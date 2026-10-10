"""Mamba 4 selective conjugate memory: chunked exact reads and cached decode.

Per head, S_t = lam_t S_(t-1) + beta_t k k^T and C_t likewise; the precision
A_t = S_t + diag(floor) keeps an undiscounted floor, so it is bounded below for
arbitrary gates. Every token reads C_t A_t^-1 q_t and q_t^T A_t^-1 q_t with an
exact Cholesky solve laid out with the batch on the TPU lane axis. All
evidence, factors and solves are float32 even under bf16 model projections.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class SelectiveState(NamedTuple):
    """Evidence-only sufficient statistics; precision adds the fixed floor."""

    evidence: jax.Array
    cross: jax.Array


class SelectiveResult(NamedTuple):
    output: jax.Array
    variance: jax.Array
    state: SelectiveState


def _cholesky_unrolled(matrix):
    """Fully unrolled Cholesky of [D,D,N]; small D only (graph grows as D^3)."""
    dim = matrix.shape[0]
    lower = [[None] * dim for _ in range(dim)]
    for j in range(dim):
        total = matrix[j, j]
        for k in range(j):
            total = total - lower[j][k] * lower[j][k]
        diagonal = jnp.sqrt(total)
        lower[j][j] = diagonal
        for i in range(j + 1, dim):
            total = matrix[i, j]
            for k in range(j):
                total = total - lower[i][k] * lower[j][k]
            lower[i][j] = total / diagonal
    zero = jnp.zeros_like(matrix[0, 0])
    return jnp.stack(
        [
            jnp.stack([lower[i][j] if j <= i else zero for j in range(dim)])
            for i in range(dim)
        ]
    )


def _cholesky_loop(matrix):
    """Left-looking Cholesky of [D,D,N] with a D-step loop and a small graph.

    Columns not yet computed are zero, so each step's full contraction equals
    the sum over earlier columns only. Each step reads the factor once.
    """
    dim = matrix.shape[0]
    rows = jnp.arange(dim)[:, None]

    def column(j, lower):
        row = lax.dynamic_index_in_dim(lower, j, axis=0, keepdims=False)
        residual = lax.dynamic_index_in_dim(matrix, j, axis=1, keepdims=False)
        residual = residual - jnp.sum(lower * row[None], axis=1)
        pivot = jnp.sqrt(lax.dynamic_index_in_dim(residual, j, axis=0, keepdims=False))
        values = jnp.where(rows > j, residual / pivot, 0.0)
        values = jnp.where(rows == j, pivot, values)
        return lax.dynamic_update_index_in_dim(lower, values, j, axis=1)

    return lax.fori_loop(0, _rolled(dim), column, jnp.zeros_like(matrix))


def _cholesky_blocked(matrix, block=8):
    """Right-looking blocked Cholesky of [D,D,N] (D a multiple of the block).

    Diagonal blocks use the unrolled factor, panels a column-sequential
    triangular solve, and trailing blocks one lane-batched outer-product
    update. Memory traffic is about D^3 N / (3 block) words instead of the
    D^3 N of the one-column loop, while the graph stays independent of N.
    """
    dim, count = matrix.shape[0], matrix.shape[2]
    if dim % block:
        raise ValueError("Blocked Cholesky needs a key dimension divisible by 8")
    blocks = dim // block
    work = {
        (i, j): matrix[i * block : (i + 1) * block, j * block : (j + 1) * block]
        for i in range(blocks)
        for j in range(i + 1)
    }
    lower = {}
    for j in range(blocks):
        diagonal = _cholesky_unrolled(work[(j, j)])
        lower[(j, j)] = diagonal
        if j + 1 == blocks:
            break
        panel = jnp.concatenate([work[(i, j)] for i in range(j + 1, blocks)], axis=0)
        columns = []
        for c in range(block):
            value = panel[:, c]
            for k in range(c):
                value = value - columns[k] * diagonal[c, k][None]
            columns.append(value / diagonal[c, c][None])
        panel = jnp.stack(columns, axis=1)
        for index, i in enumerate(range(j + 1, blocks)):
            lower[(i, j)] = panel[index * block : (index + 1) * block]
        trailing = jnp.einsum("ikn,jkn->ijn", panel, panel)
        for a, i in enumerate(range(j + 1, blocks)):
            for b, k in enumerate(range(j + 1, i + 1)):
                update = trailing[
                    a * block : (a + 1) * block, b * block : (b + 1) * block
                ]
                work[(i, k)] = work[(i, k)] - update
    zero = jnp.zeros((block, block, count), matrix.dtype)
    return jnp.concatenate(
        [
            jnp.concatenate(
                [lower[(i, j)] if j <= i else zero for j in range(blocks)], axis=1
            )
            for i in range(blocks)
        ],
        axis=0,
    )


def _rolled(count):
    """An opaque trip count keeps the factor/solve loops rolled.

    Python-unrolled factors made full-model compilation impractically slow
    (an 18-layer model did not finish compiling in 16 minutes). The rolled
    loop keeps the HLO size independent of the key dimension.
    """
    return lax.optimization_barrier(jnp.asarray(count, jnp.int32))


def _triangular_solves(lower, rhs):
    """Solve L L^T y = rhs for lane-major [D,D,N] factor and [D,N] rhs."""
    dim = rhs.shape[0]

    def forward(i, z):
        row = lax.dynamic_index_in_dim(lower, i, axis=0, keepdims=False)
        pivot = lax.dynamic_index_in_dim(row, i, axis=0, keepdims=False)
        value = lax.dynamic_index_in_dim(rhs, i, axis=0, keepdims=False)
        value = (value - jnp.sum(row * z, axis=0)) / pivot
        return lax.dynamic_update_index_in_dim(z, value, i, axis=0)

    def backward(step, y):
        i = dim - 1 - step
        col = lax.dynamic_index_in_dim(lower, i, axis=1, keepdims=False)
        pivot = lax.dynamic_index_in_dim(col, i, axis=0, keepdims=False)
        value = lax.dynamic_index_in_dim(z, i, axis=0, keepdims=False)
        value = (value - jnp.sum(col * y, axis=0)) / pivot
        return lax.dynamic_update_index_in_dim(y, value, i, axis=0)

    z = lax.fori_loop(0, _rolled(dim), forward, jnp.zeros_like(rhs))
    return lax.fori_loop(0, _rolled(dim), backward, jnp.zeros_like(rhs))


# Default factorization for spd_solve callers that do not choose one. "loop"
# keeps compile time bounded; "blocked" cuts memory traffic for D >= 16;
# "unrolled" is retained as a measured alternative for very small D.
SOLVE_BACKEND = "loop"


def _factor_and_solve(precision, rhs, backend):
    dim = rhs.shape[-1]
    count = rhs.size // dim
    matrix = jnp.transpose(precision.reshape(count, dim, dim), (1, 2, 0))
    vector = rhs.reshape(count, dim).T
    backend = SOLVE_BACKEND if backend is None else backend
    # The fused path solves whole chunks on chip; single solves use blocks.
    backend = "auto" if backend == "fused" else backend
    if backend == "unrolled":
        lower = _cholesky_unrolled(matrix)
    elif backend == "blocked" or (backend == "auto" and dim % 8 == 0):
        lower = _cholesky_blocked(matrix)
    elif backend in ("loop", "auto"):
        lower = _cholesky_loop(matrix)
    else:
        raise ValueError(f"Unknown SPD solve backend {backend}")
    solution = _triangular_solves(lower, vector).T.reshape(rhs.shape)
    return solution, lower


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def spd_solve(precision, rhs, backend=None):
    """Solve precision @ y = rhs for SPD [...,d,d] precision and [...,d] rhs.

    Only the lower triangle is read; the batch lies on the minor (TPU lane)
    axis. The backward pass reuses the Cholesky factor: z = A^-1 g, d rhs = z
    and d A = -sym(z y^T). No inverse is formed. The factorization backend
    changes only the arithmetic schedule, never the solved system.
    """
    return _factor_and_solve(
        jnp.asarray(precision, jnp.float32), jnp.asarray(rhs, jnp.float32), backend
    )[0]


def _spd_solve_forward(precision, rhs, backend):
    solution, lower = _factor_and_solve(
        jnp.asarray(precision, jnp.float32), jnp.asarray(rhs, jnp.float32), backend
    )
    return solution, (lower, solution)


def _spd_solve_backward(backend, residuals, gradient):
    del backend
    lower, solution = residuals
    dim = solution.shape[-1]
    vector = gradient.astype(jnp.float32).reshape(-1, dim).T
    adjoint = _triangular_solves(lower, vector).T.reshape(solution.shape)
    outer = adjoint[..., :, None] * solution[..., None, :]
    return -0.5 * (outer + jnp.swapaxes(outer, -1, -2)), adjoint


spd_solve.defvjp(_spd_solve_forward, _spd_solve_backward)


def selective_initial_state(batch, heads, key_dim, value_dim):
    return SelectiveState(
        jnp.zeros((batch, heads, key_dim, key_dim), jnp.float32),
        jnp.zeros((batch, heads, value_dim, key_dim), jnp.float32),
    )


def selective_gaussian_memory(
    keys,
    values,
    queries,
    beta,
    log_decay,
    floor,
    *,
    chunk_size=64,
    initial=None,
    solver=None,
):
    """Exact chunked reads of the fixed-floor selective Gaussian memory.

    keys/queries [B,T,H,D], values [B,T,H,P], beta and log_decay [B,T,H]
    (log_decay <= 0), floor [H,D] positive prior-precision diagonal. With
    S_t = lam_t S_(t-1) + beta_t k k^T and C_t likewise, each token reads
    C_t y_t and q_t^T y_t, where (S_t + diag(floor)) y_t = q_t. Intra-chunk
    evidence is a masked decay matrix product; chunk boundaries use an
    associative affine scan. Solves are exact Cholesky solves, O(T D^3).
    """
    if solver == "fused":
        from lm.kernels.mamba4_fused import selective_memory_fused

        return selective_memory_fused(
            keys,
            values,
            queries,
            beta,
            log_decay,
            floor,
            chunk_size=chunk_size,
            initial=initial,
            interpret=jax.default_backend() != "tpu",
        )
    keys, values, queries = (
        jnp.asarray(a, jnp.float32) for a in (keys, values, queries)
    )
    batch, time, heads, dim = keys.shape
    value_dim = values.shape[-1]
    if queries.shape != keys.shape or values.shape[:3] != keys.shape[:3]:
        raise ValueError("Selective memory keys, values and queries disagree")
    shape = (batch, time, heads)
    beta = jnp.broadcast_to(jnp.asarray(beta, jnp.float32), shape)
    log_decay = jnp.broadcast_to(jnp.asarray(log_decay, jnp.float32), shape)
    floor = jnp.broadcast_to(jnp.asarray(floor, jnp.float32), (heads, dim))
    state = (
        selective_initial_state(batch, heads, dim, value_dim)
        if initial is None
        else initial
    )
    padding = (-time) % chunk_size
    count = (time + padding) // chunk_size

    def chunks(x):
        # Padding writes nothing and does not decay: beta = 0 and log_decay = 0.
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        x = x.reshape(batch, count, chunk_size, *x.shape[2:])
        return jnp.swapaxes(x, 2, 3)

    k, v, q = chunks(keys), chunks(values), chunks(queries)
    b, g = chunks(beta), chunks(log_decay)
    prefix = jnp.cumsum(g, axis=-1)
    position = jnp.arange(chunk_size)
    causal = position[:, None] >= position[None, :]
    difference = prefix[..., :, None] - prefix[..., None, :]
    weights = jnp.where(causal, jnp.exp(jnp.where(causal, difference, 0.0)), 0.0)
    weights *= b[..., None, :]
    highest = lax.Precision.HIGHEST
    outer = (k[..., :, None] * k[..., None, :]).reshape(*k.shape[:-1], dim * dim)
    local = jnp.einsum("bnhts,bnhse->bnhte", weights, outer, precision=highest)
    chunk_evidence = local[..., -1, :].reshape(*local.shape[:3], dim, dim)
    chunk_cross = jnp.einsum(
        "bnhs,bnhsp,bnhsd->bnhpd", weights[..., -1, :], v, k, precision=highest
    )
    chunk_decay = jnp.exp(prefix[..., -1])

    def combine(earlier, later):
        d0, s0, c0 = earlier
        d1, s1, c1 = later
        return (
            d1 * d0,
            s1 + d1[..., None, None] * s0,
            c1 + d1[..., None, None] * c0,
        )

    total, ends_s, ends_c = lax.associative_scan(
        combine, (chunk_decay, chunk_evidence, chunk_cross), axis=1
    )
    ends_s = ends_s + total[..., None, None] * state.evidence[:, None]
    ends_c = ends_c + total[..., None, None] * state.cross[:, None]
    starts_s = jnp.concatenate((state.evidence[:, None], ends_s[:, :-1]), axis=1)
    starts_c = jnp.concatenate((state.cross[:, None], ends_c[:, :-1]), axis=1)
    carried = jnp.exp(prefix)
    evidence = local.reshape(*local.shape[:-1], dim, dim)
    evidence = evidence + carried[..., None, None] * starts_s[:, :, :, None]
    diagonal = floor[:, :, None] * jnp.eye(dim)
    precision = evidence + diagonal[None, None, :, None]
    solved = spd_solve(precision, q, solver)
    variance = jnp.sum(q * solved, axis=-1)
    scores = jnp.einsum("bnhtd,bnhsd->bnhts", solved, k, precision=highest)
    within = jnp.einsum("bnhts,bnhsp->bnhtp", scores * weights, v, precision=highest)
    inherited = jnp.einsum("bnhtd,bnhpd->bnhtp", solved, starts_c, precision=highest)
    output = within + carried[..., None] * inherited

    def unchunk(x):
        x = jnp.swapaxes(x, 2, 3)
        return x.reshape(batch, count * chunk_size, *x.shape[3:])[:, :time]

    final = SelectiveState(ends_s[:, -1], ends_c[:, -1])
    return SelectiveResult(unchunk(output), unchunk(variance), final)


def selective_decode(state, key, value, query, beta, log_decay, floor, solver=None):
    """One exact write/read: O(D^3 + P D) refactor of the fixed-floor precision."""
    lam = jnp.exp(jnp.asarray(log_decay, jnp.float32))[..., None, None]
    beta = jnp.asarray(beta, jnp.float32)[..., None, None]
    key, value, query = (jnp.asarray(a, jnp.float32) for a in (key, value, query))
    evidence = lam * state.evidence + beta * key[..., :, None] * key[..., None, :]
    cross = lam * state.cross + beta * value[..., :, None] * key[..., None, :]
    floor = jnp.asarray(floor, jnp.float32)
    precision = evidence + floor[..., :, None] * jnp.eye(key.shape[-1])
    solved = spd_solve(precision, query, solver)
    output = jnp.einsum(
        "...pd,...d->...p", cross, solved, precision=lax.Precision.HIGHEST
    )
    variance = jnp.sum(query * solved, axis=-1)
    return SelectiveResult(output, variance, SelectiveState(evidence, cross))


def selective_diagnostics(state, floor):
    """Spectral and finiteness probes of the final selective precision."""
    dim = state.evidence.shape[-1]
    precision = state.evidence + jnp.asarray(floor)[..., :, None] * jnp.eye(dim)
    with jax.default_matmul_precision("highest"):
        eigenvalues = jnp.linalg.eigvalsh(
            (precision + jnp.swapaxes(precision, -1, -2)) * 0.5
        )
    minimum, maximum = eigenvalues[..., 0], eigenvalues[..., -1]
    return {
        "precision_min_eigenvalue": jnp.min(minimum),
        "precision_max_eigenvalue": jnp.max(maximum),
        "precision_max_condition": jnp.max(maximum / jnp.maximum(minimum, 1e-30)),
        "state_all_finite": jnp.all(jnp.isfinite(state.evidence))
        & jnp.all(jnp.isfinite(state.cross)),
        "cross_max_abs": jnp.max(jnp.abs(state.cross)),
        "floor_min": jnp.min(floor),
        "floor_max": jnp.max(floor),
    }
