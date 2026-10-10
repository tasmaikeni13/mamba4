"""Fused selective memory: on-chip per-token solves with matrix-product gradients.

The exact read needs, for every token t of a chunk, the solve A_t y_t = r_t with

    A_t = exp(P_t) S0 + sum_{s<=t} beta_s exp(P_t - P_s) k_s k_s^T + diag(floor),

where S0 is the evidence at the chunk start and P is the inclusive prefix sum
of the chunk's log decays. Materializing A_t for every token, and factoring it
column by column in HBM, dominates the plain XLA path. Here one Pallas kernel
keeps each chunk on chip. It streams the rank-one updates
E_t = lam_t E_(t-1) + beta_t k_t k_t^T, factors E_t + diag(floor) by Cholesky
and solves, with 128 independent (batch, head, chunk) sequences across the
vector lanes. Chunk-boundary statistics, reads and every gradient are matrix
products. With y = A^-1 r and z = A^-1 g (a second kernel call), dr = z and
dA_t = -z_t y_t^T is rank one, so the gradients of the keys, precisions,
decays, floor and S0 reduce to c-by-c and c-by-d products. No inverse is
formed.
"""

from functools import partial

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lm.kernels.mamba4 import SelectiveResult, SelectiveState, selective_initial_state

LANES = 128
HIGHEST = lax.Precision.HIGHEST


def _solve_kernel(
    s0_ref, k_ref, beta_ref, lam_ref, floor_ref, rhs_ref, y_ref, e_ref, a_ref
):
    """Lane-major blocks: s0 [d,d,L]; k, rhs, y [c,d,L]; beta, lam [c,1,L]; floor [d,L].

    e_ref[j] and a_ref[j] hold column j of the evidence and of the working
    matrix, rows on sublanes. Only rows from the column's 8-row tile down are
    kept current: the lower triangle plus a few finite entries above it that
    no step reads. The factorization runs over a dynamic pivot column; its
    trailing update is unrolled over 8-column blocks behind dynamic guards,
    so it costs about d^3/24 vector operations instead of d^3. Triangular
    solves are fully unrolled, which turns every element pick into a static
    sublane slice.
    """
    chunk, dim = k_ref.shape[0], k_ref.shape[1]
    blocks = dim // 8
    rows = lax.broadcasted_iota(jnp.int32, (dim, 1), 0)

    def tile(c):
        return slice(8 * (c // 8), dim)

    # The grid's second axis walks the chunk's tokens in blocks; the evidence
    # carries between them in e_ref.
    @pl.when(pl.program_id(1) == 0)
    def _():
        for c in range(dim):
            e_ref[c, tile(c)] = s0_ref[c, tile(c)]

    def token(t, carry):
        key, lam, beta = k_ref[t], lam_ref[t], beta_ref[t]
        floor = floor_ref[...]
        for c in range(dim):
            part = tile(c)
            evidence = lam * e_ref[c, part] + (beta * key[c : c + 1]) * key[part]
            e_ref[c, part] = evidence
            on_diagonal = rows[part] == c
            a_ref[c, part] = evidence + jnp.where(on_diagonal, floor[c : c + 1], 0.0)

        def column(j, inner):
            col = a_ref[j]
            pivot = jnp.sqrt(
                jnp.sum(jnp.where(rows == j, col, 0.0), axis=0, keepdims=True)
            )
            below = jnp.where(rows > j, col / pivot, 0.0)
            a_ref[j] = jnp.where(rows == j, pivot, below)
            for block in range(blocks):

                @pl.when(j < 8 * block + 7)
                def _():
                    part = slice(8 * block, dim)
                    for c in range(8 * block, 8 * block + 8):
                        a_ref[c, part] = a_ref[c, part] - below[part] * below[c : c + 1]

            return inner

        lax.fori_loop(0, dim, column, 0)
        value = rhs_ref[t]
        for j in range(dim):
            col = a_ref[j]
            step = value[j : j + 1] / col[j : j + 1]
            value = jnp.where(rows > j, value - col * step, value)
            value = jnp.where(rows == j, step, value)
        for j in reversed(range(dim)):
            col = a_ref[j]
            total = jnp.sum(
                jnp.where(rows > j, col * value, 0.0), axis=0, keepdims=True
            )
            value = jnp.where(
                rows == j, (value[j : j + 1] - total) / col[j : j + 1], value
            )
        y_ref[t] = value
        return carry

    lax.fori_loop(0, chunk, token, 0)


def _lane_solve(s0, keys, beta, lam, floor, rhs, interpret):
    """Lane-major arrays whose last axis is a multiple of 128.

    Grid: lane groups (parallel) by blocks of up to 16 tokens (sequential),
    which keeps the double-buffered blocks well inside scoped VMEM.
    """
    dim, lanes = s0.shape[0], s0.shape[-1]
    chunk = keys.shape[0]
    block = next(size for size in (16, 8, 4, 2, 1) if chunk % size == 0)

    def whole(shape):
        leading = (0,) * (len(shape) - 1)
        return pl.BlockSpec((*shape[:-1], LANES), lambda i, j: (*leading, i))

    def tokens(shape):
        return pl.BlockSpec(
            (block, *shape[1:-1], LANES),
            lambda i, j: (j,) + (0,) * (len(shape) - 2) + (i,),
        )

    return pl.pallas_call(
        _solve_kernel,
        grid=(lanes // LANES, chunk // block),
        in_specs=[
            whole(s0.shape),
            tokens(keys.shape),
            tokens(beta.shape),
            tokens(lam.shape),
            whole(floor.shape),
            tokens(rhs.shape),
        ],
        out_specs=tokens(rhs.shape),
        out_shape=jax.ShapeDtypeStruct(rhs.shape, jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
    )(s0, keys, beta, lam, floor, rhs)


def batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret=False):
    """Solve A_t y_t = rhs_t for every token of S independent chunks.

    s0 [S,d,d] symmetric; keys, rhs [S,c,d]; beta, log_prefix [S,c]; floor
    [S,d]. log_prefix is the inclusive prefix sum of the chunk's log decays.
    """
    sequences, dim = s0.shape[0], s0.shape[-1]
    pad = (-sequences) % LANES
    extra = (-dim) % 8
    lam = jnp.exp(jnp.diff(log_prefix, prepend=0.0, axis=-1))
    if extra:
        # Padded coordinates form an identity block: they solve to zero and
        # leave the true coordinates unchanged.
        s0 = jnp.pad(s0, [(0, 0), (0, extra), (0, extra)])
        keys = jnp.pad(keys, [(0, 0), (0, 0), (0, extra)])
        rhs = jnp.pad(rhs, [(0, 0), (0, 0), (0, extra)])
        floor = jnp.pad(floor, [(0, 0), (0, extra)], constant_values=1.0)

    def lane(x, expand=False):
        x = jnp.moveaxis(x.astype(jnp.float32), 0, -1)
        if expand:
            x = x[..., None, :]
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])

    # Padded sequences solve the identity system and are discarded.
    floor_lanes = lane(floor)
    floor_lanes = floor_lanes.at[..., sequences:].set(1.0)
    solved = _lane_solve(
        lane(s0),
        lane(keys),
        lane(beta, True),
        lane(lam, True).at[..., sequences:].set(1.0),
        floor_lanes,
        lane(rhs),
        interpret,
    )
    return jnp.moveaxis(solved[..., :sequences], -1, 0)[..., :dim]


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def chunk_solve(s0, keys, beta, log_prefix, floor, rhs, interpret=False):
    """Differentiable batched solve; see the module docstring."""
    return batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret)


def _chunk_solve_forward(s0, keys, beta, log_prefix, floor, rhs, interpret):
    y = batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret)
    # A "kernels" remat policy keeps this; the backward then needs no new solve.
    y = checkpoint_name(y, "memory_solve")
    return y, (s0, keys, beta, log_prefix, floor, y)


def _chunk_solve_backward(interpret, saved, grad):
    s0, keys, beta, log_prefix, floor, y = saved
    z = batched_solve(s0, keys, beta, log_prefix, floor, grad, interpret)
    chunk = keys.shape[1]
    causal = jnp.arange(chunk)[:, None] >= jnp.arange(chunk)[None, :]
    difference = log_prefix[:, :, None] - log_prefix[:, None, :]
    decay = jnp.where(causal, jnp.exp(jnp.where(causal, difference, 0.0)), 0.0)
    weights = decay * beta[:, None, :]
    carried = jnp.exp(log_prefix)
    gy = jnp.einsum("std,sud->stu", y, keys, precision=HIGHEST)
    gz = jnp.einsum("std,sud->stu", z, keys, precision=HIGHEST)
    d_weights = -gz * gy
    d_keys = -(
        jnp.einsum("stu,std->sud", weights * gy, z, precision=HIGHEST)
        + jnp.einsum("stu,std->sud", weights * gz, y, precision=HIGHEST)
    )
    d_beta = jnp.sum(d_weights * decay, axis=1)
    weighted = d_weights * weights
    s0y = jnp.einsum("std,sed->ste", y, s0, precision=HIGHEST)
    d_prefix = (
        jnp.sum(weighted, axis=2)
        - jnp.sum(weighted, axis=1)
        - carried * jnp.sum(s0y * z, axis=-1)
    )
    d_s0 = -jnp.einsum("std,ste->sde", z * carried[..., None], y, precision=HIGHEST)
    d_floor = -jnp.sum(z * y, axis=1)
    return d_s0, d_keys, d_beta, d_prefix, d_floor, z


chunk_solve.defvjp(_chunk_solve_forward, _chunk_solve_backward)


def selective_memory_fused(
    keys,
    values,
    queries,
    beta,
    log_decay,
    floor,
    *,
    chunk_size=64,
    initial=None,
    interpret=False,
):
    """Same contract as `selective_gaussian_memory`, with the fused solve."""
    keys, values, queries = (
        jnp.asarray(a, jnp.float32) for a in (keys, values, queries)
    )
    batch, time, heads, dim = keys.shape
    value_dim = values.shape[-1]
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
        # Padding writes nothing and does not decay: beta = 0, log_decay = 0.
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        x = x.reshape(batch, count, chunk_size, *x.shape[2:])
        return jnp.swapaxes(x, 2, 3)

    k, v, q = chunks(keys), chunks(values), chunks(queries)
    b, g = chunks(beta), chunks(log_decay)
    prefix = jnp.cumsum(g, axis=-1)
    tail = b * jnp.exp(prefix[..., -1:] - prefix)
    chunk_evidence = jnp.einsum(
        "bnhs,bnhsd,bnhse->bnhde", tail, k, k, precision=HIGHEST
    )
    chunk_cross = jnp.einsum("bnhs,bnhsp,bnhsd->bnhpd", tail, v, k, precision=HIGHEST)
    chunk_decay = jnp.exp(prefix[..., -1])

    def combine(earlier, later):
        d0, s0, c0 = earlier
        d1, s1, c1 = later
        return d1 * d0, s1 + d1[..., None, None] * s0, c1 + d1[..., None, None] * c0

    total, ends_s, ends_c = lax.associative_scan(
        combine, (chunk_decay, chunk_evidence, chunk_cross), axis=1
    )
    ends_s = ends_s + total[..., None, None] * state.evidence[:, None]
    ends_c = ends_c + total[..., None, None] * state.cross[:, None]
    starts_s = jnp.concatenate((state.evidence[:, None], ends_s[:, :-1]), axis=1)
    starts_c = jnp.concatenate((state.cross[:, None], ends_c[:, :-1]), axis=1)
    sequences = batch * count * heads
    flat = lambda x: x.reshape(sequences, *x.shape[3:])  # noqa: E731
    solved = chunk_solve(
        flat(starts_s),
        flat(k),
        flat(b),
        flat(prefix),
        jnp.broadcast_to(floor[None, None], (batch, count, heads, dim)).reshape(
            sequences, dim
        ),
        flat(q),
        interpret,
    ).reshape(q.shape)
    causal = jnp.arange(chunk_size)[:, None] >= jnp.arange(chunk_size)[None, :]
    difference = prefix[..., :, None] - prefix[..., None, :]
    weights = jnp.where(causal, jnp.exp(jnp.where(causal, difference, 0.0)), 0.0)
    weights = weights * b[..., None, :]
    scores = jnp.einsum("bnhtd,bnhsd->bnhts", solved, k, precision=HIGHEST)
    within = jnp.einsum("bnhts,bnhsp->bnhtp", scores * weights, v, precision=HIGHEST)
    inherited = jnp.einsum("bnhtd,bnhpd->bnhtp", solved, starts_c, precision=HIGHEST)
    output = within + jnp.exp(prefix)[..., None] * inherited
    variance = jnp.sum(q * solved, axis=-1)

    def unchunk(x):
        x = jnp.swapaxes(x, 2, 3)
        return x.reshape(batch, count * chunk_size, *x.shape[3:])[:, :time]

    final = SelectiveState(ends_s[:, -1], ends_c[:, -1])
    return SelectiveResult(unchunk(output), unchunk(variance), final)
