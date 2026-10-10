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
import jax.numpy as jnp

from lm.kernels.mamba4 import SelectiveResult, SelectiveState, selective_initial_state

LANES = 128
HIGHEST = lax.Precision.HIGHEST


def _solve_kernel(
    s0_ref, k_ref, beta_ref, lam_ref, floor_ref, rhs_ref, y_ref, e_ref, a_ref
):
    """Lane-major blocks: s0 [d,d,L]; k, rhs, y [c,d,L]; beta, lam [c,1,L]; floor [d,L].

    a_ref[j] holds column j of the working matrix, so every dynamic index is on
    the leading axis; values are read out of a column with masked reductions.
    """
    chunk, dim = k_ref.shape[0], k_ref.shape[1]
    column_index = lax.broadcasted_iota(jnp.int32, (dim, dim, 1), 0)
    row_index = lax.broadcasted_iota(jnp.int32, (dim, dim, 1), 1)
    eye = (column_index == row_index).astype(jnp.float32)
    rows = lax.broadcasted_iota(jnp.int32, (dim, 1), 0)

    def pick(vector, j):
        return jnp.sum(jnp.where(rows == j, vector, 0.0), axis=0, keepdims=True)

    e_ref[...] = s0_ref[...]

    def token(t, carry):
        key = k_ref[t]
        evidence = lam_ref[t] * e_ref[...] + beta_ref[t] * (
            key[:, None, :] * key[None, :, :]
        )
        e_ref[...] = evidence
        a_ref[...] = evidence + eye * floor_ref[...][None, :, :]

        def column(j, inner):
            col = a_ref[j]
            pivot = jnp.sqrt(pick(col, j))
            below = jnp.where(rows > j, col / pivot, 0.0)
            a_ref[j] = jnp.where(rows == j, pivot, below)
            # a_ref[c][r] holds A[r, c]; only the trailing block changes.
            update = below[:, None, :] * below[None, :, :]
            trailing = (column_index > j) & (row_index > j)
            a_ref[...] = a_ref[...] - jnp.where(trailing, update, 0.0)
            return inner

        lax.fori_loop(0, dim, column, 0)

        def forward(j, value):
            col = a_ref[j]
            step = pick(value, j) / pick(col, j)
            value = jnp.where(rows > j, value - col * step, value)
            return jnp.where(rows == j, step, value)

        solved = lax.fori_loop(0, dim, forward, rhs_ref[t])

        def backward(i, value):
            j = dim - 1 - i
            col = a_ref[j]
            total = jnp.sum(
                jnp.where(rows > j, col * value, 0.0), axis=0, keepdims=True
            )
            step = (pick(value, j) - total) / pick(col, j)
            return jnp.where(rows == j, step, value)

        y_ref[t] = lax.fori_loop(0, dim, backward, solved)
        return carry

    lax.fori_loop(0, chunk, token, 0)


def _lane_solve(s0, keys, beta, lam, floor, rhs, interpret):
    """Lane-major arrays whose last axis is a multiple of 128."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    dim, lanes = s0.shape[0], s0.shape[-1]

    def spec(shape):
        leading = (0,) * (len(shape) - 1)
        return pl.BlockSpec((*shape[:-1], LANES), lambda i: (*leading, i))

    arrays = (s0, keys, beta, lam, floor, rhs)
    return pl.pallas_call(
        _solve_kernel,
        grid=(lanes // LANES,),
        in_specs=[spec(a.shape) for a in arrays],
        out_specs=spec(rhs.shape),
        out_shape=jax.ShapeDtypeStruct(rhs.shape, jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
        ],
        interpret=interpret,
    )(*arrays)


def batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret=False):
    """Solve A_t y_t = rhs_t for every token of S independent chunks.

    s0 [S,d,d] symmetric; keys, rhs [S,c,d]; beta, log_prefix [S,c]; floor
    [S,d]. log_prefix is the inclusive prefix sum of the chunk's log decays.
    """
    sequences = s0.shape[0]
    pad = (-sequences) % LANES
    lam = jnp.exp(jnp.diff(log_prefix, prepend=0.0, axis=-1))

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
    return jnp.moveaxis(solved[..., :sequences], -1, 0)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def chunk_solve(s0, keys, beta, log_prefix, floor, rhs, interpret=False):
    """Differentiable batched solve; see the module docstring."""
    return batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret)


def _chunk_solve_forward(s0, keys, beta, log_prefix, floor, rhs, interpret):
    y = batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret)
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
