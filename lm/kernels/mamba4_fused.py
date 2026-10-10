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
from lm.kernels.prefix import chunk_carry, cumulative

LANES = 128
HIGHEST = lax.Precision.HIGHEST


def _packed_offsets(dim):
    """Column c keeps rows 8*(c//8)..dim; offsets of each column when packed."""
    offsets, total = [], 0
    for column in range(dim):
        offsets.append(total)
        total += dim - 8 * (column // 8)
    return offsets, total


def _substitute(tile_of, value, dim):
    """Solve L L^T y = value from the lower factor L, held as 8-row tiles.

    tile_of(j, b) returns tile b (rows 8b..8b+7) of column j for b >= j // 8,
    whose row j holds the reciprocal pivot 1 / L_jj, so no step divides.
    value is a list of [8, L] tiles. Every pick is a static slice, and each
    column touches only the tiles at or below it.
    """
    rows = lax.broadcasted_iota(jnp.int32, (8, 1), 0)
    blocks = dim // 8
    for j in range(dim):
        first, local = j // 8, j % 8
        reciprocal = tile_of(j, first)[local : local + 1]
        step = value[first][local : local + 1] * reciprocal
        for b in range(first, blocks):
            column = tile_of(j, b)
            index = rows + 8 * b
            tile = jnp.where(index > j, value[b] - column * step, value[b])
            value[b] = jnp.where(index == j, step, tile) if b == first else tile
    for j in reversed(range(dim)):
        first, local = j // 8, j % 8
        total = None
        for b in range(first, blocks):
            index = rows + 8 * b
            term = jnp.where(index > j, tile_of(j, b) * value[b], 0.0)
            total = term if total is None else total + term
        total = jnp.sum(total, axis=0, keepdims=True)
        reciprocal = tile_of(j, first)[local : local + 1]
        new = (value[first][local : local + 1] - total) * reciprocal
        value[first] = jnp.where(rows + 8 * first == j, new, value[first])
    return value


def _factor_token(e_ref, a_ref, key, lam, beta, floor, dim):
    """Evidence update E = lam E + beta k k^T, then Cholesky of E + diag(floor).

    e_ref[j] and a_ref[j] hold column j of the evidence and of the factor L,
    rows on sublanes; only rows from the column's 8-row tile down are used.
    The factorization is left-looking over static columns: column j starts
    from E's column plus the floor and subtracts every earlier column scaled
    by its row-j entry, so the working column stays in registers and only the
    finished column is stored. The diagonal keeps 1 / L_jj, so neither the
    factorization nor the solves divide.
    """
    rows = lax.broadcasted_iota(jnp.int32, (dim, 1), 0)
    for c in range(dim):
        part = slice(8 * (c // 8), dim)
        e_ref[c, part] = lam * e_ref[c, part] + (beta * key[c : c + 1]) * key[part]
    for j in range(dim):
        part = slice(8 * (j // 8), dim)
        index = rows[part]
        column = e_ref[j, part] + jnp.where(index == j, floor[j : j + 1], 0.0)

        # Earlier columns in blocks of eight (a dynamic loop with a static
        # body), then the remainder of the current block.
        def block(index, column, j=j, part=part):
            for offset in range(8):
                k = 8 * index + offset
                column = column - a_ref[k, part] * a_ref[k, j : j + 1]
            return column

        if j >= 8:
            column = lax.fori_loop(0, j // 8, block, column)
        for k in range(8 * (j // 8), j):
            column = column - a_ref[k, part] * a_ref[k, j : j + 1]
        local = j % 8
        reciprocal = lax.rsqrt(column[local : local + 1])
        below = jnp.where(index > j, column * reciprocal, 0.0)
        a_ref[j, part] = jnp.where(index == j, reciprocal, below)


def _solve_kernel(
    s0_ref, k_ref, beta_ref, lam_ref, floor_ref, rhs_ref, *refs, store_factors
):
    """Lane-major blocks: s0 [d,d,L]; k, rhs, y [c,d,L]; beta, lam [c,1,L];
    floor [d,L]; with store_factors, also each token's packed factor."""
    if store_factors:
        y_ref, factor_ref, e_ref, a_ref = refs
    else:
        (y_ref, e_ref, a_ref), factor_ref = refs, None
    chunk, dim = k_ref.shape[0], k_ref.shape[1]
    offsets, _ = _packed_offsets(dim)

    # The grid's second axis walks the chunk's tokens in blocks; the evidence
    # carries between them in e_ref.
    @pl.when(pl.program_id(1) == 0)
    def _():
        for c in range(dim):
            part = slice(8 * (c // 8), dim)
            e_ref[c, part] = s0_ref[c, part]

    def token(t, carry):
        _factor_token(
            e_ref, a_ref, k_ref[t], lam_ref[t], beta_ref[t], floor_ref[...], dim
        )
        if factor_ref is not None:
            for c in range(dim):
                start = 8 * (c // 8)
                factor_ref[t, offsets[c] : offsets[c] + dim - start] = a_ref[
                    c, start:dim
                ]
        value = [rhs_ref[t, 8 * b : 8 * b + 8] for b in range(dim // 8)]
        value = _substitute(lambda j, b: a_ref[j, 8 * b : 8 * b + 8], value, dim)
        for b in range(dim // 8):
            y_ref[t, 8 * b : 8 * b + 8] = value[b]
        return carry

    lax.fori_loop(0, chunk, token, 0)


def _factored_kernel(factor_ref, rhs_ref, z_ref):
    """Solve with stored packed factors: factor [c,R,L], rhs and z [c,d,L]."""
    chunk, dim = rhs_ref.shape[0], rhs_ref.shape[1]
    offsets, _ = _packed_offsets(dim)

    def tile_of(t):
        def tile(j, b):
            start = offsets[j] + 8 * (b - j // 8)
            return factor_ref[t, start : start + 8]

        return tile

    def token(t, carry):
        value = [rhs_ref[t, 8 * b : 8 * b + 8] for b in range(dim // 8)]
        value = _substitute(tile_of(t), value, dim)
        for b in range(dim // 8):
            z_ref[t, 8 * b : 8 * b + 8] = value[b]
        return carry

    lax.fori_loop(0, chunk, token, 0)


def _token_block(chunk, limit):
    return next(
        size for size in (16, 8, 4, 2, 1) if size <= limit and chunk % size == 0
    )


def _lane_solve(s0, keys, beta, lam, floor, rhs, interpret, store_factors=False):
    """Lane-major arrays whose last axis is a multiple of 128.

    Grid: lane groups (parallel) by blocks of tokens (sequential), which keeps
    the double-buffered blocks well inside scoped VMEM. With store_factors the
    packed factor of every token is also returned, [c, R, lanes].
    """
    dim, lanes = s0.shape[0], s0.shape[-1]
    chunk = keys.shape[0]
    block = _token_block(chunk, 1 if store_factors else 16)
    _, packed = _packed_offsets(dim)

    def whole(shape):
        leading = (0,) * (len(shape) - 1)
        return pl.BlockSpec((*shape[:-1], LANES), lambda i, j: (*leading, i))

    def tokens(shape):
        return pl.BlockSpec(
            (block, *shape[1:-1], LANES),
            lambda i, j: (j,) + (0,) * (len(shape) - 2) + (i,),
        )

    out_specs = tokens(rhs.shape)
    out_shape = jax.ShapeDtypeStruct(rhs.shape, jnp.float32)
    if store_factors:
        factor_shape = (chunk, packed, lanes)
        out_specs = [out_specs, tokens(factor_shape)]
        out_shape = [out_shape, jax.ShapeDtypeStruct(factor_shape, jnp.float32)]
    return pl.pallas_call(
        partial(_solve_kernel, store_factors=store_factors),
        grid=(lanes // LANES, chunk // block),
        in_specs=[
            whole(s0.shape),
            tokens(keys.shape),
            tokens(beta.shape),
            tokens(lam.shape),
            whole(floor.shape),
            tokens(rhs.shape),
        ],
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
            pltpu.VMEM((dim, dim, LANES), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
    )(s0, keys, beta, lam, floor, rhs)


def _lane_factored_solve(factors, rhs, interpret):
    chunk, dim, lanes = rhs.shape
    block = _token_block(chunk, 1)

    def tokens(shape):
        return pl.BlockSpec(
            (block, *shape[1:-1], LANES),
            lambda i, j: (j,) + (0,) * (len(shape) - 2) + (i,),
        )

    return pl.pallas_call(
        _factored_kernel,
        grid=(lanes // LANES, chunk // block),
        in_specs=[tokens(factors.shape), tokens(rhs.shape)],
        out_specs=tokens(rhs.shape),
        out_shape=jax.ShapeDtypeStruct(rhs.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
    )(factors, rhs)


def _lanes(x, sequences, expand=False):
    pad = (-sequences) % LANES
    x = jnp.moveaxis(x.astype(jnp.float32), 0, -1)
    if expand:
        x = x[..., None, :]
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])


def _padded_dim(dim):
    return dim + (-dim) % 8


def _rhs_lanes(rhs, sequences):
    extra = _padded_dim(rhs.shape[-1]) - rhs.shape[-1]
    return _lanes(jnp.pad(rhs, [(0, 0), (0, 0), (0, extra)]), sequences)


def _from_lanes(solved, sequences, dim):
    return jnp.moveaxis(solved[..., :sequences], -1, 0)[..., :dim]


def batched_solve(
    s0, keys, beta, log_prefix, floor, rhs, interpret=False, store_factors=False
):
    """Solve A_t y_t = rhs_t for every token of S independent chunks.

    s0 [S,d,d] symmetric; keys, rhs [S,c,d]; beta, log_prefix [S,c]; floor
    [S,d]. log_prefix is the inclusive prefix sum of the chunk's log decays.
    With store_factors, also returns the packed lane-major factors, which
    `factored_solve` reuses for further right-hand sides.
    """
    sequences, dim = s0.shape[0], s0.shape[-1]
    extra = _padded_dim(dim) - dim
    lam = jnp.exp(jnp.diff(log_prefix, prepend=0.0, axis=-1))
    if extra:
        # Padded coordinates form an identity block: they solve to zero and
        # leave the true coordinates unchanged.
        s0 = jnp.pad(s0, [(0, 0), (0, extra), (0, extra)])
        keys = jnp.pad(keys, [(0, 0), (0, 0), (0, extra)])
        floor = jnp.pad(floor, [(0, 0), (0, extra)], constant_values=1.0)
    # Padded sequences solve the identity system and are discarded.
    floor_lanes = _lanes(floor, sequences).at[..., sequences:].set(1.0)
    result = _lane_solve(
        _lanes(s0, sequences),
        _lanes(keys, sequences),
        _lanes(beta, sequences, True),
        _lanes(lam, sequences, True).at[..., sequences:].set(1.0),
        floor_lanes,
        _rhs_lanes(rhs, sequences),
        interpret,
        store_factors,
    )
    if store_factors:
        solved, factors = result
        return _from_lanes(solved, sequences, dim), factors
    return _from_lanes(result, sequences, dim)


def factored_solve(factors, rhs, interpret=False):
    """Solve A_t z_t = rhs_t with the factors stored by `batched_solve`."""
    sequences, dim = rhs.shape[0], rhs.shape[-1]
    solved = _lane_factored_solve(factors, _rhs_lanes(rhs, sequences), interpret)
    return _from_lanes(solved, sequences, dim)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def chunk_solve(s0, keys, beta, log_prefix, floor, rhs, interpret=False):
    """Differentiable batched solve; see the module docstring."""
    return batched_solve(s0, keys, beta, log_prefix, floor, rhs, interpret)


def _chunk_solve_forward(s0, keys, beta, log_prefix, floor, rhs, interpret):
    y, factors = batched_solve(
        s0, keys, beta, log_prefix, floor, rhs, interpret, store_factors=True
    )
    # A "kernels" remat policy keeps these, so the backward reruns no kernel
    # and needs only triangular solves with the stored factors.
    y = checkpoint_name(y, "memory_solve")
    factors = checkpoint_name(factors, "memory_solve")
    return y, (s0, keys, beta, log_prefix, floor, y, factors)


def _chunk_solve_backward(interpret, saved, grad):
    s0, keys, beta, log_prefix, floor, y, factors = saved
    z = factored_solve(factors, grad, interpret)
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
    prefix = cumulative(g)
    tail = b * jnp.exp(prefix[..., -1:] - prefix)
    chunk_evidence = jnp.einsum(
        "bnhs,bnhsd,bnhse->bnhde", tail, k, k, precision=HIGHEST
    )
    chunk_cross = jnp.einsum("bnhs,bnhsp,bnhsd->bnhpd", tail, v, k, precision=HIGHEST)

    def carry(updates, initial):
        flat = updates.reshape(*updates.shape[:3], -1)
        ends = chunk_carry(prefix[..., -1], flat, initial.reshape(batch, heads, -1))
        return ends.reshape(updates.shape)

    ends_s = carry(chunk_evidence, state.evidence)
    ends_c = carry(chunk_cross, state.cross)
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
