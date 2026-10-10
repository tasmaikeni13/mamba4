"""FlashMamba: Pallas TPU kernels for the Mamba-3 SISO state-space scan.

Same recurrence as `lm.kernels.mamba3.mamba3_chunked` for rank-one heads. The
forward kernel visits the chunks of one (batch, head) in order and keeps the
[N, P] state in VMEM; every chunk is four MXU products (scores, the masked
mix with the values, the read of the carried state and the state update).
The backward kernel visits the chunks in reverse and carries the state
cotangent the same way, recomputing the chunk products from the inputs and
the saved chunk-start states. Rotary frames and trapezoidal weights stay in
XLA with their autodiff derivatives; only the scan is custom.

Per-token scalars (the chunk-local log-decay prefix, the current and source
weights, the decay to the chunk end) travel as lane-dense rows; a kernel
forms the column it needs by masking the diagonal of a broadcast row, so no
[T, 1] array is padded to 128 lanes and no in-kernel transpose of a vector
is needed. The chunk-end decay arrives replicated across the value lanes.
Scalar gradient bookkeeping (the chunk-end term and the reverse prefix sum)
happens in XLA. Matrix operands are in the model dtype with float32
accumulation; decays, weights and the carried state are float32.
"""

from functools import partial

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from lm.kernels.mamba3 import _rotary_frame

CHUNK = 128
# Most heads one grid step may process; the largest divisor of H up to it is used.
HEADS_PER_STEP = 1
TRANS_B = (((1,), (1,)), ((), ()))
F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST


def _rotate_pairs(x, angles):
    """`lm.kernels.mamba3.rotate` with pairwise=True, without size-2 axes.

    Adjacent coordinates (2i, 2i+1) rotate by angle i; coordinates beyond the
    angles stay fixed. Each coordinate's partner comes from a lane roll, so
    no [..., N/2, 2] intermediate is formed in either direction of autodiff.
    """
    width = x.shape[-1]
    angles = jnp.pad(
        angles, [(0, 0)] * (angles.ndim - 1) + [(0, width // 2 - angles.shape[-1])]
    )
    angles = jnp.repeat(angles, 2, axis=-1)
    x32 = x.astype(F32)
    even = jnp.arange(width) % 2 == 0
    partner = jnp.where(even, -jnp.roll(x32, -1, axis=-1), jnp.roll(x32, 1, axis=-1))
    return (x32 * jnp.cos(angles) + partner * jnp.sin(angles)).astype(x.dtype)


def prefix_sum(x, chunk=CHUNK):
    """Inclusive cumulative sum over axis 1 by chunked triangular products.

    XLA lowers jnp.cumsum on TPU to a full-length reduce-window whose reverse
    (its transpose) runs in quadratic time. Here each chunk's prefix is one
    float32-accurate product with a lower-triangular matrix of ones, and chunk
    totals carry forward by a short cumulative sum; the transpose is the same
    kind of product.
    """
    length = x.shape[1]
    padding = (-length) % chunk
    padded = jnp.pad(x.astype(F32), [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
    blocks = padded.reshape(x.shape[0], -1, chunk, *x.shape[2:])
    ones = jnp.tril(jnp.ones((chunk, chunk), F32))
    within = jnp.einsum("ts,bns...->bnt...", ones, blocks, precision=HIGHEST)
    totals = within[:, :, -1]
    carried = jnp.cumsum(totals, axis=1) - totals
    output = within + carried[:, :, None]
    return output.reshape(padded.shape)[:, :length]


def _rotary_frame_pairs(q, k, dt, angles, q_bias, k_bias):
    """`lm.kernels.mamba3._rotary_frame` for pairwise rotation."""
    if q_bias is not None:
        q = (q.astype(F32) + q_bias).astype(q.dtype)
    if k_bias is not None:
        k = (k.astype(F32) + k_bias).astype(k.dtype)
    phase = prefix_sum(angles.astype(F32) * dt[..., None])[..., None, :]
    return _rotate_pairs(q, phase), _rotate_pairs(k, phase)


def _transpose(x):
    """Model-dtype transpose through float32, as Mosaic transposes 32-bit tiles."""
    return x.astype(F32).T.astype(x.dtype)


def _masks(chunk):
    row = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 0)
    column = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 1)
    return row > column, row == column


def _column(row, diagonal):
    """A [1, c] row as a [c, 1] column, read off the diagonal."""
    return jnp.sum(jnp.where(diagonal, row, 0.0), axis=1, keepdims=True)


def _row(column, diagonal):
    """A [c, 1] column as a [1, c] row, read off the diagonal."""
    return jnp.sum(jnp.where(diagonal, column, 0.0), axis=0, keepdims=True)


def _chunk_terms(p_row, gamma_row, scale_row, tail_row, lower, diagonal):
    p_col = _column(p_row, diagonal)
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, p_col - p_row, 0.0)), 0.0)
    weights = decay * scale_row + jnp.where(diagonal, gamma_row, 0.0)
    source_col = _column(scale_row * tail_row, diagonal)
    return p_col, decay, weights, source_col


def _forward_head(
    index,
    last_index,
    q_ref,
    k_ref,
    v_ref,
    p_ref,
    gamma_ref,
    scale_ref,
    tail_ref,
    end_ref,
    y_ref,
    starts_ref,
    final_ref,
    state_ref,
):
    chunk = q_ref.shape[0]

    @pl.when(index == 0)
    def _():
        state_ref[...] = jnp.zeros(state_ref.shape, F32)

    q, k, v = q_ref[...], k_ref[...], v_ref[...]
    lower, diagonal = _masks(chunk)
    p_col, _, weights, source_col = _chunk_terms(
        p_ref[...], gamma_ref[...], scale_ref[...], tail_ref[...], lower, diagonal
    )
    scores = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
    within = jnp.dot((scores * weights).astype(v.dtype), v, preferred_element_type=F32)
    state = state_ref[...]
    starts_ref[...] = state
    inherited = jnp.dot(q, state.astype(q.dtype), preferred_element_type=F32)
    y_ref[...] = (within + jnp.exp(p_col) * inherited).astype(y_ref.dtype)
    weighted = (v.astype(F32) * source_col).astype(v.dtype)
    update = jnp.dot(_transpose(k), weighted, preferred_element_type=F32)
    state = end_ref[...] * state + update
    state_ref[...] = state

    @pl.when(index == last_index)
    def _():
        final_ref[...] = state


def _forward_kernel(*refs):
    """One grid step: a chunk of every head in a block of heads."""
    index, last_index = pl.program_id(2), pl.num_programs(2) - 1
    for head in range(refs[0].shape[0]):
        _forward_head(index, last_index, *(ref.at[head] for ref in refs))


def _backward_head(
    index,
    q_ref,
    k_ref,
    v_ref,
    p_ref,
    gamma_ref,
    scale_ref,
    tail_ref,
    end_ref,
    start_ref,
    dy_ref,
    d_final_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dp_ref,
    d_gamma_ref,
    d_scale_ref,
    flow_ref,
    d_end_ref,
    carry_ref,
):
    chunk = q_ref.shape[0]

    @pl.when(index == 0)
    def _():
        carry_ref[...] = d_final_ref[...]

    q, k, v, dy = q_ref[...], k_ref[...], v_ref[...], dy_ref[...]
    scale_row, tail_row = scale_ref[...], tail_ref[...]
    lower, diagonal = _masks(chunk)
    p_col, decay, weights, source_col = _chunk_terms(
        p_ref[...], gamma_ref[...], scale_row, tail_row, lower, diagonal
    )
    end = carry_ref[...]  # cotangent of this chunk's end state
    start = start_ref[...]
    end_decay = end_ref[...]
    scores = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
    d_mixed = lax.dot_general(dy, v, TRANS_B, preferred_element_type=F32)
    mixed = scores * weights
    dv = jnp.dot(mixed.T.astype(v.dtype), dy, preferred_element_type=F32)
    d_scores = d_mixed * weights
    dq = jnp.dot(d_scores.astype(q.dtype), k, preferred_element_type=F32)
    dk = jnp.dot(d_scores.T.astype(q.dtype), q, preferred_element_type=F32)
    d_weights = d_mixed * scores

    carried = jnp.exp(p_col)
    scaled_dy = (dy.astype(F32) * carried).astype(dy.dtype)
    dq += lax.dot_general(
        scaled_dy, start.astype(q.dtype), TRANS_B, preferred_element_type=F32
    )
    inherited = jnp.dot(q, start.astype(q.dtype), preferred_element_type=F32)
    dp_col = carried * jnp.sum(dy.astype(F32) * inherited, axis=1, keepdims=True)
    d_start = jnp.dot(_transpose(q), scaled_dy, preferred_element_type=F32)
    d_start += end_decay * end

    end_low = end.astype(v.dtype)
    v_end = lax.dot_general(v, end_low, TRANS_B, preferred_element_type=F32)
    dk += source_col * v_end
    dv += source_col * jnp.dot(k, end_low, preferred_element_type=F32)
    d_source = jnp.sum(k.astype(F32) * v_end, axis=1, keepdims=True)
    # source = scale * tail with tail_s = exp(p_last - p_s): this flow leaves
    # p_s and reaches p_last.
    flow = d_source * source_col
    off = jnp.where(lower, d_weights * weights, 0.0)
    dp_col += jnp.sum(off, axis=1, keepdims=True) - flow
    dp_ref[...] = _row(dp_col, diagonal) - jnp.sum(off, axis=0, keepdims=True)
    d_gamma_ref[...] = jnp.sum(
        jnp.where(diagonal, d_weights, 0.0), axis=0, keepdims=True
    )
    tail_col = _column(tail_row, diagonal)
    d_scale_ref[...] = _row(d_source * tail_col, diagonal) + jnp.sum(
        d_weights * decay, axis=0, keepdims=True
    )
    flow_ref[...] = _row(flow, diagonal)
    d_end_ref[...] = end_decay * jnp.sum(end * start, axis=0, keepdims=True)
    dq_ref[...] = dq.astype(dq_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)
    carry_ref[...] = d_start


def _backward_kernel(*refs):
    """One reverse grid step: a chunk of every head in a block of heads."""
    index = pl.program_id(2)
    for head in range(refs[0].shape[0]):
        _backward_head(index, *(ref.at[head] for ref in refs))


def _specs(heads, chunks, chunk, state_dim, value_dim, reverse=False):
    """Block specs over a (batch, head block, chunk) grid; heads is the block."""

    def at(i):
        return chunks - 1 - i if reverse else i

    def tokens(width):
        return pl.BlockSpec(
            (None, heads, chunk, width), lambda b, h, i: (b, h, at(i), 0)
        )

    def row(width=chunk):
        return pl.BlockSpec(
            (None, heads, None, 1, width), lambda b, h, i: (b, h, at(i), 0, 0)
        )

    start = pl.BlockSpec(
        (None, heads, None, state_dim, value_dim),
        lambda b, h, i: (b, h, at(i), 0, 0),
    )
    whole = pl.BlockSpec(
        (None, heads, state_dim, value_dim), lambda b, h, i: (b, h, 0, 0)
    )
    return tokens, row, start, whole


def _layout(q, k, v, log_decay, gamma, scale, chunk):
    """[B,T,H,*] inputs to padded [B,H,T,*] blocks plus per-chunk rows."""
    batch, length, heads, _ = q.shape
    value_dim = v.shape[-1]
    padding = (-length) % chunk
    chunks = (length + padding) // chunk

    def tokens(x):
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        return jnp.swapaxes(x, 1, 2)

    def rows(x):
        return x.reshape(batch, heads, chunks, 1, chunk)

    a, g, s = (tokens(x.astype(F32)) for x in (log_decay, gamma, scale))
    p = jnp.cumsum(a.reshape(batch, heads, chunks, chunk), axis=-1)
    tail = jnp.exp(p[..., -1:] - p)
    end = jnp.broadcast_to(
        jnp.exp(p[..., -1])[..., None, None], (batch, heads, chunks, 1, value_dim)
    )
    blocks = (
        tokens(q),
        tokens(k),
        tokens(v),
        rows(p),
        rows(g),
        rows(s),
        rows(tail),
        end,
    )
    return blocks, chunks


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def ssd(
    q, k, v, log_decay, gamma, scale, chunk=CHUNK, interpret=False, heads_per_step=1
):
    """Scan output [B,T,H,P] and final state [B,H,N,P]; inputs as mamba3_fast.ssd.

    heads_per_step heads share one grid step (it must divide H).
    """
    output, final, _ = _ssd_forward_pass(
        q, k, v, log_decay, gamma, scale, chunk, interpret, heads_per_step
    )
    return output, final


def _ssd_forward_pass(
    q, k, v, log_decay, gamma, scale, chunk, interpret, heads_per_step
):
    batch, length, heads, state_dim = q.shape
    value_dim = v.shape[-1]
    if heads % heads_per_step:
        raise ValueError("heads_per_step must divide the number of heads")
    blocks, chunks = _layout(q, k, v, log_decay, gamma, scale, chunk)
    tokens, row, start, whole = _specs(
        heads_per_step, chunks, chunk, state_dim, value_dim
    )
    padded = chunks * chunk
    y, starts, final = pl.pallas_call(
        _forward_kernel,
        grid=(batch, heads // heads_per_step, chunks),
        in_specs=[
            tokens(state_dim),
            tokens(state_dim),
            tokens(value_dim),
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
        ],
        out_specs=[tokens(value_dim), start, whole],
        out_shape=[
            jax.ShapeDtypeStruct((batch, heads, padded, value_dim), v.dtype),
            jax.ShapeDtypeStruct((batch, heads, chunks, state_dim, value_dim), F32),
            jax.ShapeDtypeStruct((batch, heads, state_dim, value_dim), F32),
        ],
        scratch_shapes=[pltpu.VMEM((heads_per_step, state_dim, value_dim), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(*blocks)
    output = jnp.swapaxes(y, 1, 2)[:, :length]
    return output, final, (blocks, starts, length)


def _ssd_forward(q, k, v, log_decay, gamma, scale, chunk, interpret, heads_per_step):
    output, final, (blocks, starts, length) = _ssd_forward_pass(
        q, k, v, log_decay, gamma, scale, chunk, interpret, heads_per_step
    )
    # A "kernels" remat policy keeps these instead of rerunning the scan.
    output = checkpoint_name(output, "ssd_scan")
    starts = checkpoint_name(starts, "ssd_scan")
    return (output, final), (blocks, starts, length)


def _ssd_backward(chunk, interpret, heads_per_step, residuals, cotangents):
    blocks, starts, length = residuals
    d_output, d_final = cotangents
    q_blocks, k_blocks, v_blocks = blocks[:3]
    batch, heads, padded, state_dim = q_blocks.shape
    value_dim = v_blocks.shape[-1]
    chunks = padded // chunk
    dy = jnp.pad(
        jnp.swapaxes(d_output.astype(v_blocks.dtype), 1, 2),
        [(0, 0), (0, 0), (0, padded - length), (0, 0)],
    )
    tokens, row, start, whole = _specs(
        heads_per_step, chunks, chunk, state_dim, value_dim, reverse=True
    )
    rows_shape = jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk), F32)
    dq, dk, dv, dp, d_gamma, d_scale, flow, d_end = pl.pallas_call(
        _backward_kernel,
        grid=(batch, heads // heads_per_step, chunks),
        in_specs=[
            tokens(state_dim),
            tokens(state_dim),
            tokens(value_dim),
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
            start,
            tokens(value_dim),
            whole,
        ],
        out_specs=[
            tokens(state_dim),
            tokens(state_dim),
            tokens(value_dim),
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(q_blocks.shape, q_blocks.dtype),
            jax.ShapeDtypeStruct(k_blocks.shape, k_blocks.dtype),
            jax.ShapeDtypeStruct(v_blocks.shape, v_blocks.dtype),
            rows_shape,
            rows_shape,
            rows_shape,
            rows_shape,
            jax.ShapeDtypeStruct((batch, heads, chunks, 1, value_dim), F32),
        ],
        scratch_shapes=[pltpu.VMEM((heads_per_step, state_dim, value_dim), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(*blocks, starts, dy, d_final.astype(F32))

    def tokens_out(x):
        return jnp.swapaxes(x, 1, 2)[:, :length]

    def per_token(x):
        return x.reshape(batch, heads, padded)

    # Every chunk's last position also carries the chunk-end decay's gradient.
    d_last = jnp.sum(flow[..., 0, :], axis=-1) + jnp.sum(d_end[..., 0, :], axis=-1)
    dp = dp[..., 0, :].at[..., -1].add(d_last)
    d_log_decay = jnp.cumsum(dp[..., ::-1], axis=-1)[..., ::-1]
    return (
        tokens_out(dq),
        tokens_out(dk),
        tokens_out(dv),
        tokens_out(per_token(d_log_decay)),
        tokens_out(per_token(d_gamma)),
        tokens_out(per_token(d_scale)),
    )


ssd.defvjp(_ssd_forward, _ssd_backward)


def mamba3_flash(
    q,
    k,
    v,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    chunk_size=CHUNK,
    q_bias=None,
    k_bias=None,
    pairwise=True,
    interpret=None,
    heads_per_step=None,
):
    """Drop-in for `mamba3_chunked` with rank-one heads ([B,T,H,1,*] shapes)."""
    if q.shape[3] != 1:
        raise ValueError("mamba3_flash supports SISO (rank-one) heads only")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    if heads_per_step is None:
        heads = q.shape[2]
        heads_per_step = max(
            size for size in range(1, HEADS_PER_STEP + 1) if heads % size == 0
        )
    if pairwise:
        q, k = _rotary_frame_pairs(q, k, dt, angles, q_bias, k_bias)
    else:
        q, k, _ = _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise)
    trap = jax.nn.sigmoid(trap_logits.astype(F32))
    gamma = dt.astype(F32) * trap
    previous_weight = dt.astype(F32) * (1 - trap)
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    output, final = ssd(
        q[..., 0, :],
        k[..., 0, :],
        v[..., 0, :],
        adt,
        gamma,
        scale,
        chunk_size,
        interpret,
        heads_per_step,
    )
    return output[..., None, :], final
