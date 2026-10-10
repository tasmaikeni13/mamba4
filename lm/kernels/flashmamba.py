"""FlashMamba: Pallas TPU kernels for the Mamba-3 SISO state-space scan.

Same recurrence as `lm.kernels.mamba3.mamba3_chunked` for rank-one heads. The
forward kernel visits the chunks of one (batch, head) in order and keeps the
[N, P] state in VMEM; every chunk is four MXU products (scores, the masked
mix with the values, the read of the carried state and the state update).
The backward kernel visits the chunks in reverse and carries the state
cotangent the same way, recomputing the chunk products from the inputs and
the saved chunk-start states. Rotary frames and trapezoidal weights stay in
XLA with their autodiff derivatives; only the scan is custom.

Chunk-local log-decay prefixes enter twice, as a column (one row per token)
and as a row, so the decay matrix exp(p_t - p_s) needs no in-kernel
transpose. Matrix operands are in the model dtype with float32 accumulation;
decays, weights and the carried state are float32.
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
TRANS_B = (((1,), (1,)), ((), ()))
F32 = jnp.float32


def _transpose(x):
    """Model-dtype transpose through float32, as Mosaic transposes 32-bit tiles."""
    return x.astype(F32).T.astype(x.dtype)


def _chunk_masks(chunk):
    row = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 0)
    column = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 1)
    return row > column, row == column


def _decay_weights(p_col, p_row, gamma_col, scale_row, lower, diagonal):
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, p_col - p_row, 0.0)), 0.0)
    return decay, decay * scale_row + jnp.where(diagonal, gamma_col, 0.0)


def _forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    p_col_ref,
    p_row_ref,
    gamma_ref,
    scale_ref,
    source_ref,
    y_ref,
    starts_ref,
    final_ref,
    state_ref,
):
    index = pl.program_id(2)
    chunk = q_ref.shape[0]

    @pl.when(index == 0)
    def _():
        state_ref[...] = jnp.zeros_like(state_ref)

    q, k, v = q_ref[...], k_ref[...], v_ref[...]
    p_col = p_col_ref[...]
    lower, diagonal = _chunk_masks(chunk)
    _, weights = _decay_weights(
        p_col, p_row_ref[...], gamma_ref[...], scale_ref[...], lower, diagonal
    )
    scores = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
    within = jnp.dot((scores * weights).astype(v.dtype), v, preferred_element_type=F32)
    state = state_ref[...]
    starts_ref[...] = state
    inherited = jnp.dot(q, state.astype(q.dtype), preferred_element_type=F32)
    y_ref[...] = (within + jnp.exp(p_col) * inherited).astype(y_ref.dtype)
    weighted = (v.astype(F32) * source_ref[...]).astype(v.dtype)
    update = jnp.dot(_transpose(k), weighted, preferred_element_type=F32)
    state = jnp.exp(p_col[chunk - 1 :, :]) * state + update
    state_ref[...] = state

    @pl.when(index == pl.num_programs(2) - 1)
    def _():
        final_ref[...] = state


def _backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    p_col_ref,
    p_row_ref,
    gamma_ref,
    scale_ref,
    source_ref,
    start_ref,
    dy_ref,
    d_final_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dp_col_ref,
    dp_row_ref,
    d_gamma_ref,
    d_scale_col_ref,
    d_scale_row_ref,
    carry_ref,
):
    index = pl.program_id(2)
    chunk = q_ref.shape[0]

    @pl.when(index == 0)
    def _():
        carry_ref[...] = d_final_ref[...]

    q, k, v, dy = q_ref[...], k_ref[...], v_ref[...], dy_ref[...]
    p_col, source = p_col_ref[...], source_ref[...]
    scale_row = scale_ref[...]
    lower, diagonal = _chunk_masks(chunk)
    decay, weights = _decay_weights(
        p_col, p_row_ref[...], gamma_ref[...], scale_row, lower, diagonal
    )
    end = carry_ref[...]  # cotangent of this chunk's end state
    start = start_ref[...]
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

    last = jnp.exp(p_col[chunk - 1 :, :])
    d_start += last * end
    end_low = end.astype(v.dtype)
    v_end = lax.dot_general(v, end_low, TRANS_B, preferred_element_type=F32)
    dk += source * v_end
    dv += source * jnp.dot(k, end_low, preferred_element_type=F32)
    d_source = jnp.sum(k.astype(F32) * v_end, axis=1, keepdims=True)
    d_last = last * jnp.sum(end * start, keepdims=True) + jnp.sum(
        d_source * source, keepdims=True
    )
    d_scale_col_ref[...] = d_source * jnp.exp(p_col[chunk - 1 :, :] - p_col)
    dp_col -= d_source * source

    d_scale_row_ref[...] = jnp.sum(d_weights * decay, axis=0, keepdims=True)
    off = jnp.where(lower, d_weights * weights, 0.0)
    dp_col += jnp.sum(off, axis=1, keepdims=True)
    dp_row_ref[...] = -jnp.sum(off, axis=0, keepdims=True)
    d_gamma_ref[...] = jnp.sum(
        jnp.where(diagonal, d_weights, 0.0), axis=1, keepdims=True
    )
    rows = lax.broadcasted_iota(jnp.int32, (chunk, 1), 0)
    dp_col_ref[...] = dp_col + jnp.where(rows == chunk - 1, d_last, 0.0)
    dq_ref[...] = dq.astype(dq_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)
    carry_ref[...] = d_start


def _specs(batch, heads, chunks, chunk, state_dim, value_dim, reverse=False):
    def at(i):
        return chunks - 1 - i if reverse else i

    def rows(width):
        return pl.BlockSpec(
            (None, None, chunk, width), lambda b, h, i: (b, h, at(i), 0)
        )

    row = pl.BlockSpec(
        (None, None, None, 1, chunk), lambda b, h, i: (b, h, at(i), 0, 0)
    )
    start = pl.BlockSpec(
        (None, None, None, state_dim, value_dim), lambda b, h, i: (b, h, at(i), 0, 0)
    )
    whole = pl.BlockSpec(
        (None, None, state_dim, value_dim), lambda b, h, i: (b, h, 0, 0)
    )
    return rows, row, start, whole


def _layout(q, k, v, log_decay, gamma, scale, chunk):
    """[B,T,H,*] inputs to padded [B,H,T,*] blocks plus prefix rows/columns."""
    batch, length, heads, _ = q.shape
    padding = (-length) % chunk
    chunks = (length + padding) // chunk

    def tokens(x):
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        return jnp.swapaxes(x, 1, 2)

    a, g, s = (tokens(x.astype(F32)) for x in (log_decay, gamma, scale))
    p = jnp.cumsum(a.reshape(batch, heads, chunks, chunk), axis=-1)
    s = s.reshape(p.shape)
    source = s * jnp.exp(p[..., -1:] - p)

    def column(x):
        return x.reshape(batch, heads, chunks * chunk, 1)

    def row(x):
        return x[:, :, :, None, :]

    blocks = (
        tokens(q),
        tokens(k),
        tokens(v),
        column(p),
        row(p),
        column(g),
        row(s),
        column(source),
    )
    return blocks, chunks


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def ssd(q, k, v, log_decay, gamma, scale, chunk=CHUNK, interpret=False):
    """Scan output [B,T,H,P] and final state [B,H,N,P]; inputs as mamba3_fast.ssd."""
    output, final, _ = _ssd_forward_pass(
        q, k, v, log_decay, gamma, scale, chunk, interpret
    )
    return output, final


def _ssd_forward_pass(q, k, v, log_decay, gamma, scale, chunk, interpret):
    batch, length, heads, state_dim = q.shape
    value_dim = v.shape[-1]
    blocks, chunks = _layout(q, k, v, log_decay, gamma, scale, chunk)
    rows, row, start, whole = _specs(batch, heads, chunks, chunk, state_dim, value_dim)
    padded = chunks * chunk
    y, starts, final = pl.pallas_call(
        _forward_kernel,
        grid=(batch, heads, chunks),
        in_specs=[
            rows(state_dim),
            rows(state_dim),
            rows(value_dim),
            rows(1),
            row,
            rows(1),
            row,
            rows(1),
        ],
        out_specs=[rows(value_dim), start, whole],
        out_shape=[
            jax.ShapeDtypeStruct((batch, heads, padded, value_dim), v.dtype),
            jax.ShapeDtypeStruct((batch, heads, chunks, state_dim, value_dim), F32),
            jax.ShapeDtypeStruct((batch, heads, state_dim, value_dim), F32),
        ],
        scratch_shapes=[pltpu.VMEM((state_dim, value_dim), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(*blocks)
    output = jnp.swapaxes(y, 1, 2)[:, :length]
    return output, final, (blocks, starts, length)


def _ssd_forward(q, k, v, log_decay, gamma, scale, chunk, interpret):
    output, final, (blocks, starts, length) = _ssd_forward_pass(
        q, k, v, log_decay, gamma, scale, chunk, interpret
    )
    # A "kernels" remat policy keeps these instead of rerunning the scan.
    output, starts = checkpoint_name((output, starts), "ssd_scan")
    return (output, final), (blocks, starts, length)


def _ssd_backward(chunk, interpret, residuals, cotangents):
    blocks, starts, length = residuals
    d_output, d_final = cotangents
    q_blocks = blocks[0]
    batch, heads, padded, state_dim = q_blocks.shape
    value_dim = blocks[2].shape[-1]
    chunks = padded // chunk
    dy = jnp.pad(
        jnp.swapaxes(d_output.astype(blocks[2].dtype), 1, 2),
        [(0, 0), (0, 0), (0, padded - length), (0, 0)],
    )
    rows, row, start, whole = _specs(
        batch, heads, chunks, chunk, state_dim, value_dim, reverse=True
    )
    shape = lambda *s: jax.ShapeDtypeStruct(s, F32)  # noqa: E731
    dq, dk, dv, dp_col, dp_row, d_gamma, d_scale_col, d_scale_row = pl.pallas_call(
        _backward_kernel,
        grid=(batch, heads, chunks),
        in_specs=[
            rows(state_dim),
            rows(state_dim),
            rows(value_dim),
            rows(1),
            row,
            rows(1),
            row,
            rows(1),
            start,
            rows(value_dim),
            whole,
        ],
        out_specs=[
            rows(state_dim),
            rows(state_dim),
            rows(value_dim),
            rows(1),
            row,
            rows(1),
            rows(1),
            row,
        ],
        out_shape=[
            jax.ShapeDtypeStruct(q_blocks.shape, q_blocks.dtype),
            jax.ShapeDtypeStruct(blocks[1].shape, blocks[1].dtype),
            jax.ShapeDtypeStruct(blocks[2].shape, blocks[2].dtype),
            shape(batch, heads, padded, 1),
            shape(batch, heads, chunks, 1, chunk),
            shape(batch, heads, padded, 1),
            shape(batch, heads, padded, 1),
            shape(batch, heads, chunks, 1, chunk),
        ],
        scratch_shapes=[pltpu.VMEM((state_dim, value_dim), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(*blocks, starts, dy, d_final.astype(F32))

    def tokens(x):
        return jnp.swapaxes(x, 1, 2)[:, :length]

    def per_token(column, row_part=None):
        x = column[..., 0]
        if row_part is not None:
            x = x + row_part[:, :, :, 0, :].reshape(x.shape)
        return x

    dp = per_token(dp_col, dp_row).reshape(batch, heads, chunks, chunk)
    d_log_decay = jnp.cumsum(dp[..., ::-1], axis=-1)[..., ::-1]
    d_log_decay = d_log_decay.reshape(batch, heads, padded)
    return (
        tokens(dq),
        tokens(dk),
        tokens(dv),
        tokens(d_log_decay),
        tokens(per_token(d_gamma)),
        tokens(per_token(d_scale_col, d_scale_row)),
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
):
    """Drop-in for `mamba3_chunked` with rank-one heads ([B,T,H,1,*] shapes)."""
    if q.shape[3] != 1:
        raise ValueError("mamba3_flash supports SISO (rank-one) heads only")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
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
    )
    return output[..., None, :], final
