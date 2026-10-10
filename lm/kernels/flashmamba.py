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
import numpy as np

from lm.kernels.prefix import cumulative, prefix_sum


CHUNK = 128
# Most heads one grid step may process; the largest divisor of H up to it is used.
HEADS_PER_STEP = 1
TRANS_B = (((1,), (1,)), ((), ()))
F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST


def _pair_matrices(width, count):
    """Signed swap of adjacent coordinates and duplication of each angle.

    partner = x @ swap gives -x[2i+1] at 2i and x[2i] at 2i+1, and
    angles @ duplicate repeats angle i at 2i and 2i+1. Entries are 0 and +-1,
    so both products are exact.
    """
    swap = np.zeros((width, width), np.float32)
    for i in range(0, width, 2):
        swap[i + 1, i] = -1.0
        swap[i, i + 1] = 1.0
    duplicate = np.zeros((count, width), np.float32)
    for i in range(count):
        duplicate[i, 2 * i] = duplicate[i, 2 * i + 1] = 1.0
    fixed = np.zeros(width, np.float32)
    fixed[2 * count :] = 1.0
    return swap, duplicate, fixed


def _rotate_pairs(x, angles):
    """`lm.kernels.mamba3.rotate` with pairwise=True as exact matrix products.

    Adjacent coordinates (2i, 2i+1) rotate by angle i; coordinates beyond the
    angles stay fixed. The partner coordinate and the per-coordinate cosine
    and sine come from fixed 0/+-1 matrices, so no lane shuffle, repeat or
    narrow slice appears in either direction of autodiff.
    """
    width, count = x.shape[-1], angles.shape[-1]
    swap, duplicate, fixed = _pair_matrices(width, count)
    precision = HIGHEST if x.dtype == F32 else lax.Precision.DEFAULT
    partner = jnp.matmul(
        x, jnp.asarray(swap, x.dtype), precision=precision, preferred_element_type=F32
    )
    duplicate = jnp.asarray(duplicate)
    cosine = jnp.matmul(jnp.cos(angles), duplicate, precision=HIGHEST) + fixed
    sine = jnp.matmul(jnp.sin(angles), duplicate, precision=HIGHEST)
    return (x.astype(F32) * cosine + partner * sine).astype(x.dtype)


def _rotary_frame_pairs(q, k, dt, angles, q_bias, k_bias):
    """`lm.kernels.mamba3._rotary_frame` for pairwise rotation of [B,T,H,N]
    arrays, with angle rates [B,T,H,A] and biases [H,N]."""
    if q_bias is not None:
        q = (q.astype(F32) + q_bias).astype(q.dtype)
    if k_bias is not None:
        k = (k.astype(F32) + k_bias).astype(k.dtype)
    phase = prefix_sum(angles.astype(F32) * dt[..., None])
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
    p = cumulative(a.reshape(batch, heads, chunks, chunk))
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
    d_log_decay = cumulative(dp, reverse=True)
    return (
        tokens_out(dq),
        tokens_out(dk),
        tokens_out(dv),
        tokens_out(per_token(d_log_decay)),
        tokens_out(per_token(d_gamma)),
        tokens_out(per_token(d_scale)),
    )


ssd.defvjp(_ssd_forward, _ssd_backward)


def mamba3_flash_heads(
    q,
    k,
    v,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    q_bias=None,
    k_bias=None,
    chunk_size=CHUNK,
    interpret=None,
    heads_per_step=None,
):
    """SISO Mamba-3 scan on [B,T,H,N] queries and keys and [B,T,H,P] values.

    angles [B,T,H,A] are rotation rates and q_bias, k_bias [H,N]; rotation is
    pairwise. Returns [B,T,H,P] outputs and the [B,H,N,P] final state. No
    array carries a singleton rank axis, which TPU tiling would pad 8-fold.
    """
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    if heads_per_step is None:
        heads = q.shape[2]
        heads_per_step = max(
            size for size in range(1, HEADS_PER_STEP + 1) if heads % size == 0
        )
    q, k = _rotary_frame_pairs(q, k, dt, angles, q_bias, k_bias)
    trap = jax.nn.sigmoid(trap_logits.astype(F32))
    gamma = dt.astype(F32) * trap
    previous_weight = dt.astype(F32) * (1 - trap)
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    return ssd(q, k, v, adt, gamma, scale, chunk_size, interpret, heads_per_step)


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
    if not pairwise:
        raise ValueError("mamba3_flash implements pairwise rotation")
    output, final = mamba3_flash_heads(
        q[..., 0, :],
        k[..., 0, :],
        v[..., 0, :],
        adt,
        dt,
        trap_logits,
        angles,
        q_bias=None if q_bias is None else q_bias[:, 0],
        k_bias=None if k_bias is None else k_bias[:, 0],
        chunk_size=chunk_size,
        interpret=interpret,
        heads_per_step=heads_per_step,
    )
    return output[..., None, :], final


# ---------------------------------------------------------------------------
# Layout-native fused path: rotation, phase and scan in one kernel pair.
# ---------------------------------------------------------------------------


def _mask_dot(mask, x):
    """mask @ x for a 0/1 mask, exact to float32 accumulation.

    x splits into three bfloat16 parts that sum to it; each product with the
    exactly representable mask is exact, and the MXU accumulates in float32.
    In-kernel float32-precision products crash this Mosaic version.
    """
    weights = mask.astype(jnp.bfloat16)
    high = x.astype(jnp.bfloat16)
    rest = x - high.astype(F32)
    middle = rest.astype(jnp.bfloat16)
    low = (rest - middle.astype(F32)).astype(jnp.bfloat16)
    exact = dict(precision=lax.Precision.DEFAULT, preferred_element_type=F32)
    total = jnp.dot(weights, high, **exact)
    total += jnp.dot(weights, middle, **exact)
    return total + jnp.dot(weights, low, **exact)


def _partner(x):
    """(-x[2i+1] at 2i, x[2i] at 2i+1) from two lane rotations (exact)."""
    width = x.shape[-1]
    lanes = lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    later = pltpu.roll(x, width - 1, x.ndim - 1)
    earlier = pltpu.roll(x, 1, x.ndim - 1)
    return jnp.where(lanes % 2 == 0, -later, earlier)


def _rotated(c_ref, b_ref, qb, kb, theta):
    """Bias-added, model-dtype-rounded queries and keys rotated by theta.

    Returns the model-dtype operands and their float32 values before
    rounding (the derivative of the rotation needs the latter).
    """
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    q = (c_ref[...].astype(F32) + qb).astype(c_ref.dtype).astype(F32)
    k = (b_ref[...].astype(F32) + kb).astype(b_ref.dtype).astype(F32)
    q_rot = q * cosine + _partner(q) * sine
    k_rot = k * cosine + _partner(k) * sine
    return q_rot, k_rot, cosine, sine


def _fused_forward_kernel(
    c_ref,
    b_ref,
    x_ref,
    angle_ref,
    qb_ref,
    kb_ref,
    p_ref,
    gamma_ref,
    scale_ref,
    tail_ref,
    end_ref,
    dt_ref,
    y_ref,
    starts_ref,
    phase0_ref,
    final_ref,
    state_ref,
    phase_ref,
):
    index, block = pl.program_id(1), pl.program_id(2)
    last_index = pl.num_programs(1) - 1
    heads, chunk = qb_ref.shape[0], c_ref.shape[0]
    width = x_ref.shape[-1] // heads

    @pl.when(index == 0)
    def _():
        state_ref[block] = jnp.zeros(state_ref.shape[1:], F32)
        phase_ref[block] = jnp.zeros(phase_ref.shape[1:], F32)

    lower, diagonal = _masks(chunk)
    inclusive = jnp.where(lower | diagonal, 1.0, 0.0)
    last_row = lax.broadcasted_iota(jnp.int32, (chunk, 1), 0) == chunk - 1
    angles = angle_ref[...]
    for head in range(heads):
        dt_col = _column(dt_ref[head], diagonal)
        start_phase = phase_ref[block, head]
        theta = start_phase + _mask_dot(inclusive, angles * dt_col)
        phase0_ref[head] = start_phase
        # The last row by selection: a slice would keep sublane offset 7,
        # which this Mosaic version cannot store into the [1, N] carry.
        phase_ref[block, head] = jnp.sum(
            jnp.where(last_row, theta, 0.0), axis=0, keepdims=True
        )
        q_rot, k_rot, _, _ = _rotated(c_ref, b_ref, qb_ref[head], kb_ref[head], theta)
        q, k = q_rot.astype(c_ref.dtype), k_rot.astype(b_ref.dtype)
        lanes = slice(head * width, (head + 1) * width)
        v = x_ref[:, lanes]
        p_col, _, weights, source_col = _chunk_terms(
            p_ref[head],
            gamma_ref[head],
            scale_ref[head],
            tail_ref[head],
            lower,
            diagonal,
        )
        scores = lax.dot_general(q, k, TRANS_B, preferred_element_type=F32)
        within = jnp.dot(
            (scores * weights).astype(v.dtype), v, preferred_element_type=F32
        )
        state = state_ref[block, head]
        starts_ref[head] = state
        inherited = jnp.dot(q, state.astype(q.dtype), preferred_element_type=F32)
        y_ref[:, lanes] = (within + jnp.exp(p_col) * inherited).astype(y_ref.dtype)
        weighted = (v.astype(F32) * source_col).astype(v.dtype)
        update = jnp.dot(_transpose(k), weighted, preferred_element_type=F32)
        state = end_ref[head] * state + update
        state_ref[block, head] = state

        @pl.when(index == last_index)
        def _():
            final_ref[head] = state


def _fused_backward_kernel(
    c_ref,
    b_ref,
    x_ref,
    angle_ref,
    qb_ref,
    kb_ref,
    p_ref,
    gamma_ref,
    scale_ref,
    tail_ref,
    end_ref,
    dt_ref,
    start_ref,
    phase0_ref,
    dy_ref,
    d_final_ref,
    dc_ref,
    db_ref,
    dx_ref,
    d_angle_ref,
    dqb_ref,
    dkb_ref,
    dp_ref,
    d_gamma_ref,
    d_scale_ref,
    flow_ref,
    d_end_ref,
    d_dt_ref,
    carry_ref,
    later_ref,
):
    index, block = pl.program_id(1), pl.program_id(2)
    heads, chunk = qb_ref.shape[0], c_ref.shape[0]
    width = x_ref.shape[-1] // heads

    @pl.when(index == 0)
    def _():
        carry_ref[block] = d_final_ref[...]
        later_ref[block] = jnp.zeros(later_ref.shape[1:], F32)

    # Query, key and angle gradients sum over every head of the chunk.
    @pl.when(block == 0)
    def _():
        dc_ref[...] = jnp.zeros(dc_ref.shape, F32)
        db_ref[...] = jnp.zeros(db_ref.shape, F32)
        d_angle_ref[...] = jnp.zeros(d_angle_ref.shape, F32)

    lower, diagonal = _masks(chunk)
    inclusive = jnp.where(lower | diagonal, 1.0, 0.0)
    suffix = jnp.where(lower, 0.0, 1.0)  # [t, s] = 1 for s >= t
    angles = angle_ref[...]
    for head in range(heads):
        dt_col = _column(dt_ref[head], diagonal)
        theta = phase0_ref[head] + _mask_dot(inclusive, angles * dt_col)
        q_rot, k_rot, cosine, sine = _rotated(
            c_ref, b_ref, qb_ref[head], kb_ref[head], theta
        )
        q, k = q_rot.astype(c_ref.dtype), k_rot.astype(b_ref.dtype)
        lanes = slice(head * width, (head + 1) * width)
        v, dy = x_ref[:, lanes], dy_ref[:, lanes]
        scale_row, tail_row = scale_ref[head], tail_ref[head]
        p_col, decay, weights, source_col = _chunk_terms(
            p_ref[head], gamma_ref[head], scale_row, tail_row, lower, diagonal
        )
        end = carry_ref[block, head]
        start = start_ref[head]
        end_decay = end_ref[head]
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
        flow = d_source * source_col
        off = jnp.where(lower, d_weights * weights, 0.0)
        dp_col += jnp.sum(off, axis=1, keepdims=True) - flow
        dp_ref[head] = _row(dp_col, diagonal) - jnp.sum(off, axis=0, keepdims=True)
        d_gamma_ref[head] = jnp.sum(
            jnp.where(diagonal, d_weights, 0.0), axis=0, keepdims=True
        )
        d_scale_ref[head] = _row(d_source * _column(tail_row, diagonal), diagonal) + (
            jnp.sum(d_weights * decay, axis=0, keepdims=True)
        )
        flow_ref[head] = _row(flow, diagonal)
        d_end_ref[head] = end_decay * jnp.sum(end * start, axis=0, keepdims=True)
        dx_ref[:, lanes] = dv.astype(dx_ref.dtype)
        carry_ref[block, head] = d_start

        # Through the rotation: q_rot = R(theta) q, so dq_pre = R(theta)^T dq
        # and d theta = dq . J q_rot, with J the pairwise quarter turn.
        dq_pre = dq * cosine - _partner(dq) * sine
        dk_pre = dk * cosine - _partner(dk) * sine
        dc_ref[...] += dq_pre
        db_ref[...] += dk_pre
        dqb_ref[head] = jnp.sum(dq_pre, axis=0, keepdims=True)
        dkb_ref[head] = jnp.sum(dk_pre, axis=0, keepdims=True)
        d_theta = dq * _partner(q_rot) + dk * _partner(k_rot)
        # theta_t sums earlier increments, so each increment collects the
        # phase gradient of its own and every later position.
        d_increment = later_ref[block, head] + _mask_dot(suffix, d_theta)
        later_ref[block, head] = later_ref[block, head] + jnp.sum(
            d_theta, axis=0, keepdims=True
        )
        d_angle_ref[...] += d_increment * dt_col
        d_dt_ref[head] = _row(
            jnp.sum(d_increment * angles, axis=1, keepdims=True), diagonal
        )


def _fused_specs(heads, width, chunks, chunk, state_dim, value_dim, reverse=False):
    """Specs over a (batch, chunk, head block) grid; heads is the block size."""

    def at(i):
        return chunks - 1 - i if reverse else i

    shared = pl.BlockSpec((None, chunk, state_dim), lambda b, i, h: (b, at(i), 0))
    values = pl.BlockSpec((None, chunk, heads * width), lambda b, i, h: (b, at(i), h))
    bias = pl.BlockSpec((heads, 1, state_dim), lambda b, i, h: (h, 0, 0))

    def row(length=chunk):
        return pl.BlockSpec(
            (None, heads, None, 1, length), lambda b, i, h: (b, h, at(i), 0, 0)
        )

    start = pl.BlockSpec(
        (None, heads, None, state_dim, value_dim), lambda b, i, h: (b, h, at(i), 0, 0)
    )
    whole = pl.BlockSpec(
        (None, heads, state_dim, value_dim), lambda b, i, h: (b, h, 0, 0)
    )
    partial_bias = pl.BlockSpec(
        (None, None, heads, 1, state_dim), lambda b, i, h: (b, at(i), h, 0, 0)
    )
    return shared, values, bias, row, start, whole, partial_bias


def _fused_layout(c, b, x, angles, dt, log_decay, gamma, scale, chunk):
    batch, length, state_dim = c.shape
    heads = dt.shape[-1]
    value_dim = x.shape[-1] // heads
    padding = (-length) % chunk
    chunks = (length + padding) // chunk

    def tokens(array):
        return jnp.pad(array, [(0, 0), (0, padding)] + [(0, 0)] * (array.ndim - 2))

    def rows(array):
        array = jnp.swapaxes(tokens(array.astype(F32)), 1, 2)
        return array.reshape(batch, heads, chunks, 1, chunk)

    a = rows(log_decay)
    p = cumulative(a[..., 0, :])[..., None, :]
    tail = jnp.exp(p[..., -1:] - p)
    end = jnp.broadcast_to(
        jnp.exp(p[..., 0, -1])[..., None, None], (batch, heads, chunks, 1, value_dim)
    )
    inputs = (
        tokens(c),
        tokens(b),
        tokens(x),
        tokens(angles.astype(F32)),
        p,
        rows(gamma),
        rows(scale),
        tail,
        end,
        rows(dt),
    )
    return inputs, chunks


def _block_size(heads, value_dim, limit):
    """Heads per grid step: divides H and fills whole 128-lane value tiles."""
    for size in range(1, heads + 1):
        if heads % size == 0 and (size * value_dim) % 128 == 0 and size >= limit:
            return size
    return heads


@partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12))
def fused_scan(
    c,
    b,
    x,
    q_bias,
    k_bias,
    angles,
    dt,
    log_decay,
    gamma,
    scale,
    chunk=CHUNK,
    interpret=False,
    heads_per_step=2,
):
    """Rotated SISO Mamba-3 scan in the projection layout.

    c, b [B,T,N]: queries and keys shared by every head before their head's
    bias [H,1,N] and rotation; x [B,T,H*P] values; angles [B,T,N] per-pair
    angle rates already duplicated onto both coordinates of each pair (zero
    on fixed coordinates); dt, log_decay, gamma, scale [B,T,H]. The phase
    is the running sum of angles * dt. Returns y [B,T,H*P], final [B,H,N,P].
    """
    y, final, _ = _fused_forward_pass(
        c,
        b,
        x,
        q_bias,
        k_bias,
        angles,
        dt,
        log_decay,
        gamma,
        scale,
        chunk,
        interpret,
        heads_per_step,
    )
    return y, final


def _fused_forward_pass(
    c,
    b,
    x,
    q_bias,
    k_bias,
    angles,
    dt,
    log_decay,
    gamma,
    scale,
    chunk,
    interpret,
    heads_per_step,
):
    batch, length, state_dim = c.shape
    heads = dt.shape[-1]
    value_dim = x.shape[-1] // heads
    hb = heads_per_step
    inputs, chunks = _fused_layout(c, b, x, angles, dt, log_decay, gamma, scale, chunk)
    shared, values, bias, row, start, whole, _ = _fused_specs(
        hb, value_dim, chunks, chunk, state_dim, value_dim
    )
    padded = chunks * chunk
    y, starts, phase0, final = pl.pallas_call(
        _fused_forward_kernel,
        grid=(batch, chunks, heads // hb),
        in_specs=[
            shared,
            shared,
            values,
            shared,
            bias,
            bias,
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
            row(),
        ],
        out_specs=[values, start, row(state_dim), whole],
        out_shape=[
            jax.ShapeDtypeStruct((batch, padded, heads * value_dim), x.dtype),
            jax.ShapeDtypeStruct((batch, heads, chunks, state_dim, value_dim), F32),
            jax.ShapeDtypeStruct((batch, heads, chunks, 1, state_dim), F32),
            jax.ShapeDtypeStruct((batch, heads, state_dim, value_dim), F32),
        ],
        scratch_shapes=[
            pltpu.VMEM((heads // hb, hb, state_dim, value_dim), F32),
            pltpu.VMEM((heads // hb, hb, 1, state_dim), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
    )(*inputs[:4], q_bias.astype(F32), k_bias.astype(F32), *inputs[4:])
    residuals = (inputs, q_bias, k_bias, starts, phase0, length)
    return y[:, :length], final, residuals


def _fused_forward(
    c,
    b,
    x,
    q_bias,
    k_bias,
    angles,
    dt,
    log_decay,
    gamma,
    scale,
    chunk,
    interpret,
    heads_per_step,
):
    y, final, residuals = _fused_forward_pass(
        c,
        b,
        x,
        q_bias,
        k_bias,
        angles,
        dt,
        log_decay,
        gamma,
        scale,
        chunk,
        interpret,
        heads_per_step,
    )
    y = checkpoint_name(y, "ssd_scan")
    return (y, final), residuals


def _fused_backward(chunk, interpret, heads_per_step, residuals, cotangents):
    inputs, q_bias, k_bias, starts, phase0, length = residuals
    d_y, d_final = cotangents
    c, b, x = inputs[:3]
    batch, padded, state_dim = c.shape
    heads = q_bias.shape[0]
    value_dim = x.shape[-1] // heads
    chunks = padded // chunk
    hb = heads_per_step
    shared, values, bias, row, start, whole, partial_bias = _fused_specs(
        hb, value_dim, chunks, chunk, state_dim, value_dim, reverse=True
    )
    dy = jnp.pad(d_y.astype(x.dtype), [(0, 0), (0, padded - length), (0, 0)])
    rows_shape = jax.ShapeDtypeStruct((batch, heads, chunks, 1, chunk), F32)
    shared_shape = jax.ShapeDtypeStruct((batch, padded, state_dim), F32)
    bias_shape = jax.ShapeDtypeStruct((batch, chunks, heads, 1, state_dim), F32)
    outputs = pl.pallas_call(
        _fused_backward_kernel,
        grid=(batch, chunks, heads // hb),
        in_specs=[
            shared,
            shared,
            values,
            shared,
            bias,
            bias,
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
            row(),
            start,
            row(state_dim),
            values,
            whole,
        ],
        out_specs=[
            shared,
            shared,
            values,
            shared,
            partial_bias,
            partial_bias,
            row(),
            row(),
            row(),
            row(),
            row(value_dim),
            row(),
        ],
        out_shape=[
            shared_shape,
            shared_shape,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            shared_shape,
            bias_shape,
            bias_shape,
            rows_shape,
            rows_shape,
            rows_shape,
            rows_shape,
            jax.ShapeDtypeStruct((batch, heads, chunks, 1, value_dim), F32),
            rows_shape,
        ],
        scratch_shapes=[
            pltpu.VMEM((heads // hb, hb, state_dim, value_dim), F32),
            pltpu.VMEM((heads // hb, hb, 1, state_dim), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
    )(
        *inputs[:4],
        q_bias.astype(F32),
        k_bias.astype(F32),
        *inputs[4:],
        starts,
        phase0,
        dy,
        d_final.astype(F32),
    )
    dc, db, dx, d_angle, dqb, dkb, dp, d_gamma, d_scale, flow, d_end, d_dt = outputs

    def per_token(rows_array):
        array = rows_array[..., 0, :].reshape(batch, heads, padded)
        return jnp.swapaxes(array, 1, 2)[:, :length]

    d_last = jnp.sum(flow[..., 0, :], axis=-1) + jnp.sum(d_end[..., 0, :], axis=-1)
    dp = dp[..., 0, :].at[..., -1].add(d_last)
    d_log_decay = cumulative(dp, reverse=True)[..., None, :]
    return (
        dc[:, :length].astype(c.dtype),
        db[:, :length].astype(b.dtype),
        dx[:, :length],
        jnp.sum(dqb, axis=(0, 1)).astype(q_bias.dtype),
        jnp.sum(dkb, axis=(0, 1)).astype(k_bias.dtype),
        d_angle[:, :length],
        per_token(d_dt),
        per_token(d_log_decay),
        per_token(d_gamma),
        per_token(d_scale),
    )


fused_scan.defvjp(_fused_forward, _fused_backward)


def angle_rates(angles, state_dim):
    """[B,T,A] rates on both coordinates of their pair, zero on fixed ones."""
    count = angles.shape[-1]
    duplicate = np.zeros((count, state_dim), np.float32)
    for i in range(count):
        duplicate[i, 2 * i] = duplicate[i, 2 * i + 1] = 1.0
    return jnp.matmul(angles.astype(F32), jnp.asarray(duplicate), precision=HIGHEST)


def mamba3_fused(
    c,
    b,
    x,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    q_bias,
    k_bias,
    chunk_size=CHUNK,
    interpret=None,
    heads_per_step=None,
):
    """SISO Mamba-3 with one shared query and key per token (one group).

    c, b [B,T,N]; x [B,T,H*P]; adt, dt, trap_logits [B,T,H]; angles [B,T,A]
    shared rates; q_bias, k_bias [H,1,N]. Returns y [B,T,H*P] and the final
    state. Mathematically identical to mamba3_chunked with heads repeating
    the group's B and C.
    """
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    heads = dt.shape[-1]
    value_dim = x.shape[-1] // heads
    if heads_per_step is None:
        heads_per_step = _block_size(heads, value_dim, HEADS_PER_STEP)
    trap = jax.nn.sigmoid(trap_logits.astype(F32))
    gamma = dt.astype(F32) * trap
    previous_weight = dt.astype(F32) * (1 - trap)
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    return fused_scan(
        c,
        b,
        x,
        q_bias,
        k_bias,
        angle_rates(angles, c.shape[-1]),
        dt.astype(F32),
        adt,
        gamma,
        scale,
        chunk_size,
        interpret,
        heads_per_step,
    )
