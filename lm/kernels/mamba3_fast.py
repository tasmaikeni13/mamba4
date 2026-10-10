"""Mamba-3 SISO SSD with a hand-derived backward pass.

Same recurrence as `lm.kernels.mamba3.mamba3_chunked` for rank-one (SISO)
heads. The forward evaluates every chunk with matrix products and carries one
[N,P] state per chunk. The backward reuses the forward residuals: cotangents of
the chunk states run through a reverse scan, and every other gradient is a
c-by-c or c-by-N product, so no autodiff of the chunk tensors or of the
associative scan is needed. Rotary frames and trapezoidal weights stay outside
this function and keep their autodiff derivatives.

Inputs: q, k [B,T,H,N]; v [B,T,H,P]; log_decay, gamma, scale [B,T,H], where
gamma is the diagonal (current-input) weight and scale the weight of an
earlier source, already including the next step's previous-input term.
"""

from functools import partial

import jax
import jax.numpy as jnp

from lm.kernels.mamba3 import _rotary_frame


def _chunked(x, chunk, padding):
    x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
    x = x.reshape(x.shape[0], -1, chunk, *x.shape[2:])
    return jnp.moveaxis(x, 2, 3)  # [B, n, H, c, ...]


def _unchunked(x, length):
    x = jnp.moveaxis(x, 3, 2)
    return x.reshape(x.shape[0], -1, *x.shape[3:])[:, :length]


def _masks(prefix, gamma, scale):
    chunk = prefix.shape[-1]
    position = jnp.arange(chunk)
    lower = position[:, None] > position[None, :]
    diagonal = position[:, None] == position[None, :]
    difference = prefix[..., :, None] - prefix[..., None, :]
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, difference, 0.0)), 0.0)
    weights = decay * scale[..., None, :] + jnp.where(
        diagonal, gamma[..., :, None], 0.0
    )
    return lower, diagonal, decay, weights


def _forward(q, k, v, log_decay, gamma, scale, chunk):
    batch, length, heads, state_dim = q.shape
    padding = (-length) % chunk
    qc, kc, vc = (_chunked(x, chunk, padding) for x in (q, k, v))
    ac, gc, sc = (
        _chunked(x.astype(jnp.float32), chunk, padding)
        for x in (log_decay, gamma, scale)
    )
    prefix = jnp.cumsum(ac, axis=-1)
    _, _, _, weights = _masks(prefix, gc, sc)
    score = jnp.einsum("bnhtk,bnhsk->bnhts", qc, kc, preferred_element_type=jnp.float32)
    mixed = score * weights
    within = jnp.einsum(
        "bnhts,bnhsp->bnhtp",
        mixed.astype(vc.dtype),
        vc,
        preferred_element_type=jnp.float32,
    )
    last = prefix[..., -1:]
    source = sc * jnp.exp(last - prefix)
    update = jnp.einsum(
        "bnhsk,bnhsp->bnhkp",
        (kc.astype(jnp.float32) * source[..., None]).astype(kc.dtype),
        vc,
        preferred_element_type=jnp.float32,
    )
    decay = jnp.exp(last[..., 0])

    def step(state, inputs):
        chunk_decay, chunk_update = inputs
        return chunk_decay[..., None, None] * state + chunk_update, state

    initial = jnp.zeros((batch, heads, state_dim, v.shape[-1]), jnp.float32)
    final, starts = jax.lax.scan(
        step, initial, (jnp.moveaxis(decay, 1, 0), jnp.moveaxis(update, 1, 0))
    )
    starts = jnp.moveaxis(starts, 0, 1)
    inherited = jnp.einsum(
        "bnhtk,bnhkp->bnhtp",
        qc,
        starts.astype(qc.dtype),
        preferred_element_type=jnp.float32,
    )
    output = within + jnp.exp(prefix)[..., None] * inherited
    residuals = (qc, kc, vc, prefix, gc, sc, starts, length)
    return _unchunked(output, length).astype(v.dtype), final, residuals


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def ssd(q, k, v, log_decay, gamma, scale, chunk=64):
    output, final, _ = _forward(q, k, v, log_decay, gamma, scale, chunk)
    return output, final


def _ssd_forward(q, k, v, log_decay, gamma, scale, chunk):
    output, final, residuals = _forward(q, k, v, log_decay, gamma, scale, chunk)
    return (output, final), residuals


def _ssd_backward(chunk, residuals, cotangents):
    qc, kc, vc, prefix, gc, sc, starts, length = residuals
    d_output, d_final = cotangents
    padding = (-length) % chunk
    dy = _chunked(d_output.astype(jnp.float32), chunk, padding)
    lower, diagonal, decay_matrix, weights = _masks(prefix, gc, sc)
    q32, k32, v32 = (x.astype(jnp.float32) for x in (qc, kc, vc))
    score = jnp.einsum("bnhtk,bnhsk->bnhts", qc, kc, preferred_element_type=jnp.float32)
    mixed = score * weights
    d_mixed = jnp.einsum("bnhtp,bnhsp->bnhts", dy, v32)
    dv = jnp.einsum("bnhts,bnhtp->bnhsp", mixed, dy)
    d_score = d_mixed * weights
    dq = jnp.einsum("bnhts,bnhsk->bnhtk", d_score, k32)
    dk = jnp.einsum("bnhts,bnhtk->bnhsk", d_score, q32)
    d_weights = d_mixed * score
    carried = jnp.exp(prefix)
    scaled_dy = dy * carried[..., None]
    dq = dq + jnp.einsum("bnhtp,bnhkp->bnhtk", scaled_dy, starts)
    d_starts = jnp.einsum("bnhtk,bnhtp->bnhkp", q32, scaled_dy)
    inherited = jnp.einsum("bnhtk,bnhkp->bnhtp", q32, starts)
    d_prefix = jnp.sum(dy * inherited, axis=-1) * carried
    last = prefix[..., -1]
    decay = jnp.exp(last)

    # ends_i = decay_i starts_i + update_i and starts_(i+1) = ends_i, so the
    # cotangent of ends_i is G_i = d_starts_(i+1) + decay_(i+1) G_(i+1), with
    # G_(n-1) = d_final (shifted decay 1 and no later start at the end).
    def back(carry, inputs):
        next_decay, next_d_starts = inputs
        carry = next_d_starts + next_decay[..., None, None] * carry
        return carry, carry

    shifted_decay = jnp.concatenate((decay[:, 1:], jnp.ones_like(decay[:, :1])), axis=1)
    shifted_d_starts = jnp.concatenate(
        (d_starts[:, 1:], jnp.zeros_like(d_starts[:, :1])), axis=1
    )
    _, ends_cotangent = jax.lax.scan(
        back,
        d_final.astype(jnp.float32),
        (jnp.moveaxis(shifted_decay, 1, 0), jnp.moveaxis(shifted_d_starts, 1, 0)),
        reverse=True,
    )
    ends_cotangent = jnp.moveaxis(ends_cotangent, 0, 1)
    d_decay = jnp.sum(ends_cotangent * starts, axis=(-1, -2))
    source = sc * jnp.exp(last[..., None] - prefix)
    k_ends = jnp.einsum("bnhsk,bnhkp->bnhsp", k32, ends_cotangent)
    dk = dk + jnp.einsum("bnhkp,bnhsp->bnhsk", ends_cotangent, v32) * source[..., None]
    dv = dv + k_ends * source[..., None]
    d_source = jnp.sum(k_ends * v32, axis=-1)
    d_scale = d_source * jnp.exp(last[..., None] - prefix)
    d_prefix = d_prefix - d_source * source
    d_last = jnp.sum(d_source * source, axis=-1) + d_decay * decay
    d_scale = d_scale + jnp.sum(d_weights * decay_matrix, axis=-2)
    off = jnp.where(lower, d_weights * weights, 0.0)
    d_prefix = d_prefix + jnp.sum(off, axis=-1) - jnp.sum(off, axis=-2)
    d_gamma = jnp.sum(jnp.where(diagonal, d_weights, 0.0), axis=-1)
    d_prefix = d_prefix.at[..., -1].add(d_last)
    d_log_decay = jnp.cumsum(d_prefix[..., ::-1], axis=-1)[..., ::-1]
    unchunk = lambda x: _unchunked(x, length)  # noqa: E731
    return (
        unchunk(dq).astype(qc.dtype),
        unchunk(dk).astype(kc.dtype),
        unchunk(dv).astype(vc.dtype),
        unchunk(d_log_decay),
        unchunk(d_gamma),
        unchunk(d_scale),
    )


ssd.defvjp(_ssd_forward, _ssd_backward)


def mamba3_fast(
    q,
    k,
    v,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    chunk_size=64,
    q_bias=None,
    k_bias=None,
    pairwise=True,
):
    """Drop-in for `mamba3_chunked` with rank-one heads ([B,T,H,1,*] shapes)."""
    if q.shape[3] != 1:
        raise ValueError("mamba3_fast supports SISO (rank-one) heads only")
    q, k, _ = _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise)
    trap = jax.nn.sigmoid(trap_logits.astype(jnp.float32))
    gamma = dt.astype(jnp.float32) * trap
    previous_weight = dt.astype(jnp.float32) * (1 - trap)
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    output, final = ssd(
        q[..., 0, :], k[..., 0, :], v[..., 0, :], adt, gamma, scale, chunk_size
    )
    return output[..., None, :], final
