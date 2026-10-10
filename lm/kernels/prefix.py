"""Prefix sums and decayed chunk carries as matrix products.

On TPU, XLA lowers jnp.cumsum to a full-length reduce-window whose reverse
(its transpose) runs in quadratic time, and lax.associative_scan over a few
chunks becomes many pads and slices. Over short axes both are exact and
cheaper as float32-accurate products with triangular or decay matrices,
whose transposes are products of the same kind.
"""

import jax.numpy as jnp
from jax import lax

HIGHEST = lax.Precision.HIGHEST
F32 = jnp.float32


def cumulative(x, reverse=False):
    """Inclusive sum along the last axis; reverse sums from the end."""
    index = jnp.arange(x.shape[-1])
    if reverse:
        mask = index[:, None] >= index[None, :]
    else:
        mask = index[:, None] <= index[None, :]
    return jnp.matmul(x.astype(F32), mask.astype(F32), precision=HIGHEST)


def chunk_carry(log_decay, updates, initial=None):
    """End states of h_i = exp(log_decay_i) h_(i-1) + u_i over chunks.

    log_decay [B,n,H] are per-chunk log decays, updates [B,n,H,D] and
    initial [B,H,D] the state before the first chunk. Returns [B,n,H,D].
    """
    total = cumulative(jnp.swapaxes(log_decay, 1, 2))  # [B,H,n]
    index = jnp.arange(total.shape[-1])
    lower = index[:, None] >= index[None, :]
    difference = total[..., :, None] - total[..., None, :]
    weights = jnp.where(lower, jnp.exp(jnp.where(lower, difference, 0.0)), 0.0)
    ends = jnp.einsum(
        "bhij,bjhd->bihd", weights, updates.astype(F32), precision=HIGHEST
    )
    if initial is not None:
        carried = jnp.exp(jnp.swapaxes(total, 1, 2))[..., None]  # [B,n,H,1]
        ends = ends + carried * initial[:, None].astype(F32)
    return ends


def prefix_sum(x, chunk=128):
    """Inclusive sum over axis 1 of [B,T,...] by chunked triangular products.

    Trailing axes are flattened so the contraction runs over lane-dense rows.
    """
    batch, length = x.shape[:2]
    rest = x.shape[2:]
    flat = x.astype(F32).reshape(batch, length, -1)
    padding = (-length) % chunk
    flat = jnp.pad(flat, [(0, 0), (0, padding), (0, 0)])
    blocks = flat.reshape(batch, -1, chunk, flat.shape[-1])
    ones = jnp.tril(jnp.ones((chunk, chunk), F32))
    within = jnp.einsum("ts,bnsc->bntc", ones, blocks, precision=HIGHEST)
    totals = within[:, :, -1]
    carried = jnp.einsum(
        "ij,bjc->bic",
        jnp.tril(jnp.ones((totals.shape[1],) * 2, F32), -1),
        totals,
        precision=HIGHEST,
    )
    output = within + carried[:, :, None]
    return output.reshape(batch, -1, *rest)[:, :length]
