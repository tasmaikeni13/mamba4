"""Causal attention with Pallas FlashAttention on TPU and an XLA reference."""

import jax
import jax.numpy as jnp


def causal_attention(q, k, v):
    """Arrays have shape [batch,time,heads,channels]; output has the same shape.

    TPU execution uses the JAX Pallas fused forward/backward kernels. Padding is
    masked by causality, so short/nonmultiple lengths have identical semantics.
    The CPU diagnostic path uses JAX's fp32-softmax implementation.
    """
    if jax.default_backend() == "tpu":
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

        length = q.shape[1]
        pad = (-length) % 128
        arrays = []
        for array in (q, k, v):
            array = jnp.pad(array, ((0, 0), (0, pad), (0, 0), (0, 0)))
            arrays.append(jnp.transpose(array, (0, 2, 1, 3)))
        output = flash_attention(*arrays, causal=True, sm_scale=q.shape[-1] ** -0.5)
        return jnp.transpose(output, (0, 2, 1, 3))[:, :length]
    return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")


def attention_reference(q, k, v):
    """Independent explicit softmax reference used for conformance checks."""
    scores = jnp.einsum("bthd,bshd->bhts", q.astype(jnp.float32), k.astype(jnp.float32))
    scores *= q.shape[-1] ** -0.5
    mask = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
    scores = jnp.where(mask[None, None], scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", probabilities, v.astype(jnp.float32)).astype(
        v.dtype
    )
