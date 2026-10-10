"""Causal attention with Pallas FlashAttention on TPU and an XLA reference."""

from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(maxsize=None)
def _splash_kernel(length, heads, block):
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as masks,
    )

    mask = masks.MultiHeadMask([masks.CausalMask((length, length))] * heads)
    sizes = splash.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        use_fused_bwd_kernel=True,
    )
    return splash.make_splash_mha(
        mask, block_sizes=sizes, head_shards=1, q_seq_shards=1
    )


def _flash_blocks(block):
    from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes

    if not block:
        return None
    return BlockSizes(
        block_q=block,
        block_k_major=block,
        block_k=block,
        block_b=1,
        block_q_major_dkv=block,
        block_k_major_dkv=block,
        block_k_dkv=block,
        block_q_dkv=block,
        block_k_major_dq=block,
        block_k_dq=block,
        block_q_dq=block,
    )


def causal_attention(q, k, v, kernel="flash", block=0):
    """Arrays have shape [batch,time,heads,channels]; output has the same shape.

    TPU execution uses JAX's Pallas fused forward/backward kernels:
    FlashAttention (block 0 keeps its default tiles) or SplashAttention with
    a fused backward. Both compute exact softmax attention. Padding is masked
    by causality, so short/nonmultiple lengths have identical semantics. The
    CPU diagnostic path uses JAX's fp32-softmax implementation.
    """
    if jax.default_backend() == "tpu":
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

        length = q.shape[1]
        if kernel == "splash":
            block = block or 512
        tile = max(block, 128)
        pad = (-length) % tile
        arrays = []
        for array in (q, k, v):
            array = jnp.pad(array, ((0, 0), (0, pad), (0, 0), (0, 0)))
            arrays.append(jnp.transpose(array, (0, 2, 1, 3)))
        scale = q.shape[-1] ** -0.5
        if kernel == "splash":
            splash = _splash_kernel(length + pad, q.shape[2], block)
            output = jax.vmap(splash)(arrays[0] * scale, arrays[1], arrays[2])
        elif kernel == "flash":
            output = flash_attention(
                *arrays, causal=True, sm_scale=scale, block_sizes=_flash_blocks(block)
            )
        else:
            raise ValueError(f"Unknown attention kernel: {kernel}")
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
