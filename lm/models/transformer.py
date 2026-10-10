"""Pre-norm causal Transformer with RoPE, SwiGLU and tied embeddings."""

from flax import linen as nn
import jax.numpy as jnp

from lm.kernels.attention import causal_attention
from lm.models.common import (
    FeedForward,
    RMSNorm,
    TokenEmbedding,
    dense_init,
    dtype_from_name,
    remat_block,
    residual_projection_init,
)


def rotary_positions(x):
    channels = x.shape[-1]
    if channels % 2:
        raise ValueError("Transformer RoPE head dimension must be even")
    frequency = 10000.0 ** (-jnp.arange(0, channels, 2, dtype=jnp.float32) / channels)
    angle = jnp.arange(x.shape[1], dtype=jnp.float32)[:, None] * frequency[None]
    pair = x.astype(jnp.float32).reshape(*x.shape[:-1], channels // 2, 2)
    cosine, sine = jnp.cos(angle)[None, :, None], jnp.sin(angle)[None, :, None]
    rotated = jnp.stack(
        (
            pair[..., 0] * cosine - pair[..., 1] * sine,
            pair[..., 0] * sine + pair[..., 1] * cosine,
        ),
        axis=-1,
    )
    return rotated.reshape(x.shape).astype(x.dtype)


class TransformerBlock(nn.Module):
    config: object

    @nn.compact
    def __call__(self, x, train=False):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        normalized = RMSNorm(
            config.d_model, config.norm_eps, dtype, param_dtype, name="attention_norm"
        )(x)
        qkv = nn.Dense(
            3 * config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=dense_init(),
            name="qkv_proj",
        )(normalized)
        qkv = qkv.reshape(
            *x.shape[:2], 3, config.n_heads, config.d_model // config.n_heads
        )
        q, k, v = (qkv[:, :, index] for index in range(3))
        output = causal_attention(
            rotary_positions(q),
            rotary_positions(k),
            v,
            config.attention_kernel,
            config.attention_block,
        )
        output = output.reshape(x.shape)
        output = nn.Dense(
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=residual_projection_init(config),
            name="out_proj",
        )(output)
        x = x + nn.Dropout(config.dropout_rate)(output, deterministic=not train)
        normalized = RMSNorm(
            config.d_model, config.norm_eps, dtype, param_dtype, name="ffn_norm"
        )(x)
        return x + FeedForward(config, name="ffn")(normalized, train=train)


class TransformerLM(nn.Module):
    config: object

    @nn.compact
    def __call__(self, token_ids, train=False):
        config = self.config
        embedding = TokenEmbedding(config, name="tokens")
        x = embedding(token_ids)
        block = remat_block(TransformerBlock, config)
        for index in range(config.n_layers):
            x = block(config, name=f"layer_{index}")(x, train)
        x = RMSNorm(
            config.d_model,
            config.norm_eps,
            dtype_from_name(config.dtype),
            dtype_from_name(config.param_dtype),
            name="final_norm",
        )(x)
        return embedding.attend(x)
