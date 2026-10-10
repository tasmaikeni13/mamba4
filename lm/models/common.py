"""Shared, explicitly matched language-model components."""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp


def dtype_from_name(name):
    """Resolve the serialized precision without changing parameter precision."""
    return {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[
        name
    ]


def dense_init():
    return nn.initializers.normal(0.02)


def residual_projection_init(config):
    return nn.initializers.normal(0.02 / math.sqrt(2 * config.n_layers))


class RMSNorm(nn.Module):
    features: int | None = None
    epsilon: float = 1e-5
    dtype: object = jnp.bfloat16
    param_dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1] if self.features is None else self.features
        if features != x.shape[-1]:
            raise ValueError("RMSNorm feature count does not match input")
        scale = self.param("scale", nn.initializers.ones, (features,), self.param_dtype)
        x32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        normalized = x32 * jax_rsqrt(variance + self.epsilon)
        return (normalized * scale).astype(self.dtype)


def jax_rsqrt(x):
    # Keeping reduction and reciprocal square root in fp32 matters for the SSM.
    return jnp.reciprocal(jnp.sqrt(x))


class FeedForward(nn.Module):
    config: object

    @nn.compact
    def __call__(self, x, train=False):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        projected = nn.Dense(
            2 * config.d_ff,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=dense_init(),
            name="in_proj",
        )(x)
        gate, value = jnp.split(projected, 2, axis=-1)
        hidden = nn.silu(gate) * value
        hidden = nn.Dropout(config.dropout_rate)(hidden, deterministic=not train)
        return nn.Dense(
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=residual_projection_init(config),
            name="out_proj",
        )(hidden)


class TokenEmbedding(nn.Module):
    config: object

    def setup(self):
        self.embedding = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=dtype_from_name(self.config.dtype),
            param_dtype=dtype_from_name(self.config.param_dtype),
            embedding_init=dense_init(),
        )

    def __call__(self, token_ids):
        return self.embedding(token_ids)

    def attend(self, x):
        # The head shares exactly the embedding parameter, including gradients.
        return self.embedding.attend(x).astype(jnp.float32)


def remat_block(block, config, memory=False, position=0):
    """The block itself, or its rematerialized form under the config's policy.

    "full" recomputes every activation of every block in the backward pass;
    "kernels" keeps the scan and memory-solve outputs; "mixers" recomputes
    every block except memory blocks, which keep all their activations,
    including the stored Cholesky factors that their backward pass reuses;
    "mixers-N" recomputes only the first N non-memory blocks (position counts
    them), trading memory for fewer recomputed forwards.
    """
    if not config.remat:
        return block
    if config.remat_policy == "full":
        policy = None
    elif config.remat_policy == "kernels":
        policy = jax.checkpoint_policies.save_only_these_names(
            "memory_solve", "ssd_scan"
        )
    elif config.remat_policy.startswith("mixers"):
        _, _, count = config.remat_policy.partition("-")
        if memory or (count and position >= int(count)):
            return block
        policy = None
    else:
        raise ValueError(f"Unknown remat_policy: {config.remat_policy}")
    return nn.remat(block, static_argnums=(2,), policy=policy)
