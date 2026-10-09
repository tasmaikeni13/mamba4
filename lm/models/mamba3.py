"""Full official Mamba-3 SISO block with an optional genuine MIMO path.

Includes learned input/output projections, normalized and biased B/C,
data-dependent negative A, learned timestep offset, exponential trapezoidal
integration, dt-integrated complex phases, D skip, and SiLU output gating.
No convolution or additional FFN is present in the official Mamba-3 block.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp

from lm.kernels.mamba3 import mamba3_chunked
from lm.models.common import (
    RMSNorm,
    TokenEmbedding,
    dense_init,
    dtype_from_name,
    residual_projection_init,
)


def heavy_tail_activation(x):
    return jnp.maximum(x, 0) + jnp.reciprocal(1 - jnp.minimum(x, 0))


def timestep_bias_init(key, shape, dtype=jnp.float32):
    dt = jnp.exp(
        jax.random.uniform(key, shape, minval=math.log(0.001), maxval=math.log(0.1))
    )
    return (dt + jnp.log(-jnp.expm1(-dt))).astype(dtype)


class Mamba3Mixer(nn.Module):
    config: object

    @nn.compact
    def __call__(self, u, train=False):
        del train
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        inner = config.d_model * config.expand
        heads = inner // config.head_dim
        if inner % config.head_dim or heads % config.n_groups:
            raise ValueError("Mamba inner/head/group dimensions must divide evenly")
        if config.rope_fraction not in (0.5, 1.0) or config.d_state % 2:
            raise ValueError(
                "Official Mamba-3 requires even state and rope_fraction 0.5 or 1"
            )
        rank = config.mimo_rank
        angles_count = int(config.d_state * config.rope_fraction) // 2
        bc_size = config.d_state * config.n_groups * rank
        sizes = (inner, inner, bc_size, bc_size, heads, heads, heads, angles_count)
        projected = nn.Dense(
            sum(sizes),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=dense_init(),
            name="in_proj",
        )(u)
        sections = []
        cursor = 0
        for size in sizes:
            sections.append(projected[..., cursor : cursor + size])
            cursor += size
        z, x, b, c, raw_dt, raw_a, trap, angles = sections
        b = b.reshape(*u.shape[:2], rank, config.n_groups, config.d_state)
        c = c.reshape(b.shape)
        b = RMSNorm(config.d_state, config.norm_eps, dtype, param_dtype, name="B_norm")(
            b
        )
        c = RMSNorm(config.d_state, config.norm_eps, dtype, param_dtype, name="C_norm")(
            c
        )
        b = jnp.transpose(b, (0, 1, 3, 2, 4))
        c = jnp.transpose(c, (0, 1, 3, 2, 4))
        b = jnp.repeat(b, heads // config.n_groups, axis=2)
        c = jnp.repeat(c, heads // config.n_groups, axis=2)
        b_bias = self.param(
            "B_bias", nn.initializers.ones, (heads, rank, config.d_state), param_dtype
        )
        c_bias = self.param(
            "C_bias", nn.initializers.ones, (heads, rank, config.d_state), param_dtype
        )
        dt_bias = self.param("dt_bias", timestep_bias_init, (heads,), jnp.float32)
        dt = jax.nn.softplus(raw_dt.astype(jnp.float32) + dt_bias)
        a = -jnp.maximum(heavy_tail_activation(raw_a.astype(jnp.float32)), 1e-4)
        angles = jnp.broadcast_to(
            angles.astype(jnp.float32)[..., None, :],
            (*u.shape[:2], heads, angles_count),
        )
        x = x.reshape(*u.shape[:2], heads, config.head_dim)
        z = z.reshape(x.shape)
        if rank > 1:
            mimo_x = self.param(
                "mimo_x",
                nn.initializers.constant(1 / rank),
                (heads, rank, config.head_dim),
                param_dtype,
            )
            mimo_z = self.param(
                "mimo_z",
                nn.initializers.ones,
                (heads, rank, config.head_dim),
                param_dtype,
            )
            mimo_o = self.param(
                "mimo_o",
                nn.initializers.constant(1 / rank),
                (heads, rank, config.head_dim),
                param_dtype,
            )
            values = (x[..., None, :] * mimo_x.astype(dtype)).astype(dtype)
            gates = (z[..., None, :] * mimo_z.astype(dtype)).astype(dtype)
        else:
            values, gates = x[..., None, :], z[..., None, :]
        y, _ = mamba3_chunked(
            c,
            b,
            values,
            a * dt,
            dt,
            trap,
            angles,
            q_bias=c_bias,
            k_bias=b_bias,
            chunk_size=config.chunk_size,
            pairwise=rank == 1,
        )
        skip = self.param("D", nn.initializers.ones, (heads,), param_dtype)
        y = (
            y.astype(jnp.float32)
            + values.astype(jnp.float32) * skip[None, None, :, None, None]
        )
        if config.mamba3_outproj_norm:
            weight = self.param(
                "out_norm_scale",
                nn.initializers.ones,
                (heads, config.head_dim),
                param_dtype,
            )
            y *= jax.lax.rsqrt(
                jnp.mean(jnp.square(y), axis=-1, keepdims=True) + config.norm_eps
            )
            y *= weight[None, None, :, None, :]
        y *= jax.nn.silu(gates.astype(jnp.float32))
        if rank > 1:
            y = jnp.sum(y * mimo_o[None, None], axis=-2)
        else:
            y = y[..., 0, :]
        y = y.reshape(*u.shape[:2], inner).astype(dtype)
        return nn.Dense(
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=residual_projection_init(config),
            name="out_proj",
        )(y)


class Mamba3Block(nn.Module):
    config: object

    @nn.compact
    def __call__(self, x, train=False):
        config = self.config
        normalized = RMSNorm(
            config.d_model,
            config.norm_eps,
            dtype_from_name(config.dtype),
            dtype_from_name(config.param_dtype),
            name="norm",
        )(x)
        return x + Mamba3Mixer(config, name="mixer")(normalized, train=train)


class Mamba3LM(nn.Module):
    config: object

    @nn.compact
    def __call__(self, token_ids, train=False):
        config = self.config
        embedding = TokenEmbedding(config, name="tokens")
        x = embedding(token_ids)
        block = (
            nn.remat(Mamba3Block, static_argnums=(2,)) if config.remat else Mamba3Block
        )
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
