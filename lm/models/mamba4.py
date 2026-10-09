"""Learned conjugate Gaussian memory with protected QR and order-sensitive scans.

Cyclic gates are learned per head and constant in time. Fixed-floor mode uses
learned token gates and refactors the exact fixed-prior precision. Neither
learned geometry routing nor Gaussian confidence claims universal exact recall.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

from lm.kernels.mamba4 import (
    GaussianResult,
    cyclic_decode,
    gaussian_diagnostics,
    gaussian_initial_state,
    gaussian_memory,
    protected_cascade_memory,
    protected_decode,
    protected_diagnostics,
    protected_initial_state,
)
from lm.models.common import (
    FeedForward,
    RMSNorm,
    TokenEmbedding,
    dense_init,
    dtype_from_name,
    residual_projection_init,
)


def _unit(x):
    x = x.astype(jnp.float32)
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def _constant(value):
    return lambda _key, shape, dtype: jnp.full(shape, value, dtype)


def order_memory(values, decay):
    """A token-gated vector affine scan; additional state and projection count."""
    x = values.astype(jnp.float32)
    gate = decay.astype(jnp.float32)[..., None]

    def combine(earlier, later):
        l0, v0 = earlier
        l1, v1 = later
        return l1 * l0, v1 + l1 * v0

    return lax.associative_scan(combine, (gate, (1 - gate) * x), axis=1)[1]


class Mamba4Mixer(nn.Module):
    config: object

    @nn.compact
    def __call__(self, x, train=False, cache=None):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        heads, dim, value_dim = config.num_heads, config.key_dim, config.value_dim
        batch, time, _ = x.shape
        if config.d_model * config.expand != heads * value_dim:
            raise ValueError("Expanded dimension must divide exactly into memory heads")

        def project(name, features):
            return nn.Dense(
                features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=dense_init(),
                name=name,
            )(x)

        key = _unit(project("key", heads * dim).reshape(batch, time, heads, dim))
        query = _unit(project("query", heads * dim).reshape(batch, time, heads, dim))
        value = project("value", heads * value_dim).reshape(
            batch, time, heads, value_dim
        )
        output_gate = project("output_gate", heads * value_dim).reshape(value.shape)
        precision_logits = project("precision", heads).astype(jnp.float32)
        beta = 2 * jax.nn.sigmoid(precision_logits) + 1e-4
        if config.memory_epsilon <= 1e-4:
            raise ValueError("memory_epsilon must exceed the learnable 1e-4 floor")
        raw_epsilon = self.param(
            "epsilon",
            _constant(math.log(math.expm1(config.memory_epsilon - 1e-4))),
            (heads,),
            jnp.float32,
        )
        epsilon = jax.nn.softplus(raw_epsilon) + 1e-4
        # Bound the cyclic anisotropy at all learned values, including gradients.
        lower, upper = 0.9, 0.9999
        if not lower < config.memory_decay < upper:
            raise ValueError(
                "Learned decay initialization must lie strictly in (0.9,0.9999)"
            )
        init_fraction = (config.memory_decay - lower) / (upper - lower)
        decay_bias = math.log(init_fraction / (1 - init_fraction))
        raw_decay = self.param("decay", _constant(decay_bias), (heads,), jnp.float32)
        if config.memory_floor == "cyclic":
            decay = lower + (upper - lower) * jax.nn.sigmoid(raw_decay)
        elif config.memory_floor == "fixed":
            decay = lower + (upper - lower) * jax.nn.sigmoid(
                project("token_decay", heads).astype(jnp.float32) + raw_decay
            )
        else:
            raise ValueError("Unsupported memory_floor")
        if cache is None:
            result = gaussian_memory(
                key,
                value,
                query,
                beta,
                decay,
                epsilon,
                floor=config.memory_floor,
                chunk_size=config.chunk_size,
            )
        else:
            if time != 1:
                raise ValueError("Cached Mamba4 mixer accepts exactly one token")
            initial, initial_lower = gaussian_initial_state(
                batch,
                heads,
                dim,
                value_dim,
                epsilon,
                decay if config.memory_floor == "cyclic" else config.memory_decay,
                floor=config.memory_floor,
            )
            state = jax.tree.map(
                lambda fresh, old: jnp.where(cache["gaussian"].step == 0, fresh, old),
                initial,
                cache["gaussian"],
            )
            lower_factor = jnp.where(state.step == 0, initial_lower, cache["lower"])
            if config.memory_floor == "cyclic":
                decoded, lower_factor = cyclic_decode(
                    state,
                    lower_factor,
                    key[:, 0],
                    value[:, 0],
                    query[:, 0],
                    beta[:, 0],
                    decay,
                    epsilon,
                )
                result = GaussianResult(
                    decoded.output[:, None], decoded.variance[:, None], decoded.state
                )
            else:
                result = gaussian_memory(
                    key,
                    value,
                    query,
                    beta,
                    decay,
                    epsilon,
                    floor="fixed",
                    chunk_size=1,
                    initial=state,
                )
                lower_factor = jnp.linalg.cholesky(result.state.precision)
        read, variance = result.output, result.variance
        # Multi-hop uses learned bridges, retains the same statistics, and pays
        # sequential reads. This implementation refactors per hop (explicit cost).
        for hop in range(1, config.memory_hops):
            bridge = self.param(
                f"hop_bridge_{hop}", dense_init(), (heads, value_dim, dim), param_dtype
            )
            query = _unit(
                jnp.einsum(
                    "bthp,hpd->bthd",
                    read,
                    bridge.astype(jnp.float32),
                    precision=lax.Precision.HIGHEST,
                )
            )
            if cache is None:
                result = gaussian_memory(
                    key,
                    value,
                    query,
                    beta,
                    decay,
                    epsilon,
                    floor=config.memory_floor,
                    chunk_size=config.chunk_size,
                )
                read, variance = result.output, result.variance
            else:
                y = solve_triangular(lower_factor, query[:, 0, ..., None], lower=True)
                y = solve_triangular(
                    jnp.swapaxes(lower_factor, -1, -2), y, lower=False
                )[..., 0]
                read = jnp.einsum(
                    "bhpd,bhd->bhp",
                    result.state.cross,
                    y,
                    precision=lax.Precision.HIGHEST,
                )[:, None]
                variance = jnp.sum(query[:, 0] * y, axis=-1)[:, None]
        if config.memory_hops < 1:
            raise ValueError("memory_hops must be positive")
        confidence_scale = self.param(
            "confidence_scale", _constant(0.1), (heads,), jnp.float32
        )
        confidence_gate = jax.nn.sigmoid(1 - confidence_scale * jnp.log1p(variance))
        read *= confidence_gate[..., None]
        if config.protected_anchor_budget:
            if config.protected_routing != "all":
                raise ValueError(
                    "Only explicitly counted all-bank protected routing is implemented"
                )
            route_raw = self.param(
                "protected_route_strength", _constant(3.98), (heads,), jnp.float32
            )
            if cache is None:
                protected_result = protected_cascade_memory(
                    key,
                    value,
                    query,
                    budget=config.protected_anchor_budget,
                    block_size=config.protected_block_size,
                    route_strength=jax.nn.softplus(route_raw),
                )
                protected = protected_result.output
                protected_state = protected_result.state
            else:
                protected, protected_cache = protected_decode(
                    cache["protected"],
                    key[:, 0],
                    value[:, 0],
                    query[:, 0],
                    budget=config.protected_anchor_budget,
                    block_size=config.protected_block_size,
                    route_strength=jax.nn.softplus(route_raw),
                )
                protected = protected[:, None]
                protected_state = protected_cache
            mix = self.param("protected_mix", _constant(-3.0), (heads,), jnp.float32)
            read += jax.nn.sigmoid(mix)[None, None, :, None] * protected
        order_logits = project("order_decay", heads).astype(jnp.float32)
        order_decay = jax.nn.sigmoid(order_logits + 2)
        if cache is None:
            ordered = order_memory(value, order_decay)
        else:
            order_cache = order_decay[:, 0, ..., None] * cache["order"] + (
                1 - order_decay[:, 0, ..., None]
            ) * value[:, 0].astype(jnp.float32)
            ordered = order_cache[:, None]
        order_mix = self.param("order_mix", _constant(-1.0), (heads,), jnp.float32)
        read += jax.nn.sigmoid(order_mix)[None, None, :, None] * ordered
        if self.is_mutable_collection("diagnostics") and not self.is_initializing():
            raw = gaussian_diagnostics(result.state)
            metrics = {
                "precision_eigenvalue_min": raw["precision_min_eigenvalue"],
                "precision_eigenvalue_max": raw["precision_max_eigenvalue"],
                "precision_condition_max": raw["precision_max_condition"],
                "gaussian_cross_abs_max": raw["cross_max_abs"],
                "prior_diagonal_min": raw["prior_min_diagonal"],
                "prior_diagonal_max": raw["prior_max_diagonal"],
                "gaussian_allfinite": raw["state_all_finite"]
                & jnp.all(jnp.isfinite(result.output))
                & jnp.all(jnp.isfinite(result.variance)),
                "epsilon_min": jnp.min(epsilon),
                "epsilon_max": jnp.max(epsilon),
                "decay_min": jnp.min(decay),
                "decay_max": jnp.max(decay),
                "beta_min": jnp.min(beta),
                "beta_max": jnp.max(beta),
            }
            if config.protected_anchor_budget:
                raw = protected_diagnostics(protected_state)
                metrics.update(
                    {
                        "protected_bank_count_max": raw["protected_active_banks"],
                        "protected_retained_anchors_min": raw[
                            "protected_retained_anchors_min"
                        ],
                        "protected_retained_anchors_max": raw[
                            "protected_retained_anchors_max"
                        ],
                        "protected_qr_diagonal_min": raw["protected_min_qr_diagonal"],
                        "protected_merge_count_max": raw["protected_merges"],
                        "protected_allfinite": raw["protected_all_finite"],
                    }
                )
            for name, statistic in metrics.items():
                self.sow("diagnostics", name, statistic)
        read = RMSNorm(
            value_dim,
            epsilon=config.norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            name="read_norm",
        )(read)
        read *= nn.silu(output_gate)
        read = read.reshape(batch, time, heads * value_dim)
        read = nn.Dropout(config.dropout_rate)(read, deterministic=not train)
        output = nn.Dense(
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=residual_projection_init(config),
            name="out_proj",
        )(read)
        if cache is None:
            return output
        return output, dict(
            gaussian=result.state,
            lower=lower_factor,
            order=order_cache,
            protected=protected_cache if config.protected_anchor_budget else None,
        )


class Mamba4Block(nn.Module):
    config: object

    @nn.compact
    def __call__(self, x, train=False, cache=None):
        config = self.config
        kwargs = dict(
            epsilon=config.norm_eps,
            dtype=dtype_from_name(config.dtype),
            param_dtype=dtype_from_name(config.param_dtype),
        )
        memory = Mamba4Mixer(config, name="memory")(
            RMSNorm(name="memory_norm", **kwargs)(x), train=train, cache=cache
        )
        if cache is not None:
            memory, cache = memory
        x = x + memory
        output = x + FeedForward(config, name="mlp")(
            RMSNorm(name="mlp_norm", **kwargs)(x), train=train
        )
        return output if cache is None else (output, cache)


class Mamba4LM(nn.Module):
    config: object

    @nn.compact
    def __call__(self, tokens, train=False):
        config = self.config
        embedding = TokenEmbedding(config, name="token_embedding")
        x = embedding(tokens)
        block = (
            nn.remat(Mamba4Block, static_argnums=(2,)) if config.remat else Mamba4Block
        )
        for i in range(config.n_layers):
            x = block(config, name=f"layer_{i}")(x, train)
        x = RMSNorm(
            epsilon=config.norm_eps,
            dtype=dtype_from_name(config.dtype),
            param_dtype=dtype_from_name(config.param_dtype),
            name="final_norm",
        )(x)
        return embedding.attend(x)

    @nn.compact
    def decode_step(self, tokens, cache):
        config = self.config
        embedding = TokenEmbedding(config, name="token_embedding")
        x = embedding(tokens)[:, None]
        updated = []
        for i in range(config.n_layers):
            x, layer_cache = Mamba4Block(config, name=f"layer_{i}")(x, False, cache[i])
            updated.append(layer_cache)
        x = RMSNorm(
            epsilon=config.norm_eps,
            dtype=dtype_from_name(config.dtype),
            param_dtype=dtype_from_name(config.param_dtype),
            name="final_norm",
        )(x)
        return embedding.attend(x)[:, 0], tuple(updated)


def initialize_cache(config, batch_size):
    """Static cache allocation; first step reinitializes priors from learned params."""
    layers = []
    for _ in range(config.n_layers):
        state, lower = gaussian_initial_state(
            batch_size,
            config.num_heads,
            config.key_dim,
            config.value_dim,
            config.memory_epsilon,
            config.memory_decay,
            floor=config.memory_floor,
        )
        protected = (
            protected_initial_state(
                batch_size,
                config.num_heads,
                config.key_dim,
                config.value_dim,
                config.protected_anchor_budget,
                config.max_seq_len,
                config.protected_block_size,
            )
            if config.protected_anchor_budget
            else None
        )
        layers.append(
            dict(
                gaussian=state,
                lower=lower,
                order=jnp.zeros((batch_size, config.num_heads, config.value_dim)),
                protected=protected,
            )
        )
    return tuple(layers)


def decode_step(config, params, tokens, cache, position=None):
    """Full cached LM decode; position is tracked in Gaussian/protected caches."""
    model = Mamba4LM(config)
    return model.apply({"params": params}, tokens, cache, method=model.decode_step)
