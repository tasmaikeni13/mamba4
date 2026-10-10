"""Mamba 4 language model: selective conjugate memory with Mamba-3 layers.

Memory layers keep regression sufficient statistics with token drift and
evidence gates and an undiscounted floor, and read them exactly. Keys can be
aligned to the preceding context so a read is an induction lookup. Official
Mamba-3 SISO layers fill the remaining depth. Learned encoders and gates do
not inherit the operator's exact-recall or calibration guarantees.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from lm.kernels.mamba4 import (
    selective_decode,
    selective_diagnostics,
    selective_gaussian_memory,
    selective_initial_state,
)
from lm.models.common import (
    FeedForward,
    RMSNorm,
    TokenEmbedding,
    dense_init,
    dtype_from_name,
    remat_block,
    residual_projection_init,
)
from lm.kernels.mamba3 import rotate
from lm.models.mamba3 import Mamba3Block, heavy_tail_activation, timestep_bias_init


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


def order_memory_chunked(values, log_decay, chunk_size=64):
    """The same gated vector scan as order_memory, computed in chunks.

    h_t = lam_t h_(t-1) + (1 - lam_t) v_t from h_0 = 0. Within a chunk the
    recurrence is one masked decay-weighted matrix product; chunk ends combine
    by the associative affine scan. Taking log(lam) keeps 1 - lam exact.
    """
    x = values.astype(jnp.float32)
    batch, time, heads, width = x.shape
    padding = (-time) % chunk_size
    count = (time + padding) // chunk_size

    def chunks(a):
        a = jnp.pad(a, [(0, 0), (0, padding)] + [(0, 0)] * (a.ndim - 2))
        a = a.reshape(batch, count, chunk_size, *a.shape[2:])
        return jnp.swapaxes(a, 2, 3)

    value, logs = chunks(x), chunks(log_decay.astype(jnp.float32))
    inflow = -jnp.expm1(logs)
    prefix = jnp.cumsum(logs, axis=-1)
    position = jnp.arange(chunk_size)
    causal = position[:, None] >= position[None, :]
    difference = prefix[..., :, None] - prefix[..., None, :]
    weights = jnp.where(causal, jnp.exp(jnp.where(causal, difference, 0.0)), 0.0)
    local = jnp.einsum(
        "bnhts,bnhsp->bnhtp",
        weights * inflow[..., None, :],
        value,
        precision=lax.Precision.HIGHEST,
    )

    def combine(earlier, later):
        d0, x0 = earlier
        d1, x1 = later
        return d1 * d0, x1 + d1[..., None] * x0

    _, ends = lax.associative_scan(
        combine, (jnp.exp(prefix[..., -1]), local[..., -1, :]), axis=1
    )
    starts = jnp.concatenate((jnp.zeros_like(ends[:, :1]), ends[:, :-1]), axis=1)
    output = local + jnp.exp(prefix)[..., None] * starts[..., None, :]
    output = jnp.swapaxes(output, 2, 3).reshape(batch, count * chunk_size, heads, width)
    return output[:, :time]


def causal_conv(x, weight, bias, history=None):
    """Depthwise causal convolution; history holds the previous K-1 inputs."""
    taps = weight.shape[0]
    if history is None:
        history = jnp.zeros((x.shape[0], taps - 1, x.shape[-1]), x.dtype)
    padded = jnp.concatenate((history.astype(x.dtype), x), axis=1)
    output = bias.astype(jnp.float32)
    for tap in range(taps):
        window = padded[:, tap : tap + x.shape[1]].astype(jnp.float32)
        output = output + window * weight[tap].astype(jnp.float32)
    return output, padded[:, x.shape[1] :]


def _inverse_softplus(value):
    return math.log(math.expm1(value))


class SelectiveMemoryMixer(nn.Module):
    """Definition 5.1 with token drift and evidence gates, and a fixed floor.

    S_t = lam_t S_(t-1) + beta_t k k^T, C_t likewise, A_t = S_t + diag(floor).
    Every read solves A_t y = q_t exactly and returns C_t y and q_t^T y. The
    undiscounted floor bounds every precision eigenvalue below for arbitrary
    gates; no cyclic or constant-gate assumption is needed.
    """

    config: object

    @nn.compact
    def __call__(self, x, train=False, cache=None):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        heads, dim, value_dim = (
            config.memory_heads,
            config.key_dim,
            config.memory_value_dim,
        )
        inner = heads * value_dim
        batch, time, _ = x.shape
        if config.d_model * config.expand != inner:
            raise ValueError("Expanded dimension must divide exactly into memory heads")
        if config.floor_min <= 0 or config.floor_init <= config.floor_min:
            raise ValueError("Selective floor must be positive and above its minimum")
        angles_count = int(dim * config.memory_rope_fraction) // 2
        sizes = (inner, inner, heads * dim, heads * dim, heads, heads, heads)
        sizes += (angles_count,) if angles_count else ()
        projected = nn.Dense(
            sum(sizes),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=dense_init(),
            name="in_proj",
        )(x)
        sections, cursor = [], 0
        for size in sizes:
            sections.append(projected[..., cursor : cursor + size])
            cursor += size
        gate, value, key, query, beta_raw, dt_raw, order_raw = sections[:7]
        mixed = jnp.concatenate((value, key, query), axis=-1)
        history = None if cache is None else cache["conv"]
        if config.conv_kernel:
            bound = 1 / math.sqrt(config.conv_kernel)
            weight = self.param(
                "conv_weight",
                nn.initializers.uniform(2 * bound),
                (config.conv_kernel, mixed.shape[-1]),
                param_dtype,
            )
            weight = weight - bound
            bias = self.param(
                "conv_bias", nn.initializers.zeros, (mixed.shape[-1],), param_dtype
            )
            mixed, history = causal_conv(mixed, weight, bias, history)
        mixed = nn.silu(mixed.astype(jnp.float32))
        value = mixed[..., :inner].reshape(batch, time, heads, value_dim)
        key = mixed[..., inner : inner + heads * dim].reshape(batch, time, heads, dim)
        query = mixed[..., inner + heads * dim :].reshape(batch, time, heads, dim)
        if config.qk_activation == "none":
            key = sections[2].astype(jnp.float32).reshape(key.shape)
            query = sections[3].astype(jnp.float32).reshape(query.shape)
        elif config.qk_activation != "silu":
            raise ValueError("qk_activation must be 'silu' or 'none'")
        shifted = None
        if config.memory_key_shift:
            # Write each value under the key of the context before it, while
            # queries see the current context: a read then returns what
            # followed the current context last time (an induction lookup).
            if angles_count:
                raise ValueError("memory_key_shift does not support memory rotary")
            if cache is None:
                previous = jnp.zeros_like(key[:, :1])
            else:
                previous = cache["shifted"][:, None]
            shifted = key[:, -1]
            key = jnp.concatenate((previous, key[:, :-1]), axis=1)
        key, query = _unit(key), _unit(query)
        if config.memory_beta_max <= 0:
            raise ValueError("memory_beta_max must be positive")
        # Evidence precision lies in (0, beta_max). Every head starts at
        # beta = 1/2 whatever the range, so a wider range adds the ability to
        # write one token far above the floor without changing initialization.
        start = 0.5 / max(config.memory_beta_max, 1.0)
        beta_bias = self.param(
            "beta_bias",
            _constant(math.log(start / (1 - start))),
            (heads,),
            jnp.float32,
        )
        beta = config.memory_beta_max * jax.nn.sigmoid(
            beta_raw.astype(jnp.float32) + beta_bias
        )
        dt_bias = self.param("dt_bias", timestep_bias_init, (heads,), jnp.float32)
        dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + dt_bias)
        a_log = self.param(
            "a_log",
            lambda key_, shape, dtype_: jnp.log(
                jax.random.uniform(key_, shape, dtype_, 1.0, 16.0)
            ),
            (heads,),
            jnp.float32,
        )
        log_decay = -jnp.exp(a_log) * dt
        phase = None
        if angles_count:
            # Data-dependent rotary phases, integrated with the same timestep
            # as Mamba-3's complex state. Rotations keep keys/queries unit.
            increment = sections[7].astype(jnp.float32)[..., None, :] * dt[..., None]
            start = 0.0 if cache is None else cache["phase"][:, None]
            phase = start + jnp.cumsum(increment, axis=1)
            key = rotate(key, phase, True)
            query = rotate(query, phase, True)
        floor_raw = self.param(
            "floor",
            _constant(_inverse_softplus(config.floor_init - config.floor_min)),
            (heads, dim),
            jnp.float32,
        )
        floor = config.floor_min + jax.nn.softplus(floor_raw)
        if cache is None:
            result = selective_gaussian_memory(
                key,
                value,
                query,
                beta,
                log_decay,
                floor,
                chunk_size=config.chunk_size,
                solver=config.memory_solver,
            )
            memory_state = result.state
            read, variance = result.output, result.variance
        else:
            if time != 1:
                raise ValueError("Cached selective mixer accepts exactly one token")
            result = selective_decode(
                cache["memory"],
                key[:, 0],
                value[:, 0],
                query[:, 0],
                beta[:, 0],
                log_decay[:, 0],
                floor,
                config.memory_solver,
            )
            memory_state = result.state
            read, variance = result.output[:, None], result.variance[:, None]
        confidence_scale = self.param(
            "confidence_scale", _constant(0.1), (heads,), jnp.float32
        )
        read = (
            read * jax.nn.sigmoid(1 - confidence_scale * jnp.log1p(variance))[..., None]
        )
        skip = self.param("D", nn.initializers.ones, (heads,), jnp.float32)
        read = read + skip[:, None] * value
        order_cache = None
        if config.order_head:
            order_logits = order_raw.astype(jnp.float32) + 2
            order_decay = jax.nn.sigmoid(order_logits)
            if cache is None:
                ordered = order_memory_chunked(
                    value, jax.nn.log_sigmoid(order_logits), config.chunk_size
                )
            else:
                order_cache = (
                    order_decay[:, 0, :, None] * cache["order"]
                    + (1 - order_decay[:, 0, :, None]) * value[:, 0]
                )
                ordered = order_cache[:, None]
            order_mix = self.param("order_mix", _constant(-1.0), (heads,), jnp.float32)
            read = read + jax.nn.sigmoid(order_mix)[:, None] * ordered
        if self.is_mutable_collection("diagnostics") and not self.is_initializing():
            raw = selective_diagnostics(memory_state, floor)
            metrics = {
                "precision_eigenvalue_min": raw["precision_min_eigenvalue"],
                "precision_eigenvalue_max": raw["precision_max_eigenvalue"],
                "precision_condition_max": raw["precision_max_condition"],
                "gaussian_cross_abs_max": raw["cross_max_abs"],
                "floor_min": raw["floor_min"],
                "floor_max": raw["floor_max"],
                "gaussian_allfinite": raw["state_all_finite"]
                & jnp.all(jnp.isfinite(read))
                & jnp.all(jnp.isfinite(variance)),
                "variance_min": jnp.min(variance),
                "variance_max": jnp.max(variance),
                "decay_min": jnp.min(jnp.exp(log_decay)),
                "decay_max": jnp.max(jnp.exp(log_decay)),
                "beta_min": jnp.min(beta),
                "beta_max": jnp.max(beta),
            }
            for name, statistic in metrics.items():
                self.sow("diagnostics", name, statistic)
        read = RMSNorm(
            value_dim,
            epsilon=config.norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            name="read_norm",
        )(read)
        read = read * nn.silu(gate.reshape(read.shape))
        read = read.reshape(batch, time, inner)
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
            memory=memory_state,
            conv=history,
            order=order_cache,
            phase=None if phase is None else phase[:, -1],
            shifted=shifted,
        )


class SelectiveMemoryBlock(nn.Module):
    """Pre-norm residual memory mixer; a gated MLP follows only when d_ff > 0."""

    config: object

    @nn.compact
    def __call__(self, x, train=False, cache=None):
        config = self.config
        kwargs = dict(
            epsilon=config.norm_eps,
            dtype=dtype_from_name(config.dtype),
            param_dtype=dtype_from_name(config.param_dtype),
        )
        memory = SelectiveMemoryMixer(config, name="memory")(
            RMSNorm(name="memory_norm", **kwargs)(x), train=train, cache=cache
        )
        if cache is not None:
            memory, cache = memory
        x = x + memory
        if config.d_ff:
            x = x + FeedForward(config, name="mlp")(
                RMSNorm(name="mlp_norm", **kwargs)(x), train=train
            )
        return x if cache is None else (x, cache)


def _layer_types(config):
    """Selective memory blocks (M) and official Mamba-3 blocks (S)."""
    if config.memory_mixer != "selective":
        raise ValueError("Mamba 4 memory layers are selective conjugate memory")
    if config.protected_anchor_budget:
        raise ValueError("Mamba 4 memory layers do not implement protected banks")
    return [
        SelectiveMemoryBlock if kind == "M" else Mamba3Block
        for kind in config.layer_kinds
    ]


class Mamba3StepBlock(nn.Module):
    """One-token official Mamba-3 SISO layer with Mamba3Block's parameter tree.

    It follows the pinned step recurrence used by lm.decode, so a hybrid model
    can decode its Mamba-3 layers without changing the peer's source files.
    """

    config: object

    @nn.compact
    def __call__(self, x, cache):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        if config.mimo_rank != 1 or config.mamba3_outproj_norm:
            raise ValueError("Hybrid Mamba-3 decode implements the SISO screen block")
        inner = config.d_model * config.expand
        heads = inner // config.head_dim
        angles_count = int(config.d_state * config.rope_fraction) // 2
        bc_size = config.d_state * config.n_groups
        sizes = (inner, inner, bc_size, bc_size, heads, heads, heads, angles_count)
        normalized = RMSNorm(
            config.d_model, config.norm_eps, dtype, param_dtype, name="norm"
        )(x[:, 0])
        mixer = _Mamba3StepMixer(config, sizes, name="mixer")
        output, cache = mixer(normalized, cache)
        return x + output[:, None], cache


class _Mamba3StepMixer(nn.Module):
    config: object
    sizes: tuple

    @nn.compact
    def __call__(self, u, cache):
        config = self.config
        dtype = dtype_from_name(config.dtype)
        param_dtype = dtype_from_name(config.param_dtype)
        inner = config.d_model * config.expand
        heads = inner // config.head_dim
        batch = u.shape[0]
        projected = nn.Dense(
            sum(self.sizes),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            name="in_proj",
        )(u)
        sections, cursor = [], 0
        for size in self.sizes:
            sections.append(projected[..., cursor : cursor + size])
            cursor += size
        z, value, b, c, raw_dt, raw_a, trap, angles = sections
        b = b.reshape(batch, 1, config.n_groups, config.d_state)
        c = c.reshape(b.shape)
        b = RMSNorm(config.d_state, config.norm_eps, dtype, param_dtype, name="B_norm")(
            b
        )
        c = RMSNorm(config.d_state, config.norm_eps, dtype, param_dtype, name="C_norm")(
            c
        )
        b, c = jnp.transpose(b, (0, 2, 1, 3)), jnp.transpose(c, (0, 2, 1, 3))
        b = jnp.repeat(b, heads // config.n_groups, axis=1)
        c = jnp.repeat(c, heads // config.n_groups, axis=1)
        shape = (heads, 1, config.d_state)
        b_bias = self.param("B_bias", nn.initializers.ones, shape, param_dtype)
        c_bias = self.param("C_bias", nn.initializers.ones, shape, param_dtype)
        dt_bias = self.param("dt_bias", timestep_bias_init, (heads,), jnp.float32)
        skip = self.param("D", nn.initializers.ones, (heads,), param_dtype)
        b = (b.astype(jnp.float32) + b_bias).astype(dtype)
        c = (c.astype(jnp.float32) + c_bias).astype(dtype)
        dt = jax.nn.softplus(raw_dt.astype(jnp.float32) + dt_bias)
        a = -jnp.maximum(heavy_tail_activation(raw_a.astype(jnp.float32)), 1e-4)
        phase = cache["phase"] + angles.astype(jnp.float32)[:, None, :] * dt[..., None]
        b = rotate(b, phase[..., None, :], True)
        c = rotate(c, phase[..., None, :], True)
        value = value.reshape(batch, heads, 1, config.head_dim)
        gates = z.reshape(value.shape)
        trap = jax.nn.sigmoid(trap.astype(jnp.float32))
        current = jnp.einsum(
            "bhrn,bhrp->bhnp", b.astype(jnp.float32), value.astype(jnp.float32)
        )
        previous = jnp.einsum(
            "bhrn,bhrp->bhnp",
            cache["previous_keys"].astype(jnp.float32),
            cache["previous_values"].astype(jnp.float32),
        )
        alpha = jnp.exp(a * dt)[..., None, None]
        state = alpha * (cache["state"] + (dt * (1 - trap))[..., None, None] * previous)
        state += (dt * trap)[..., None, None] * current
        y = jnp.einsum("bhrn,bhnp->bhrp", c.astype(jnp.float32), state)
        y += value.astype(jnp.float32) * skip[None, :, None, None]
        y *= jax.nn.silu(gates.astype(jnp.float32))
        y = y[..., 0, :].reshape(batch, inner).astype(dtype)
        output = nn.Dense(
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            name="out_proj",
        )(y)
        return output, dict(
            state=state, previous_keys=b, previous_values=value, phase=phase
        )


class Mamba4LM(nn.Module):
    config: object

    @nn.compact
    def __call__(self, tokens, train=False):
        config = self.config
        embedding = TokenEmbedding(config, name="token_embedding")
        x = embedding(tokens)
        for i, kind in enumerate(_layer_types(config)):
            block = remat_block(kind, config, memory=kind is SelectiveMemoryBlock)
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
        for i, kind in enumerate(_layer_types(config)):
            if kind is Mamba3Block:
                x, layer_cache = Mamba3StepBlock(config, name=f"layer_{i}")(x, cache[i])
            else:
                x, layer_cache = kind(config, name=f"layer_{i}")(x, False, cache[i])
            updated.append(layer_cache)
        x = RMSNorm(
            epsilon=config.norm_eps,
            dtype=dtype_from_name(config.dtype),
            param_dtype=dtype_from_name(config.param_dtype),
            name="final_norm",
        )(x)
        return embedding.attend(x)[:, 0], tuple(updated)


def initialize_cache(config, batch_size):
    """Constant-size cache for every layer; its size does not grow with context."""
    _layer_types(config)
    return _selective_cache(config, batch_size)


def decode_step(config, params, tokens, cache, position=None):
    """One cached decode step; recurrent caches make position implicit."""
    model = Mamba4LM(config)
    return model.apply({"params": params}, tokens, cache, method=model.decode_step)


def _selective_cache(config, batch_size):
    """Evidence, cross statistics, conv history, order, previous key, SSM state."""
    heads, dtype = config.num_heads, dtype_from_name(config.dtype)
    memory_heads, memory_dim = config.memory_heads, config.memory_value_dim
    channels = memory_heads * (memory_dim + 2 * config.key_dim)
    layers = []
    for kind in config.layer_kinds:
        if kind == "M":
            layers.append(
                dict(
                    memory=selective_initial_state(
                        batch_size, memory_heads, config.key_dim, memory_dim
                    ),
                    conv=jnp.zeros(
                        (batch_size, max(config.conv_kernel - 1, 0), channels), dtype
                    )
                    if config.conv_kernel
                    else None,
                    order=jnp.zeros((batch_size, memory_heads, memory_dim))
                    if config.order_head
                    else None,
                    phase=jnp.zeros(
                        (batch_size, memory_heads, int(config.key_dim * rope) // 2),
                        jnp.float32,
                    )
                    if (rope := config.memory_rope_fraction)
                    else None,
                    shifted=jnp.zeros(
                        (batch_size, memory_heads, config.key_dim), jnp.float32
                    )
                    if config.memory_key_shift
                    else None,
                )
            )
        else:
            angles = int(config.d_state * config.rope_fraction) // 2
            layers.append(
                dict(
                    state=jnp.zeros(
                        (batch_size, heads, config.d_state, config.head_dim),
                        jnp.float32,
                    ),
                    previous_keys=jnp.zeros(
                        (batch_size, heads, 1, config.d_state), dtype
                    ),
                    previous_values=jnp.zeros(
                        (batch_size, heads, 1, config.head_dim), dtype
                    ),
                    phase=jnp.zeros((batch_size, heads, angles), jnp.float32),
                )
            )
    return tuple(layers)
