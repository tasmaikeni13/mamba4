"""Actual one-token LM decoding with KV or constant-size recurrent caches.

These helpers share trained Flax parameters; they do not re-evaluate prefixes.
The optional Mamba-4 implementation is delegated to its architecture module.
"""

import jax
import jax.numpy as jnp

from lm.kernels.mamba3 import rotate
from lm.models.common import dtype_from_name
from lm.models.mamba3 import heavy_tail_activation


def initialize_cache(config, batch_size):
    dtype = dtype_from_name(config.dtype)
    if config.architecture == "transformer":
        shape = (
            config.n_layers,
            batch_size,
            config.n_heads,
            config.max_seq_len,
            config.d_model // config.n_heads,
        )
        return {"keys": jnp.zeros(shape, dtype), "values": jnp.zeros(shape, dtype)}
    if config.architecture == "mamba3":
        heads = config.num_heads
        return {
            "state": jnp.zeros(
                (config.n_layers, batch_size, heads, config.d_state, config.head_dim),
                jnp.float32,
            ),
            "previous_keys": jnp.zeros(
                (config.n_layers, batch_size, heads, config.mimo_rank, config.d_state),
                dtype,
            ),
            "previous_values": jnp.zeros(
                (config.n_layers, batch_size, heads, config.mimo_rank, config.head_dim),
                dtype,
            ),
            "phase": jnp.zeros(
                (
                    config.n_layers,
                    batch_size,
                    heads,
                    int(config.d_state * config.rope_fraction) // 2,
                ),
                jnp.float32,
            ),
        }
    if config.architecture == "mamba4":
        from lm.models.mamba4 import initialize_cache as initialize_mamba4_cache

        return initialize_mamba4_cache(config, batch_size)
    raise ValueError(f"Unknown architecture {config.architecture}")


def _dense(x, parameters, dtype):
    return jnp.matmul(x.astype(dtype), parameters["kernel"].astype(dtype))


def _norm(x, parameters, config):
    x32 = x.astype(jnp.float32)
    output = x32 * jax.lax.rsqrt(
        jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + config.norm_eps
    )
    return (output * parameters["scale"]).astype(dtype_from_name(config.dtype))


def _finish(x, params, config):
    x = _norm(x, params["final_norm"], config)
    embedding = params["tokens"]["embedding"]["embedding"]
    return jnp.matmul(x, embedding.astype(x.dtype).T).astype(jnp.float32)


def _transformer(config, params, token, cache, position):
    dtype = dtype_from_name(config.dtype)
    x = params["tokens"]["embedding"]["embedding"][token].astype(dtype)
    keys, values = cache["keys"], cache["values"]
    channels = config.d_model // config.n_heads
    frequencies = 10000.0 ** (-jnp.arange(0, channels, 2, dtype=jnp.float32) / channels)
    angles = position.astype(jnp.float32) * frequencies
    for index in range(config.n_layers):
        layer = params[f"layer_{index}"]
        normalized = _norm(x, layer["attention_norm"], config)
        qkv = _dense(normalized, layer["qkv_proj"], dtype)
        qkv = qkv.reshape(x.shape[0], 3, config.n_heads, channels)
        q, k, v = (qkv[:, part] for part in range(3))
        q, k = rotate(q, angles), rotate(k, angles)
        keys = keys.at[index, :, :, position, :].set(k)
        values = values.at[index, :, :, position, :].set(v)
        scores = (
            jnp.einsum(
                "bhd,bhsd->bhs", q, keys[index], preferred_element_type=jnp.float32
            )
            * channels**-0.5
        )
        scores = jnp.where(
            jnp.arange(config.max_seq_len)[None, None] <= position, scores, -jnp.inf
        )
        probabilities = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum(
            "bhs,bhsd->bhd", probabilities.astype(dtype), values[index]
        ).reshape(x.shape)
        x = x + _dense(output, layer["out_proj"], dtype)
        normalized = _norm(x, layer["ffn_norm"], config)
        gate, value = jnp.split(
            _dense(normalized, layer["ffn"]["in_proj"], dtype), 2, axis=-1
        )
        x = x + _dense(jax.nn.silu(gate) * value, layer["ffn"]["out_proj"], dtype)
    return _finish(x, params, config), {"keys": keys, "values": values}


def _mamba3(config, params, token, cache):
    dtype = dtype_from_name(config.dtype)
    x = params["tokens"]["embedding"]["embedding"][token].astype(dtype)
    heads, rank, inner = (
        config.num_heads,
        config.mimo_rank,
        config.d_model * config.expand,
    )
    angles_count = int(config.d_state * config.rope_fraction) // 2
    bc_size = config.d_state * config.n_groups * rank
    sizes = (inner, inner, bc_size, bc_size, heads, heads, heads, angles_count)
    states, old_keys, old_values, phases = (
        cache[name] for name in ("state", "previous_keys", "previous_values", "phase")
    )
    for index in range(config.n_layers):
        layer = params[f"layer_{index}"]
        mixer = layer["mixer"]
        normalized = _norm(x, layer["norm"], config)
        projected = _dense(normalized, mixer["in_proj"], dtype)
        sections, cursor = [], 0
        for size in sizes:
            sections.append(projected[..., cursor : cursor + size])
            cursor += size
        z, value, b, c, raw_dt, raw_a, trap, angles = sections
        b = b.reshape(x.shape[0], rank, config.n_groups, config.d_state)
        c = c.reshape(b.shape)
        b, c = _norm(b, mixer["B_norm"], config), _norm(c, mixer["C_norm"], config)
        b, c = jnp.transpose(b, (0, 2, 1, 3)), jnp.transpose(c, (0, 2, 1, 3))
        b = jnp.repeat(b, heads // config.n_groups, axis=1)
        c = jnp.repeat(c, heads // config.n_groups, axis=1)
        b = (b.astype(jnp.float32) + mixer["B_bias"]).astype(dtype)
        c = (c.astype(jnp.float32) + mixer["C_bias"]).astype(dtype)
        dt = jax.nn.softplus(raw_dt.astype(jnp.float32) + mixer["dt_bias"])
        a = -jnp.maximum(heavy_tail_activation(raw_a.astype(jnp.float32)), 1e-4)
        phase = phases[index] + angles.astype(jnp.float32)[:, None, :] * dt[..., None]
        b, c = (
            rotate(b, phase[..., None, :], rank == 1),
            rotate(c, phase[..., None, :], rank == 1),
        )
        value = value.reshape(x.shape[0], heads, config.head_dim)
        z = z.reshape(value.shape)
        if rank > 1:
            value = (value[..., None, :] * mixer["mimo_x"].astype(dtype)).astype(dtype)
            gates = (z[..., None, :] * mixer["mimo_z"].astype(dtype)).astype(dtype)
        else:
            value, gates = value[..., None, :], z[..., None, :]
        trap = jax.nn.sigmoid(trap.astype(jnp.float32))
        current = jnp.einsum(
            "bhrn,bhrp->bhnp", b.astype(jnp.float32), value.astype(jnp.float32)
        )
        previous = jnp.einsum(
            "bhrn,bhrp->bhnp",
            old_keys[index].astype(jnp.float32),
            old_values[index].astype(jnp.float32),
        )
        alpha = jnp.exp(a * dt)[..., None, None]
        state = alpha * (states[index] + (dt * (1 - trap))[..., None, None] * previous)
        state += (dt * trap)[..., None, None] * current
        y = jnp.einsum("bhrn,bhnp->bhrp", c.astype(jnp.float32), state)
        y += value.astype(jnp.float32) * mixer["D"][None, :, None, None]
        if config.mamba3_outproj_norm:
            y *= jax.lax.rsqrt(
                jnp.mean(jnp.square(y), axis=-1, keepdims=True) + config.norm_eps
            )
            y *= mixer["out_norm_scale"][None, :, None, :]
        y *= jax.nn.silu(gates.astype(jnp.float32))
        y = jnp.sum(y * mixer["mimo_o"][None], axis=-2) if rank > 1 else y[..., 0, :]
        x = x + _dense(
            y.reshape(x.shape[0], inner).astype(dtype), mixer["out_proj"], dtype
        )
        states = states.at[index].set(state)
        old_keys, old_values, phases = (
            old_keys.at[index].set(b),
            old_values.at[index].set(value),
            phases.at[index].set(phase),
        )
    return _finish(x, params, config), {
        "state": states,
        "previous_keys": old_keys,
        "previous_values": old_values,
        "phase": phases,
    }


def decode_step(config, params, token, cache, position):
    """Evaluate just the supplied token, using and updating the supplied cache."""
    if config.architecture == "transformer":
        return _transformer(config, params, token, cache, jnp.asarray(position))
    if config.architecture == "mamba3":
        return _mamba3(config, params, token, cache)
    if config.architecture == "mamba4":
        from lm.models.mamba4 import decode_step as mamba4_decode_step

        return mamba4_decode_step(config, params, token, cache, position)
    raise ValueError(f"Unknown architecture {config.architecture}")
