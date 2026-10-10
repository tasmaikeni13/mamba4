"""FlashMamba's Pallas scan reproduces the reference kernel exactly (CPU interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lm.kernels.flashmamba import mamba3_flash
from lm.kernels.mamba3 import mamba3_chunked
from lm.models.mamba4 import Mamba4LM, decode_step, initialize_cache
from lm.tests.test_mamba4 import model_config


@pytest.mark.parametrize("length, chunk", [(45, 16), (32, 8), (7, 16), (40, 128)])
def test_flash_ssd_matches_reference_values_and_gradients(length, chunk):
    rng = np.random.default_rng(length)
    batch, heads, state, width = 2, 3, 8, 6
    f32 = jnp.float32
    q = jnp.asarray(rng.normal(size=(batch, length, heads, 1, state)), f32)
    k = jnp.asarray(rng.normal(size=q.shape), f32)
    v = jnp.asarray(rng.normal(size=(batch, length, heads, 1, width)), f32)
    dt = jnp.asarray(np.log1p(np.exp(rng.normal(size=(batch, length, heads)) - 1)), f32)
    adt = -dt * jnp.asarray(rng.uniform(0.5, 2.0, dt.shape), f32)
    trap = jnp.asarray(rng.normal(size=dt.shape), f32)
    angles = jnp.asarray(rng.normal(size=(batch, length, heads, state // 4)), f32)
    q_bias = jnp.asarray(rng.normal(size=(heads, 1, state)), f32) * 0.1
    k_bias = jnp.asarray(rng.normal(size=(heads, 1, state)), f32) * 0.1
    args = (q, k, v, adt, dt, trap, angles, q_bias, k_bias)

    def reference(*a):
        return mamba3_chunked(*a[:7], chunk_size=16, q_bias=a[7], k_bias=a[8])

    def flash(*a):
        return mamba3_flash(*a[:7], chunk_size=chunk, q_bias=a[7], k_bias=a[8])

    for a, b in zip(reference(*args), flash(*args)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
    weight_y = jnp.asarray(rng.normal(size=v.shape), f32)
    weight_f = jnp.asarray(rng.normal(size=(batch, heads, state, width)), f32)

    def loss(kernel, *a):
        y, final = kernel(*a)
        return jnp.sum(y * weight_y) + jnp.sum(final * weight_f)

    expected = jax.grad(lambda *a: loss(reference, *a), argnums=range(9))(*args)
    actual = jax.grad(lambda *a: loss(flash, *a), argnums=range(9))(*args)
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=5e-5)


def test_flash_kernels_keep_hybrid_prefill_equal_to_decode():
    config = model_config(
        "SM", memory_solver="fused", memory_key_shift=True, ssd_kernel="flash"
    )
    model = Mamba4LM(config)
    tokens = (jnp.arange(11)[None] * 3) % config.vocab_size
    params = model.init(jax.random.key(4), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    cache = initialize_cache(config, 1)
    step = jax.jit(lambda token, cache: decode_step(config, params, token, cache))
    outputs = []
    for position in range(tokens.shape[1]):
        output, cache = step(tokens[:, position], cache)
        outputs.append(output)
    np.testing.assert_allclose(jnp.stack(outputs, 1), logits, rtol=2e-5, atol=2e-6)


def test_jitted_hybrid_gradients_match_across_remat_policies():
    """Under jit, keeping the kernel outputs gives the full-remat gradients."""
    import dataclasses

    config = model_config(
        "SMS",
        memory_solver="fused",
        memory_key_shift=True,
        ssd_kernel="flash",
        remat=True,
        remat_policy="kernels",
    )
    tokens = (jnp.arange(24)[None] * 7) % config.vocab_size
    params = Mamba4LM(config).init(jax.random.key(1), tokens)["params"]

    def gradients(policy):
        model = Mamba4LM(dataclasses.replace(config, remat_policy=policy))

        def loss(params):
            logits = model.apply({"params": params}, tokens)
            return jnp.mean(logits.astype(jnp.float32) ** 2)

        return jax.jit(jax.grad(loss))(params)

    reference = jax.tree.leaves(gradients("full"))
    for policy in ("kernels", "mixers", "mixers-1"):
        for a, b in zip(jax.tree.leaves(gradients(policy)), reference):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "length, chunk, heads_per_step", [(45, 16, 2), (40, 8, 4), (7, 16, 1)]
)
def test_fused_layout_scan_matches_reference_values_and_gradients(
    length, chunk, heads_per_step
):
    from lm.kernels.flashmamba import mamba3_fused

    rng = np.random.default_rng(length + chunk)
    batch, heads, state, width, angles_count = 2, 4, 8, 6, 2
    f32 = jnp.float32
    c = jnp.asarray(rng.normal(size=(batch, length, state)), f32)
    b = jnp.asarray(rng.normal(size=c.shape), f32)
    x = jnp.asarray(rng.normal(size=(batch, length, heads * width)), f32)
    dt = jnp.asarray(np.log1p(np.exp(rng.normal(size=(batch, length, heads)) - 1)), f32)
    adt = -dt * jnp.asarray(rng.uniform(0.5, 2.0, dt.shape), f32)
    trap = jnp.asarray(rng.normal(size=dt.shape), f32)
    angles = jnp.asarray(rng.normal(size=(batch, length, angles_count)), f32)
    q_bias = jnp.asarray(rng.normal(size=(heads, 1, state)), f32) * 0.1
    k_bias = jnp.asarray(rng.normal(size=(heads, 1, state)), f32) * 0.1
    args = (c, b, x, adt, dt, trap, angles, q_bias, k_bias)

    def reference(c, b, x, adt, dt, trap, angles, q_bias, k_bias):
        q = jnp.broadcast_to(c[:, :, None, None], (batch, length, heads, 1, state))
        k = jnp.broadcast_to(b[:, :, None, None], q.shape)
        v = x.reshape(batch, length, heads, 1, width)
        per_head = jnp.broadcast_to(
            angles[:, :, None], (batch, length, heads, angles_count)
        )
        y, final = mamba3_chunked(
            q,
            k,
            v,
            adt,
            dt,
            trap,
            per_head,
            chunk_size=16,
            q_bias=q_bias,
            k_bias=k_bias,
        )
        return y.reshape(batch, length, heads * width), final

    def fused(c, b, x, adt, dt, trap, angles, q_bias, k_bias):
        return mamba3_fused(
            c,
            b,
            x,
            adt,
            dt,
            trap,
            angles,
            q_bias=q_bias,
            k_bias=k_bias,
            chunk_size=chunk,
            heads_per_step=heads_per_step,
        )

    for a, b_ in zip(reference(*args), fused(*args)):
        np.testing.assert_allclose(a, b_, rtol=2e-5, atol=2e-5)
    weight_y = jnp.asarray(rng.normal(size=x.shape), f32)
    weight_f = jnp.asarray(rng.normal(size=(batch, heads, state, width)), f32)

    def loss(kernel, *a):
        y, final = kernel(*a)
        return jnp.sum(y * weight_y) + jnp.sum(final * weight_f)

    expected = jax.grad(lambda *a: loss(reference, *a), argnums=range(9))(*args)
    actual = jax.grad(lambda *a: loss(fused, *a), argnums=range(9))(*args)
    for a, b_ in zip(expected, actual):
        np.testing.assert_allclose(a, b_, rtol=5e-5, atol=5e-5)
