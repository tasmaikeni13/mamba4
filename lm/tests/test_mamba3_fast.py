"""The hand-derived SSD backward reproduces the reference kernel exactly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lm.kernels.mamba3 import mamba3_chunked
from lm.kernels.mamba3_fast import mamba3_fast
from lm.models.mamba4 import Mamba4LM, decode_step, initialize_cache
from lm.tests.test_mamba4 import model_config


@pytest.mark.parametrize("length, chunk", [(45, 16), (32, 8), (7, 16)])
def test_fast_ssd_matches_reference_values_and_gradients(length, chunk):
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

    def run(kernel, *a):
        return kernel(*a[:7], chunk_size=chunk, q_bias=a[7], k_bias=a[8])

    for a, b in zip(run(mamba3_chunked, *args), run(mamba3_fast, *args)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
    weight_y = jnp.asarray(rng.normal(size=v.shape), f32)
    weight_f = jnp.asarray(rng.normal(size=(batch, heads, state, width)), f32)

    def loss(kernel, *a):
        y, final = run(kernel, *a)
        return jnp.sum(y * weight_y) + jnp.sum(final * weight_f)

    reference = jax.grad(lambda *a: loss(mamba3_chunked, *a), argnums=range(9))(*args)
    candidate = jax.grad(lambda *a: loss(mamba3_fast, *a), argnums=range(9))(*args)
    for a, b in zip(reference, candidate):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=5e-5)


def test_fast_kernels_keep_hybrid_prefill_equal_to_decode():
    config = model_config(
        "SM", memory_solver="fused", memory_key_shift=True, ssd_kernel="fast"
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
