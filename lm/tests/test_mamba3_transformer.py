"""Full-model and independently derived kernel conformance checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from lm.config import ModelConfig
from lm.decode import decode_step, initialize_cache
from lm.kernels.attention import attention_reference, causal_attention
from lm.kernels.mamba3 import mamba3_chunked, mamba3_sequential, numpy_reference
from lm.models.mamba3 import Mamba3LM
from lm.models.transformer import TransformerLM


def kernel_inputs(rank=1, length=11):
    rng = np.random.default_rng(919 + rank)
    shape = (2, length, 2, rank, 8)
    q, k = (rng.normal(size=shape).astype(np.float32) * 0.3 for _ in range(2))
    v = rng.normal(size=(*shape[:4], 4)).astype(np.float32) * 0.3
    dt = rng.uniform(0.05, 0.6, size=shape[:3]).astype(np.float32)
    adt = -rng.uniform(0.1, 2.0, size=shape[:3]).astype(np.float32) * dt
    trap = rng.normal(size=shape[:3]).astype(np.float32)
    # Fractional rotary dimensions exercise both rotating and fixed coordinates.
    angles = rng.normal(size=(*shape[:3], 2)).astype(np.float32)
    return tuple(jnp.asarray(array) for array in (q, k, v, adt, dt, trap, angles))


@pytest.mark.parametrize("rank,pairwise", [(1, True), (2, False), (3, False)])
@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16])
def test_ssd_matches_local_complex_reference(rank, pairwise, chunk_size):
    inputs = kernel_inputs(rank)
    bias = jnp.arange(2 * rank * 8).reshape(2, rank, 8).astype(jnp.float32) / 100
    options = {"q_bias": bias, "k_bias": -bias, "pairwise": pairwise}
    actual, state = mamba3_chunked(*inputs, chunk_size=chunk_size, **options)
    expected = numpy_reference(*inputs, **options)
    sequential, sequential_state = mamba3_sequential(*inputs, **options)
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert_allclose(actual, sequential, rtol=2e-5, atol=2e-6)
    assert_allclose(state, sequential_state, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("rank", [1, 2])
def test_ssd_all_input_gradients_match_recurrence(rank):
    inputs = kernel_inputs(rank, length=7)
    cotangent = jax.random.normal(jax.random.PRNGKey(222), inputs[2].shape)
    options = {"pairwise": rank == 1}

    def objective(function, *arrays):
        output, state = function(*arrays, **options)
        return jnp.sum(output * cotangent) + 0.03 * jnp.sum(state * state)

    def production(*arrays):
        return objective(
            lambda *xs, **kw: mamba3_chunked(*xs, chunk_size=4, **kw), *arrays
        )

    def reference(*arrays):
        return objective(mamba3_sequential, *arrays)

    indices = tuple(range(len(inputs)))
    actual = jax.grad(production, argnums=indices)(*inputs)
    expected = jax.grad(reference, argnums=indices)(*inputs)
    for gradient, reference_gradient in zip(actual, expected):
        assert_allclose(gradient, reference_gradient, rtol=3e-4, atol=3e-6)
        assert np.all(np.isfinite(gradient))


def test_chunked_no_future_trap_or_phase_leakage():
    inputs = kernel_inputs(length=11)
    expected, _ = mamba3_chunked(*inputs, chunk_size=4)
    changed = tuple(array.at[:, 5:].set(array[:, 5:] + 0.7) for array in inputs)
    actual, _ = mamba3_chunked(*changed, chunk_size=4)
    assert_allclose(actual[:, :5], expected[:, :5], atol=1e-7)


def test_attention_forward_and_backward_conformance():
    q, k, v = (
        jax.random.normal(jax.random.PRNGKey(seed), (2, 9, 2, 4)) for seed in range(3)
    )
    assert_allclose(causal_attention(q, k, v), attention_reference(q, k, v), atol=1e-6)
    cotangent = jax.random.normal(jax.random.PRNGKey(8), v.shape)
    arguments = (0, 1, 2)
    actual = jax.grad(
        lambda *xs: jnp.sum(causal_attention(*xs) * cotangent), argnums=arguments
    )(q, k, v)
    expected = jax.grad(
        lambda *xs: jnp.sum(attention_reference(*xs) * cotangent), argnums=arguments
    )(q, k, v)
    for grad, ref in zip(actual, expected):
        assert_allclose(grad, ref, rtol=1e-5, atol=2e-6)


def tiny_config(architecture):
    return ModelConfig(
        architecture=architecture,
        vocab_size=31,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=24,
        d_state=8,
        head_dim=8,
        chunk_size=4,
        dtype="float32",
        max_seq_len=16,
        remat=True,
    )


@pytest.mark.parametrize(
    "model_cls,architecture", [(TransformerLM, "transformer"), (Mamba3LM, "mamba3")]
)
def test_full_lm_causality_tied_embedding_and_learning(model_cls, architecture):
    model = model_cls(tiny_config(architecture))
    tokens = (jnp.arange(18).reshape(2, 9) % 31).astype(jnp.int32)
    params = model.init(jax.random.PRNGKey(42), tokens)["params"]
    output = jax.jit(model.apply)({"params": params}, tokens)
    changed = tokens.at[:, 5:].set(30)
    other = model.apply({"params": params}, changed)
    assert output.shape == (2, 9, 31)
    assert_allclose(output[:, :5], other[:, :5], atol=2e-6)
    assert "tokens" in params
    assert "lm_head" not in params
    labels = (tokens + 1) % 31

    def loss(parameters):
        logits = model.apply({"params": parameters}, tokens, train=True)
        return -jnp.mean(
            jnp.take_along_axis(jax.nn.log_softmax(logits), labels[..., None], axis=-1)
        )

    before, gradients = jax.jit(jax.value_and_grad(loss))(params)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    after_params = jax.tree.map(
        lambda parameter, gradient: parameter - 0.01 * gradient, params, gradients
    )
    after = loss(after_params)
    assert after < before


def test_optional_full_mimo_norm_path_has_gradients():
    config = replace(tiny_config("mamba3"), mimo_rank=2, mamba3_outproj_norm=True)
    model = Mamba3LM(config)
    tokens = jnp.array([[1, 2, 3, 4, 5]])
    params = model.init(jax.random.PRNGKey(71), tokens)["params"]
    gradients = jax.grad(
        lambda p: jnp.sum(jnp.square(model.apply({"params": p}, tokens)))
    )(params)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    for key in ("mimo_x", "mimo_z", "mimo_o", "B_bias", "C_bias", "dt_bias"):
        assert np.linalg.norm(gradients["layer_0"]["mixer"][key]) > 0


@pytest.mark.parametrize(
    "architecture,rank", [("transformer", 1), ("mamba3", 1), ("mamba3", 2)]
)
def test_cached_decode_matches_full_prefill(architecture, rank):
    config = replace(tiny_config(architecture), mimo_rank=rank)
    model = TransformerLM(config) if architecture == "transformer" else Mamba3LM(config)
    tokens = jnp.array([[1, 4, 2, 8, 3, 5, 7], [2, 5, 1, 7, 9, 6, 8]])
    params = model.init(jax.random.PRNGKey(441), tokens)["params"]
    expected = model.apply({"params": params}, tokens)
    step = jax.jit(
        lambda token, cache, position: decode_step(
            config, params, token, cache, position
        )
    )
    cache = initialize_cache(config, tokens.shape[0])
    actual = []
    for position in range(tokens.shape[1]):
        logits, cache = step(tokens[:, position], cache, jnp.array(position, jnp.int32))
        actual.append(logits)
    assert_allclose(jnp.stack(actual, axis=1), expected, rtol=3e-5, atol=3e-6)


def test_recurrent_decode_cache_does_not_grow_with_sequence_length():
    config = tiny_config("mamba3")
    small = initialize_cache(config, 2)
    large = initialize_cache(replace(config, max_seq_len=65536), 2)
    assert jax.tree.map(lambda x: x.shape, small) == jax.tree.map(
        lambda x: x.shape, large
    )
