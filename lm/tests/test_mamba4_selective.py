"""Conformance of the selective fixed-floor memory used by screen-60m-v2."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lm.config import ModelConfig
from lm.kernels.mamba4 import (
    selective_decode,
    selective_gaussian_memory,
    selective_initial_state,
    spd_solve,
)
from lm.kernels.mamba4_reference import selective_dense_reference
from lm.models.mamba4 import Mamba4LM, decode_step, initialize_cache


def arrays(time=13, heads=3, dim=4, value_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(2, time, heads, dim)).astype(np.float32)
    q = rng.normal(size=k.shape).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = rng.normal(size=(2, time, heads, value_dim)).astype(np.float32)
    beta = rng.uniform(0.05, 1.0, size=k.shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.3, 0.999, size=k.shape[:-1]).astype(np.float32)
    floor = rng.uniform(0.25, 1.5, size=(heads, dim)).astype(np.float32)
    return k, v, q, beta, decay, floor


@pytest.mark.parametrize("chunk", [1, 4, 5, 13, 16])
def test_chunked_reads_match_sequential_dense_solves(chunk):
    k, v, q, beta, decay, floor = arrays()
    expected = selective_dense_reference(k, v, q, beta, decay, floor)
    actual = jax.jit(
        lambda *a: selective_gaussian_memory(*a, chunk_size=chunk),
    )(k, v, q, beta, np.log(decay), floor)
    for x, y in zip(
        (actual.output, actual.variance, actual.state.evidence, actual.state.cross),
        expected,
    ):
        np.testing.assert_allclose(x, y, rtol=3e-5, atol=3e-6)


def test_every_precision_respects_the_undiscounted_floor():
    k, v, q, beta, decay, floor = arrays(time=40)
    # Arbitrary token gates, including near-total forgetting and no forgetting.
    decay[:, ::7] = 1e-4
    decay[:, 3::7] = 1.0
    result = selective_gaussian_memory(
        k, v, q, beta, np.log(decay), floor, chunk_size=8
    )
    precision = result.state.evidence + floor[None, :, :, None] * np.eye(4)
    smallest = np.linalg.eigvalsh(np.asarray(precision))[..., 0]
    assert np.all(smallest >= floor.min(axis=-1)[None] - 1e-5)
    # A unit query cannot have latent variance above 1 / min(floor).
    assert np.all(np.asarray(result.variance) <= 1 / floor.min() + 1e-5)


def test_custom_solve_derivative_matches_dense_autodiff():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(7, 6, 6)).astype(np.float32)
    a = a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(6, dtype=np.float32)
    rhs = rng.normal(size=(7, 6)).astype(np.float32)

    def lanes(matrix, vector):
        symmetric = (matrix + jnp.swapaxes(matrix, -1, -2)) / 2
        return jnp.sum(jnp.sin(spd_solve(symmetric, vector)))

    def dense(matrix, vector):
        symmetric = (matrix + jnp.swapaxes(matrix, -1, -2)) / 2
        return jnp.sum(jnp.sin(jnp.linalg.solve(symmetric, vector[..., None])[..., 0]))

    np.testing.assert_allclose(
        spd_solve(a, rhs), np.linalg.solve(a, rhs[..., None])[..., 0], rtol=2e-5
    )
    for x, y in zip(
        jax.grad(lanes, argnums=(0, 1))(a, rhs), jax.grad(dense, argnums=(0, 1))(a, rhs)
    ):
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=2e-5)


def test_all_input_gradients_match_dense_reference():
    k, v, q, beta, decay, floor = map(jnp.asarray, arrays(time=9))

    def fused(k, v, q, beta, log_decay, floor):
        result = selective_gaussian_memory(
            k, v, q, beta, log_decay, floor, chunk_size=4
        )
        return jnp.sum(jnp.sin(result.output)) + 0.1 * jnp.sum(result.variance)

    def dense(k, v, q, beta, log_decay, floor):
        read, variance, *_ = selective_dense_reference(
            k, v, q, beta, jnp.exp(log_decay), floor
        )
        return jnp.sum(jnp.sin(read)) + 0.1 * jnp.sum(variance)

    arguments = (k, v, q, beta, jnp.log(decay), floor)
    actual = jax.grad(fused, argnums=tuple(range(6)))(*arguments)
    expected = jax.grad(dense, argnums=tuple(range(6)))(*arguments)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=2e-5)
        assert float(jnp.linalg.norm(x)) > 1e-6


def test_decode_and_continuation_match_prefill():
    k, v, q, beta, decay, floor = arrays()
    log_decay = np.log(decay)
    full = selective_gaussian_memory(k, v, q, beta, log_decay, floor, chunk_size=4)
    first = selective_gaussian_memory(
        k[:, :6], v[:, :6], q[:, :6], beta[:, :6], log_decay[:, :6], floor, chunk_size=4
    )
    second = selective_gaussian_memory(
        k[:, 6:],
        v[:, 6:],
        q[:, 6:],
        beta[:, 6:],
        log_decay[:, 6:],
        floor,
        chunk_size=4,
        initial=first.state,
    )
    np.testing.assert_allclose(second.output, full.output[:, 6:], rtol=3e-5, atol=3e-6)
    state = selective_initial_state(2, 3, 4, 5)
    for t in range(k.shape[1]):
        step = selective_decode(
            state, k[:, t], v[:, t], q[:, t], beta[:, t], log_decay[:, t], floor
        )
        state = step.state
        np.testing.assert_allclose(step.output, full.output[:, t], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(
            step.variance, full.variance[:, t], rtol=3e-5, atol=3e-6
        )


def test_exact_recall_of_independent_keys_as_floor_vanishes():
    # Orthonormal keys, no forgetting: read(k_i) = v_i / (1 + floor / beta).
    dim, value_dim = 4, 3
    keys = np.eye(dim, dtype=np.float32)[None, :, None, :]
    values = np.random.default_rng(1).normal(size=(1, dim, 1, value_dim))
    beta = np.full((1, dim, 1), 2.0, np.float32)
    floor = np.full((1, dim), 1e-3, np.float32)
    result = selective_gaussian_memory(
        keys, values, keys, beta, np.zeros_like(beta), floor, chunk_size=2
    )
    final = result.state
    precision = final.evidence + 1e-3 * np.eye(dim)
    for i in range(dim):
        solved = np.linalg.solve(np.asarray(precision[0, 0]), keys[0, i, 0])
        read = np.asarray(final.cross[0, 0]) @ solved
        np.testing.assert_allclose(read, values[0, i, 0] / (1 + 5e-4), rtol=1e-5)


def model_config(pattern, **overrides):
    values = dict(
        architecture="mamba4",
        vocab_size=31,
        d_model=16,
        n_layers=len(pattern),
        d_ff=0,
        expand=2,
        head_dim=8,
        key_dim=4,
        d_state=8,
        chunk_size=4,
        protected_anchor_budget=0,
        dtype="float32",
        remat=True,
        memory_mixer="selective",
        layer_pattern=pattern,
        conv_kernel=4,
        max_seq_len=11,
    )
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    "pattern, rope, width, shift",
    [
        ("MM", 0.0, 0, False),
        ("SM", 0.0, 0, False),
        ("MS", 0.0, 0, False),
        ("MM", 1.0, 0, False),
        ("SM", 0.5, 0, False),
        ("SM", 0.5, 16, False),
        ("MM", 0.0, 0, True),
        ("SM", 0.0, 16, True),
    ],
)
def test_selective_lm_causality_gradients_and_cached_decode(
    pattern, rope, width, shift
):
    config = model_config(
        pattern,
        memory_rope_fraction=rope,
        memory_head_dim=width,
        memory_key_shift=shift,
    )
    model = Mamba4LM(config)
    tokens = (jnp.arange(11)[None] * 7) % config.vocab_size
    params = model.init(jax.random.key(0), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    changed = model.apply({"params": params}, tokens.at[:, 6:].set(3))
    np.testing.assert_allclose(logits[:, :6], changed[:, :6], atol=1e-6)

    def loss(p):
        y = model.apply({"params": p}, tokens)
        return -jnp.mean(
            jax.nn.log_softmax(y[:, :-1], -1)[0, jnp.arange(10), tokens[0, 1:]]
        )

    grads = jax.grad(loss)(params)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    memory = grads[f"layer_{pattern.index('M')}"]["memory"]
    for name in (
        "floor",
        "a_log",
        "dt_bias",
        "beta_bias",
        "conv_weight",
        "D",
        "confidence_scale",
        "order_mix",
    ):
        assert float(jnp.linalg.norm(memory[name])) > 1e-9, name
    cache = initialize_cache(config, 1)
    step = jax.jit(lambda token, cache: decode_step(config, params, token, cache))
    outputs = []
    for position in range(tokens.shape[1]):
        output, cache = step(tokens[:, position], cache)
        outputs.append(output)
    np.testing.assert_allclose(jnp.stack(outputs, 1), logits, rtol=2e-5, atol=2e-6)


def test_selective_repeated_tokens_and_diagnostics_are_finite():
    config = model_config("MSM", n_layers=3, max_seq_len=24)
    model = Mamba4LM(config)
    tokens = jnp.full((2, 24), 2, dtype=jnp.int32)
    variables = model.init(jax.random.key(5), tokens)

    def loss(p):
        logits = model.apply({"params": p}, tokens)
        return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[..., 2])

    value, grads = jax.value_and_grad(loss)(variables["params"])
    assert np.isfinite(value)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    ordinary = model.apply(variables, tokens)
    inspected, collections = model.apply(variables, tokens, mutable=["diagnostics"])
    np.testing.assert_array_equal(ordinary, inspected)
    for layer in ("layer_0", "layer_2"):
        metrics = collections["diagnostics"][layer]["memory"]
        assert float(metrics["precision_eigenvalue_min"][0]) >= 0.25 - 1e-5
        assert bool(metrics["gaussian_allfinite"][0])
        assert all(np.asarray(value[0]).ndim == 0 for value in metrics.values())
    assert "layer_1" not in collections["diagnostics"]


def test_v1_and_selective_options_are_rejected_when_inconsistent():
    with pytest.raises(ValueError, match="protected banks"):
        Mamba4LM(model_config("M", protected_anchor_budget=2)).init(
            jax.random.key(0), jnp.zeros((1, 4), jnp.int32)
        )
    with pytest.raises(ValueError, match="layer_pattern"):
        model_config("MX").layer_kinds


def test_loop_and_unrolled_solve_backends_agree():
    from lm.kernels import mamba4 as kernels

    rng = np.random.default_rng(9)
    a = rng.normal(size=(5, 3, 16, 16)).astype(np.float32)
    a = a @ np.swapaxes(a, -1, -2) / 16 + 0.3 * np.eye(16, dtype=np.float32)
    rhs = rng.normal(size=(5, 3, 16)).astype(np.float32)
    results = {
        backend: np.asarray(kernels._factor_and_solve(a, rhs, backend)[0])
        for backend in ("loop", "unrolled", "blocked", "auto")
    }
    expected = np.linalg.solve(a, rhs[..., None])[..., 0]
    for value in results.values():
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=3e-5)


def test_blocked_backend_gradients_match_loop():
    k, v, q, beta, decay, floor = map(jnp.asarray, arrays(time=9, dim=16, heads=2))

    def loss(solver, k, q, beta):
        result = selective_gaussian_memory(
            k, v, q, beta, jnp.log(decay), floor, chunk_size=4, solver=solver
        )
        return jnp.sum(jnp.sin(result.output)) + 0.1 * jnp.sum(result.variance)

    blocked = jax.grad(lambda *a: loss("blocked", *a), argnums=(0, 1, 2))(k, q, beta)
    loop = jax.grad(lambda *a: loss("loop", *a), argnums=(0, 1, 2))(k, q, beta)
    for x, y in zip(blocked, loop):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


def test_chunked_order_scan_matches_sequential_scan():
    from lm.models.mamba4 import order_memory, order_memory_chunked

    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.normal(size=(2, 37, 3, 5)), jnp.float32)
    logits = jnp.asarray(rng.normal(size=(2, 37, 3)) + 2, jnp.float32)

    def chunked(v, z):
        return order_memory_chunked(v, jax.nn.log_sigmoid(z), chunk_size=8)

    def reference(v, z):
        return order_memory(v, jax.nn.sigmoid(z))

    np.testing.assert_allclose(
        chunked(values, logits), reference(values, logits), rtol=1e-5, atol=1e-6
    )
    gradients = [
        jax.grad(lambda v, z, f=f: jnp.sum(jnp.sin(f(v, z))), argnums=(0, 1))(
            values, logits
        )
        for f in (chunked, reference)
    ]
    for x, y in zip(*gradients):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


def test_default_evidence_range_keeps_the_trained_parameterization():
    tokens = jnp.zeros((1, 8), jnp.int32)
    params = Mamba4LM(model_config("M")).init(jax.random.key(0), tokens)["params"]
    np.testing.assert_array_equal(params["layer_0"]["memory"]["beta_bias"], 0.0)
    wide = Mamba4LM(model_config("M", memory_beta_max=16.0))
    bias = wide.init(jax.random.key(0), tokens)["params"]["layer_0"]["memory"]
    np.testing.assert_allclose(16 * jax.nn.sigmoid(bias["beta_bias"]), 0.5, rtol=1e-6)


def test_wide_evidence_range_keeps_floor_decode_and_gradients():
    config = model_config("SM", memory_beta_max=8.0)
    model = Mamba4LM(config)
    tokens = (jnp.arange(11)[None] * 7) % config.vocab_size
    params = model.init(jax.random.key(0), tokens)["params"]
    memory = params["layer_1"]["memory"]
    memory["beta_bias"] = jnp.full_like(memory["beta_bias"], 6.0)
    logits, collections = model.apply(
        {"params": params}, tokens, mutable=["diagnostics"]
    )
    metrics = collections["diagnostics"]["layer_1"]["memory"]
    assert float(metrics["beta_max"][0]) > 7.5
    assert float(metrics["precision_eigenvalue_min"][0]) >= 0.25 - 1e-5
    grads = jax.grad(lambda p: jnp.mean(model.apply({"params": p}, tokens) ** 2))(
        params
    )
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    cache = initialize_cache(config, 1)
    step = jax.jit(lambda token, cache: decode_step(config, params, token, cache))
    outputs = []
    for position in range(tokens.shape[1]):
        output, cache = step(tokens[:, position], cache)
        outputs.append(output)
    np.testing.assert_allclose(jnp.stack(outputs, 1), logits, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="memory_beta_max"):
        Mamba4LM(model_config("M", memory_beta_max=0.0)).init(jax.random.key(0), tokens)


def test_key_shift_writes_each_value_under_the_previous_context():
    """With the shift, the first token writes nothing and later writes move."""
    config = model_config("M", memory_key_shift=True, max_seq_len=8)
    model = Mamba4LM(config)
    tokens = jnp.array([[3, 5, 7, 11, 13, 17, 19, 23]])
    params = model.init(jax.random.key(2), tokens)["params"]
    _, state = model.apply({"params": params}, tokens[:, :1], mutable=["diagnostics"])
    memory = state["diagnostics"]["layer_0"]["memory"]
    # A single shifted token has a zero key: its evidence is exactly the floor.
    assert float(memory["precision_eigenvalue_max"][0]) == pytest.approx(
        float(memory["floor_max"][0]), rel=1e-6
    )
    with pytest.raises(ValueError, match="memory_key_shift"):
        Mamba4LM(
            model_config("M", memory_key_shift=True, memory_rope_fraction=1.0)
        ).init(jax.random.key(0), tokens)
