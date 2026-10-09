"""Independent Gaussian, gradient, floor and protected-routing conformance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from analysis.memory import CyclicFloorMemory, ProtectedCascade
from lm.config import ModelConfig
from lm.kernels.mamba4 import (
    cyclic_decode,
    gaussian_memory,
    protected_cascade_memory,
    protected_route_by_id,
)
from lm.kernels.mamba4_reference import dense_reference
from lm.models.mamba4 import Mamba4LM, order_memory


def arrays(time=11):
    rng = np.random.default_rng(23)
    k = rng.normal(size=(1, time, 2, 4)).astype(np.float32) * 0.4
    v = rng.normal(size=(1, time, 2, 3)).astype(np.float32)
    q = rng.normal(size=k.shape).astype(np.float32)
    b = rng.uniform(0.1, 1.5, size=k.shape[:-1]).astype(np.float32)
    return k, v, q, b


@pytest.mark.parametrize("floor", ["fixed", "cyclic"])
@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_fused_prefix_against_independent_dense(floor, chunk):
    k, v, q, b = arrays()
    decay = np.array([0.97, 0.99], np.float32)
    if floor == "fixed":
        decay = np.broadcast_to(decay, b.shape).copy()
        decay[:, 3:5] = 0.6
    eps = np.array([0.08, 0.13], np.float32)
    expected = dense_reference(k, v, q, b, decay, eps, floor)
    actual = jax.jit(
        lambda k, v, q, b: gaussian_memory(
            k, v, q, b, decay, eps, floor=floor, chunk_size=chunk
        )
    )(k, v, q, b)
    for x, y in zip(
        (
            actual.output,
            actual.variance,
            actual.state.precision,
            actual.state.cross,
            actual.state.prior,
        ),
        expected,
    ):
        np.testing.assert_allclose(x, y, rtol=4e-5, atol=5e-6)
    if floor == "fixed":
        np.testing.assert_allclose(
            actual.state.prior, eps[None, :, None, None] * np.eye(4), atol=2e-6
        )
    else:
        diag = np.diagonal(actual.state.prior, axis1=-2, axis2=-1)
        assert np.all(diag >= eps[None, :, None] - 1e-6)
        assert np.all(diag <= (eps * decay**-3)[None, :, None] + 1e-6)


@pytest.mark.parametrize("floor", ["fixed", "cyclic"])
def test_read_and_gate_gradients_against_dense(floor):
    k, v, q, b = map(jnp.asarray, arrays(5))
    decay = jnp.array([0.96, 0.98])
    if floor == "fixed":
        decay = jnp.broadcast_to(decay, b.shape)
    eps = jnp.array([0.1, 0.2])
    args = (k, v, q, b, decay, eps)

    def fused(*args):
        out = gaussian_memory(*args, floor=floor, chunk_size=3)
        return jnp.sum(jnp.sin(out.output)) + 0.01 * jnp.sum(out.variance)

    def dense(*args):
        y, variance, *_ = dense_reference(*args, floor)
        return jnp.sum(jnp.sin(y)) + 0.01 * jnp.sum(variance)

    actual = jax.grad(fused, argnums=tuple(range(6)))(*args)
    expected = jax.grad(dense, argnums=tuple(range(6)))(*args)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=8e-5)
        assert np.all(np.isfinite(x))
    assert np.linalg.norm(actual[-1]) > 1e-3


def test_variable_gate_requires_explicit_fixed_fallback():
    k, v, q, b = arrays(3)
    with pytest.raises(ValueError, match="fixed-floor fallback"):
        gaussian_memory(k, v, q, b, np.ones_like(b) * 0.99, 0.1, floor="cyclic")
    assert np.isfinite(
        gaussian_memory(k, v, q, b, np.ones_like(b) * 0.99, 0.1, floor="fixed").output
    ).all()


def test_chunk_continuation_and_quadratic_cyclic_decode():
    k, v, q, b = arrays(8)
    decay = jnp.array([0.97, 0.99])
    full = gaussian_memory(k, v, q, b, decay, 0.1, chunk_size=3)
    first = gaussian_memory(
        k[:, :3], v[:, :3], q[:, :3], b[:, :3], decay, 0.1, chunk_size=2
    )
    second = gaussian_memory(
        k[:, 3:],
        v[:, 3:],
        q[:, 3:],
        b[:, 3:],
        decay,
        0.1,
        chunk_size=2,
        initial=first.state,
    )
    np.testing.assert_allclose(second.output, full.output[:, 3:], rtol=3e-5, atol=2e-6)
    state = first.state
    lower = jnp.linalg.cholesky(state.precision)
    for t in range(3, 8):
        decoded, lower = cyclic_decode(
            state,
            lower,
            jnp.asarray(k[:, t]),
            jnp.asarray(v[:, t]),
            jnp.asarray(q[:, t]),
            b[:, t],
            decay,
            0.1,
        )
        state = decoded.state
        np.testing.assert_allclose(
            decoded.output, full.output[:, t], rtol=4e-5, atol=4e-6
        )
        np.testing.assert_allclose(
            lower @ jnp.swapaxes(lower, -1, -2), state.precision, rtol=2e-5, atol=3e-6
        )
    cpu = CyclicFloorMemory.create(4, 3, 0.1, float(decay[0]))
    for t in range(8):
        cpu.write(k[0, t, 0], v[0, t, 0], b[0, t, 0])
        np.testing.assert_allclose(
            cpu.read(q[0, t, 0]), full.output[0, t, 0][None], rtol=4e-5, atol=4e-6
        )


def test_protected_redundant_cascade_exact_ids_and_discarded_status():
    k, v, q, _ = arrays(24)
    # B=H=1 lets the existing CPU audit provide an independent bank ledger.
    k, v, q = k[:, :, :1], v[:, :, :1], q[:, :, :1]
    actual = jax.jit(
        lambda k, v, q: protected_cascade_memory(k, v, q, budget=2, block_size=3)
    )(k, v, q)
    cpu = ProtectedCascade(4, 3, anchor_budget=2)
    for start in range(0, 24, 3):
        cpu.append(
            k[0, start : start + 3, 0],
            v[0, start : start + 3, 0],
            np.arange(start, start + 3),
        )
    retained = sorted(int(i) for bank in cpu.blocks for i in bank.ids)
    got = []
    for level, count in enumerate(np.asarray(actual.state.counts)):
        assert count in (0, 1, 2)
        for slot in range(int(count)):
            ids = np.asarray(actual.state.banks.ids[level, slot, 0, 0])
            got.extend(int(i) for i in ids if i >= 0)
    assert sorted(got) == retained
    assert int(actual.state.merges) == cpu.merges
    for item in range(24):
        answer, found = protected_route_by_id(
            actual.state, jnp.asarray(k[:, item]), jnp.array([[item]])
        )
        assert bool(found[0, 0]) == (item in retained)
        if item in retained:
            np.testing.assert_allclose(
                answer[0, 0], v[0, item, 0], rtol=3e-5, atol=4e-6
            )
    # Background/evicted value changes cannot contaminate retained QR reads.
    changed = v.copy()
    changed[:, [i for i in range(24) if i not in retained]] = 1e5
    again = protected_cascade_memory(k, changed, q, budget=2, block_size=3)
    for item in retained:
        answer, _ = protected_route_by_id(
            again.state, jnp.asarray(k[:, item]), jnp.array([[item]])
        )
        np.testing.assert_allclose(answer[0, 0], v[0, item, 0], rtol=3e-5, atol=4e-6)


def test_order_auxiliary_distinguishes_permutations():
    v = jnp.array([1.0, 3.0]).reshape(1, 2, 1, 1)
    gates = jnp.ones((1, 2, 1)) * 0.5
    assert float(order_memory(v, gates)[0, -1, 0, 0]) != float(
        order_memory(v[:, ::-1], gates)[0, -1, 0, 0]
    )


@pytest.mark.parametrize("floor", ["cyclic", "fixed"])
def test_full_lm_causality_and_loss_gradients(floor):
    config = ModelConfig(
        architecture="mamba4",
        vocab_size=31,
        d_model=16,
        n_layers=1,
        d_ff=24,
        expand=1,
        head_dim=8,
        key_dim=4,
        chunk_size=3,
        protected_anchor_budget=2,
        protected_block_size=3,
        dtype="float32",
        remat=True,
        memory_floor=floor,
    )
    model = Mamba4LM(config)
    tokens = jnp.arange(9)[None] % config.vocab_size
    params = model.init(jax.random.key(42), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    changed = model.apply({"params": params}, tokens.at[:, 5:].set(12))
    assert logits.shape == (1, 9, 31)
    np.testing.assert_allclose(logits[:, :5], changed[:, :5], atol=3e-6, rtol=1e-5)

    def loss(p):
        y = model.apply({"params": p}, tokens)
        return -jnp.mean(
            jax.nn.log_softmax(y[:, :-1], axis=-1)[0, jnp.arange(8), tokens[0, 1:]]
        )

    grads = jax.grad(loss)(params)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    for field in (
        "epsilon",
        "decay",
        "protected_mix",
        "protected_route_strength",
        "order_mix",
    ):
        assert float(jnp.linalg.norm(grads["layer_0"]["memory"][field])) > 1e-9


@pytest.mark.parametrize("floor", ["cyclic", "fixed"])
def test_full_cached_decode_matches_prefill_with_learned_prior(floor):
    from lm.models.mamba4 import decode_step, initialize_cache

    config = ModelConfig(
        architecture="mamba4",
        vocab_size=23,
        d_model=8,
        n_layers=1,
        d_ff=12,
        expand=1,
        head_dim=4,
        key_dim=4,
        chunk_size=3,
        protected_anchor_budget=2,
        protected_block_size=3,
        dtype="float32",
        remat=True,
        memory_floor=floor,
        memory_hops=2,
        max_seq_len=9,
    )
    model = Mamba4LM(config)
    tokens = jnp.arange(9)[None] % config.vocab_size
    params = model.init(jax.random.key(31), tokens)["params"]
    # Cached allocation from config must not override learned checkpoint priors.
    params["layer_0"]["memory"]["epsilon"] += 0.3
    params["layer_0"]["memory"]["decay"] -= 0.2
    expected = model.apply({"params": params}, tokens)
    cache = initialize_cache(config, 1)
    step = jax.jit(lambda token, cache: decode_step(config, params, token, cache))
    outputs = []
    for position in range(9):
        output, cache = step(tokens[:, position], cache)
        outputs.append(output)
    np.testing.assert_allclose(
        jnp.stack(outputs, axis=1), expected, rtol=3e-4, atol=6e-6
    )


@pytest.mark.parametrize("residual", [0.0, 0.5e-5, 2e-5, 2e-3])
def test_protected_duplicate_cluster_and_rank_cut_gradients(residual):
    # Exact duplicates, rejected subthreshold keys, and accepted clustered keys.
    keys = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (1, 12, 1, 4))
    keys = keys.at[:, 1::3, :, 1].set(residual)
    keys = keys.at[:, 2::3, :, 2].set(residual)
    values = jax.random.normal(jax.random.key(11), (1, 12, 1, 3))
    queries = jnp.broadcast_to(jnp.array([1.0, 0.2, -0.1, 0.3]), keys.shape)

    def loss(k, v, q):
        return jnp.sum(
            jnp.tanh(protected_cascade_memory(k, v, q, budget=3, block_size=3).output)
        )

    output = loss(keys, values, queries)
    grads = jax.grad(loss, argnums=(0, 1, 2))(keys, values, queries)
    assert np.isfinite(output)
    assert all(np.isfinite(g).all() for g in grads)
    bank_result = protected_cascade_memory(
        keys, values, queries, budget=3, block_size=3
    )
    eligible = np.asarray(bank_result.state.banks.valid)
    triangular = np.asarray(bank_result.state.banks.triangular)
    diagonals = np.diagonal(triangular, axis1=-2, axis2=-1)
    assert np.all(diagonals[eligible] > 1e-5)


@pytest.mark.parametrize("floor", ["cyclic", "fixed"])
def test_repeated_eos_full_lm_gradients_are_finite(floor):
    config = ModelConfig(
        architecture="mamba4",
        vocab_size=23,
        d_model=8,
        n_layers=2,
        d_ff=12,
        expand=1,
        head_dim=4,
        key_dim=4,
        chunk_size=3,
        protected_anchor_budget=3,
        protected_block_size=3,
        dtype="float32",
        remat=True,
        memory_floor=floor,
        max_seq_len=12,
    )
    model = Mamba4LM(config)
    tokens = jnp.full((1, 12), 2, dtype=jnp.int32)
    params = model.init(jax.random.key(66), tokens)["params"]

    def loss(p):
        logits = model.apply({"params": p}, tokens)
        return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[..., 2])

    value, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(value)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))


def test_opt_in_full_lm_diagnostics_are_scalar_and_loss_unchanged():
    config = ModelConfig(
        architecture="mamba4",
        vocab_size=23,
        d_model=8,
        n_layers=1,
        d_ff=12,
        expand=1,
        head_dim=4,
        key_dim=4,
        chunk_size=3,
        protected_anchor_budget=2,
        protected_block_size=3,
        dtype="float32",
        max_seq_len=9,
    )
    model = Mamba4LM(config)
    tokens = jnp.full((2, 9), 2, dtype=jnp.int32)
    variables = model.init(jax.random.key(17), tokens)
    assert "diagnostics" not in variables
    ordinary = model.apply(variables, tokens)
    inspected, collections = model.apply(variables, tokens, mutable=["diagnostics"])
    np.testing.assert_array_equal(ordinary, inspected)
    metrics = collections["diagnostics"]["layer_0"]["memory"]
    assert float(metrics["precision_eigenvalue_min"][0]) > 0
    assert float(metrics["precision_condition_max"][0]) >= 1
    assert bool(metrics["gaussian_allfinite"][0])
    assert bool(metrics["protected_allfinite"][0])
    assert int(metrics["protected_retained_anchors_min"][0]) >= 1
    assert all(np.asarray(value[0]).ndim == 0 for value in metrics.values())
