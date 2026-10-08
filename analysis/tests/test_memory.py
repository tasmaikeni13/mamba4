import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.gradients import terminal_stream_vjp
from analysis.memory import (
    AffineSummary,
    CyclicFloorMemory,
    EvidenceState,
    ProtectedCascade,
    effective_weights,
    hebbian_read,
    interpolation,
    mamba3_core,
    mamba3_read,
    rotate_pairs,
    softmax_read,
    tree_prefix,
    unit_rows,
)


def test_discounted_scan_every_prefix_and_merge():
    rng = np.random.default_rng(18)
    keys, values = unit_rows(rng.normal(size=(37, 8))), rng.normal(size=(37, 3))
    beta, decay = rng.uniform(0, 2, 37), rng.uniform(0, 1, 37)
    decay[9] = 0
    elements = [
        AffineSummary(lam, b * np.outer(k, k), b * np.outer(v, k))
        for k, v, b, lam in zip(keys, values, beta, decay)
    ]
    state = EvidenceState.zeros(8, 3)
    for i, prefix in enumerate(tree_prefix(elements)):
        state.write(keys[i], values[i], beta[i], decay[i])
        assert_allclose(prefix.s, state.s, atol=1e-14)
        assert_allclose(prefix.c, state.c, atol=1e-14)
        weights = effective_weights(beta[: i + 1], decay[: i + 1])
        batch = EvidenceState.from_batch(keys[: i + 1], values[: i + 1], weights)
        assert_allclose(batch.s, state.s, atol=1e-14)
    left = EvidenceState.from_batch(keys[:17], values[:17])
    right = EvidenceState.from_batch(keys[17:], values[17:])
    full = EvidenceState.from_batch(keys, values)
    assert_allclose(left.merge(right).s, full.s)
    assert_allclose(left.merge(right).c, full.c)


def test_qr_interpolates_independent_keys_and_signed_queries():
    rng = np.random.default_rng(7)
    keys, values = unit_rows(rng.normal(size=(6, 8))), rng.normal(size=(6, 3))
    coeff = rng.normal(size=(5, 6))
    assert_allclose(interpolation(keys, values, keys), values, atol=1e-12)
    assert_allclose(
        interpolation(keys, values, coeff @ keys), coeff @ values, atol=1e-12
    )
    with pytest.raises(ValueError):
        interpolation(np.repeat(keys[:1], 2, axis=0), values[:2], keys)


def test_precision_smoothing_can_average_heteroscedastic_writes():
    keys = np.array([[1.0, 0.0], [1.0, 0.0]])
    values = np.array([[1.0], [3.0]])
    result = softmax_read(keys, values, keys[:1], 16, np.array([1.0, 3.0]))
    assert_allclose(result, [[2.5]])


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_mamba3_matches_independent_rotary_frame_reference(rank):
    """The independent form follows the pinned upstream step kernel's frame.

    Compare nonzero phases, variable dt/decay, previous-input term and MIMO.
    This catches the opposite-rotation-sign error that zero-phase tests miss.
    """
    rng = np.random.default_rng(23 + rank)
    n, dim, dv = 11, 8, 3
    b, x = rng.normal(size=(n, dim, rank)), rng.normal(size=(n, dv, rank))
    q = rng.normal(size=(4, dim, rank))
    dt, a, trap = (
        rng.uniform(0.1, 1, n),
        -rng.uniform(0.01, 0.3, n),
        rng.uniform(0.2, 0.8, n),
    )
    angles = rng.normal(size=(n, dim // 2))
    actual, _ = mamba3_core(b, x, q, dt, a, angles, trap)
    cumulative = np.zeros(dim // 2)
    h = np.zeros((dim, dv))
    previous = np.zeros_like(h)
    for i in range(n):
        cumulative += angles[i]
        rotated_b = rotate_pairs(b[i], cumulative)
        current = rotated_b @ x[i].T
        alpha = np.exp(dt[i] * a[i])
        h = (
            alpha * h
            + (1 - trap[i]) * dt[i] * alpha * previous
            + trap[i] * dt[i] * current
        )
        previous = current
    expected = np.einsum(
        "qdr,dv->qvr", np.stack([rotate_pairs(qi, cumulative) for qi in q]), h
    )
    assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2])
def test_mamba3_euler_limit_is_hebbian(rank):
    rng = np.random.default_rng(24)
    k, v, q = rng.normal(size=(9, 8)), rng.normal(size=(9, 3)), rng.normal(size=(4, 8))
    assert_allclose(mamba3_read(k, v, q, rank=rank), hebbian_read(k, v, q), atol=1e-12)


@pytest.mark.parametrize("decay", [1.0, 0.999, 0.95, 0.8])
def test_cyclic_floor_factor_matches_dense_and_never_loses_floor(decay):
    rng = np.random.default_rng(2)
    memory = CyclicFloorMemory.create(8, 3, 0.01, decay)
    for i in range(97):
        k, v = unit_rows(rng.normal(size=(1, 8)))[0], rng.normal(size=3)
        memory.write(k, v, beta=0.0 if i % 7 == 0 else 1.0)
        a = memory.state.s + np.diag(memory.prior)
        assert memory.prior.min() >= 0.01 - 1e-14
        assert_allclose(memory.lower @ memory.lower.T, a, atol=1e-12)
        q = rng.normal(size=(2, 8))
        assert_allclose(
            memory.read(q), (memory.state.c @ np.linalg.solve(a, q.T)).T, atol=1e-10
        )


def test_protected_cascade_capacity_mass_and_background():
    rng = np.random.default_rng(3)
    cascade = ProtectedCascade(8, 3)
    all_keys, all_values = [], []
    for block_id in range(100):
        k, v = unit_rows(rng.normal(size=(8, 8))), rng.normal(size=(8, 3))
        all_keys.append(k)
        all_values.append(v)
        cascade.append(k, v, np.arange(block_id * 8, (block_id + 1) * 8))
        assert all(1 <= len(level) <= 2 for level in cascade.levels)
        assert sum(b.span for b in cascade.blocks) == (block_id + 1) * 8
        assert cascade.merges < block_id + 1
        assert (
            (2 ** len(cascade.levels) - 1)
            <= block_id + 1
            <= 2 * (2 ** len(cascade.levels) - 1)
        )
        for block in cascade.blocks:
            assert len(block.keys) == 8
            assert_allclose(block.read(block.keys), block.values, atol=2e-10)
    evidence = EvidenceState.from_batch(
        np.concatenate(all_keys), np.concatenate(all_values)
    )
    assert_allclose(sum(b.background.s for b in cascade.blocks), evidence.s, atol=1e-11)
    assert_allclose(sum(b.background.c for b in cascade.blocks), evidence.c, atol=1e-11)


def test_every_stream_gradient_against_finite_differences():
    rng = np.random.default_rng(21)
    args = {
        "keys": rng.normal(size=(5, 4)),
        "values": rng.normal(size=(5, 2)),
        "beta": rng.uniform(0.2, 2, 5),
        "decay": rng.uniform(0.5, 0.9, 5),
        "query": rng.normal(size=4),
    }
    upstream, ridge = rng.normal(size=2), 0.1
    gradient = terminal_stream_vjp(**args, upstream=upstream, ridge=ridge)

    def loss(inputs):
        state = EvidenceState.from_batch(
            inputs["keys"],
            inputs["values"],
            effective_weights(inputs["beta"], inputs["decay"]),
        )
        return float(upstream @ state.read(inputs["query"], ridge)[0])

    for name, arr in args.items():
        numerical = np.zeros_like(arr)
        for index in np.ndindex(arr.shape):
            plus = {k: v.copy() for k, v in args.items()}
            minus = {k: v.copy() for k, v in args.items()}
            plus[name][index] += 1e-6
            minus[name][index] -= 1e-6
            numerical[index] = (loss(plus) - loss(minus)) / 2e-6
        assert_allclose(gradient[name], numerical, atol=2e-8, rtol=1e-5)
