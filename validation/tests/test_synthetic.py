"""Synthetic recall tasks: answers, masks and state accounting on CPU."""

import numpy as np
import pytest

from validation import synthetic


@pytest.mark.parametrize("unknown", [False, True])
def test_mqar_answers_are_stored_values_or_none(unknown):
    rng = np.random.default_rng(0)
    for count in (8, 32):
        inputs, targets, mask, kind = synthetic.mqar_sequence(rng, count, 256, unknown)
        store = dict(zip(inputs[0 : 2 * count : 2], inputs[1 : 2 * count : 2]))
        slots = np.flatnonzero(mask)
        assert len(slots) == count
        for slot in slots:
            key, answer = inputs[slot], targets[slot]
            if key in store:
                assert answer == store[key] and kind[slot] == 0
            else:
                assert unknown and answer == synthetic.NONE and kind[slot] == 1
        assert np.all(inputs[4 * count - 1 :] == synthetic.PAD)


def test_noisy_queries_recall_scattered_pairs():
    rng = np.random.default_rng(1)
    for length in (512, 2048):
        inputs, targets, mask, _ = synthetic.noisy_sequence(rng, length)
        slots = np.flatnonzero(mask)
        assert len(slots) == synthetic.NOISY_PAIRS and slots[-1] == length - 1
        low, high = synthetic.KEYS
        stored = {
            inputs[p]: targets[p] for p in range(slots[0]) if low <= inputs[p] < high
        }
        assert len(stored) == synthetic.NOISY_PAIRS
        assert all(stored[inputs[p]] == targets[p] for p in slots)


def test_batches_are_reproducible_for_a_seed():
    first = synthetic.batch("mqar", np.random.default_rng([1, 0]), 4)
    second = synthetic.batch("mqar", np.random.default_rng([1, 0]), 4)
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_state_accounting():
    transformer = synthetic.SPECS["transformer"]
    assert transformer.state_floats(512) == 2 * 2 * 512 * 128
    mamba3 = synthetic.SPECS["mamba3"]
    assert mamba3.state_floats(512) == 2 * 4 * 48 * 64
    mamba4 = synthetic.SPECS["mamba4"]
    assert mamba4.state_floats(512) == 2 * 4 * (32 * 33 // 2 + 32 * 64)


def test_hops_answers_follow_the_stored_permutation():
    rng = np.random.default_rng(2)
    inputs, targets, mask, kind = synthetic.hops_sequence(rng, 16, 256)
    successor = dict(zip(inputs[0:32:2].tolist(), inputs[1:32:2].tolist()))
    assert sorted(successor) == sorted(successor.values())
    slots = np.flatnonzero(mask)
    assert len(slots) == 16 and set(kind[slots].tolist()) <= {0, 1}
    for slot in slots:
        node, hop = int(inputs[slot]), int(inputs[slot - 1])
        expected = (
            successor[node] if hop == synthetic.HOP1 else successor[successor[node]]
        )
        assert hop in (synthetic.HOP1, synthetic.HOP2)
        assert targets[slot] == expected
        assert kind[slot] == (0 if hop == synthetic.HOP1 else 1)
