"""Claims-suite task construction, scoring probes and statistics on CPU."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lm.config import ModelConfig
from validation import claims


TOKENIZER = "data/fineweb-edu-1b/tokenizer.json"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(TOKENIZER)


def check_teacher_forcing(rows):
    for row in rows:
        assert np.array_equal(row["ids"][row["positions"] + 1], row["targets"])
        assert row["positions"].max() < len(row["ids"]) - 1


def test_copy_and_associative_rows_score_the_right_tokens(tokenizer, monkeypatch):
    monkeypatch.setitem(claims.ROWS, "copy", 3)
    monkeypatch.setitem(claims.ROWS, "mqar", 3)
    words, keys, values = claims.word_pools(tokenizer)
    assert not set(keys.tolist()) & set(values.tolist())
    rng = np.random.default_rng(0)
    copies = claims.copy_rows(words, rng)
    check_teacher_forcing(copies)
    for row in copies:
        length = row["group"]
        assert np.array_equal(row["ids"][:length], row["ids"][length:])
        assert len(set(row["ids"][:length].tolist())) == length
    stores = claims.mqar_rows(keys, values, rng)
    check_teacher_forcing(stores)
    for row in stores:
        count = row["group"]
        lookup = dict(zip(row["ids"][0 : 4 * count : 4], row["ids"][2 : 4 * count : 4]))
        queried = row["ids"][row["positions"] - 1]
        assert [lookup[key] for key in queried] == row["targets"].tolist()
        assert set(row["targets"].tolist()) <= set(row["candidates"].tolist())
        assert len(row["ids"]) == 4 * count + 4 * claims.MQAR_QUERIES


def test_passkey_rows_fit_and_ask_for_the_hidden_key(tokenizer, monkeypatch):
    monkeypatch.setitem(claims.ROWS, "passkey", 2)
    monkeypatch.setattr(claims, "PASSKEY_LENGTHS", (512, 2048))
    rows = claims.passkey_rows(tokenizer, np.random.default_rng(1))
    check_teacher_forcing(rows)
    for row in rows:
        length = row["group"] // 10
        assert length - 30 <= len(row["ids"]) <= length
        text = tokenizer.decode(row["ids"].tolist())
        key = tokenizer.decode(row["targets"].tolist()).strip()
        assert f"The pass key is {key}." in text
        assert text.endswith(f"The pass key is {key}")
        assert np.all(row["distance"] > 0)


def test_pack_pads_without_changing_scored_slots():
    rows = [
        dict(
            family="copy",
            group=2,
            ids=np.array([5, 6, 5, 6]),
            positions=np.array([2]),
            targets=np.array([6]),
            candidates=None,
            distance=np.array([2]),
        ),
        dict(
            family="mqar",
            group=1,
            ids=np.arange(3000),
            positions=np.array([10, 20]),
            targets=np.array([11, 21]),
            candidates=np.array([11, 21, 7]),
            distance=np.array([3, 4]),
        ),
    ]
    arrays, metadata = claims.pack(rows)
    assert metadata == {
        "1024": {"rows": 1, "score_width": 1},
        "4096": {"rows": 1, "score_width": 2},
    }
    assert arrays["b1024_input_ids"][0, 4:].tolist() == [claims.EOS] * 1020
    assert arrays["b4096_candidates"][0].tolist() == [11, 21, 7]
    assert arrays["b4096_family"][0] == 1


def tiny(architecture, **extra):
    base = dict(
        architecture=architecture,
        vocab_size=97,
        d_model=64,
        n_layers=3,
        n_heads=2,
        d_ff=96,
        d_state=16,
        expand=2,
        head_dim=32,
        chunk_size=16,
        dtype="float32",
        remat=False,
        max_seq_len=64,
    )
    base.update(extra)
    return ModelConfig(**base)


def selective():
    return tiny(
        "mamba4",
        d_ff=0,
        key_dim=8,
        memory_head_dim=32,
        memory_mixer="selective",
        layer_pattern="MSM",
        conv_kernel=4,
        memory_floor="fixed",
        protected_anchor_budget=0,
        memory_solver="loop",
        remat=True,
    )


def model_and_params(config):
    from lm.train import create_model

    model = create_model(config)
    ids = jnp.zeros((1, 32), jnp.int32)
    return model, model.init(jax.random.PRNGKey(0), ids)["params"]


def scored_inputs(rng, batch=2, length=48, width=5, options=4):
    ids = rng.integers(0, 97, size=(1, batch, length)).astype(np.int32)
    positions = np.sort(rng.choice(length - 1, size=(1, batch, width)), axis=-1)
    positions[0, 1, -1] = -1
    targets = rng.integers(0, 97, size=(1, batch, width)).astype(np.int32)
    candidates = rng.integers(0, 97, size=(1, batch, options)).astype(np.int32)
    candidates[..., 0] = targets[..., 0]
    return ids, positions.astype(np.int32), targets, candidates


def test_scored_function_matches_a_direct_log_softmax():
    model, params = model_and_params(tiny("transformer"))
    ids, positions, targets, candidates = scored_inputs(np.random.default_rng(2))
    replicated = jax.device_put_replicated(params, jax.local_devices())
    output = claims.scored_function(model, False)(
        replicated, ids, positions, targets, candidates
    )
    logits = model.apply({"params": params}, ids[0])
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    for row in range(2):
        for slot, position in enumerate(positions[0, row]):
            position = max(int(position), 0)
            expected = logp[row, position, targets[0, row, slot]]
            got = output["target_logp"][0, row, slot]
            assert got == pytest.approx(float(expected), abs=1e-5)
            assert bool(output["correct"][0, row, slot]) == bool(
                jnp.argmax(logp[row, position]) == targets[0, row, slot]
            )
            options = logp[row, position, candidates[0, row]]
            assert bool(output["candidate_correct"][0, row, slot]) == bool(
                expected >= jnp.max(options)
            )


def test_variance_probe_keeps_logits_and_orders_memory_layers():
    config = selective()
    model, params = model_and_params(config)
    ids, positions, targets, candidates = scored_inputs(np.random.default_rng(3))
    replicated = jax.device_put_replicated(params, jax.local_devices())
    plain = claims.scored_function(model, False)(
        replicated, ids, positions, targets, candidates
    )
    with claims.mamba4_probe("variance"):
        probed = claims.scored_function(model, True)(
            replicated, ids, positions, targets, candidates
        )
    np.testing.assert_allclose(plain["target_logp"], probed["target_logp"], atol=1e-6)
    heads = config.memory_heads
    assert probed["variance"].shape == (1, 2, 2, positions.shape[-1], heads)
    assert np.all(probed["variance"] >= 0)
    bound = 1 / config.floor_min
    assert np.all(probed["variance"] <= bound + 1e-5)
    with claims.mamba4_probe("no_read"):
        ablated = claims.scored_function(model, False)(
            replicated, ids, positions, targets, candidates
        )
    assert not np.allclose(plain["target_logp"], ablated["target_logp"])
    import lm.models.mamba4 as module
    from lm.kernels.mamba4 import selective_gaussian_memory

    assert module.selective_gaussian_memory is selective_gaussian_memory


def test_long_function_accounts_every_position(monkeypatch):
    monkeypatch.setattr(claims, "POSITION_EDGES", (0, 8, 24, 48))
    monkeypatch.setattr(claims, "TRAINING_CONTEXT", 24)
    model, params = model_and_params(tiny("mamba3", d_ff=0))
    tokens = np.random.default_rng(4).integers(0, 97, size=(1, 2, 49)).astype(np.int32)
    replicated = jax.device_put_replicated(params, jax.local_devices())
    output = claims.long_function(model)(replicated, tokens[..., :-1], tokens[..., 1:])
    assert output["bucket_count"][0, 0].tolist() == [8, 16, 24]
    nll = -np.asarray(output["target_logp"][0])
    assert output["bucket_nll"][0, :, 0] == pytest.approx(nll[:, :8].sum(axis=1))
    assert output["hist_count"][0].sum(axis=-1).tolist() == [48, 48]
    assert output["hist_count"][0, :, : claims.CALIBRATION_BINS].sum() == 48


def test_run_rows_returns_global_order_and_drops_padding():
    arrays = {"input_ids": np.arange(10)[:, None].repeat(3, axis=1)}

    def mapped(_, ids):
        return {"first": ids[..., 0] * 2}

    timings = []
    output = claims.run_rows(mapped, None, arrays, 3, timings)
    assert output["first"].tolist() == [2 * i for i in range(10)]
    assert len(timings) == 4


def test_statistics_helpers():
    assert claims.holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    first = np.array([True] * 8 + [False] * 2)
    second = np.array([False] * 8 + [True] * 2)
    result = claims.mcnemar(first, second)
    assert (result["only_first"], result["only_second"]) == (8, 2)
    assert result["p"] == pytest.approx(0.109375)
    assert claims.auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    rng = np.random.default_rng(5)
    confidence = rng.uniform(size=400)
    hits = rng.uniform(size=400) < confidence
    rows = [
        {"confidence": confidence[i : i + 40], "hits": hits[i : i + 40]}
        for i in range(0, 400, 40)
    ]
    bins = np.stack([claims._row_bins(row) for row in rows]).sum(axis=0)
    assert claims._ece_from_bins(bins) == pytest.approx(
        claims.expected_calibration_error(confidence, hits)
    )
    test = claims.paired_test(np.ones(50) * 0.6, np.ones(50) * 0.5, rng)
    assert test["mean"] == pytest.approx(0.1) and test["p"] == 0.0


def test_decode_contexts_cover_long_range():
    assert claims.DECODE_CONTEXTS[-1] >= 16 * claims.TRAINING_CONTEXT
    assert dataclasses.is_dataclass(tiny("transformer"))
