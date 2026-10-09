"""Packing and split tests for exact, matched language-model target budgets."""

import hashlib
import json

import numpy as np
import pytest

from lm.data import TokenCorpus, sha256_file
from scripts.prepare_fineweb import LEDGER_DTYPE, split_documents, write_stream


def make_corpus(directory, train, evaluation):
    splits = {}
    for name, ids in (("train", train), ("eval", evaluation)):
        path = directory / f"{name}.bin"
        np.asarray(ids, dtype="<u2").tofile(path)
        splits[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "token_count": len(ids),
            "target_count": len(ids) - 1,
        }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mamba4.tokens.v1",
                "dtype": "<u2",
                "tokenizer": {"eos_id": 50256, "vocab_size": 50257},
                "splits": splits,
            }
        )
    )
    return TokenCorpus(directory, verify_hashes=True)


def test_every_target_is_consumed_once_and_padding_is_masked(tmp_path):
    corpus = make_corpus(tmp_path, list(range(18)), [20, 21, 22, 23])
    batches = list(corpus.iter_batches(2, 4))
    assert len(batches) == 3
    assert sum(batch["loss_mask"].sum() for batch in batches) == 17
    observed = np.concatenate(
        [batch["targets"][batch["loss_mask"].astype(bool)] for batch in batches]
    )
    np.testing.assert_array_equal(observed, np.arange(1, 18))
    for batch in batches:
        mask = batch["loss_mask"].astype(bool)
        np.testing.assert_array_equal(
            batch["targets"][mask], batch["input_ids"][mask] + 1
        )
    assert np.all(batches[-1]["targets"].reshape(-1)[1:] == 50256)
    assert batches[0]["input_ids"].shape == (2, 4)


def test_resume_and_smaller_budget_preserve_the_identical_stream(tmp_path):
    corpus = make_corpus(tmp_path, list(range(30)), [40, 41, 42])
    all_batches = list(corpus.iter_batches(2, 4, target_budget=19))
    resumed = list(corpus.iter_batches(2, 4, target_budget=19, start_step=1))
    for a, b in zip(all_batches[1:], resumed, strict=True):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert sum(batch["loss_mask"].sum() for batch in all_batches) == 19
    with pytest.raises(ValueError, match="contains"):
        corpus.batch(0, 2, 4, target_budget=30)
    with pytest.raises(IndexError, match="beyond"):
        corpus.batch(3, 2, 4, target_budget=19)
    with pytest.raises(ValueError):
        corpus.batch(-1, 2, 4)


def test_validation_rejects_truncated_or_altered_tokens(tmp_path):
    make_corpus(tmp_path, [1, 2, 3], [4, 5, 6])
    np.asarray([1, 9, 3], dtype="<u2").tofile(tmp_path / "train.bin")
    with pytest.raises(ValueError, match="hash"):
        TokenCorpus(tmp_path, verify_hashes=True)
    (tmp_path / "train.bin").write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="size"):
        TokenCorpus(tmp_path)


def test_eval_is_a_separate_bounded_stream(tmp_path):
    corpus = make_corpus(tmp_path, [1, 2, 3], [8, 9, 10, 11, 12])
    batch = corpus.batch(0, 2, 3, split="eval")
    np.testing.assert_array_equal(batch["targets"].reshape(-1)[:4], [9, 10, 11, 12])
    assert batch["loss_mask"].sum() == 4


def make_ledger(texts, lengths=None):
    ledger = np.zeros(len(texts), dtype=LEDGER_DTYPE)
    ledger["length"] = 10 if lengths is None else lengths
    ledger["row"] = np.arange(len(texts))
    for i, text in enumerate(texts):
        ledger["sha256"][i] = np.frombuffer(
            hashlib.sha256(text.encode()).digest(), dtype=np.uint8
        )
    return ledger


def test_hash_group_split_excludes_all_heldout_duplicates():
    texts = ["alpha", "alpha", "beta", "beta", "gamma", "delta", "epsilon"]
    ledger = make_ledger(texts)
    train, evaluation, summary = split_documents(ledger, seed=42, eval_targets=15)
    train_hashes = {ledger["sha256"][i].tobytes() for i in train}
    eval_hashes = {ledger["sha256"][i].tobytes() for i in evaluation}
    assert not train_hashes.intersection(eval_hashes)
    assert len(eval_hashes) == len(evaluation)
    assert summary["heldout_tokens_available"] >= 16
    for text in set(texts):
        group = [i for i, current in enumerate(texts) if current == text]
        assert len(set(ledger["split"][group])) == 1
    train2, eval2, _ = split_documents(make_ledger(texts), 42, 15)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(evaluation, eval2)


def test_stream_retains_eos_and_records_final_document_truncation(tmp_path):
    source = np.asarray([7, 8, 50256, 9, 10, 50256], dtype="<u2")
    ledger = make_ledger(["first", "second"], [3, 3])
    ledger["offset"] = [0, 3]
    summary = write_stream(tmp_path, "train", np.asarray([1, 0]), ledger, [source], 4)
    stream = np.fromfile(tmp_path / "train.bin", dtype="<u2")
    np.testing.assert_array_equal(stream, [9, 10, 50256, 7, 8])
    assert summary["target_count"] == 4
    assert summary["partial_final_document"]["tokens_used"] == 2
    assert not summary["source_replay"]
    with pytest.raises(ValueError, match="Refusing silent"):
        write_stream(tmp_path, "short", np.asarray([0]), ledger, [source], 5)


def test_billion_target_arithmetic():
    batch_targets = 128 * 1024
    steps = (1_000_000_000 + batch_targets - 1) // batch_targets
    assert steps == 7630
    remainder = 1_000_000_000 - (steps - 1) * batch_targets
    assert remainder == 51712
    assert (steps - 1) * batch_targets + remainder == 1_000_000_000


def test_explicit_replay_uses_training_only_and_records_its_budget(tmp_path):
    source = np.asarray([7, 8, 50256, 9, 10, 50256, 42, 50256], dtype="<u2")
    ledger = make_ledger(["train-a", "train-b", "heldout"], [3, 3, 2])
    ledger["offset"] = [0, 3, 6]
    ledger["split"] = [0, 0, 1]
    summary = write_stream(
        tmp_path,
        "train",
        np.asarray([0, 1]),
        ledger,
        [source],
        10,
        allow_replay=True,
        replay_seed=43,
    )
    stream = np.fromfile(tmp_path / "train.bin", dtype="<u2")
    assert len(stream) == 11
    assert 42 not in stream
    assert summary["available_source_tokens"] == 6
    assert summary["replayed_stream_tokens"] == 5
    assert summary["source_replay"]
    order = np.load(tmp_path / "train-document-order.npy")
    assert set(order) == {0, 1}
    assert len(order) > 2
    np.testing.assert_array_equal(stream[:6], source[:6])
    with pytest.raises(ValueError, match="never"):
        write_stream(
            tmp_path, "eval", np.asarray([2]), ledger, [source], 10, allow_replay=True
        )
