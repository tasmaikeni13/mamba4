"""Adversarial artifact checks for the final three-run completion audit."""

import copy
import hashlib
import json
import math
from types import SimpleNamespace

from flax import serialization
import numpy as np
import pytest

from lm.recall import RecallDataset, summarize_recall
from scripts.verify_lm_runs import (
    audit_run,
    fingerprint,
    validate_data,
    validate_hardware,
    validate_training_steps,
)


def hardware(process):
    return {
        "process_index": process,
        "hostname": f"physical-worker-{process}",
        "backend": "tpu",
        "device_count": 16,
        "process_count": 4,
        "local_device_count": 4,
        "devices": [
            {
                "id": i,
                "process_index": i // 4,
                "kind": "TPU v4",
                "coords": [i % 2, i // 2 % 2, i // 4],
            }
            for i in range(16)
        ],
    }


@pytest.fixture
def run_fixture(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    corpus_path = tmp_path / "data"
    corpus_path.mkdir()
    (corpus_path / "manifest.json").write_text('{"dataset": "test fixture"}\n')
    training = {
        "seed": 42,
        "token_budget": 35,
        "global_batch": 8,
        "sequence_length": 2,
        "eval_tokens": 7,
    }
    frozen = {"model": {"architecture": "transformer"}, "training": training}
    sources = {"lm/train.py": "a" * 64}
    identity = {
        "protocol": frozen,
        "sources": sources,
        "data_manifest_sha256": hashlib.sha256(
            (corpus_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    ledger = {
        "total": 8,
        "embedding": 4,
        "non_embedding": 4,
        "parameters": [{"name": "weight", "shape": [2, 4], "count": 8}],
    }
    manifest = {
        **identity,
        "fingerprint": fingerprint(identity),
        "parameter_ledger": ledger,
    }
    gold = np.asarray([0, 1, 2, 0], np.int32)
    recall_dataset = RecallDataset(
        {"gold_index": gold, "candidate_token_ids": np.tile(np.arange(3), (4, 1))},
        tuple({"group": "fixture", "text_sha256": str(i)} for i in range(4)),
        {
            "protocol": "trained-textual-recall-v1",
            "seed": 20261008,
            "array_sha256": "b" * 64,
            "prompt_count": 4,
        },
    )
    raw_recall = summarize_recall(recall_dataset, np.full((4, 3), -math.log(10)))
    recall_summary = {key: raw_recall[key] for key in ("protocol", "overall", "groups")}
    result = {
        "status": "completed",
        "architecture": "transformer",
        "seed": 42,
        "training_targets": 35,
        "optimizer_steps": 3,
        "parameter_count": 8,
        "fingerprint": manifest["fingerprint"],
        "heldout": {"targets": 7, "nll": 2.0, "perplexity": math.exp(2)},
        "recall": recall_summary,
        "observed_device_count": 16,
        "steady_tokens_per_second": 160,
    }
    protocol = {
        "seed": 42,
        "training_targets_per_architecture": 35,
        "heldout_targets": 7,
        "recall_protocol": "trained-textual-recall-v1",
        "recall_seed": 20261008,
        "recall_array_sha256": "b" * 64,
        "recall_prompt_count": 4,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "recall.json").write_text(json.dumps(raw_recall))
    for process in range(4):
        (directory / f"result-host-{process}.json").write_text(json.dumps(result))
        (directory / f"hardware-{process}.json").write_text(
            json.dumps(hardware(process))
        )
    metrics = [{"event": "initial_eval", "targets": 7, "nll": 3.0}]
    metrics.extend(
        {
            "event": "train",
            "step": step,
            "training_targets": min(16 * step, 35),
            "loss": 2.0,
            "grad_norm": 1.0,
            "step_seconds": 0.1,
        }
        for step in range(1, 4)
    )
    (directory / "metrics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in metrics) + "\n"
    )
    checkpoint = directory / "checkpoints/step-00000003"
    checkpoint.mkdir(parents=True)
    payload = serialization.msgpack_serialize(
        {
            "step": np.asarray(3, np.int32),
            "params": {"weight": np.arange(8, dtype=np.float32).reshape(2, 4)},
            "opt_state": {},
        }
    )
    (checkpoint / "state.msgpack").write_bytes(payload)
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "step": 3,
                "training_targets": 35,
                "fingerprint": manifest["fingerprint"],
                "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    (directory / "latest.json").write_text(
        json.dumps({"checkpoint": "checkpoints/step-00000003"})
    )

    def run_audit():
        return audit_run(
            directory,
            "transformer",
            protocol,
            SimpleNamespace(path=corpus_path),
            frozen=frozen,
            expected_sources=sources,
            parameter_target=8,
        )

    return directory, run_audit, metrics, training, result


def test_valid_small_artifacts_cover_actual_parameter_state(run_fixture):
    _, run_audit, _, _, _ = run_fixture
    result = run_audit()
    assert result["training_targets"] == 35
    assert result["accounting"]["final_step_targets"] == 3
    assert result["all_four_workers_completed"]


def test_reject_missing_final_step_despite_completed_result(run_fixture):
    directory, run_audit, metrics, _, _ = run_fixture
    (directory / "metrics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in metrics[:-1]) + "\n"
    )
    with pytest.raises(ValueError, match="final step"):
        run_audit()


def test_reject_missing_recall_despite_complete_budget(run_fixture):
    directory, run_audit, _, _, _ = run_fixture
    for path in [directory / "result.json", *directory.glob("result-host-*.json")]:
        result = json.loads(path.read_text())
        result["recall"] = None
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Missing frozen"):
        run_audit()


def test_reject_wrong_hardware_or_missing_process_index(run_fixture):
    directory, run_audit, _, _, _ = run_fixture
    wrong = hardware(3)
    wrong["backend"] = "cpu"
    (directory / "hardware-3.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="TPU"):
        run_audit()
    records = [hardware(i) for i in range(4)]
    records[3]["process_index"] = 2
    with pytest.raises(ValueError, match="indices"):
        validate_hardware(records)


def test_reject_stale_sources_and_unreproducible_fingerprint(run_fixture):
    directory, run_audit, _, _, _ = run_fixture
    path = directory / "manifest.json"
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["sources"]["lm/train.py"] = "c" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Source hashes"):
        run_audit()
    changed = copy.deepcopy(original)
    changed["fingerprint"] = "c" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        run_audit()


def test_reject_checkpoint_step_or_parameter_shape_mismatch(run_fixture):
    directory, run_audit, _, _, _ = run_fixture
    checkpoint = directory / "checkpoints/step-00000003"
    payload_path = checkpoint / "state.msgpack"
    original = serialization.msgpack_restore(payload_path.read_bytes())
    for mutation, expected_error in (
        ("step", "optimizer-state step"),
        ("shape", "shape/count"),
    ):
        state = copy.deepcopy(original)
        if mutation == "step":
            state["step"] = np.asarray(2, np.int32)
        else:
            state["params"]["weight"] = np.arange(8, dtype=np.float32)
        payload = serialization.msgpack_serialize(state)
        payload_path.write_bytes(payload)
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match=expected_error):
            run_audit()


def test_real_billion_token_ceiling_and_resume_log_duplicates():
    training = {
        "token_budget": 1_000_000_000,
        "global_batch": 128,
        "sequence_length": 1024,
    }
    result = {"optimizer_steps": 7630, "training_targets": 1_000_000_000}
    metrics = [
        {
            "event": "train",
            "step": step,
            "training_targets": min(step * 131072, 1_000_000_000),
            "loss": 2.0,
            "grad_norm": 1.0,
            "step_seconds": 0.1,
        }
        for step in range(1, 7631)
    ]
    metrics.append(copy.deepcopy(metrics[100]))
    audit = validate_training_steps(metrics, training, result)
    assert audit["final_step_targets"] == 51712
    assert audit["replayed_log_steps_after_resume"] == 1
    with pytest.raises(ValueError, match="ceiling"):
        validate_training_steps(metrics, training, {**result, "optimizer_steps": 1})


def test_reject_wrong_dataset_before_accepting_matching_token_budget():
    corpus = SimpleNamespace(
        manifest={"dataset": {"repo": "another/source", "revision": "revision"}}
    )
    with pytest.raises(ValueError, match="acquisition identity"):
        validate_data(
            corpus,
            {"dataset": "codelion/fineweb-edu-1B", "dataset_revision": "revision"},
        )
