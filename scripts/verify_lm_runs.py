"""Audit actual checkpoints, exact budgets, all hosts, and frozen screen evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from flax import serialization
import numpy as np

from lm.data import TokenCorpus, sha256_file
from lm.recall import build_recall_dataset, summarize_recall
from lm.runtime import atomic_json, source_provenance


ARCHITECTURES = ("transformer", "mamba3", "mamba4")


def require(condition, message):
    """Audits remain active even if Python is invoked with optimization enabled."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_sources(recorded, expected):
    require(recorded == expected, "Source hashes changed or differ between runs")


def validate_data(corpus, protocol):
    manifest = corpus.manifest
    for key in ("repo", "revision"):
        require(
            manifest["dataset"][key]
            == protocol["dataset" if key == "repo" else "dataset_revision"],
            "Dataset acquisition identity differs from frozen protocol",
        )
        require(
            manifest["tokenizer"][key]
            == protocol["tokenizer" if key == "repo" else "tokenizer_revision"],
            "Tokenizer identity differs from frozen protocol",
        )
    require(corpus.vocab_size == protocol["vocabulary"] == 50257, "Vocabulary mismatch")
    require(corpus.eos_id == 50256, "EOS mismatch")
    require(
        corpus.target_count("train") == protocol["training_targets_per_architecture"],
        "Training corpus budget mismatch",
    )
    require(
        corpus.target_count("eval") == protocol["heldout_targets"],
        "Held-out corpus budget mismatch",
    )
    require(
        manifest["splits"]["train"]["replayed_stream_tokens"]
        == protocol["training_replay_targets"],
        "Training replay budget mismatch",
    )
    require(
        not manifest["splits"]["eval"]["source_replay"],
        "Evaluation source was replayed",
    )
    require(
        len(manifest["dataset"]["files"]) == 10, "Missing source parquet identities"
    )
    require(
        sha256_file(corpus.path / "tokenizer.json") == manifest["tokenizer"]["sha256"],
        "Tokenizer file checksum mismatch",
    )
    ledger_item = manifest["document_ledger"]
    require(
        sha256_file(corpus.path / ledger_item["file"]) == ledger_item["sha256"],
        "Document ledger checksum mismatch",
    )
    ledger = np.load(corpus.path / ledger_item["file"], allow_pickle=False)
    hashes = {}
    for split in ("train", "eval"):
        item = manifest["splits"][split]
        path = corpus.path / item["document_order_file"]
        require(
            sha256_file(path) == item["document_order_sha256"],
            f"{split} document order checksum mismatch",
        )
        order = np.load(path, allow_pickle=False)
        require(
            np.all(ledger["split"][order] == (0 if split == "train" else 1)),
            f"{split} document split mismatch",
        )
        hashes[split] = np.unique(
            ledger["sha256"][order].copy().view("S32").reshape(-1)
        )
    require(
        np.intersect1d(hashes["train"], hashes["eval"]).size == 0,
        "Training/evaluation exact-document leakage",
    )


def validate_hardware(records):
    require(len(records) == 4, "Missing a worker's hardware evidence")
    require(
        {item["process_index"] for item in records} == set(range(4)),
        "Hardware process indices must be exactly 0,1,2,3",
    )
    require(
        len({item["hostname"] for item in records}) == 4,
        "Hardware evidence does not identify four distinct hosts",
    )
    topologies = []
    for item in records:
        require(item["backend"] == "tpu", "Training did not use TPU devices")
        require(
            item["device_count"] == 16
            and item["process_count"] == 4
            and item["local_device_count"] == 4,
            "Hardware requires 16 devices on four hosts, four local devices each",
        )
        devices = item["devices"]
        require(
            len(devices) == 16 and len({device["id"] for device in devices}) == 16,
            "Hardware does not identify 16 unique devices",
        )
        require(
            all("v4" in device["kind"] for device in devices),
            "Hardware devices are not all TPU v4",
        )
        require(
            len({tuple(device["coords"]) for device in devices}) == 16,
            "TPU chip coordinates are not unique",
        )
        for process in range(4):
            require(
                sum(device["process_index"] == process for device in devices) == 4,
                "Device ownership does not cover every host",
            )
        topologies.append(
            sorted(
                (
                    device["id"],
                    device["process_index"],
                    tuple(device["coords"]),
                    device["kind"],
                )
                for device in devices
            )
        )
    require(
        all(topology == topologies[0] for topology in topologies),
        "Workers disagree on hardware topology",
    )


def validate_training_steps(metrics, training, result):
    budget = int(training["token_budget"])
    capacity = int(training["global_batch"] * training["sequence_length"])
    expected_steps = math.ceil(budget / capacity)
    require(
        result["optimizer_steps"] == expected_steps,
        "Optimizer-step count differs from exact target-budget ceiling",
    )
    train_entries = [entry for entry in metrics if entry["event"] == "train"]
    steps = {entry["step"]: entry for entry in train_entries}
    require(
        set(steps) == set(range(1, expected_steps + 1)),
        "Missing optimizer-step evidence, including the final step",
    )
    for entry in train_entries:
        step = entry["step"]
        require(
            entry["training_targets"] == min(step * capacity, budget),
            f"Training target accounting mismatch at step {step}",
        )
        require(
            np.isfinite(entry["loss"]) and np.isfinite(entry["grad_norm"]),
            f"Nonfinite training evidence at step {step}",
        )
        require(
            np.isfinite(entry["step_seconds"]) and entry["step_seconds"] > 0,
            "Invalid training timing",
        )
    require(
        steps[expected_steps]["training_targets"]
        == budget
        == result["training_targets"],
        "Final optimizer step does not prove complete target budget",
    )
    return {
        "unique_steps": expected_steps,
        "logged_steps": len(train_entries),
        "replayed_log_steps_after_resume": len(train_entries) - expected_steps,
        "final_step_targets": budget - (expected_steps - 1) * capacity,
    }


def validate_recall(directory, result, protocol, expected_dataset=None):
    summary = result.get("recall")
    require(isinstance(summary, dict), "Missing frozen trained recall result")
    raw = read_json(directory / "recall.json")
    recall_protocol = raw["protocol"]
    require(
        recall_protocol["protocol"] == protocol["recall_protocol"],
        "Recall protocol changed",
    )
    require(recall_protocol["seed"] == protocol["recall_seed"], "Recall seed changed")
    require(
        recall_protocol["array_sha256"] == protocol["recall_array_sha256"],
        "Frozen recall prompt array hash changed",
    )
    count = protocol.get("recall_prompt_count", 128)
    require(recall_protocol["prompt_count"] == count, "Recall prompt count changed")
    records = raw["records"]
    require(
        len(records) == count
        and [entry["prompt_index"] for entry in records] == list(range(count)),
        "Missing or reordered recall prompt evidence",
    )
    for key in ("protocol", "overall", "groups"):
        require(
            summary[key] == raw[key], "Recall summary and raw prompt evidence disagree"
        )
    require(
        raw["overall"]["prompts"] == count,
        "Recall aggregate prompt accounting mismatch",
    )
    require(0 <= raw["overall"]["accuracy"] <= 1, "Invalid recall accuracy")
    for entry in records:
        require(
            np.isfinite(entry["candidate_log_probs"]).all(), "Nonfinite recall scores"
        )
        require(np.isfinite(entry["gold_full_vocab_nll"]), "Nonfinite recall NLL")
    if expected_dataset is not None:
        require(
            recall_protocol == expected_dataset.manifest,
            "Recall tokenizer or construction provenance differs",
        )
        saved_arrays = np.load(
            directory / "recall-prompts/prompts.npz", allow_pickle=False
        )
        for key, expected in expected_dataset.arrays.items():
            require(
                np.array_equal(saved_arrays[key], expected),
                f"Saved frozen recall array changed: {key}",
            )
        require(
            read_json(directory / "recall-prompts/prompts.json")
            == list(expected_dataset.prompts),
            "Saved frozen recall texts changed",
        )
        scores = np.asarray([entry["candidate_log_probs"] for entry in records])
        require(
            summarize_recall(expected_dataset, scores) == raw,
            "Recall metrics do not reproduce from whole-prompt scores",
        )
    return summary


def validate_checkpoint(directory, result, ledger):
    pointer = read_json(directory / "latest.json")["checkpoint"]
    checkpoint = directory / pointer
    metadata = read_json(checkpoint / "metadata.json")
    require(
        metadata["step"] == result["optimizer_steps"],
        "Final checkpoint metadata step mismatch",
    )
    require(
        metadata["training_targets"] == result["training_targets"],
        "Final checkpoint target accounting mismatch",
    )
    require(
        metadata["fingerprint"] == result["fingerprint"],
        "Final checkpoint fingerprint mismatch",
    )
    payload = (checkpoint / "state.msgpack").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(
        digest == metadata["checkpoint_sha256"], "Final checkpoint checksum mismatch"
    )
    state = serialization.msgpack_restore(payload)
    require(
        int(state["step"]) == result["optimizer_steps"],
        "Actual optimizer-state step differs from declared completion",
    )
    parameters = {}

    def visit(tree, prefix=""):
        if isinstance(tree, dict):
            for key, value in tree.items():
                visit(value, prefix + ("/" if prefix else "") + str(key))
        else:
            array = np.asarray(tree)
            require(
                np.isfinite(array).all(), f"Nonfinite checkpoint parameter: {prefix}"
            )
            parameters[prefix] = array

    visit(state["params"])
    entries = {entry["name"]: entry for entry in ledger["parameters"]}
    require(
        set(parameters) == set(entries),
        "Actual checkpoint parameter names differ from ledger",
    )
    for name, array in parameters.items():
        require(
            list(array.shape) == entries[name]["shape"]
            and array.size == entries[name]["count"],
            f"Actual checkpoint parameter shape/count differs: {name}",
        )
    return digest


def audit_run(
    directory,
    architecture,
    protocol,
    corpus,
    *,
    frozen=None,
    expected_sources=None,
    expected_recall=None,
    parameter_target=60_000_000,
):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    result = read_json(directory / "result.json")
    training = manifest["protocol"]["training"]
    frozen = frozen or read_json(f"lm/configs/{architecture}-60m.json")
    expected_sources = expected_sources or source_provenance()["sha256"]
    require(manifest["protocol"] == frozen, "Architecture or training protocol changed")
    validate_sources(manifest["sources"], expected_sources)
    identity = {
        "protocol": manifest["protocol"],
        "sources": manifest["sources"],
        "data_manifest_sha256": manifest["data_manifest_sha256"],
    }
    require(
        fingerprint(identity) == manifest["fingerprint"] == result["fingerprint"],
        "Run fingerprint does not reproduce from its protocol, sources, and corpus",
    )
    require(result["status"] == "completed", "Run is not completed")
    require(result["architecture"] == architecture, "Architecture result mismatch")
    require(
        result["seed"] == protocol["seed"] == training["seed"], "Training seed mismatch"
    )
    require(
        result["training_targets"]
        == training["token_budget"]
        == protocol["training_targets_per_architecture"],
        "Run training budget mismatch",
    )
    require(
        result["heldout"]["targets"]
        == training["eval_tokens"]
        == protocol["heldout_targets"],
        "Run held-out budget mismatch",
    )
    require(result["observed_device_count"] == 16, "Missing all-device result")
    require(
        manifest["data_manifest_sha256"] == sha256_file(corpus.path / "manifest.json"),
        "Corpus manifest changed",
    )
    ledger = manifest["parameter_ledger"]
    require(
        result["parameter_count"] == ledger["total"],
        "Parameter count result differs from ledger",
    )
    require(
        abs(ledger["total"] / parameter_target - 1) <= 0.02,
        "Parameter count differs from declared target by more than 2%",
    )
    require(
        sum(item["count"] for item in ledger["parameters"]) == ledger["total"],
        "Parameter ledger total mismatch",
    )
    require(
        ledger["embedding"] + ledger["non_embedding"] == ledger["total"],
        "Embedding/nonembedding ledger mismatch",
    )
    expected_hosts = {f"result-host-{index}.json" for index in range(4)}
    require(
        {file.name for file in directory.glob("result-host-*.json")} == expected_hosts,
        "Missing a worker's completed result or unexpected process index",
    )
    for name in expected_hosts:
        worker = read_json(directory / name)
        for field in (
            "status",
            "architecture",
            "seed",
            "training_targets",
            "optimizer_steps",
            "parameter_count",
            "fingerprint",
            "heldout",
            "recall",
            "observed_device_count",
        ):
            require(
                worker[field] == result[field], f"Worker disagreement: {name} {field}"
            )
    hardware = [read_json(directory / f"hardware-{index}.json") for index in range(4)]
    validate_hardware(hardware)
    metrics = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    accounting = validate_training_steps(metrics, training, result)
    initial = next(entry for entry in metrics if entry["event"] == "initial_eval")
    require(
        initial["targets"] == protocol["heldout_targets"],
        "Initial evaluation target budget mismatch",
    )
    require(
        np.isfinite(initial["nll"]) and np.isfinite(result["heldout"]["nll"]),
        "Nonfinite held-out loss",
    )
    require(result["heldout"]["nll"] < initial["nll"], "Held-out loss did not improve")
    require(
        np.isclose(
            result["heldout"]["perplexity"],
            math.exp(result["heldout"]["nll"]),
            rtol=1e-10,
        ),
        "Held-out perplexity differs from exp(NLL)",
    )
    require(
        np.isfinite(result["steady_tokens_per_second"])
        and result["steady_tokens_per_second"] > 0,
        "Missing positive steady training throughput",
    )
    recall = validate_recall(directory, result, protocol, expected_recall)
    digest = validate_checkpoint(directory, result, ledger)
    return {
        "architecture": architecture,
        "total_parameters": ledger["total"],
        "embedding_parameters": ledger["embedding"],
        "non_embedding_parameters": ledger["non_embedding"],
        "training_targets": result["training_targets"],
        "seed": result["seed"],
        "optimizer_steps": result["optimizer_steps"],
        "accounting": accounting,
        "heldout": result["heldout"],
        "initial_nll": initial["nll"],
        "steady_tokens_per_second": result["steady_tokens_per_second"],
        "final_checkpoint_sha256": digest,
        "fingerprint": result["fingerprint"],
        "recall": recall,
        "all_four_workers_completed": True,
        "sources": manifest["sources"],
    }


def validate_engineering(benchmarks, conformance, sources):
    evidence = {"benchmarks": {}, "kernel_conformance": {}}
    for architecture in ARCHITECTURES:
        rows = [
            read_json(Path(benchmarks) / f"{architecture}-host-{index}.json")
            for index in range(4)
        ]
        validate_hardware([row["hardware"] for row in rows])
        frozen_model = read_json(f"lm/configs/{architecture}-60m.json")["model"]
        for row in rows:
            validate_sources(row["provenance"]["sha256"], sources)
            require(
                row["architecture"] == architecture
                and row["model_config"] == frozen_model,
                "Benchmark architecture configuration changed",
            )
            require(
                row["global_batch"] == 128 and row["sequence_length"] == 1024,
                "Benchmark does not measure production batch/length",
            )
            for phase in ("train", "prefill_loss", "cached_decode"):
                measured = row[phase]
                require(
                    len(measured["steady_seconds"]) > 0
                    and all(
                        np.isfinite(value) and value > 0
                        for value in measured["steady_seconds"]
                    ),
                    f"Missing finite steady {phase} timing",
                )
                require(
                    np.isfinite(measured["tokens_per_second"])
                    and measured["tokens_per_second"] > 0,
                    f"Missing {phase} throughput",
                )
                compiler = measured["compiler"]
                require(
                    "memory_analysis" in compiler and "cost_analysis" in compiler,
                    f"Missing {phase} compiler cost/memory analysis",
                )
                require(
                    "unavailable" not in compiler["memory_analysis"]
                    and "unavailable" not in compiler["cost_analysis"],
                    f"Unavailable {phase} compiler evidence",
                )
            require(
                row["cached_decode"]["finite_logits"], "Nonfinite cached decode logits"
            )
            require(
                len(row["runtime_memory"]) == 4
                and all(
                    item and item.get("peak_bytes_in_use", 0) > 0
                    for item in row["runtime_memory"]
                ),
                "Missing observed device peak memory",
            )
        evidence["benchmarks"][architecture] = [
            sha256_file(Path(benchmarks) / f"{architecture}-host-{index}.json")
            for index in range(4)
        ]
    rows = [read_json(Path(conformance) / f"host-{index}.json") for index in range(4)]
    validate_hardware([row["hardware"] for row in rows])
    for row in rows:
        validate_sources(row["provenance"]["sha256"], sources)
        require(
            row["passed"] and row["pallas_tpu_verified"],
            "Fused TPU kernel conformance did not pass",
        )
        require(len(row["cases"]) >= 7, "Missing declared kernel conformance cases")
        for case in row["cases"]:
            require(
                case["passed"] and case["local_chips_checked"] == 4,
                "Kernel case did not cover every local chip",
            )
            require(
                case["forward"]["passed"]
                and case["gradients"]
                and all(gradient["passed"] for gradient in case["gradients"]),
                "Kernel forward/backward conformance failed",
            )
    evidence["kernel_conformance"] = [
        sha256_file(Path(conformance) / f"host-{index}.json") for index in range(4)
    ]
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="lm/runs/screen-60m-v1")
    parser.add_argument("--data", default="data/fineweb-edu-1b")
    parser.add_argument("--output", default="lm/results/screen-60m-v1")
    parser.add_argument("--benchmarks", default="lm/results/benchmarks")
    parser.add_argument("--conformance", default="lm/results/kernel-conformance")
    options = parser.parse_args()
    protocol = read_json("lm/configs/screen-protocol.json")
    require(
        protocol["training_targets_per_architecture"] == 1_000_000_000
        and protocol["seed"] == 42,
        "The requested 1B-token seed42 scope changed",
    )
    corpus = TokenCorpus(options.data, verify_hashes=True)
    validate_data(corpus, protocol)
    sources = source_provenance()["sha256"]
    expected_recall = build_recall_dataset(corpus.path / "tokenizer.json")
    engineering = validate_engineering(options.benchmarks, options.conformance, sources)
    rows = [
        audit_run(
            Path(options.runs) / name,
            name,
            protocol,
            corpus,
            expected_sources=sources,
            expected_recall=expected_recall,
        )
        for name in ARCHITECTURES
    ]
    for field in ("total_parameters", "non_embedding_parameters"):
        counts = [row[field] for row in rows]
        require(max(counts) / min(counts) < 1.01, f"Parameter matching failed: {field}")
    require(
        len({row["embedding_parameters"] for row in rows}) == 1,
        "Embedding parameter counts differ",
    )
    training_protocols = [
        read_json(Path(options.runs) / name / "manifest.json")["protocol"]["training"]
        for name in ARCHITECTURES
    ]
    require(
        all(value == training_protocols[0] for value in training_protocols),
        "Training protocols differ between architectures",
    )
    win = all(rows[2]["heldout"]["nll"] < row["heldout"]["nll"] for row in rows[:2])
    value = {
        "protocol": protocol,
        "all_requested_runs_completed": True,
        "screen_win": win,
        "runs": rows,
        "engineering": engineering,
        "data_manifest_sha256": sha256_file(corpus.path / "manifest.json"),
        "limitation": "One seed per model; no between-seed uncertainty estimate",
    }
    output = Path(options.output)
    atomic_json(output / "audit.json", value)
    lines = [
        "# 60M, 1B-token, single-seed screen",
        "",
        "All three actual final checkpoint states, exact target budgets, frozen recall results, and four-worker TPU results were audited.",
        "",
        "| Model | Parameters | Held-out NLL | Perplexity | Train tokens/s | Recall accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['total_parameters']:,} | {row['heldout']['nll']:.6f} | {row['heldout']['perplexity']:.4f} | {row['steady_tokens_per_second']:,.0f} | {row['recall']['overall']['accuracy']:.3f} |"
        )
    lines += [
        "",
        f"Declared Mamba4 screen win: **{win}**.",
        "",
        "The same dataset, tokenizer, batch, optimizer, source hashes and training seed were used.",
        "The final 0.11316% of training targets reuse training documents; held-out",
        "documents remain excluded. One seed cannot establish robustness.",
        "",
        "See audit.json for final checkpoint hashes, exact target counts, per-model",
        "fingerprints, raw-evidence references and every outcome. Scaling beyond 60M was not performed.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
