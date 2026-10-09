"""Versioned final audit, separate from the immutable training implementation.

Run with ``python -m validation.verify_screen`` after all three runs and
source-consistent mature-context benchmarks finish. This module never opens
TPU devices, launches jobs, or changes execution sources.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from lm.data import TokenCorpus, sha256_file
from lm.recall import build_recall_dataset
from lm.runtime import atomic_json, source_provenance
from scripts.pod import HOSTS
from scripts.verify_lm_runs import (
    ARCHITECTURES,
    audit_run,
    read_json,
    require,
    validate_data,
    validate_hardware,
    validate_sources,
)


AUDIT_REVISION = "screen-final-audit-v2"
STANDARD_CASES = {
    "attention_bf16_T128_H2": (0.035, 0.075, 3),
    "attention_bf16_T256_H4": (0.035, 0.075, 3),
    "mamba3_bf16_T128_R1": (0.04, 0.08, 7),
    "mamba3_bf16_T139_R1": (0.04, 0.08, 7),
    "mamba3_bf16_T139_finalstate": (0.04, 0.08, 7),
    "mamba3_bf16_T65_R2": (0.04, 0.08, 7),
    "mamba4_bf16_cyclic": (0.002, 0.025, 6),
    "mamba4_bf16_fixed": (0.002, 0.025, 6),
}
QR_CASES = (
    "mamba4_protected_repeated_rankcut_0.0",
    "mamba4_protected_repeated_rankcut_0.002",
)
DECODE_CASE = "mamba4_bf16_protected_cached_vs_prefill_T65_and_eos_full_gradients"


def validate_errors(errors, tolerance, description):
    require(errors["finite"] is True, f"Nonfinite {description}")
    require(errors["passed"] is True, f"Failed {description}")
    require(errors["tolerance"] == tolerance, f"Changed {description} tolerance")
    for name in (
        "max_absolute_error",
        "max_error_over_reference_max",
        "relative_l2_error",
    ):
        value = errors[name]
        require(np.isfinite(value) and value >= 0, f"Invalid {description} {name}")
    require(
        errors["max_error_over_reference_max"] <= tolerance
        and errors["relative_l2_error"] <= tolerance,
        f"Numeric {description} errors exceed the frozen tolerance",
    )


def validate_kernel_cases(cases):
    expected = set(STANDARD_CASES) | set(QR_CASES) | {DECODE_CASE}
    require(
        len(cases) == len(expected) and {case["name"] for case in cases} == expected,
        "Missing, duplicated, or changed declared kernel cases",
    )
    for case in cases:
        name = case["name"]
        require(
            case["passed"] is True and case["local_chips_checked"] == 4,
            f"Kernel case failed or did not cover four chips: {name}",
        )
        if name in STANDARD_CASES:
            forward_tolerance, gradient_tolerance, inputs = STANDARD_CASES[name]
            require(
                len(case["input_shapes"]) == inputs
                and len(case["gradients"]) == inputs,
                f"Missing input-gradient coverage: {name}",
            )
            validate_errors(case["forward"], forward_tolerance, f"{name} forward")
            for gradient in case["gradients"]:
                validate_errors(gradient, gradient_tolerance, f"{name} gradient")
        elif name in QR_CASES:
            require(
                case["finite_loss_and_all_key_value_query_gradients"] is True,
                f"Nonfinite repeated-key loss or gradients: {name}",
            )
            require(
                case["positive_retained_qr_diagonals"] is True,
                f"Invalid QR rank margin: {name}",
            )
            diagnostics = case["diagnostics"]
            fields = (
                "protected_active_banks",
                "protected_all_finite",
                "protected_merges",
                "protected_min_qr_diagonal",
                "protected_retained_anchors",
                "protected_retained_anchors_min",
                "protected_retained_anchors_max",
            )
            for field in fields:
                values = np.asarray(diagnostics[field])
                require(
                    values.shape == (4,) and np.isfinite(values).all(),
                    f"Missing finite per-chip QR diagnostics: {name} {field}",
                )
            require(
                np.all(np.asarray(diagnostics["protected_all_finite"]) == 1),
                f"Nonfinite protected banks: {name}",
            )
            require(
                np.all(np.asarray(diagnostics["protected_min_qr_diagonal"]) > 1e-5),
                f"Numeric QR rank margin failed: {name}",
            )
            for field in (
                "protected_active_banks",
                "protected_merges",
                "protected_retained_anchors",
                "protected_retained_anchors_min",
                "protected_retained_anchors_max",
            ):
                values = np.asarray(diagnostics[field])
                require(
                    np.all(values >= 1) and np.all(values == np.floor(values)),
                    f"Invalid protected bank/resource count: {name} {field}",
                )
            require(
                np.all(
                    np.asarray(diagnostics["protected_retained_anchors_min"])
                    <= np.asarray(diagnostics["protected_retained_anchors_max"])
                )
                and np.all(
                    np.asarray(diagnostics["protected_retained_anchors_max"])
                    <= np.asarray(diagnostics["protected_retained_anchors"])
                ),
                f"Inconsistent retained-anchor range: {name}",
            )
        else:
            require(
                case["context_tokens"] == 65 and case["protected_block_size"] == 64,
                "Cached/full protected conformance did not cross a block boundary",
            )
            validate_errors(
                case["cached_decode"], 0.035, "protected cached/full decode"
            )
            require(
                case["finite_repeated_eos_loss_and_all_parameter_gradients"] is True,
                "Repeated-EOS full-model loss or parameter gradient became nonfinite",
            )


def validate_benchmark(row, architecture, frozen_model, sources, minimum_context=1024):
    validate_sources(row["provenance"]["sha256"], sources)
    require(
        row["architecture"] == architecture and row["model_config"] == frozen_model,
        "Benchmark architecture configuration changed",
    )
    require(
        row["global_batch"] == 128 and row["sequence_length"] == 1024,
        "Benchmark does not measure the production batch and length",
    )
    summary = {}
    for phase in ("train", "prefill_loss", "cached_decode"):
        measured = row[phase]
        times = np.asarray(measured["steady_seconds"])
        require(
            times.ndim == 1
            and len(times) >= 1
            and np.isfinite(times).all()
            and np.all(times > 0),
            f"Missing finite steady {phase} timings",
        )
        throughput = measured["tokens_per_second"]
        require(
            np.isfinite(throughput) and throughput > 0, f"Invalid {phase} throughput"
        )
        tokens = row["global_batch"] * (
            1 if phase == "cached_decode" else row["sequence_length"]
        )
        require(
            np.isclose(throughput, tokens / np.median(times), rtol=1e-6),
            f"{phase} throughput does not reproduce from measured timings",
        )
        compiler = measured["compiler"]
        memory, cost = compiler["memory_analysis"], compiler["cost_analysis"]
        require(
            "unavailable" not in memory and "unavailable" not in cost,
            f"Unavailable {phase} compiler evidence",
        )
        for field in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        ):
            require(
                np.isfinite(memory[field]) and memory[field] >= 0,
                f"Invalid {phase} compiler memory: {field}",
            )
        require(
            cost["flops"] > 0
            and np.isfinite(cost["flops"])
            and cost["bytes accessed"] > 0
            and np.isfinite(cost["bytes accessed"]),
            f"Missing positive {phase} compiler FLOP/byte counts",
        )
        summary[phase] = {
            "tokens_per_second": float(throughput),
            "flops": float(cost["flops"]),
            "bytes_accessed": float(cost["bytes accessed"]),
            "arithmetic_intensity": float(cost["flops"] / cost["bytes accessed"]),
        }
    require(
        np.isfinite(row["train"]["loss"]) and np.isfinite(row["train"]["grad_norm"]),
        "Nonfinite production-shape benchmark loss or gradients",
    )
    require(
        np.isfinite(row["prefill_loss"]["nll_sum"])
        and row["prefill_loss"]["nll_sum"] > 0,
        "Nonfinite production-shape prefill loss",
    )
    decode = row["cached_decode"]
    require(decode["finite_logits"] is True, "Nonfinite benchmark cached-decode logits")
    require(
        decode["prefix_context_tokens"] >= minimum_context,
        "Cached decode benchmark lacks a mature 1024-token context",
    )
    require(
        decode["cache_capacity_tokens"]
        >= decode["prefix_context_tokens"] + len(decode["steady_seconds"]) + 1,
        "Decode cache capacity does not cover measured positions",
    )
    require(
        decode["cache_bytes_per_device"] > 0, "Missing cached-state memory accounting"
    )
    memory = row["runtime_memory"]
    require(len(memory) == 4, "Missing a local chip's runtime memory evidence")
    for item in memory:
        require(
            item and 0 < item["peak_bytes_in_use"] <= item["bytes_limit"],
            "Missing or impossible observed device peak memory",
        )
    summary["context_tokens"] = decode["prefix_context_tokens"]
    summary["peak_device_bytes"] = max(item["peak_bytes_in_use"] for item in memory)
    if architecture == "mamba4":
        execution = row["memory_diagnostic_execution"]
        require(
            execution["passed"] is True, "Mamba4 benchmark memory diagnostics failed"
        )
        require(
            np.isfinite(execution["compile_seconds"])
            and execution["compile_seconds"] >= 0
            and np.isfinite(execution["seconds"])
            and execution["seconds"] > 0,
            "Missing finite Mamba4 benchmark diagnostic execution timings",
        )
        summary["learned_memory_diagnostics"] = validate_memory_metrics(
            row["memory_diagnostics"], frozen_model
        )
        summary["memory_diagnostic_execution"] = {
            "passed": True,
            "compile_seconds": execution["compile_seconds"],
            "seconds": execution["seconds"],
        }
    return summary


def validate_engineering(benchmarks, conformance, sources, frozen_models=None):
    result = {"benchmarks": {}, "kernel_conformance": []}
    for architecture in ARCHITECTURES:
        paths = [Path(benchmarks) / f"{architecture}-host-{i}.json" for i in range(4)]
        rows = [read_json(path) for path in paths]
        validate_hardware([row["hardware"] for row in rows])
        model = (
            frozen_models[architecture]
            if frozen_models is not None
            else read_json(f"lm/configs/{architecture}-60m.json")["model"]
        )
        result["benchmarks"][architecture] = [
            {
                "file": str(path),
                "sha256": sha256_file(path),
                "measurements": validate_benchmark(row, architecture, model, sources),
            }
            for path, row in zip(paths, rows, strict=True)
        ]
    paths = [Path(conformance) / f"host-{i}.json" for i in range(4)]
    rows = [read_json(path) for path in paths]
    validate_hardware([row["hardware"] for row in rows])
    for path, row in zip(paths, rows, strict=True):
        validate_sources(row["provenance"]["sha256"], sources)
        require(
            row["passed"] is True and row["pallas_tpu_verified"] is True,
            "Fused TPU conformance did not pass",
        )
        validate_kernel_cases(row["cases"])
        result["kernel_conformance"].append(
            {
                "file": str(path),
                "sha256": sha256_file(path),
                "cases": len(row["cases"]),
                "local_chips": 4,
            }
        )
    return result


def validate_memory_metrics(metrics, model):
    """Inspect actual scalar diagnostics emitted by each frozen memory module."""
    required = (
        "precision_eigenvalue_min",
        "precision_eigenvalue_max",
        "precision_condition_max",
        "prior_diagonal_min",
        "prior_diagonal_max",
        "gaussian_cross_abs_max",
        "gaussian_allfinite",
        "epsilon_min",
        "epsilon_max",
        "decay_min",
        "decay_max",
        "beta_min",
        "beta_max",
    )
    protected = (
        "protected_qr_diagonal_min",
        "protected_allfinite",
        "protected_bank_count_max",
        "protected_retained_anchors_min",
        "protected_retained_anchors_max",
        "protected_merge_count_max",
    )
    require(
        metrics
        and all(
            np.asarray(value).ndim == 0 and np.isfinite(value)
            for value in metrics.values()
        ),
        "Missing finite scalar Mamba4 learned diagnostics",
    )
    eigen_min, condition_max, qr_min = np.inf, 0.0, np.inf
    for layer in range(model["n_layers"]):
        prefix = f"layer_{layer}/memory/"
        names = required + (protected if model["protected_anchor_budget"] else ())
        require(
            all(prefix + name in metrics for name in names),
            f"Missing Mamba4 layer diagnostics: {layer}",
        )
        values = {name: metrics[prefix + name] for name in names}
        require(
            values["gaussian_allfinite"] == 1,
            "Nonfinite learned Gaussian factors/state",
        )
        require(
            values["precision_eigenvalue_min"] > 0
            and values["precision_eigenvalue_max"]
            >= values["precision_eigenvalue_min"],
            "Learned precision lost positive definiteness",
        )
        require(
            1
            <= values["precision_condition_max"]
            <= 1.001
            * values["precision_eigenvalue_max"]
            / values["precision_eigenvalue_min"],
            "Invalid learned precision condition evidence",
        )
        require(
            0 < values["prior_diagonal_min"] <= values["prior_diagonal_max"],
            "Nonpositive learned prior",
        )
        require(
            0 < values["epsilon_min"] <= values["epsilon_max"],
            "Nonpositive learned epsilon",
        )
        require(
            0 < values["decay_min"] <= values["decay_max"] < 1,
            "Invalid learned decay gate",
        )
        require(
            0 < values["beta_min"] <= values["beta_max"],
            "Nonpositive learned write precision",
        )
        require(
            values["gaussian_cross_abs_max"] >= 0, "Invalid cross-statistic magnitude"
        )
        if model["protected_anchor_budget"]:
            require(
                values["protected_allfinite"] == 1
                and values["protected_qr_diagonal_min"] > 0,
                "Nonfinite protected state or nonpositive learned QR margin",
            )
            require(
                0
                < values["protected_retained_anchors_min"]
                <= values["protected_retained_anchors_max"],
                "Protected state retained no valid anchors",
            )
            require(
                values["protected_bank_count_max"] > 0
                and values["protected_merge_count_max"] >= 0,
                "Invalid protected resource diagnostics",
            )
            qr_min = min(qr_min, values["protected_qr_diagonal_min"])
        eigen_min = min(eigen_min, values["precision_eigenvalue_min"])
        condition_max = max(condition_max, values["precision_condition_max"])
    return {
        "precision_eigenvalue_min": float(eigen_min),
        "precision_condition_max": float(condition_max),
        "protected_qr_diagonal_min": float(qr_min)
        if model["protected_anchor_budget"]
        else None,
    }


def validate_mamba4_diagnostics(directory, frozen):
    training, model = frozen["training"], frozen["model"]
    steps = (
        training["token_budget"]
        + training["global_batch"] * training["sequence_length"]
        - 1
    ) // (training["global_batch"] * training["sequence_length"])
    expected_steps = {0, steps} | set(
        range(training["eval_every"], steps, training["eval_every"])
    )
    entries = [
        json.loads(line)
        for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()
    ]
    diagnostics = [entry for entry in entries if entry["event"] == "diagnostics"]
    require(
        {entry["step"] for entry in diagnostics} == expected_steps,
        "Missing initial, periodic, or final Mamba4 learned-state diagnostics",
    )
    summaries = [
        validate_memory_metrics(entry["metrics"], model) for entry in diagnostics
    ]
    final = [entry for entry in diagnostics if entry["step"] == steps][-1]
    require(
        read_json(Path(directory) / "diagnostics.json") == final,
        "Final diagnostic sidecar differs from training evidence",
    )
    return {
        "probe_steps": sorted(expected_steps),
        "logged_probe_entries": len(diagnostics),
        "precision_eigenvalue_min": min(
            row["precision_eigenvalue_min"] for row in summaries
        ),
        "precision_condition_max": max(
            row["precision_condition_max"] for row in summaries
        ),
        "protected_qr_diagonal_min": min(
            row["protected_qr_diagonal_min"] for row in summaries
        )
        if model["protected_anchor_budget"]
        else None,
        "scope": "All hosts/chips, first fixed held-out batch; not every training token or a calibration proof",
    }


def validate_crosshost_checkpoint(path, run, required=False):
    path = Path(path)
    if not path.exists():
        require(
            not required,
            "Missing actual-payload verification from every physical worker",
        )
        return {"available": False}
    sidecar = read_json(path)
    require(
        sidecar["architecture"] == run["architecture"]
        and sidecar["optimizer_steps"] == run["optimizer_steps"]
        and sidecar["training_targets"] == run["training_targets"]
        and sidecar["fingerprint"] == run["fingerprint"],
        "Physical-worker checkpoint identity differs from audited run",
    )
    require(
        sidecar["all_actual_payloads_equal"] is True,
        "Physical-worker final checkpoint payloads disagree",
    )
    workers = sidecar["workers"]
    require(
        len(workers) == 4 and {worker["host"] for worker in workers} == set(HOSTS),
        "Missing physical worker checkpoint checksum",
    )
    require(
        all(
            worker["checkpoint_sha256"] == run["final_checkpoint_sha256"]
            for worker in workers
        ),
        "Actual physical-worker checkpoint checksum mismatch",
    )
    return {
        "available": True,
        "file": str(path),
        "sha256": sha256_file(path),
        "workers": 4,
    }


def audit_provenance():
    files = sorted(Path("validation").rglob("*.py"))
    return {
        "revision": AUDIT_REVISION,
        "sha256": {str(path): sha256_file(path) for path in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="lm/runs/screen-60m-v1")
    parser.add_argument("--data", default="data/fineweb-edu-1b")
    parser.add_argument("--output", default="lm/results/screen-60m-v1")
    parser.add_argument("--benchmarks", default="lm/results/benchmarks")
    parser.add_argument("--conformance", default="lm/results/kernel-conformance")
    parser.add_argument("--require-crosshost-checkpoints", action="store_true")
    args = parser.parse_args()
    protocol = read_json("lm/configs/screen-protocol.json")
    require(
        protocol["training_targets_per_architecture"] == 1_000_000_000
        and protocol["seed"] == 42,
        "Requested 60M/1B/seed42 scope changed",
    )
    corpus = TokenCorpus(args.data, verify_hashes=True)
    validate_data(corpus, protocol)
    sources = source_provenance()["sha256"]
    recall = build_recall_dataset(corpus.path / "tokenizer.json")
    engineering = validate_engineering(args.benchmarks, args.conformance, sources)
    rows = [
        audit_run(
            Path(args.runs) / name,
            name,
            protocol,
            corpus,
            expected_sources=sources,
            expected_recall=recall,
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
    training = [
        read_json(Path(args.runs) / name / "manifest.json")["protocol"]["training"]
        for name in ARCHITECTURES
    ]
    require(
        all(item == training[0] for item in training),
        "Training protocols differ between architectures",
    )
    diagnostics = validate_mamba4_diagnostics(
        Path(args.runs) / "mamba4", read_json("lm/configs/mamba4-60m.json")
    )
    output = Path(args.output)
    crosshost = {
        row["architecture"]: validate_crosshost_checkpoint(
            output / f"{row['architecture']}-checkpoints.json",
            row,
            args.require_crosshost_checkpoints,
        )
        for row in rows
    }
    win = all(rows[2]["heldout"]["nll"] < row["heldout"]["nll"] for row in rows[:2])
    value = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "auditor": audit_provenance(),
        "frozen_execution_source_count": len(sources),
        "frozen_execution_source_hashes": sources,
        "protocol": protocol,
        "all_requested_runs_completed": True,
        "screen_win": win,
        "runs": rows,
        "engineering": engineering,
        "mamba4_learned_diagnostics": diagnostics,
        "physical_worker_checkpoint_verification": crosshost,
        "data_manifest_sha256": sha256_file(corpus.path / "manifest.json"),
        "limitation": "One training seed; diagnostic intervals quantify prompt variability only",
    }
    atomic_json(output / "audit.json", value)
    lines = [
        "# 60M, 1B-token, single-seed screen",
        "",
        "All three exact target budgets, actual final states, frozen recall results and four-host TPU evidence passed the separate final audit.",
        "",
        "| Model | Parameters | Held-out NLL | Perplexity | Train targets/s | Recall accuracy |",
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
        f"Frozen execution provenance covers {len(sources)} files. Separate auditor revision: `{AUDIT_REVISION}`; its hashes appear in audit.json.",
        "One seed does not establish robustness. The final 0.11316% of training targets reuse training documents; held-out documents remain excluded.",
        "All benchmark paths use the production batch/length and cached decode after at least 1024 context tokens. Numerical conformance and learned-state diagnostics remain distinct from predictive superiority.",
        "Scaling beyond the requested 60M screen was not performed.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "all_requested_runs_completed": True,
                "screen_win": win,
                "audit": str(output / "audit.json"),
                "auditor": AUDIT_REVISION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
