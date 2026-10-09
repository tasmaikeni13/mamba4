"""Reject corrupt or incomplete final evidence without accessing TPU devices."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from validation.verify_screen import (
    DECODE_CASE,
    QR_CASES,
    STANDARD_CASES,
    validate_benchmark,
    validate_crosshost_checkpoint,
    validate_engineering,
    validate_kernel_cases,
    validate_mamba4_diagnostics,
)


def errors(tolerance):
    return {
        "finite": True,
        "passed": True,
        "tolerance": tolerance,
        "max_absolute_error": 0.001,
        "max_error_over_reference_max": tolerance / 10,
        "relative_l2_error": tolerance / 10,
    }


def cases():
    result = []
    for name, (forward, backward, inputs) in STANDARD_CASES.items():
        result.append(
            {
                "name": name,
                "local_chips_checked": 4,
                "passed": True,
                "input_shapes": [[1]] * inputs,
                "forward": errors(forward),
                "gradients": [errors(backward) for _ in range(inputs)],
            }
        )
    for name in QR_CASES:
        result.append(
            {
                "name": name,
                "local_chips_checked": 4,
                "passed": True,
                "finite_loss_and_all_key_value_query_gradients": True,
                "positive_retained_qr_diagonals": True,
                "diagnostics": {
                    "protected_active_banks": [3] * 4,
                    "protected_all_finite": [True] * 4,
                    "protected_merges": [1] * 4,
                    "protected_min_qr_diagonal": [0.002] * 4,
                    "protected_retained_anchors": [3] * 4,
                    "protected_retained_anchors_min": [3] * 4,
                    "protected_retained_anchors_max": [3] * 4,
                },
            }
        )
    result.append(
        {
            "name": DECODE_CASE,
            "passed": True,
            "local_chips_checked": 4,
            "context_tokens": 65,
            "protected_block_size": 64,
            "cached_decode": errors(0.035),
            "finite_repeated_eos_loss_and_all_parameter_gradients": True,
        }
    )
    return result


def hardware(index):
    return {
        "process_index": index,
        "hostname": f"worker-{index}",
        "backend": "tpu",
        "device_count": 16,
        "local_device_count": 4,
        "process_count": 4,
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


def benchmark(architecture="mamba4", index=0):
    compiler = {
        "memory_analysis": {
            "argument_size_in_bytes": 10,
            "output_size_in_bytes": 10,
            "temp_size_in_bytes": 20,
            "alias_size_in_bytes": 5,
        },
        "cost_analysis": {"flops": 100, "bytes accessed": 200},
    }
    phases = {
        name: {
            "steady_seconds": [0.5, 0.5, 0.5],
            "tokens_per_second": 128 * (1 if name == "cached_decode" else 1024) / 0.5,
            "compiler": copy.deepcopy(compiler),
        }
        for name in ("train", "prefill_loss", "cached_decode")
    }
    phases["train"].update({"loss": 2.0, "grad_norm": 1.0})
    phases["prefill_loss"]["nll_sum"] = 100
    phases["cached_decode"].update(
        {
            "finite_logits": True,
            "prefix_context_tokens": 1024,
            "cache_capacity_tokens": 1028,
            "cache_bytes_per_device": 500,
        }
    )
    row = {
        "architecture": architecture,
        "model_config": {"architecture": architecture},
        "global_batch": 128,
        "sequence_length": 1024,
        "provenance": {"sha256": {"source": "hash"}},
        "hardware": hardware(index),
        **phases,
        "runtime_memory": [{"peak_bytes_in_use": 100, "bytes_limit": 1000}] * 4,
    }
    if architecture == "mamba4":
        row["model_config"].update({"n_layers": 1, "protected_anchor_budget": 4})
        row["memory_diagnostics"] = diagnostic_fixture(None)[0][-1]["metrics"]
        row["memory_diagnostic_execution"] = {
            "passed": True,
            "compile_seconds": 0.1,
            "seconds": 0.01,
            "compiler": copy.deepcopy(compiler),
        }
    return row


def test_all_eleven_schemas_and_numeric_error_limits():
    validate_kernel_cases(cases())
    changed = cases()
    changed[0]["forward"]["relative_l2_error"] = 0.1
    with pytest.raises(ValueError, match="exceed"):
        validate_kernel_cases(changed)
    changed = cases()
    changed[0]["gradients"].pop()
    with pytest.raises(ValueError, match="gradient coverage"):
        validate_kernel_cases(changed)
    changed = cases()[:-1]
    with pytest.raises(ValueError, match="Missing"):
        validate_kernel_cases(changed)


@pytest.mark.parametrize(
    "mutation", ["negative_margin", "nan_margin", "false_finite", "false_gradient"]
)
def test_reject_qr_failures_even_when_case_passed_is_true(mutation):
    changed = cases()
    case = next(case for case in changed if case["name"] == QR_CASES[0])
    if mutation == "negative_margin":
        case["diagnostics"]["protected_min_qr_diagonal"][0] = -0.1
    elif mutation == "nan_margin":
        case["diagnostics"]["protected_min_qr_diagonal"][0] = np.nan
    elif mutation == "false_finite":
        case["diagnostics"]["protected_all_finite"][0] = False
    else:
        case["finite_loss_and_all_key_value_query_gradients"] = False
    with pytest.raises(ValueError):
        validate_kernel_cases(changed)


@pytest.mark.parametrize("mutation", ["nan_error", "large_error", "false_eos_gradient"])
def test_reject_bad_decode_and_repeated_eos_gradients(mutation):
    changed = cases()
    case = changed[-1]
    if mutation == "nan_error":
        case["cached_decode"]["max_absolute_error"] = np.nan
    elif mutation == "large_error":
        case["cached_decode"]["max_error_over_reference_max"] = 0.9
    else:
        case["finite_repeated_eos_loss_and_all_parameter_gradients"] = False
    with pytest.raises(ValueError):
        validate_kernel_cases(changed)


def test_bench_requires_frozen_sources_mature_context_and_numeric_cost():
    row = benchmark()
    validate_benchmark(row, "mamba4", row["model_config"], {"source": "hash"})
    for mutation, message in (
        ("source", "Source hashes"),
        ("context", "mature"),
        ("memory", "memory"),
        ("cost", "FLOP"),
    ):
        changed = copy.deepcopy(row)
        if mutation == "source":
            changed["provenance"]["sha256"]["source"] = "stale"
        elif mutation == "context":
            changed["cached_decode"]["prefix_context_tokens"] = 0
        elif mutation == "memory":
            changed["runtime_memory"][0]["peak_bytes_in_use"] = 0
        else:
            changed["train"]["compiler"]["cost_analysis"]["flops"] = np.nan
        with pytest.raises(ValueError, match=message):
            validate_benchmark(
                changed, "mamba4", row["model_config"], {"source": "hash"}
            )


def test_all_three_full_model_benches_and_four_hosts_are_required(tmp_path):
    benches, kernels = tmp_path / "benches", tmp_path / "kernels"
    benches.mkdir()
    kernels.mkdir()
    models = {}
    for architecture in ("transformer", "mamba3", "mamba4"):
        models[architecture] = benchmark(architecture)["model_config"]
        for index in range(4):
            (benches / f"{architecture}-host-{index}.json").write_text(
                json.dumps(benchmark(architecture, index))
            )
    for index in range(4):
        (kernels / f"host-{index}.json").write_text(
            json.dumps(
                {
                    "hardware": hardware(index),
                    "provenance": {"sha256": {"source": "hash"}},
                    "passed": True,
                    "pallas_tpu_verified": True,
                    "cases": cases(),
                }
            )
        )
    result = validate_engineering(benches, kernels, {"source": "hash"}, models)
    assert set(result["benchmarks"]) == {"transformer", "mamba3", "mamba4"}
    assert len(result["kernel_conformance"]) == 4
    (benches / "mamba4-host-3.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_engineering(benches, kernels, {"source": "hash"}, models)


def diagnostic_fixture(tmp_path):
    metrics = {
        "precision_eigenvalue_min": 1,
        "precision_eigenvalue_max": 3,
        "precision_condition_max": 2,
        "prior_diagonal_min": 0.2,
        "prior_diagonal_max": 1,
        "gaussian_cross_abs_max": 1,
        "gaussian_allfinite": 1,
        "epsilon_min": 0.1,
        "epsilon_max": 0.2,
        "decay_min": 0.96,
        "decay_max": 0.98,
        "beta_min": 0.2,
        "beta_max": 0.5,
        "protected_qr_diagonal_min": 0.002,
        "protected_allfinite": 1,
        "protected_bank_count_max": 3,
        "protected_retained_anchors_min": 4,
        "protected_retained_anchors_max": 12,
        "protected_merge_count_max": 1,
    }
    entries = [
        {
            "event": "diagnostics",
            "step": step,
            "metrics": {
                "layer_0/memory/" + key: value for key, value in metrics.items()
            },
        }
        for step in (0, 2, 3)
    ]
    frozen = {
        "model": {"n_layers": 1, "protected_anchor_budget": 4},
        "training": {
            "token_budget": 35,
            "global_batch": 8,
            "sequence_length": 2,
            "eval_every": 2,
        },
    }
    return entries, frozen


def write_diagnostics(path, entries):
    (path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n"
    )
    (path / "diagnostics.json").write_text(json.dumps(entries[-1]))


@pytest.mark.parametrize(
    "mutation",
    ["missing_final", "negative_eigen", "nan_condition", "false_finite", "bad_qr"],
)
def test_learned_diagnostic_checks_reject_missing_or_unstable_states(
    tmp_path, mutation
):
    entries, frozen = diagnostic_fixture(tmp_path)
    write_diagnostics(tmp_path, entries)
    valid = validate_mamba4_diagnostics(tmp_path, frozen)
    assert valid["probe_steps"] == [0, 2, 3]
    if mutation == "missing_final":
        entries.pop()
    else:
        suffix, value = {
            "negative_eigen": ("precision_eigenvalue_min", -1),
            "nan_condition": ("precision_condition_max", np.nan),
            "false_finite": ("gaussian_allfinite", 0),
            "bad_qr": ("protected_qr_diagonal_min", 0),
        }[mutation]
        entries[-1]["metrics"]["layer_0/memory/" + suffix] = value
    write_diagnostics(tmp_path, entries)
    with pytest.raises(ValueError):
        validate_mamba4_diagnostics(tmp_path, frozen)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed_execution",
        "nan_execution",
        "negative_precision",
        "nan_condition",
        "bad_qr",
        "false_finite",
    ],
)
def test_mamba4_benchmark_requires_actual_finite_memory_probe(mutation):
    row = benchmark()
    if mutation == "failed_execution":
        row["memory_diagnostic_execution"]["passed"] = False
    elif mutation == "nan_execution":
        row["memory_diagnostic_execution"]["seconds"] = np.nan
    else:
        suffix, value = {
            "negative_precision": ("precision_eigenvalue_min", -1),
            "nan_condition": ("precision_condition_max", np.nan),
            "bad_qr": ("protected_qr_diagonal_min", 0),
            "false_finite": ("protected_allfinite", 0),
        }[mutation]
        row["memory_diagnostics"]["layer_0/memory/" + suffix] = value
    with pytest.raises(ValueError):
        validate_benchmark(row, "mamba4", row["model_config"], {"source": "hash"})


def test_actual_frozen_model_diagnostics_namespace_and_auditor_integration(tmp_path):
    """Generate probes through real production collection/pmap code on four CPUs."""
    child = r"""
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from lm.config import ModelConfig
from lm.models.mamba4 import Mamba4LM
from lm.train import make_diagnostic_step, unreplicate

config = ModelConfig(architecture="mamba4", vocab_size=23, d_model=8, n_layers=1,
                     d_ff=12, expand=1, head_dim=4, key_dim=4, chunk_size=3,
                     protected_anchor_budget=2, protected_block_size=3,
                     dtype="float32", max_seq_len=9, remat=False)
model = Mamba4LM(config)
tokens = jnp.full((1, 9), 2, jnp.int32)
params = model.init(jax.random.key(17), tokens)["params"]
replicated = jax.device_put_replicated(params, jax.local_devices())
mapped_tokens = jnp.broadcast_to(tokens, (jax.local_device_count(), 1, 9))
metrics = unreplicate(make_diagnostic_step(model)(replicated, mapped_tokens))
metrics = {name: float(value) for name, value in metrics.items()}
Path(sys.argv[1]).write_text(json.dumps({"model": config.to_dict(), "metrics": metrics,
                                       "cpu_devices": jax.local_device_count()}))
"""
    evidence = tmp_path / "actual-diagnostics.json"
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "OMP_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", child, str(evidence)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(evidence.read_text())
    assert actual["cpu_devices"] == 4
    assert "layer_0/memory/precision_eigenvalue_min" in actual["metrics"]
    assert not any("/mixer/" in name for name in actual["metrics"])
    entries = [
        {"event": "diagnostics", "step": step, "metrics": actual["metrics"]}
        for step in (0, 2, 3)
    ]
    frozen = {
        "model": actual["model"],
        "training": {
            "token_budget": 35,
            "global_batch": 8,
            "sequence_length": 2,
            "eval_every": 2,
        },
    }
    write_diagnostics(tmp_path, entries)
    result = validate_mamba4_diagnostics(tmp_path, frozen)
    assert result["precision_eigenvalue_min"] > 0
    assert result["protected_qr_diagonal_min"] > 0
    row = benchmark()
    row["model_config"] = actual["model"]
    row["memory_diagnostics"] = actual["metrics"]
    summary = validate_benchmark(row, "mamba4", actual["model"], {"source": "hash"})
    assert (
        summary["learned_memory_diagnostics"]["precision_eigenvalue_min"]
        == result["precision_eigenvalue_min"]
    )


def test_physical_worker_sidecar_rejects_one_actual_checksum_difference(tmp_path):
    from scripts.pod import HOSTS

    run = {
        "architecture": "transformer",
        "optimizer_steps": 7630,
        "training_targets": 1_000_000_000,
        "fingerprint": "frozen",
        "final_checkpoint_sha256": "checksum",
    }
    sidecar = {
        **{
            key: run[key]
            for key in (
                "architecture",
                "optimizer_steps",
                "training_targets",
                "fingerprint",
            )
        },
        "all_actual_payloads_equal": True,
        "workers": [{"host": host, "checkpoint_sha256": "checksum"} for host in HOSTS],
    }
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps(sidecar))
    assert validate_crosshost_checkpoint(path, run, required=True)["workers"] == 4
    sidecar["workers"][-1]["checkpoint_sha256"] = "other"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_crosshost_checkpoint(path, run, required=True)


def test_actual_frozen_kernel_reports_when_available():
    paths = list(Path("lm/results/kernel-conformance").glob("host-*.json"))
    if not paths:
        pytest.skip("Actual conformance artifacts are generated by the TPU run")
    for path in paths:
        validate_kernel_cases(json.loads(path.read_text())["cases"])
