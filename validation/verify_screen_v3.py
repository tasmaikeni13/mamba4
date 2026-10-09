"""Final screen-60m-v3 audit: three completed runs, statistics and report.

Re-audits the reused Transformer/Mamba-3 runs and the v3 Mamba 4 run from
their raw logs and checkpoints, checks shared-source identity, parameter and
protocol matching, and the declared strict screen win. Per-sequence NLL from
``validation.fresh_holdout evaluate`` must reproduce every audited held-out
NLL; paired moving-block bootstrap intervals use whole 1,024-token sequences
as sampling units. One training seed cannot estimate seed variance.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from lm.data import TokenCorpus, sha256_file
from lm.recall import build_recall_dataset
from lm.runtime import atomic_json, source_provenance
from scripts.verify_lm_runs import audit_run, read_json, require, validate_data
from validation.run_screen_v2 import peer_source_identity


ROOT = Path(__file__).resolve().parents[1]
BLOCK, RESAMPLES, SEED = 8, 10_000, 20261009


def block_bootstrap(differences, rng):
    """Circular moving-block bootstrap of a mean over ordered sequences."""
    count = len(differences)
    blocks = -(-count // BLOCK)
    starts = rng.integers(0, count, size=(RESAMPLES, blocks))
    index = (starts[..., None] + np.arange(BLOCK)) % count
    means = differences[index.reshape(RESAMPLES, -1)[:, :count]].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_statistics(sequence_nll, audits):
    rng = np.random.default_rng(SEED)
    models = sequence_nll["models"]
    summary = {}
    for split in ("screen_heldout", "fresh_holdout"):
        per_sequence = {}
        for name, entry in models.items():
            sums = np.asarray(entry[split]["sequence_nll_sum"], np.float64)
            counts = np.asarray(entry[split]["sequence_targets"], np.float64)
            require(np.all(counts > 0), f"Empty evaluation sequence: {name} {split}")
            per_sequence[name] = sums / counts
            nll = float(sums.sum() / counts.sum())
            if split == "screen_heldout":
                audited = audits[name]["heldout"]["nll"]
                require(
                    abs(nll - audited) < 2e-4,
                    f"{name} sequence NLL {nll} does not reproduce audit {audited}",
                )
        rows = {name: float(values.mean()) for name, values in per_sequence.items()}
        comparisons = {}
        for peer in ("transformer", "mamba3"):
            difference = per_sequence["mamba4"] - per_sequence[peer]
            comparisons[f"mamba4_minus_{peer}"] = {
                "mean": float(difference.mean()),
                "block_bootstrap_95": block_bootstrap(difference, rng),
                "sequences_mamba4_lower": int(np.sum(difference < 0)),
                "sequences": int(difference.size),
            }
        summary[split] = {"nll": rows, "paired": comparisons}
    return summary


def matched_curves(runs):
    """Scheduled held-out NLL by step for every completed run."""
    table = {}
    for name, directory in runs.items():
        log = directory / "metrics.jsonl"
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "eval":
                table.setdefault(row["step"], {})[name] = row["nll"]
    return {step: table[step] for step in sorted(table)}


def memory_trajectory(directory):
    """Extremes of the learned memory diagnostics at every scheduled probe."""
    rows = []
    for line in (directory / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "diagnostics":
            continue
        metrics = row["metrics"]

        def extreme(suffix, reduce):
            values = [v for k, v in metrics.items() if k.endswith(suffix)]
            return reduce(values) if values else None

        rows.append(
            {
                "step": row["step"],
                "floor_min": extreme("floor_min", min),
                "precision_eigenvalue_min": extreme("precision_eigenvalue_min", min),
                "precision_condition_max": extreme("precision_condition_max", max),
                "variance_max": extreme("variance_max", max),
                "decay_max": extreme("decay_max", max),
                "allfinite": extreme("allfinite", min),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="lm/results/screen-60m-v3")
    options = parser.parse_args()
    output = ROOT / options.output
    protocol = read_json(ROOT / "lm/configs/screen-protocol-v3.json")
    corpus = TokenCorpus(ROOT / "data/fineweb-edu-1b", verify_hashes=True)
    validate_data(corpus, protocol)
    recall = build_recall_dataset(corpus.path / "tokenizer.json")
    sources = source_provenance()["sha256"]
    identity = peer_source_identity(sources)
    runs = {
        "transformer": ROOT / "lm/runs/screen-60m-v1/transformer",
        "mamba3": ROOT / "lm/runs/screen-60m-v1/mamba3",
        "mamba4": ROOT / "lm/runs/screen-60m-v3/mamba4",
    }
    configs = {
        "transformer": ROOT / "lm/configs/transformer-60m.json",
        "mamba3": ROOT / "lm/configs/mamba3-60m.json",
        "mamba4": ROOT / "lm/configs/mamba4-60m-v3.json",
    }
    audits = {}
    for name, directory in runs.items():
        recorded = read_json(directory / "manifest.json")["sources"]
        audits[name] = audit_run(
            directory,
            name,
            protocol,
            corpus,
            frozen=read_json(configs[name]),
            expected_sources=recorded if name != "mamba4" else sources,
            expected_recall=recall,
        )
    for name in ("transformer", "mamba3"):
        stored = read_json(ROOT / f"lm/results/screen-60m-v1/{name}-audit.json")
        for field in ("final_checkpoint_sha256", "fingerprint", "heldout"):
            require(stored[field] == audits[name][field], f"{name} {field} changed")
    for name in runs:
        stage = "v1" if name != "mamba4" else "v3"
        sidecar = read_json(
            ROOT / f"lm/results/screen-60m-{stage}/{name}-checkpoints.json"
        )
        require(sidecar["all_actual_payloads_equal"], f"{name} checkpoint sidecar")
        require(
            {w["checkpoint_sha256"] for w in sidecar["workers"]}
            == {audits[name]["final_checkpoint_sha256"]},
            f"{name} cross-host checkpoint hashes differ from the audit",
        )
    for field in ("total_parameters", "non_embedding_parameters"):
        counts = [audit[field] for audit in audits.values()]
        require(max(counts) / min(counts) < 1.01, f"Parameter matching failed: {field}")
    require(
        len({audit["embedding_parameters"] for audit in audits.values()}) == 1,
        "Embedding parameter counts differ",
    )
    training = [
        read_json(directory / "manifest.json")["protocol"]["training"]
        for directory in runs.values()
    ]
    require(all(t == training[0] for t in training), "Training protocols differ")
    win = all(
        audits["mamba4"]["heldout"]["nll"] < audits[peer]["heldout"]["nll"]
        for peer in ("transformer", "mamba3")
    )
    statistics = paired_statistics(read_json(output / "sequence-nll.json"), audits)
    curves = matched_curves(runs)
    trajectory = memory_trajectory(runs["mamba4"])
    require(
        all(row["allfinite"] == 1 for row in trajectory)
        and all(
            row["precision_eigenvalue_min"] >= row["floor_min"] - 1e-3
            for row in trajectory
        ),
        "A learned precision violated its floor or became nonfinite",
    )
    value = {
        "protocol": protocol,
        "matched_step_heldout_nll": curves,
        "mamba4_memory_trajectory": trajectory,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "all_requested_runs_completed": True,
        "screen_win": win,
        "runs": audits,
        "shared_source_identity": identity,
        "paired_statistics": statistics,
        "data_manifest_sha256": sha256_file(corpus.path / "manifest.json"),
        "limitation": "One training seed per model; intervals cover evaluation "
        "sequences only, not training-seed variability",
    }
    atomic_json(output / "audit.json", value)
    lines = [
        "# 60M, 1B-token, single-seed screen (protocol screen-60m-v3)",
        "",
        "| Model | Parameters | Held-out NLL | Fresh-holdout NLL | Train tokens/s"
        " | Recall accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, audit in audits.items():
        fresh = statistics["fresh_holdout"]["nll"][name]
        lines.append(
            f"| {name} | {audit['total_parameters']:,} | "
            f"{audit['heldout']['nll']:.6f} | {fresh:.6f} | "
            f"{audit['steady_tokens_per_second']:,.0f} | "
            f"{audit['recall']['overall']['accuracy']:.3f} |"
        )
    lines += ["", f"Declared Mamba 4 screen win: **{win}**.", ""]
    for split, entry in statistics.items():
        for label, row in entry["paired"].items():
            low, high = row["block_bootstrap_95"]
            lines.append(
                f"- {split}, {label}: {row['mean']:+.5f} nats "
                f"(95% block bootstrap {low:+.5f} to {high:+.5f}; "
                f"{row['sequences_mamba4_lower']}/{row['sequences']} sequences lower)"
            )
    lines += ["", "Scheduled held-out NLL at matched optimizer steps:", ""]
    lines += ["| Step | Transformer | Mamba-3 | Mamba 4 v3 |", "|---:|---:|---:|---:|"]
    for step, row in curves.items():
        cells = [f"{row[n]:.4f}" if n in row else "—" for n in runs]
        lines.append(f"| {step:,} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Learned memory at every probe: minimum precision eigenvalue "
        f"{min(r['precision_eigenvalue_min'] for r in trajectory):.4f} against "
        f"floor minimum {min(r['floor_min'] for r in trajectory):.4f}; maximum "
        f"condition {max(r['precision_condition_max'] for r in trajectory):.1f}.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
        "Peers are the audited screen-60m-v1 runs; their execution sources are",
        "byte-identical to v3 apart from Mamba 4 files and additive config fields.",
        "Intervals treat whole 1,024-token sequences in blocks of eight as units.",
        "One training seed cannot establish robustness. No 125M run was performed.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"screen_win": win, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
