"""Report of the 125M study from the audited runs and the evaluations.

Primary outcome (declared in docs/scaling-125m.md): from the held-out NLL of
each seed's runs, a win if Mamba 4's mean is lower and all three seed
differences favour it, a loss if the mean is higher and all three favour the
Transformer, otherwise inconclusive. Paired per-sequence intervals use
whole 1,024-token sequences in circular blocks of eight, for each seed's
pair and for sequence NLL averaged over seeds, on the held-out split and on
the fresh holdout. Downstream and claims summaries are produced by their own
report commands; this file collects them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lm.runtime import atomic_json
from validation.verify_screen import SEED, block_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "lm/runs/lm-125m"
RESULTS = ROOT / "lm/results/lm-125m"
SEEDS = (42, 43, 44)
ARCHITECTURES = ("transformer", "mamba4")
NAMES = {"transformer": "Transformer", "mamba4": "Mamba 4"}
COLORS = {"transformer": "#1baf7a", "mamba4": "#2a78d6"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read(path):
    return json.loads(Path(path).read_text())


def outcome(differences):
    differences = np.asarray(differences)
    if differences.mean() < 0 and np.all(differences < 0):
        return "win"
    if differences.mean() > 0 and np.all(differences > 0):
        return "loss"
    return "inconclusive"


def runs_table():
    table = {}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            name = f"{architecture}-s{seed}"
            result = read(RUNS / name / "result.json")
            recall = result.get("recall") or {}
            table[name] = {
                "heldout_nll": result["heldout"]["nll"],
                "parameters": result["parameter_count"],
                "steady_tokens_per_second": result["steady_tokens_per_second"],
                "recall_accuracy": (recall.get("overall") or {}).get("accuracy"),
                "audited": (RESULTS / f"{name}-audit.json").exists(),
            }
    return table


def sequence_comparisons(sequence_nll):
    rng = np.random.default_rng(SEED)
    models = sequence_nll["models"]
    report = {}
    for split in ("screen_heldout", "fresh_holdout"):
        per_sequence = {}
        for name, entry in models.items():
            sums = np.asarray(entry[split]["sequence_nll_sum"], np.float64)
            counts = np.asarray(entry[split]["sequence_targets"], np.float64)
            per_sequence[name] = sums / counts
        split_report = {
            "nll": {name: float(models[name][split]["nll"]) for name in models},
            "per_seed": {},
        }
        for seed in SEEDS:
            difference = (
                per_sequence[f"mamba4-s{seed}"] - per_sequence[f"transformer-s{seed}"]
            )
            split_report["per_seed"][str(seed)] = {
                "mean": float(difference.mean()),
                "block_bootstrap_95": block_bootstrap(difference, rng),
                "sequences_mamba4_lower": int(np.sum(difference < 0)),
                "sequences": int(difference.size),
            }
        pooled = np.mean(
            [per_sequence[f"mamba4-s{s}"] for s in SEEDS], axis=0
        ) - np.mean([per_sequence[f"transformer-s{s}"] for s in SEEDS], axis=0)
        split_report["seed_averaged"] = {
            "mean": float(pooled.mean()),
            "block_bootstrap_95": block_bootstrap(pooled, rng),
            "sequences_mamba4_lower": int(np.sum(pooled < 0)),
            "sequences": int(pooled.size),
        }
        report[split] = split_report
    return report


def curves(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot, axis = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    for architecture in ARCHITECTURES:
        for index, seed in enumerate(SEEDS):
            steps, values = [], []
            log = RUNS / f"{architecture}-s{seed}" / "metrics.merged.jsonl"
            if not log.exists():
                log = RUNS / f"{architecture}-s{seed}" / "metrics.jsonl"
            for line in log.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("event") in ("initial_eval", "eval"):
                    steps.append(entry["step"])
                    values.append(entry["nll"])
            result = read(RUNS / f"{architecture}-s{seed}" / "result.json")
            steps.append(result["optimizer_steps"])
            values.append(result["heldout"]["nll"])
            axis.plot(
                steps[1:],
                values[1:],
                color=COLORS[architecture],
                linewidth=1.6,
                alpha=0.9,
                label=NAMES[architecture] if index == 0 else None,
            )
    axis.set_xlabel("Optimizer step (262,144 tokens per step)", color=MUTED)
    axis.set_ylabel("Held-out NLL (nats)", color=MUTED)
    axis.set_title(
        "Held-out NLL during training, three seeds each", color=TEXT, loc="left"
    )
    legend = axis.legend(frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT)
    plot.tight_layout()
    plot.savefig(path, dpi=160, facecolor=SURFACE)


def markdown(summary):
    runs = summary["runs"]
    lines = [
        "# 125M, 3B FineWeb-Edu tokens, three seeds",
        "",
        "Protocol: `docs/scaling-125m.md`. Learning rates from the recorded sweep "
        f"rule: Transformer {summary['learning_rates']['transformer']:g}, "
        f"Mamba 4 {summary['learning_rates']['mamba4']:g}.",
        "",
        "| Run | Parameters | Held-out NLL | Fresh-holdout NLL | Recall | Train tokens/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fresh = summary["sequence_nll"]["fresh_holdout"]["nll"]
    for name, entry in runs.items():
        recall = entry["recall_accuracy"]
        lines.append(
            f"| {name} | {entry['parameters']:,} | {entry['heldout_nll']:.4f} | "
            f"{fresh[name]:.4f} | {'—' if recall is None else f'{100 * recall:.1f}%'} | "
            f"{entry['steady_tokens_per_second']:,.0f} |"
        )
    primary = summary["primary"]
    lines += [
        "",
        f"**Primary outcome: {primary['outcome']}.** Held-out NLL, Mamba 4 minus "
        "Transformer, per seed: "
        + ", ".join(f"{d:+.4f}" for d in primary["differences"])
        + f" (mean {primary['mean_difference']:+.4f}).",
        "",
        "Paired per-sequence differences (Mamba 4 minus Transformer; 2,048 "
        "sequences, circular blocks of eight):",
        "",
        "| Split | Seed 42 | Seed 43 | Seed 44 | Seed-averaged |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, label in (
        ("screen_heldout", "Held-out"),
        ("fresh_holdout", "Fresh holdout"),
    ):
        entry = summary["sequence_nll"][split]
        cells = []
        for item in [entry["per_seed"][str(s)] for s in SEEDS] + [
            entry["seed_averaged"]
        ]:
            low, high = item["block_bootstrap_95"]
            cells.append(f"{item['mean']:+.4f} [{low:+.4f}, {high:+.4f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "![Held-out NLL during training](learning-curves.png)"]
    if summary.get("downstream_markdown"):
        lines += [
            "",
            "## Zero-shot downstream tasks (mean ± sd over seeds)",
            "",
            summary["downstream_markdown"].strip(),
        ]
    if summary.get("claims_markdown"):
        claims = summary["claims_markdown"].strip().splitlines()
        lines += ["", "## Claims suite", "", *claims[2:]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-nll",
        default=str(RUNS / "evaluations/sequence-nll/sequence-nll.json"),
    )
    options = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = runs_table()
    sequence = sequence_comparisons(read(options.sequence_nll))
    differences = [
        runs[f"mamba4-s{s}"]["heldout_nll"] - runs[f"transformer-s{s}"]["heldout_nll"]
        for s in SEEDS
    ]
    selection = read(RESULTS / "sweep-selection.json")
    summary = {
        "runs": runs,
        "learning_rates": {a: selection[a]["selected"] for a in ARCHITECTURES},
        "primary": {
            "rule": "win: mean lower and all three seed differences negative; "
            "loss: mean higher and all three positive; otherwise inconclusive",
            "differences": differences,
            "mean_difference": float(np.mean(differences)),
            "outcome": outcome(differences),
        },
        "sequence_nll": sequence,
    }
    downstream = RESULTS / "downstream/summary.md"
    if downstream.exists():
        summary["downstream_markdown"] = downstream.read_text()
    claims = RESULTS / "claims/REPORT.md"
    if claims.exists():
        summary["claims_markdown"] = claims.read_text()
    curves(RESULTS / "learning-curves.png")
    atomic_json(RESULTS / "summary.json", summary)
    (RESULTS / "REPORT.md").write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
