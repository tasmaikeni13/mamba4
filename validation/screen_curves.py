"""Preserve screen learning curves and interrupted-run evidence.

Training logs live in ignored run directories. This module copies their
per-step training losses and scheduled held-out evaluations into a compact
committed record, and documents a run that stopped before its token budget.
It never opens TPU devices or changes execution sources.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from lm.runtime import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def read_log(path):
    """Return train/eval/diagnostic rows; resumed duplicate steps keep the last."""
    train, evals, diagnostics = {}, {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        event = row.get("event")
        if event == "train":
            train[row["step"]] = row
        elif event in ("eval", "initial_eval"):
            evals[row["step"]] = row
        elif event == "diagnostics":
            diagnostics[row["step"]] = row
    steps = sorted(train)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"Training steps are not contiguous in {path}")
    return train, evals, diagnostics


def curve(path):
    train, evals, _ = read_log(path)
    steps = sorted(train)
    losses = [train[step]["loss"] for step in steps]
    if not np.all(np.isfinite(losses)):
        raise ValueError(f"Nonfinite training loss in {path}")
    return {
        "logged_steps": len(steps),
        "training_targets": train[steps[-1]]["training_targets"],
        "train_loss": [round(value, 6) for value in losses],
        "median_step_seconds": float(
            np.median([train[step]["step_seconds"] for step in steps[1:]])
        ),
        "heldout": {
            str(step): {"nll": row["nll"], "targets": row["targets"]}
            for step, row in sorted(evals.items())
        },
    }


def window(losses, end, width=100):
    values = losses[max(0, end - width) : end]
    return float(np.mean(values)) if len(values) == width else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="lm/runs/screen-60m-v1")
    parser.add_argument("--output", default="lm/results/screen-60m-v1")
    parser.add_argument("--interruption", type=Path)
    options = parser.parse_args()
    runs = ROOT / options.runs
    curves = {
        name: curve(runs / name / "metrics.jsonl")
        for name in ("transformer", "mamba3", "mamba4")
        if (runs / name / "metrics.jsonl").exists()
    }
    windows = {}
    for end in (500, 1000):
        windows[str(end)] = {
            name: window(item["train_loss"], end) for name, item in curves.items()
        }
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Copied per-step training losses and scheduled held-out NLL",
        "matched_step_mean_train_loss_previous_100_steps": windows,
        "curves": curves,
    }
    output = ROOT / options.output
    atomic_json(output / "learning-curves.json", record)
    if options.interruption:
        details = json.loads(options.interruption.read_text())
        mamba4 = curves["mamba4"]
        details["logged_steps"] = mamba4["logged_steps"]
        details["logged_training_targets"] = mamba4["training_targets"]
        details["median_step_seconds"] = mamba4["median_step_seconds"]
        details["heldout"] = mamba4["heldout"]
        details["matched_step_heldout_nll"] = {
            step: {
                name: item["heldout"].get(step, {}).get("nll")
                for name, item in curves.items()
            }
            for step in ("500", "1000")
        }
        details["matched_step_mean_train_loss_previous_100_steps"] = windows
        atomic_json(output / "mamba4-interrupted.json", details)


if __name__ == "__main__":
    main()
