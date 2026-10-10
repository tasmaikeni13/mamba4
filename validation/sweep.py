"""Learning-rate sweep for the 125M protocol on unseen training batches.

``configs`` writes one short-schedule configuration per architecture and
learning rate: the frozen 125M configuration with its token budget cut to
the sweep budget, so warmup and cosine decay complete within the run.
``validation.pilot`` trains each through every step and records the loss of
each batch before it updates the model. ``report`` applies the rule fixed in
docs/scaling-125m.md before any sweep run: per architecture, the lowest mean
pre-update loss over the final 100 steps wins. A run that stops on a
nonfinite loss scores infinity. A winner at the edge of the grid triggers
the next point beyond it (factor 2), at most twice, before the choice is
final. The held-out split is never opened.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lm.runtime import atomic_json

WINDOW = 100


def name_for(architecture, rate):
    return f"{architecture}-lr{rate:g}"


def write_configs(options):
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    for path in options.configs:
        frozen = json.loads(Path(path).read_text())
        architecture = frozen["model"]["architecture"]
        for rate in options.learning_rates:
            sweep = json.loads(json.dumps(frozen))
            sweep["protocol_id"] = frozen["protocol_id"] + "-sweep"
            sweep["training"]["token_budget"] = options.budget
            sweep["training"]["learning_rate"] = rate
            target = output / f"{name_for(architecture, rate)}.json"
            target.write_text(json.dumps(sweep, indent=2) + "\n")
            print(target)


def choose(runs):
    """runs: {rate: per-step losses or None}; returns (rate, scores, edge)."""
    scores = {
        rate: float("inf") if losses is None else float(np.mean(losses[-WINDOW:]))
        for rate, losses in runs.items()
    }
    rates = sorted(scores)
    best = min(rates, key=lambda rate: scores[rate])
    edge = None
    if len(rates) > 1 and best == rates[-1]:
        edge = best * 2
    elif len(rates) > 1 and best == rates[0]:
        edge = best / 2
    return best, scores, edge


def report(options):
    raw = Path(options.raw)
    grouped = {
        architecture: {rate: None for rate in options.grid}
        for architecture in options.architectures
    }
    for path in sorted(raw.glob("*.json")):
        record = json.loads(path.read_text())
        if "train_loss" not in record:
            continue
        training = record["config"]["training"]
        architecture = record["config"]["model"]["architecture"]
        steps = -(
            -training["token_budget"]
            // (training["global_batch"] * training["sequence_length"])
        )
        if record["steps"] != steps:
            raise ValueError(f"{path.name} stopped before its schedule ended")
        grouped.setdefault(architecture, {})[training["learning_rate"]] = record[
            "train_loss"
        ]
    summary = {"rule": f"lowest mean pre-update loss over the final {WINDOW} steps"}
    for architecture, runs in grouped.items():
        best, scores, edge = choose(runs)
        summary[architecture] = {
            "scores": {f"{rate:g}": score for rate, score in sorted(scores.items())},
            "selected": best,
            "extend_grid_to": edge,
            "final": edge is None,
        }
    atomic_json(Path(options.output), summary)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("configs")
    make.add_argument("--configs", nargs="+", required=True)
    make.add_argument("--learning-rates", nargs="+", type=float, required=True)
    make.add_argument("--budget", type=int, default=300_000_000)
    make.add_argument("--output", required=True)
    summarize = sub.add_parser("report")
    summarize.add_argument("--raw", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument(
        "--architectures", nargs="+", default=["transformer", "mamba4"]
    )
    summarize.add_argument("--grid", nargs="+", type=float, required=True)
    options = parser.parse_args()
    {"configs": write_configs, "report": report}[options.action](options)


if __name__ == "__main__":
    main()
