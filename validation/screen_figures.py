"""Learning-curve figure for the 60M screen report (reads logs only).

Panel (a): scheduled held-out NLL against optimizer step. Panel (b): the
100-step moving average of each model's training loss minus Mamba-3's loss
on the identical batch, so the shared data difficulty cancels.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Validated categorical slots (light surface), fixed per entity.
COLORS = {
    "Mamba 4 v2": "#2a78d6",
    "Mamba-3": "#eb6834",
    "Transformer": "#1baf7a",
    "Mamba 4 v1 (stopped)": "#eda100",
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def v2_curve(path):
    """Train losses (last entry per step) and held-out evaluations from a log."""
    train, held = {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "train":
            train[row["step"]] = row["loss"]
        elif row.get("event") in ("eval", "initial_eval"):
            held[row["step"]] = row["nll"]
    steps = sorted(train)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("Training log has gaps")
    return [train[s] for s in steps], held


def moving(values, width=100):
    values = np.asarray(values, np.float64)
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-log", default="lm/runs/screen-60m-v2/mamba4/metrics.jsonl"
    )
    parser.add_argument(
        "--output", default="lm/results/screen-60m-v2/learning-curves.png"
    )
    options = parser.parse_args()
    curves = json.loads(
        (ROOT / "lm/results/screen-60m-v1/learning-curves.json").read_text()
    )["curves"]
    v2_train, v2_held = v2_curve(ROOT / options.v2_log)
    finals = {}
    for name, path in (
        ("Transformer", "lm/runs/screen-60m-v1/transformer/result.json"),
        ("Mamba-3", "lm/runs/screen-60m-v1/mamba3/result.json"),
        ("Mamba 4 v2", "lm/runs/screen-60m-v2/mamba4/result.json"),
    ):
        if (ROOT / path).exists():
            result = json.loads((ROOT / path).read_text())
            finals[name] = (result["optimizer_steps"], result["heldout"]["nll"])
    series = {
        "Transformer": (
            curves["transformer"]["train_loss"],
            curves["transformer"]["heldout"],
        ),
        "Mamba-3": (curves["mamba3"]["train_loss"], curves["mamba3"]["heldout"]),
        "Mamba 4 v1 (stopped)": (
            curves["mamba4"]["train_loss"],
            curves["mamba4"]["heldout"],
        ),
        "Mamba 4 v2": (v2_train, {str(k): {"nll": v} for k, v in v2_held.items()}),
    }
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Optimizer step (1 step = 131,072 tokens)")
    for name, (_, held) in series.items():
        points = sorted((int(step), value["nll"]) for step, value in held.items())
        points = [(s, v) for s, v in points if s > 0]
        if name in finals:
            points.append(finals[name])
        if not points:
            continue
        steps, values = zip(*points)
        left.plot(
            steps,
            values,
            color=COLORS[name],
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
        # Stagger end labels whose values nearly coincide (Mamba-3 vs v2).
        lift = {"Mamba 4 v2": -10}.get(name, 0)
        left.annotate(
            f"{values[-1]:.3f}",
            (steps[-1], values[-1]),
            xytext=(6, lift),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    left.set_ylabel("Held-out NLL (nats)")
    left.set_title(
        "(a) Held-out NLL (last point: final, step 7,630)", color=TEXT, loc="left"
    )
    left.set_ylim(top=4.8)
    left.legend(frameon=False, fontsize=9)
    reference = np.asarray(series["Mamba-3"][0], np.float64)
    right.axhline(0, color=COLORS["Mamba-3"], linewidth=2)
    right.annotate(
        "Mamba-3 (reference)",
        (len(reference) * 0.6, 0),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=TEXT,
    )
    for name in ("Transformer", "Mamba 4 v1 (stopped)", "Mamba 4 v2"):
        losses = np.asarray(series[name][0], np.float64)
        difference = moving(losses - reference[: len(losses)])
        steps = np.arange(100, 100 + len(difference))
        right.plot(steps, difference, color=COLORS[name], linewidth=2, label=name)
        right.annotate(
            name,
            (steps[-1], difference[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    right.set_ylabel("Training loss minus Mamba-3 (100-step mean)")
    right.set_title(
        "(b) Paired training-loss gap on identical batches", color=TEXT, loc="left"
    )
    right.set_ylim(-0.12, 0.7)
    figure.tight_layout()
    output = ROOT / options.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor=SURFACE)
    print(output)


if __name__ == "__main__":
    main()
