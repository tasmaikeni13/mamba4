"""Learning-curve figure for the 60M screen report (reads committed curves).

Panel (a): scheduled held-out NLL against optimizer step; the last point of
each series is the audited final NLL. Panel (b): the 100-step moving average
of each model's training loss minus Mamba-3's loss on the identical batch, so
the shared data difficulty cancels.
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
STYLE = {
    "transformer": ("Transformer", "#1baf7a", "-"),
    "mamba3": ("Mamba-3", "#eb6834", "-"),
    "mamba4": ("Mamba 4", "#2a78d6", "-"),
    "mamba4_no_shift": ("Mamba 4 without key shift (ablation)", "#2a78d6", ":"),
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def moving(values, width=100):
    return np.convolve(np.asarray(values, np.float64), np.ones(width) / width, "valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves", default="lm/results/screen-60m/learning-curves.json"
    )
    parser.add_argument("--output", default="lm/results/screen-60m/learning-curves.png")
    options = parser.parse_args()
    curves = json.loads((ROOT / options.curves).read_text())["curves"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.set_xlabel("Optimizer step (1 step = 131,072 tokens)", color=MUTED)
    for name, (label, color, dash) in STYLE.items():
        if name not in curves:
            continue
        points = sorted(
            (int(step), value["nll"])
            for step, value in curves[name]["heldout"].items()
            if int(step) > 0
        )
        steps, values = zip(*points)
        left.plot(steps, values, dash, color=color, linewidth=2, label=label)
        if dash == "-":
            left.annotate(
                f"{values[-1]:.4f}",
                (steps[-1], values[-1]),
                xytext=(6, {"mamba3": 6, "mamba4": -6}.get(name, 0)),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT,
            )
    left.set_ylabel("Held-out NLL (nats)", color=MUTED)
    left.set_title("(a) Held-out NLL; last point is final", color=TEXT, loc="left")
    left.set_ylim(3.4, 4.8)
    left.legend(frameon=False, fontsize=8)
    reference = np.asarray(curves["mamba3"]["train_loss"], np.float64)
    right.axhline(0, color=STYLE["mamba3"][1], linewidth=2, label="Mamba-3 (reference)")
    for name in ("transformer", "mamba4", "mamba4_no_shift"):
        if name not in curves:
            continue
        label, color, dash = STYLE[name]
        losses = np.asarray(curves[name]["train_loss"], np.float64)
        difference = moving(losses - reference[: len(losses)])
        steps = np.arange(100, 100 + len(difference))
        right.plot(steps, difference, dash, color=color, linewidth=2, label=label)
    right.set_ylabel("Training loss minus Mamba-3 (100-step mean)", color=MUTED)
    right.set_title("(b) Paired gap on identical batches", color=TEXT, loc="left")
    right.set_ylim(-0.08, 0.3)
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output = ROOT / options.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor=SURFACE)
    print(output)


if __name__ == "__main__":
    main()
