"""Markdown summary and figure of the in-context regression study."""

import argparse
import json
from pathlib import Path

NAMES = {"transformer": "Transformer", "mamba3": "Mamba-3", "mamba4-shift": "Mamba 4"}
COLORS = {"transformer": "#1baf7a", "mamba3": "#eb6834", "mamba4-shift": "#2a78d6"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def figure(record, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = record["runs"]
    bayes = next(iter(runs.values()))["evaluation"]["per_example_index"]
    index = list(range(1, len(bayes["bayes_mse"]) + 1))
    plot, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURFACE)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.set_xlabel("Example index within the sequence", color=MUTED)
    left.plot(
        index, bayes["bayes_mse"], "--", color=TEXT, linewidth=1.6, label="Exact Bayes"
    )
    right.plot(
        index,
        [a - a for a in bayes["bayes_nll"]],
        "--",
        color=TEXT,
        linewidth=1.6,
        label="Exact Bayes",
    )
    for name in NAMES:
        values = runs[name]["evaluation"]["per_example_index"]
        left.plot(
            index,
            values["model_mse"],
            color=COLORS[name],
            linewidth=2,
            label=NAMES[name],
        )
        excess = [m - b for m, b in zip(values["model_nll"], values["bayes_nll"])]
        right.plot(index, excess, color=COLORS[name], linewidth=2, label=NAMES[name])
    left.set_yscale("log")
    ticks = [0.3, 0.5, 1, 2, 4, 8]
    left.set_yticks(ticks, [f"{t:g}" for t in ticks])
    left.minorticks_off()
    left.set_ylabel("Squared error of the predicted mean", color=MUTED)
    left.set_title("(a) Prediction error", color=TEXT, loc="left")
    right.set_ylabel("Gaussian NLL minus exact Bayes (nats)", color=MUTED)
    right.set_title("(b) Excess predictive NLL", color=TEXT, loc="left")
    for axis in (left, right):
        legend = axis.legend(frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(TEXT)
    plot.tight_layout()
    plot.savefig(path, dpi=160, facecolor=SURFACE)


def markdown(record):
    runs = record["runs"]
    task = record["task"]
    lines = [
        "# In-context regression against exact Bayes",
        "",
        f"Each sequence draws w ~ N(0, I_{task['dim']}) and {task['examples']} "
        f"examples with y = w·x + {task['noise']}ε. At every x the model outputs a "
        "mean and a log variance for y, trained by Gaussian NLL. The exact "
        "posterior predictive under this model is the reference. Matched small "
        f"models (two blocks of width 128) train for {record['steps']:,} steps of "
        f"{record['global_batch']} sequences. The learning rate is chosen by "
        "development NLL (seed 12); scores use evaluation seed 13. The design was "
        "recorded before the run (`validation/synthetic-protocol.md`).",
        "",
        "| Model | LR | MSE (all) | MSE (late) | NLL (all) | NLL (late) | 90% coverage (late) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    bayes = next(iter(runs.values()))["evaluation"]
    for label, values in (("Exact Bayes", None),) + tuple(
        (name, runs[name]) for name in NAMES
    ):
        if values is None:
            every, late = bayes["all_examples"], bayes["late_examples"]
            prefix, rate = "bayes", "—"
        else:
            every = values["evaluation"]["all_examples"]
            late = values["evaluation"]["late_examples"]
            prefix, rate = "model", f"{values['selected_learning_rate']:g}"
            label = NAMES[label]
        lines.append(
            f"| {label} | {rate} | {every[prefix + '_mse']:.3f} | "
            f"{late[prefix + '_mse']:.3f} | {every[prefix + '_nll']:.4f} | "
            f"{late[prefix + '_nll']:.4f} | {late[prefix + '_coverage90']:.3f} |"
        )
    rows = bayes["rows"]
    lines += [
        "",
        f"Late = examples 17–32 of each sequence. Means over {rows:,} evaluation "
        "sequences from one training seed per model; the per-sequence arrays were "
        "not stored, so no interval is given. The gaps between models are large "
        "relative to the sequence-level noise of these means.",
        "",
        "![Error and excess NLL by example index](regression.png)",
    ]
    diverged = sorted(
        name
        for name, run in runs.items()
        if not all(c["finite"] for c in run["candidates"].values())
    )
    if diverged:
        lines += [
            "",
            "Nonfinite training at some learning rate: " + ", ".join(diverged),
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="regression.json")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    record = json.loads(Path(options.record).read_text())
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    figure(record, output / "regression.png")
    (output / "README.md").write_text(markdown(record))
    print(markdown(record))


if __name__ == "__main__":
    main()
