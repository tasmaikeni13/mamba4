"""Four-panel figure for a claims report (reads its summary and curves only).

(a) exact-copy top-1 accuracy against the copied length; (b) passkey exact
match against prompt length, pooled over depths; (c) long-document NLL by
position, averaged in logarithmic position bins; (d) one-token decode latency
against cache context. Dashed verticals mark the 1,024-token training length.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots (light surface), fixed per entity.
STYLE = {
    "mamba4": ("Mamba 4", "#2a78d6", "-"),
    "mamba3": ("Mamba-3", "#eb6834", "-"),
    "transformer": ("Transformer", "#1baf7a", "-"),
    "mamba4_v2": ("Mamba 4 without key shift", "#eda100", "-"),
    "mamba4_no_read": ("Mamba 4, read zeroed", "#2a78d6", ":"),
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TRAINING = 1024


def _axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.axvline(TRAINING, color=MUTED, linestyle="--", linewidth=1)


def _plot(ax, name, x, y):
    label, color, dash = STYLE[name]
    ax.plot(x, y, dash, color=color, linewidth=2, marker="o", markersize=4, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="lm/results/claims-60m")
    options = parser.parse_args()
    results = Path(options.results)
    summary = json.loads((results / "summary.json").read_text())
    names = [n for n in STYLE if n in summary["evaluation"]]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="white")
    copy = summary["families"]["copy"]["groups"]
    ax = axes[0, 0]
    _axes(ax, "(a) Exact copy", "Copied length L (sequence 2L)", "Top-1 accuracy")
    lengths = sorted(int(g) for g in copy)
    for name in names:
        _plot(ax, name, lengths, [copy[str(g)][name]["mean"] for g in lengths])
    ax.set_xscale("log", base=2)
    passkey = summary["families"]["passkey"]["groups"]
    ax = axes[0, 1]
    _axes(ax, "(b) Passkey retrieval", "Prompt length (tokens)", "Exact match")
    pooled = sorted({int(g) // 10 for g in passkey})
    for name in names:
        values = [
            np.mean([e[name]["mean"] for g, e in passkey.items() if int(g) // 10 == n])
            for n in pooled
        ]
        _plot(ax, name, pooled, values)
    ax.set_xscale("log", base=2)
    curves = np.load(results / "long-context-curves.npz")
    ax = axes[1, 0]
    _axes(ax, "(c) Long documents", "Token position", "Mean NLL (nats)")
    edges = np.unique(np.geomspace(16, 16384, 25).astype(int))
    centers = np.sqrt(edges[:-1] * edges[1:])
    for name in names:
        if name not in curves.files:
            continue
        curve = curves[name]
        binned = [curve[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
        label, color, dash = STYLE[name]
        ax.plot(centers, binned, dash, color=color, linewidth=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_ylim(3.2, 4.4)
    ax.text(
        1500, 4.3, "Transformer rises off the axis", color=MUTED, fontsize=8, va="top"
    )
    ax = axes[1, 1]
    _axes(ax, "(d) One-token decode", "Cache context (tokens)", "Milliseconds")
    if summary.get("decode"):
        for name in names:
            if name not in summary["decode"]:
                continue
            entry = summary["decode"][name]["contexts"]
            contexts = sorted(int(c) for c in entry)
            values = [1000 * entry[str(c)]["median_step_seconds"] for c in contexts]
            _plot(ax, name, contexts, values)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(results / "claims.png", dpi=150)
    print(results / "claims.png")


if __name__ == "__main__":
    main()
