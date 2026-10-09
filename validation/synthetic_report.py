"""Markdown summary of a synthetic study (reads its synthetic.json only)."""

import argparse
import json
from pathlib import Path

NAMES = {
    "transformer": "Transformer",
    "mamba3": "Mamba-3",
    "mamba4": "Mamba 4 without key shift",
    "mamba4-shift": "Mamba 4",
    "mamba4-strong-lowfloor": "Mamba 4, β ≤ 16 and floor 0.02",
}
TITLES = {
    "mqar": ("Associative recall: accuracy over all K queries", "K"),
    "unknown": ("Unknown keys: accuracy; half the queries were never stored", "K"),
    "noisy": ("Retention among distractors: 8 pairs, trained at 512 tokens", "Length"),
    "hops": ("Successor lookups: one hop / two hops", "K"),
}


def table(runs, task):
    title, unit = TITLES[task]
    rows = [(n, runs[f"{n}/{task}"]) for n in NAMES if f"{n}/{task}" in runs]
    if not rows:
        return []
    groups = sorted(rows[0][1]["evaluation"], key=int)
    lines = ["", f"## {title}", ""]
    lines.append("| Model | " + " | ".join(f"{unit}={g}" for g in groups) + " | LR |")
    lines.append("|---" + "|---:" * (len(groups) + 1) + "|")
    for name, run in rows:
        cells = []
        for group in groups:
            entry = run["evaluation"][group]
            if task == "hops":
                cells.append(
                    f"{100 * entry['hop1_accuracy']:.1f}% / "
                    f"{100 * entry['hop2_accuracy']:.1f}%"
                )
            elif task == "unknown":
                cells.append(
                    f"{100 * entry['known_accuracy']:.1f}% / "
                    f"{100 * entry['unknown_accuracy']:.1f}%"
                )
            else:
                cells.append(f"{100 * entry['accuracy']:.1f}%")
        lines.append(
            f"| {NAMES[name]} | "
            + " | ".join(cells)
            + f" | {run['selected_learning_rate']} |"
        )
    if task == "unknown":
        lines += ["", "Cells: accuracy on stored keys / on never-stored keys (NONE)."]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="directory holding synthetic.json")
    parser.add_argument("--title", default="Synthetic study")
    parser.add_argument("--note", default="")
    options = parser.parse_args()
    directory = Path(options.results)
    record = json.loads((directory / "synthetic.json").read_text())
    runs = record["runs"]
    lines = [f"# {options.title}", ""]
    lines += [
        f"Matched small models trained from scratch for {record['steps']:,} steps of "
        f"{record['global_batch']} sequences at 512 tokens. The learning rate is "
        "chosen on a development seed; accuracy uses a separate evaluation seed. "
        "One training seed per model."
    ]
    if options.note:
        lines += ["", options.note]
    state = {}
    for key, run in runs.items():
        state.setdefault(key.split("/")[0], run["state_floats_at_512"])
    lines += [
        "",
        "Per-sequence state at 512 tokens (floats): "
        + "; ".join(f"{NAMES.get(n, n)} {v:,}" for n, v in state.items())
        + ".",
    ]
    for task in TITLES:
        lines += table(runs, task)
    diverged = sorted(
        key
        for key, run in runs.items()
        if not all(c["finite"] for c in run["candidates"].values())
    )
    if diverged:
        lines += [
            "",
            "Non-finite training at some learning rate: " + ", ".join(diverged),
        ]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
