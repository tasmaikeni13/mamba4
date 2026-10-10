"""Unattended 125M pipeline: sweeps, main configurations, runs, evaluations.

1. Learning-rate sweeps under the rule in docs/scaling-125m.md, including
   the rule-triggered grid extensions, for each architecture not yet final.
2. Per-seed main configurations with the selected rates, committed before
   any main-run step.
3. The six runs and their audits (validation.run_125m).
4. Evaluations of the final checkpoints: sequence NLL on the held-out split
   and the fresh holdout, zero-shot downstream tasks and the claims suite.

Every stage is recorded in lm/runs/lm-125m/pipeline.json, and a rerun skips
completed stages. Only the pod controller's own jobs touch the TPUs.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from lm.runtime import atomic_json
from scripts.pod import HOSTS, ROOT
from validation import sweep
from validation.run_125m import ORDER
from validation.run_screen import clear_caches, remote, wait_for_hosts

RECORD = ROOT / "lm/runs/lm-125m/pipeline.json"
SWEEP_RAW = ROOT / "lm/runs/sweep-125m"
SWEEP_CONFIGS = ROOT / "lm/configs/sweep-125m"
MAIN_CONFIGS = ROOT / "lm/configs/125m"
RESULTS = ROOT / "lm/results/lm-125m"
EVALUATIONS = ROOT / "lm/runs/lm-125m/evaluations"
GRID = (0.0006, 0.0012, 0.0024)
BASES = {
    "transformer": ROOT / "lm/configs/transformer-125m.json",
    "mamba4": ROOT / "lm/configs/mamba4-125m.json",
}


def load_record():
    return json.loads(RECORD.read_text()) if RECORD.exists() else {"stages": {}}


def save(record, stage, **details):
    record["stages"][stage] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    atomic_json(RECORD, record)
    print(json.dumps({stage: record["stages"][stage]}), flush=True)


def run(command):
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


def sync():
    wait_for_hosts()
    if run([sys.executable, "-m", "scripts.pod", "sync", "--data"]):
        raise RuntimeError("pod sync failed")


def sweep_runs(architecture, rates):
    runs = {}
    for rate in rates:
        path = SWEEP_RAW / f"{sweep.name_for(architecture, rate)}.json"
        runs[rate] = None
        if path.exists():
            record = json.loads(path.read_text())
            training = record["config"]["training"]
            if record["steps"] == sweep._steps(training):
                runs[rate] = record["train_loss"]
    return runs


def select_rate(architecture):
    """Run the grid, then at most two rule-triggered extensions."""
    rates, attempted = list(GRID), set()
    for _ in range(3):
        missing = [
            rate
            for rate in rates
            if rate not in attempted
            and not (SWEEP_RAW / f"{sweep.name_for(architecture, rate)}.json").exists()
        ]
        attempted.update(rates)
        if missing:
            sweep.write_configs(
                SimpleNamespace(
                    configs=[str(BASES[architecture])],
                    learning_rates=missing,
                    budget=300_000_000,
                    output=str(SWEEP_CONFIGS),
                )
            )
            sync()
            configs = [
                str(SWEEP_CONFIGS / f"{sweep.name_for(architecture, rate)}.json")
                for rate in missing
            ]
            sweep.launch(
                SimpleNamespace(
                    configs=configs, data="data/fineweb-edu-3b", raw=str(SWEEP_RAW)
                )
            )
        best, scores, edge = sweep.choose(sweep_runs(architecture, rates))
        if edge is None:
            return best, scores, rates
        rates.append(edge)
    best, scores, _ = sweep.choose(sweep_runs(architecture, rates))
    return best, scores, rates


def write_main_configs(selected):
    MAIN_CONFIGS.mkdir(parents=True, exist_ok=True)
    for architecture, seed in ORDER:
        config = json.loads(BASES[architecture].read_text())
        config["training"]["learning_rate"] = selected[architecture]
        config["training"]["seed"] = seed
        path = MAIN_CONFIGS / f"{architecture}-s{seed}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")


def commit(message, paths):
    subprocess.run(["git", "add", *map(str, paths)], cwd=ROOT, check=True)
    body = message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    subprocess.run(["git", "commit", "-q", "-m", body], cwd=ROOT, check=False)


def pod_job(tag, module, arguments, output):
    for host in HOSTS[1:]:
        remote(host, f"mkdir -p {ROOT / output}")
    (ROOT / output).mkdir(parents=True, exist_ok=True)
    wait_for_hosts()
    clear_caches()
    command = [sys.executable, "-u", "-m", "scripts.pod", "run", "--tag", tag]
    return run(command + [module, *arguments])


def models():
    entries = []
    for architecture, seed in ORDER:
        name = f"{architecture}-s{seed}"
        config = MAIN_CONFIGS / f"{name}.json"
        directory = ROOT / "lm/runs/lm-125m" / name
        entries.append(
            f"{name}={config.relative_to(ROOT)},{directory.relative_to(ROOT)}"
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", default="")
    options = parser.parse_args()
    os.chdir(ROOT)
    record = load_record()
    stages = record["stages"]
    if "selection" not in stages:
        selected, details = {}, {}
        for architecture in BASES:
            best, scores, rates = select_rate(architecture)
            selected[architecture] = best
            details[architecture] = {
                "selected": best,
                "rates": rates,
                "scores": {f"{rate:g}": value for rate, value in scores.items()},
            }
        RESULTS.mkdir(parents=True, exist_ok=True)
        atomic_json(
            RESULTS / "sweep-selection.json",
            {"rule": "lowest mean pre-update loss over the final 100 steps", **details},
        )
        write_main_configs(selected)
        commit(
            "125M learning rates selected by the recorded sweep rule; main configurations",
            [MAIN_CONFIGS, SWEEP_CONFIGS, RESULTS / "sweep-selection.json"],
        )
        sync()
        save(record, "selection", selected=selected)
    if options.stop_after == "selection":
        return
    if "runs" not in stages:
        code = run([sys.executable, "-u", "-m", "validation.run_125m"])
        if code:
            save(record, "runs_failed", exit=code)
            raise SystemExit(code)
        save(record, "runs")
    entries = models()
    evaluations = {
        "sequence_nll": (
            "validation.fresh_holdout",
            [
                "evaluate",
                "--screen-data",
                "data/fineweb-edu-3b",
                "--fresh-data",
                "data/fresh-holdout",
                "--output",
                str((EVALUATIONS / "sequence-nll").relative_to(ROOT)),
                *entries,
            ],
            EVALUATIONS / "sequence-nll",
        ),
        "downstream": (
            "validation.downstream",
            [
                "evaluate",
                "--output",
                str((EVALUATIONS / "downstream").relative_to(ROOT)),
                *entries,
            ],
            EVALUATIONS / "downstream",
        ),
        "claims": (
            "validation.claims",
            [
                "evaluate",
                "--tasks",
                "data/claims",
                "--output",
                str((EVALUATIONS / "claims").relative_to(ROOT)),
                *entries,
            ],
            EVALUATIONS / "claims",
        ),
        "claims_decode": (
            "validation.claims",
            [
                "decode",
                "--tasks",
                "data/claims",
                "--output",
                str((EVALUATIONS / "claims").relative_to(ROOT)),
                *entries,
            ],
            EVALUATIONS / "claims",
        ),
    }
    for stage, (module, arguments, output) in evaluations.items():
        if stage in stages and stages[stage].get("exit") == 0:
            continue
        code = pod_job(
            f"eval-125m-{stage}", module, arguments, output.relative_to(ROOT)
        )
        save(record, stage, exit=code)


if __name__ == "__main__":
    main()
