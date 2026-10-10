"""Guarded 125M runs: train each (architecture, seed) to completion and audit it.

Runs go in the declared order (seed 42 Transformer, seed 42 Mamba 4, then
seeds 43 and 44), each through scripts.pod from its frozen configuration in
lm/configs/125m/. Before each attempt every host's compilation cache is
cleared. An interrupted run resumes from its latest durable checkpoint
(lm.train verifies the fingerprint). Two consecutive failures without
checkpoint progress stop the chain. A completed run is audited against
lm/configs/protocol-125m.json and its final checkpoint is hashed on every
host. Every stage is recorded in status.json.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from lm.data import TokenCorpus, sha256_file
from lm.recall import build_recall_dataset
from lm.runtime import atomic_json, source_provenance
from scripts.pod import HOSTS, ROOT
from scripts.verify_lm_runs import audit_run, read_json, require, validate_data
from validation.run_screen import clear_caches, remote, wait_for_hosts

RUNS = ROOT / "lm/runs/lm-125m"
EVIDENCE = ROOT / "lm/results/lm-125m"
CONFIGS = ROOT / "lm/configs/125m"
PROTOCOL = ROOT / "lm/configs/protocol-125m.json"
DATA = "data/fineweb-edu-3b"
STATUS = RUNS / "status.json"
ORDER = [
    (architecture, seed)
    for seed in (42, 43, 44)
    for architecture in ("transformer", "mamba4")
]


def record(stage, **details):
    history = read_json(STATUS)["history"] if STATUS.exists() else []
    entry = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **details}
    history.append(entry)
    atomic_json(STATUS, {"controller_pid": os.getpid(), **entry, "history": history})
    print(json.dumps(entry), flush=True)


class Run:
    def __init__(self, architecture, seed):
        self.name = f"{architecture}-s{seed}"
        self.architecture = architecture
        self.config = CONFIGS / f"{self.name}.json"
        self.directory = RUNS / self.name
        self.log = self.directory / "metrics.jsonl"
        self.merged = self.directory / "metrics.merged.jsonl"

    def _write_log(self, lines):
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in (self.merged, self.log):
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(line + "\n" for line in lines))
            temporary.replace(path)

    def prepare_logs(self):
        """Restore the master log locally and remove remote copies, so a
        remote JAX process 0 starts a file holding only its new lines."""
        if self.merged.exists():
            self._write_log(_lines(self.merged.read_text()))
        for host in HOSTS[1:]:
            remote(host, "rm -f " + shlex.quote(str(self.log)))

    def absorb_logs(self):
        merged = _lines(self.merged.read_text()) if self.merged.exists() else []
        candidates = [_lines(self.log.read_text()) if self.log.exists() else []]
        for host in HOSTS[1:]:
            answer = remote(
                host, "cat " + shlex.quote(str(self.log)) + " 2>/dev/null", 300
            )
            candidates.append(_lines(answer.stdout) if answer.returncode == 0 else [])
        new = []
        for lines in candidates:
            suffix = lines[len(merged) :] if lines[: len(merged)] == merged else lines
            if len(suffix) > len(new):
                new = suffix
        self._write_log(merged + new)
        return len(new)

    def checkpoint_step(self):
        pointer = self.directory / "latest.json"
        if not pointer.exists():
            return 0
        path = self.directory / read_json(pointer)["checkpoint"] / "metadata.json"
        return read_json(path)["step"]

    def completed(self):
        result = self.directory / "result.json"
        return result.exists() and read_json(result)["status"] == "completed"


def _lines(text):
    return [line for line in text.splitlines() if line.strip()]


def train(run, max_attempts):
    stalled = 0
    for attempt in range(1, max_attempts + 1):
        wait_for_hosts()
        clear_caches()
        before = run.checkpoint_step()
        tag = f"train-{run.name}-a{attempt}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.pod",
            "run",
            "--tag",
            tag,
            "lm.train",
            "--config",
            str(run.config.relative_to(ROOT)),
            "--data",
            DATA,
            "--output",
            str(run.directory),
        ]
        environment = os.environ.copy()
        environment.pop("JAX_PLATFORMS", None)
        run.prepare_logs()
        record("training", run=run.name, attempt=attempt, resume_step=before)
        code = subprocess.run(command, cwd=ROOT, env=environment).returncode
        record("logs_absorbed", run=run.name, new_lines=run.absorb_logs())
        if code == 0 and run.completed():
            record("trained", run=run.name, attempt=attempt)
            return
        after = run.checkpoint_step()
        record("attempt_failed", run=run.name, exit=code, checkpoint_step=after)
        stalled = stalled + 1 if after == before else 0
        require(stalled < 2, f"{run.name}: two failures without progress; inspect")
        time.sleep(120)
    raise RuntimeError(f"{run.name} did not complete within the attempt budget")


def checkpoint_sidecar(run, audited):
    pointer = read_json(run.directory / "latest.json")["checkpoint"]
    payload = run.directory / pointer / "state.msgpack"
    workers = [{"host": HOSTS[0], "checkpoint_sha256": sha256_file(payload)}]
    for host in HOSTS[1:]:
        answer = remote(host, "sha256sum -- " + shlex.quote(str(payload)), 300)
        require(answer.returncode == 0, f"Checkpoint read failed on {host}")
        workers.append({"host": host, "checkpoint_sha256": answer.stdout.split()[0]})
    require(
        all(
            w["checkpoint_sha256"] == audited["final_checkpoint_sha256"]
            for w in workers
        ),
        "Final checkpoint payloads differ across physical workers",
    )
    atomic_json(
        EVIDENCE / f"{run.name}-checkpoints.json",
        {
            "run": run.name,
            "protocol": "lm-125m",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Independent SHA256 of actual final state.msgpack on each worker",
            "optimizer_steps": audited["optimizer_steps"],
            "training_targets": audited["training_targets"],
            "fingerprint": audited["fingerprint"],
            "all_actual_payloads_equal": True,
            "workers": workers,
        },
    )


def prune_checkpoints(run):
    """Keep only the final checkpoint on every host once the run is audited."""
    pointer = read_json(run.directory / "latest.json")["checkpoint"]
    keep = shlex.quote(Path(pointer).name)
    directory = shlex.quote(str(run.directory / "checkpoints"))
    command = (
        f"cd {directory} && for d in step-*; do "
        f'[ "$d" = {keep} ] || rm -rf -- "$d"; done'
    )
    subprocess.run(command, shell=True, check=True)
    for host in HOSTS[1:]:
        require(remote(host, command).returncode == 0, f"Prune failed on {host}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--runs", nargs="*", help="subset, e.g. transformer-s42")
    options = parser.parse_args()
    os.chdir(ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    sources = source_provenance()["sha256"]
    protocol = read_json(PROTOCOL)
    corpus = TokenCorpus(ROOT / DATA, verify_hashes=True)
    validate_data(corpus, protocol)
    recall = build_recall_dataset(corpus.path / "tokenizer.json")
    record("started", sources=len(sources))
    runs = [Run(*item) for item in ORDER]
    if options.runs:
        runs = [run for run in runs if run.name in options.runs]
    try:
        for run in runs:
            if not run.completed():
                train(run, options.max_attempts)
            # An audit failure is recorded and the remaining runs continue; the
            # audit can be repeated later from the kept checkpoints.
            try:
                require(
                    source_provenance()["sha256"] == sources,
                    "Execution sources changed",
                )
                audited = audit_run(
                    run.directory,
                    run.architecture,
                    protocol,
                    corpus,
                    frozen=read_json(run.config),
                    expected_sources=sources,
                    expected_recall=recall,
                    parameter_target=read_json(run.config)["training"][
                        "parameter_target"
                    ],
                )
                atomic_json(EVIDENCE / f"{run.name}-audit.json", audited)
                checkpoint_sidecar(run, audited)
                prune_checkpoints(run)
                record("audited", run=run.name, heldout=audited["heldout"])
            except Exception as error:
                record("audit_failed", run=run.name, error=repr(error))
    except BaseException as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
