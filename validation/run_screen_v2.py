"""Guarded screen-60m-v2 Mamba 4 run: train, resume after preemption, audit.

Only this controller launches the full v2 run, always through scripts.pod.
Before each attempt every host's persistent compilation cache is cleared, so
no host can execute a cached program while the others compile a fresh one.
After an external interruption the next attempt resumes from the latest
durable checkpoint (lm.train verifies the fingerprint). Two consecutive
failures without checkpoint progress stop the chain for inspection. Every
attempt, failure and audit is recorded in ``status.json``.
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
from scripts.pod import HOSTS, ROOT, SSH
from scripts.verify_lm_runs import audit_run, read_json, require, validate_data


RUNS = ROOT / "lm/runs/screen-60m-v2"
EVIDENCE = ROOT / "lm/results/screen-60m-v2"
CONFIG = ROOT / "lm/configs/mamba4-60m-v2.json"
PROTOCOL = ROOT / "lm/configs/screen-protocol-v2.json"
STATUS = RUNS / "status.json"
LOG = RUNS / "mamba4/metrics.jsonl"
MERGED = RUNS / "mamba4/metrics.merged.jsonl"


def record(stage, **details):
    history = read_json(STATUS)["history"] if STATUS.exists() else []
    entry = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **details}
    history.append(entry)
    atomic_json(STATUS, {"controller_pid": os.getpid(), **entry, "history": history})
    print(json.dumps(entry), flush=True)


def remote(host, command, timeout=60):
    return subprocess.run(
        SSH + [f"tasma@{host}", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def wait_for_hosts():
    """Block until every worker answers and has its four TPU devices."""
    while True:
        ready = True
        for host in HOSTS[1:]:
            try:
                answer = remote(host, "ls /dev/accel* | wc -l", timeout=30)
                ready &= answer.returncode == 0 and answer.stdout.strip() == "4"
            except subprocess.TimeoutExpired:
                ready = False
        ready &= len(list(Path("/dev").glob("accel*"))) == 4
        if ready:
            return
        time.sleep(60)


def clear_caches():
    command = f"rm -rf {shlex.quote(str(ROOT / '.jax_cache'))}/*"
    subprocess.run(command, shell=True, check=True)
    for host in HOSTS[1:]:
        require(remote(host, command).returncode == 0, f"Cache clear failed: {host}")


def _lines(text):
    return [line for line in text.splitlines() if line.strip()]


def _write_log(lines):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for path in (MERGED, LOG):
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(line + "\n" for line in lines))
        temporary.replace(path)


def prepare_logs():
    """JAX process 0 writes metrics.jsonl and may move host after a reboot.

    Before an attempt the master log is restored locally and remote copies are
    removed, so a remote process 0 starts a file holding only its new lines.
    """
    if MERGED.exists():
        _write_log(_lines(MERGED.read_text()))
    for host in HOSTS[1:]:
        remote(host, "rm -f " + shlex.quote(str(LOG)))


def absorb_logs():
    """Append exactly the lines this attempt produced, wherever they were written."""
    merged = _lines(MERGED.read_text()) if MERGED.exists() else []
    candidates = [_lines(LOG.read_text()) if LOG.exists() else []]
    for host in HOSTS[1:]:
        answer = remote(host, "cat " + shlex.quote(str(LOG)) + " 2>/dev/null", 300)
        candidates.append(_lines(answer.stdout) if answer.returncode == 0 else [])
    new = []
    for lines in candidates:
        suffix = lines[len(merged) :] if lines[: len(merged)] == merged else lines
        if len(suffix) > len(new):
            new = suffix
    _write_log(merged + new)
    return len(new)


def checkpoint_step():
    pointer = RUNS / "mamba4/latest.json"
    if not pointer.exists():
        return 0
    path = RUNS / "mamba4" / read_json(pointer)["checkpoint"] / "metadata.json"
    return read_json(path)["step"]


def train(max_attempts):
    stalled = 0
    for attempt in range(1, max_attempts + 1):
        wait_for_hosts()
        clear_caches()
        before = checkpoint_step()
        tag = f"train-mamba4-60m-v2-a{attempt}"
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
            str(CONFIG.relative_to(ROOT)),
            "--data",
            "data/fineweb-edu-1b",
            "--output",
            str(RUNS / "mamba4"),
        ]
        environment = os.environ.copy()
        environment.pop("JAX_PLATFORMS", None)
        prepare_logs()
        record("training", attempt=attempt, tag=tag, resume_step=before)
        code = subprocess.run(command, cwd=ROOT, env=environment).returncode
        record("logs_absorbed", attempt=attempt, new_lines=absorb_logs())
        result = RUNS / "mamba4/result.json"
        if code == 0 and result.exists() and read_json(result)["status"] == "completed":
            record("trained", attempt=attempt)
            return
        after = checkpoint_step()
        record("attempt_failed", attempt=attempt, exit=code, checkpoint_step=after)
        stalled = stalled + 1 if after == before else 0
        require(stalled < 2, "Two failures without checkpoint progress; inspect")
        time.sleep(120)
    raise RuntimeError("Training did not complete within the attempt budget")


def checkpoint_sidecar(audited):
    pointer = read_json(RUNS / "mamba4/latest.json")["checkpoint"]
    payload = RUNS / "mamba4" / pointer / "state.msgpack"
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
        EVIDENCE / "mamba4-checkpoints.json",
        {
            "architecture": "mamba4",
            "protocol": "screen-60m-v2",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Independent SHA256 of actual final state.msgpack on each worker",
            "optimizer_steps": audited["optimizer_steps"],
            "training_targets": audited["training_targets"],
            "fingerprint": audited["fingerprint"],
            "all_actual_payloads_equal": True,
            "workers": workers,
        },
    )


def pod(module, arguments, tag):
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    wait_for_hosts()
    clear_caches()
    command = [sys.executable, "-u", "-m", "scripts.pod", "run", "--tag", tag]
    code = subprocess.run(command + [module, *arguments], cwd=ROOT, env=environment)
    require(code.returncode == 0, f"Pod job failed: {tag}")


# Files whose behavior defines the reused peers or the shared training loop.
SHARED_PREFIXES = ("lm/kernels/mamba4", "lm/models/mamba4", "lm/tests/", "lm/configs/")
SHARED_EXCEPTIONS = {"lm/config.py"}


def peer_source_identity(sources):
    """Every peer/shared execution file must equal its completed-run hash."""
    report = {}
    for architecture in ("transformer", "mamba3"):
        recorded = read_json(
            ROOT / f"lm/runs/screen-60m-v1/{architecture}/manifest.json"
        )
        differing = sorted(
            name
            for name, digest in recorded["sources"].items()
            if sources.get(name) != digest
            and not name.startswith(SHARED_PREFIXES)
            and name not in SHARED_EXCEPTIONS
        )
        require(not differing, f"Shared or peer sources changed: {differing}")
        for name in (f"lm/configs/{architecture}-60m.json",):
            require(sources[name] == recorded["sources"][name], f"{name} changed")
        report[architecture] = "identical apart from Mamba 4 files and config.py"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=6)
    options = parser.parse_args()
    os.chdir(ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    sources = source_provenance()["sha256"]
    record("started", config_sha256=sha256_file(CONFIG), config=str(CONFIG))
    try:
        identity = peer_source_identity(sources)
        record("peer_sources_checked", identity=identity)
        if not (RUNS / "mamba4/latest.json").exists():
            # Deploy exactly once; a resume must run byte-identical sources.
            wait_for_hosts()
            subprocess.run(
                [sys.executable, "-m", "scripts.pod", "sync"], cwd=ROOT, check=True
            )
            record("synced")
        train(options.max_attempts)
        require(source_provenance()["sha256"] == sources, "Execution sources changed")
        protocol = read_json(PROTOCOL)
        corpus = TokenCorpus(ROOT / "data/fineweb-edu-1b", verify_hashes=True)
        validate_data(corpus, protocol)
        recall = build_recall_dataset(corpus.path / "tokenizer.json")
        audited = audit_run(
            RUNS / "mamba4",
            "mamba4",
            protocol,
            corpus,
            frozen=read_json(CONFIG),
            expected_sources=sources,
            expected_recall=recall,
        )
        atomic_json(EVIDENCE / "mamba4-audit.json", audited)
        checkpoint_sidecar(audited)
        record("audited", heldout=audited["heldout"])
        v1 = "lm/runs/screen-60m-v1"
        pod(
            "validation.fresh_holdout",
            [
                "evaluate",
                "--output",
                str(EVIDENCE),
                f"transformer=lm/configs/transformer-60m.json,{v1}/transformer",
                f"mamba3=lm/configs/mamba3-60m.json,{v1}/mamba3",
                f"mamba4=lm/configs/mamba4-60m-v2.json,{RUNS}/mamba4",
            ],
            "eval-sequence-nll-v2",
        )
        record("evaluated", evidence=str(EVIDENCE / "sequence-nll.json"))
    except BaseException as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
