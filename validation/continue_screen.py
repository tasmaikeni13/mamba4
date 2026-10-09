"""Guarded continuation of the existing single-seed 60M TPU screen.

This controller keeps the frozen execution files intact and never restarts an
active run. A failed audit or job stops the chain for inspection.
"""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

from lm.data import TokenCorpus, sha256_file
from lm.recall import build_recall_dataset
from lm.runtime import atomic_json, source_provenance
from scripts.pod import HOSTS, ROOT, SSH
from scripts.verify_lm_runs import audit_run, read_json, require, validate_data
from validation.verify_screen import validate_benchmark, validate_hardware


RUNS = ROOT / "lm/runs/screen-60m-v1"
EVIDENCE = ROOT / "lm/results/screen-60m-v1"
BENCHMARKS = ROOT / "lm/results/benchmarks-frozen-v1"
STATUS = RUNS / "continuation.json"


def update(stage, **details):
    value = {
        "stage": stage,
        "controller_pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "orchestrator_sha256": sha256_file(__file__),
        **details,
    }
    atomic_json(STATUS, value)
    print(json.dumps(value), flush=True)


def process_live(pid, module, tag):
    """Check the actual process, including zombie state and PID reuse."""
    path = Path(f"/proc/{pid}")
    try:
        state = (path / "stat").read_text().split(") ", 1)[1][0]
        if state == "Z":
            return False
        arguments = (path / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    require(
        module.encode() in arguments and tag.encode() in arguments,
        f"Controller PID {pid} was reused; refusing a new launch",
    )
    return True


def pod_run(module, arguments, tag):
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.pod",
        "run",
        "--tag",
        tag,
        module,
        *arguments,
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    update(f"running_{tag}", child_pid=process.pid, command=command)
    code = process.wait()
    require(code == 0, f"Pod job failed ({code}): {tag}; inspect its retained logs")
    exit_record = read_json(ROOT / "lm/runs" / tag / "exit.json")
    require(
        exit_record["exit_codes"] == [0, 0, 0, 0],
        f"Not every physical worker completed: {tag}",
    )


def checkpoint_workers(architecture, audited):
    directory = RUNS / architecture
    pointer = read_json(directory / "latest.json")["checkpoint"]
    payload = directory / pointer / "state.msgpack"
    expected = audited["final_checkpoint_sha256"]
    workers = [{"host": HOSTS[0], "checkpoint_sha256": sha256_file(payload)}]
    remote = "sha256sum -- " + shlex.quote(str(payload))
    processes = [
        (
            host,
            subprocess.Popen(
                SSH + [f"tasma@{host}", remote],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )
        for host in HOSTS[1:]
    ]
    for host, process in processes:
        output, error = process.communicate(timeout=60)
        require(process.returncode == 0, f"Checkpoint read failed on {host}: {error}")
        digest = output.split()[0]
        workers.append({"host": host, "checkpoint_sha256": digest})
    require(
        all(worker["checkpoint_sha256"] == expected for worker in workers),
        "Actual final checkpoint payloads differ across physical workers",
    )
    atomic_json(
        EVIDENCE / f"{architecture}-checkpoints.json",
        {
            "architecture": architecture,
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Independent SHA256 of actual final state.msgpack on each worker",
            "optimizer_steps": audited["optimizer_steps"],
            "training_targets": audited["training_targets"],
            "fingerprint": audited["fingerprint"],
            "all_actual_payloads_equal": True,
            "workers": workers,
        },
    )


def audit_completed(architecture, protocol, corpus, sources, recall):
    update(f"auditing_{architecture}")
    audited = audit_run(
        RUNS / architecture,
        architecture,
        protocol,
        corpus,
        expected_sources=sources,
        expected_recall=recall,
    )
    checkpoint_workers(architecture, audited)
    atomic_json(EVIDENCE / f"{architecture}-audit.json", audited)
    return audited


def benchmark(architecture, sources):
    require(source_provenance()["sha256"] == sources, "Frozen execution source changed")
    pod_run(
        "scripts.lm_benchmark",
        [
            "--config",
            f"lm/configs/{architecture}-60m.json",
            "--steps",
            "3",
            "--decode",
            "--decode-context",
            "1024",
            "--output",
            str(BENCHMARKS),
        ],
        f"bench-{architecture}-frozen-v1",
    )
    rows = [read_json(BENCHMARKS / f"{architecture}-host-{i}.json") for i in range(4)]
    validate_hardware([row["hardware"] for row in rows])
    model = read_json(ROOT / f"lm/configs/{architecture}-60m.json")["model"]
    for row in rows:
        validate_benchmark(row, architecture, model, sources)
    update(
        f"benchmark_passed_{architecture}",
        tokens_per_second=rows[0]["train"]["tokens_per_second"],
    )


def main():
    os.chdir(ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / "continuation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        sources = source_provenance()["sha256"]
        require(
            sources == read_json(RUNS / "transformer/manifest.json")["sources"],
            "Frozen execution source differs from completed Transformer",
        )
        launch_path = ROOT / "lm/runs/train-mamba3-60m-v1"
        record = read_json(launch_path / "launch.json")
        pid = record["controller_pid"]
        update("waiting_mamba3", existing_controller_pid=pid)
        while process_live(pid, "scripts.pod", "train-mamba3-60m-v1"):
            time.sleep(10)
        require(
            read_json(launch_path / "exit.json")["exit_codes"] == [0, 0, 0, 0],
            "Mamba-3 did not terminate successfully; do not start another job",
        )
        protocol = read_json(ROOT / "lm/configs/screen-protocol.json")
        corpus = TokenCorpus(ROOT / "data/fineweb-edu-1b", verify_hashes=True)
        validate_data(corpus, protocol)
        recall = build_recall_dataset(corpus.path / "tokenizer.json")
        audit_completed("mamba3", protocol, corpus, sources, recall)
        benchmark("mamba4", sources)
        require(
            source_provenance()["sha256"] == sources, "Frozen execution source changed"
        )
        pod_run(
            "lm.train",
            [
                "--config",
                "lm/configs/mamba4-60m.json",
                "--data",
                "data/fineweb-edu-1b",
                "--output",
                str(RUNS / "mamba4"),
            ],
            "train-mamba4-60m-v1",
        )
        audit_completed("mamba4", protocol, corpus, sources, recall)
        for architecture in ("transformer", "mamba3"):
            benchmark(architecture, sources)
        update("final_audit")
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "validation.verify_screen",
                "--benchmarks",
                str(BENCHMARKS),
                "--require-crosshost-checkpoints",
            ],
            cwd=ROOT,
            check=True,
        )
        update("completed", audit=str(EVIDENCE / "audit.json"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
