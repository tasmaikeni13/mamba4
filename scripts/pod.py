"""Deploy and run the existing four-host v4-32 pod; never allocates resources."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
HOSTS = ["10.130.0.15", "10.130.0.14", "10.130.0.16", "10.130.0.17"]
SSH = [
    "ssh",
    "-i",
    str(Path.home() / ".ssh/google_compute_engine"),
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


def deploy(data=False):
    """Copy frozen sources and install the same lockfile on every worker."""
    processes = []
    for host in HOSTS[1:]:
        command = ["rsync", "-az", "-e", shlex.join(SSH)]
        for exclude in [
            ".git",
            ".venv",
            ".lake",
            ".jax_cache",
            "__pycache__",
            "checkpoints",
            "lm/runs",
            "analysis/runs",
            "vendor",
            ".pytest_cache",
            ".ruff_cache",
        ]:
            command += ["--exclude", exclude]
        if not data:
            command += ["--exclude", "data"]
        command += [str(ROOT) + "/", f"tasma@{host}:{ROOT}/"]
        processes.append((host, subprocess.Popen(command)))
    for host, process in processes:
        if process.wait() != 0:
            raise RuntimeError(f"rsync failed on {host}")
    processes = []
    for host in HOSTS[1:]:
        command = f"cd {shlex.quote(str(ROOT))} && ~/.local/bin/uv sync --frozen"
        processes.append((host, subprocess.Popen(SSH + [f"tasma@{host}", command])))
    for host, process in processes:
        if process.wait() != 0:
            raise RuntimeError(f"Environment installation failed on {host}")


def execute(module, args, tag):
    output = ROOT / "lm/runs" / tag
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (ROOT / "lm/runs/pod.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [str(ROOT / ".venv/bin/python"), "-u", "-m", module] + args
    shell_command = f"cd {shlex.quote(str(ROOT))} && exec {shlex.join(command)}"
    processes, handles = [], []
    for index, host in enumerate(HOSTS):
        handle = (output / f"host-{index}.log").open("a")
        handles.append(handle)
        pid_file = output / f"worker-{index}.pid"
        pid_file.unlink(missing_ok=True)
        if index == 0:
            call = command
        else:
            body = f"echo $$ > {shlex.quote(str(pid_file))}; {shell_command}"
            remote = (
                f"mkdir -p {shlex.quote(str(output))} && "
                f"exec setsid sh -c {shlex.quote(body)}"
            )
            call = SSH + [f"tasma@{host}", remote]
        process = subprocess.Popen(
            call, cwd=ROOT, stdout=handle, stderr=handle, start_new_session=True
        )
        processes.append(process)
        if index == 0:
            pid_file.write_text(str(process.pid) + "\n")
    record = {
        "controller_pid": os.getpid(),
        "started_unix": time.time(),
        "module": module,
        "args": args,
        "hosts": HOSTS,
        "controller_child_pids": [p.pid for p in processes],
    }
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    try:
        while any(p.poll() is None for p in processes):
            failure = [p.returncode for p in processes if p.returncode not in (None, 0)]
            if failure:
                # A distributed worker failure makes this collective run unusable.
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(2)
    finally:
        # SSH exit alone does not establish remote process termination.
        cleanup = []
        for index, host in enumerate(HOSTS):
            stop = [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "scripts.stop_pod_worker",
                "--pid-file",
                str(output / f"worker-{index}.pid"),
                "--module",
                module,
            ]
            if index:
                stop = SSH + [
                    f"tasma@{host}",
                    f"cd {shlex.quote(str(ROOT))} && {shlex.join(stop)}",
                ]
            cleanup.append(subprocess.Popen(stop, cwd=ROOT))
        for process in cleanup:
            if process.wait(timeout=25) != 0:
                raise RuntimeError("Unable to verify distributed worker cleanup")
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record["exit_codes"] = [p.returncode for p in processes]
        record["finished_unix"] = time.time()
        (output / "exit.json").write_text(json.dumps(record, indent=2) + "\n")
        for handle in handles:
            handle.close()
        lock_file.close()
    # Preserve remote failure evidence as well as completed runs.
    collect(args)
    if any(record["exit_codes"]):
        raise RuntimeError(
            f"Pod execution failed: {record['exit_codes']}; see {output}"
        )


def collect(args):
    """JAX process0 can be any physical host; preserve every host's evidence."""
    paths = ["lm/results"]
    if "--output" in args:
        path = Path(args[args.index("--output") + 1])
        if path.is_absolute():
            path = path.relative_to(ROOT)
        if str(path) not in paths:
            paths.append(str(path))
    for path in paths:
        (ROOT / path).mkdir(parents=True, exist_ok=True)
        for host in HOSTS[1:]:
            command = [
                "rsync",
                "-az",
                "-e",
                shlex.join(SSH),
                "--exclude",
                "checkpoints",
                "--exclude",
                "manifest.json",
                f"tasma@{host}:{ROOT / path}/",
                str(ROOT / path) + "/",
            ]
            subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("--data", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--tag", required=True)
    run.add_argument("module")
    run.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.action == "sync":
        deploy(options.data)
    else:
        execute(options.module, options.args, options.tag)


if __name__ == "__main__":
    main()
