"""Stop only the verified worker process group created by this pod controller."""

import argparse
import os
from pathlib import Path
import signal
import time


def alive(pid, module):
    directory = Path(f"/proc/{pid}")
    if not directory.exists():
        return False
    try:
        state = directory.joinpath("stat").read_text().split(") ", 1)[1][0]
    except FileNotFoundError:
        return False
    if state == "Z":
        return False
    try:
        arguments = directory.joinpath("cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    if module.encode() not in arguments:
        raise RuntimeError(f"Refusing to stop reused PID {pid}: module differs")
    if directory.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
        raise RuntimeError(f"Refusing unowned process group {pid}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    file = Path(args.pid_file)
    if not file.exists():
        return
    pid = int(file.read_text().strip())
    if alive(pid, args.module):
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while alive(pid, args.module) and time.monotonic() < deadline:
            time.sleep(0.1)
        if alive(pid, args.module):
            os.killpg(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while alive(pid, args.module) and time.monotonic() < deadline:
                time.sleep(0.1)
    if alive(pid, args.module):
        raise RuntimeError(f"Worker remains live: {pid}")


if __name__ == "__main__":
    main()
