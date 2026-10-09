"""Sync sources, create the output directory on every worker, run one pod job.

``scripts.pod`` collects ``--output`` from every worker afterwards, so the
directory must exist on all of them even when JAX process 0 writes it alone.
Usage: ``python -m validation.pod_job TAG OUTPUT MODULE [ARGS...]``.
"""

import subprocess
import sys

from scripts.pod import HOSTS, ROOT, SSH
from validation.run_screen_v2 import pod, wait_for_hosts


def main():
    tag, output, module, *args = sys.argv[1:]
    wait_for_hosts()
    subprocess.run([sys.executable, "-m", "scripts.pod", "sync"], cwd=ROOT, check=True)
    for host in HOSTS[1:]:
        subprocess.run(SSH + [f"tasma@{host}", f"mkdir -p {ROOT / output}"], check=True)
    pod(module, ["--output", output, *args], tag)


if __name__ == "__main__":
    main()
