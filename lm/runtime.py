"""Initialize the existing TPU pod before creating arrays or devices."""

import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import jax


def initialize(distributed: bool = True) -> dict:
    if distributed:
        jax.distributed.initialize(initialization_timeout=180)
    jax.config.update("jax_compilation_cache_dir", str(Path(".jax_cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    devices = jax.devices()
    info = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": len(devices),
        "local_device_count": jax.local_device_count(),
        "devices": [
            {
                "id": d.id,
                "process_index": d.process_index,
                "kind": d.device_kind,
                "coords": list(d.coords) if hasattr(d, "coords") else None,
                "core_on_chip": getattr(d, "core_on_chip", None),
            }
            for d in devices
        ],
    }
    return info


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_provenance() -> dict:
    files = sorted(Path("lm").rglob("*.py"))
    files += sorted(Path("lm/configs").glob("*.json"))
    files += sorted(Path("scripts").glob("*.py"))
    files += [Path("pyproject.toml"), Path("uv.lock")]
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {"git_commit": commit.stdout.strip(), "sha256": hashes}
