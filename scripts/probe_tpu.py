"""Collect an actual all-host TPU collective and memory evidence."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lm.runtime import atomic_json, initialize


def main():
    info = initialize()
    collective = jax.pmap(lambda x: jax.lax.psum(x, "pod"), axis_name="pod")
    result = np.asarray(collective(jnp.ones((jax.local_device_count(),))))
    if not np.all(result == jax.device_count()):
        raise RuntimeError(f"Incorrect all-device collective: {result}")
    info["collective_sum"] = result.tolist()
    info["local_memory_stats"] = [d.memory_stats() for d in jax.local_devices()]
    out = Path("lm/results/hardware") / f"host-{jax.process_index()}.json"
    atomic_json(out, info)
    print(json.dumps(info), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
