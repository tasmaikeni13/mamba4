"""Profile one configuration's training step on the pod and rank device ops.

Same variant syntax as scripts.step_benchmark (``config.json:key=value``).
After compilation and two warm-up steps, two steps run under
jax.profiler.trace; JAX process 0 reads its own device trace and writes the
total device time of every XLA operation, largest first, to
``<output>/profile.json``. Run through scripts.pod.
"""

import argparse
import json
from pathlib import Path
import time

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig, TrainConfig
from lm.runtime import atomic_json, initialize
from lm.train import create_model, make_steps, optimizer
from scripts.step_benchmark import parse


def summarize(directory, top):
    """Total device time per operation on the first TPU core of the trace."""
    import gzip

    paths = sorted(Path(directory).rglob("*.trace.json.gz"))
    if not paths:
        return {"error": "no perfetto trace written"}
    with gzip.open(paths[-1], "rt") as stream:
        trace = json.load(stream)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    names = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            names[event["pid"]] = event.get("args", {}).get("name", "")
    device = sorted(pid for pid, name in names.items() if "TPU:0" in name)
    totals = {}
    for event in events:
        if event.get("ph") != "X" or event.get("pid") not in device[:1]:
            continue
        entry = totals.setdefault(event["name"], [0.0, 0])
        entry[0] += event.get("dur", 0) / 1e3
        entry[1] += 1
    ranked = sorted(totals.items(), key=lambda item: -item[1][0])
    return {
        "device_processes": [names[pid] for pid in device],
        "total_ms": sum(value[0] for value in totals.values()),
        "ops": [
            {"name": name, "milliseconds": value[0], "count": value[1]}
            for name, value in ranked[:top]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("variant")
    options = parser.parse_args()
    info = initialize(True)
    _, path, overrides = parse("profile=" + options.variant)
    frozen = json.loads(Path(path).read_text())
    config = ModelConfig(**{**frozen["model"], **overrides})
    training = TrainConfig(**frozen["training"])
    model = create_model(config)
    params = model.init(
        jax.random.PRNGKey(training.seed),
        jnp.zeros((1, config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    tx, _ = optimizer(training, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    replicated = jax.device_put_replicated(state, jax.local_devices())
    per_device = training.global_batch // jax.device_count()
    shape = (jax.local_device_count(), per_device, training.sequence_length)
    tokens = (np.arange(np.prod(shape)).reshape(shape) % config.vocab_size).astype(
        np.int32
    )
    batch = {
        "input_ids": tokens,
        "targets": (tokens + 1) % config.vocab_size,
        "loss_mask": np.ones(shape, np.float32),
    }
    train_step, _ = make_steps(model)
    executable = train_step.lower(replicated, batch).compile()
    for _ in range(2):
        replicated, metrics = executable(replicated, batch)
        jax.block_until_ready(metrics)
    trace = Path(options.output) / f"trace-{jax.process_index()}"
    started = time.monotonic()
    with jax.profiler.trace(str(trace), create_perfetto_trace=True):
        for _ in range(2):
            replicated, metrics = executable(replicated, batch)
            jax.block_until_ready(metrics)
    seconds = (time.monotonic() - started) / 2
    if jax.process_index() == 0:
        summary = {
            "hardware": info,
            "variant": options.variant,
            "step_seconds": seconds,
        }
        summary.update(summarize(trace, options.top))
        atomic_json(Path(options.output) / "profile.json", summary)
        for op in summary.get("ops", [])[:40]:
            print(f"{op['milliseconds']:9.2f} ms {op['count']:5d}  {op['name'][:110]}")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
