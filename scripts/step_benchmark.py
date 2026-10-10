"""Time full training steps of configuration variants on all chips.

Each variant is ``name=config.json`` followed by optional ``:key=value``
model overrides (values parse as JSON, else as strings), for example
``mamba4-noremat=lm/configs/mamba4-125m.json:remat=false``. Every variant
builds its model and AdamW state, compiles the real data-parallel training
step and times steady steps on synthetic tokens with the configuration's
global batch. A variant that fails (for example out of device memory) is
recorded and the next one runs. Run through scripts.pod.
"""

import argparse
import gc
import json
from pathlib import Path
import time

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig, TrainConfig
from lm.runtime import atomic_json, initialize, source_provenance
from lm.train import create_model, make_steps, optimizer, parameter_ledger


def parse(variant):
    name, rest = variant.split("=", 1)
    path, *settings = rest.split(":")
    overrides = {}
    for setting in settings:
        key, value = setting.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return name, path, overrides


def measure(path, overrides, steps):
    frozen = json.loads(Path(path).read_text())
    config = ModelConfig(**{**frozen["model"], **overrides})
    training = TrainConfig(**frozen["training"])
    model = create_model(config)
    params = model.init(
        jax.random.PRNGKey(training.seed),
        jnp.zeros((1, config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    ledger = parameter_ledger(params)
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
    started = time.monotonic()
    executable = train_step.lower(replicated, batch).compile()
    compile_seconds = time.monotonic() - started
    times = []
    for _ in range(steps + 1):
        started = time.monotonic()
        replicated, metrics = executable(replicated, batch)
        jax.block_until_ready(metrics)
        times.append(time.monotonic() - started)
    median = float(np.median(times[1:]))
    stats = jax.local_devices()[0].memory_stats() or {}
    result = {
        "overrides": overrides,
        "parameters": ledger["total"],
        "compile_seconds": compile_seconds,
        "step_seconds": times[1:],
        "median_step_seconds": median,
        "tokens_per_second": training.global_batch * training.sequence_length / median,
        "loss": float(jax.device_get(metrics["loss"])[0]),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }
    del replicated, executable, state, params
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("variants", nargs="+")
    options = parser.parse_args()
    info = initialize(True)
    results = {"hardware": info, "sources": source_provenance()["sha256"], "runs": {}}
    for variant in options.variants:
        name, path, overrides = parse(variant)
        try:
            entry = measure(path, overrides, options.steps)
        except Exception as error:  # record and continue with the next variant
            entry = {
                "overrides": overrides,
                "error": f"{type(error).__name__}: {str(error)[:3000]}",
            }
        entry["config"] = path
        results["runs"][name] = entry
        if jax.process_index() == 0:
            summary = {k: v for k, v in entry.items() if k != "step_seconds"}
            print(name, json.dumps(summary)[:900], flush=True)
        gc.collect()
    atomic_json(Path(options.output) / f"host-{jax.process_index()}.json", results)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
