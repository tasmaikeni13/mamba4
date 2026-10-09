"""Short development pilots with the frozen training step, never the holdout.

A pilot runs the first ``--steps`` optimizer steps of the full screen protocol:
same seed, initialization, data order, batch, optimizer and complete 7,630-step
learning-rate schedule. Each step's loss is computed on a training batch before
that batch updates the model, so it is an unseen-data development signal. The
held-out split is never opened. Selection compares these per-step losses with
the peers' logged losses on the identical batches.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig, TrainConfig
from lm.data import TokenCorpus
from lm.runtime import atomic_json, initialize, source_provenance
from lm.train import (
    create_model,
    make_steps,
    optimizer,
    parameter_ledger,
    shard_batch,
    unreplicate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/fineweb-edu-1b")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    options = parser.parse_args()
    info = initialize(True)
    # Hash the executed sources before any later local edit can change them.
    sources = source_provenance()["sha256"]
    frozen = json.loads(Path(options.config).read_text())
    model_config = ModelConfig(**frozen["model"])
    train_config = TrainConfig(**frozen["training"])
    if options.steps > train_config.steps:
        raise ValueError("A pilot cannot exceed the protocol's optimizer steps")
    corpus = TokenCorpus(options.data, verify_hashes=False)
    model = create_model(model_config)
    params = model.init(
        jax.random.PRNGKey(train_config.seed),
        jnp.zeros((1, model_config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    ledger = parameter_ledger(params)
    tx, schedule = optimizer(train_config, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax.device_put_replicated(state, jax.local_devices())
    train_step, _ = make_steps(model)
    losses, norms, seconds = [], [], []
    for step in range(options.steps):
        batch = corpus.batch(
            step,
            train_config.global_batch,
            train_config.sequence_length,
            target_budget=train_config.token_budget,
        )
        started = time.monotonic()
        state, metrics = train_step(
            state, shard_batch(batch, train_config.global_batch)
        )
        metrics = unreplicate(metrics)
        seconds.append(time.monotonic() - started)
        loss, norm = float(metrics["loss"]), float(metrics["grad_norm"])
        if not (np.isfinite(loss) and np.isfinite(norm)):
            raise RuntimeError(f"Nonfinite pilot step {step + 1}: {loss}, {norm}")
        losses.append(loss)
        norms.append(norm)
        if jax.process_index() == 0 and (step + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": loss,
                        "grad_norm": norm,
                        "lr": float(schedule(step)),
                        "step_seconds": seconds[-1],
                    }
                ),
                flush=True,
            )
    result = {
        "name": options.name,
        "purpose": "development selection on unseen training batches; no holdout",
        "config": frozen,
        "config_sha256": hashlib.sha256(Path(options.config).read_bytes()).hexdigest(),
        "steps": options.steps,
        "parameter_count": ledger["total"],
        "non_embedding_parameters": ledger["non_embedding"],
        "train_loss": losses,
        "grad_norm": norms,
        "median_step_seconds": float(np.median(seconds[1:])),
        "hardware": info,
        "sources": sources,
    }
    if jax.process_index() == 0:
        atomic_json(Path(options.output) / f"{options.name}.json", result)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
