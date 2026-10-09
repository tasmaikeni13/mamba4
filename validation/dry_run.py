"""Pre-flight of every TPU program a full screen run executes; no holdout.

Before committing hours of pod time, compile and execute the training step,
the evaluation step, the memory-diagnostic step and the recall step for a
frozen configuration on all 16 chips. Evaluation and diagnostics run on a
training batch, so the held-out split is never opened. Outputs are checked
for finiteness only; nothing here is a result.
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
from lm.data import TokenCorpus
from lm.recall import build_recall_dataset, evaluate_recall, make_recall_step
from lm.runtime import atomic_json, initialize, source_provenance
from lm.train import (
    create_model,
    make_diagnostic_step,
    make_steps,
    optimizer,
    shard_batch,
    unreplicate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/fineweb-edu-1b")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    hardware = initialize(True)
    sources = source_provenance()["sha256"]
    frozen = json.loads(Path(options.config).read_text())
    model_config = ModelConfig(**frozen["model"])
    training = TrainConfig(**frozen["training"])
    corpus = TokenCorpus(options.data)
    model = create_model(model_config)
    params = model.init(
        jax.random.PRNGKey(training.seed),
        jnp.zeros((1, model_config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    tx, _ = optimizer(training, params)
    state = jax.device_put_replicated(
        TrainState.create(apply_fn=model.apply, params=params, tx=tx),
        jax.local_devices(),
    )
    train_step, eval_step = make_steps(model)
    timings, values = {}, {}
    batch = shard_batch(
        corpus.batch(0, training.global_batch, training.sequence_length),
        training.global_batch,
    )
    for step in range(2):
        started = time.monotonic()
        state, metrics = train_step(state, batch)
        metrics = unreplicate(metrics)
        timings[f"train_{step}"] = time.monotonic() - started
        values[f"train_loss_{step}"] = float(metrics["loss"])
    started = time.monotonic()
    metrics = unreplicate(eval_step(state.params, batch))
    timings["eval"] = time.monotonic() - started
    values["eval_nll"] = float(metrics["nll_sum"]) / float(metrics["targets"])
    if model_config.architecture == "mamba4":
        started = time.monotonic()
        diagnostics = unreplicate(
            make_diagnostic_step(model)(state.params, batch["input_ids"])
        )
        timings["diagnostics"] = time.monotonic() - started
        values["diagnostics"] = {name: float(v) for name, v in diagnostics.items()}
    dataset = build_recall_dataset(corpus.path / "tokenizer.json")
    started = time.monotonic()
    recall = evaluate_recall(dataset, state.params, make_recall_step(model))
    timings["recall"] = time.monotonic() - started
    values["recall_accuracy_untrained"] = recall["overall"]["accuracy"]
    flat = [values[k] for k in values if k != "diagnostics"]
    flat += list(values.get("diagnostics", {}).values())
    if not all(np.isfinite(flat)):
        raise RuntimeError(f"Nonfinite dry-run output: {values}")
    if jax.process_index() == 0:
        atomic_json(
            Path(options.output) / f"{Path(options.config).stem}.json",
            {
                "config": frozen,
                "hardware": hardware,
                "sources": sources,
                "timings_seconds_including_compile": timings,
                "values": values,
                "passed": True,
            },
        )
    print(json.dumps({"timings": timings, "passed": True}), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
