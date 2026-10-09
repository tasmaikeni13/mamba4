"""One matched causal-LM run with exact target accounting on every TPU chip."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from lm.config import ModelConfig, TrainConfig
from lm.runtime import atomic_json, initialize, source_provenance


def create_model(config):
    if config.architecture == "transformer":
        from lm.models.transformer import TransformerLM

        return TransformerLM(config)
    if config.architecture == "mamba3":
        from lm.models.mamba3 import Mamba3LM

        return Mamba3LM(config)
    if config.architecture == "mamba4":
        from lm.models.mamba4 import Mamba4LM

        return Mamba4LM(config)
    raise ValueError(f"Unknown architecture: {config.architecture}")


def parameter_ledger(params):
    entries = []
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(item, "key", item)) for item in path)
        entries.append({"name": name, "shape": list(value.shape), "count": value.size})
    total = sum(entry["count"] for entry in entries)
    embeddings = sum(
        entry["count"]
        for entry in entries
        if "embed" in entry["name"] and len(entry["shape"]) == 2
    )
    return {
        "total": total,
        "embedding": embeddings,
        "non_embedding": total - embeddings,
        "parameters": entries,
    }


def optimizer(config, params):
    warmup = min(config.warmup_steps, max(1, config.steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate / warmup,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=max(config.steps, warmup + 1),
        end_value=config.learning_rate * config.min_lr_ratio,
    )
    mask = jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
    transform = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(
            schedule,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
            mask=mask,
        ),
    )
    return transform, schedule


def local_loss(model, params, batch, training):
    logits = model.apply({"params": params}, batch["input_ids"], train=training)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), batch["targets"]
    )
    mask = batch["loss_mask"]
    return jnp.sum(losses * mask), jnp.sum(mask)


def make_steps(model):
    def train_step(state, batch):
        (numerator, count), grads = jax.value_and_grad(
            lambda params: local_loss(model, params, batch, True), has_aux=True
        )(state.params)
        total = jax.lax.psum(count, "data")
        grads = jax.tree_util.tree_map(
            lambda grad: jax.lax.psum(grad, "data") / jnp.maximum(total, 1), grads
        )
        loss = jax.lax.psum(numerator, "data") / jnp.maximum(total, 1)
        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss, "targets": total, "grad_norm": norm}

    def eval_step(params, batch):
        numerator, count = local_loss(model, params, batch, False)
        return {
            "nll_sum": jax.lax.psum(numerator, "data"),
            "targets": jax.lax.psum(count, "data"),
        }

    return (
        jax.pmap(train_step, axis_name="data", donate_argnums=(0,)),
        jax.pmap(eval_step, axis_name="data"),
    )


def make_diagnostic_step(model):
    """Observe actual learned factors and QR geometry outside optimizer steps."""

    def inspect(params, tokens):
        _, collections = model.apply(
            {"params": params}, tokens, train=False, mutable=["diagnostics"]
        )
        metrics = {}

        def flatten(node, prefix):
            if isinstance(node, dict):
                for name, item in node.items():
                    flatten(item, prefix + [name])
            else:
                value = node[0]
                name = prefix[-1]
                if name.endswith("allfinite"):
                    value = value.astype(jnp.int32)
                reduce = jax.lax.pmax if name.endswith("_max") else jax.lax.pmin
                metrics["/".join(prefix)] = reduce(value, "data")

        flatten(collections["diagnostics"], [])
        return metrics

    return jax.pmap(inspect, axis_name="data")


def record_diagnostics(corpus, config, params, diagnostic_step, step, output, log):
    if diagnostic_step is None:
        return
    batch = corpus.batch(
        0,
        config.global_batch,
        config.sequence_length,
        target_budget=config.eval_tokens,
        split="eval",
    )
    metrics = unreplicate(
        diagnostic_step(params, shard_batch(batch, config.global_batch)["input_ids"])
    )
    metrics = {name: float(value) for name, value in metrics.items()}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite memory factors: {metrics}")
    if any(value != 1 for name, value in metrics.items() if name.endswith("allfinite")):
        raise RuntimeError(f"Memory state became nonfinite: {metrics}")
    if jax.process_index() == 0:
        entry = {"event": "diagnostics", "step": step, "metrics": metrics}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        atomic_json(output / "diagnostics.json", entry)


def shard_batch(batch, global_batch):
    local_devices = jax.local_device_count()
    local_batch = global_batch // jax.process_count()
    start = jax.process_index() * local_batch
    return {
        name: array[start : start + local_batch].reshape(
            local_devices, local_batch // local_devices, array.shape[-1]
        )
        for name, array in batch.items()
    }


def unreplicate(tree):
    return jax.device_get(jax.tree_util.tree_map(lambda value: value[0], tree))


def save_checkpoint(output, state, metadata):
    """Each host keeps a complete checkpoint; publish pointer after durable files."""
    host_state = unreplicate(state)
    checkpoint = output / "checkpoints" / f"step-{int(host_state.step):08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes(host_state)
    temporary = checkpoint / "state.msgpack.tmp"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint / "state.msgpack")
    metadata = {
        **metadata,
        "step": int(host_state.step),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
    }
    atomic_json(checkpoint / "metadata.json", metadata)
    atomic_json(
        output / "latest.json", {"checkpoint": str(checkpoint.relative_to(output))}
    )
    # Retain final and immediately previous checkpoint without filling worker disks.
    older = sorted((output / "checkpoints").glob("step-*"))[:-2]
    for path in older:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def evaluate(corpus, config, params, eval_step):
    available = corpus.target_count("eval")
    budget = min(config.eval_tokens, available)
    count, numerator = 0, 0.0
    for step in range(math.ceil(budget / config.tokens_per_step)):
        batch = corpus.batch(
            step,
            config.global_batch,
            config.sequence_length,
            target_budget=budget,
            split="eval",
        )
        metrics = unreplicate(
            eval_step(params, shard_batch(batch, config.global_batch))
        )
        numerator += float(metrics["nll_sum"])
        count += int(metrics["targets"])
    if count != budget:
        raise RuntimeError(f"Evaluation accounting mismatch: {count} != {budget}")
    return {
        "nll": numerator / count,
        "perplexity": math.exp(numerator / count),
        "targets": count,
    }


def run(args):
    info = initialize(not args.single_host)
    if not args.single_host and (
        info["device_count"] != 16 or info["process_count"] != 4
    ):
        raise RuntimeError(f"Expected all 16 v4 chips on four hosts: {info}")
    frozen = json.loads(Path(args.config).read_text())
    model_config = ModelConfig(**frozen["model"])
    train_config = TrainConfig(**frozen["training"])
    if train_config.global_batch % jax.device_count():
        raise ValueError("Global batch must divide the total device count")
    from lm.data import TokenCorpus

    corpus = TokenCorpus(args.data, verify_hashes=True)
    if model_config.vocab_size != corpus.vocab_size:
        raise ValueError("Model and corpus vocabularies disagree")
    if corpus.target_count("train") < train_config.token_budget:
        raise ValueError(
            "Training corpus cannot cover the full budget without repetition"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"hardware-{jax.process_index()}.json", info)
    model = create_model(model_config)
    params = model.init(
        jax.random.PRNGKey(train_config.seed),
        jnp.zeros((1, model_config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    ledger = parameter_ledger(params)
    if not args.allow_small and abs(ledger["total"] / 60_000_000 - 1) > 0.02:
        raise RuntimeError(f"Model not within 2% of 60M: {ledger['total']}")
    tx, schedule = optimizer(train_config, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    provenance = source_provenance()
    identity = {
        "protocol": frozen,
        "sources": provenance["sha256"],
        "data_manifest_sha256": hashlib.sha256(
            (corpus.path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    tokens_seen = 0
    if (output / "latest.json").exists():
        pointer = json.loads((output / "latest.json").read_text())["checkpoint"]
        directory = output / pointer
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["fingerprint"] != fingerprint:
            raise RuntimeError("Resume refused: protocol, source, or corpus changed")
        payload = (directory / "state.msgpack").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint checksum mismatch")
        state = serialization.from_bytes(state, payload)
        tokens_seen = metadata["training_targets"]
    first_step = int(state.step)
    if tokens_seen != min(
        first_step * train_config.tokens_per_step, train_config.token_budget
    ):
        raise RuntimeError("Resume step/target accounting mismatch")
    if not args.single_host:
        from jax.experimental import multihost_utils

        identity_vector = np.array(
            [first_step, tokens_seen, ledger["total"]]
            + list(bytes.fromhex(fingerprint)),
            dtype=np.int32,
        )
        identities = np.asarray(multihost_utils.process_allgather(identity_vector))
        if not np.all(identities == identities[0]):
            raise RuntimeError(
                "Workers disagree on resume state, source, corpus, or protocol"
            )
    atomic_json(
        output / "manifest.json",
        {
            "fingerprint": fingerprint,
            **identity,
            "parameter_ledger": ledger,
            "provenance": provenance,
        },
    )
    state = jax.device_put_replicated(state, jax.local_devices())
    train_step, eval_step = make_steps(model)
    diagnostic_step = (
        make_diagnostic_step(model) if model_config.architecture == "mamba4" else None
    )
    log = (output / "metrics.jsonl").open("a") if jax.process_index() == 0 else None
    steady_seconds, steady_targets = 0.0, 0
    metadata = {"fingerprint": fingerprint, "training_targets": tokens_seen}
    if first_step == 0:
        initial = evaluate(corpus, train_config, state.params, eval_step)
        record_diagnostics(
            corpus, train_config, state.params, diagnostic_step, 0, output, log
        )
        if log:
            log.write(
                json.dumps({"event": "initial_eval", "step": 0, **initial}) + "\n"
            )
            log.flush()
    for step in range(first_step, train_config.steps):
        batch = corpus.batch(
            step,
            train_config.global_batch,
            train_config.sequence_length,
            target_budget=train_config.token_budget,
        )
        start = time.monotonic()
        state, metrics = train_step(
            state, shard_batch(batch, train_config.global_batch)
        )
        metrics = unreplicate(metrics)
        elapsed = time.monotonic() - start
        count = int(metrics["targets"])
        expected = min(
            train_config.tokens_per_step, train_config.token_budget - tokens_seen
        )
        if count != expected:
            raise RuntimeError(
                f"Training target mismatch at {step}: {count} != {expected}"
            )
        if not np.isfinite(float(metrics["loss"])) or not np.isfinite(
            float(metrics["grad_norm"])
        ):
            raise RuntimeError(f"Nonfinite training at {step}: {metrics}")
        tokens_seen += count
        if step > first_step:
            steady_seconds += elapsed
            steady_targets += count
        entry = {
            "event": "train",
            "step": step + 1,
            "training_targets": tokens_seen,
            "loss": float(metrics["loss"]),
            "grad_norm": float(metrics["grad_norm"]),
            "learning_rate": float(schedule(step)),
            "step_seconds": elapsed,
            "tokens_per_second": count / elapsed,
        }
        if log:
            log.write(json.dumps(entry) + "\n")
            log.flush()
        if (step + 1) % train_config.log_every == 0 or step == first_step:
            if jax.process_index() == 0:
                print(json.dumps(entry), flush=True)
                atomic_json(output / "status.json", {"status": "running", **entry})
        if (step + 1) % train_config.eval_every == 0 and step + 1 < train_config.steps:
            result = evaluate(corpus, train_config, state.params, eval_step)
            record_diagnostics(
                corpus,
                train_config,
                state.params,
                diagnostic_step,
                step + 1,
                output,
                log,
            )
            if log:
                log.write(
                    json.dumps({"event": "eval", "step": step + 1, **result}) + "\n"
                )
                log.flush()
        if (step + 1) % train_config.checkpoint_every == 0:
            metadata["training_targets"] = tokens_seen
            save_checkpoint(output, state, metadata)
    metadata["training_targets"] = tokens_seen
    save_checkpoint(output, state, metadata)
    result = evaluate(corpus, train_config, state.params, eval_step)
    record_diagnostics(
        corpus,
        train_config,
        state.params,
        diagnostic_step,
        train_config.steps,
        output,
        log,
    )
    if tokens_seen != train_config.token_budget:
        raise RuntimeError("Run ended without the exact full training budget")
    from lm.recall import build_recall_dataset, evaluate_recall, make_recall_step

    recall_summary = None
    if model_config.vocab_size == 50257:
        recall_dataset = build_recall_dataset(corpus.path / "tokenizer.json")
        recall = evaluate_recall(recall_dataset, state.params, make_recall_step(model))
        recall_summary = {
            "protocol": recall["protocol"],
            "overall": recall["overall"],
            "groups": recall["groups"],
        }
        if jax.process_index() == 0:
            recall_dataset.save(output / "recall-prompts")
            atomic_json(output / "recall.json", recall)
    elif not args.allow_small:
        raise RuntimeError("Production recall requires the pinned GPT2 vocabulary")
    result = {
        "status": "completed",
        "architecture": model_config.architecture,
        "seed": train_config.seed,
        "training_targets": tokens_seen,
        "optimizer_steps": train_config.steps,
        "parameter_count": ledger["total"],
        "fingerprint": fingerprint,
        "heldout": result,
        "recall": recall_summary,
        "steady_tokens_per_second": steady_targets / max(steady_seconds, 1e-9),
        "observed_device_count": info["device_count"],
        "memory_stats": [d.memory_stats() for d in jax.local_devices()],
    }
    atomic_json(output / f"result-host-{jax.process_index()}.json", result)
    if jax.process_index() == 0:
        atomic_json(output / "result.json", result)
        atomic_json(output / "status.json", result)
        print(json.dumps(result), flush=True)
    if log:
        log.close()
    if not args.single_host:
        jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/fineweb-edu-1b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--allow-small", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
