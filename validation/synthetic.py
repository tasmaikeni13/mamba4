"""Synthetic trained tests of the memory claims at matched model width.

Small models are trained from scratch with identical data streams, optimizer,
steps and batch, then scored on held-out evaluation seeds:

``mqar``     K stored key/value pairs, then every key queried (capacity).
``unknown``  half of the queries ask for keys that were never stored; the
             target is a NONE token (knowing what is absent).
``noisy``    eight pairs scattered in random distractors; trained at 512
             tokens, scored up to 8,192 (selective retention over long range).
``hops``     a stored permutation of K nodes, then queries that ask for the
             successor (one hop) or the successor's successor (two hops).

Every model is a stack of pre-norm residual blocks between a tied embedding
and head: Transformer blocks (attention + SwiGLU MLP), official Mamba-3 SISO
blocks, or Mamba 4 selective conjugate-memory blocks. Learning rates are
chosen on a development seed; reported accuracies use a separate seed.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from lm.runtime import atomic_json


PAD, NONE, HOP1, HOP2 = 0, 1, 2, 3
KEYS = (16, 4096)
VALUES = (4096, 8192)
NOISE = (8192, 12288)
VOCAB = 12288
TRAIN_LENGTH = 512
MQAR_LOADS = (8, 16, 32, 64, 128)
NOISY_PAIRS = 8
HOP_LOADS = (8, 16, 32, 64)
NOISY_LENGTHS = (512, 1024, 2048, 4096, 8192)
SEEDS = {"train": 1, "development": 2, "evaluation": 3}
TASK_CODES = {"mqar": 0, "unknown": 1, "noisy": 2, "hops": 3}


def _pairs(rng, count):
    keys = rng.choice(np.arange(*KEYS), size=count, replace=False)
    values = rng.choice(np.arange(*VALUES), size=count, replace=False)
    return keys, values


def mqar_sequence(rng, count, length, unknown=False):
    """Stored pairs, then queries with teacher-forced answers.

    Inputs end at the final answer; targets are next tokens and the mask marks
    the answer slots. With ``unknown`` half of the queried keys are fresh and
    their answer is NONE.
    """
    keys, values = _pairs(rng, count)
    order = rng.permutation(count)
    queried, answers = keys[order], values[order]
    if unknown:
        fresh = np.setdiff1d(np.arange(*KEYS), keys)
        absent = rng.random(count) < 0.5
        queried = np.where(
            absent, rng.choice(fresh, size=count, replace=False), queried
        )
        answers = np.where(absent, NONE, answers)
    else:
        absent = np.zeros(count, bool)
    store = np.stack([keys, values], axis=1).reshape(-1)
    query = np.stack([queried, answers], axis=1).reshape(-1)
    tokens = np.concatenate([store, query])
    inputs = np.full(length, PAD, np.int32)
    targets = np.full(length, PAD, np.int32)
    mask = np.zeros(length, np.float32)
    inputs[: len(tokens) - 1] = tokens[:-1]
    targets[: len(tokens) - 1] = tokens[1:]
    slots = 2 * count + 2 * np.arange(count)
    mask[slots] = 1
    kind = np.full(length, -1, np.int8)
    kind[slots] = absent.astype(np.int8)
    return inputs, targets, mask, kind


def noisy_sequence(rng, length, pairs=NOISY_PAIRS):
    """Pairs at random positions in distractor tokens, queries at the end."""
    keys, values = _pairs(rng, pairs)
    body = rng.integers(*NOISE, size=length + 1 - 4 * pairs)
    slots = np.sort(rng.choice(len(body) // 2, size=pairs, replace=False)) * 2
    tokens = []
    cursor = 0
    for slot, key, value in zip(slots, keys, values):
        tokens.extend(body[cursor:slot])
        tokens.extend([key, value])
        cursor = slot
    tokens.extend(body[cursor:])
    order = rng.permutation(pairs)
    tokens.extend(np.stack([keys[order], values[order]], axis=1).reshape(-1))
    tokens = np.asarray(tokens[-(length + 1) :], np.int32)
    inputs, targets = tokens[:-1], tokens[1:]
    mask = np.zeros(length, np.float32)
    mask[length + 1 - 2 * pairs :: 2] = 1
    kind = np.where(mask > 0, 0, -1).astype(np.int8)
    return inputs, targets, mask, kind


def hops_sequence(rng, count, length):
    """Store node -> successor along a random cycle, then hop queries.

    Each query is [HOP1 or HOP2, node, answer]; the mask marks the node slot,
    whose target is the successor or the successor's successor.
    """
    nodes = rng.choice(np.arange(*KEYS), size=count, replace=False)
    # One random cycle: neither answer can equal the queried node, so copying
    # the current token never scores.
    successor = dict(zip(nodes.tolist(), np.roll(nodes, -1).tolist()))
    store = np.stack([nodes, [successor[n] for n in nodes.tolist()]], axis=1)
    asked = rng.choice(nodes, size=count, replace=False)
    hops = rng.integers(1, 3, size=count)
    answers = [
        successor[n] if h == 1 else successor[successor[n]]
        for n, h in zip(asked.tolist(), hops.tolist())
    ]
    query = np.stack([np.where(hops == 1, HOP1, HOP2), asked, answers], axis=1)
    tokens = np.concatenate([store.reshape(-1), query.reshape(-1)])
    inputs = np.full(length, PAD, np.int32)
    targets = np.full(length, PAD, np.int32)
    mask = np.zeros(length, np.float32)
    inputs[: len(tokens) - 1] = tokens[:-1]
    targets[: len(tokens) - 1] = tokens[1:]
    slots = 2 * count + 3 * np.arange(count) + 1
    mask[slots] = 1
    kind = np.full(length, -1, np.int8)
    kind[slots] = hops - 1
    return inputs, targets, mask, kind


def batch(task, rng, size, length=TRAIN_LENGTH, load=None):
    rows = []
    for _ in range(size):
        if task == "noisy":
            rows.append(noisy_sequence(rng, length))
        elif task == "hops":
            count = load or int(rng.choice(HOP_LOADS))
            rows.append(hops_sequence(rng, count, length))
        else:
            count = load or int(rng.choice(MQAR_LOADS))
            rows.append(mqar_sequence(rng, count, length, unknown=task == "unknown"))
    inputs, targets, mask, kind = (np.stack(part) for part in zip(*rows))
    return {"inputs": inputs, "targets": targets, "mask": mask, "kind": kind}


@dataclasses.dataclass(frozen=True)
class Spec:
    """One small architecture; kinds are A (attention), S (Mamba-3), M (memory)."""

    name: str
    kinds: str
    d_model: int = 128
    n_heads: int = 2
    d_ff: int = 256
    d_state: int = 48
    head_dim: int = 64
    key_dim: int = 32
    memory_head_dim: int = 64
    floor_min: float = 0.25
    floor_init: float = 1.0
    conv_kernel: int = 4
    order_head: bool = True
    beta_max: float = 1.0
    key_shift: bool = False
    dtype: str = "float32"

    def model_config(self):
        from lm.config import ModelConfig

        architecture = (
            "mamba4"
            if "M" in self.kinds
            else ("mamba3" if "S" in self.kinds else "transformer")
        )
        return ModelConfig(
            architecture=architecture,
            vocab_size=VOCAB,
            d_model=self.d_model,
            n_layers=len(self.kinds),
            n_heads=self.n_heads,
            d_ff=self.d_ff if "A" in self.kinds else 0,
            d_state=self.d_state,
            expand=2,
            head_dim=self.head_dim,
            chunk_size=64,
            dtype=self.dtype,
            remat=False,
            max_seq_len=max(NOISY_LENGTHS),
            key_dim=self.key_dim,
            memory_floor="fixed",
            protected_anchor_budget=0,
            memory_mixer="selective",
            layer_pattern=self.kinds.replace("A", "S"),
            conv_kernel=self.conv_kernel,
            floor_min=self.floor_min,
            floor_init=self.floor_init,
            order_head=self.order_head,
            memory_head_dim=self.memory_head_dim,
            memory_solver="blocked",
            memory_beta_max=self.beta_max,
            memory_key_shift=self.key_shift,
        )

    def state_floats(self, length):
        """Per-sequence recurrent state, or the KV cache at ``length``."""
        config = self.model_config()
        total = 0
        for kind in self.kinds:
            if kind == "A":
                total += 2 * length * self.d_model
            elif kind == "S":
                total += config.num_heads * self.d_state * self.head_dim
            else:
                heads = self.d_model * 2 // self.memory_head_dim
                dim = self.key_dim
                total += heads * (dim * (dim + 1) // 2 + dim * self.memory_head_dim)
        return total


def build_model(spec):
    from flax import linen as nn

    from lm.models.common import RMSNorm, TokenEmbedding
    from lm.models.mamba3 import Mamba3Block
    from lm.models.mamba4 import SelectiveMemoryBlock
    from lm.models.transformer import TransformerBlock

    blocks = {"A": TransformerBlock, "S": Mamba3Block, "M": SelectiveMemoryBlock}
    config = spec.model_config()

    class Stack(nn.Module):
        @nn.compact
        def __call__(self, ids, train=False):
            embedding = TokenEmbedding(config, name="tokens")
            x = embedding(ids)
            for index, kind in enumerate(spec.kinds):
                x = blocks[kind](config, name=f"layer_{index}")(x, train)
            return embedding.attend(RMSNorm(config.d_model, name="final_norm")(x))

    return Stack()


def make_steps(model, tx):
    import jax
    import jax.numpy as jnp

    def local_loss(params, data):
        logits = model.apply({"params": params}, data["inputs"], train=True)
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(logp, data["targets"][..., None], axis=-1)[..., 0]
        count = jax.lax.psum(jnp.sum(data["mask"]), "data")
        return jnp.sum(nll * data["mask"]) / jnp.maximum(count, 1.0)

    def train_step(params, opt_state, data):
        loss, grads = jax.value_and_grad(local_loss)(params, data)
        grads = jax.lax.psum(grads, "data")
        loss = jax.lax.psum(loss, "data")
        updates, opt_state = tx.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    def score_step(params, data):
        logits = model.apply({"params": params}, data["inputs"])
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        hit = jnp.argmax(logp, axis=-1) == data["targets"]
        target = jnp.take_along_axis(logp, data["targets"][..., None], axis=-1)[..., 0]
        return {
            "hit": hit,
            "target_logp": target,
            "none_logp": logp[..., NONE],
            "confidence": jnp.exp(jnp.max(logp, axis=-1)),
        }

    return (
        jax.pmap(train_step, axis_name="data", donate_argnums=(0, 1)),
        jax.pmap(score_step, axis_name="data"),
    )


def _local(data, global_batch):
    import jax

    local = global_batch // jax.process_count()
    start = jax.process_index() * local
    return {
        name: value[start : start + local].reshape(
            jax.local_device_count(), -1, *value.shape[1:]
        )
        for name, value in data.items()
    }


def compiled(mapped, label, *arguments):
    """Compile ahead of time, then wait for every host before launching.

    The TPU backend can emit slightly different bundles on different hosts for
    the same HLO; launching in lockstep avoids launch-identity mismatches.
    """
    import jax
    from jax.experimental import multihost_utils

    executable = mapped.lower(*arguments).compile()
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(label)
    return executable


def train_one(spec, task, learning_rate, steps, global_batch, log_every=100):
    import jax
    import optax

    model = build_model(spec)
    params = model.init(jax.random.PRNGKey(0), np.zeros((1, TRAIN_LENGTH), np.int32))[
        "params"
    ]
    count = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    schedule = optax.warmup_cosine_decay_schedule(
        learning_rate / 20, learning_rate, steps // 20, steps, learning_rate / 10
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        # As in lm.train: decay matrices only, never gates, floors or norms.
        optax.adamw(
            schedule,
            b1=0.9,
            b2=0.98,
            weight_decay=0.1,
            mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params),
        ),
    )
    train_step, score_step = make_steps(model, tx)
    devices = jax.local_devices()
    params = jax.device_put_replicated(params, devices)
    label = f"{spec.name}-{task}-{learning_rate}"
    initialize = compiled(jax.pmap(tx.init), f"init-{label}", params)
    opt_state = initialize(params)
    rng = np.random.default_rng([SEEDS["train"], TASK_CODES[task]])
    curve, started, executable = [], time.perf_counter(), None
    for step in range(steps):
        data = batch(task, rng, global_batch)
        local = _local({k: v for k, v in data.items() if k != "kind"}, global_batch)
        if executable is None:
            executable = compiled(
                train_step, f"train-{label}", params, opt_state, local
            )
        params, opt_state, loss = executable(params, opt_state, local)
        if step % log_every == 0 or step == steps - 1:
            value = float(np.asarray(loss)[0])
            curve.append([step, value])
            if not np.isfinite(value):
                break
    seconds = time.perf_counter() - started
    return (
        params,
        score_step,
        {
            "parameters": count,
            "curve": curve,
            "seconds": seconds,
            "finite": bool(np.isfinite(curve[-1][1])),
        },
    )


def score(score_step, params, task, seed, rows, global_batch):
    """Per-row answer accuracy for every load or length of the task."""
    import jax
    from jax.experimental import multihost_utils

    results = {}
    groups = {"noisy": NOISY_LENGTHS, "hops": HOP_LOADS}.get(task, MQAR_LOADS)
    for group in groups:
        rng = np.random.default_rng([seed, group, TASK_CODES[task]])
        length = group if task == "noisy" else TRAIN_LENGTH
        per = max(1, global_batch * TRAIN_LENGTH // length)
        per -= per % jax.device_count()
        per = max(per, jax.device_count())
        collected = {"hit": [], "mask": [], "kind": [], "none_logp": [], "conf": []}
        executable = None
        for _ in range(-(-rows // per)):
            data = batch(task, rng, per, length=length, load=group)
            local = _local(
                {k: v for k, v in data.items() if k in ("inputs", "targets")}, per
            )
            if executable is None:
                executable = compiled(
                    score_step, f"score-{task}-{seed}-{group}", params, local
                )
            out = jax.device_get(executable(params, local))
            for key, name in (
                ("hit", "hit"),
                ("none_logp", "none_logp"),
                ("confidence", "conf"),
            ):
                value = np.asarray(out[key])
                gathered = np.asarray(multihost_utils.process_allgather(value))
                collected[name].append(gathered.reshape(per, length))
            collected["mask"].append(data["mask"])
            collected["kind"].append(data["kind"])
        arrays = {k: np.concatenate(v)[:rows] for k, v in collected.items()}
        mask = arrays["mask"] > 0
        hits = arrays["hit"] & mask
        per_row = hits.sum(axis=1) / mask.sum(axis=1)
        entry = {
            "rows": rows,
            "accuracy": float(per_row.mean()),
            "per_row_accuracy": per_row.round(5).tolist(),
            "exact_rate": float(np.mean(hits.sum(axis=1) == mask.sum(axis=1))),
        }
        if task == "hops":
            for hop in (1, 2):
                chosen = mask & (arrays["kind"] == hop - 1)
                entry[f"hop{hop}_accuracy"] = float(hits[chosen].mean())
        if task == "unknown":
            known = mask & (arrays["kind"] == 0)
            absent = mask & (arrays["kind"] == 1)
            entry["known_accuracy"] = float(hits[known].mean())
            entry["unknown_accuracy"] = float(hits[absent].mean())
            from validation.claims import auroc

            entry["none_auroc"] = auroc(
                arrays["none_logp"][known | absent], absent[known | absent]
            )
        results[str(group)] = entry
    return results


SPECS = {
    "transformer": Spec("transformer", "AA"),
    "mamba3": Spec("mamba3", "SS"),
    "mamba4": Spec("mamba4", "MM"),
    "mamba4-hybrid": Spec("mamba4-hybrid", "SM"),
    "mamba4-shift": Spec("mamba4-shift", "MM", key_shift=True),
    "mamba4-strong": Spec("mamba4-strong", "MM", beta_max=16.0),
    "mamba4-lowfloor": Spec("mamba4-lowfloor", "MM", floor_min=0.02, floor_init=0.1),
    "mamba4-strong-lowfloor": Spec(
        "mamba4-strong-lowfloor", "MM", beta_max=16.0, floor_min=0.02, floor_init=0.1
    ),
}


def run(options):
    import jax

    from lm.runtime import initialize

    hardware = initialize(not options.local)
    # Only process 0 writes the persistent cache, and XLA reuses its kernel
    # choices across these many small programs; hosts must compile alike.
    jax.config.update("jax_enable_compilation_cache", False)
    output = Path(options.output)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "steps": options.steps,
        "global_batch": options.batch,
        "runs": {},
    }
    for name in options.models:
        spec = SPECS[name]
        if options.key_dim:
            spec = dataclasses.replace(spec, key_dim=options.key_dim)
        if options.floor_min:
            spec = dataclasses.replace(
                spec, floor_min=options.floor_min, floor_init=options.floor_min * 4
            )
        for task in options.tasks:
            candidates = {}
            for learning_rate in options.learning_rates:
                params, score_step, info = train_one(
                    spec, task, learning_rate, options.steps, options.batch
                )
                development = score(
                    score_step,
                    params,
                    task,
                    SEEDS["development"],
                    options.rows,
                    options.batch,
                )
                mean = float(np.mean([g["accuracy"] for g in development.values()]))
                candidates[str(learning_rate)] = {
                    **info,
                    "development_mean_accuracy": mean,
                    "development": development,
                }
                if mean >= max(
                    c["development_mean_accuracy"] for c in candidates.values()
                ):
                    best = (learning_rate, params, score_step)
                print(
                    name,
                    task,
                    learning_rate,
                    f"dev={mean:.3f}",
                    f"{info['seconds']:.0f}s",
                    flush=True,
                )
            learning_rate, params, score_step = best
            evaluation = score(
                score_step,
                params,
                task,
                SEEDS["evaluation"],
                options.rows,
                options.batch,
            )
            results["runs"][f"{name}/{task}"] = {
                "spec": dataclasses.asdict(spec),
                "state_floats_at_512": spec.state_floats(TRAIN_LENGTH),
                "selected_learning_rate": learning_rate,
                "candidates": candidates,
                "evaluation": evaluation,
            }
            print(
                name,
                task,
                "eval",
                {g: round(v["accuracy"], 3) for g, v in evaluation.items()},
                flush=True,
            )
            if jax.process_index() == 0:
                atomic_json(output / "synthetic.json", results)
    if not options.local:
        jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=list(SPECS))
    parser.add_argument(
        "--tasks", nargs="+", default=["mqar", "unknown", "noisy", "hops"]
    )
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[5e-4, 1.5e-3, 5e-3]
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--key-dim", type=int, default=0)
    parser.add_argument("--floor-min", type=float, default=0.0)
    parser.add_argument("--local", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
