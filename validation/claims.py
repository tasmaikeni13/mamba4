"""Claims-60m: pre-registered tests of retrieval, context, calibration and decode.

The 60M screen gated only on held-out NLL. This suite tests the paper's
distinctive claims on the trained 60M checkpoints without further training:
exact copying and associative recall across loads (exact retrieval), passkey
retrieval and long-document NLL up to 16,384 tokens (context), confidence at
answer positions (calibrated retrieval) and per-token decode cost versus
context length. The protocol is fixed in ``validation/claims/PROTOCOL.md``
before any model is scored; every task array is deterministic and hashed.

``build``    (CPU) writes task arrays to data/claims-v1 and their manifest.
``evaluate`` (pod) scores every model on identical arrays.
``decode``   (pod) times one-token decode and counts cache bytes per context.
``report``   (CPU) computes the pre-registered statistics and verdicts.
``launch``   (local) syncs sources/data, then runs evaluate and decode.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

from lm.data import sha256_file
from lm.runtime import atomic_json


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261010
EOS, COLON, NEWLINE = 50256, 25, 198
WORD = re.compile(r"^ [a-z]{3,10}$")
BUCKETS = (1024, 2048, 4096, 8192, 16384)
PER_DEVICE = {1024: 8, 2048: 4, 4096: 2, 8192: 1, 16384: 1}
COPY_LENGTHS = (16, 64, 256, 512, 1024, 2048, 4096)
MQAR_PAIRS = (16, 32, 64, 128, 240, 496, 1008, 2032)
MQAR_QUERIES = 16
PASSKEY_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
PASSKEY_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
ROWS = {"copy": 128, "mqar": 128, "passkey": 64}
FAMILIES = ("copy", "mqar", "passkey")
LONG_CONTEXT, LONG_DOCUMENTS = 16384, 512
POSITION_EDGES = (0, 128, 512, 1024, 2048, 4096, 8192, 16384)
CALIBRATION_BINS = 15
DECODE_CONTEXTS = (1024, 4096, 16384, 65536)
DECODE_STEPS = 32
TRAINING_CONTEXT = 1024
RESAMPLES = 10_000
FORMAT = "mamba4.claims-tasks.v1"
TASKS = ROOT / "data/claims-v1"
RESULTS = ROOT / "lm/results/claims-60m"
RAW = ROOT / "lm/runs/claims-60m"
SHARD = ROOT / "data/fresh-holdout-v2/source/013_00000.parquet"

# Mohtashami & Jaggi (2023), arXiv:2305.16300, passkey retrieval template,
# repeated at sentence granularity so no filler word is cut.
DESCRIPTION = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize them. I will quiz you about the important "
    "information there."
)
GARBAGE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
QUESTION = "What is the pass key? The pass key is"


def bucket_for(length):
    for bucket in BUCKETS:
        if length <= bucket:
            return bucket
    raise ValueError(f"Sequence of {length} tokens exceeds every bucket")


def word_pools(tokenizer):
    """Single-token lowercase words: all (copy), keys and values (disjoint)."""
    ids = sorted(
        index
        for _, index in tokenizer.get_vocab().items()
        if WORD.match(tokenizer.decode([index]))
    )
    ids = np.asarray(ids, np.int32)[np.random.default_rng(SEED).permutation(len(ids))]
    half = len(ids) // 2
    return ids, ids[:half], ids[half:]


def copy_rows(words, rng):
    """Random distinct words, then the same words again; score the second copy.

    Position p in [L, 2L-2] holds second-copy word p-L; its target is the next
    word. The first second-copy word has no cue and is not scored.
    """
    rows = []
    for length in COPY_LENGTHS:
        for _ in range(ROWS["copy"]):
            sample = rng.choice(words, size=length, replace=False)
            sequence = np.concatenate([sample, sample])
            positions = np.arange(length, 2 * length - 1)
            rows.append(
                dict(
                    family="copy",
                    group=length,
                    ids=sequence,
                    positions=positions,
                    targets=sequence[positions + 1],
                    candidates=None,
                    distance=np.full(length - 1, length),
                )
            )
    return rows


def mqar_rows(keys, values, rng):
    """K stored 'key: value' lines, then 16 queried keys with teacher forcing."""
    rows = []
    for count in MQAR_PAIRS:
        for _ in range(ROWS["mqar"]):
            key = rng.choice(keys, size=count, replace=False)
            value = rng.choice(values, size=count, replace=False)
            store = np.stack(
                [key, np.full(count, COLON), value, np.full(count, NEWLINE)], axis=1
            ).reshape(-1)
            chosen = rng.choice(count, size=MQAR_QUERIES, replace=False)
            query = np.stack(
                [
                    key[chosen],
                    np.full(MQAR_QUERIES, COLON),
                    value[chosen],
                    np.full(MQAR_QUERIES, NEWLINE),
                ],
                axis=1,
            ).reshape(-1)
            positions = 4 * count + 4 * np.arange(MQAR_QUERIES) + 1
            rows.append(
                dict(
                    family="mqar",
                    group=count,
                    ids=np.concatenate([store, query]),
                    positions=positions,
                    targets=value[chosen],
                    candidates=value,
                    distance=positions - (4 * chosen + 1),
                )
            )
    return rows


def passkey_text(key, before, after):
    information = f"The pass key is {key}. Remember it. {key} is the pass key."
    lines = [
        DESCRIPTION,
        " ".join([GARBAGE] * before),
        information,
        " ".join([GARBAGE] * after),
        QUESTION,
    ]
    return "\n".join(lines)


def passkey_rows(tokenizer, rng):
    """Largest filler count whose prompt plus answer fits the target length."""

    def encode(text):
        return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids)

    unit = len(encode(" " + GARBAGE))
    rows = []
    for length in PASSKEY_LENGTHS:
        for depth_index, depth in enumerate(PASSKEY_DEPTHS):
            for _ in range(ROWS["passkey"]):
                key = int(rng.integers(10000, 100000))
                answer = encode(f" {key}")

                def prompt_for(total, key=key, depth=depth):
                    before = int(round(depth * total))
                    return encode(passkey_text(key, before, total - before)), before

                def fits(total, answer=answer, length=length):
                    return len(prompt_for(total)[0]) + len(answer) <= length

                overhead = len(prompt_for(0)[0]) + len(answer)
                total = max(0, (length - overhead) // unit)
                while fits(total + 1):
                    total += 1
                while total > 0 and not fits(total):
                    total -= 1
                prompt, before = prompt_for(total)
                sequence = np.concatenate([prompt, answer])
                if len(sequence) > length:
                    raise ValueError("Passkey prompt exceeds its target length")
                positions = len(prompt) - 1 + np.arange(len(answer))
                header = encode(
                    passkey_text(key, before, total - before).split("The pass key is")[
                        0
                    ]
                )
                rows.append(
                    dict(
                        family="passkey",
                        group=length * 10 + depth_index,
                        ids=sequence,
                        positions=positions,
                        targets=answer,
                        candidates=None,
                        distance=positions - len(header),
                    )
                )
    return rows


def pack(rows):
    """Group rows by padded length; pad ids with EOS and score slots with -1."""
    arrays, metadata = {}, {}
    for bucket in BUCKETS:
        members = [row for row in rows if bucket_for(len(row["ids"])) == bucket]
        if not members:
            continue
        width = max(len(row["positions"]) for row in members)
        candidates = max(
            len(row["candidates"]) if row["candidates"] is not None else 1
            for row in members
        )
        count = len(members)
        ids = np.full((count, bucket), EOS, np.int32)
        positions = np.full((count, width), -1, np.int32)
        targets = np.zeros((count, width), np.int32)
        distance = np.zeros((count, width), np.int32)
        choice = np.zeros((count, candidates), np.int32)
        family = np.zeros(count, np.int32)
        group = np.zeros(count, np.int32)
        for index, row in enumerate(members):
            ids[index, : len(row["ids"])] = row["ids"]
            scored = len(row["positions"])
            positions[index, :scored] = row["positions"]
            targets[index, :scored] = row["targets"]
            distance[index, :scored] = row["distance"]
            options = (
                row["candidates"]
                if row["candidates"] is not None
                else row["targets"][:1]
            )
            choice[index, : len(options)] = options
            choice[index, len(options) :] = options[0]
            family[index] = FAMILIES.index(row["family"])
            group[index] = row["group"]
        prefix = f"b{bucket}"
        arrays.update(
            {
                f"{prefix}_input_ids": ids,
                f"{prefix}_positions": positions,
                f"{prefix}_targets": targets,
                f"{prefix}_distance": distance,
                f"{prefix}_candidates": choice,
                f"{prefix}_family": family,
                f"{prefix}_group": group,
            }
        )
        metadata[str(bucket)] = {"rows": count, "score_width": width}
    return arrays, metadata


def long_documents(tokenizer, screen_data, fresh_data):
    """Unused long FineWeb-Edu documents from the pinned fresh shard.

    Excludes every screen-corpus document hash (training and held-out) and
    every document of the fresh confirmatory holdout. Each kept document
    supplies [EOS] + its first 16,384 tokens.
    """
    import pyarrow.parquet as pq

    manifest = json.loads((fresh_data / "manifest.json").read_text())
    if sha256_file(SHARD) != manifest["source"]["lfs_sha256"]:
        raise ValueError("Fresh shard hash differs from its manifest")
    ledger = np.load(screen_data / "documents.npy", mmap_mode="r")
    excluded = set(np.asarray(ledger["sha256"]).view("S32").reshape(-1).tolist())
    used = json.loads((fresh_data / "documents.json").read_text())["rows"]
    excluded |= {bytes.fromhex(row[4]) for row in used}
    documents, seen = [], set()
    for batch in pq.ParquetFile(SHARD).iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if len(text) < 40_000:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in excluded or digest in seen:
                continue
            seen.add(digest)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) >= LONG_CONTEXT:
                documents.append((digest.hex(), [EOS] + ids[:LONG_CONTEXT]))
    order = np.random.default_rng(SEED).permutation(len(documents))
    keep = min(LONG_DOCUMENTS, len(documents) // 16 * 16)
    chosen = [documents[index] for index in order[:keep]]
    tokens = np.asarray([ids for _, ids in chosen], np.uint16)
    return tokens, [digest for digest, _ in chosen], len(documents)


def array_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_tasks(tokenizer_path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (
        tokenizer.get_vocab_size() != 50257
        or tokenizer.token_to_id("<|endoftext|>") != EOS
    ):
        raise ValueError("The claims tasks require the pinned GPT2 tokenizer")
    words, keys, values = word_pools(tokenizer)
    rng = np.random.default_rng(SEED)
    rows = copy_rows(words, rng) + mqar_rows(keys, values, rng)
    rows += passkey_rows(tokenizer, rng)
    arrays, metadata = pack(rows)
    return tokenizer, arrays, metadata, len(words)


def build(options):
    tokenizer_path = Path(options.screen_data) / "tokenizer.json"
    tokenizer, arrays, metadata, words = build_tasks(tokenizer_path)
    tokens, hashes, available = long_documents(
        tokenizer, Path(options.screen_data), Path(options.fresh_data)
    )
    arrays["long_tokens"] = tokens
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "tasks.npz", **arrays)
    manifest = {
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "array_sha256": array_digest(arrays),
        "file_sha256": sha256_file(output / "tasks.npz"),
        "word_pool_tokens": words,
        "buckets": metadata,
        "families": {
            "copy": {"lengths": COPY_LENGTHS, "rows_per_length": ROWS["copy"]},
            "mqar": {
                "pairs": MQAR_PAIRS,
                "queries": MQAR_QUERIES,
                "rows_per_load": ROWS["mqar"],
            },
            "passkey": {
                "lengths": PASSKEY_LENGTHS,
                "depths": PASSKEY_DEPTHS,
                "rows_per_cell": ROWS["passkey"],
            },
        },
        "long_documents": {
            "source": str(SHARD.relative_to(ROOT)),
            "source_sha256": sha256_file(SHARD),
            "available": available,
            "kept": len(hashes),
            "context": LONG_CONTEXT,
            "document_sha256": hashes,
            "exclusions": "every screen-corpus document hash and every fresh-holdout document",
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "long_documents"}))
    print("long documents kept", len(hashes), "of", available)


def load_tasks(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "tasks.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    if array_digest(arrays) != manifest["array_sha256"]:
        raise ValueError("Claims task arrays differ from their manifest")
    return arrays, manifest


@contextmanager
def mamba4_probe(mode):
    """Capture per-token memory variance, or zero the conjugate read.

    The wrapper only observes or replaces the read inside the evaluation
    process; trained parameters and model sources are untouched.
    """
    import jax.numpy as jnp
    from flax.linen import module as flax_module

    import lm.models.mamba4 as model_module

    original = model_module.selective_gaussian_memory

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if mode == "variance":
            flax_module._context.module_stack[-1].sow(
                "intermediates", "memory_variance", result.variance
            )
        elif mode == "no_read":
            result = result._replace(output=jnp.zeros_like(result.output))
        return result

    model_module.selective_gaussian_memory = wrapped
    try:
        yield
    finally:
        model_module.selective_gaussian_memory = original


def scored_function(model, capture):
    """Per scored slot: target log probability, top-1 hit, confidence."""
    import jax
    import jax.numpy as jnp

    def function(params, ids, positions, targets, candidates):
        if capture:
            logits, state = model.apply(
                {"params": params}, ids, mutable=["intermediates"]
            )
        else:
            logits, state = model.apply({"params": params}, ids), None
        rows = jnp.arange(ids.shape[0])[:, None]
        safe = jnp.maximum(positions, 0)
        logp = jax.nn.log_softmax(logits[rows, safe].astype(jnp.float32), axis=-1)
        target = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        options = jnp.broadcast_to(
            candidates[:, None, :], (*positions.shape, candidates.shape[-1])
        )
        best_option = jnp.max(jnp.take_along_axis(logp, options, axis=-1), axis=-1)
        output = {
            "target_logp": target,
            "correct": jnp.argmax(logp, axis=-1) == targets,
            "confidence": jnp.exp(jnp.max(logp, axis=-1)),
            "candidate_correct": target >= best_option,
        }
        if capture:
            leaves = jax.tree_util.tree_flatten_with_path(state["intermediates"])[0]
            ordered = sorted(
                leaves,
                key=lambda item: int(
                    re.search(r"layer_(\d+)", jax.tree_util.keystr(item[0])).group(1)
                ),
            )
            output["variance"] = jnp.stack(
                [leaf[rows, safe] for _, leaf in ordered], axis=1
            )
        return output

    return jax.pmap(function)


def long_function(model):
    """Per-document NLL by position bucket, calibration histograms, NLL curve."""
    import jax
    import jax.numpy as jnp

    edges = jnp.asarray(POSITION_EDGES[1:])

    def function(params, ids, targets):
        logp = jax.nn.log_softmax(
            model.apply({"params": params}, ids).astype(jnp.float32), axis=-1
        )
        target = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        correct = (jnp.argmax(logp, axis=-1) == targets).astype(jnp.float32)
        confidence = jnp.exp(jnp.max(logp, axis=-1))
        position = jnp.arange(ids.shape[1])
        bucket = jax.nn.one_hot(
            jnp.searchsorted(edges, position, side="right"), len(edges)
        )
        bins = jnp.minimum(
            (confidence * CALIBRATION_BINS).astype(jnp.int32), CALIBRATION_BINS - 1
        )
        beyond = (position >= TRAINING_CONTEXT).astype(jnp.int32)
        slot = jax.nn.one_hot(beyond * CALIBRATION_BINS + bins, 2 * CALIBRATION_BINS)
        return {
            "bucket_nll": jnp.einsum("bt,tk->bk", -target, bucket),
            "bucket_count": jnp.broadcast_to(
                jnp.sum(bucket, axis=0), (ids.shape[0], len(edges))
            ),
            "hist_count": jnp.sum(slot, axis=1),
            "hist_confidence": jnp.einsum("bt,btk->bk", confidence, slot),
            "hist_correct": jnp.einsum("bt,btk->bk", correct, slot),
            "target_logp": target,
        }

    return jax.pmap(function)


def load_model(config_path, run):
    """Create the frozen architecture and restore its final checkpoint."""
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from flax.training.train_state import TrainState

    from lm.config import ModelConfig, TrainConfig
    from lm.train import create_model, optimizer

    frozen = json.loads(Path(config_path).read_text())
    config = ModelConfig(**frozen["model"])
    training = TrainConfig(**frozen["training"])
    model = create_model(config)
    params = model.init(
        jax.random.PRNGKey(training.seed),
        jnp.zeros((1, config.chunk_size), jnp.int32),
        train=False,
    )["params"]
    tx, _ = optimizer(training, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pointer = json.loads((Path(run) / "latest.json").read_text())["checkpoint"]
    payload = (Path(run) / pointer / "state.msgpack").read_bytes()
    state = serialization.from_bytes(state, payload)
    return config, model, state.params, hashlib.sha256(payload).hexdigest()


def run_rows(mapped, replicated, arrays, per_device, timings):
    """Score rows in fixed global order; pad the final batch with copies."""
    import jax
    from jax.experimental import multihost_utils

    names = list(arrays)
    total = len(arrays[names[0]])
    global_batch = per_device * jax.device_count()
    local = global_batch // jax.process_count()
    steps = -(-total // global_batch)
    collected = {}
    for step in range(steps):
        index = np.arange(step * global_batch, (step + 1) * global_batch) % total
        start = jax.process_index() * local
        batch = [
            arrays[name][index[start : start + local]].reshape(
                jax.local_device_count(), per_device, *arrays[name].shape[1:]
            )
            for name in names
        ]
        started = time.perf_counter()
        output = jax.device_get(mapped(replicated, *batch))
        timings.append(time.perf_counter() - started)
        for key, value in output.items():
            value = np.asarray(value)
            gathered = np.asarray(multihost_utils.process_allgather(value))
            gathered = gathered.reshape(global_batch, *value.shape[2:])
            collected.setdefault(key, []).append(gathered)
    return {key: np.concatenate(values)[:total] for key, values in collected.items()}


def evaluate(options):
    import jax

    from lm.runtime import initialize

    hardware = initialize(True)
    arrays, manifest = load_tasks(options.tasks)
    raw = Path(options.output)
    summary = {
        "hardware": hardware,
        "task_manifest_sha256": sha256_file(Path(options.tasks) / "manifest.json"),
        "task_array_sha256": manifest["array_sha256"],
        "models": {},
    }
    for item in options.models:
        name, paths = item.split("=", 1)
        config_path, run = paths.split(",")
        config, model, params, digest = load_model(config_path, run)
        replicated = jax.device_put_replicated(params, jax.local_devices())
        variants = [(name, None)]
        if config.architecture == "mamba4":
            variants = [(name, "variance"), (f"{name}_no_read", "no_read")]
        for label, mode in variants:
            entry = {
                "config": config_path,
                "checkpoint": run,
                "checkpoint_sha256": digest,
                "probe": mode,
                "seconds": {},
            }
            context = mamba4_probe(mode) if mode else _nothing()
            with context:
                outputs = {}
                for bucket in BUCKETS:
                    prefix = f"b{bucket}"
                    if f"{prefix}_input_ids" not in arrays:
                        continue
                    mapped = scored_function(model, mode == "variance")
                    timings = []
                    result = run_rows(
                        mapped,
                        replicated,
                        {
                            key: arrays[f"{prefix}_{key}"]
                            for key in (
                                "input_ids",
                                "positions",
                                "targets",
                                "candidates",
                            )
                        },
                        PER_DEVICE[bucket],
                        timings,
                    )
                    entry["seconds"][prefix] = timings
                    for key, value in result.items():
                        outputs[f"{prefix}_{key}"] = value
                    print(label, prefix, f"{sum(timings):.1f}s", flush=True)
                tokens = arrays["long_tokens"].astype(np.int32)
                timings = []
                result = run_rows(
                    long_function(model),
                    replicated,
                    {"input_ids": tokens[:, :-1], "targets": tokens[:, 1:]},
                    PER_DEVICE[LONG_CONTEXT],
                    timings,
                )
                entry["seconds"]["long"] = timings
                for key, value in result.items():
                    outputs[f"long_{key}"] = value
                print(label, "long", f"{sum(timings):.1f}s", flush=True)
            if jax.process_index() == 0:
                raw.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw / f"{label}.npz", **outputs)
                entry["raw"] = str(raw / f"{label}.npz")
                entry["raw_sha256"] = sha256_file(raw / f"{label}.npz")
            summary["models"][label] = entry
    if jax.process_index() == 0:
        atomic_json(raw / "evaluation.json", summary)
    jax.distributed.shutdown()


@contextmanager
def _nothing():
    yield


def decode(options):
    """Median one-token latency with a cache sized for each context.

    Transformer attention reads the whole cache, so its cost follows the
    cache capacity; recurrent caches have fixed shapes. Timing does not
    depend on cache contents, so caches start empty.
    """
    import dataclasses

    import jax
    import jax.numpy as jnp

    from lm.decode import decode_step, initialize_cache
    from lm.runtime import initialize

    hardware = initialize(True)
    results = {"hardware": hardware, "models": {}}
    for item in options.models:
        name, paths = item.split("=", 1)
        config_path, run = paths.split(",")
        config, _, params, digest = load_model(config_path, run)
        replicated = jax.device_put_replicated(params, jax.local_devices())
        entry = {"checkpoint_sha256": digest, "contexts": {}}
        for context in DECODE_CONTEXTS:
            local = dataclasses.replace(
                config, max_seq_len=context + DECODE_STEPS + 8, remat=False
            )
            per_sequence = initialize_cache(local, 1)
            cache_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(per_sequence))
            cache = jax.pmap(lambda _: initialize_cache(local, 1))(
                jnp.zeros(jax.local_device_count())
            )
            mapped = jax.pmap(
                _decoder(decode_step, local),
                in_axes=(0, 0, 0, None),
                donate_argnums=(2,),
            )
            tokens = jnp.ones((jax.local_device_count(), 1), jnp.int32)
            seconds = []
            for offset in range(DECODE_STEPS + 3):
                started = time.perf_counter()
                logits, cache = mapped(
                    replicated, tokens, cache, jnp.asarray(context + offset, jnp.int32)
                )
                jax.block_until_ready(logits)
                seconds.append(time.perf_counter() - started)
            entry["contexts"][str(context)] = {
                "cache_bytes_per_sequence": int(cache_bytes),
                "median_step_seconds": float(np.median(seconds[3:])),
                "step_seconds": seconds,
                "finite_logits": bool(np.isfinite(np.asarray(logits)).all()),
            }
            print(name, context, entry["contexts"][str(context)]["median_step_seconds"])
        results["models"][name] = entry
    if jax.process_index() == 0:
        atomic_json(Path(options.output) / "decode.json", results)
    jax.distributed.shutdown()


def _decoder(step, config):
    def function(params, token, cache, position):
        return step(config, params, token, cache, position)

    return function


# ----------------------------------------------------------------------------
# Statistics (CPU)


def wilson(successes, count):
    if count == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    )
    return [center - radius, center + radius]


def paired_test(first, second, rng):
    """Mean difference, bootstrap 95% interval and two-sided paired t p-value."""
    from scipy import stats

    difference = np.asarray(first, np.float64) - np.asarray(second, np.float64)
    count = len(difference)
    means = difference[rng.integers(count, size=(RESAMPLES, count))].mean(axis=1)
    if np.allclose(difference, difference[0]):
        p = 1.0 if difference[0] == 0 else 0.0
    else:
        p = float(stats.ttest_1samp(difference, 0.0).pvalue)
    return {
        "mean": float(difference.mean()),
        "bootstrap_95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "p": p,
    }


def mcnemar(first, second):
    """Exact two-sided McNemar test for paired binary outcomes."""
    from scipy import stats

    only_first = int(np.sum(first & ~second))
    only_second = int(np.sum(~first & second))
    discordant = only_first + only_second
    p = (
        1.0
        if discordant == 0
        else float(stats.binomtest(only_first, discordant, 0.5).pvalue)
    )
    return {"only_first": only_first, "only_second": only_second, "p": p}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def expected_calibration_error(confidence, correct, bins=CALIBRATION_BINS):
    confidence = np.asarray(confidence, np.float64)
    correct = np.asarray(correct, np.float64)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    confidence_sum = np.bincount(index, confidence, minlength=bins)
    correct_sum = np.bincount(index, correct, minlength=bins)
    return float(np.sum(np.abs(confidence_sum - correct_sum)) / max(1, len(index)))


def histogram_ece(count, confidence, correct):
    total = count.sum()
    return float(np.sum(np.abs(confidence - correct)) / max(1.0, total))


def auroc(score, label):
    """Probability that a random positive outscores a random negative."""
    from scipy import stats

    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float(
        (ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def row_metrics(arrays, outputs):
    """Per-row (sampling unit) statistics for the scored families."""
    rows = {family: {} for family in FAMILIES}
    for bucket in BUCKETS:
        prefix = f"b{bucket}"
        if f"{prefix}_family" not in arrays:
            continue
        valid = arrays[f"{prefix}_positions"] >= 0
        family = arrays[f"{prefix}_family"]
        group = arrays[f"{prefix}_group"]
        correct = outputs[f"{prefix}_correct"] & valid
        candidate = outputs[f"{prefix}_candidate_correct"] & valid
        logp = np.where(valid, outputs[f"{prefix}_target_logp"], 0.0)
        confidence = outputs[f"{prefix}_confidence"]
        scored = valid.sum(axis=1)
        variance = outputs.get(f"{prefix}_variance")
        for index in range(len(family)):
            name = FAMILIES[family[index]]
            slots = valid[index]
            record = rows[name].setdefault(int(group[index]), [])
            entry = {
                "accuracy": float(correct[index].sum() / scored[index]),
                "exact": bool(correct[index][slots].all()),
                "candidate_accuracy": float(candidate[index].sum() / scored[index]),
                "nll": float(-logp[index].sum() / scored[index]),
                "confidence": confidence[index][slots].astype(np.float64),
                "hits": correct[index][slots],
            }
            if variance is not None:
                entry["variance"] = variance[index][:, slots].mean(axis=-1)
            record.append(entry)
    return rows


def compare_families(models, rng):
    """Pre-registered retrieval and passkey comparisons against both peers."""
    report = {}
    for family in FAMILIES:
        metric = "exact" if family == "passkey" else "accuracy"
        groups = sorted(models["mamba4"][family])
        table = {}
        for group in groups:
            entry = {}
            for name, rows in models.items():
                values = [row[metric] for row in rows[family][group]]
                entry[name] = {
                    "mean": float(np.mean(values)),
                    "exact_rate": float(
                        np.mean([row["exact"] for row in rows[family][group]])
                    ),
                    "nll": float(np.mean([row["nll"] for row in rows[family][group]])),
                    "rows": len(values),
                }
                if family == "mqar":
                    entry[name]["candidate_accuracy"] = float(
                        np.mean(
                            [row["candidate_accuracy"] for row in rows[family][group]]
                        )
                    )
            table[group] = entry
        comparisons = {}
        for peer in ("transformer", "mamba3", "mamba4_no_read"):
            if peer not in models:
                continue
            tests = {}
            for group in groups:
                mine = models["mamba4"][family][group]
                theirs = models[peer][family][group]
                if family == "passkey":
                    tests[group] = mcnemar(
                        np.array([row["exact"] for row in mine]),
                        np.array([row["exact"] for row in theirs]),
                    )
                    tests[group]["mean"] = float(
                        np.mean([row["exact"] for row in mine])
                        - np.mean([row["exact"] for row in theirs])
                    )
                else:
                    tests[group] = paired_test(
                        [row[metric] for row in mine],
                        [row[metric] for row in theirs],
                        rng,
                    )
            adjusted = holm(np.array([tests[group]["p"] for group in groups]))
            for group, value in zip(groups, adjusted):
                tests[group]["holm_p"] = float(value)
                mean = tests[group]["mean"]
                tests[group]["verdict"] = (
                    "matches"
                    if value >= 0.05
                    else ("exceeds" if mean > 0 else "trails")
                )
            comparisons[peer] = tests
        report[family] = {"metric": metric, "groups": table, "mamba4_vs": comparisons}
    return report


def _row_bins(row):
    """Per-row calibration sufficient statistics: count, confidence, hits."""
    index = np.minimum(
        (row["confidence"] * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1
    )
    return np.stack(
        [
            np.bincount(index, minlength=CALIBRATION_BINS),
            np.bincount(index, row["confidence"], minlength=CALIBRATION_BINS),
            np.bincount(index, row["hits"], minlength=CALIBRATION_BINS),
        ]
    )


def _ece_from_bins(bins):
    """bins [..., 3, CALIBRATION_BINS] summed over rows."""
    return np.sum(np.abs(bins[..., 1, :] - bins[..., 2, :]), axis=-1) / np.maximum(
        1.0, np.sum(bins[..., 0, :], axis=-1)
    )


def calibration(models, long_outputs, rng):
    """Answer-slot ECE with row-resampled paired intervals; LM-level ECE."""
    report, bins = {}, {}
    for name, rows in models.items():
        entry = {}
        for family in FAMILIES:
            ordered = [
                row for group in sorted(rows[family]) for row in rows[family][group]
            ]
            confidence = np.concatenate([row["confidence"] for row in ordered])
            hits = np.concatenate([row["hits"] for row in ordered])
            bins[name, family] = np.stack([_row_bins(row) for row in ordered])
            entry[family] = {
                "ece": expected_calibration_error(confidence, hits),
                "auroc": auroc(confidence, hits),
                "mean_confidence": float(confidence.mean()),
                "accuracy": float(hits.mean()),
                "slots": int(len(hits)),
            }
            if "variance" in ordered[0]:
                variance = np.concatenate([row["variance"] for row in ordered], axis=1)
                entry[family]["variance_auroc_by_layer"] = [
                    auroc(-layer, hits) for layer in variance
                ]
        if name in long_outputs:
            out = long_outputs[name]
            for label, part in (
                ("within_1024", slice(0, CALIBRATION_BINS)),
                ("beyond_1024", slice(CALIBRATION_BINS, None)),
            ):
                count = out["long_hist_count"][:, part].sum(axis=0)
                conf = out["long_hist_confidence"][:, part].sum(axis=0)
                hit = out["long_hist_correct"][:, part].sum(axis=0)
                entry[f"long_{label}"] = {
                    "ece": histogram_ece(count, conf, hit),
                    "mean_confidence": float(conf.sum() / count.sum()),
                    "accuracy": float(hit.sum() / count.sum()),
                    "tokens": int(count.sum()),
                }
        report[name] = entry
    for peer in ("transformer", "mamba3"):
        if peer not in models:
            continue
        for family in FAMILIES:
            mine, theirs = bins["mamba4", family], bins[peer, family]
            draws = rng.integers(len(mine), size=(1000, len(mine)))
            difference = _ece_from_bins(mine[draws].sum(axis=1)) - _ece_from_bins(
                theirs[draws].sum(axis=1)
            )
            interval = [
                float(np.percentile(difference, 2.5)),
                float(np.percentile(difference, 97.5)),
            ]
            report["mamba4"][family][f"ece_minus_{peer}"] = {
                "mean": report["mamba4"][family]["ece"] - report[peer][family]["ece"],
                "bootstrap_95": interval,
                "verdict": (
                    "exceeds"
                    if interval[1] < 0
                    else ("trails" if interval[0] > 0 else "matches")
                ),
            }
    return report


def long_context(long_outputs, rng):
    """Per-document NLL by position bucket with paired comparisons."""
    edges = list(POSITION_EDGES)
    names = list(long_outputs)
    per_document = {
        name: out["long_bucket_nll"] / out["long_bucket_count"]
        for name, out in long_outputs.items()
    }
    buckets = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    table = {
        name: {b: float(per_document[name][:, i].mean()) for i, b in enumerate(buckets)}
        for name in names
    }
    comparisons = {}
    for peer in ("transformer", "mamba3", "mamba4_no_read"):
        if peer not in per_document:
            continue
        tests = {
            b: paired_test(per_document["mamba4"][:, i], per_document[peer][:, i], rng)
            for i, b in enumerate(buckets)
        }
        adjusted = holm(np.array([tests[b]["p"] for b in buckets]))
        for b, value in zip(buckets, adjusted):
            tests[b]["holm_p"] = float(value)
            tests[b]["verdict"] = (
                "matches"
                if value >= 0.05
                else ("exceeds" if tests[b]["mean"] < 0 else "trails")
            )
        comparisons[peer] = tests
    self_test = {}
    for name in names:
        late = per_document[name][:, buckets.index("8192-16384")]
        early = per_document[name][:, buckets.index("512-1024")]
        self_test[name] = paired_test(late, early, rng)
    curves = {
        name: (-out["long_target_logp"].mean(axis=0)).tolist()
        for name, out in long_outputs.items()
    }
    return {
        "documents": int(len(per_document["mamba4"])),
        "nll_by_bucket": table,
        "mamba4_vs": comparisons,
        "late_minus_early": self_test,
    }, curves


def verdicts(families, context, calibration_report, decoding):
    """Apply the pre-registered claim rules of PROTOCOL.md."""
    within = {
        "copy": [g for g in families["copy"]["groups"] if 2 * g <= TRAINING_CONTEXT],
        "mqar": [
            g for g in families["mqar"]["groups"] if 4 * g + 64 <= TRAINING_CONTEXT
        ],
    }
    beyond = {
        "copy": [g for g in families["copy"]["groups"] if 2 * g > TRAINING_CONTEXT],
        "mqar": [
            g for g in families["mqar"]["groups"] if 4 * g + 64 > TRAINING_CONTEXT
        ],
    }
    result = {}
    for peer in ("transformer", "mamba3"):
        calls = [
            families[f]["mamba4_vs"][peer][g]["verdict"]
            for f in ("copy", "mqar")
            for g in within[f]
        ]
        retrieval = (
            "refuted"
            if "trails" in calls
            else ("supported" if "exceeds" in calls else "matched")
        )
        calls_beyond = [
            families[f]["mamba4_vs"][peer][g]["verdict"]
            for f in ("copy", "mqar")
            for g in beyond[f]
        ]
        passkey_long = [
            families["passkey"]["mamba4_vs"][peer][g]["verdict"]
            for g in families["passkey"]["groups"]
            if g // 10 > TRAINING_CONTEXT
        ]
        nll_long = [
            context["mamba4_vs"][peer][b]["verdict"]
            for b in ("1024-2048", "2048-4096", "4096-8192", "8192-16384")
        ]
        long_calls = calls_beyond + passkey_long + nll_long
        tolerant = context["late_minus_early"]["mamba4"]["bootstrap_95"][1] <= 0
        long_verdict = (
            "refuted"
            if "trails" in long_calls or not tolerant
            else ("supported" if "exceeds" in long_calls else "matched")
        )
        ece_calls = [
            calibration_report["mamba4"][f][f"ece_minus_{peer}"]["verdict"]
            for f in FAMILIES
        ]
        result[peer] = {
            "exact_retrieval_within_training_context": retrieval,
            "long_context": long_verdict,
            "calibrated_confidence": (
                "refuted"
                if "trails" in ece_calls
                else ("supported" if "exceeds" in ece_calls else "matched")
            ),
        }
    if decoding:
        own = decoding["models"]["mamba4"]["contexts"]
        times = [own[str(c)]["median_step_seconds"] for c in DECODE_CONTEXTS]
        transformer = decoding["models"]["transformer"]["contexts"]
        last = str(DECODE_CONTEXTS[-1])
        flat = max(times) <= 1.1 * min(times)
        faster = (
            own[last]["median_step_seconds"] < transformer[last]["median_step_seconds"]
        )
        result["context_independent_decode"] = (
            "supported" if flat and faster else "not_supported"
        )
    return result


def report(options):
    arrays, manifest = load_tasks(options.tasks)
    raw = Path(options.raw)
    evaluation = json.loads((raw / "evaluation.json").read_text())
    if evaluation["task_array_sha256"] != manifest["array_sha256"]:
        raise ValueError("Evaluation used different task arrays")
    outputs, long_outputs = {}, {}
    for name, entry in evaluation["models"].items():
        if sha256_file(Path(entry["raw"])) != entry["raw_sha256"]:
            raise ValueError(f"Raw scores for {name} changed after evaluation")
        with np.load(entry["raw"]) as stored:
            data = {key: stored[key] for key in stored.files}
        outputs[name] = row_metrics(arrays, data)
        long_outputs[name] = {k: v for k, v in data.items() if k.startswith("long_")}
    rng = np.random.default_rng(SEED)
    families = compare_families(outputs, rng)
    context, curves = long_context(long_outputs, rng)
    calibration_report = calibration(outputs, long_outputs, rng)
    decoding = (
        json.loads((raw / "decode.json").read_text())
        if (raw / "decode.json").exists()
        else None
    )
    summary = {
        "protocol": "claims-60m",
        "protocol_sha256": sha256_file(ROOT / "validation/claims/PROTOCOL.md"),
        "task_manifest": manifest["array_sha256"],
        "evaluation": {
            name: {
                k: v
                for k, v in entry.items()
                if k in ("checkpoint", "checkpoint_sha256", "probe", "raw_sha256")
            }
            for name, entry in evaluation["models"].items()
        },
        "families": families,
        "long_context": context,
        "calibration": calibration_report,
        "decode": decoding["models"] if decoding else None,
        "verdicts": verdicts(families, context, calibration_report, decoding),
        "prefill_seconds_per_16k_document": {
            name: float(np.median(entry["seconds"]["long"][1:]))
            / PER_DEVICE[LONG_CONTEXT]
            for name, entry in evaluation["models"].items()
        },
    }
    output = Path(options.output)
    atomic_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "long-context-curves.npz",
        **{name: np.asarray(curve, np.float32) for name, curve in curves.items()},
    )
    per_row = {
        name: {
            family: {
                str(group): {
                    "accuracy": [row["accuracy"] for row in rows],
                    "exact": [row["exact"] for row in rows],
                    "candidate_accuracy": [row["candidate_accuracy"] for row in rows],
                    "nll": [row["nll"] for row in rows],
                }
                for group, rows in models[family].items()
            }
            for family in FAMILIES
        }
        for name, models in outputs.items()
    }
    atomic_json(output / "per-row.json", per_row)
    (output / "REPORT.md").write_text(markdown(summary))
    print(json.dumps(summary["verdicts"], indent=2))


NAMES = {
    "transformer": "Transformer",
    "mamba3": "Mamba-3",
    "mamba4": "Mamba 4",
    "mamba4_no_read": "Mamba 4, read zeroed",
}


def _percent(value):
    return f"{100 * value:.1f}%"


def markdown(summary):
    """Human-readable results; verdict rules are those of PROTOCOL.md."""
    families, context = summary["families"], summary["long_context"]
    models = [name for name in NAMES if name in summary["evaluation"]]
    header = "| " + " | ".join(NAMES[m] for m in models) + " |"
    lines = [
        "# Claims-60m results",
        "",
        "Pre-registered tests (`validation/claims/PROTOCOL.md`) of the paper's",
        "retrieval, long-context, calibration and decode claims on the trained",
        "60M checkpoints. One training seed; prompts or documents are the",
        "sampling units. Verdicts compare Mamba 4 with each peer under the frozen",
        'rules; "read zeroed" is an evaluation-time ablation, not a model.',
        "",
        "## Verdicts",
        "",
        "| Claim | vs Transformer | vs Mamba-3 |",
        "|---|---|---|",
    ]
    verdict = summary["verdicts"]
    for key, label in (
        ("exact_retrieval_within_training_context", "Exact retrieval (within 1,024)"),
        ("long_context", "Long context"),
        ("calibrated_confidence", "Calibrated confidence"),
    ):
        lines.append(
            f"| {label} | {verdict['transformer'][key]} | {verdict['mamba3'][key]} |"
        )
    if "context_independent_decode" in verdict:
        lines.append(
            "| Context-independent decode | "
            f"{verdict['context_independent_decode']} | — |"
        )
    for family, title, unit in (
        ("copy", "Exact copy: top-1 accuracy on the second copy", "L"),
        ("mqar", "Associative recall: top-1 accuracy at 16 queries", "K"),
    ):
        lines += ["", f"## {title}", "", f"| {unit} |" + header[1:] + " vs T | vs M3 |"]
        lines.append("|---:" * (len(models) + 1) + "|---|---|")
        for group, entry in sorted(
            families[family]["groups"].items(), key=lambda x: int(x[0])
        ):
            cells = " | ".join(_percent(entry[m]["mean"]) for m in models)
            tests = families[family]["mamba4_vs"]
            lines.append(
                f"| {group} | {cells} | {tests['transformer'][group]['verdict']}"
                f" | {tests['mamba3'][group]['verdict']} |"
            )
    lines += [
        "",
        "## Passkey retrieval: exact match, pooled over five depths",
        "",
        "| Length |" + header[1:],
        "|---:" * (len(models) + 1) + "|",
    ]
    pooled = {}
    for group, entry in families["passkey"]["groups"].items():
        for m in models:
            pooled.setdefault(int(group) // 10, {}).setdefault(m, []).append(
                entry[m]["mean"]
            )
    for length in sorted(pooled):
        cells = " | ".join(_percent(float(np.mean(pooled[length][m]))) for m in models)
        lines.append(f"| {length:,} | {cells} |")
    order = [f"{POSITION_EDGES[i]}-{POSITION_EDGES[i + 1]}" for i in range(7)]
    lines += [
        "",
        f"## Long-document NLL by position ({context['documents']} documents)",
        "",
        "| Positions |" + header[1:],
        "|---" + "|---:" * len(models) + "|",
    ]
    for bucket in order:
        cells = " | ".join(f"{context['nll_by_bucket'][m][bucket]:.3f}" for m in models)
        lines.append(f"| {bucket} | {cells} |")
    late = context["late_minus_early"]
    lines += [
        "",
        "NLL at 8,192–16,384 minus NLL at 512–1,024 (paired over documents): "
        + "; ".join(
            f"{NAMES[m]} {late[m]['mean']:+.3f} "
            f"[{late[m]['bootstrap_95'][0]:+.3f}, {late[m]['bootstrap_95'][1]:+.3f}]"
            for m in models
        )
        + ".",
        "",
        "## Calibration at answer slots",
        "",
        "| Family | " + " | ".join(f"{NAMES[m]} ECE / AUROC" for m in models) + " |",
        "|---" + "|---:" * len(models) + "|",
    ]
    for family in FAMILIES:
        cells = " | ".join(
            f"{summary['calibration'][m][family]['ece']:.4f} / "
            f"{summary['calibration'][m][family]['auroc']:.3f}"
            for m in models
        )
        lines.append(f"| {family} | {cells} |")
    variance = summary["calibration"]["mamba4"]
    lines += [
        "",
        "Mamba 4 memory-variance AUROC (lower variance predicting a correct",
        "answer), memory layers in order: "
        + "; ".join(
            f"{family} "
            + ", ".join(f"{v:.2f}" for v in variance[family]["variance_auroc_by_layer"])
            for family in FAMILIES
        )
        + ".",
    ]
    if summary["decode"]:
        timed = [m for m in models if m in summary["decode"]]
        lines += [
            "",
            "## One-token decode (median, one sequence per chip)",
            "",
            "| Context | "
            + " | ".join(f"{NAMES[m]} ms (cache MiB)" for m in timed)
            + " |",
            "|---:" + "|---:" * len(timed) + "|",
        ]
        for c in DECODE_CONTEXTS:
            cells = []
            for m in timed:
                entry = summary["decode"][m]["contexts"][str(c)]
                cells.append(
                    f"{1000 * entry['median_step_seconds']:.2f} "
                    f"({entry['cache_bytes_per_sequence'] / 2**20:.1f})"
                )
            lines.append(f"| {c:,} | " + " | ".join(cells) + " |")
    prefill = summary["prefill_seconds_per_16k_document"]
    lines += [
        "",
        "Full-sequence forward time per 16,384-token document (one per chip): "
        + "; ".join(f"{NAMES[m]} {prefill[m]:.3f} s" for m in models)
        + ".",
        "",
    ]
    return "\n".join(lines)


def launch(options):
    """Sync once, then evaluate and time decode through the pod controller."""
    import subprocess

    from scripts.pod import HOSTS, SSH
    from validation.run_screen_v2 import pod, wait_for_hosts

    v1, v2 = "lm/runs/screen-60m-v1", "lm/runs/screen-60m-v2"
    models = [
        f"transformer=lm/configs/transformer-60m.json,{v1}/transformer",
        f"mamba3=lm/configs/mamba3-60m.json,{v1}/mamba3",
        f"mamba4=lm/configs/mamba4-60m-v2.json,{v2}/mamba4",
    ]
    wait_for_hosts()
    subprocess.run(
        [sys.executable, "-m", "scripts.pod", "sync", "--data"], cwd=ROOT, check=True
    )
    # scripts.pod collects --output from every worker; create it everywhere.
    for host in HOSTS[1:]:
        subprocess.run(
            SSH + [f"tasma@{host}", f"mkdir -p {ROOT / RAW.relative_to(ROOT)}"],
            check=True,
        )
    if "evaluate" in options.stages:
        pod(
            "validation.claims",
            ["evaluate", "--output", str(RAW.relative_to(ROOT)), *models],
            "claims-60m-evaluate",
        )
    if "decode" in options.stages:
        pod(
            "validation.claims",
            ["decode", "--output", str(RAW.relative_to(ROOT)), *models],
            "claims-60m-decode",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("build")
    make.add_argument("--screen-data", default="data/fineweb-edu-1b")
    make.add_argument("--fresh-data", default="data/fresh-holdout-v2")
    make.add_argument("--output", default=str(TASKS.relative_to(ROOT)))
    for action in ("evaluate", "decode"):
        run = sub.add_parser(action)
        run.add_argument("--tasks", default=str(TASKS.relative_to(ROOT)))
        run.add_argument("--output", required=True)
        run.add_argument("models", nargs="+", help="name=config.json,run_directory")
    summarize = sub.add_parser("report")
    summarize.add_argument("--tasks", default=str(TASKS.relative_to(ROOT)))
    summarize.add_argument("--raw", default=str(RAW.relative_to(ROOT)))
    summarize.add_argument("--output", default=str(RESULTS.relative_to(ROOT)))
    start = sub.add_parser("launch")
    start.add_argument("--stages", nargs="+", default=["evaluate", "decode"])
    options = parser.parse_args()
    {
        "build": build,
        "evaluate": evaluate,
        "decode": decode,
        "report": report,
        "launch": launch,
    }[options.action](options)


if __name__ == "__main__":
    main()
