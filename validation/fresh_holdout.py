"""Fresh confirmatory holdout and per-sequence evaluation of final checkpoints.

``prepare`` (CPU) downloads one pinned shard of the official FineWeb-Edu
sample, removes every document whose exact UTF-8 SHA256 occurs anywhere in the
screen corpus (training or held-out), permutes the remaining unique documents
with a new seed and tokenizes them exactly like the screen corpus. No model or
training choice may use this set; it exists for the versioned v2 comparison.

``evaluate`` (pod only) loads final checkpoints and records per-sequence NLL
sums on the original held-out stream and on the fresh stream, so model
differences can receive paired whole-sequence intervals.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np

from lm.data import sha256_file
from lm.runtime import atomic_json


REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/013_00000.parquet"
SEED = 20261009
TARGETS = 2_097_152
EOS = 50256
FORMAT = "mamba4.fresh-holdout.v1"


def download(directory):
    tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/sample/10BT"
    with urllib.request.urlopen(tree_url, timeout=60) as response:
        tree = {item["path"]: item for item in json.load(response)}
    item = tree[SHARD]
    path = directory / Path(SHARD).name
    expected = item["lfs"]["oid"]
    if not (path.exists() and sha256_file(path) == expected):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SHARD}"
        temporary = path.with_suffix(".download")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=300) as response:
                    with temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
                if sha256_file(temporary) != expected:
                    raise ValueError("Downloaded shard has the wrong SHA256")
                temporary.replace(path)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    return path, {
        "repo": REPO,
        "revision": REVISION,
        "path": SHARD,
        "size_bytes": item["size"],
        "lfs_sha256": expected,
    }


def earlier_documents(path):
    """Document hashes recorded by a claims manifest or a holdout ledger."""
    record = json.loads(Path(path).read_text())
    if "long_documents" in record:
        return {bytes.fromhex(h) for h in record["long_documents"]["document_sha256"]}
    return {bytes.fromhex(row[4]) for row in record["rows"]}


def prepare(options):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    output = Path(options.output)
    (output / "source").mkdir(parents=True, exist_ok=True)
    screen = Path(options.screen_data)
    if options.shard:
        shard = Path(options.shard)
        source = json.loads(Path(options.shard_source).read_text())
        if sha256_file(shard) != source["lfs_sha256"]:
            raise ValueError("Local shard differs from its pinned hash")
    else:
        shard, source = download(output / "source")
    ledger = np.load(screen / "documents.npy", mmap_mode="r")
    excluded = set(np.asarray(ledger["sha256"]).view("S32").reshape(-1).tolist())
    earlier = set()
    for path in options.exclude:
        earlier |= earlier_documents(path)
    excluded |= earlier
    tokenizer_file = screen / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    parquet = pq.ParquetFile(shard)
    texts, hashes, seen = [], [], set()
    overlap, duplicates = 0, 0
    for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column(0).to_pylist():
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in excluded:
                overlap += 1
            elif digest in seen:
                duplicates += 1
            else:
                seen.add(digest)
                texts.append(text)
                hashes.append(digest)
    order = np.random.default_rng(options.seed).permutation(len(texts))
    stream, rows, written = [], [], 0
    for index in order:
        ids = tokenizer.encode(texts[index], add_special_tokens=False).ids + [EOS]
        take = min(len(ids), TARGETS + 1 - written)
        stream.append(np.asarray(ids[:take], dtype="<u2"))
        rows.append((int(index), written, take, len(ids), hashes[index].hex()))
        written += take
        if written == TARGETS + 1:
            break
    if written != TARGETS + 1:
        raise ValueError("The fresh shard cannot supply the evaluation budget")
    tokens = np.concatenate(stream)
    (output / "eval.bin").write_bytes(tokens.tobytes())
    atomic_json(
        output / "documents.json",
        {
            "columns": ["candidate_index", "offset", "tokens_used", "length", "sha256"],
            "rows": rows,
        },
    )
    manifest = {
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_rows": parquet.metadata.num_rows,
        "excluded_screen_hashes": len(excluded) - len(earlier),
        "excluded_earlier_evaluation_documents": len(earlier),
        "screen_overlap_rows_removed": overlap,
        "within_shard_duplicates_removed": duplicates,
        "candidate_documents": len(texts),
        "selected_documents": len(rows),
        "permutation": "numpy PCG64",
        "seed": options.seed,
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "tokenization": "GPT2, no implicit special tokens, one EOS per document",
        "token_count": TARGETS + 1,
        "target_count": TARGETS,
        "eval_sha256": sha256_file(output / "eval.bin"),
        "near_duplicate_exclusion": False,
        "use": "confirmatory only; never used for development or selection",
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


class Stream:
    """Batches with the screen's evaluation packing over one token stream."""

    def __init__(self, path, targets=TARGETS):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        if self.tokens.size != targets + 1:
            raise ValueError(f"{path} does not contain targets + 1 tokens")
        self.targets = targets

    def batch(self, step, global_batch, length):
        capacity = global_batch * length
        offset = step * capacity
        valid = min(capacity, self.targets - offset)
        inputs = np.full(capacity, EOS, np.int32)
        labels = np.full(capacity, EOS, np.int32)
        mask = np.zeros(capacity, np.float32)
        inputs[:valid] = self.tokens[offset : offset + valid]
        labels[:valid] = self.tokens[offset + 1 : offset + valid + 1]
        mask[:valid] = 1
        shape = (global_batch, length)
        return {
            "input_ids": inputs.reshape(shape),
            "targets": labels.reshape(shape),
            "loss_mask": mask.reshape(shape),
        }


def evaluate(options):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from flax.training.train_state import TrainState
    from jax.experimental import multihost_utils
    import optax

    from lm.config import ModelConfig, TrainConfig
    from lm.runtime import initialize
    from lm.train import create_model, optimizer, shard_batch

    hardware = initialize(True)
    streams = {
        "screen_heldout": Stream(Path(options.screen_data) / "eval.bin"),
        "fresh_holdout": Stream(Path(options.fresh_data) / "eval.bin"),
    }
    results = {"hardware": hardware, "models": {}}
    for item in options.models:
        name, config_path, run = item.split("=")[0], *item.split("=")[1].split(",")
        frozen = json.loads(Path(config_path).read_text())
        model_config = ModelConfig(**frozen["model"])
        training = TrainConfig(**frozen["training"])
        model = create_model(model_config)
        params = model.init(
            jax.random.PRNGKey(training.seed),
            jnp.zeros((1, model_config.chunk_size), jnp.int32),
            train=False,
        )["params"]
        tx, _ = optimizer(training, params)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        pointer = json.loads((Path(run) / "latest.json").read_text())["checkpoint"]
        payload = (Path(run) / pointer / "state.msgpack").read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        state = serialization.from_bytes(state, payload)
        replicated = jax.device_put_replicated(state.params, jax.local_devices())

        def sequence_losses(parameters, batch):
            logits = model.apply({"params": parameters}, batch["input_ids"])
            losses = optax.softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), batch["targets"]
            )
            mask = batch["loss_mask"]
            return jnp.sum(losses * mask, axis=-1), jnp.sum(mask, axis=-1)

        mapped = jax.pmap(sequence_losses)
        entry = {"checkpoint": str(Path(run) / pointer), "checkpoint_sha256": digest}
        for label, stream in streams.items():
            sums, counts = [], []
            steps = -(-stream.targets // training.tokens_per_step)
            for step in range(steps):
                batch = stream.batch(
                    step, training.global_batch, training.sequence_length
                )
                local = shard_batch(batch, training.global_batch)
                total, count = mapped(replicated, local)
                gathered = multihost_utils.process_allgather(
                    np.stack([np.asarray(total), np.asarray(count)])
                )
                # [process, 2, local device, per-device batch] -> global row order.
                gathered = np.asarray(gathered).transpose(1, 0, 2, 3).reshape(2, -1)
                sums.append(gathered[0])
                counts.append(gathered[1])
            sums, counts = np.concatenate(sums), np.concatenate(counts)
            if int(counts.sum()) != stream.targets:
                raise RuntimeError(f"{name} {label} target accounting mismatch")
            entry[label] = {
                "nll": float(sums.sum() / counts.sum()),
                "targets": int(counts.sum()),
                "sequence_nll_sum": sums.tolist(),
                "sequence_targets": counts.astype(int).tolist(),
            }
            print(name, label, entry[label]["nll"], flush=True)
        results["models"][name] = entry
    if jax.process_index() == 0:
        atomic_json(Path(options.output) / "sequence-nll.json", results)
    jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--screen-data", default="data/fineweb-edu-1b")
    build.add_argument("--output", default="data/fresh-holdout-v2")
    build.add_argument("--seed", type=int, default=SEED)
    build.add_argument("--shard", help="local pinned shard instead of a download")
    build.add_argument("--shard-source", help="JSON with the shard's lfs_sha256")
    build.add_argument("--exclude", nargs="*", default=[], help="earlier ledgers")
    run = sub.add_parser("evaluate")
    run.add_argument("--screen-data", default="data/fineweb-edu-1b")
    run.add_argument("--fresh-data", default="data/fresh-holdout-v2")
    run.add_argument("--output", required=True)
    run.add_argument("models", nargs="+", help="name=config.json,run_directory")
    options = parser.parse_args()
    prepare(options) if options.action == "prepare" else evaluate(options)


if __name__ == "__main__":
    main()
