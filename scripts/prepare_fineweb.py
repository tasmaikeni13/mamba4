"""Prepare pinned FineWeb-Edu documents for matched language-model runs.

Downloads and tokenized source shards are reusable. A final manifest is
published atomically only after split exclusion, target counts and hashes
have been checked. The complete ordered document ledger remains available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm.data import sha256_file  # noqa: E402


DATASET_REPO = "codelion/fineweb-edu-1B"
DATASET_REVISION = "66b78cd134ab7b0545a1a618a1cc029f3091eb20"
TOKENIZER_REPO = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
INDEX_DTYPE = np.dtype([("offset", "<u8"), ("length", "<u4"), ("sha256", "u1", (32,))])
LEDGER_DTYPE = np.dtype(
    [
        ("shard", "<u2"),
        ("row", "<u4"),
        ("offset", "<u8"),
        ("length", "<u4"),
        ("sha256", "u1", (32,)),
        ("split", "u1"),
    ]
)


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def download_file(
    item: dict, directory: Path, revision: str, repo: str = DATASET_REPO
) -> dict:
    path = directory / Path(item["path"]).name
    expected = item.get("lfs", {}).get("oid")
    if (
        path.exists()
        and path.stat().st_size == item["size"]
        and (expected is None or sha256_file(path) == expected)
    ):
        log(f"Verified cached source {path.name}")
    else:
        url = (
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{item['path']}"
        )
        temp = path.with_suffix(".download")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    with temp.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
                actual = sha256_file(temp)
                if temp.stat().st_size != item["size"]:
                    raise ValueError(f"Wrong download size for {path.name}")
                if expected and actual != expected:
                    raise ValueError(f"Wrong SHA256 for {path.name}")
                temp.replace(path)
                log(f"Downloaded and verified {path.name}")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    return {
        "path": item["path"],
        "local_file": str(path.name),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "hub_git_oid": item["oid"],
    }


def load_tokenizer(directory: Path):
    from tokenizers import Tokenizer

    file = directory / "tokenizer.json"
    if not file.exists():
        url = (
            f"https://huggingface.co/{TOKENIZER_REPO}/resolve/"
            f"{TOKENIZER_REVISION}/tokenizer.json"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            temp = file.with_suffix(".download")
            with temp.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temp.replace(file)
    tokenizer = Tokenizer.from_file(str(file))
    if tokenizer.get_vocab_size() != 50257:
        raise ValueError("The pinned GPT2 vocabulary must contain 50257 tokens")
    if tokenizer.token_to_id("<|endoftext|>") != 50256:
        raise ValueError("Unexpected GPT2 EOS token ID")
    return tokenizer, {
        "repo": TOKENIZER_REPO,
        "revision": TOKENIZER_REVISION,
        "file": "tokenizer.json",
        "sha256": sha256_file(file),
        "vocab_size": 50257,
        "eos_id": 50256,
        "add_special_tokens": False,
        "append_eos_per_document": True,
    }


def tokenize_shard(task: tuple[str, str, str]) -> dict:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    source_name, tokenizer_name, output_name = task
    source, output = Path(source_name), Path(output_name)
    token_file = output.with_suffix(".bin")
    index_file = output.with_suffix(".npy")
    meta_file = output.with_suffix(".json")
    tokenizer_hash = sha256_file(tokenizer_name)
    if token_file.exists() and index_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if (
            meta["tokenizer_sha256"] == tokenizer_hash
            and meta["source_sha256"] == sha256_file(source)
            and meta["tokens_sha256"] == sha256_file(token_file)
            and meta["index_sha256"] == sha256_file(index_file)
        ):
            log(f"Verified reusable tokenized shard {source.name}")
            return meta
    tokenizer = Tokenizer.from_file(tokenizer_name)
    parquet = pq.ParquetFile(source)
    records = np.empty(parquet.metadata.num_rows, dtype=INDEX_DTYPE)
    row, offset = 0, 0
    temp = token_file.with_suffix(".bin.tmp")
    started = time.monotonic()
    with temp.open("wb", buffering=8 * 1024 * 1024) as stream:
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if any(not isinstance(text, str) for text in texts):
                raise ValueError("Null or non-string document in source dataset")
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for text, encoding in zip(texts, encodings, strict=True):
                ids = np.asarray(encoding.ids + [50256], dtype="<u2")
                records[row]["offset"] = offset
                records[row]["length"] = len(ids)
                records[row]["sha256"] = np.frombuffer(
                    hashlib.sha256(text.encode("utf-8")).digest(), dtype=np.uint8
                )
                stream.write(ids.tobytes())
                row += 1
                offset += len(ids)
            if row % (256 * 40) == 0:
                log(
                    f"{source.name}: {row:,}/{len(records):,} documents, "
                    f"{offset:,} tokens, {time.monotonic() - started:.0f}s"
                )
    if row != len(records):
        raise ValueError("Parquet row count changed during tokenization")
    temp.replace(token_file)
    with index_file.with_suffix(".npy.tmp").open("wb") as stream:
        np.save(stream, records, allow_pickle=False)
    index_file.with_suffix(".npy.tmp").replace(index_file)
    meta = {
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "tokenizer_sha256": tokenizer_hash,
        "documents": row,
        "tokens_including_eos": offset,
        "tokens_sha256": sha256_file(token_file),
        "index_sha256": sha256_file(index_file),
        "tokens_file": token_file.name,
        "index_file": index_file.name,
        "seconds": time.monotonic() - started,
    }
    atomic_json(meta_file, meta)
    log(f"Finished {source.name}: {offset:,} tokens in {meta['seconds']:.0f}s")
    return meta


def earlier_hashes(paths) -> set:
    """Document SHA256 digests recorded by earlier evaluation sets."""
    digests = set()
    for path in paths:
        record = json.loads(Path(path).read_text())
        if "long_documents" in record:
            digests |= {
                bytes.fromhex(h) for h in record["long_documents"]["document_sha256"]
            }
        else:
            digests |= {bytes.fromhex(row[4]) for row in record["rows"]}
    return digests


def split_documents(
    ledger: np.ndarray, seed: int, eval_targets: int, excluded_hashes=frozenset()
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hold out whole content-hash groups before ordering training documents.

    Documents whose hash is in ``excluded_hashes`` (earlier evaluation sets)
    enter neither stream.
    """
    hashes = ledger["sha256"].copy().view("S32").reshape(-1)
    unique, representative, inverse = np.unique(
        hashes, return_index=True, return_inverse=True
    )
    banned = np.asarray([digest in excluded_hashes for digest in unique.tolist()])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(unique))
    holdout_groups = []
    heldout_tokens = 0
    for group in permutation:
        if banned[group]:
            continue
        holdout_groups.append(group)
        heldout_tokens += int(ledger[representative[group]]["length"])
        if heldout_tokens >= eval_targets + 1:
            break
    if heldout_tokens < eval_targets + 1:
        raise ValueError("Source cannot supply the requested held-out token budget")
    heldout_group_mask = np.zeros(len(unique), dtype=bool)
    heldout_group_mask[holdout_groups] = True
    excluded = heldout_group_mask[inverse]
    removed = banned[inverse]
    # Evaluation uses one representative per hash, avoiding duplicated eval text.
    eval_order = representative[np.asarray(holdout_groups)]
    train_candidates = np.flatnonzero(~excluded & ~removed)
    train_order = rng.permutation(train_candidates)
    if train_order.size == 0:
        raise ValueError("No training documents remain after holdout exclusion")
    ledger["split"] = excluded.astype(np.uint8) + 2 * removed.astype(np.uint8)
    summary = {
        "algorithm": "numpy-PCG64 seed; sorted content SHA256 groups; permutation",
        "seed": seed,
        "document_hash": "SHA256 of exact UTF-8 text before tokenization",
        "holdout_policy": "whole content-hash groups excluded from training",
        "near_duplicate_exclusion": False,
        "source_documents": len(ledger),
        "unique_content_hashes": len(unique),
        "train_candidate_documents": len(train_order),
        "heldout_unique_documents": len(eval_order),
        "heldout_source_documents_excluded": int(excluded.sum()),
        "heldout_tokens_available": heldout_tokens,
        "earlier_evaluation_documents_removed": int(removed.sum()),
        "split_values_in_ledger": {"0": "train", "1": "heldout", "2": "removed"},
    }
    return train_order, eval_order, summary


def write_stream(
    directory: Path,
    name: str,
    order: np.ndarray,
    ledger: np.ndarray,
    source_streams: list[np.ndarray],
    targets: int,
    *,
    allow_replay: bool = False,
    replay_seed: int = 43,
) -> dict:
    """Write exactly targets + 1 tokens, documenting any source reuse."""
    desired = targets + 1
    available = int(ledger["length"][order].sum())
    if available < desired:
        if not allow_replay:
            raise ValueError(
                f"{name} has only {available - 1:,} distinct stream targets; "
                f"requested {targets:,}. Refusing silent training-document replay."
            )
        if name != "train":
            raise ValueError("Evaluation documents must never be replayed")
        rng = np.random.default_rng(replay_seed)
        epochs = [order]
        accumulated = available
        while accumulated < desired:
            epoch = rng.permutation(order)
            cumulative = np.cumsum(ledger["length"][epoch], dtype=np.uint64)
            used = min(
                len(epoch), int(np.searchsorted(cumulative, desired - accumulated)) + 1
            )
            epochs.append(epoch[:used])
            accumulated += int(cumulative[used - 1])
        order = np.concatenate(epochs)
        log(
            f"Explicit {name} replay: {desired - available:,} stream tokens "
            f"({100 * (desired - available) / targets:.4f}% of target budget), "
            f"fresh training-only permutation seed {replay_seed}"
        )
    path = directory / f"{name}.bin"
    temp = path.with_suffix(".bin.tmp")
    digest = hashlib.sha256()
    written = 0
    used_documents = 0
    partial_document = None
    with temp.open("wb", buffering=8 * 1024 * 1024) as stream:
        for index in order:
            record = ledger[index]
            begin = int(record["offset"])
            count = min(int(record["length"]), desired - written)
            source = source_streams[int(record["shard"])]
            chunk = source[begin : begin + count].tobytes()
            stream.write(chunk)
            digest.update(chunk)
            written += count
            used_documents += 1
            if count < int(record["length"]):
                partial_document = {
                    "ledger_row": int(index),
                    "tokens_used": count,
                    "document_tokens": int(record["length"]),
                }
            if written == desired:
                break
    if written != desired:
        raise ValueError("Stream target budget was not fulfilled")
    temp.replace(path)
    np.save(directory / f"{name}-document-order.npy", order, allow_pickle=False)
    return {
        "file": path.name,
        "sha256": digest.hexdigest(),
        "token_count": written,
        "target_count": targets,
        "available_source_tokens": available,
        "documents_used": used_documents,
        "source_replay": available < desired,
        "replayed_stream_tokens": max(0, desired - available),
        "replay_policy": (
            "new PCG64 permutation of training-only documents"
            if available < desired
            else None
        ),
        "replay_seed": replay_seed if available < desired else None,
        "partial_final_document": partial_document,
        "document_order_file": f"{name}-document-order.npy",
        "document_order_sha256": sha256_file(directory / f"{name}-document-order.npy"),
    }


def prepare(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists() and not args.rebuild:
        from lm.data import TokenCorpus

        corpus = TokenCorpus(directory, verify_hashes=True)
        if (
            corpus.target_count("train") != args.train_targets
            or corpus.target_count("eval") != args.eval_targets
            or corpus.manifest["split"]["seed"] != args.seed
            or (
                corpus.manifest["splits"]["train"]["source_replay"]
                and not args.allow_train_replay
            )
        ):
            raise ValueError("Existing corpus has another protocol; use --rebuild")
        log("Complete corpus verified; nothing to rebuild")
        return corpus.manifest
    sources = directory / "sources"
    tokenized = directory / "tokenized"
    sources.mkdir(exist_ok=True)
    tokenized.mkdir(exist_ok=True)
    metadata_url = (
        f"https://huggingface.co/api/datasets/{args.repo}/tree/"
        f"{args.revision}/{args.prefix}?recursive=true&expand=true"
    )
    with urllib.request.urlopen(metadata_url, timeout=120) as response:
        tree = json.load(response)
    files = sorted(
        (item for item in tree if item["path"].endswith(".parquet")),
        key=lambda item: item["path"],
    )
    if args.files:
        wanted = set(args.files)
        files = [item for item in files if Path(item["path"]).name in wanted]
        if len(files) != len(wanted):
            raise ValueError("Some requested parquet shards are not in the pin")
    elif len(files) != args.expected_files:
        raise ValueError("Pinned source must contain every expected parquet shard")
    existing_bytes = sum(
        (sources / Path(item["path"]).name).stat().st_size
        for item in files
        if (sources / Path(item["path"]).name).exists()
    )
    required = max(0, sum(item["size"] for item in files) - existing_bytes)
    required += 2 * (args.train_targets + args.eval_targets) + 4_000_000_000
    if shutil.disk_usage(directory).free < required:
        raise OSError(f"Preparation needs at least {required:,} free bytes")
    tokenizer, tokenizer_meta = load_tokenizer(directory)
    del tokenizer
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        source_meta = list(
            executor.map(
                lambda item: download_file(item, sources, args.revision, args.repo),
                files,
            )
        )
    tasks = [
        (
            str(sources / Path(item["path"]).name),
            str(directory / "tokenizer.json"),
            str(tokenized / Path(item["path"]).stem),
        )
        for item in files
    ]
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.tokenizer_threads))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        shard_meta = list(executor.map(tokenize_shard, tasks))
    ledger = np.empty(sum(item["documents"] for item in shard_meta), LEDGER_DTYPE)
    source_streams = []
    start = 0
    for shard, item in enumerate(shard_meta):
        index = np.load(tokenized / item["index_file"], allow_pickle=False)
        end = start + len(index)
        ledger["shard"][start:end] = shard
        ledger["row"][start:end] = np.arange(len(index))
        for field in ("offset", "length", "sha256"):
            ledger[field][start:end] = index[field]
        source_streams.append(
            np.memmap(tokenized / item["tokens_file"], dtype="<u2", mode="r")
        )
        start = end
    train_order, eval_order, split_meta = split_documents(
        ledger, args.seed, args.eval_targets, earlier_hashes(args.exclude)
    )
    np.save(directory / "documents.npy", ledger, allow_pickle=False)
    log(
        f"Split {len(ledger):,} documents: "
        f"{split_meta['heldout_unique_documents']:,} unique held-out documents"
    )
    train_meta = write_stream(
        directory,
        "train",
        train_order,
        ledger,
        source_streams,
        args.train_targets,
        allow_replay=args.allow_train_replay,
        replay_seed=args.seed + 1,
    )
    eval_meta = write_stream(
        directory, "eval", eval_order, ledger, source_streams, args.eval_targets
    )
    manifest = {
        "format": "mamba4.tokens.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": "<u2",
        "dataset": {
            "repo": args.repo,
            "revision": args.revision,
            "prefix": args.prefix,
            "files": source_meta,
            "source_tokens_including_eos": sum(
                item["tokens_including_eos"] for item in shard_meta
            ),
        },
        "tokenizer": tokenizer_meta,
        "split": split_meta,
        "document_ledger": {
            "file": "documents.npy",
            "sha256": sha256_file(directory / "documents.npy"),
        },
        "splits": {"train": train_meta, "eval": eval_meta},
        "accounting": {
            "target": "one non-masked next-token cross-entropy term",
            "eos_targets_count": True,
            "packing": "concatenate documents with one EOS after each",
            "sequence_context": "independent fixed-length contexts; EOS can be crossed",
            "last_batch": "EOS padding with zero loss mask beyond exact budget",
            "matching": "all architectures consume identical step/batch/length mapping",
        },
        "provenance": {
            "prepare_script_sha256": sha256_file(__file__),
            "data_module_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "lm/data.py"
            ),
            "numpy_version": np.__version__,
            "tokenizer_library_version": __import__("tokenizers").__version__,
            "pyarrow_version": __import__("pyarrow").__version__,
            "preparation_seconds": time.monotonic() - started,
        },
    }
    atomic_json(directory / "manifest.json", manifest)
    from lm.data import TokenCorpus

    TokenCorpus(directory, verify_hashes=True)
    log(
        f"Verified corpus complete: {args.train_targets:,} training targets, "
        f"{args.eval_targets:,} held-out targets"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/fineweb-edu-1b")
    parser.add_argument("--repo", default=DATASET_REPO)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--prefix", default="data", help="parquet folder in the repo")
    parser.add_argument("--files", nargs="*", default=[], help="parquet shard names")
    parser.add_argument("--expected-files", type=int, default=10)
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="earlier evaluation document lists"
    )
    parser.add_argument("--train-targets", type=int, default=1_000_000_000)
    parser.add_argument("--eval-targets", type=int, default=2_097_152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--allow-train-replay",
        action="store_true",
        help="Explicitly permit training-only document replay if source is insufficient",
    )
    args = parser.parse_args()
    if args.train_targets <= 0 or args.eval_targets <= 0:
        parser.error("Target budgets must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
