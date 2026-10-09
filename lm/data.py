"""Deterministic, bounded next-token batches from an audited token corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


def sha256_file(path: str | Path) -> str:
    """Hash large files without loading them into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TokenCorpus:
    """Open train/eval uint16 streams and preserve exact target accounting.

    Each sequence has L inputs and the L immediately following targets.
    Consecutive sequences share one context token; target intervals never
    overlap. Corpus files therefore contain target_count + 1 tokens. Padding
    in the last batch has a zero loss mask, so it cannot increase the budget.
    All architectures use the same step-to-token mapping.
    """

    def __init__(self, path: str | Path, *, verify_hashes: bool = False):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != "mamba4.tokens.v1":
            raise ValueError("Unsupported or incomplete corpus manifest")
        self.dtype = np.dtype(self.manifest["dtype"])
        self.eos_id = int(self.manifest["tokenizer"]["eos_id"])
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        if self.dtype != np.dtype("<u2") or self.vocab_size > 65536:
            raise ValueError("Corpus must use uint16 with a compatible vocabulary")
        self.streams = {}
        for split in ("train", "eval"):
            item = self.manifest["splits"][split]
            file = self.path / item["file"]
            expected_size = int(item["token_count"]) * self.dtype.itemsize
            if file.stat().st_size != expected_size:
                raise ValueError(f"{split} corpus size does not match its manifest")
            if int(item["target_count"]) + 1 != int(item["token_count"]):
                raise ValueError(f"{split} corpus has inconsistent target accounting")
            if verify_hashes and sha256_file(file) != item["sha256"]:
                raise ValueError(f"{split} corpus hash does not match its manifest")
            self.streams[split] = np.memmap(file, dtype=self.dtype, mode="r")

    def target_count(self, split: str = "train") -> int:
        return int(self.manifest["splits"][split]["target_count"])

    def num_steps(
        self,
        global_batch: int,
        sequence_length: int,
        *,
        target_budget: int | None = None,
        split: str = "train",
    ) -> int:
        if global_batch <= 0 or sequence_length <= 0:
            raise ValueError("Batch and sequence length must be positive")
        budget = self._budget(split, target_budget)
        return (budget + global_batch * sequence_length - 1) // (
            global_batch * sequence_length
        )

    def _budget(self, split: str, target_budget: int | None) -> int:
        available = self.target_count(split)
        budget = available if target_budget is None else int(target_budget)
        if budget <= 0 or budget > available:
            raise ValueError(
                f"Requested {budget} targets; {split} contains {available}"
            )
        return budget

    def batch(
        self,
        step: int,
        global_batch: int,
        sequence_length: int,
        *,
        target_budget: int | None = None,
        split: str = "train",
    ) -> dict[str, np.ndarray]:
        """Return input_ids, targets, and loss_mask with shape [B, L]."""
        if step < 0 or global_batch <= 0 or sequence_length <= 0:
            raise ValueError("Step must be nonnegative; batch and length positive")
        budget = self._budget(split, target_budget)
        capacity = global_batch * sequence_length
        offset = step * capacity
        if offset >= budget:
            raise IndexError("The requested step is beyond the target budget")
        valid = min(capacity, budget - offset)
        inputs = np.full(capacity, self.eos_id, dtype=np.int32)
        targets = np.full(capacity, self.eos_id, dtype=np.int32)
        mask = np.zeros(capacity, dtype=np.float32)
        stream = self.streams[split]
        inputs[:valid] = stream[offset : offset + valid]
        targets[:valid] = stream[offset + 1 : offset + valid + 1]
        mask[:valid] = 1.0
        shape = (global_batch, sequence_length)
        return {
            "input_ids": inputs.reshape(shape),
            "targets": targets.reshape(shape),
            "loss_mask": mask.reshape(shape),
        }

    def iter_batches(
        self,
        global_batch: int,
        sequence_length: int,
        *,
        start_step: int = 0,
        target_budget: int | None = None,
        split: str = "train",
    ) -> Iterator[dict[str, np.ndarray]]:
        for step in range(
            start_step,
            self.num_steps(
                global_batch,
                sequence_length,
                target_budget=target_budget,
                split=split,
            ),
        ):
            yield self.batch(
                step,
                global_batch,
                sequence_length,
                target_budget=target_budget,
                split=split,
            )
