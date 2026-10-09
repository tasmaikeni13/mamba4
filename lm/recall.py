"""Frozen textual key/value diagnostic with prompt-level uncertainty.

This evaluates trained weights without additional optimization. It is a small
recall diagnostic; its intervals describe sampled prompts, not training seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from lm.data import sha256_file


RECALL_SEED = 20261008
COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")
FILLER = (
    "A quiet village has a library, a school, and a market. "
    "People read books, solve arithmetic exercises, and learn about history. "
    "The calendar lists months and seasons. Rivers flow through valleys. "
)


@dataclass(frozen=True)
class RecallDataset:
    """One fixed global batch, with a separate scored query per prompt."""

    arrays: dict[str, np.ndarray]
    prompts: tuple[dict, ...]
    manifest: dict

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "prompts.npz", **self.arrays)
        (path / "prompts.json").write_text(json.dumps(self.prompts, indent=2) + "\n")
        (path / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")


def build_recall_dataset(
    tokenizer_path: str | Path = "data/fineweb-edu-1b/tokenizer.json",
    *,
    prompt_count: int = 128,
    sequence_length: int = 1024,
    seed: int = RECALL_SEED,
) -> RecallDataset:
    """Create paired prompts with short/medium/long and near/far placements.

    The queried fact occurs exactly once; seven other key/value facts and
    neutral prose distractors surround it. The answer is absent after the
    query. Padding follows the scored query and cannot affect causal models.
    Gold colors are balanced to within one prompt independently of candidate
    ordering, preventing a constant candidate preference from beating chance.
    """
    from tokenizers import Tokenizer

    if prompt_count <= 0 or sequence_length < 128:
        raise ValueError("Recall requires positive prompts and length at least 128")
    tokenizer_path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 50257:
        raise ValueError("Recall requires the pinned 50257-token GPT2 vocabulary")
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos != 50256:
        raise ValueError("Unexpected GPT2 EOS")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    candidate_ids = []
    for color in COLORS:
        ids = encode(" " + color)
        if len(ids) != 1:
            raise ValueError(f"Recall candidate {color!r} is not one GPT2 token")
        candidate_ids.append(ids[0])
    filler_ids = encode(FILLER * (sequence_length // 16 + 2))
    rng = np.random.default_rng(seed)
    balanced_gold = rng.permutation(np.arange(prompt_count) % len(COLORS))
    lengths = sorted({min(sequence_length, length) for length in (128, 512, 1024)})
    groups = [
        (length, placement) for length in lengths for placement in ("near", "far")
    ]
    input_ids = np.full((prompt_count, sequence_length), eos, dtype=np.int32)
    positions = np.empty(prompt_count, dtype=np.int32)
    candidates = np.empty((prompt_count, len(COLORS)), dtype=np.int32)
    gold = np.empty(prompt_count, dtype=np.int32)
    records = []
    for prompt_index in range(prompt_count):
        requested_length, placement = groups[prompt_index % len(groups)]
        keys = [f"key-{code:04x}" for code in rng.choice(65536, size=8, replace=False)]
        target_key_index = int(rng.integers(8))
        values = rng.permutation(8)
        gold_color_index = int(balanced_gold[prompt_index])
        old_gold = int(np.flatnonzero(values == gold_color_index)[0])
        values[old_gold], values[target_key_index] = (
            values[target_key_index],
            values[old_gold],
        )
        target_key = keys[target_key_index]
        target_fact = f"{target_key} -> {COLORS[gold_color_index]}\n"
        others = [i for i in range(8) if i != target_key_index]
        rng.shuffle(others)
        distractor_facts = "".join(
            f"{keys[i]} -> {COLORS[values[i]]}\n" for i in others
        )
        header = "Remember this name-to-color table.\n"
        query = f"\nFind the color for {target_key}.\nAnswer:"
        baseline = header + target_fact + distractor_facts + query
        spare = requested_length - len(encode(baseline))
        if spare < 0:
            # Random keys can occupy many byte-BPE tokens; keep short prompts
            # valid without dropping any facts or shrinking their candidate set.
            requested_length = min(sequence_length, len(encode(baseline)) + 8)
            spare = requested_length - len(encode(baseline))
        for _ in range(8):
            prose = tokenizer.decode(filler_ids[: max(0, spare)]) + "\n"
            if placement == "near":
                before_target = header + distractor_facts + prose
                prefix = before_target + target_fact
            else:
                before_target = header
                prefix = before_target + target_fact + distractor_facts + prose
            text = prefix + query
            ids = encode(text)
            delta = requested_length - len(ids)
            if delta == 0 or spare == 0:
                break
            spare = max(0, spare + delta)
        if len(ids) > sequence_length:
            raise ValueError("Recall construction exceeded the frozen sequence length")
        end_of_fact = len(encode(before_target + target_fact)) - 1
        input_ids[prompt_index, : len(ids)] = ids
        positions[prompt_index] = len(ids) - 1
        permutation = rng.permutation(8)
        candidates[prompt_index] = np.asarray(candidate_ids)[permutation]
        gold[prompt_index] = int(np.flatnonzero(permutation == gold_color_index)[0])
        records.append(
            {
                "prompt_index": prompt_index,
                "group": f"length-{groups[prompt_index % len(groups)][0]}-{placement}",
                "placement": placement,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "effective_length": len(ids),
                "target_position": int(positions[prompt_index]),
                "fact_to_query_tokens": int(positions[prompt_index] - end_of_fact),
                "target_key": target_key,
                "gold_color": COLORS[gold_color_index],
                "candidate_colors": [COLORS[i] for i in permutation],
                "gold_index": int(gold[prompt_index]),
            }
        )
    arrays = {
        "input_ids": input_ids,
        "target_positions": positions,
        "candidate_token_ids": candidates,
        "gold_index": gold,
    }
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.astype("<i4").tobytes())
    manifest = {
        "protocol": "trained-textual-recall-v1",
        "seed": seed,
        "prompt_count": prompt_count,
        "sequence_length": sequence_length,
        "candidate_count": len(COLORS),
        "chance_accuracy": 1 / len(COLORS),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "array_sha256": digest.hexdigest(),
        "effective_length_range": [int(positions.min() + 1), int(positions.max() + 1)],
        "sampling_unit": "one independently generated key/value prompt",
        "selection": "frozen diagnostic; no architecture or optimizer selection",
        "uncertainty": "prompt variability only; one training seed",
    }
    return RecallDataset(arrays, tuple(records), manifest)


def _bootstrap_mean(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize_recall(dataset: RecallDataset, candidate_log_probs: np.ndarray) -> dict:
    """Report full-vocabulary and candidate-conditional NLL for whole prompts."""
    scores = np.asarray(candidate_log_probs, dtype=np.float64)
    gold = dataset.arrays["gold_index"]
    expected_shape = dataset.arrays["candidate_token_ids"].shape
    if scores.shape != expected_shape or not np.isfinite(scores).all():
        raise ValueError(
            "Recall scores must be finite, with shape [prompts, candidates]"
        )
    predicted = scores.argmax(axis=1)
    correct = (predicted == gold).astype(np.float64)
    log_normalizer = np.logaddexp.reduce(scores, axis=1)
    full_nll = -scores[np.arange(len(gold)), gold]
    conditional_nll = full_nll + log_normalizer

    def aggregate(indices, seed):
        n = len(indices)
        accuracy = float(correct[indices].mean())
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (accuracy + z * z / (2 * n)) / denominator
        radius = (
            z
            * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
            / denominator
        )
        return {
            "prompts": n,
            "accuracy": accuracy,
            "accuracy_wilson_95": [float(center - radius), float(center + radius)],
            "gold_full_vocab_nll": float(full_nll[indices].mean()),
            "gold_full_vocab_nll_bootstrap_95": _bootstrap_mean(
                full_nll[indices], seed
            ),
            "gold_candidate_nll": float(conditional_nll[indices].mean()),
            "gold_candidate_nll_bootstrap_95": _bootstrap_mean(
                conditional_nll[indices], seed + 1
            ),
        }

    groups = {}
    for group in sorted({prompt["group"] for prompt in dataset.prompts}):
        indices = np.asarray(
            [i for i, prompt in enumerate(dataset.prompts) if prompt["group"] == group]
        )
        groups[group] = aggregate(indices, RECALL_SEED + len(groups) + 2)
    return {
        "protocol": dataset.manifest,
        "overall": aggregate(np.arange(len(gold)), RECALL_SEED + 1),
        "groups": groups,
        "records": [
            {
                "prompt_index": i,
                "text_sha256": dataset.prompts[i]["text_sha256"],
                "group": dataset.prompts[i]["group"],
                "gold_index": int(gold[i]),
                "predicted_index": int(predicted[i]),
                "correct": bool(correct[i]),
                "gold_full_vocab_nll": float(full_nll[i]),
                "gold_candidate_nll": float(conditional_nll[i]),
                "candidate_log_probs": scores[i].tolist(),
            }
            for i in range(len(gold))
        ],
    }


def make_recall_step(model):
    """Return a pmap that emits only eight log probabilities per prompt."""
    import jax
    import jax.numpy as jnp

    def step(params, batch):
        logits = model.apply({"params": params}, batch["input_ids"], train=False)
        queried = logits[jnp.arange(logits.shape[0]), batch["target_positions"]]
        log_probs = jax.nn.log_softmax(queried.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(log_probs, batch["candidate_token_ids"], axis=-1)

    return jax.pmap(step, axis_name="data")


def evaluate_recall(dataset: RecallDataset, replicated_params, recall_step) -> dict:
    """Every host participates; aggregate ordered small outputs across hosts."""
    import jax
    from jax.experimental import multihost_utils

    total = len(dataset.prompts)
    if total % jax.device_count():
        raise ValueError("Recall prompt count must divide the global device count")
    local_count = total // jax.process_count()
    start = jax.process_index() * local_count
    batch = {
        name: value[start : start + local_count].reshape(
            jax.local_device_count(),
            local_count // jax.local_device_count(),
            *value.shape[1:],
        )
        for name, value in dataset.arrays.items()
        if name != "gold_index"
    }
    local_scores = np.asarray(jax.device_get(recall_step(replicated_params, batch)))
    local_scores = local_scores.reshape(
        local_count, dataset.manifest["candidate_count"]
    )
    scores = (
        np.asarray(multihost_utils.process_allgather(local_scores)).reshape(total, -1)
        if jax.process_count() > 1
        else local_scores
    )
    return summarize_recall(dataset, scores)
