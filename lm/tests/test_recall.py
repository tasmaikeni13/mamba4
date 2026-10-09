"""Token-level synthetic recall construction and whole-prompt scoring checks."""

from pathlib import Path

import numpy as np
import pytest

from lm.recall import RecallDataset, build_recall_dataset, summarize_recall


@pytest.fixture(scope="module")
def actual_prompts():
    tokenizer = Path("data/fineweb-edu-1b/tokenizer.json")
    if not tokenizer.exists():
        pytest.skip("Pinned GPT2 tokenizer is provided by FineWeb preparation")
    return build_recall_dataset(tokenizer)


def test_actual_gpt2_prompts_are_frozen_causal_and_balanced(actual_prompts):
    from tokenizers import Tokenizer

    dataset = actual_prompts
    tokenizer = Tokenizer.from_file("data/fineweb-edu-1b/tokenizer.json")
    repeated = build_recall_dataset()
    assert dataset.manifest == repeated.manifest
    for name in dataset.arrays:
        np.testing.assert_array_equal(dataset.arrays[name], repeated.arrays[name])
    assert dataset.arrays["input_ids"].shape == (128, 1024)
    assert dataset.arrays["candidate_token_ids"].shape == (128, 8)
    gold_counts = {}
    for i, prompt in enumerate(dataset.prompts):
        assert prompt["text"].count(prompt["target_key"]) == 2  # fact and question
        assert prompt["text"].endswith("Answer:")
        assert prompt["text"].count(" -> ") == 8
        position = dataset.arrays["target_positions"][i]
        assert position == prompt["effective_length"] - 1
        assert np.all(dataset.arrays["input_ids"][i, position + 1 :] == 50256)
        encoded = tokenizer.encode(prompt["text"], add_special_tokens=False).ids
        np.testing.assert_array_equal(
            dataset.arrays["input_ids"][i, : position + 1], encoded
        )
        gold = dataset.arrays["gold_index"][i]
        token = dataset.arrays["candidate_token_ids"][i, gold]
        assert tokenizer.decode([int(token)]) == " " + prompt["gold_color"]
        gold_counts[prompt["gold_color"]] = gold_counts.get(prompt["gold_color"], 0) + 1
    assert set(gold_counts.values()) == {16}
    for length in (128, 512, 1024):
        near = [
            p["fact_to_query_tokens"]
            for p in dataset.prompts
            if p["group"] == f"length-{length}-near"
        ]
        far = [
            p["fact_to_query_tokens"]
            for p in dataset.prompts
            if p["group"] == f"length-{length}-far"
        ]
        assert np.mean(far) > np.mean(near) + length / 3


def scoring_fixture():
    gold = np.asarray([0, 1, 2, 0], dtype=np.int32)
    arrays = {"gold_index": gold, "candidate_token_ids": np.tile(np.arange(3), (4, 1))}
    prompts = tuple(
        {"group": "near" if i < 2 else "far", "text_sha256": str(i)} for i in range(4)
    )
    return RecallDataset(arrays, prompts, {"prompt_count": 4})


def test_nll_uses_full_vocab_scores_and_candidate_conditioning():
    dataset = scoring_fixture()
    probabilities = np.asarray(
        [[0.6, 0.1, 0.1], [0.05, 0.4, 0.05], [0.1, 0.1, 0.6], [0.1, 0.3, 0.1]]
    )
    result = summarize_recall(dataset, np.log(probabilities))
    gold_probabilities = np.asarray([0.6, 0.4, 0.6, 0.1])
    full_nll = -np.log(gold_probabilities)
    conditional_nll = -np.log(gold_probabilities / probabilities.sum(axis=1))
    assert result["overall"]["accuracy"] == 0.75
    assert result["overall"]["prompts"] == 4
    assert result["overall"]["gold_full_vocab_nll"] == pytest.approx(full_nll.mean())
    assert result["overall"]["gold_candidate_nll"] == pytest.approx(
        conditional_nll.mean()
    )
    assert result["groups"]["near"]["accuracy"] == 1
    assert result["groups"]["far"]["accuracy"] == 0.5
    assert len(result["records"]) == 4
    lower, upper = result["overall"]["accuracy_wilson_95"]
    assert 0 <= lower < 0.75 < upper <= 1
    assert result == summarize_recall(dataset, np.log(probabilities))


def test_uniform_candidate_nll_and_invalid_score_rejection():
    dataset = scoring_fixture()
    result = summarize_recall(dataset, np.full((4, 3), -np.log(50257)))
    assert result["overall"]["gold_full_vocab_nll"] == pytest.approx(np.log(50257))
    assert result["overall"]["gold_candidate_nll"] == pytest.approx(np.log(3))
    with pytest.raises(ValueError, match="finite"):
        summarize_recall(dataset, np.full((4, 3), np.nan))
    with pytest.raises(ValueError, match="shape"):
        summarize_recall(dataset, np.zeros((4, 2)))


def test_model_scores_use_query_position_and_global_prompt_order():
    import jax
    import jax.numpy as jnp

    from lm.recall import evaluate_recall, make_recall_step

    class PositionModel:
        def apply(self, variables, inputs, train=False):
            # Each token's own value identifies the candidate whose logit wins.
            return jax.nn.one_hot(inputs % 3, 3) * 2

    count = jax.device_count() * 2
    arrays = {
        "input_ids": np.tile(np.asarray([0, 1, 2, 0], np.int32), (count, 1)),
        "target_positions": np.arange(count, dtype=np.int32) % 3,
        "candidate_token_ids": np.tile(np.arange(3, dtype=np.int32), (count, 1)),
        "gold_index": np.arange(count, dtype=np.int32) % 3,
    }
    prompts = tuple({"group": "test", "text_sha256": str(i)} for i in range(count))
    dataset = RecallDataset(
        arrays, prompts, {"prompt_count": count, "candidate_count": 3}
    )
    result = evaluate_recall(
        dataset,
        jax.device_put_replicated({"dummy": jnp.asarray(0)}, jax.local_devices()),
        make_recall_step(PositionModel()),
    )
    assert result["overall"]["accuracy"] == 1
    expected_nll = np.log(np.exp(2) + 2) - 2
    assert result["overall"]["gold_full_vocab_nll"] == pytest.approx(
        expected_nll, abs=1e-6
    )
    assert [record["prompt_index"] for record in result["records"]] == list(
        range(count)
    )
