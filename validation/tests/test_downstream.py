"""Downstream benchmark prompts, row construction and scoring on CPU."""

import numpy as np
import pytest

from validation import downstream

TOKENIZER = "data/fineweb-edu-1b/tokenizer.json"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(TOKENIZER)


def test_requests_follow_the_harness_definitions():
    arc = {
        "question": "Which is a mammal?",
        "choices": {"text": ["shark", "whale"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    contexts, choices, gold, delimiter = downstream.requests("arc_easy", arc)
    assert contexts == ["Question: Which is a mammal?\nAnswer:"] * 2
    assert (choices, gold, delimiter) == (["shark", "whale"], 1, " ")
    hellaswag = {
        "activity_label": "Cooking",
        "ctx_a": "A man cracks eggs.",
        "ctx_b": "he",
        "endings": ["whisks them [title] well.", "eats the shells."],
        "label": "0",
    }
    contexts, choices, gold, _ = downstream.requests("hellaswag", hellaswag)
    assert contexts[0] == "Cooking: A man cracks eggs. He"
    assert choices == ["whisks them. well.", "eats the shells."] and gold == 0
    winogrande = {
        "sentence": "The cup fell off the table because _ was tilted.",
        "option1": "the cup",
        "option2": "the table",
        "answer": "2",
    }
    contexts, choices, gold, _ = downstream.requests("winogrande", winogrande)
    assert contexts == ["The cup fell off the table because the cup"] + [
        "The cup fell off the table because the table"
    ]
    assert choices == ["was tilted."] * 2 and gold == 1
    contexts, choices, gold, delimiter = downstream.requests(
        "lambada", {"text": "she opened the door"}
    )
    assert (contexts, choices, gold, delimiter) == (
        ["she opened the"],
        [" door"],
        0,
        "",
    )
    boolq = {"passage": "Water is wet.", "question": "is water wet", "answer": True}
    contexts, choices, gold, _ = downstream.requests("boolq", boolq)
    assert contexts[0] == "Water is wet.\nQuestion: is water wet?\nAnswer:"
    assert choices == ["no", "yes"] and gold == 1
    book = {
        "question_stem": "Plants need",
        "choices": {"text": ["sunlight", "rocks"], "label": ["A", "B"]},
        "answerKey": " A",
    }
    assert downstream.requests("openbookqa", book)[2] == 0


def test_rows_start_after_end_of_text_and_score_only_the_continuation(tokenizer):
    documents = [
        {"goal": "Open a jar", "sol1": "twist the lid", "sol2": "shout", "label": 0}
    ]
    rows, meta = downstream.build_rows(tokenizer, "piqa", documents)
    assert [m[:4] for m in meta] == [("piqa", 0, 0, 0), ("piqa", 0, 1, 0)]
    assert meta[0][4] == len("twist the lid")
    for row, choice in zip(rows, ("twist the lid", "shout")):
        assert row["ids"][0] == downstream.EOT
        full = np.concatenate((row["ids"], row["targets"][-1:]))
        assert np.array_equal(full[row["positions"] + 1], row["targets"])
        assert tokenizer.decode(row["targets"].tolist()) == " " + choice
        prompt = tokenizer.decode(row["ids"][1 : row["positions"][0] + 1].tolist())
        assert prompt == "Question: Open a jar\nAnswer:"


def test_long_rows_keep_the_most_recent_tokens(tokenizer):
    long = {"text": "word " * 3000 + "end"}
    rows, _ = downstream.build_rows(tokenizer, "lambada", [long])
    row = rows[0]
    assert len(row["ids"]) == downstream.MAX_LENGTH
    assert tokenizer.decode(row["targets"].tolist()) == " end"
    arrays, counts = downstream.pack(rows)
    assert counts == {"1024": 1}
    assert arrays["b1024_positions"][0, 0] == downstream.MAX_LENGTH - 1


def test_trailing_context_spaces_move_to_the_continuation(tokenizer):
    context, continuation = downstream.encode_pair(tokenizer, "Answer: ", "yes")
    assert tokenizer.decode(context) == "Answer:"
    assert tokenizer.decode(continuation) == " yes"


def test_accuracy_normalization_and_paired_summary():
    rows = [
        ["piqa", 0, 0, 1, 10],
        ["piqa", 0, 1, 1, 40],
        ["piqa", 1, 0, 0, 10],
        ["piqa", 1, 1, 0, 10],
        ["lambada", 0, 0, 0, 5],
        ["lambada", 1, 0, 0, 5],
    ]
    table = {"rows": rows}
    loglik = np.array([-5.0, -8.0, -1.0, -2.0, -0.5, -3.0])
    greedy = np.array([False, False, False, False, True, False])
    scores = downstream.score(table, loglik, greedy)
    # Raw sums prefer the short first choice; per character the long one wins.
    assert scores["piqa"]["acc"].tolist() == [False, True]
    assert scores["piqa"]["acc_norm"].tolist() == [True, True]
    assert scores["lambada"]["acc"].tolist() == [True, False]
    np.testing.assert_allclose(scores["lambada"]["loglik"], [-0.5, -3.0])

    def fake(hits):
        task = {"acc": np.array(hits), "acc_norm": np.array(hits)}
        lambada = {"acc": np.array(hits), "loglik": -np.ones(len(hits))}
        return {t: lambada if t == "lambada" else task for t in downstream.SOURCES}

    models = {
        "a0": fake([True, True, False, True]),
        "a1": fake([True, False, False, True]),
        "b0": fake([False, False, False, True]),
    }
    summary = downstream.summarize(models, {"a": ["a0", "a1"], "b": ["b0"]})
    piqa = summary["tasks"]["piqa"]
    assert piqa["groups"]["a"]["acc"]["mean"] == pytest.approx(0.625)
    assert piqa["groups"]["a"]["acc"]["sd"] == pytest.approx(
        np.std([0.75, 0.5], ddof=1)
    )
    assert piqa["paired"]["mean"] == pytest.approx(0.375)
    assert 0 < piqa["paired"]["p_holm"] <= 1
    assert summary["tasks"]["lambada"]["groups"]["b"]["perplexity"]["mean"] == (
        pytest.approx(np.e)
    )
    assert "| piqa | acc | 4 |" in downstream.markdown(summary)
