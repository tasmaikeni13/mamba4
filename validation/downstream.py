"""Zero-shot downstream benchmarks scored by continuation log-likelihood.

Prompts, answer choices, tokenization and metrics follow the EleutherAI LM
Evaluation Harness task definitions for ARC-Easy, ARC-Challenge, HellaSwag,
PIQA, WinoGrande (partial scoring), LAMBADA (OpenAI), SciQ, OpenBookQA and
BoolQ. Every answer choice becomes one row: an end-of-text token (every
training document follows one), the context, then the continuation, of which
only the continuation is scored. A choice's score is its summed
log-likelihood; ``acc`` takes the argmax and ``acc_norm`` first divides by the
choice's character length. LAMBADA reports greedy whole-word accuracy and
perplexity. Every file is pinned to a hub revision and recorded by SHA-256.

``prepare``  (CPU) downloads the pinned files and builds the rows.
``evaluate`` (pod) scores every row for every model.
``report``   (CPU) accuracies, seed means and paired document-level tests.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np

from lm.data import sha256_file
from lm.runtime import atomic_json
from validation.claims import BOOTSTRAP_SEED, array_digest, holm, mcnemar, paired_test

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/downstream"
EOT = 50256
MAX_LENGTH = 1024
BUCKETS = (128, 256, 512, 1024)
PER_DEVICE = {128: 64, 256: 32, 512: 16, 1024: 8}
ARC = ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453")
SOURCES = {
    "arc_easy": (*ARC, "ARC-Easy/test-00000-of-00001.parquet"),
    "arc_challenge": (*ARC, "ARC-Challenge/test-00000-of-00001.parquet"),
    "hellaswag": (
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "data/validation-00000-of-00001.parquet",
    ),
    "piqa": (
        "baber/piqa",
        "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "piqa_validation.parquet",
    ),
    "winogrande": (
        "allenai/winogrande",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "winogrande_xl/validation-00000-of-00001.parquet",
    ),
    "lambada": (
        "EleutherAI/lambada_openai",
        "900124bf3b8235c6daf21033af9948b3f07346c4",
        "default/test/default.parquet",
    ),
    "sciq": (
        "allenai/sciq",
        "2c94ad3e1aafab77146f384e23536f97a4849815",
        "data/test-00000-of-00001.parquet",
    ),
    "openbookqa": (
        "allenai/openbookqa",
        "388097ea7776314e93a529163e0fea805b8a6454",
        "main/test-00000-of-00001.parquet",
    ),
    "boolq": (
        "google/boolq",
        "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "data/validation-00000-of-00001.parquet",
    ),
}
# Declared before any model is scored: length-normalized accuracy where the
# answer choices differ much in length, as in the Mamba papers' tables.
HEADLINE = {
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc",
    "winogrande": "acc",
    "lambada": "acc",
    "sciq": "acc",
    "openbookqa": "acc_norm",
    "boolq": "acc",
}


def _hellaswag_text(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def requests(task, row):
    """(contexts, choices, gold, delimiter) for one document."""
    if task in ("arc_easy", "arc_challenge"):
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"])
        context = f"Question: {row['question']}\nAnswer:"
        return [context] * len(choices), choices, gold, " "
    if task == "hellaswag":
        context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = _hellaswag_text(row["activity_label"] + ": " + context)
        choices = [_hellaswag_text(ending) for ending in row["endings"]]
        return [query] * len(choices), choices, int(row["label"]), " "
    if task == "piqa":
        context = f"Question: {row['goal']}\nAnswer:"
        return [context] * 2, [row["sol1"], row["sol2"]], int(row["label"]), " "
    if task == "winogrande":
        sentence = row["sentence"]
        blank = sentence.index("_")
        target = sentence[blank + 1 :].strip()
        contexts = [
            sentence[:blank] + row["option1"],
            sentence[:blank] + row["option2"],
        ]
        return contexts, [target, target], int(row["answer"]) - 1, " "
    if task == "lambada":
        words = row["text"].split(" ")
        return [" ".join(words[:-1])], [" " + words[-1]], 0, ""
    if task == "sciq":
        context = f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:"
        choices = [
            row["distractor1"],
            row["distractor2"],
            row["distractor3"],
            row["correct_answer"],
        ]
        return [context] * 4, choices, 3, " "
    if task == "openbookqa":
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"].lstrip())
        return [row["question_stem"]] * len(choices), choices, gold, " "
    if task == "boolq":
        context = f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:"
        return [context] * 2, ["no", "yes"], int(bool(row["answer"])), " "
    raise ValueError(task)


def encode_pair(tokenizer, context, continuation):
    """Harness tokenization: trailing context spaces join the continuation,
    which is the whole encoding past the context's own tokens."""
    spaces = len(context) - len(context.rstrip())
    if spaces:
        context, continuation = context[:-spaces], context[-spaces:] + continuation
    whole = tokenizer.encode(context + continuation, add_special_tokens=False).ids
    context_ids = tokenizer.encode(context, add_special_tokens=False).ids
    return context_ids, whole[len(context_ids) :]


def bucket_for(length):
    for bucket in BUCKETS:
        if length <= bucket:
            return bucket
    raise ValueError(f"Row of {length} tokens exceeds every bucket")


def download(task, directory):
    repo, revision, path = SOURCES[task]
    target = directory / f"{task}.parquet"
    if not target.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
        with urllib.request.urlopen(url, timeout=300) as response:
            payload = response.read()
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return target, {
        "repo": repo,
        "revision": revision,
        "path": path,
        "sha256": sha256_file(target),
    }


def build_rows(tokenizer, task, documents):
    """Teacher-forced rows plus (task, document, choice, gold, characters)."""
    rows, meta = [], []
    for index, document in enumerate(documents):
        contexts, choices, gold, delimiter = requests(task, document)
        for choice, (context, text) in enumerate(zip(contexts, choices)):
            context_ids, continuation_ids = encode_pair(
                tokenizer, context, delimiter + text
            )
            if not continuation_ids:
                raise ValueError(f"{task} document {index} has an empty continuation")
            ids = ([EOT] + context_ids + continuation_ids)[-(MAX_LENGTH + 1) :]
            scored = min(len(continuation_ids), len(ids) - 1)
            start = len(ids) - 1 - scored
            rows.append(
                dict(
                    ids=np.asarray(ids[:-1], np.int32),
                    positions=np.arange(start, len(ids) - 1, dtype=np.int32),
                    targets=np.asarray(ids[start + 1 :], np.int32),
                )
            )
            meta.append((task, index, choice, gold, len(text)))
    return rows, meta


def pack(rows):
    """Bucketed, padded arrays; ``row`` maps each slot back to its request."""
    arrays, counts = {}, {}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    for bucket in BUCKETS:
        members = [i for i in order if bucket_for(len(rows[i]["ids"])) == bucket]
        if not members:
            continue
        width = max(len(rows[i]["positions"]) for i in members)
        ids = np.full((len(members), bucket), EOT, np.int32)
        positions = np.full((len(members), width), -1, np.int32)
        targets = np.zeros((len(members), width), np.int32)
        for slot, i in enumerate(members):
            ids[slot, : len(rows[i]["ids"])] = rows[i]["ids"]
            positions[slot, : len(rows[i]["positions"])] = rows[i]["positions"]
            targets[slot, : len(rows[i]["targets"])] = rows[i]["targets"]
        prefix = f"b{bucket}"
        arrays[f"{prefix}_input_ids"] = ids
        arrays[f"{prefix}_positions"] = positions
        arrays[f"{prefix}_targets"] = targets
        arrays[f"{prefix}_candidates"] = np.zeros((len(members), 1), np.int32)
        arrays[f"{prefix}_row"] = np.asarray(members, np.int32)
        counts[str(bucket)] = len(members)
    return arrays, counts


def prepare(options):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    output = Path(options.output)
    (output / "sources").mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(options.tokenizer)
    if tokenizer.token_to_id("<|endoftext|>") != EOT:
        raise ValueError("Expected the GPT-2 tokenizer")
    rows, meta, sources = [], [], {}
    for task in SOURCES:
        path, sources[task] = download(task, output / "sources")
        documents = pq.read_table(path).to_pylist()
        sources[task]["documents"] = len(documents)
        task_rows, task_meta = build_rows(tokenizer, task, documents)
        rows += task_rows
        meta += task_meta
        print(task, len(documents), "documents", len(task_rows), "rows", flush=True)
    arrays, counts = pack(rows)
    np.savez(output / "requests.npz", **arrays)
    columns = ["task", "document", "choice", "gold", "characters"]
    (output / "requests.json").write_text(
        json.dumps({"columns": columns, "rows": meta})
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer_sha256": sha256_file(options.tokenizer),
        "array_sha256": array_digest(arrays),
        "requests_sha256": sha256_file(output / "requests.json"),
        "requests": len(rows),
        "buckets": counts,
        "sources": sources,
        "max_length": MAX_LENGTH,
        "prefix": "<|endoftext|> before every context",
        "headline": HEADLINE,
        "definitions": "EleutherAI LM Evaluation Harness tasks, zero-shot",
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=1))


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "requests.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    if array_digest(arrays) != manifest["array_sha256"]:
        raise ValueError("Downstream arrays differ from their manifest")
    if sha256_file(directory / "requests.json") != manifest["requests_sha256"]:
        raise ValueError("Downstream request table differs from its manifest")
    table = json.loads((directory / "requests.json").read_text())
    return arrays, table, manifest


def evaluate(options):
    import jax

    from lm.runtime import initialize
    from validation.claims import load_model, run_rows, scored_function

    hardware = initialize(True)
    arrays, table, manifest = load(options.data)
    raw = Path(options.output)
    results = {
        "hardware": hardware,
        "array_sha256": manifest["array_sha256"],
        "models": {},
    }
    for item in options.models:
        name, paths = item.split("=", 1)
        config_path, run = paths.split(",")
        _, model, params, digest = load_model(config_path, run)
        replicated = jax.device_put_replicated(params, jax.local_devices())
        mapped = scored_function(model, False)
        loglik = np.zeros(len(table["rows"]))
        greedy = np.zeros(len(table["rows"]), bool)
        started = time.perf_counter()
        for bucket in BUCKETS:
            prefix = f"b{bucket}"
            if f"{prefix}_input_ids" not in arrays:
                continue
            timings = []
            fields = ("input_ids", "positions", "targets", "candidates")
            out = run_rows(
                mapped,
                replicated,
                {key: arrays[f"{prefix}_{key}"] for key in fields},
                PER_DEVICE[bucket],
                timings,
            )
            valid = arrays[f"{prefix}_positions"] >= 0
            rows = arrays[f"{prefix}_row"]
            loglik[rows] = np.sum(np.where(valid, out["target_logp"], 0.0), axis=1)
            greedy[rows] = np.all(np.where(valid, out["correct"], True), axis=1)
            print(name, prefix, f"{sum(timings):.1f}s", flush=True)
        results["models"][name] = {
            "config": config_path,
            "checkpoint": run,
            "checkpoint_sha256": digest,
            "seconds": time.perf_counter() - started,
        }
        if jax.process_index() == 0:
            raw.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw / f"{name}.npz", loglik=loglik, greedy=greedy)
    if jax.process_index() == 0:
        atomic_json(raw / "evaluation.json", results)
    jax.distributed.shutdown()


def score(table, loglik, greedy):
    """Per task and document: acc and acc_norm hits; for LAMBADA, greedy hits
    and the target log-likelihood."""
    grouped = {}
    for index, (task, document, choice, gold, characters) in enumerate(table["rows"]):
        grouped.setdefault(task, {}).setdefault(document, []).append(
            (choice, gold, characters, index)
        )
    result = {}
    for task, documents in grouped.items():
        acc, norm, hits, logs = [], [], [], []
        for document in sorted(documents):
            entries = sorted(documents[document])
            gold = entries[0][1]
            indices = np.asarray([entry[3] for entry in entries])
            scores = loglik[indices]
            lengths = np.asarray([max(entry[2], 1) for entry in entries], np.float64)
            acc.append(int(np.argmax(scores)) == gold)
            norm.append(int(np.argmax(scores / lengths)) == gold)
            hits.append(bool(greedy[indices[gold]]))
            logs.append(float(scores[gold]))
        if task == "lambada":
            result[task] = {"acc": np.asarray(hits), "loglik": np.asarray(logs)}
        else:
            result[task] = {"acc": np.asarray(acc), "acc_norm": np.asarray(norm)}
    return result


def _spread(values):
    values = np.asarray(values, np.float64)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "sd": sd}


def summarize(scores, groups):
    """Seed means per group and, for two groups, paired document-level tests
    of the seed-averaged hits with Holm adjustment across tasks."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(groups)
    summary = {"groups": groups, "headline": HEADLINE, "tasks": {}}
    for task in SOURCES:
        metric = HEADLINE[task]
        entry = {"metric": metric, "groups": {}}
        for group, members in groups.items():
            item = {"members": {}}
            for member in members:
                values = scores[member][task]
                entry["documents"] = int(len(values["acc"]))
                record = {
                    key: float(values[key].mean())
                    for key in ("acc", "acc_norm")
                    if key in values
                }
                if "loglik" in values:
                    record["perplexity"] = float(np.exp(-values["loglik"].mean()))
                item["members"][member] = record
            item[metric] = _spread([item["members"][m][metric] for m in members])
            if task == "lambada":
                item["perplexity"] = _spread(
                    [item["members"][m]["perplexity"] for m in members]
                )
            entry["groups"][group] = item
        if len(names) == 2:
            first, second = (
                np.mean([scores[m][task][metric] for m in groups[name]], axis=0)
                for name in names
            )
            entry["paired"] = {"groups": names, **paired_test(first, second, rng)}
            if all(len(groups[name]) == 1 for name in names):
                entry["paired"]["mcnemar"] = mcnemar(first > 0.5, second > 0.5)
        summary["tasks"][task] = entry
    if len(names) == 2:
        adjusted = holm(
            np.asarray([summary["tasks"][t]["paired"]["p"] for t in SOURCES])
        )
        for task, value in zip(SOURCES, adjusted):
            summary["tasks"][task]["paired"]["p_holm"] = float(value)
    summary["average"] = {
        group: _spread(
            [
                np.mean([scores[m][t][HEADLINE[t]].mean() for t in SOURCES])
                for m in members
            ]
        )
        for group, members in groups.items()
    }
    return summary


def markdown(summary):
    names = list(summary["groups"])
    paired = len(names) == 2
    columns = ["Task", "Metric", "Documents", *names]
    if paired:
        columns += [f"{names[0]} − {names[1]} [95% bootstrap]", "Holm p"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]

    def cell(spread, count):
        text = f"{100 * spread['mean']:.2f}"
        return text + (f" ± {100 * spread['sd']:.2f}" if count > 1 else "")

    for task, entry in summary["tasks"].items():
        metric = entry["metric"]
        cells = [task, metric, str(entry["documents"])]
        cells += [
            cell(entry["groups"][name][metric], len(summary["groups"][name]))
            for name in names
        ]
        if paired:
            test = entry["paired"]
            low, high = test["bootstrap_95"]
            cells.append(
                f"{100 * test['mean']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"
            )
            cells.append(f"{test['p_holm']:.3g}")
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["average", "headline", ""]
    cells += [
        cell(summary["average"][name], len(summary["groups"][name])) for name in names
    ]
    lines.append("| " + " | ".join(cells + ([""] * 2 if paired else [])) + " |")
    lambada = summary["tasks"]["lambada"]["groups"]
    perplexity = ", ".join(
        f"{name} {lambada[name]['perplexity']['mean']:.2f}" for name in names
    )
    return "\n".join(lines) + f"\n\nLAMBADA perplexity: {perplexity}\n"


def report(options):
    _, table, manifest = load(options.data)
    raw = Path(options.raw)
    evaluation = json.loads((raw / "evaluation.json").read_text())
    if evaluation["array_sha256"] != manifest["array_sha256"]:
        raise ValueError("The evaluation scored different requests")
    scores = {}
    for name in evaluation["models"]:
        with np.load(raw / f"{name}.npz") as stored:
            scores[name] = score(table, stored["loglik"], stored["greedy"])
    if options.groups:
        groups = {}
        for item in options.groups:
            group, members = item.split("=", 1)
            groups[group] = members.split(",")
    else:
        groups = {name: [name] for name in scores}
    summary = summarize(scores, groups)
    summary["evaluation"] = evaluation["models"]
    summary["array_sha256"] = manifest["array_sha256"]
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    (output / "summary.md").write_text(markdown(summary))
    print(markdown(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--tokenizer", default="data/fineweb-edu-1b/tokenizer.json")
    make.add_argument("--output", default=str(DATA.relative_to(ROOT)))
    run = sub.add_parser("evaluate")
    run.add_argument("--data", default=str(DATA.relative_to(ROOT)))
    run.add_argument("--output", required=True)
    run.add_argument("models", nargs="+", help="name=config.json,run_directory")
    summary = sub.add_parser("report")
    summary.add_argument("--data", default=str(DATA.relative_to(ROOT)))
    summary.add_argument("--raw", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--groups", nargs="*", help="group=model,model,...")
    options = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "report": report}[options.action](
        options
    )


if __name__ == "__main__":
    main()
