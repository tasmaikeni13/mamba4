# Claims-60m protocol

Frozen on 2026-10-09 before any model was scored. The 60M screen gated only
on held-out NLL. This protocol tests the paper's distinctive claims on the
three trained 60M checkpoints, with no further training and no selection:

| Model | Checkpoint |
|---|---|
| Transformer | `lm/runs/screen-60m-v1/transformer` (final, 7,630 steps) |
| Mamba-3 | `lm/runs/screen-60m-v1/mamba3` (final, 7,630 steps) |
| Mamba 4 | `lm/runs/screen-60m-v2/mamba4` (final, 7,630 steps) |

All three models were trained on 1,024-token sequences. Tasks are built by
`validation/claims.py build` with seed 20261010 and the pinned GPT-2
tokenizer; the manifest (`data/claims-v1/manifest.json`) records the array
hash `487d94eda8a64b2027bbebfb742837941fceea8484e166100a1d133201bd90bf`.
Every model scores identical arrays. Prompts are the statistical units;
slots within a prompt are not independent observations.

## Tasks

**Exact copy (exact retrieval).** L distinct random single-token words, then
the same L words again, for L in {16, 64, 256, 512, 1024, 2048, 4096}
(sequence 2L), 128 sequences per L. The L−1 predictable tokens of the second
copy are scored by top-1 accuracy under teacher forcing. Primary metric:
per-sequence mean top-1 accuracy. Also reported: the exact-match rate (every
scored token correct) and NLL.

**Multi-query associative recall (exact retrieval at load K).** K stored
`key: value` lines of distinct single-token words (disjoint key and value
pools), followed by 16 queried keys from the store, for K in {16, 32, 64, 128,
240, 496, 1008, 2032}, 128 sequences per K. Sequence length is 4K + 64, so
K ≤ 240 lies within the training context. Primary metric: per-sequence top-1
accuracy over the full vocabulary at the 16 value slots. Also reported:
accuracy among the K stored values, the exact-match rate and NLL.

**Passkey retrieval (long context).** The template of Mohtashami & Jaggi
(2023, arXiv:2305.16300), with filler repeated at sentence granularity. Total
lengths {512, 1,024, 2,048, 4,096, 8,192, 16,384} tokens, key depths {0.1, 0.3,
0.5, 0.7, 0.9}, 64 prompts per cell. Primary metric: exact match, meaning
every token of the five-digit key is top-1 under teacher forcing.

**Long-document NLL (long context).** 336 FineWeb-Edu documents of at least
16,384 GPT-2 tokens. They come from the pinned fresh shard
`sample/10BT/013_00000.parquet`, after excluding every screen-corpus
document hash and every fresh-holdout document. Each document gives
[EOS] + its first 16,384 tokens. Metric: per-document mean NLL in the position
buckets 0–128, 128–512, 512–1,024, 1,024–2,048, 2,048–4,096, 4,096–8,192 and
8,192–16,384.

**Calibration (calibrated retrieval confidence).** At every scored slot of the
three retrieval families: the top-1 probability and top-1 correctness.
Metrics: expected calibration error (15 equal-width bins) and AUROC of
confidence for correctness. LM-level ECE is computed on the long documents
within and beyond 1,024 positions. For Mamba 4, the memory variance
q^T A^-1 q of each memory layer (mean over heads) is captured at the scored
slots. Its AUROC (lower variance predicting correctness) is a diagnostic
with no peer analog.

**Decode cost versus context.** Median one-token decode latency (16 chips,
one sequence per chip, 32 timed steps after 3 warm-up steps) and cache bytes
per sequence. The cache is sized for contexts {1,024, 4,096, 16,384, 65,536}.
Transformer attention reads its whole cache, so its cost follows capacity;
recurrent caches have fixed shapes. Timing does not depend on cache contents.

**Ablation (diagnostic).** Mamba 4 with its conjugate read replaced by zero
inside every memory layer. The skip, order head and gates are kept. This
measures how much each result depends on the conjugate memory rather than on
the fifteen Mamba-3 layers. The ablation is evaluation-time only and is not
a trained model.

## Comparisons

Mamba 4 is compared with each peer on the same prompts or documents.
Accuracy and NLL use the paired mean difference with a 10,000-draw bootstrap
interval and a two-sided paired t-test. Passkey exact match uses the exact
McNemar test. Holm adjustment runs across the groups of each family, per
peer. The call is "exceeds" or "trails" when the adjusted p < 0.05, and
"matches" otherwise. For calibration, rows are resampled jointly (1,000 draws)
for the paired ECE difference. ECE "exceeds" or "trails" when that interval
excludes zero.

## Pre-registered verdict rules (per peer)

- **Exact retrieval within the training context**: copy with 2L ≤ 1,024 and
  MQAR with K ≤ 240.
  - *refuted* if Mamba 4 trails at any group;
  - *supported* if it exceeds at one or more and trails at none;
  - *matched* otherwise.
- **Long context**: copy L ≥ 1,024, MQAR K ≥ 496, passkey lengths above
  1,024, and NLL buckets beyond position 1,024.
  - *refuted* if Mamba 4 trails anywhere, or if its own NLL at positions
    8,192–16,384 is not significantly below its NLL at 512–1,024 (upper
    bootstrap bound above zero), meaning it does not benefit from context
    beyond its training length;
  - *supported* if it exceeds somewhere and neither condition holds;
  - *matched* otherwise.
- **Calibrated confidence**: paired ECE per retrieval family.
  - *refuted* if trails in any family;
  - *supported* if it exceeds in one or more and trails in none;
  - *matched* otherwise.
- **Context-independent decode**: supported when Mamba 4's latency varies by at
  most 10% across the four contexts and is below the Transformer's at 65,536.

A claim counts as achieved at 60M only when it is supported against both
peers. These are single-seed, single-checkpoint results. They describe these
trained models and cannot establish robustness across training seeds.

## Predictions recorded before scoring

- Mamba 4 will trail the Transformer on exact copy and MQAR within 1,024
  tokens. This version has no exact-retrieval mechanism: no protected
  anchors, and a positive ridge floor.
- The Transformer, trained with RoPE at 1,024 tokens, will degrade sharply
  beyond 1,024.
- Mamba 4 versus Mamba-3 is uncertain.

## Use of results

The results directory keeps every outcome. A failed claim is diagnosed and
the method revised under a new protocol version, with fresh task seeds and a
different pinned document shard. The prompts and documents here are never
used to select a later method.
