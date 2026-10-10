# The memory mechanism under direct training

The 60M language model did not show exact retrieval or long-context use
(`lm/results/claims-60m/`). These studies ask whether the mechanism
itself works when a model is trained for the job. Small matched models (two
blocks of width 128, tied 12,288-token embedding) are trained from scratch
on synthetic recall and retention tasks:
- the Transformer uses attention and a SwiGLU MLP;
- Mamba-3 uses official SISO blocks with state 48;
- Mamba 4 uses memory blocks with keys of dimension 32, four heads of width
  64 and aligned keys.

Study B and the in-context regression study were designed and recorded
before they ran (`validation/synthetic-protocol.md`). Study A was an
earlier development study. One training seed per model; the learning rate
is chosen on a development seed and every accuracy comes from a separate
evaluation seed, with 256 sequences per group. Raw records:
`study-a.json` and `study-b.json`.

Per-sequence state at 512 tokens, in floats: Transformer 262,144 (its cache);
Mamba-3 24,576; Mamba 4 20,608.

## Study B (recorded design, 6,000 steps)

### Associative recall: accuracy over all K queries

| Model | K=8 | K=16 | K=32 | K=64 | K=128 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 33.7% | 18.5% | 10.8% | 5.8% | 2.8% | 0.0005 |
| Mamba-3 | 21.1% | 15.5% | 8.2% | 4.3% | 2.9% | 0.0005 |
| Mamba 4 | 100.0% | 100.0% | 99.9% | 97.5% | 87.5% | 0.005 |

### Unknown keys: accuracy; half the queries were never stored

| Model | K=8 | K=16 | K=32 | K=64 | K=128 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.005 |
| Mamba-3 | 39.6% / 100.0% | 38.7% / 100.0% | 33.7% / 100.0% | 23.0% / 99.7% | 12.2% / 99.5% | 0.005 |
| Mamba 4 | 95.1% / 94.6% | 85.5% / 94.0% | 49.1% / 93.2% | 8.3% / 95.3% | 0.2% / 99.6% | 0.0015 |

Cells: accuracy on stored keys / on never-stored keys (NONE).

### Retention among distractors: 8 pairs, trained at 512 tokens

| Model | Length=512 | Length=1024 | Length=2048 | Length=4096 | Length=8192 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 30.8% | 9.5% | 8.1% | 0.0% | 0.0% | 0.005 |
| Mamba-3 | 98.1% | 0.1% | 0.1% | 0.0% | 0.5% | 0.005 |
| Mamba 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.005 |

### Successor lookups: one hop / two hops

| Model | K=8 | K=16 | K=32 | K=64 | LR |
|---|---:|---:|---:|---:|---:|
| Transformer | 15.1% / 16.7% | 6.7% / 6.4% | 1.9% / 2.1% | 0.7% / 0.8% | 0.005 |
| Mamba-3 | 97.4% / 85.1% | 92.7% / 87.3% | 82.6% / 79.9% | 59.2% / 63.1% | 0.0005 |
| Mamba 4 | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 0.0005 |

Non-finite training at some learning rate: mamba4-shift/unknown

Mamba 4's highest learning rate diverged on the unknown-keys task; the
selection rule then picked from the finite runs.

AUROC of P(NONE) for telling never-stored keys from stored ones:

| Model | K=8 | K=16 | K=32 | K=64 | K=128 |
|---|---:|---:|---:|---:|---:|
| Transformer | 0.496 | 0.491 | 0.496 | 0.496 | 0.496 |
| Mamba-3 | 1.000 | 1.000 | 1.000 | 0.999 | 0.995 |
| Mamba 4 | 0.984 | 0.973 | 0.937 | 0.842 | 0.736 |

Predictions recorded before the run, against the outcome:
- **Near-exact recall up to the key dimension (32) and degradation beyond
  it:** borne out. Accuracy is at least 99.9% for K ≤ 32, then 97.5% and
  87.5%.
- **Retention to 8,192 tokens while the Transformer falls off beyond 512:**
  borne out. Mamba 4 is exact at every length; Mamba-3 is exact at the
  trained length but fails beyond it.
- **The Transformer approaches exact recall with enough steps:** not borne
  out within 6,000 steps (at most 33.7%).

Two-hop successor lookups are exact for Mamba 4 at every K. With unknown
keys, Mamba 4 answers stored keys far better at low load (95.1% at K = 8,
against 39.6% for Mamba-3), but its stored-key accuracy collapses beyond
K = 32. Mamba-3 separates absent keys better at every load.

## Study A (development, 3,000 steps)

The memory layer here had no key alignment, and weight decay applied to
every parameter, including gates, floors and norms. A variant with evidence
precision up to 16 and floor 0.02 diverged.

### Associative recall: accuracy over all K queries

| Model | K=8 | K=16 | K=32 | K=64 | K=128 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 32.4% | 17.5% | 9.2% | 3.7% | 1.2% | 0.0005 |
| Mamba-3 | 14.0% | 8.8% | 8.4% | 3.9% | 3.2% | 0.0015 |
| Mamba 4 without key shift | 98.5% | 98.3% | 84.5% | 43.2% | 28.9% | 0.0005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

### Retention among distractors: 8 pairs, trained at 512 tokens

| Model | Length=512 | Length=1024 | Length=2048 | Length=4096 | Length=8192 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 33.7% | 11.4% | 11.7% | 3.8% | 0.4% | 0.0015 |
| Mamba-3 | 54.6% | 0.4% | 0.1% | 0.0% | 0.0% | 0.0015 |
| Mamba 4 without key shift | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

Non-finite training at some learning rate: mamba4-strong-lowfloor/mqar, mamba4-strong-lowfloor/noisy

## In-context regression

A trained model reads (x, y) examples of a random linear map and predicts y
with a variance. Mamba 4 comes within 0.04 nats of the exact Bayes
predictive NLL on late examples, against 0.20 for Mamba-3 and 0.83 for the
Transformer. All three reach about 90% interval coverage. See
`lm/results/regression/`.
