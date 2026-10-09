# Synthetic study B and in-context regression: design

Recorded 2026-10-09 22:31 UTC, before either study runs.

## Study B (token tasks)

Models are matched small stacks: two blocks of width 128 with a tied
12,288-token embedding.
- Transformer: attention plus SwiGLU MLP.
- Mamba-3: official SISO blocks, d_state 48.
- Mamba 4: selective memory blocks, key dimension 32, four heads of width 64,
  aligned keys (`memory_key_shift`).

The no-key-shift memory layer was measured in study A. Pure stacks are used
because a small Mamba-3 + memory hybrid hits the TPU launch-identity halt
(docs/iterations.md).

Every model trains on every task with the same data stream:
- 6,000 steps of 256 sequences at 512 tokens;
- AdamW (β2 0.98) with weight decay 0.1 on matrices only;
- 5% warm-up, then cosine decay to 10%;
- learning rates {5e-4, 1.5e-3, 5e-3}.

Each model's learning rate is chosen by mean development accuracy (seed 2);
every reported accuracy comes from evaluation seed 3, with 256 sequences per
group.

| Task | Content | Primary metric |
|---|---|---|
| `mqar` | K ∈ {8, 16, 32, 64, 128} stored pairs, every key queried | accuracy per K |
| `unknown` | as `mqar`, but half of the queries ask for never-stored keys whose answer is NONE | accuracy on stored and on unknown keys; AUROC of P(NONE) for absence |
| `noisy` | eight pairs among random distractors, trained at 512 tokens, scored at 512–8,192 | accuracy per length |
| `hops` | a stored random cycle over K ∈ {8, 16, 32, 64} nodes, with one-hop and two-hop queries | accuracy per hop count and K |

Predictions:
- Mamba 4 is near-exact on `mqar` up to its per-head key dimension, and
  degrades beyond it.
- Mamba 4 keeps most of its `noisy` accuracy to 8,192 tokens; the Transformer
  falls off beyond 512.
- With enough steps the Transformer can approach exact `mqar` at every K,
  because its cache grows with the sequence.

## In-context regression (`validation/regression.py`)

Task: w ~ N(0, I_8); 32 examples per sequence, with y = w·x + 0.5ε. At every
x the model outputs a mean and a log variance, trained by Gaussian NLL.

The same three models train for 4,000 steps of 256 sequences, with the same
learning-rate grid. Each model's learning rate is chosen by development NLL
(seed 12); scores use evaluation seed 13, with 2,048 sequences.

Metrics are squared error, Gaussian NLL and 90% coverage, for the model and
for the exact posterior predictive. They are reported per example index and
over the second half of each sequence.

Prediction: Mamba 4 approaches Bayes in both error and coverage, because its
memory computes ridge regression in context.
