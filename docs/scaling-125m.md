# 125M, 3B-token study: protocol fixed before the sweep

This file records phases 05–06 as declared before any 125M training step. The
user authorized both phases on 10 October 2026. Changes after a sweep run
starts are appended as dated amendments; nothing above them is edited.

## Models

| Model | Shape | Total parameters | Non-embedding |
|---|---|---:|---:|
| Transformer | width 768, 12 layers, 12 heads of 64, SwiGLU 2,048, RoPE | 123,551,232 | 84,953,856 |
| Mamba 4 | width 768: 17 official Mamba-3 SISO layers (state 128, 24 heads of 64) and 4 memory layers (12 heads, keys 64, values 128), `SSSMSSSSMSSSSMSSSSMSS` | 122,636,368 | 84,038,992 |

Both tie a 50,257 × 768 GPT-2 embedding. Mamba 4 keeps every 60M design
choice: key alignment, the undiscounted floor (minimum 0.25, initial 1), the
evidence range, the width-4 convolution, the order scan and the confidence
gate. Only width, depth and the Mamba-3 state size change, to reach the
parameter count with TPU-aligned shapes. Mamba 4 has 0.74% fewer parameters.

## Data

`data/fineweb-edu-3b`: shards 000–004 of FineWeb-Edu `sample/10BT` at
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, GPT-2 tokenization and
a seed-125 document split. Documents of the claims tasks and both fresh
holdouts are excluded by exact content hash (131 removed). The split has
3,000,000,000 training targets and 2,097,152 held-out targets from 2,090
whole documents. Every run reads the same batches in the same order.

## Training

- 3,000,000,000 targets per run: global batch 256 × 1,024 tokens, 11,445
  steps. The last step is partial, and its padded targets have zero loss
  weight.
- AdamW with β = (0.9, 0.95), ε = 1e-8 and weight decay 0.1 on matrices;
  gradient clipping at 1.0.
- 300 warm-up steps, then cosine decay to 10% of the peak.
- Peak learning rate chosen per architecture by the sweep below.
- Seeds 42, 43 and 44 change initialization only. Runs are ordered by seed,
  alternating architecture.

## Learning-rate sweep (phase 05)

Each architecture gets the same grid, {6e-4, 1.2e-3, 2.4e-3}. Every point is
a complete short run: the main configuration with its budget cut to
300,000,000 targets (1,145 steps, warm-up capped at a tenth of the steps).
`validation.pilot` records each batch's loss before that batch updates the
model, so every loss is on unseen training data. The held-out split is never
opened.

Selection rule (`validation/sweep.py`):
- **Choice:** per architecture, take the rate with the lowest mean
  pre-update loss over the last 100 steps.
- **Divergence:** a run that stops on a nonfinite loss scores infinity.
- **Grid edge:** if the winner sits at the edge of the grid, the next point
  beyond it (factor 2) is run for that architecture. This repeats at most
  twice.

Both architectures get identical trial budgets, apart from any rule-triggered
extensions. Every trial is kept.

## Evaluation (phase 06)

All are computed after training, on frozen checkpoints:
- **Held-out NLL** on the 2,097,152 held-out targets (primary).
- **Fresh-holdout NLL** on `data/fresh-holdout`, documents from an unused
  shard.
- **Zero-shot downstream:** ARC-Easy/Challenge, HellaSwag, PIQA, WinoGrande,
  LAMBADA, SciQ, OpenBookQA and BoolQ, from hub files pinned by revision and
  SHA-256 (`validation/downstream.py`). The headline metric per task is fixed
  in the code.
- **Claims suite:** the frozen claims protocol (`validation/claims/`) on its
  fresh arrays: copying, associative recall, passkey, long-document NLL,
  calibration and decode cost.
- **Recall diagnostic:** recorded at the end of training.

Seed-level results are reported for every metric. Architecture differences
use the mean over seeds with the three paired seed differences, and for the
benchmarks the paired document-level bootstrap of seed-averaged correctness,
with Holm adjustment across tasks. Three seeds give limited power. A Mamba 4
loss is a publishable result.

Declared outcome for the primary metric:
- **Win:** Mamba 4's mean held-out NLL is lower and all three seed
  differences favour it.
- **Loss:** the mean is higher and all three seed differences favour the
  Transformer.
- **Inconclusive:** any other pattern.
