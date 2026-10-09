# Phase 04: 60M screening, 1B tokens, one seed

Authorized and configured in `lm/configs/*-60m.json`; full-run completion
remains unproven until final checkpoints and budgets are audited. Train
Mamba 4, Mamba-3 and Transformer at 60M parameters using
all designated v4-32 resources, one training seed per architecture and exactly
1B training tokens from
[codelion/fineweb-edu-1B](https://huggingface.co/datasets/codelion/fineweb-edu-1B).
Pin the dataset revision, tokenizer, token counting and leakage-free held-out
split. Record effective embedding/model parameter matching, FLOPs, batch,
tokens per second and observed device use; do not infer these from names.

Declare the loss/recall/stability/latency criteria before the run. Use the same
protocol for all peers. Diagnose any failure through residuals, key geometry,
factor/gradient traces and phase-01 assumptions. Research and replace failed
methods, update all dependent contracts and rerun the fair screen with a
fresh evaluation protocol. The target is a measured Mamba 4 win before scaling;
one seed cannot prove robustness. Retain each failed iteration and its cost.

Gate: validated complete token budgets, healthy learning, fair matched peers
and the declared screen win. Do not scale on an operator-only win or fabricate
success when the fair trained comparison fails.

The frozen `lm/configs/screen-protocol.json` specifies seed 42, 7,630 optimizer
steps, the final 51,712-target mask, held-out NLL, stability, textual recall
and latency metrics. The whole-document held-out split contains 2,097,152
targets. After reserving it, 998,868,400 source training targets remain under
the pinned GPT2 tokenizer, so the final 1,131,600 training targets use an
explicit second training-document permutation (0.11316% reuse). No held-out
document is reused. The actual corpus ledger and independent full-stream
audit are in `docs/fineweb-data.md`.

The audited report must retain a failed screen win. Completing the three
requested runs does not itself establish the promotion gate or authorize
125M work.
