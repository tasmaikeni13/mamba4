# Phase 04: 60M screening, 1B tokens, one seed

Planned only. Train Mamba 4, Mamba-3 and Transformer at 60M parameters using
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
