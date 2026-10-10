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

## Result

The Transformer, Mamba-3 and Mamba 4 runs each completed exactly 1B targets on
all 16 chips and passed the audit (`lm/results/screen-60m/`).

**NLL screen.** Mamba 4 reached held-out NLL 3.4719, against 3.4758 for Mamba-3
and 3.5352 for the Transformer. Paired 95% intervals against Mamba-3 are
−0.0052 to −0.0025 on the held-out split and −0.0059 to −0.0031 on a fresh
confirmatory holdout. The declared single-seed screen win holds.

**Claims suite.** The pre-registered suite (`validation/claims/`) tests what
NLL cannot. On fresh arrays:
- **Supported:** context-independent decode.
- **Exact retrieval inside the 1,024-token window:** refuted against the
  Transformer, supported against Mamba-3.
- **Long context:** refuted. No passkey is retrieved at any length, and NLL
  does not improve beyond the training length.
- **Calibrated confidence:** refuted.

Trained directly on synthetic tasks, the same memory recalls and retains
exactly within capacity (`lm/results/synthetic/`). The NLL gate is met, but
the paper's retrieval and long-context claims are not yet demonstrated in the
60M language model. One seed cannot establish robustness, and 125M scaling
awaits the user's authorization.

Two earlier Mamba 4 designs were tried and abandoned before this one
(`docs/iterations.md`; git history `fc618d6`, `5269fdd`).
