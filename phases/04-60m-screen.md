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

## Versioned iteration: screen-60m-v2

The v1 Mamba 4 run was interrupted after 1,026 steps and trailed both peers
at matched steps; its record is `lm/results/screen-60m-v1/mamba4-interrupted.json`.
Following the iteration rule above, `lm/configs/screen-protocol-v2.json`
changes only the Mamba 4 method. Data, tokenizer, seed, budget, batch,
optimizer, schedule and held-out documents are unchanged, and the completed
Transformer and Mamba-3 runs are reused. Their execution sources must stay
byte-identical apart from Mamba 4 files and additive config fields; the run
controller and final audit check this.

Settings were chosen only from development pilots: the first 1,000 protocol
steps, scored by loss on unseen training batches against the peers' logged
losses on identical batches. The held-out split was never used for
selection. The rule, recorded before the decisive pilots, is in
`validation/pilots/SELECTION.md`; every pilot is kept in
`lm/results/pilots-v2/`. The selected composition (fifteen official Mamba-3
layers and four selective memory layers) is frozen in
`lm/configs/mamba4-60m-v2.json`.

The run is driven by `validation/run_screen_v2.py`. It resumes after external
interruptions and merges logs written by whichever host is JAX process 0. It
then audits the run, hashes the final checkpoint on all four hosts, and
evaluates every model per sequence on the original held-out split and on a
fresh confirmatory holdout. The fresh holdout has 2,097,152 targets from a
pinned official FineWeb-Edu shard, with every exact screen-document hash
removed. `validation/verify_screen_v2.py` re-audits all three runs and
decides the strict screen win. It reports paired moving-block bootstrap
intervals over whole sequences. A loss remains a reported outcome, and one
seed cannot establish robustness.

**Result.** Mamba 4 v2 reached held-out NLL 3.4727 against 3.4758 for Mamba-3
and 3.5352 for the Transformer (paired 95% intervals versus Mamba-3: −0.0044
to −0.0017 on the held-out split, −0.0056 to −0.0031 on the fresh holdout).
The declared single-seed screen win holds and is recorded in
`lm/results/screen-60m-v2/audit.json` and `REPORT.md`. The margin is small and
narrowed during training; one seed does not authorize 125M scaling by itself,
which also awaits the user's authorization.

