# Diagnosis and repair log

Iterations A–G are the phase-01/02 investigations of the operator theory. The original monograph is preserved; the active paper and
downstream contracts reflect every correction. Holdout losses remain in the
artifacts.

| Iteration | Cause / witness | Repair | Verification / consequence |
|---|---|---|---|
| A: scan/prior semantics | Discounts do not commute; merging posteriors counts two priors. | Chronological affine summaries, evidence-only addition, prior-aware merge. | Lean composition/fold/prior identities; every-prefix/merge tests. |
| B: exactness/conditioning | Positive ridge shrinks a unit write; near-dependent normal equations lose precision. | Separate Bayesian ridge and exact cached QR anchors, count geometry/IDs. | Left-inverse/Gram proofs, weighted bound MC, QR/float32 diagnostics. |
| C: uncertainty/risk | Conflicting keys give small c; discounted noise covariance uses squared weights. | Conditional calibration, corrected bias/weighted covariance. | Finite-distribution risk proofs; risk and calibration holdouts. |
| D: floor | Original rule gives 0.716–1.349 epsilon at d=64, lambda=.99; fixed-floor forgetting has full-rank injection. | Guaranteed initialized cyclic anisotropic prior; constant-gate contract and explicit fixed-floor fallback. | Five Lean cycle/bound results; factor/dense checks across widths and decays. |
| E: cascade | Merges overload d; binary occupancy collapses at powers of two; background contaminates anchors. | Redundant banks, <=d anchor reselection, separate background evidence. | Lean mass/bank invariants; all-prefix tests; 72 retained/1,976 discarded exact items. |
| F: hops/order | Unit codes may expand; token-local undiscounted evidence loses AB/BA order. | Actual Lipschitz bound and H costs; an auxiliary affine order head for the witness. | General geometric envelope and graph MC; constructive order example. |
| G: peer target | Growing-cache softmax recalls beyond d; precision smoothing is optimal for repeated noisy keys. | Keep signed-query/conditional-regression advantages; retract universal dominance and averaging impossibilities. | All recall/functional holdouts and baseline losses retained; trained win stays a later gate. |

The protocol was frozen before full evaluation. A quick conformance run exposed
a JSON serialization issue, repaired before full artifact release. A second
full execution verified cached versus recomputed QR with the same estimator,
hypothesis and tuning choices; its source hashes/output are committed. No
holdout-driven hyperparameter change is presented as a new win.

Future estimator/hypothesis changes need a new protocol version, development
selection and fresh held-out seeds. Diagnostics can reuse a benchmark, but an
adaptively changed method cannot certify a new win on that same holdout.
Information-theoretic obstructions require declared budget tradeoffs rather
than hidden caches or unlimited retries.

## Language-model design record

Two earlier Mamba 4 language-model designs were tried and abandoned before
the current one. The first used constant per-head gates and protected QR
banks, trailed both peers at matched steps and ran 33× slower than Mamba-3.
The second lacked the key alignment described below. Their code and records
are in the git history (commits `fc618d6` and `5269fdd`); the second survives
in this tree only as the no-key-shift ablation.

**Composition.** One-thousand-step development pilots on unseen training
batches compared hybrid compositions under a rule recorded before the
deciding pilots (`validation/pilots/SELECTION.md`, records in
`lm/results/pilots/`). Replacing many Mamba-3 layers with narrow memory layers
hurt. Spending five Mamba-3 layers' budget on four wide memory layers (key
dimension 64) helped. The held-out split was never used for selection.

**Key alignment.** The pre-registered claims suite showed that the model
without alignment rarely copied and never retrieved a passkey, while the same
memory layer, trained directly on synthetic recall, reached 98% recall within
capacity and 100% retention to 16× its training length. The mechanism works;
language-model training rarely teaches it to retrieve. Writing each value
under the key of the preceding context makes an induction lookup the
memory's default geometry. It adds no parameters. Its pilot passed a rule
recorded before it ran (`validation/pilots/SELECTION-key-alignment.md`).

**Diagnosis kept for the record.** On passkey prompts, the trained gates of
the unaligned model keep 30–80% of the key-token evidence across about 400
filler tokens. But the learned floor stays near 1 while β ≤ 1, and many heads
write the repetitive filler strongly, so one write is shrunk by half or more
(`Retention.lean`). A strong-write, low-floor variant (β ≤ 16, floor 0.02)
diverged in every synthetic run and was dropped.

**Outcome.** With alignment, the final model:
- has the best NLL of all three architectures on both holdouts;
- answers 69 of 128 frozen recall prompts, against 45 without alignment;
- still trails the Transformer on in-window copying and on passkey retrieval
  (`lm/results/claims-60m/`).

The synthetic and regression studies (`lm/results/synthetic/`,
`lm/results/regression/`) measure the mechanism directly.

**Engineering findings.**
- A Python-unrolled factor made an 18-layer program compile for more than
  16 minutes; the blocked lane-major factor replaced it.
- Isolated multi-host microbenchmarks hit TPU launch-identity mismatches even
  though the hosts' optimized HLO was identical. Compiling ahead of time
  behind a barrier fixed them.
- A small Mamba-3 + memory hybrid in the synthetic harness hit the same halt
  on every attempt (f32 and bf16, one flat all-reduce), so the synthetic
  studies use pure stacks.
- Only JAX process 0 writes the persistent compile cache, so caches are
  cleared on every host before each launch.
