# Diagnosis and repair log

These are phase-01/02 investigations, not training iterations. The original
monograph is preserved; the active paper and downstream contracts reflect
every correction. Holdout losses remain in the artifacts.

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
