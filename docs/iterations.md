# Diagnosis and repair log

Iterations A–G are phase-01/02 investigations; H is the first trained-screen
iteration. The original monograph is preserved; the active paper and
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

## Training-screen iterations

| Iteration | Cause / witness | Repair | Verification / consequence |
|---|---|---|---|
| H: screen v1 → v2 | The frozen v1 language model trailed both completed peers at matched steps: held-out NLL 4.6073 at step 1,000 versus 4.3596 (Transformer) and 4.2364 (Mamba-3). It had constant per-head cyclic gates, 16-wide keys, no short convolution, an XLA per-token Cholesky and a 1,024-step sequential protected-QR scan per layer, and took 25.53 s per step (33× Mamba-3). A Cloud TPU defragmentation preemption stopped it after 1,026 steps; it was not resumed. | Restore Definition 5.1's token drift and evidence gates with an undiscounted diagonal floor, valid for arbitrary gates. Add causal width-4 convolutions, exact lane-major Cholesky solves with an analytic reverse pass, and a chunked order scan. Compose selective memory layers with unmodified official Mamba-3 layers. Protected banks leave the language model; the operator, its contracts and tests remain. | Nine Lean theorems in `Selective.lean`, dense-reference forward/gradient/decode tests, TPU kernel timings, five parameter-matched development pilots on unseen training batches under a pre-recorded rule, a dry run of every TPU program, then one full screen-60m-v2 run. |

Development pilots replay the first 1,000 protocol steps. Each loss is taken on
a training batch before that batch updates the model; the held-out split is
never opened. Values are mean training loss over steps 901–1,000 minus the
peer's logged loss on the same batches; ± is a normal 95% interval over the
100 paired steps, which are autocorrelated, so it is optimistic.

| Pilot | Layers (Mamba-3 + memory), key dim | Parameters | vs Mamba-3 | vs Transformer | Median step |
|---|---|---:|---:|---:|---:|
| p1 | 0 + 18, d=18 | 59,955,712 | not run: compile exceeded 16 min (unrolled factor) | — | — |
| p2 | 8 + 11, d=16 | 59,976,928 | +0.0625 ± 0.0011 | −0.0746 ± 0.0015 | 1.615 s |
| p3 | 18 + 2, d=16 | 60,281,600 | −0.0128 ± 0.0010 | −0.1498 ± 0.0015 | 1.004 s |
| p4 | 15 + 4, d=32 | 59,942,304 | −0.0438 ± 0.0019 | −0.1809 ± 0.0018 | 1.571 s |
| p5 | 15 + 4, d=64 (8 heads × 128) | 59,893,216 | −0.0395 ± 0.0017 | −0.1765 ± 0.0017 | 1.451 s |

Many 16-wide memory layers (p2) were worse than the Mamba-3 layers they
replaced; a few wider memory layers were better, and their margin grew with
training (p4 against Mamba-3: −0.008, −0.008, −0.022, −0.044 over successive
windows). The rule in `validation/pilots/SELECTION.md` was recorded before
p4/p5 were observed: p4 had the lowest loss, p5 lay within 0.005 nats and was
faster, so p5 was selected. p4 had used the slower one-column solve schedule,
which partly explains its step time; the frozen run uses the blocked
schedule, which computes the same factor. A p1b pilot (pure memory, 32
narrow heads) was dropped in favor of the capacity tests p4/p5.

Engineering findings are kept as well. A Python-unrolled factor made an
18-layer program compile for more than 16 minutes. Isolated multi-host
microbenchmarks twice stopped with a TPU launch-identity mismatch, although
the hosts' optimized HLO was identical (backend bundle counts 546,264 vs
546,266); barriers between timed cases removed the failures. Training
programs, which run in collective lockstep, were unaffected. Per-block
timings showed the order head's associative scan cost as much as the
conjugate read; the chunked order scan computes the same recurrence.

**Outcome of H.** The frozen screen-60m-v2 run completed 1B targets without
interruption and passed the full audit. Held-out NLL 3.4727 against 3.4758
(Mamba-3) and 3.5352 (Transformer); paired per-sequence intervals exclude zero
on both the original and the fresh holdout. The margin over Mamba-3 narrowed
from −0.050 at step 1,000 to −0.003 at the end, so the next scale must test
whether it persists; synthetic recall rose to 45 of 128 prompts (30 for
Mamba-3). The declared single-seed screen win holds; robustness across seeds
is unmeasured.
