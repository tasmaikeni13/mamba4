# Autonomous, self-correcting research workflow

Current authorized execution scope: phases 01–04, through the three 60M,
1B-token, single-seed runs on the existing v4-32 pod. Phases05–07 remain
plans for later work.
See the machine-readable [status](status.json) and checked-in evidence.

The research target is a useful Mamba 4 that preserves efficient scan memory,
strong recall, in-context regression, uncertainty and adaptive retrieval. Seek
to match or exceed peers on declared, resource-accounted tasks. An impossible
universal guarantee cannot become a passing gate through wording, selective
benchmarks or changing the baseline; record the obstruction and develop a
stronger feasible method. This is part of solving the mathematical problem.

## Operating loop

1. Read `prompt.md`, the revised paper, claim ledger, phase prerequisites and
   status before resuming. Inspect the actual source/results/remote state.
2. Translate each theory claim into a precise assumption, output, resource
   budget, falsification test and evidence artifact. Literature-derived claims
   need primary-source links and versions; use web research when necessary.
3. Run an adversarial analytic example first. If it fails, derive the cause.
   Repair the method while preserving the intended capability; retain the old
   witness. For an information-theoretic obstruction, explicitly state the
   feasible resource/error tradeoff. Never assert that an unproved stronger
   theorem has been established.
4. Formalize the corrected algebra in Lean with explicit hypotheses. No
   placeholder proofs, new axioms or conclusion-as-hypothesis shortcuts. Map
   each theorem to its exact paper statement and audit axiom dependencies.
5. Implement an independent numerical reference, test conformance, gradients,
   edge cases and numerical conditioning. A future fast path must match it.
6. Freeze a versioned protocol before evaluating. Use development seeds for
   choices; independent held-out seeds for evaluation; paired data for peers;
   trial-level intervals and multiple-comparison accounting. Keep every result.
7. When a hypothesis fails, record an iteration with cause, replacement,
   cost and outcome. Altering a method requires a new protocol/version and
   fresh holdout. Do not repeatedly inspect and optimize the same holdout.
8. Propagate the correction: update theory, claim ledger, downstream phase
   contracts, implementation specifications, gates and result provenance.
   Mark stale evidence invalid; rerun affected checks rather than silently
   reusing results for a different method. Preserve original artifacts.
9. A phase finishes when every planned investigation has a checked result,
   including a valid falsification. Promotion to training requires its own
   scientific/engineering gate. “Investigation complete” does not mean all
   aspirational performance claims are true.
10. Before committing, run the exact validation commands in `CLAUDE.md`,
    inspect the diff and audit artifacts against the task. Push when authorized
    and verify the remote commit; do not claim a run/proof/benchmark not done.

## Phases and dependency order

| Phase | Purpose | Current state |
|---|---|---|
| [01](01-mathematical-formal.md) | Deep mathematical audit, corrected theory, multi-file Lean proofs | Completed investigation |
| [02](02-statistical-monte-carlo.md) | Reproducible statistical/Monte Carlo comparisons and repair evaluation | Completed investigation; universal superiority falsified |
| [03](03-models-and-kernels.md) | Full JAX models, fused kernels and v4-32 conformance | v1 and v2 implementations conformance-tested; engineering benchmarks partial |
| [04](04-60m-screen.md) | 60M, 1B FineWeb-Edu tokens, one seed, all target TPU chips | Screen-60m-v2 completed and audited; single-seed screen win established; 125M not authorized |
| [05](05-125m-sweep.md) | Budget-limited 125M hyperparameter selection | Planned |
| [06](06-125m-main.md) | 125M, 3B tokens, three seeds, Mamba 4 vs Transformer | Planned; losses are publishable outcomes |
| [07](07-publication.md) | Clean, document, humanize and prepare the paper/repository | Planned |

## Decision invariants

- Exact QR anchors, Bayesian ridge and a cyclic anisotropic prior are different
  contracts. Do not transfer calibration or floor claims between them.
- Protected cascade exactness covers retained anchors, not discarded items.
  Count key geometry, factor caches, routing IDs, background state and FLOPs.
- Operator tests are diagnostic; trained Mamba-3/Transformer comparisons need
  full official architecture features and matched training/data/parameters.
- A softmax or peer win remains in the report. Improving Mamba 4 must improve
  the method or explain a real tradeoff; suppressing a competitor is not a fix.
- Single-seed screening is useful for iteration but does not estimate training
  seed variability. The later three-seed study must retain that uncertainty.
- Mathematical impossibilities are not overcome by more retries. For feasible
  unresolved goals, continue the diagnose/derive/test loop within the active
  authorized phase; log failures and revise dependent work.
