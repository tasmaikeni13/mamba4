# Phase 01: mathematical and formal analysis

Scope: every named result and major unnumbered claim of the original paper,
including capacity, scan algebra, conjugacy, ridge estimation, interpolation,
confidence, risk, multi-hop reads, numerical floors, work and cascades.

Deliverables:

- Preserve the original; revise the active theory paper when proofs/witnesses
  change a statement. Maintain a complete claim-by-claim ledger.
- Derive the weighted objective/posterior, interpolation and risk identities,
  assumptions for calibration/BLUE, byte/work budgets and counterexamples.
- Build a real multi-file Lean 4 project with pinned Lean/mathlib versions,
  checked finite-dimensional matrix/linear-map proofs, capacity, scan,
  uncertainty monotonicity, cyclic floor and cascade count invariants.
- Record precisely which results are analytic, imported literature, numerical
  tests, or Lean theorems. Inspect every local theorem's kernel dependencies.
- Repair errors with explicit resource changes and downstream consequences.
  Preserve the fixed-state capacity constraint; retain multiple estimators
  when one cannot simultaneously offer noiseless exactness and calibrated risk.

Evidence: `docs/mathematical-analysis.md`, `docs/claim-ledger.md`, `mamba4.md`,
`formal/Mamba4/`, `formal/Verify.lean`, `analysis/results/formal-audit.json`.

Acceptance: complete ledger, corrected paper, clean `lake build`, no forbidden
axiom dependencies, and concrete repairs/counterexamples. Generic complexity
theory/PKD/Gaussian integrals remain explicitly literature/analytic rather
than falsely claimed fully formalized results.
