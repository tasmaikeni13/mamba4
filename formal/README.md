# Lean verification

The project pins Lean 4.34.1, mathlib v4.34.1 and the exact transitive revisions
in `lake-manifest.json`. Install elan from the
[Lean project](https://github.com/leanprover/elan) if `lake` is unavailable.
From the repository root:

```sh
lake exe cache get
lake build
uv run python scripts/audit_formal.py
```

The default `Mamba4` target imports every research module. `Verify.lean` prints
the kernel axiom dependencies of every local theorem; the audit regenerates
that file, checks each declaration and rejects dependencies beyond propext,
Classical.choice and Quot.sound. It saves source hashes and the exact output
in `analysis/results/formal-audit.json` and `formal-axioms.txt`. No extra axiom
or admitted proof is used. `.lake/` is a disposable downloaded build/cache
directory and is excluded from Git.

The 16 modules cover general scan/fold algebra, deterministic bit and linear
capacity, invariant-space embeddings, log-kernel conjugacy, PSD/ridge evidence,
matrix read/interpolation identities, finite-sum ridge optimality, an end-to-end
weighted ridge solver, variational confidence, finite-distribution risk,
bounded scalar evidence/decay, cyclic prior bounds, hop envelopes, cascade
count invariants, exact counterexample witnesses, and the selective
fixed-floor memory of the v2 language model (arbitrary-gate floor, variance
bound, anisotropic ridge optimality of the exact solve, scan equivalence).

Formal scope is precise: a theorem's supplied normal/solve/moment hypotheses
are not a proof of a stronger statistical premise. `Regression.lean` does
derive positivity, invertibility, normal equations and optimality for the
actual finite weighted-design solve. The Gaussian integrals, PKD/Fano/complexity
theory, spectral interpolation bound, hardware work estimates and Python
implementation conformance are analytic/literature/numerical evidence, not
silently attributed to Lean. See the complete claim ledger for each mapping.
