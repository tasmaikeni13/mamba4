# Phase 02: statistical and Monte Carlo analysis

Prerequisite: phase 01's corrected contracts. Operator behavior is studied
before expensive language-model training; the model/kernels phase follows.

Deliverables:

- Independent NumPy/SciPy reference for weighted evidence, QR interpolation,
  attention, Hebbian/diagonal/delta memories, and actual Mamba-3
  exponential-trapezoidal complex SISO/MIMO primitives.
- Nonzero-angle, variable-gate, previous-input and rank>1 conformance checks;
  all-input read/write gradient finite differences; every-prefix scan checks.
- Frozen development/holdout seed protocol, raw independent trial arrays,
  paired comparisons, confidence intervals, Holm/Bonferroni correction and
  explicit comparator state/parameter/cost limitations.
- Capacity/geometry/load sweeps, signed queries, noisy repeated writes,
  heteroscedastic and discounted risk, in-model/out-of-model calibration,
  graph hops, floating-point conditioning, stable floor and cascade repairs,
  causal drift/window/forgetting comparisons and dimensional bound sweeps.
- Standalone plots, a generated results report, machine evidence verifier,
  repair log, limitations and next-phase specifications.

Commands:

```sh
uv sync --frozen
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analysis.run
uv run python scripts/report.py
uv run python scripts/verify_results.py
```

Quick runs go to `analysis/runs/` and never satisfy full-evidence acceptance.
Research completion means the full investigation and repair evaluations are
verified. A claim of dominance requires evidence in its declared domain;
universal superiority over a growing attention cache is falsified, so it is
not a promotion criterion we can mark as passed. No trained-LM or TPU speed
result is claimed. Read `analysis/results/REPORT.md` before phase 03.
