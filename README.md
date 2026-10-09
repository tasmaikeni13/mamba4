# Mamba 4: audited conjugate sequence memory

Phases 01–02 are complete: deep mathematical analysis, 70 checked Lean 4
theorems across 16 modules, and reproducible statistical/Monte Carlo operator
comparisons. Several original claims are repaired; attention's wins are
retained. Phases 03–04 implement full JAX models and run the authorized 60M,
1B-token, single-seed screen against a Transformer and official Mamba-3 on
the v4-32 pod.

The first Mamba 4 language model (constant cyclic gates) trailed both peers at
matched steps, was 33× slower than Mamba-3 and was stopped by TPU maintenance;
that record is kept. The second (screen-60m-v2) restores the original
token-dependent gates with an undiscounted floor and exact lane-major solves.
It interleaves four conjugate memory layers with fifteen official Mamba-3
layers. Its held-out NLL is 3.4727, against 3.4758 for Mamba-3 and 3.5352 for
the Transformer: a narrow single-seed win that a fresh holdout confirms, with
markedly better synthetic recall (45 of 128 prompts, against 30 and 18). See
the [screen report](lm/results/screen-60m-v2/REPORT.md).

Start with the [results report](analysis/results/REPORT.md),
[corrected paper](mamba4.md), [claim ledger](docs/claim-ledger.md), and
[deep derivations](docs/mathematical-analysis.md). The
[original monograph](docs/mamba4-original.md) is preserved.

Supported results include exact protected interpolation within rank capacity,
signed linear-functional recall and calibration under a specified Gaussian
model. Repairs include cached QR anchors, a guaranteed cyclic anisotropic
prior and protected redundant cascade banks. Their assumptions/resource
changes are explicit. Universal dominance over growing attention caches is
falsified. The [screen report](lm/results/screen-60m-v2/REPORT.md) contains
the only trained comparison; one seed cannot establish robustness.

```sh
uv sync --frozen
lake exe cache get
lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis lm scripts validation
uv run ruff format --check analysis lm scripts validation
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analysis.run
uv run python scripts/report.py
uv run python scripts/verify_results.py
```

Lean/mathlib and Python dependencies are pinned. `analysis/results/` contains
raw trial arrays, intervals, figures, source hashes, proof audit and scientific
gates. This CPU reference is analysis code, not the later TPU implementation.
A smoke run uses `python -m analysis.run --quick --output analysis/runs/quick`.
`bash scripts/check.sh` repeats all local checks; `--full` also regenerates the
full numerical study and report.

The [seven-phase workflow](phases/README.md), [status](phases/status.json),
[repair log](docs/iterations.md), and [CLAUDE.md](CLAUDE.md) guide continuation.
The [language-model workflow](lm/README.md) records full JAX models, fused
kernels, exact data accounting, the matched parameter ledger and measured
all-chip hardware evidence. The later 125M/3B-token three-seed study is
outside the current authorized scope.
