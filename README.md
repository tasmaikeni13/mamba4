# Mamba 4: audited conjugate sequence memory

Phases 01 and 02 are complete: deep mathematical analysis, 61 checked Lean 4
theorems across 15 modules, and reproducible statistical/Monte Carlo operator
comparisons. Several original claims are repaired; attention's wins are
retained. Phases 03–04 now implement the full models and run the authorized
60M, 1B-token, single-seed screen on the existing v4-32 pod. Training completion
and a trained win require actual final checkpoints and evaluation results.

Start with the [results report](analysis/results/REPORT.md),
[corrected paper](mamba4.md), [claim ledger](docs/claim-ledger.md), and
[deep derivations](docs/mathematical-analysis.md). The
[original monograph](docs/mamba4-original.md) is preserved.

Supported results include exact protected interpolation within rank capacity,
signed linear-functional recall and calibration under a specified Gaussian
model. Repairs include cached QR anchors, a guaranteed cyclic anisotropic
prior and protected redundant cascade banks. Their assumptions/resource
changes are explicit. Universal dominance over growing attention caches is
falsified; no trained Mamba-3/Transformer win or TPU speedup is claimed.

```sh
uv sync --frozen
lake exe cache get
lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis scripts
uv run ruff format --check analysis scripts
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
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
all-chip hardware evidence. The 60M runs are in progress. The later
125M/3B-token three-seed study is outside the current authorized scope.
