#!/usr/bin/env bash
# Verify committed evidence; add --full to regenerate the complete experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
uv sync --frozen
lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis lm scripts validation
uv run ruff format --check analysis lm scripts validation
uv run pytest -q
uv run pytest -q validation/tests
if [[ "${1:-}" == "--full" ]]; then
    uv run python -m analysis.run
    uv run python scripts/report.py
fi
uv run python scripts/verify_results.py
