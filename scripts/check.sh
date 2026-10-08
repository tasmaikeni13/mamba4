#!/usr/bin/env bash
# Verify committed evidence; add --full to regenerate the complete experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
uv sync --frozen
lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis scripts
uv run ruff format --check analysis scripts
uv run pytest -q
if [[ "${1:-}" == "--full" ]]; then
    uv run python -m analysis.run
    uv run python scripts/report.py
fi
uv run python scripts/verify_results.py
