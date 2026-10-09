# Mamba 4 research

The current task covers phases 01–04 through the 60M, 1B-token, single-seed
screen on the existing 16-chip v4-32 pod: completed Transformer/Mamba-3 runs,
the interrupted Mamba 4 v1 run and the versioned screen-60m-v2 Mamba 4 run.
Later scaling/publication phases are plans.
Read `phases/README.md`, `phases/status.json`, the claim ledger and the
results report when resuming; inspect the real worktree and remote state.

## Commands

```sh
uv sync --frozen
lake exe cache get
lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis lm scripts validation
uv run ruff format --check analysis lm scripts validation
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
JAX_PLATFORMS=cpu uv run pytest -q validation/tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analysis.run
uv run python scripts/report.py
uv run python scripts/verify_results.py
```

Use `python -m analysis.run --quick --output analysis/runs/quick` for a smoke
run. It cannot certify a full phase. Lean/mathlib and Python packages are pinned.

## Scientific constraints

- No placeholder Lean proofs, new axioms, invented benchmark/proof results or
  suppressed peer wins. Audit local theorem axiom dependencies.
- The original paper is preserved in `docs/mamba4-original.md`. Update the active
  paper, claim ledger and dependent phase contracts when a claim changes.
- Gaussian posterior calibration needs the specified model. Exact QR anchors,
  finite ridge and cyclic anisotropic priors have different guarantees.
- Exact cascade recall concerns retained independent anchors. Count discarded
  items, routing IDs, cached factors and background/temporary memory.
- Mamba-3 primitives here are untrained operator baselines. Full language-model
  and TPU performance claims require the later matched training runs.
- Select settings using development seeds. Keep the frozen protocol and every
  holdout result; changed hypotheses/methods require versioned fresh holdouts.
- Save raw trial arrays and provenance; statistical sampling units are whole
  trials/trajectories, not dependent queries/tokens within them.
- Phases 03–04 are now authorized by the user; stop before 125M scaling.
- CPU tests use JAX_PLATFORMS=cpu. Only the pod controller launches TPU jobs.
- Run `uv run python -m scripts.pod sync --data` before pod execution; never
  redeploy changed source during a live run. Inspect live processes before resume.
- While a pod job runs, do not edit `lm/`, `scripts/`, configs or lockfiles:
  host 0 imports the local tree, and a resume refuses changed fingerprints.
- JAX process 0 writes the persistent compile cache; clear `.jax_cache` on all
  hosts before launches (as `validation/run_screen_v2.py` does).
- Architecture choices come from `validation.pilot` runs on unseen training
  batches under a rule recorded in advance; never select on held-out data.

## Working notes

Keep module conventions: observation rows; S is key-by-key; C is value-by-key.
Use Cholesky/QR solves, not explicit numeric inverses. The rotating floor
reference assumes constant decay; the v2 selective memory instead keeps the
undiscounted fixed floor, proven for arbitrary gates in `Selective.lean`.
Keep research details in linked phase/docs files, rather than expanding this
instruction file. When compacting, preserve current phase, changed contracts,
verified results, live process handles and remaining acceptance gates.

This concise, command-oriented file follows Anthropic's
[CLAUDE.md guidance](https://code.claude.com/docs/en/best-practices#write-an-effective-claudemd)
and [project-memory guidance](https://code.claude.com/docs/en/memory#write-effective-instructions),
consulted 2026-10-08. These are workflow references, not scientific evidence.
