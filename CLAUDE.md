# Mamba 4 research

The current task covers phases 01–02. Later TPU training/publication phases are
plans. Read `phases/README.md`, `phases/status.json`, the claim ledger and the
results report when resuming; inspect the real worktree and remote state.

## Commands

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
- Do not launch later training merely because investigation artifacts pass.

## Working notes

Keep module conventions: observation rows; S is key-by-key; C is value-by-key.
Use Cholesky/QR solves, not explicit numeric inverses. The rotating floor
reference assumes constant decay; variable gate semantics require new proofs.
Keep research details in linked phase/docs files, rather than expanding this
instruction file. When compacting, preserve current phase, changed contracts,
verified results, live process handles and remaining acceptance gates.

This concise, command-oriented file follows Anthropic's
[CLAUDE.md guidance](https://code.claude.com/docs/en/best-practices#write-an-effective-claudemd)
and [project-memory guidance](https://code.claude.com/docs/en/memory#write-effective-instructions),
consulted 2026-10-08. These are workflow references, not scientific evidence.
