# Mamba 4: conjugate sequence memory

Mamba 4 is a sequence-memory architecture whose memory layers keep weighted
regression sufficient statistics and read them exactly. At every token,
evidence `S_t = λ_t S_(t−1) + β_t k_t k_tᵀ` and cross statistics `C_t`
are updated by token-dependent gates. A read solves
`(S_t + diag(floor)) y_t = q_t` by Cholesky, returning `C_t y_t` and the
latent variance `q_tᵀ y_t`. The floor is undiscounted, so every precision
stays bounded below for arbitrary gates. Each value is written under the
key of the preceding position, so a read is an induction lookup. The
language model interleaves four such layers with official Mamba-3 SISO
layers.

This repository holds:
- the theory, with 74 Lean 4 theorems in 17 modules and no new axioms;
- statistical studies of the operator;
- JAX models, with FlashMamba Pallas TPU kernels;
- matched training against a Transformer and Mamba-3, with every trained
  result, including the losses.

Start with the [paper](mamba4.md) and the [claim ledger](docs/claim-ledger.md).
The [derivations](docs/mathematical-analysis.md) and the
[original monograph](docs/mamba4-original.md) are kept beside them.

## Results

**60M, 1B FineWeb-Edu tokens, one seed** ([report](lm/results/screen-60m/REPORT.md)):

| Model | Parameters | Held-out NLL | Fresh-holdout NLL | Recall (of 128) |
|---|---:|---:|---:|---:|
| Transformer | 59,985,920 | 3.5352 | 3.4960 | 18 |
| Mamba-3 | 59,968,896 | 3.4758 | 3.4378 | 30 |
| Mamba 4 | 59,893,216 | 3.4719 | 3.4333 | 69 |

Mamba 4's NLL win over Mamba-3 is small (−0.0039 nats; 95% interval −0.0052
to −0.0025 over evaluation sequences). It is one training seed.

**Trained claims at 60M** ([report](lm/results/claims-60m/REPORT.md)): a
protocol frozen before scoring tests the paper's distinctive claims.
- **Exact retrieval inside the window:** refuted against the Transformer,
  supported against Mamba-3.
- **Long context and calibrated confidence:** refuted against both.
- **Context-independent decode:** supported.

The memory saturates in text: about a thousand tokens pass through
64-dimensional keys. Training at 1,024 tokens also teaches nothing beyond
that length.

**The mechanism under direct training**
([synthetic studies](lm/results/synthetic/README.md)): small matched models
trained on recall and retention tasks. The memory layer recalls nearly
perfectly at loads where the Transformer and Mamba-3 fail, and it retains
pairs among distractors far beyond its training length.

**125M, 3B tokens, three seeds** ([protocol](docs/scaling-125m.md)): in
progress.

## FlashMamba kernels

`lm/kernels/flashmamba.py` is a Pallas TPU implementation of the Mamba-3
SISO scan. It processes 128-token chunks with the state kept on chip, with a
forward pass in chunk order and a backward pass in reverse.
`lm/kernels/mamba4_fused.py` performs the memory read's per-token Cholesky
solves on chip, with 128 sequences across the vector lanes, and computes
every gradient as matrix products. Both match the reference implementations
to float32 rounding in values and gradients (`lm/tests`).

## Reproduce

```sh
uv sync --frozen
lake exe cache get && lake build
uv run python scripts/audit_formal.py
uv run ruff check analysis lm scripts validation
uv run ruff format --check analysis lm scripts validation
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m analysis.run
uv run python scripts/report.py
uv run python scripts/verify_results.py
```

Lean, mathlib and Python dependencies are pinned. `analysis/results/` holds
the operator study's raw trial arrays, intervals and proof audit. The
[language-model workflow](lm/README.md) covers data preparation, pod
execution, training, audits and evaluation. The
[phase records](phases/README.md) and [CLAUDE.md](CLAUDE.md) guide
continuation.

Development history, including two earlier memory-layer designs that did not
survive (constant cyclic gates, then token gates without key alignment), is
in the git history (commits `fc618d6` and `5269fdd`).
