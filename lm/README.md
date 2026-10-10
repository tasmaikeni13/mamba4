# Matched 60M language-model screen

The screen compares a Transformer, official Mamba-3 SISO and Mamba 4 on the
existing four-host v4-32 pod. Each model has about 60M parameters, a tied GPT-2
embedding and exactly 1,000,000,000 training targets with seed 42.
[The frozen protocol](configs/screen-protocol.json) declares the metrics,
matching, data provenance and win criterion before training. There is no 125M
run in the current scope.

| Model | Total parameters | Non-embedding parameters |
|---|---:|---:|
| Transformer, 11 layers | 59,985,920 | 34,254,336 |
| Mamba-3 SISO, 20 layers, 96-dimensional state | 59,968,896 | 34,237,312 |
| Mamba 4: 15 Mamba-3 + 4 memory layers, 64-dimensional keys | 59,893,216 | 34,161,632 |

Shared settings:
- width 512, vocabulary 50,257, sequence length 1,024, global batch 128;
- AdamW with a 6e-4 peak learning rate, 200 warm-up steps and cosine decay;
- no dropout, float32 parameters and bf16 dense projections;
- 7,630 updates per run. The last update has 51,712 valid targets; padded
  targets get zero loss weight.

Distributed gradients divide the global loss sum by the global count of valid
targets, including partly empty devices.

The [data audit](../docs/fineweb-data.md) excludes held-out documents by
whole-document content hash and keeps 2,097,152 held-out targets. GPT-2
tokenization leaves 998,868,400 training targets after the holdout, so the
final 1,131,600 targets come from a second deterministic permutation of the
training documents (seed 43). That 0.11316% reuse is identical for all models.

**Implementations.**
- **Transformer:** JAX Pallas fused causal FlashAttention with its fused
  backward.
- **Mamba-3:** a chunked SSD program with complex phases, trapezoidal
  previous-input terms, learned B/C normalization and bias, a skip and output
  gating.
- **Mamba 4** ([configuration](configs/mamba4-60m.json)): interleaves
  token-gated fixed-floor conjugate memory, with exact lane-major Cholesky
  solves, keys aligned to the preceding context, causal short convolutions and
  a chunked order scan, with unmodified official Mamba-3 blocks. See
  `docs/mamba4-implementation.md`.

The ablation without key alignment is
`configs/ablations/mamba4-60m-no-key-shift.json`.

```sh
uv sync --frozen
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
uv run python scripts/prepare_fineweb.py --allow-train-replay
uv run python -m scripts.pod sync --data
uv run python -m scripts.pod run --tag train-transformer lm.train \
  --config lm/configs/transformer-60m.json --output lm/runs/screen-60m/transformer
uv run python -m scripts.pod run --tag train-mamba3 lm.train \
  --config lm/configs/mamba3-60m.json --output lm/runs/screen-60m/mamba3
uv run python -m validation.run_screen      # Mamba 4: train, audit, evaluate
uv run python -m validation.verify_screen   # re-audit all three, report
uv run python -m validation.claims launch   # claims suite and decode timing
uv run python -m validation.claims report
```

The run controller holds an exclusive pod lock, deploys no new cloud
resources and copies every worker's result back. Development pilots
(`validation.pilot`) never open the held-out split. JAX process 0 can differ
from physical host 0; the recorded topology is authoritative.

Every worker verifies corpus hashes and matching checkpoint, source and
protocol identity before resuming. Checkpoints keep the optimizer state and
exact target counts, and payload checksums with atomic pointers detect
corruption. Keep sources and the protocol fixed during a live run. A run
counts as complete only with:
- the final checkpoint;
- exact target accounting;
- the held-out evaluation;
- finite training;
- results from every host.

The recorded runs were produced and audited at commit `5234621`. The audits
compare execution sources byte for byte, so re-run them from that commit.
