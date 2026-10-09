# Matched 60M language-model screen

The authorized experiment compares Transformer, official Mamba-3 SISO and
Mamba4 on the existing four-host v4-32 pod. Each model has approximately 60M
parameters, a tied GPT2 embedding, and exactly 1,000,000,000 training targets
with seed 42. [The frozen protocol](configs/screen-protocol.json) declares
metrics, matching, data provenance, reuse and the win criterion before training.
There is no 125M run in the current scope.

| Model | Total parameters | Nonembedding parameters |
|---|---:|---:|
| Transformer, 11 layers |59,985,920|34,254,336|
| Mamba-3 SISO, 20 layers, 96-dimensional state |59,968,896|34,237,312|
| Mamba4, 9 layers, 16-dimensional keys |59,989,408|34,257,824|
| Mamba 4 v2: 15 Mamba-3 + 4 selective memory layers, 64-dimensional keys |59,893,216|34,161,632|

All use width 512, vocabulary 50,257, sequence length 1024, global batch 128,
AdamW with 6e-4 peak learning rate, 200 warmup steps and cosine decay, no
dropout, float32 parameters and bf16 dense projections. Each run takes 7,630
updates. The last update includes 51,712 valid targets; padded targets receive
zero loss weight. Distributed gradients divide the global loss sum by the
global valid-target count, including partially empty devices.

The [data audit](../docs/fineweb-data.md) excludes held-out whole-document
content hashes. It retains 2,097,152 held-out targets. GPT2 tokenization leaves
998,868,400 training targets after this holdout, so the final 1,131,600 targets
use a second deterministic permutation of training documents, seed 43. The
manifest and independent full-stream reconstruction record this 0.11316%
reuse. It is identical for all models.

Transformer uses JAX Pallas fused causal FlashAttention with its fused
backward. Mamba-3 uses a chunked SSD XLA program with complex phases,
trapezoidal previous-input terms, learned B/C normalization and bias, skip
and output gating. Mamba4 uses learned conjugate Gaussian evidence, cyclic
prior, protected redundant QR banks, all-bank routing, confidence and an
order-sensitive auxiliary scan (the frozen v1 row, interrupted after 1,026
steps). Mamba 4 v2 (`configs/mamba4-60m-v2.json`, protocol
[screen-60m-v2](configs/screen-protocol-v2.json)) uses token-gated
fixed-floor Gaussian memory with exact lane-major Cholesky solves, causal
short convolutions and the chunked order scan, interleaved with unmodified
official Mamba-3 blocks; see `docs/mamba4-implementation.md`.

```sh
uv sync --frozen
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
uv run ruff check analysis lm scripts
uv run ruff format --check analysis lm scripts
uv run python scripts/prepare_fineweb.py --allow-train-replay
uv run python -m scripts.pod sync --data
uv run python -m scripts.pod run --tag hardware-probe scripts.probe_tpu
uv run python -m scripts.pod run --tag bench-transformer scripts.lm_benchmark \
  --config lm/configs/transformer-60m.json --steps 3 --decode
uv run python -m scripts.pod run --tag train-transformer lm.train \
  --config lm/configs/transformer-60m.json --output lm/runs/screen-60m-v1/transformer
```

Run the benchmark/training command sequentially for `mamba3` and `mamba4`
with their checked-in configurations. The v2 Mamba 4 run, its audit and the
per-sequence evaluations use `uv run python -m validation.run_screen_v2`
followed by `uv run python -m validation.verify_screen_v2`; development pilots
use `validation.pilot` and never open the held-out split. The controller holds an exclusive pod
lock, deploys no new cloud resources, and copies every worker's result back.
JAX process 0 can differ from physical host 0; the recorded topology is
authoritative. No CPU reference path is used for TPU training.

Every worker verifies corpus hashes and matching checkpoint/source/protocol
identity before resuming. Checkpoints preserve optimizer state and exact target
counts; payload checksums and durable atomic pointers detect corruption. Keep
source and protocol fixed during a live run. Changed models require a new
version and retained failed-run evidence. A completed run requires final
checkpoint, exact target accounting, held-out evaluation, finite training and
results from every host; job launch alone is not completion.

CPU operator checks and a successful all-chip collective are evidence of
implementation and hardware access. They do not establish a trained Mamba4
win. See `phases/status.json` and actual run results for current completion.
