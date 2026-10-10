# Mamba 4 language-model implementation

The language model in `lm/models/mamba4.py` interleaves selective conjugate
memory layers with unmodified official Mamba-3 SISO layers. The production
kernel is `lm/kernels/mamba4.py`; the independent sequential oracle is
`lm/kernels/mamba4_reference.py`. Operator guarantees keep their stated
hypotheses. Learned encoders, forgetting and the neural read path do not
inherit exact-recall or calibration guarantees; trained behaviour is measured
separately.

## Memory operator

Per memory head, with unit keys and queries,
`S_t = lam_t S_(t-1) + beta_t k_t k_t^T`, `C_t = lam_t C_(t-1) + beta_t v_t k_t^T`
and `A_t = S_t + diag(floor)`. Each token reads `C_t y_t` and the latent
variance `q_t^T y_t`, where `A_t y_t = q_t` is solved exactly. The floor is
never discounted, so `A_t >= min(floor) I` for every gate sequence.
`Selective.lean` proves this and the variance bound
`0 <= q^T A^-1 q <= |q|^2 / min(floor)`. The read exactly minimizes the
discounted weighted ridge objective with per-coordinate penalty `floor_j`
(`selective_solve_minimizes`); it is a posterior mean only under the static
model without discount. `Retention.lean` gives the price of the undiscounted
floor: a single write whose discounted weight is `w` is read back as
`w / (w + f)` of its value under an isotropic floor `f`.

## Gates, encoders and key alignment

- **Decay:** `lam_t = exp(-exp(a_h) softplus(w_dt x_t + b_h))`, with
  Mamba-style initialization (`b_h` a log-uniform timestep in [0.001, 0.1],
  `a_h` in [1, 16]).
- **Evidence precision:** `beta_t = sigmoid(w_beta x_t + c_h)`.
- **Floor:** `floor_min + softplus(raw)` per head and key coordinate,
  initialized to 1 with minimum 0.25.

Values, keys and queries pass through a causal depthwise convolution of width
four and SiLU, then keys and queries are normalized to unit length.

**Key alignment** (`memory_key_shift`). Each value is written under the key
computed at the previous position, while the query uses the current one. A
read therefore returns what followed the current context last time: an
induction lookup made into the memory's default geometry instead of a pattern
the convolution must discover. H3's shift SSM uses the same idea. It adds no
parameters and no measurable compute. The first token writes nothing, and the
cache carries the previous key.

## Read path

The latent variance feeds a learned confidence gate
`sigmoid(1 - s_h log1p(c))`. A skip `D_h v_t` and an order-sensitive gated
vector scan are added. Then come a per-head RMS norm, a SiLU output gate and
the output projection.

## Prefill and decode

Within each 64-token chunk, every token's evidence is one masked,
decay-weighted matrix product over the chunk's key outer products. Chunk
boundaries use the associative affine scan, which `selective_scan_equals_sequential`
proves equal to the sequential recurrence. Per-token precisions are factored
by an exact Cholesky with the batch on the TPU lane axis, followed by two
triangular solves. The reverse pass reuses the factor: for upstream `g` it
computes `z = A^-1 g`, `d q = z` and `d A = -sym(z y^T)`, with no explicit
inverse. Work is `O(T (c d^2 + d^3 + c p + p d))` for chunk length `c`, key
width `d` and value width `p`. Everything runs in float32 at HIGHEST
precision.

Cached decode carries:
- the evidence and cross statistics;
- the convolution history;
- the order state;
- the previous key.

Each token pays an exact `O(d^3 + p d)` refactor and solve. Mamba-3 layers use
the pinned official step recurrence. The cache size does not depend on context
length.

## Composition and parameters

The 60M model (`lm/configs/mamba4-60m.json`) has 19 residual layers in the
pattern `SSSMSSSMSSSMSSSMSSS`:

- **Fifteen Mamba-3 SISO blocks** (1,711,840 parameters each): unmodified
  official blocks, `d_state` 96, sixteen 64-wide heads.
- **Four memory blocks** (2,120,880 parameters each):
  - eight heads of value width 128, key dimension 64;
  - a 512×3,096 input projection and a 1,024×512 output projection;
  - a width-4 convolution over 2,048 channels;
  - an 8×64 floor and per-head gate scalars.

With the 25,731,584 tied embedding parameters, the model has 59,893,216
parameters (34,161,632 non-embedding), within 1% of both peers.

## Conformance

`lm/tests/test_mamba4.py` checks:
- chunked reads against sequential dense solves, for chunk sizes 1–16 with
  padding;
- the floor bound;
- the custom solve derivative, and all-input gradients against the dense
  reference;
- decode and continuation against prefill;
- exact recall as the floor vanishes;
- causality, gradients and cached decode, for pure, hybrid, rotary,
  wide-head and key-shifted models;
- solve-backend parity and the chunked order scan.
