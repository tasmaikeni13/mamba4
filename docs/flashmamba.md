# FlashMamba: TPU kernels for Mamba 4

Mamba 4 has two kinds of sequence mixer: official Mamba-3 SISO layers and
the memory layers. Each needs its own TPU kernels. This note records:
- what the kernels compute;
- how they are verified;
- what the TPU compiler accepts;
- where the training time went.

All numbers are for the 125M models on the 16-chip v4-32 pod, at a global
batch of 256 × 1,024 tokens.

## Kernels

**Mamba-3 scan** (`lm/kernels/flashmamba.py`, `mamba3_fused`). One Pallas
kernel pair evaluates a layer's whole state-space recurrence in the
projection's own layout:
- **Inputs:** the shared B and C rows `[B,T,N]`, the values `[B,T,H·P]` and
  the per-head gates. No per-head copies of B and C, and no transposes.
- **Forward kernel:** walks the 128-token chunks of every (batch, head) in
  order. It adds the head's bias to B and C, accumulates the rotary phase in
  VMEM and rotates pairs with exact lane rolls. It forms the decay matrix
  from lane-dense prefix rows and runs the chunk's four products on the MXU,
  carrying the `[N,P]` state on chip.
- **Backward kernel:** walks the chunks in reverse and carries the state
  cotangent and the phase cotangent. It recomputes the rotation and returns
  the gradients of B, C, the values, biases, angle rates, timesteps and
  decays. The gradients of the shared B and C are summed across heads in
  place.

**Memory read** (`lm/kernels/mamba4_fused.py`). Every token needs an exact
solve `A_t y_t = q_t`, with `A_t` the token's decayed evidence plus the
undiscounted floor.
- **Forward:** 128 independent sequences share the vector lanes. Each token
  updates its evidence and factors `A_t` by a left-looking Cholesky, building
  each column in registers and storing it once. The diagonal holds `1/L_jj`,
  so neither the factorization nor the solves divide.
- **Stored factors:** the forward kernel writes every token's packed factor
  to HBM, about 1.8 GB per layer per chip at 125M.
- **Backward:** `z = A⁻¹g` costs two triangular solves instead of a second
  factorization. Every other gradient is a chunk-level matrix product, and no
  inverse is formed.

## Verification

- **Values and gradients:** every kernel is compared with an independent
  implementation in values and in all gradients (`lm/tests/test_flashmamba.py`,
  `lm/tests/test_mamba4.py`).
  - FlashMamba against the chunked reference `mamba3_chunked`; the reference
    itself is checked against a float64 local-frame recurrence.
  - The memory kernels against dense NumPy solves.
- **CPU:** these tests run in Pallas interpret mode and agree to float32
  rounding.
- **TPU, FlashMamba:** the kernel agrees with the reference to about 1e-6
  relative in outputs and in every gradient.
- **TPU, memory read:** the solves agree with NumPy to 1.5e-5 relative. That
  is ordinary float32 Cholesky error at these condition numbers; the in-kernel
  `rsqrt` is accurate to about one ulp.
- **Model level:** the Mamba-3 mixer and the hybrid model give the same
  initialization and the same gradients under every rematerialization policy,
  and prefill equals token-by-token decode.

## What this TPU compiler accepts

These Mosaic limits in JAX 0.6.2 shaped the designs:
- **Broadcasts:** a `[1,1]` value cannot be broadcast along both axes at
  once, so per-chunk scalars arrive as rows replicated across lanes.
- **Narrow minor axes:** arrays shaped `[T,1]`, or with a singleton
  second-minor axis such as `[...,1,N]`, are padded 128-fold or 8-fold in
  HBM. Per-token scalars therefore travel as lane-dense rows.
- **In-kernel float32 products:** they crash the compiler (`lower_to_llo`
  check failure). A 0/1 matrix times a value split into three bfloat16 parts
  is exact with float32 accumulation and is used instead.
- **Offset rows:** a row sliced at sublane offset 7 cannot be stored into a
  row-shaped carry; the row is selected with a masked sum instead.
- **Loop unrolling:** `fori_loop` lowers only without unrolling or fully
  unrolled.
- **XLA around the kernels:** `jnp.cumsum` lowers on TPU to a full-length
  reduce-window whose transpose is quadratic, so prefix sums are triangular
  matrix products. `jnp.roll` and `jnp.repeat` on the state axis turn into
  1- and 127-lane copies, so pair rotations use exact signed permutations.

## Where the time went

A device profile of one training step (`scripts/profile_step.py`) guided
each change.

| Stage (125M, measured) | Mamba 4 tokens/s |
|---|---:|
| First FlashMamba kernels, XLA rotary frame, stored Cholesky factors | 0.22M |
| No singleton axes; memory blocks not rematerialized | 0.27M |
| Exact permutation rotations, matrix-product prefix sums and chunk carries | 0.39M |
| Rotation and phase fused into the scan kernel; left-looking Cholesky | 0.45M |
| Only three Mamba-3 blocks rematerialized | 0.505M |

For scale: the implementation used in the 60M screen trained the 60M
Mamba 4 at 0.093M tokens/s, against 1.38M for that Transformer.

The Transformer baseline uses JAX's Pallas FlashAttention:
- stock 128-token tiles with rematerialization: 0.82M tokens/s;
- 512-token tiles without rematerialization: 1.55–1.62M tokens/s.

Mamba 4 therefore trains at about a third of the optimized Transformer's
speed. Its 17 Mamba-3 layers each run a scan that costs as much as attention
at 1,024 tokens. Its 4 memory layers each factor 64 × 64 matrices per token
on the vector units, which the MXU cannot help with. The step records are in
`lm/results/speed-125m/`.
