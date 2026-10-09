# Mamba 4 language-model implementation

The sections below the v2 heading describe the selected screen-60m-v2 model;
the earlier sections describe the frozen v1 composition, retained for
diagnosis and the interrupted v1 run.

The language model in `lm/models/mamba4.py` composes a learned Gaussian head,
protected redundant QR banks, and an order-sensitive vector head. These are
separate states with separate guarantees. Their output mixture is learned;
operator exactness does not establish language-model accuracy or calibration.
The production kernel is `lm/kernels/mamba4.py`; the independent conformance
oracle is `lm/kernels/mamba4_reference.py`.

## Learned Gaussian head

Learned key and query projections are normalized to unit length. Value and
output-gate projections are unrestricted. Token precision is
`2 sigmoid(logit) + 1e-4`. Per-head epsilon is learned through
`softplus(raw_epsilon) + 1e-4`. The cyclic mode learns one constant gate per
head in `(0.9, 0.9999)` and differentiates both the initialized anisotropic
prior and every scheduled injection. It implements exactly the cyclic floor
in paper §6, rather than dropping the fixed-prior injection.

The alternative `memory_floor="fixed"` learns token gates and retains
`epsilon I` with the full-rank `(1-lambda_t) epsilon I` injection. Arbitrary
variable gates are rejected by the cyclic kernel; the fixed mode is the
explicit dense-refactor fallback. It retains different prior semantics.

The prefill implementation builds chronological affine prefix summaries
inside chunks, carries boundary statistics through a rematerialized chunk
scan, then batches **a Cholesky
factorization and two triangular solves for every token**. Its factor work
is `O(T d^3)`, plus `O(T(d^2+p d))` evidence work. Chunking bounds the live prefix tensors and stores boundary statistics
for backward recomputation, but does not replace those factors with rank
updates. The diagnostic `gaussian_memory_prefix` path retains the earlier
whole-sequence two-level affine prefix realization. The ideal compatible-update training bound in paper §6.2 therefore
is not claimed for this first production prefill path. Factors, statistics,
and solves use float32, including when projections use bfloat16.

Cached cyclic Gaussian decode uses a scaled Cholesky factor and two positive
rank-one updates. Its Gaussian work is `O(d^2+p d)`. Fixed-floor cached decode
refactors the full precision and pays `O(d^3+p d)`. Learned epsilon and gate
values are used when initializing a cache from a checkpoint. No explicit
matrix inverse appears in either path.

`q^T A^-1 q` is returned as latent variance and enters a learned confidence
gate. It has the paper's static in-model meaning; the learned token encoders,
forgetting, arbitrary labels and neural output mixture do not supply a new
calibration theorem. Repeated/conflicting labels can still produce small
variance and failed item recall.

## Protected branch and routing

Every base block selects at most `a<=d` chronological independent anchors by
twice-reorthogonalized Gram–Schmidt, caching a padded thin QR factor, values,
valid flags and token IDs. On a third bank at a scale, the two oldest banks
merge, reselect at most `a` independent anchors, and carry upward; the newest
bank remains. Thus every occupied scale holds one or two banks, including
powers of two. Background writes never modify retained QR values or geometry.

Prefill reads the current causal partial block and **all** earlier occupied
banks. Cached decode maintains the same local QR state and performs the same
all-bank reads and boundary reselections. Softmax routing is learned from
query projection onto each bank's span. This geometry router is not proved
to identify the correct bank, and the output mixture does not guarantee exact
answers for all retained items. The separate `protected_route_by_id` API
searches stored IDs, returns a found flag, and satisfies the exact retained
anchor contract within numerical conditioning. Missing/evicted IDs have no
exact status. QR reads carry no posterior variance.

Selection decisions and integer IDs are piecewise constant. Gradients pass
through accepted QR factors and retained values; no gradient through discrete
rank/ID decisions is fabricated. A bank read costs
`O(d a + a^2 + p a)`; multiply by the number of queried banks. Reselection of
at most `2a` candidates costs `O(d a^2+p a)` plus sorting/metadata work. No
constant-cost routing claim is made. A budget of four anchors at width sixteen
provides four retained independent anchors per eligible bank, not sixteen.

The composition keeps one global discounted Gaussian background. It does not
implement the optional per-bank **additive** background merge subsystem from
paper §8.5. All writes remain in the global discounted evidence, subject to
its gates. This distinction affects mergeable background semantics, not the
isolated protected-bank left-inverse contract.

## Order, hops and resource accounting

A separate learned vector affine scan uses token gates and value projections.
It distinguishes order even when additive Gaussian evidence cannot. A learned
mixture weights its output. Optional hops use learned value-to-query bridges
and sequential Gaussian reads. Prefill currently repeats the factor/read
path per hop; cached decode reuses the updated factor. Hops do not imply a
unit Lipschitz bound or general fixed-state reasoning.

The configured 60M candidate (`d_model=512`, `expand=2`, `head_dim=64`,
`key_dim=16`, nine layers, `d_ff=1272`, vocabulary 50,257, four protected
anchors) has **59,989,408 parameters**, including tied embeddings, all head
projections, QR routing mixtures, epsilon/gates, order scan and feed-forward
layers. This count matches the exact model parameter tree; it is not a FLOP
or speed claim.

Cached Gaussian state explicitly stores full precision, full prior, a factor,
and cross statistics: `3 d^2+p d` float32 words per head. Order adds `p` words.
An allocated QR bank stores `(2 d a+p a+a^2)` float32 words, `a` int32 IDs,
and `a` boolean flags, plus cascade counters/steps. For sequence length 1,024
and base span 64 the cache allocates five scales with two bank slots each,
plus a current bank. Occupied slots can be fewer; allocation still counts.
Training additionally materializes **chunk-local** prefix statistics and
factors, partial QR states, all-bank query reads, and chunk-boundary Gaussian
statistics for backward. At local batch eight, length 1,024, sixteen heads,
`d=16,p=64`, a whole-sequence cross tensor would contain 512 MiB; the default
span-64 chunk cross tensor contains 32 MiB, and sixteen boundary states carry
roughly 10 MiB for precision plus cross statistics. These are logical tensor
sizes, not measured peak memory. Compiler memory analysis and hardware
measurements must establish actual costs. Layer/block rematerialization is
enabled in the candidate to limit stored activations.

## Conformance scope

The CPU tests compare multiple chunk sizes, fixed variable gates and constant
cyclic gates against an independent sequential dense solve; compare gradients
for every key/value/query/precision/gate/epsilon input; check floor bounds and
continuation; compare cyclic rank-one decode with the NumPy/SciPy reference;
compare redundant bank IDs, retained exact answers and eviction status with
the phase-02 protected cascade; check background isolation; and verify full
LM causality, nonzero learned-parameter loss gradients, and cached/full
prefill agreement across block boundaries. Hardware conformance, complete
training and empirical comparison remain separate acceptance gates.

The production rank selector compares squared residuals to the tolerance squared
and uses a floored normalization norm for rejected anchors. This leaves
accepted QR columns unchanged and prevents an undefined zero-residual norm
derivative for exact duplicate/EOS keys. Tests exercise duplicates, clustered
keys on both sides of the rank threshold, and repeated-EOS full LM losses.
`gaussian_diagnostics` exposes spectral precision bounds/condition, floor
diagonals, cross magnitude and finiteness; `protected_diagnostics` exposes
active bank/anchor counts, minimum retained QR diagonal, merges and finiteness.
These probes have explicit diagnostic cost and should run at evaluation/check
intervals; a small QR diagonal flags conditioning, not a calibrated route.

Full-model evaluation can capture scalar probes with
`model.apply({"params": params}, tokens, mutable=["diagnostics"])`.
The returned collection is organized as
`diagnostics/layer_i/memory/metric_name`, with one scalar in each Flax sow tuple.
Probe names distinguish `_min`, `_max` and `_allfinite` so a distributed
caller can use global min/max reductions. Probes are skipped during parameter
initialization and ordinary training; they add no work to the default path.
Per-layer anchor bounds count retained anchors per sequence and head. Exact
query routing and output quality still require separate evaluation.

## screen-60m-v2: selective fixed-floor composition

The v1 composition above is frozen and retained for diagnosis
(`memory_mixer="v1"`). Its trained screen was interrupted and trailed both
peers at matched steps (iteration H). The v2 composition
(`memory_mixer="selective"`) restores Definition 5.1 of the original paper:
token-dependent drift gates and evidence precisions with an undiscounted
prior.

**Operator.** Per memory head, with unit keys and queries,
`S_t = lam_t S_(t-1) + beta_t k_t k_t^T`, `C_t = lam_t C_(t-1) + beta_t v_t k_t^T`
and `A_t = S_t + diag(floor)`. Each token reads `C_t y_t` and the latent
variance `q_t^T y_t`, where `A_t y_t = q_t` is solved exactly. Because the
floor is never discounted, `A_t >= min(floor) I` for **every** gate sequence;
`Selective.lean` proves this and the variance bound
`0 <= q^T A^-1 q <= |q|^2 / min(floor)`. The read is the exact minimizer of the
discounted weighted ridge objective with per-coordinate penalty `floor_j`
(`selective_solve_minimizes`). The cyclic constant-gate floor is not used;
its quadratic rank-one decode is replaced by an exact `O(d^3 + p d)` refactor,
which the analysis already documented as the fixed-floor fallback cost.

**Gates and encoders.** `lam_t = exp(-exp(a_h) softplus(w_dt x_t + b_h))`
with Mamba-style initialization (`b_h` log-uniform timestep in
[0.001, 0.1], `a_h` in [1, 16]); `beta_t = sigmoid(w_beta x_t + c_h)`; the
floor is `floor_min + softplus(raw)`, per head and key coordinate
(initialized to 1, minimum 0.25). Values, keys and queries pass through a
causal depthwise convolution of width four and SiLU; keys and queries are
then normalized to unit length. Optional data-dependent rotary phases
(`memory_rope_fraction`) rotate keys and queries by the cumulative
timestep-weighted angle, as Mamba-3 does for B/C; rotations preserve norms.

**Read path.** The latent variance feeds the learned confidence gate
`sigmoid(1 - s_h log1p(c))`. A skip `D_h v_t` and the order-sensitive gated
vector scan are added, then a per-head RMS norm, SiLU output gate and output
projection. Protected redundant QR banks are **not** part of the v2 language
model (see "Protected banks" below).

**Prefill algorithm.** Within a chunk of 64 tokens, evidence for every token
is one masked decay-weighted matrix product over the chunk's key outer
products; chunk boundaries use the associative affine scan, whose equality
with the sequential recurrence is `selective_scan_equals_sequential`. The
per-token precisions are factorized by an exact Cholesky with the batch on the
TPU lane axis (loop or blocked schedule), followed by two triangular solves.
The reverse pass reuses the factor: for upstream `g`, `z = A^-1 g`,
`d q = z` and `d A = -sym(z y^T)`; no inverse is formed and no derivative is
taken through the factorization steps. Work is `O(T (c d^2 + d^3 + c p + p d))`
for chunk length `c`, key width `d` and value width `p`.
All evidence, factors and solves use float32 HIGHEST precision.

**Cached decode.** Evidence, cross statistics, the convolution history, the
order state and (optionally) the rotary phase are carried. Each token pays an
exact `O(d^3 + p d)` refactor and solve; hybrid Mamba-3 layers use the pinned
official step recurrence with the official parameter tree.

**Composition and parameter ledger.** The selected screen-60m-v2 model
(`lm/configs/mamba4-60m-v2.json`) has 19 residual layers in the pattern
`SSSMSSSMSSSMSSSMSSS`: fifteen unmodified official Mamba-3 SISO blocks
(1,711,840 parameters each, `d_state=96`, sixteen 64-wide heads) and four
selective memory blocks (2,120,880 each: eight heads of value width 128, key
dimension 64, input projection 512×3,096, width-4 convolution over 2,048
channels, output projection 1,024×512, per-head floor 8×64 and gate scalars).
With the shared 25,731,584 tied embedding parameters the model has
**59,893,216** parameters (34,161,632 non-embedding), within 1% of both peers.
The v1-only fields `memory_decay`, `memory_epsilon`, `memory_hops` and
`protected_*` are inert in this composition; the frozen file records
`memory_floor="fixed"` to name the semantics.

**Measured block costs.** Per-chip timings of one block on 8 × 1,024 tokens,
run on all 16 v4 chips (`lm/results/bench-blocks-dev/`; median of three
blocked executions after a synchronizing barrier, compile excluded):

| Block | Forward | Forward + backward |
|---|---:|---:|
| Official Mamba-3 | 5.5 ms | 39.0 ms |
| Memory, d=16, sixteen heads, loop factor | 40.8 ms | 73.8 ms |
| same without the order head | 20.1 ms | 32.0 ms |
| same with block rematerialization | 40.9 ms | 113.6 ms |
| Memory, d=32, loop / blocked factor | 91.9 / 61.9 ms | 137.4 / 107.3 ms |
| Memory, d=64, eight 128-wide heads, blocked | 77.9 ms | 106.9 ms |

These timings used the associative order scan; the frozen run uses the
chunked order scan (`order_memory_chunked`), which computes the same
recurrence (test: values within 1e-5, gradients within 1e-4). The full
model's measured step time is recorded by the screen run itself.

**Protected banks.** The v1 language model's protected redundant QR cascade
selected the first independent keys of every 64-token block and processed all
1,024 positions sequentially in every layer. The v1 speed review listed it,
its broadcast all-bank reads and the per-token XLA factors as the visible
bottlenecks of the 25.5-second step; v1 was never profiled, so no exact share
is claimed. Its learned output mix started at sigmoid(-3). The v2 language
model omits the branch: no protected anchors, routing or bank state are
trained or claimed for the v2 screen. The operator
(`protected_cascade_memory`, `protected_route_by_id`), its exact
retained-anchor contract and its CPU/TPU conformance tests remain in the
library. A parallel completed-block variant (score-ordered selection,
cascade schedule fixed by the block count, reads of completed banks only) is
the specified replacement for a later screen, not part of these results.

**Full-model benchmarks.** `lm/results/benchmarks-v2/` holds source-matched
measurements of all three screen models on all 16 chips at batch 128 ×
1,024 (blocked medians after compilation; decode after a 1,024-token
context):

| Model | Train tokens/s | Prefill tokens/s | Cached decode tokens/s | Cache per device |
|---|---:|---:|---:|---:|
| Transformer | 1,335,387 | 5,720,960 | 24,686 | 177.5 MiB (KV, 1,033 tokens) |
| Mamba-3 | 150,568 | 1,277,411 | 26,535 | 61.0 MiB |
| Mamba 4 v2 | 92,624 | 345,127 | 16,014 | 58.3 MiB |

The v1 benchmark of the frozen composition measured 5,131 training, 15,670
prefill and 6,792 decode tokens/s. v2 trains 18× and prefills 22× faster, but
it remains slower than Mamba-3: 0.62× training, 0.27× prefill and 0.60×
decode throughput, with 9.3 GiB of compiler temporary memory against 2.9 GiB.
Its cache is constant-size like Mamba-3's.

