# Mamba 4 language-model implementation

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
