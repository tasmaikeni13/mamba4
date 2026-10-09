# Mamba 4 speed review: frozen 60M implementation

Reviewed 8 October 2026. This is a read-only diagnosis of the completed
`bench-mamba4-frozen-v1` benchmark. It does not change a model, kernel,
configuration, run controller, or the ongoing 60M/1B-token screen.

## Verified benchmark outcome

The benchmark's `exit.json` records exit code zero on all four workers.
`lm/results/benchmarks-frozen-v1/mamba4-host-{0,1,2,3}.json` contains the final
blocked device timings. By the time of review, benchmark worker PID 2362936
had terminated and guarded continuation had started `train-mamba4-60m-v1`
(worker zero PID 2369204). The transition was checked from process state and
both launch records; no work was restarted.

The frozen candidate has 59,989,408 parameters, global batch 128, length
1,024, sixteen logical v4 devices across four hosts, and local batch eight
per device. All protected, Gaussian, uncertainty and order branches are
present. Each benchmark performs one first execution and three measured
steady training executions, followed by similarly timed forward/loss work.
The same synthetic token batch is reused, so these are kernel measurements
rather than real-corpus training-throughput or language-model-quality results.

| Measurement | Host zero | Range across four hosts |
|---|---:|---:|
| Train compile, including lowering | 278.423 s | 272.010–289.298 s |
| First train execution | 33.659 s | 26.145–40.077 s |
| Median steady train execution | 25.54264 s | 25.54206–25.54266 s |
| Global train tokens/s | 5,131.498 | 5,131.493–5,131.614 |
| Forward/loss compile | 56.454 s | 55.969–67.128 s |
| Median steady forward/loss | 8.36473 s | 8.36463–8.36476 s |
| Global forward/loss tokens/s | 15,669.608 | 15,669.547–15,669.791 |
| Cached decode tokens/s, after 1,024-token context | 6,792.141 | 6,659.118–6,792.141 |

Host zero's measured steady training samples are
`[25.5428631, 25.5426409, 25.5407951]` seconds. Thus the long gap after
training compilation is substantially actual execution: its four timed
training calls total 110.285 seconds. Compilation and first-use effects
cannot explain the steady 25.54-second step. A projection using the synthetic
rate alone is **54.13 hours for one billion training targets**, before
validation, checkpoints, diagnostics and startup. This is not a completion
ETA for the real-corpus run.

Synthetic optimizer updates remain finite: final loss 10.738824 and
pre-clipping gradient norm 12.838545. The separately executed memory
inspection passed on every host. Its worst Gaussian precision condition is
4.249490, minimum precision eigenvalue 2.620284, and minimum retained QR
factor diagonal 0.431766. These healthy synthetic factors make a numerical
failure an unsupported explanation for this particular measured slowdown.
They do not establish future real-corpus stability or model-relative
calibration under learned encoders.

For context, archived full-size peer benchmarks at the same B128/T1024 report
172,012 train tokens/s for Mamba-3 and 1,339,916 for Transformer, compared with
5,131 for Mamba 4. The corresponding observed training gaps are about 33.5×
and 261×. Those peer files use an older benchmark-result schema without the
new complete source-provenance field, so this comparison is contextual;
it is not a new simultaneous, source-matched hardware experiment.

The current source hashes were checked against Mamba 4's final benchmark
provenance and matched. Relevant frozen hashes are:

- `lm/kernels/mamba4.py`: `55bc2833540e0236586f0a4275043fc4dcd18c65ed86732a66ddc75e609ac852`
- `lm/models/mamba4.py`: `66287689ba569d5f9962d55440714f10ed46ab509d964f4b69f993c3ce108b5c`
- `lm/configs/mamba4-60m.json`: `b2b11f630955e7b1ec8c5a4ee7aa75f4cdd6dc5216cf1ad2381f72c76cf93896`

## What the compiler evidence establishes

The train executable reports 7,198,633,984 bytes of temporary storage and
182,951,424 bytes of generated code. Its compiler cost report estimates
3.4504e12 FLOPs and 6.7979e11 bytes accessed. The archived Mamba-3 report has
nearly the same FLOP estimate, but its byte-access estimate is about 3.6×
smaller and its measured step about 33.5× faster. Nominal FLOPs therefore do
not account for the measured gap. These compiler counters are static cost
estimates, not measured memory traffic or achieved arithmetic throughput.
In particular, the negative `optimal_seconds` field is unusable as a roofline
estimate.

The local TPU driver log for benchmark PID 2362936 records:

| Compiled module | Overlays | Program Cmem (driver units) | Program Vmem (driver units) |
|---|---:|---:|---:|
| `pmap_train_step` | 1,404, with HLO functions | 128.00M/128.00M | 15.78M/16.00M |
| `pmap_eval_step` | 352 | 128.00M/128.00M | 15.78M/16.00M |
| `pmap_inspect` | 337 | 128.00M/128.00M | 15.78M/16.00M |
| Cached decode | 76 | 95.46M/128.00M | 11.34M/16.00M |

This supports inspecting nested loops, small numerical operations, layouts
and code/data movement. It is not an execution profile assigning time to a
specific branch. Generated code size alone is also insufficient: the archived
Mamba-3 executable has larger generated code and runs faster.

## Exact-operator bottlenecks visible in the source

1. **Protected selection performs work after the bank is full.**
   `_insert_anchor` computes two QR projection/subtraction passes, residual
   norms and updates for every one of 64 positions in a base block. Its
   acceptance mask checks `count < 4` only after those computations. Once
   four anchors are retained, the bank cannot accept another key, but the
   numerical work is still expressed in the scan. Four small `HIGHEST`
   projection dots execute per candidate in the expressed algorithm.
   Accepted/rejected key patterns are data dependent; one cannot assume that
   the first four positions always suffice.

2. **Protected factors are broadcast over query time.**
   Earlier-bank Q/R/value caches are constant during a base block, yet the
   code expands them over all 64 query positions and calls a batched
   one-column triangular solve for each query. Its geometry scores separately
   express another Q projection. XLA may remove some broadcast or common
   work; no profile currently proves that it does. Grouping queries as matrix
   right-hand sides is an exact alternative with more useful shapes.

3. **All allocated earlier-bank slots are read before occupancy masking.**
   Length 1,024 with base span 64 allocates five levels and two slots at each
   level. The code evaluates ten old-bank reads per query, then masks slots
   using occupied counts and eligible-anchor flags. The actual occupied-bank
   counts before the sixteen blocks are
   `[0,1,2,2,3,3,4,3,4,4,5,4,5,5,6,4]`, averaging 3.4375. The current partial
   bank is additional. Omitting *unoccupied* slots can save expressed work
   while continuing to read every occupied bank.

4. **Gaussian factors and their automatic derivatives are per token.**
   The default Gaussian path materializes local evidence prefixes and
   Cholesky-factors each 16×16 precision, then solves two triangular systems.
   Its forward factor work is O(T d³), despite bounded chunk activation memory.
   Automatic differentiation also passes through these factors/solves.
   The existing quadratic cached cyclic update does not make this prefill
   algorithm quadratic.

5. **Nested rematerialization multiplies expressed numerical work.**
   The model rematerializes whole layers; both Gaussian and protected outer
   chunk scans also rematerialize their bodies. This bounds stored activation
   memory in the verified production preflight, but recomputation
   includes small factors, QR selection, routing and solves. Its actual cost
   must be profiled before changing checkpoint placement. Larger Gaussian
   chunk sizes could reduce outer-loop iterations without changing the
   operator, at a higher live-memory cost.

The order-sensitive affine vector scan and dense projections remain necessary
parts of the architecture. Neither is isolated by the current timing evidence
as the dominant bottleneck.

## Proposals that preserve the operator and feature contracts

These are proposals only. None was applied to frozen sources or the live run.
No speedup factor is claimed before a measured full-size experiment.

**First, short-circuit saturated selection and group constant-bank reads.**
A scalar `all(count == budget)` guard can return the current bank directly
when every batch/head bank is full. For partially full batches, continue the
existing per-head eligibility mask. This preserves chronological selection,
all retained IDs, accepted QR gradients, the safe rejected-residual norm,
and zero updates for keys outside a full bank. It also preserves gradients
through the Gaussian and order branches, which still consume those tokens.

For an old bank, compute `U = Q^T Queries` with all block queries as columns;
solve `R Coefficients = U` once with 64 right-hand-side columns; compute
`Values Coefficients`; and reuse U for span-energy geometry. This keeps the
same all-bank softmax weights and signed protected read. A must-pass check is
forward and every key/value/query/routing gradient against the frozen path,
including empty banks, repeated keys and cluster/rank-cut cases.

**Second, avoid reads of unoccupied slots and preserve chronological order
without unnecessary sorting.** Counts depend on block boundaries, not rank
eligibility, and are shared across sequences/heads. A packed occupied list or
carefully guarded per-bank reads can omit empty slots while retaining all
occupied reads, IDs and merge costs. For merge candidates, the redundant
counter invariant orders the entire older bank before the newer bank; invalid
padding can be skipped. If this invariant is established for every merge,
concatenating their chronological retained candidates can avoid an argsort.
This requires independent retained-ID, mass/count and conditioning tests.

**Third, vectorize the chronological selector in at most a selection rounds.**
For each next anchor, project every remaining candidate onto the already
selected basis in parallel, perform the same two reorthogonalization passes,
and choose the earliest eligible candidate after the previously selected ID.
Repeat at most `a=4` times. Construct each token's causal prefix by masking
selected columns whose IDs are later than that token; old basis columns do
not depend on future accepted columns. Selection continues past dependent early keys to find later eligible keys. Piecewise
selection gradients and finite arithmetic near the rank threshold need direct
conformance, including adverse EOS/cluster sequences. Prefix-factor grouping
may then permit matrix-right-hand-side reads for equal prefix ranks.

**Fourth, use an analytic Gaussian mean-and-variance read VJP.**
Let `y=A^-1 q`, `z=A^-1 C^T g`, and let α be the upstream scalar for the
variance `q^T A^-1 q`. The same forward mean/variance have derivatives

- `dC = g y^T`;
- `dq = z + 2 α y`;
- `dA = -sym(z y^T) - α y y^T`.

Reuse the Cholesky factor and compute z by two triangular solves. This avoids
backpropagating through the Cholesky algorithm while retaining its exact
symmetric-SPD derivative. Gradients through chronological evidence, cyclic
prior initialization/injections, learned epsilon/decay/precision and fixed
variable-gate fallback still flow through A and C. The variance term must be
included because the production confidence gate consumes it. This proposal
uses solves and retains the full precision and covariance read.

**Finally, examine packed float32 layouts and checkpoint placement.**
A fused vector-oriented layout for the tiny QR and Gaussian operations can
retain float32 arithmetic while avoiding poorly shaped repeated dots. The
compatible cyclic boundary-factor/rank-update realization is also worth an
isolated comparison, but its extra sequential factor updates could run slower
than batched factors on this hardware; its quadratic work bound is not a
speed measurement. Any implementation must keep the constant cyclic prior,
fixed-floor fallback, all uncertainty reads, protected eligibility/reselection,
all occupied-bank routing and order state.

## Evidence needed before integrating any proposal

Use independent dense/QR forward and gradient conformance, causality, cache
agreement across base-block and cascade-carry boundaries, duplicate/cluster
rank tests, learned epsilon/gate checks, and spectral/QR diagnostics. Then
measure the complete B128/T1024 model on all sixteen devices, with first-use
and compile time separate from blocked steady optimizer execution. Capture
an execution profile if component attribution remains uncertain. Record a
new source freeze and preserve this version's benchmark and training history.
The scope remains the requested matched 60M/1B-token one-seed runs.
