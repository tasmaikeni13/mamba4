# Mamba 4: conjugate sequence memory with audited recall and uncertainty

*Revised paper, 9 October 2026. Phases 01–04: theory, formal and statistical audit, and the 60M single-seed screen.*

## Abstract

Mamba 4 is a sequence-memory proposal that stores weighted regression
sufficient statistics and reads an estimate of a latent operator. Its
evidence updates admit an associative affine scan. Under a static Gaussian
linear model, the ridge read is a posterior mean with an explicit predictive
variance. Within independent-key capacity, an unregularized left-inverse read
exactly recovers associations and their signed linear combinations. Positive
ridge introduces bias, and fixed state cannot retain unbounded arbitrary
associations.

We audit the original monograph, formalize 70 algebraic and finite-dimensional
results across 16 Lean 4 modules, and run a reproducible NumPy/SciPy study
against softmax, Mamba-3 recurrence primitives and additional memories. The
study supports signed-query retrieval and in-model calibration, and exposes
failures under overloaded or ill-conditioned keys, model mismatch and cascade
merges. Protected QR anchors, redundant cascade banks and an initialized
cyclic anisotropic prior provide explicit repairs with different resource
and statistical contracts.

A 60M-parameter, 1B-token, single-seed language-model screen against a
Transformer and official Mamba-3 is then reported in full (§12). The first
realization, with constant cyclic gates, trailed both peers and was 33 times
slower than Mamba-3; the interrupted run is retained. Restoring the original
token-dependent gates with an undiscounted floor, exact lane-major solves and
a composition with Mamba-3 layers gives the second realization. On the frozen
held-out split it reaches NLL 3.4727, against 3.4758 for Mamba-3 and 3.5352
for the Transformer, a narrow but paired-significant win over evaluation
sequences that a fresh holdout confirms; it also answers markedly more
synthetic recall prompts. No universal peer-dominance claim is made, and one
training seed cannot establish robustness.

## 1. Hypothesis and existing work

A recurrent state can retain a useful inference problem rather than arbitrary
history. If future queries ask about a latent linear map, its regression
statistics are a natural state. This is a modeling choice, not a unique
universal solution. The regression-memory viewpoint has precedents in
[test-time regression](https://arxiv.org/html/2501.12352v3), retention choices in
[Miras](https://arxiv.org/html/2504.13173v1), and curvature-aware memories in
[Preconditioned DeltaNet](https://arxiv.org/html/2604.21100v1). The closest
trained precedent is the [MesaNet](https://arxiv.org/abs/2506.05233) Mesa
layer: gated evidence $H_t+\Lambda$ with a fixed regularizer and a per-token
optimal ridge read, solved by up to 30 conjugate-gradient steps at key width
128. At 145M parameters it was slightly behind a Transformer; at 400M–1B it
matched or slightly beat one, and its hybrid with a recurrent layer was the
strongest recurrent model reported there. [Mamba-3](https://arxiv.org/abs/2603.15569)
is the selective state-space peer used here.

The candidate contribution is the audited combination of scan evidence,
conditional uncertainty, protected exact anchors and explicit floors (the
cyclic constant-gate prior and the undiscounted arbitrary-gate floor), and,
for the trained screen, exact factorized reads with a Bayesian variance gate,
composed with Mamba-3 layers. Publication novelty against MesaNet-style layers
rests on those audited contracts and measured comparisons, not on the read
itself.

## 2. Capacity and useful scan realizations

**Finite-bit recall (original 3.1).** A deterministic m-bit state reconstructing
all Kb-bit assignments must satisfy m>=Kb: encoding is injective and
2^(Kb)<=2^m. Lean checks this counting theorem. Randomized entropy extensions
need explicit error/randomness assumptions and are not Lean-formalized here.
Real-coordinate counts do not imply bit capacity without finite precision.

**General recurrence hardness (3.2).** Growing-state recurrences can simulate
circuit evaluation. This generic worst-case obstruction does not forbid
special nonlinear scans or require every efficient monoid to have a small
linear realization. Representation and combine costs matter. This is a
written complexity argument, not a Lean result.

**Value-linear converse (3.3).** For fixed keys, exact retrieval factoring the
identity on K*d_v real value coordinates linearly through p_v state coordinates
requires K*d_v<=p_v. Lean proves injectivity and the finrank inequality. Extra
stores and nonlinear encoders change the hypothesis.

**Invariant features (3.5).** A finite basis of a precomposition-invariant
function space yields a linear feature recurrence; state separation makes
the feature map injective on that domain. Lean constructs these matrices.
No practically small embedding or unconditional converse is asserted.

## 3. Conjugacy, scan and prior semantics

An exponential-family conjugate log kernel is
$\langle\chi,\eta\rangle-\nu A(\eta)$. Power-weighted updates give
$$
\chi_t=\lambda_t\chi_{t-1}+\beta_tT(o_t),\qquad
\nu_t=\lambda_t\nu_{t-1}+\beta_t.
$$
An affine summary $(\Lambda,X)$ acts as $z\mapsto\Lambda z+X$. Later after
earlier is
$$
(\Lambda_2,X_2)\circ(\Lambda_1,X_1)
=(\Lambda_2\Lambda_1,X_2+\Lambda_2X_1).
$$
The identity is (1,0); combine is associative. Lean proves composition and
sequential-fold equality (4.1). A work-efficient tree scan uses O(Np)
coordinate work for p-coordinate summaries and O(log N) combine dependency
depth. Discounted summaries preserve chronology and generally do not commute.

The proposed layer (4.2) learns token/query maps and mixes head/hop reads.
Protected anchors and order-sensitive auxiliary scans are separate modules;
their selection, state, addressing and gradients count toward the model.
Their learnability is not established by an encoded-operator test.

Likelihood factorization supplies model-relative sufficiency (4.3).
Minimality requires regularity and removal of redundant/ancillary components.
Pitman–Koopman–Darmois (4.4) concerns regular IID/common-support models and
regular statistics; it does not force a conjugate prior or characterize
arbitrary history encodings. See the
[primary lecture notes](https://www.stat.umn.edu/geyer/8054/notes/expfam.html).
Neither sufficiency theorem is claimed as fully Lean-formalized.

Zero-initialized undiscounted evidence adds (4.5). Initialized posterior
parameters merge by adding and subtracting one shared prior. Other additive
memories can merge too. Evidence precision is a likelihood parameter when
the noise model says so (4.6). Forgetting is a generalized-posterior choice
unless an explicit dynamic model specifies the prediction. Pure covariance
inflation and fixed-prior discount are different operations.

## 4. Gaussian regression and exact protected recall

For keys $k_t\in\mathbb R^d$, values $v_t\in\mathbb R^p$, nonnegative beta
and decay in [0,1], zero-initialized evidence is
$$
S_t=\lambda_tS_{t-1}+\beta_tk_tk_t^\top,\quad
C_t=\lambda_tC_{t-1}+\beta_tv_tk_t^\top,\quad
A_t=S_t+\varepsilon I,\quad M_t=C_tA_t^{-1}.
$$
For epsilon>0, M uniquely minimizes (5.1)
$$
J(M)=\sum_{i\le t}a_{i,t}\|Mk_i-v_i\|^2+\varepsilon\|M\|_F^2,
\quad a_{i,t}=\beta_i\prod_{j=i+1}^t\lambda_j.
$$
PSD/ridge positivity, the normal equation and completing-square optimizer
inequality are Lean-checked. Numerically solve A*y=q; do not form an inverse.

With no discount, independent noise of precision beta and prior
$W\sim\mathcal{MN}(0,I_p,\varepsilon^{-1}I_d)$ give posterior
$\mathcal{MN}(M,I_p,A^{-1})$ by square completion. With fixed-prior forgetting,
the precision update also contains $(1-\lambda)\varepsilon I$; pure scaling
of the old factor omits it and changes the model.

For n independent keys, n<=d, positive weights and Gram $G=K^\top K$, the
original interpolation bound (5.2) is valid under its assumptions:
$$
\|Mk_i-v_i\|\le
\frac{\varepsilon\sqrt{\beta_{\max}/\beta_i}\|V\|_2}
{\beta_{\min}\lambda_{\min}(G)+\varepsilon}.
$$
The analysis derives and numerically checks the bound; it is a convergence
statement, not exact finite-ridge recall. The protected exact branch factors
$K=QR$ and reads $VR^{-1}Q^\top q$. Its left-inverse property gives exact
real-arithmetic protected recall and signed linear-functional retrieval.
Lean proves the general left-inverse and Gram contracts. QR avoids squaring
the key condition number, but rank/conditioning limits remain. It stores
additional geometry/IDs and has no calibrated posterior variance.

The ridge read uses signed coefficients $a_i k_i^\top A^{-1}q$ (5.3).
The posterior mean is squared-loss optimal under the specified static model
(5.4); Lean proves the finite-distribution mean-risk identity, not continuous
Gaussian integration. Zero-ridge generalized least squares is BLUE only for
estimable queries and correct independent-noise weights. Finite ridge is
biased, and repeated-write risk includes that bias.

A normalized positive smoother stays in the value convex hull (5.5), limiting
signed extrapolation. However, sharp softmax separates distinct unit keys
even beyond d stored associations, and log-precision score offsets implement
inverse-variance smoothing. Identical conflicting keys cannot have distinct
deterministic answers. The original stronger smoothing-impossibility claims
are withdrawn.

## 5. Uncertainty, corrected risk and dependent reads

Under the static Gaussian model, latent per-coordinate variance is
$c(q)=q^\top A^{-1}q$; new observation variance is $c(q)+\beta_q^{-1}$ (5.6).
Adding PSD evidence with the same prior lowers c, as Lean proves by a
variational identity. For an unwritten direction, $c(q)=\|q\|^2/\varepsilon$.
Conflicting labels can make c small while item recall fails. No universal
out-of-model calibration or overload-warning claim survives.

For fixed $W^*$, independent noise covariance $\tau_i^2I_p$ and arbitrary
deterministic weights a_i, the corrected risk (5.7) is
$$
\mathbb E\|\hat Mq-W^*q\|^2=
\varepsilon^2\|W^*A^{-1}q\|^2+
p q^\top A^{-1}\left[\sum_i a_i^2\tau_i^2k_ik_i^\top\right]A^{-1}q.
$$
The bracket equals S only for specific weight/noise matches. With discounted
true precisions it is $\sum_i\gamma_i^2\beta_i k_ik_i^\top$, not S.
c does not include fixed-W bias or mismatch. Lean checks the centered-noise
decomposition; the covariance substitution is derived and Monte Carlo checked.

Orthonormal full-capacity graph codes give exact zero-ridge chase (5.8).
If true intermediate queries have one-step error at most e_1 and the
approximate map has Lipschitz L, Lean proves
$e_H\le e_1\sum_{j=0}^{H-1}L^j$. Many-to-one graph codes have norm
$\sqrt{\max\operatorname{indegree}}$, so unit output vectors do not imply
L<=1. H reads incur H sequential read costs.

Protected/unregularized recall reaches d independent associations and the
value-linear capacity converse (5.9). Prior, geometry, factor caches and
addresses remain extra resources. Real dimensions do not establish Shannon
bit efficiency, and positive ridge is not exact.

## 6. Work, floor, gradients and numerical precision

The matrix scan element is $(\lambda,\beta kk^\top,\beta vk^\top)$ (6.1).
Prefix evidence plus factorization at chunk boundaries and compatible rank
updates inside chunks costs (6.2)
$$O\big(N(d^2+pd)+(N/c)d^3\big).$$
For c>=d this is quadratic work per token, including boundary factors, for
no discount or the cyclic-prior model. Depth includes chunk, scan and
factor/solve depth; all-prefix factor storage is O(Nd squared) without
recomputation. Arbitrary fixed isotropic-floor forgetting needs a separate
algorithm before the same cheap bound can be claimed.

No-discount rank-one Cholesky decode is quadratic (6.3). The repaired cyclic
model uses two rank-one updates. In contrast, forgetting with a fixed
isotropic floor has a full-rank injection; d standard rank-one injections
cost cubic work. Omitting them is not an implementation of the same posterior.

The replacement floor lemma (6.4) assumes constant lambda in (0,1]. Initialize
$F_0=\varepsilon\operatorname{diag}(1,\lambda^{-1},\ldots,\lambda^{-(d-1)})$;
after scaling, inject $a=\varepsilon(\lambda^{-(d-1)}-\lambda)$ into the
rotating coordinate. Then every diagonal lies between epsilon and
$\varepsilon\lambda^{-(d-1)}$ from initialization. Lean proves the floor,
ceiling and cycle identities; factors match dense solves numerically. This is
an anisotropic prior with explicit gate/width constraints. The original
“few percent” phantom condition is false. Variable gates need a new schedule
or a documented refactor fallback.

The fallback keeps the prior undiscounted: $A_t=S_t+\operatorname{diag}(f)$ with
$S_t=\lambda_tS_{t-1}+\beta_tk_tk_t^\top$. For **every** sequence of gates
$\lambda_t\ge0$ and precisions $\beta_t\ge0$, $S_t$ is PSD, so
$x^\top A_tx\ge\min_jf_j\,\|x\|^2$ and the latent variance obeys
$0\le q^\top A_t^{-1}q\le\|q\|^2/\min_jf_j$. The exact solve minimizes the
discounted weighted ridge objective with penalty $\sum_jf_jM_j^2$ per value
coordinate. `Selective.lean` proves these statements and the equality of the
chunked affine scan with the sequential recurrence. The price is an exact
$O(d^3+pd)$ refactor per decoded token instead of two rank-one updates. The
trained screen (§12) measures the realization that pays it; that realization
also changes encoders, width and composition, so the screen does not isolate
selectivity alone.

The state adjoint is $g_t=\ell_t+\lambda_{t+1}g_{t+1}$ (6.5).
For y=A inverse q and upstream g, read derivatives are
$$
\partial_C L=g y^\top,\quad
\partial_S L=-\operatorname{sym}[(A^{-1}C^\top g)y^\top],\quad
\partial_q L=A^{-1}C^\top g.
$$
All terminal-stream k/v/beta/lambda/q derivatives are finite-difference checked.
Cyclic-floor parameter and discrete anchor-selection gradients are later work.

Normalized keys, bounded beta, zero initialization and decay below one give
$\|S_t\|_2\le\beta_{\max}/(1-\bar\lambda)$ and the associated fixed-ridge
condition bound (6.6). Lean checks a scalar evidence invariant and the
decay-product bound. Only the state Jacobian is nonexpansive; read/gate
derivatives can grow as inverse epsilon. Float32 factor/read conformance and
conditioning tests remain necessary.

## 7. Resource and hardware claims

GEMMs, rank updates and triangular solves have different hardware behavior
(7.1). Arithmetic intensity does not establish an end-to-end compute-bound
training kernel. Measured TPU costs of the trained realizations appear in §12
and the implementation contract; no roofline bound is claimed.

Ideal packed evidence/factor state uses d(d+1)/2+pd words per head (7.2).
Keeping both S and a factor, metadata and temporary buffers increases this.
For d=p=64, 16 heads and 48 layers, 4,743,168 packed words occupy 18.10 MiB
at float32 or 9.05 MiB at 16-bit precision. Word counts do not establish latency.

| Operator | Decode work | State | Contract |
|---|---|---|---|
| Full-cache softmax | O(K(d+p)) | O(K(d+p)) | Sharp distinct-key lookup; larger cache at increasing K |
| Hebbian/Mamba-3 primitive | O(dp), rank-dependent MIMO | O(dp) plus metadata | Encoded recurrence, not trained performance |
| Gaussian ridge | O(d squared+pd) for compatible factor update | O(d squared+pd) | Conditional calibration; positive-ridge bias |
| Gaussian ridge, any gates, fixed floor | O(d cubed+pd) exact refactor | O(d squared+pd) | Floor for every gate sequence; conditional calibration; positive-ridge bias |
| Protected cached QR | O(d squared+pd) per bank | O(d squared+pd) plus IDs | Exact retained independent anchors |
| Repaired cascade | O(R(d squared+pd)) for R queried banks | O((d squared+pd) log m) plus IDs | Retained capacity with counted selection/routing |

## 8. Protected redundant cascade

Keep one or two banks at each occupied scale (8.1). On the third, merge the
two oldest. Each bank has <=d independent protected anchors with cached QR
and a separate additive background state. Reselect <=d anchors on merge;
evicted items lose exact status while their background evidence remains.
Background writes cannot contaminate exact anchor reads.

There are fewer than m merges over m base blocks (8.2). Background addition
is quadratic, but protected selection/refactor can cost O(d cubed+pd squared)
per merge. Base span c0>=d keeps amortized maintenance within quadratic work
per token, not the old O(d) claim.

For counts b_l in {1,2} at L contiguous occupied scales, Lean checks (8.3)
$$
2^L-1\le m=\sum_l b_l2^l\le2(2^L-1),\qquad L\le\sum_l b_l\le2L.
$$
Full-rank eligible blocks supply d retained anchors per bank, giving
Theta(d log m) retained exact items. The old one-bank binary counter has
popcount(m) banks and only one at powers of two; child unions also overload d.
The repair changes retention, storage and maintenance contracts explicitly.

The universal age-only power-law theorem is withdrawn (8.4). Risk depends on
rank, noise, drift, selection and query. Aligned undiscounted backgrounds add
(8.5); protected selection is not generally associative. Dense message sizes,
block boundaries and IDs count. Reading all banks has a logarithmic factor
(8.6); constant expected routed cost requires a validated query/routing model.
c alone cannot certify reliable routing under conflicts or overload.

## 9. Statistical findings and limitations

The [full report](analysis/results/REPORT.md) retains the frozen protocol,
raw arrays, source hashes, every peer result, intervals and multiplicity
accounting. Development and held-out seeds are separate. Whole trials and
trajectories are sampling units.

Within independent-key capacity, protected QR reaches numerical exactness.
Signed-query ridge MSE is about 7.15e-15 for random keys and 4.31e-13 for
clustered keys, versus softmax 6.29 and 6.65. Full-cache softmax retains distinct
arbitrary values beyond d; fixed-state ridge incurs expected compression loss.
Static in-model latent 95% coverage is 95.24%, versus 17.03% under prior
mismatch and 0% for duplicate-conflict recall. The corrected weighted-risk
identity agrees with Monte Carlo.

After 256 eight-item blocks, the old binary cascade has one bank and all-item
MSE about 0.972. The repair retains 72 exact anchors in nine banks at eight
levels, with error around 1.43e-28; 1,976 items lose exact status. This is not
all-history preservation or a learned-routing result.

Mamba-3 primitives follow its [paper](https://arxiv.org/html/2603.15569v1) and
[pinned step kernel](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py).
Nonzero-phase, variable-gate, previous-input and rank>1 checks establish
primitive conformance. The tied encoded-operator comparisons do not include
full learned models and cannot rank language-model perplexity or TPU speed.

## 10. Remaining dependencies

Phases 03–04 implemented matched models and fused kernels on v4-32 and ran
the requested 60M/1B-token one-seed screen (§11–§12). A budgeted 125M
selection and 3B-token three-seed comparison remain plans; they need the
user's authorization. Still open: protected banks and routing inside the
selective composition, a profiled breakdown of hardware cost, and
training-seed variability.

Other conjugate families need their own likelihood, calibration, capacity and
resource proofs. An affine auxiliary head can distinguish AB from BA when
undiscounted token-local Gaussian evidence loses order, but this does not prove
general fixed-state reasoning. Universal growing-cache dominance is disproved.

The [complete claim ledger](docs/claim-ledger.md),
[deep derivations](docs/mathematical-analysis.md),
[Lean project](formal/Mamba4.lean), and [phases](phases/README.md) preserve the
original scope and make every corrected contract and remaining gate reviewable.

## 11. Language-model architecture

**Memory layer.** The layer implements Definition 5.1:
- **Gates:** token drift `λ_t = exp(−e^{a} softplus(·))` and evidence precision
  `β_t = σ(·)`.
- **Floor:** undiscounted, per head and key coordinate, initialized to 1 with
  minimum 0.25.
- **Encoders:** values, keys and queries pass through a causal width-4
  convolution and SiLU, then keys and queries are normalized to unit length.
- **Key alignment:** each value is written under the key of the preceding
  position, while the query uses the current one. A read therefore returns
  what followed the current context last time, an induction lookup built into
  the memory's geometry. It adds no parameters.
- **Exact read:** every token reads `C_t y_t` and `q_t^T y_t`, with
  `A_t y_t = q_t` solved exactly. Chunked evidence products feed a Cholesky
  factor with the batch on the TPU lane axis. The reverse pass reuses the
  factor: `z = A^{-1}g`, `∂q = z`, `∂A = −sym(z y^T)`.
- **Read path:** the variance drives a learned confidence gate. A skip term and
  an order-sensitive scan are added before a gated output projection.

**Composition.** The 60M model interleaves four memory layers (key dimension
64, eight heads of width 128) with fifteen unmodified official Mamba-3 SISO
layers, `SSSMSSSMSSSMSSSMSSS`. It has 59,893,216 parameters, within 1% of both
peers. Development pilots chose this composition and then the key alignment,
on unseen training batches only. Protected QR banks are not part of the
language model; their operator contracts are unchanged, and no trained
protected-bank claim is made. Costs, state and conformance are in the
[implementation contract](docs/mamba4-implementation.md).

Operator guarantees keep their hypotheses. Learned encoders, forgetting and
the neural read path do not inherit exact recall or calibration, and the
measured advantage belongs to this composition, not to the memory layer
alone.

## 12. Trained 60M screen

Each model trains once (seed 42) on exactly 1B FineWeb-Edu targets with the
pinned GPT-2 tokenizer, batch 128 × 1,024, AdamW and a 7,630-step cosine
schedule, on all sixteen v4 chips. Total and non-embedding parameters match
within 1%. The primary metric is held-out NLL on 2,097,152 targets from whole
held-out documents. A fresh confirmatory holdout comes from an unused official
shard.

| Model | Held-out NLL | Fresh-holdout NLL | Recall (of 128) | Train tokens/s |
|---|---:|---:|---:|---:|
| Transformer | 3.5352 | 3.4960 | 18 | 1,379,725 |
| Mamba-3 | 3.4758 | 3.4378 | 30 | 152,335 |
| Mamba 4 | 3.4719 | 3.4333 | 69 | 93,393 |

The declared strict screen win holds. Paired per-sequence differences
(Mamba 4 minus peer, over 2,048 sequences of 1,024 targets, with a circular
block bootstrap in blocks of eight sequences) are:

| | Held-out split | Fresh holdout |
|---|---:|---:|
| vs Mamba-3 | −0.0039 nats (95%: −0.0052 to −0.0025) | −0.0045 (−0.0059 to −0.0031) |
| vs Transformer | −0.0633 (−0.0653 to −0.0614) | −0.0628 (−0.0647 to −0.0608) |

The intervals describe evaluation samples only. With one training seed they do
not estimate seed variance; the 125M three-seed study remains the test of
robustness. The same model without key alignment reached 3.4727, so alignment
leaves NLL essentially unchanged.

Recall is a frozen 8-way diagnostic, not a selection target. Mamba 4 answered
69 of 128 prompts, against 30 for Mamba-3, 18 for the Transformer and 45 for
the model without alignment. Exact paired McNemar tests give p < 10⁻⁷ against
both peers and p = 0.0004 against the ablation (`docs/trained-recall.md`).

Training throughput is 0.61× Mamba-3's. The cache is a constant 7.3 MiB per
sequence. Every learned precision respected its floor at every scheduled
diagnostic: the minimum eigenvalue was 0.7208 against a floor of 0.7179
(`lm/results/screen-60m/`).
