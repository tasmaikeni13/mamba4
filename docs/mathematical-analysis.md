# Mathematical analysis and repaired contracts

Analysis date: 2026-10-08. The original monograph is preserved verbatim in
[mamba4-original.md](mamba4-original.md). Every named original claim is assessed
in [claim-ledger.md](claim-ledger.md). `mamba4.md` is the corrected theory paper.
These documents distinguish a derivation, a Lean theorem, an experiment, and a
future implementation claim. None is interchangeable with another.

## 1. The state, objective, and statistical model

Write keys as columns of \(K\in\mathbb R^{d\times n}\), values as columns of
\(V\in\mathbb R^{p\times n}\). Let
\[
\gamma_{i,t}=\prod_{j=i+1}^t\lambda_j,\quad a_{i,t}=\gamma_{i,t}\beta_i,
\quad B_t=\operatorname{diag}(a_{1,t},\ldots,a_{t,t}).
\]
The empty product is 1; the write's own decay does not discount its new evidence.
Starting from zero evidence gives
\[
S_t=K B_t K^\top,\quad C_t=V B_t K^\top,\quad A_t=S_t+\varepsilon I.
\]
For \(\varepsilon>0\), the read is \(M_tq\), \(M_t=C_tA_t^{-1}\).
An implementation solves \(A_ty=q\) and returns \(C_ty\); it should not
materialize an inverse. The Python code uses Cholesky or QR solves.

Expanding the loss gives
\[
J(M)=\sum_i a_{i,t}\|Mk_i-v_i\|^2+\varepsilon\|M\|_F^2,
\quad \nabla J(M)=2(MA_t-C_t).
\]
For \(E=M-M_t\), cancellation of the normal-equation cross term gives
\[
J(M)-J(M_t)=\operatorname{tr}(E A_t E^\top)
 =\sum_i a_{i,t}\|Ek_i\|^2+\varepsilon\|E\|_F^2.
\]
It is nonnegative and is strictly positive for nonzero \(E\). This proves
existence, optimality and uniqueness. `Ridge.lean` proves the finite-sum
identity and optimizer inequality for each output coordinate;
`Readout.lean` proves the matrix normal equation. `Regression.lean` connects
these statements end to end: build weighted evidence from observations,
prove precision positive/invertible, solve it, derive normal equations and
prove the actual returned coefficients minimize the weighted objective.
The strict-uniqueness
consequence is a written derivation, rather than a separate Lean theorem.

For the static probabilistic interpretation, assume independent rows of
\(W\sim N(0,\varepsilon^{-1}I)\), fixed/predictable keys and independent
\(v_i\mid W,k_i\sim N(Wk_i,\beta_i^{-1}I_p)\), with \(\beta_i>0\).
With no discount, completing the square in the log likelihood gives
\[
W\mid D\sim\mathcal{MN}(C A^{-1},I_p,A^{-1}).
\]
Thus \(M q\) is the conditional posterior mean,
\(\operatorname{Var}(Wq\mid D)=c(q)I_p\), \(c(q)=q^\top A^{-1}q\).
For a new independent observation with precision \(\beta_q\), add
\(\beta_q^{-1}I_p\) to the covariance. Calibration refers to repeated data
generated from this prior/likelihood model, not every fixed unknown \(W^*\).
Learned keys/precisions depending on the same observations also require a
coherent conditional/generative model; calling a learned gate a precision
does not establish the independence assumptions or calibration of that head.
`Risk.lean` proves squared-loss posterior-mean optimality for a finite
distribution whose mean is supplied. It does **not** prove the continuous
matrix-normal posterior integration. That integration is the square-completion
derivation above, checked numerically by the calibration experiment.

## 2. What tempering does

The unanchored conjugate update raises the previous posterior kernel to power
\(\lambda\) and the new likelihood to power \(\beta\). That is a
generalized/power posterior unless an explicit dynamic model supplies the
corresponding predictive distribution. In a Gaussian, pure covariance
inflation preserves the mean and changes precision to \(\lambda A\).
Discounting only evidence while keeping the prior fixed instead gives
\[
A_t=\lambda_t A_{t-1}+(1-\lambda_t)\varepsilon I
       +\beta_t k_tk_t^\top,\quad C_t=\lambda_t C_{t-1}+\beta_t v_tk_t^\top.
\]
The extra diagonal term anchors the prior. Its prediction also changes the
mean; it is not merely covariance inflation around the old mean. Equivalently
the new posterior kernel is
\(p_{t-1}(W)^{\lambda_t}p_0(W)^{1-\lambda_t}p(v_t\mid W)\).
No unspecified maximum-entropy argument makes all of these rules exact Bayes.

For a general likelihood precision \(\beta_i\), \(\beta_i\) is already
in the observation likelihood. An extra evidence multiplier is a separate
quantity. Calling both of them beta hides the distinction between a noise
model and a deliberately downweighted observation.

## 3. Scan algebra and its actual limits

An affine summary is \((\Lambda,X)\), acting as
\(z\mapsto\Lambda z+X\). Chronological combination is
\[
(\Lambda_2,X_2)\circ(\Lambda_1,X_1)
=(\Lambda_2\Lambda_1,X_2+\Lambda_2X_1).
\]
Associativity follows from distributivity; identity is \((1,0)\).
`Scan.lean` proves associativity, both identities, composition of the action,
and equality of summary application and sequential execution for any finite
list in a real module. It is instantiated by the product of the S/C spaces.
The tree prefix reference independently checks every prefix, including
zero-decay resets and non-power-of-two lengths. A tree uses O(N) combines and
O(log N) dependency depth. Its coordinate work is
\(O(N(d^2+pd))\); parallel hardware and communication determine actual span.

At \(\lambda=1\), evidence addition commutes. With discounts it generally
does not: \((1/2,1)\circ(1/2,2)\ne(1/2,2)\circ(1/2,1)\).
Chronology is part of a discounted summary. Arbitrarily adding discounted
posteriors is wrong. Add undiscounted evidence; merge summaries in sequence
order when discounting.

Generic circuit simulation gives a P-hard recurrence-evaluation problem when
state dimension/description grows with input size; polynomial-time evaluation
puts the appropriately encoded problem in P. This standard complexity result
does not say that every nonlinear recurrence is sequential or that every
efficient monoid must have a small linear realization. Finite automata can
scan their transition tables, and special nonlinear functions can have compact
composition formulas. The size/cost of a summary is essential.

`Embedding.lean` proves the *sufficient* linearization result: a finite basis
of a precomposition-invariant function space yields matrices on evaluation
features, and separation makes that feature map injective. It does not prove
an unconditional converse or a practically small embedding. A separating
space on reachable states gives an embedding on those states, not unreachable
states. The paper's statement needed this domain correction.

## 4. Sufficiency is not uniquely conjugacy

The factorization of exponential-family likelihoods gives a sufficient sum of
statistics. For a regular, identifiable full family, the appropriate statistic
can be minimal sufficient. Adding constants/sample count or ancillary key
geometry can introduce redundancy; for fixed Gaussian design, S is known and
C carries the W-dependent statistic. A conjugate prior is convenient for
representing a posterior, but a nonconjugate prior can also use sufficient
statistics. Pitman–Koopman–Darmois applies to regular IID sampling with common
support and suitably regular statistics. Merely permitting an arbitrary
measurable encoding into one real defeats a dimension-only statement.

These distinctions follow the [exponential-family lecture notes](https://www.stat.umn.edu/geyer/8054/notes/expfam.html).
We do not claim a new uniqueness theorem or a Lean formalization of that
characterization. A conditional Gaussian design is not an IID observation
model over arbitrary learned, history-dependent encoders.

Initialized natural parameters include the prior. The union of two streams is
\[
\chi_{12}=\chi_1+\chi_2-\chi_0,\quad
\nu_{12}=\nu_1+\nu_2-\nu_0.
\]
Only zero-initialized **evidence** adds without correction. Lean checks the
additive identity and exponential-kernel product identity. Some attention
summaries and linear RNNs are mergeable too; exclusivity claims are removed.

## 5. Exact interpolation, the bound, and a stronger numerical method

For independent keys, \(n\le d\), \(B=\operatorname{diag}(\beta_i)>0\),
let \(H=B^{1/2}K^\top K B^{1/2}\). The push-through identity gives
\[
M K=V B^{1/2}H(H+\varepsilon I)^{-1}B^{-1/2},
\quad M K-V=-\varepsilon V B^{1/2}(H+\varepsilon I)^{-1}B^{-1/2}.
\]
Consequently the original columnwise bound is valid under these assumptions:
\[
\|Mk_i-v_i\|\le
\frac{\varepsilon\sqrt{\beta_{\max}/\beta_i}\|V\|_2}
{\beta_{\min}\lambda_{\min}(K^\top K)+\varepsilon}.
\]
It is a convergence-to-interpolation bound, **not exactness at positive
epsilon**. A one-dimensional unit-key/unit-value example returns
\(1/(1+\varepsilon)\ne1\). At the square random capacity boundary,
rare nearly singular matrices make average finite-ridge error appreciable.
The experiment checks the bound for dimensions 8, 16 and 32, two loading
fractions, and random positive evidence in [exp(-3), exp(3)].
The bound is derived here and tested, not fully formalized as a spectral norm
inequality in Lean.

The stronger exact branch keeps at most d independent **protected** keys and
their values. Factor \(K=QR\) with thin QR and read
\[
M_{\rm exact}q=V R^{-1}Q^\top q.
\]
Since \(R^{-1}Q^\top K=I\), it exactly reads each protected key and
every signed linear-functional query in their span, in real arithmetic.
`Readout.lean` proves this left-inverse contract and the equivalent Gram
construction, deriving Gram invertibility from injectivity of the key map.
QR avoids forming a normal equation with squared condition number. Cached
factors give \(O(dn+n^2+pn)\) read work, at most \(O(d^2+pd)\).
Choosing/replacing anchors and factorization cost are counted separately.
Exact mode has no positive ridge floor and no calibrated Bayesian variance.
It is a different estimator with additional stored geometry and addresses.

Equal/overlapping distinct keys are not the same as identical conflicting
addresses. No function can return both 0 and 2 for the same key. Distinct
unit keys have self-similarity 1 and cross-similarity below 1, so arbitrarily
sharp softmax also tends to exact lookup, even when n exceeds d. Claims that
smoothing can never separate overlapping distinct keys are false.

A softmax read remains in the convex hull of the observed values; signed or
scaled linear-functional outputs can be outside that hull. This is the
well-defined extrapolation advantage established here. Multiple Transformer
layers, heads, projections and nonlinear features have a larger function
class than this single-head operator, so this is not a universal Transformer
expressivity theorem.

### Estimable-query Gauss–Markov statement

For independent noise with covariance B inverse, an estimator V a is unbiased
for Wq exactly when K a=q. For q in range(K), take
\(a_*=B K^\top S^+q\), where \(S=KBK^\top\) and S plus is the
Moore–Penrose inverse. Any other unbiased a is a_*+h with Kh=0. The covariance
cross term vanishes:
\(a_*^\top B^{-1}h=q^\top S^+Kh=0\). Its excess variance is
\(p\,h^\top B^{-1}h\ge0\). This proves the BLUE claim only for
estimable queries and correct noise weights. At positive ridge the estimator
is generally biased, so this unbiased-estimator comparison no longer applies.
This derivation is analytic; it is not a separate Lean Gauss–Markov theorem.

## 6. Risk with weighted and discounted noise

Assume fixed \(W^*\), independent noise of row covariance
\(\tau_i^2 I_p\), and arbitrary deterministic nonnegative weights a_i.
Let \(S=\sum_i a_i k_i k_i^\top\), \(A=S+\varepsilon I\). Then
\[
\hat M q-W^*q=-\varepsilon W^*A^{-1}q+
       \sum_i a_i\xi_i(k_i^\top A^{-1}q),
\]
so, using zero mean and zero cross covariance,
\[
\mathbb E\|\hat M q-W^*q\|^2=
\varepsilon^2\|W^*A^{-1}q\|^2+
p\,q^\top A^{-1}\Big[\sum_i a_i^2\tau_i^2 k_i k_i^\top\Big]A^{-1}q.
\]
The bracket is S only when \(a_i^2\tau_i^2=a_i\), for example
undiscounted correctly specified precisions. With homoscedastic noise and
arbitrary weights it is \(\sigma^2\sum_i a_i^2k_i k_i^\top\).
With true precisions beta and forgetting, it is
\(Q=\sum_i\gamma_i^2\beta_i k_i k_i^\top\), not
\(S=\sum_i\gamma_i\beta_i k_i k_i^\top\).
Since \(0\le\gamma_i\le1\), \(Q\preceq S\), and
\(A^{-1}SA^{-1}\preceq A^{-1}\). Thus posterior c can upper-bound this
variance contribution; it does not include fixed-W bias or model mismatch.
Under the correctly specified no-discount Bayesian prior, averaging the bias
term over W contributes \(p\varepsilon\|A^{-1}q\|^2\). Adding the
noise variance gives \(p q^\top A^{-1}(S+\varepsilon I)A^{-1}q=p c(q)\),
which reconciles Bayesian variance with the conditional frequentist formula.
`Risk.lean` verifies the abstract centered-noise squared-error identity;
the Gaussian weighted covariance substitution is an analytic derivation.

For repeated scalar noisy observations of one value, zero-ridge precision
averaging has risk \(1/\sum_i\beta_i\). Equal noise gives
\(\sigma^2/n\) per coordinate. At finite epsilon and true value theta,
\[
R=\frac{\varepsilon^2\theta^2+\sum_i\beta_i}
{(\varepsilon+\sum_i\beta_i)^2}.
\]
Repeated identical keys with scores offset by log beta produce the same
precision-weighted softmax mean. Unequal multiplicities alone are already
handled by ordinary averaging when observation variances agree. The original
claim of unavoidable smoothing suboptimality is retracted.

For exponential forgetting with equal precision and stationary noise,
\[
n_{\rm eff}=\frac{(\sum_j\lambda^j)^2}{\sum_j\lambda^{2j}}
\longrightarrow\frac{1+\lambda}{1-\lambda}.
\]
Under a drifting scalar random walk with increment variance Q and observation
variance sigma squared, the infinite-horizon unregularized exponential read
has variance/noise risk
\[
R(\lambda)=\sigma^2\frac{1-\lambda}{1+\lambda}
+Q\frac{\lambda^2}{1-\lambda^2}.
\]
The second term follows by expressing the lag error as a sum of random-walk
increments weighted by \(\lambda^r\). Differentiating this expression
motivates tuning forgetting to drift/noise, instead of demanding a universal
gate. The drift suite compares independently tuned windows and decays on
causal held-out trajectories, with one trajectory as a statistical cluster.

## 7. Confidence is conditional, not a universal error detector

For SPD A and \(x=A^{-1}q\),
\[
c(q)=\max_z\{2q^\top z-z^\top A z\},\qquad
c(q)-(2q^\top z-z^\top A z)=(z-x)^\top A(z-x).
\]
Adding PSD evidence lowers this maximum. `Confidence.lean` proves both
identities in bilinear form, with explicit symmetry, PSD and solve
hypotheses. On an unwritten direction, \(Sq=0\) gives
\(c(q)=\|q\|^2/\varepsilon\), which equals epsilon inverse only for a
unit query. Discounting evidence can increase c; monotonicity concerns
addition with the same prior, not arbitrary stream updates.

Write the same unit key n times with contradictory labels. Then c is
\(1/(n+\varepsilon)\), approaching zero while arbitrary item-specific
recall error stays large. The fixed linear class identifies one shared value,
so it cannot express both labels. Large fixed-W bias also defeats model-free
calibration. Unwritten-direction confidence cannot identify all overload or
conflict failures. A future reliability head needs residual/consistency
statistics, a noise model, and an out-of-distribution validation protocol.

## 8. Capacity, bytes, and unavoidable comparator wins

If all Kb-bit assignments can be reconstructed, encoding into an m-bit state
is injective. There are \(2^{Kb}\) assignments and \(2^m\) states, so
\(m\ge Kb\). `Capacity.lean` proves this finite-function counting theorem.
The randomized extension needs a specified error convention, independent
random coins and information inequalities. For per-bit error delta at most
one half, conditional entropy per bit is bounded by binary entropy and
\(m\ge Kb(1-H_2(\delta))\) under the usual model. We do not claim a Lean
formalization of Fano/information inequalities.

For fixed keys, a value-linear exact read factors the identity on
\(\mathbb R^{Kp}\) through \(\mathbb R^{p_v}\); therefore
\(Kp\le p_v\). The finrank argument is proved in Lean. This is a
conditional bound on the value-dependent state. It excludes general
nonlinear encoders, other memory stores and infinitely precise encodings.
The Gaussian state has p_v=dp, but additionally uses d(d+1)/2 scalar words
for symmetric S (or its factor), plus metadata, prior and temporary buffers.

Counting real dimensions cannot establish a bit-optimal Shannon rate.
Each real scalar may encode infinitely many bits; a finite precision word has
a fixed range and rounding error. Protected anchors also store K, V, Q/R and
routing IDs. For byte accounting, specify dtype, key/value precision, factor
duplication and address representation. A full KV cache uses K(d+p) words,
with exact addressable values up to its capacity; it is intentionally a
larger-state comparator when K grows. A recent-window KV cache can be
budget-matched but has different eviction semantics. No fixed-state method
can dominate an unbounded cache on arbitrary exact recall.

## 9. Hops and order

Orthonormal keys and successor codes in the same space give exact unregularized
functional-graph chase. At finite ridge, a fully written basis gives
\(\hat T=T/(1+\varepsilon)\); H reads return
\((1+\varepsilon)^{-H}T^Hq\). For a functional graph,
\(\|T\|_2=\sqrt{\max_j\operatorname{indegree}(j)}\), because columns
that share a successor coincide. Unit value vectors do not imply norm at
most 1. Only a permutation/injective map has that property.

If one-step approximation error on each true intermediate query is at most
e_1 and the approximate map has Lipschitz constant L, then
\[
e_H\le e_1\sum_{j=0}^{H-1}L^j.
\]
`Hops.lean` proves the recurrence and geometric envelope for arbitrary
nonnegative L. It does not replace L with 1. Normalizing hop queries can
repair shrinkage for known unit codes, but changes the estimator and requires
an angular error analysis; it does not solve arbitrary graph amplification.
H dependent reads also incur H solve/read costs and H sequential read depth.

Undiscounted evidence loses order when encoders only depend on the token.
AB and BA then have identical S/C. Their ordered targets cannot both be
recovered by any deterministic state-only read. This is a specific limitation,
not a price imposed on attention or all scans. A supplementary affine head
with A:z->-z+1 and B:z->z+2 returns 3 for AB and -1 for BA. It is scannable
using the same monoid and costs constant extra state for this witness.
This repair demonstrates useful order information without claiming general
unbounded stack/algorithmic reasoning in fixed state.

## 10. Decode floor: counterexample and replacement

For undiscounted evidence, rank-one Cholesky updates maintain a fixed prior
in O(d squared) work. With forgetting and a fixed isotropic prior, the exact
update additionally needs \((1-\lambda)\varepsilon I\). Scaling the
factor and doing only the token rank-one update omits that full-rank term.
Applying d standard rank-one prior updates costs O(d cubed). No generic exact
O(d squared) implementation of this fixed-floor step is established here.

The original cycling approximation with injection
\(a=d\varepsilon(1-\lambda)\) has stationary floor extremes
\[
\frac{F_{\max}}{\varepsilon}=\frac{d(1-\lambda)}{1-\lambda^d},
\quad\frac{F_{\min}}{\varepsilon}
=\frac{d(1-\lambda)\lambda^{d-1}}{1-\lambda^d}.
\]
For d=64 and lambda=0.99 these are 1.3491 and 0.7162 even though
\((1-\lambda)d=0.64\le1\). “A few percent” is false.

The repaired cyclic prior uses
\[
a=\varepsilon\big(\lambda^{-(d-1)}-\lambda\big),\quad
F_0=\varepsilon\operatorname{diag}(1,\lambda^{-1},\ldots,\lambda^{-(d-1)}).
\]
At time t inject \(a e_{t\bmod d}e_{t\bmod d}^\top\).
Its diagonal rotates through
\(\varepsilon,\varepsilon/\lambda,\ldots,\varepsilon/\lambda^{d-1}\).
Therefore, from initialization, every coordinate has
\[
\varepsilon\le F_{t,jj}\le\varepsilon\lambda^{-(d-1)}.
\]
`Floor.lean` proves the floor, ceiling, age step, oldest value, and reinjection
identities. Initialize at the specified cycle; a zero prior needs burn-in.
The Python reference scales a lower Cholesky factor and applies one token and
one phantom rank-one update; it verifies \(LL^\top=S+F\) at every step.
This is exactly the **anisotropic** ridge model with prior F, not the fixed
isotropic model. For lambda=1, F=epsilon I and no phantom is needed.
The implemented guarantee assumes constant lambda. Variable learned gates
need a separately proved schedule, lower bound or explicit refactor fallback.

The bound on anisotropy grows when d(1-lambda) grows. To bound it by 1+delta,
require \(-(d-1)\log\lambda\le\log(1+\delta)\). Use this as a
head-width/gate constraint, not an assertion that every forget rate is safe.

## 11. Work, gradients and hardware

At a chunk boundary, build its SPD matrix from a prefix evidence scan and
factor it in O(d cubed) work. Within a chunk of c steps, use exact rank-one
updates for lambda=1, or the repaired cyclic model for constant lambda.
The total work is
\[
O\big(N(d^2+pd)+(N/c)d^3\big).
\]
For c at least d this is O(N(d squared+pd)), including boundary factors.
Time dependency depth contains c within a chunk, log(N/c) across chunk
summaries, and the cost/depth of a factor/solve. All-prefix factor/activation
storage is O(N(d squared+pd)) unless recomputation is used. For the exact
fixed isotropic floor at arbitrary lambda, a naive implementation is
O(N d cubed); the cheaper cost has not been proved for that different model.

For y=A inverse q and scalar loss g transpose C y, the read derivatives are
\[
\partial_C L=g y^\top,\quad
\partial_S L=-\operatorname{sym}[(A^{-1}C^\top g)y^\top],\quad
\partial_q L=A^{-1}C^\top g.
\]
The state adjoint has the affine recurrence
\(g_t=\ell_t+\lambda_{t+1}g_{t+1}\). This does not include every gate,
encoder or solve derivative automatically. The implementation checks all
terminal-loss derivatives with respect to k, v, beta, lambda and q against
central finite differences. \(\partial_\varepsilon L=\operatorname{tr}(\partial_S L)\)
for a fixed isotropic floor. Differentiating the cyclic prior or QR anchor
selection is additional work for phase 3.

Bounded normalized-key evidence gives
\(\|S_t\|_2\le\beta_{\max}/(1-\bar\lambda)\) from zero initialization
when \(0\le\lambda_t\le\bar\lambda<1\). Thus a fixed ridge yields
\(\kappa(A_t)\le1+\beta_{\max}/(\varepsilon(1-\bar\lambda))\).
Only the state-to-state scalar Jacobian is nonexpansive. Reads and gate
derivatives involve A inverse and C, so they can be large, even unbounded
as epsilon tends to zero. Floating-point stability also depends on rank
updates, cancellation, factor residual, dtype and gradients; a single
O(kappa u) slogan is insufficient. The precision suite explicitly exposes
float32 errors and singular normal equations at near-colliding keys.

A flop ledger is not a measured roofline. Solves/rank updates can be
bandwidth or latency limited while GEMMs are compute bound. Packed symmetric
state for d=p=64 is 6,176 words per head; 16 heads and 48 layers use
4,743,168 words, 18.10 MiB at float32 or 9.05 MiB at 16-bit precision.
Keeping S and a separate factor, metadata and gradients increases this.
Decode model latency is a future measurement, not implied by state size.

## 12. Cascade failure and protected redundant banks

In a one-block-per-level binary counter with m completed blocks, the number
of occupied banks is popcount(m), not always Theta(log m); at powers of two
it is 1. Worse, two children each storing d independent associations merge
into 2d arbitrary values in a d-dimensional value-linear map. Exactness is
lost. Unmarked noisy writes in the same regression state can also disturb
marked values. Confidence cannot generally diagnose this overload.

The repair is a redundant binary counter: keep one or two banks at each
occupied level. On the third, merge the two oldest and leave one. Each bank
has at most d independent protected anchors with a cached QR factor, and a
separate background S/C containing all writes. A merge reselects at most d
anchors from the child union, and adds background evidence. Discarded anchors
lose exact-retention status; no missing arbitrary value is hidden in a claim
about the background posterior. Distinct banks have disjoint source blocks
and retained item IDs. A routing oracle is used only to verify local reads;
learning/query routing is unproved and belongs in later experiments.

Let counts per scale be b_l in {1,2}, L occupied scales. Then
\[
2^L-1\le m=\sum_{l=0}^{L-1}b_l2^l\le2(2^L-1),\quad
L\le\sum_l b_l\le2L.
\]
If each initial block offers d independent keys, reselection keeps d in every
bank, giving Theta(d log m) **retained** exact items, including powers of two.
This fails when a block's key rank is smaller; use its actual rank instead.
`Cascade.lean` proves carry mass conservation and both count bounds given the
invariant. The Python tests verify the implementation maintains that invariant
for every prefix; Lean does not formalize the Python selection algorithm.

There are fewer than m merges because every merge reduces the bank count by
one. Anchor selection/refactor costs O(d cubed+pd squared), rather than mere
state addition's O(d squared+pd). Amortized maintenance is
O((d cubed+pd squared)/c0); c0 at least d keeps this within
O(d squared+pd). Cached bank reads cost O(d squared+pd); reading all banks
costs an additional log m factor. The original “free” O(d) maintenance and
universally context-free routed-read claim do not carry over.

Only backgrounds remain additive. Deterministic anchor reselection can be
partition-order dependent and is not an associative merge. A composable
top-priority policy could fix selection associativity but may sacrifice
rank/diversity guarantees; no combination of both is claimed here. Raw
background blocks merge exactly only when their boundaries/weights match.
Messages of dense state size O(d squared) are constant in token count for
fixed d, not constant in d or in the number of levels.

Age alone does not specify risk. It also depends on key covariance, noise,
capacity, selector, query and drift. No general power-law risk theorem survives
this audit. The correct artifact is the retained-set contract and an explicit
future query-age/routing distribution, not a replacement universal slogan.

## 13. Relation to peers and novelty

The regression-memory lens and its connections to attention/SSMs already
appear in [test-time regression](https://arxiv.org/html/2501.12352v3).
Retention/objective choices are studied in
[Miras](https://arxiv.org/html/2504.13173v1), and curvature-aware delta rules in
[Preconditioned DeltaNet](https://arxiv.org/html/2604.21100v1).
These precedents rule out claiming that ridge retrieval, Bayesian linear
regression or regression-as-memory is new. The candidate contribution is the
audited combination of scan evidence, explicit uncertainty, protected exact
anchors and a guaranteed cyclic floor, subject to fair trained validation.

The [Mamba-3 paper](https://arxiv.org/html/2603.15569v1) and
[pinned upstream step kernel](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py)
provide the exponential-trapezoidal and rotary recurrence used here. Our
operator comparison is explicitly untrained; it cannot establish a language
model win, equal learned parameter count, TPU latency, or publication novelty.
