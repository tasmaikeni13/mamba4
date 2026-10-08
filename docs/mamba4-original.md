# Mamba 4: Memory as Inference and the Dissolution of the Expressivity–Efficiency Frontier in Linear-Time Sequence Modeling

*A condensed theoretical monograph. June 2026.*

## Abstract

Recurrent and state-space models achieve linear-time training and constant-cost inference by compressing history into a fixed state, but lose associative recall, multi-hop reasoning, and faithful in-context learning. Attention avoids compression, but pays quadratic training and growing per-token inference. This paper argues both families compress the wrong object. The correct object is the **posterior belief over the latent operator that generated the context**.

We derive **Mamba 4**, a conjugate-posterior sequence architecture, from four bedrock results:

1. Exact recall of \(K\) arbitrary associations requires \(\Omega(K)\) state (Theorem 3.1).
2. Parallel-in-time evaluation of general nonlinear recurrences is P-complete, so parallel training forces essentially linear state transitions (Theorem 3.2).
3. State transitions whose composites span a finite-dimensional function space are exactly the linearly realizable ones (Lemma 3.5).
4. Fixed-dimensional sufficient statistics exist only for exponential families; hence lossless history summarization relative to a hypothesis class forces conjugate posterior parameters, whose updates are linear and scannable (Theorems 4.1, 4.3, 4.4).

Mamba 4 instantiates this with a matrix-Gaussian state: online ridge regression sufficient statistics. It achieves exact associative recall up to the information-theoretic capacity, strict per-access advantages over softmax attention, calibrated retrieval confidence, single-layer multi-hop retrieval, derived forgetting/input gates, \(\Theta(Nd^2)\) training with \(O(\log N)\) span, \(O(d^2)\) decode with a conditioning floor, and a dyadic cascade with \(\Theta(d\log N)\) selectable exact recalls and additive mergeable states.

---

## 1. Thesis

The usual frontier is false: one wall is universal, the rest are self-inflicted.

- **Universal wall:** no fixed-size state can exactly recall unbounded adversarial associations.
- **Self-inflicted wall:** recurrent models compress the signal, not the posterior; update with wrong algebra; read with the wrong estimator.

Mamba 4 compresses the posterior over the latent structure future queries probe. The state is a conjugate natural parameter. The update is Bayes’ rule in natural coordinates, hence additive, associative, and scannable. The read is the posterior predictive, where all nonlinearity lives.

---

## 2. Formal Preliminaries

A streaming machine is \((U,R)\), state \(z_t=U(z_{t-1},x_t)\), read \(R(z_t,q)\). It is scan-realizable if there is a monoid \((\mathcal M,\bullet,e)\) and encoding \(\iota\) such that
\[
z_t=\alpha(\iota(x_t)\bullet\cdots\bullet\iota(x_1),z_0).
\]
Parallel prefix computes all \(z_t\) in \(\Theta(N)\) monoid products and \(O(\log N)\) span.

Tasks:

- \(\mathrm{EAR}(K,b)\): exact recall of \(K\) values of \(b\) bits.
- Linear-functional recall: query \(q=\sum_j\alpha_j k_j\), target \(\sum_j\alpha_j v_j\).
- \(H\)-hop chase: follow functional graph \(H\) steps.
- Adaptive memory rounds: reads whose addressing depends on prior reads.

Generative lens: latent linear map
\[
v_t=Wk_t+\beta_t^{-1/2}\xi_t,\qquad
W\sim\mathcal{MN}(0,I_{d_v},\varepsilon^{-1}I_{d_k}).
\]
Inside the model we get Bayes-optimality; outside it, algebraic and computational guarantees remain.

---

## 3. Three Walls and One Door

### Theorem 3.1 — Exact recall requires linear state

Any deterministic streaming machine solving \(\mathrm{EAR}(K,b)\) for every assignment must have state
\[
m\ge Kb \quad\text{bits}.
\]
Randomized machines require
\[
m\ge (1-H_2(\delta))Kb-1
\]
bits.

*Proof idea:* Deterministic: \(m<Kb\) forces two assignments to share a state; query differing coordinate gives contradiction. Randomized: Fano.

Consequences: “unbounded lossless context in \(O(1)\) state” is void. The real goal is selectable exactness at a state budget with optimal degradation.

### Theorem 3.2 — No free parallelism

The prefix-recurrence problem for general nonlinear \(f\),
\[
z_t=f(z_{t-1},x_t),
\]
is P-complete under logspace reductions. Unless \(\mathrm{NC}=\mathrm P\), no polylog-span parallel-in-time algorithm exists for general nonlinear recurrences.

*Proof idea:* Reduce from the circuit value problem. State holds evaluated gates; token encodes gate type and inputs; \(f\) evaluates one gate per step.

Consequence: every parallel-trainable architecture must restrict state transitions to an essentially linearizable class. Nonlinearity must migrate to token encoders and readout.

### Theorem 3.3 — Capacity converse for value-linear memories

If a memory has value-dependent state
\[
z_{\mathrm{val}}=\sum_i L(k_i)v_i
\]
and read linear in \(z_{\mathrm{val}}\), and it exactly recalls \(K\) associations of dimension \(d_v\), then
\[
Kd_v\le p_v,\qquad K\le \frac{p_v}{d_v}.
\]
*Proof idea:* The composite map from values to recalled values factors through \(\mathbb R^{p_v}\), so rank \(\le p_v\). Exactness forces rank \(Kd_v\).

### Lemma 3.5 — Linearization of composition-closed transitions

If a family \(\{f_x\}\) has a finite-dimensional space \(V\) of functions \(\mathcal Z\to\mathbb R\), \(\dim V=m\), invariant under precomposition \(v\mapsto v\circ f_x\), and separating reachable states, then there is an embedding \(\varphi:\mathcal Z\to\mathbb R^m\) and matrices \(A_x\) such that
\[
\varphi(f_x(z))=A_x\varphi(z).
\]
Thus the recurrence becomes a linear scan.

*Proof idea:* Evaluate basis functions after \(f_x\), expand in basis, stack coefficients.

This identifies the design space: expressive scannable recurrences are exactly useful finite-dimensional precomposition-invariant function spaces.

---

## 4. Bridge: Bayesian Conjugacy Is the Algebra of Scannable Memory

Let observations \(o\) have exponential-family likelihood
\[
\ell(o\mid\eta)=h(o)\exp(\langle T(o),\eta\rangle-A(\eta)).
\]
With conjugate prior \(\pi(\eta\mid\chi_0,\nu_0)\propto\exp(\langle\chi_0,\eta\rangle-\nu_0A(\eta))\), the posterior after \(o_1,\dots,o_t\) is
\[
\chi_t=\chi_0+\sum_{s\le t}T(o_s),\qquad
\nu_t=\nu_0+t.
\]
With evidence weights \(\beta_t\) and tempering \(\lambda_t\),
\[
\boxed{\chi_t=\lambda_t\chi_{t-1}+\beta_tT(o_t),\qquad
\nu_t=\lambda_t\nu_{t-1}+\beta_t.}
\]

### Theorem 4.1 — Conjugate updates are associative scans

The maps
\[
u_t:(\chi,\nu)\mapsto(\lambda_t\chi+\beta_tT(o_t),\lambda_t\nu+\beta_t)
\]
are closed under composition. A block \(s..t\) has multiplier \(\Lambda_{s:t}=\prod_{r=s}^t\lambda_r\) and increment
\[
(X_{s:t},n_{s:t})=\sum_{r=s}^t\Lambda_{r+1:t}\beta_r(T(o_r),1).
\]
The combine
\[
(\Lambda_2,X_2)\bullet(\Lambda_1,X_1)
=(\Lambda_2\Lambda_1,\;X_2+\Lambda_2X_1)
\]
is associative with identity \((1,0)\). Hence all prefix posteriors are computable in \(\Theta(N)\) work and \(O(\dim T\cdot\log N)\) span. For \(\lambda\equiv1\), the monoid is commutative: posterior is order-free.

### Definition 4.2 — Mamba 4 layer

A Mamba 4 layer has:

1. Learned encoders producing observation \(o_t\), evidence \(\beta_t\ge0\), drift \(\lambda_t\in(0,1]\).
2. State \((\chi_t,\nu_t)\), natural parameters of the running conjugate posterior.
3. Learned query maps \(q_t\).
4. Read: posterior predictive at \(q_t\), with optional uncertainty.
5. Outputs of reads, concatenated over heads/hops, mixed by a learned projection.

Between tokens: linear scan. At each token: nonlinear encoders and read. Across layers: reads feed encoders.

### Theorem 4.3 — Sufficiency and minimality

Under the conjugate exponential-family model, \((\chi_t,\nu_t)\) is sufficient for \(\eta\) given history. If the family is minimal, it is minimal sufficient.

### Theorem 4.4 — Canonicity (Koopman–Pitman–Darmois)

Among positive smooth densities with parameter-independent support, only exponential families admit sufficient statistics of dimension bounded in sample size.

Thus the only fixed-size states that losslessly summarize unbounded history relative to an inference class are conjugate posterior parameters.

### Corollary 4.5 — Mergeability

With \(\lambda\equiv1\), the state of a union of streams is the sum of independently computed states:
\[
(\chi^{(1\cup2)},\nu^{(1\cup2)})
=(\chi^{(1)}+\chi^{(2)},\nu^{(1)}+\nu^{(2)}).
\]
Attention cannot do this except by concatenating growing caches. Nonlinear RNNs cannot do it at all.

### Proposition 4.6 — Gates derived, not designed

- Discounting by \(\lambda\) is exact Bayesian tempering under maximum-entropy drift. For Gaussian families it is covariance inflation with process noise proportional to current uncertainty.
- Evidence weight \(\beta\) is the precision of the token’s evidence.

Thus forget gates are drift rates; input gates are evidence precisions.

---

## 5. Gaussian Instance: Exact, Optimal, Introspective Associative Memory

### Definition 5.1 — Gauss–Markov memory

Per head, encoders emit \(k_t\in\mathbb R^{d_k}\), \(\|k_t\|=1\), \(v_t\in\mathbb R^{d_v}\), \(q_t\in\mathbb R^{d_k}\), \(\beta_t\ge0\), \(\lambda_t\in(0,1]\). State:
\[
S_t=\lambda_tS_{t-1}+\beta_tk_tk_t^\top,\qquad
C_t=\lambda_tC_{t-1}+\beta_tv_tk_t^\top.
\]
Let \(A_t=S_t+\varepsilon I\). Read and confidence:
\[
\mathrm{read}_t(q)=C_tA_t^{-1}q,\qquad
c_t(q)=q^\top A_t^{-1}q.
\]
Model-free, \(M_t=C_tA_t^{-1}\) solves discounted ridge regression:
\[
\min_M\sum_{s\le t}\Lambda_{s+1:t}\beta_s\|Mk_s-v_s\|^2+\varepsilon\|M\|_F^2.
\]
The recurrence has no inverse, normalization, or nonlinearity. The solve is the relocated nonlinearity.

### Theorem 5.2 — Interpolation

For \(\lambda\equiv1\), \(K\le d_k\), independent keys with Gram \(G=K^\top K\), weights \(\beta_i>0\),
\[
\|\mathrm{read}(k_i)-v_i\|
\le \varepsilon\sqrt{\frac{\beta_{\max}}{\beta_i}}
\frac{\|V\|_2}{\beta_{\min}\sigma_{\min}(G)+\varepsilon}.
\]
Thus \(\mathrm{read}(k_i)\to v_i\) as \(\varepsilon\to0\): exact recall with no crosstalk for arbitrary independent keys.

### Proposition 5.3 — Whitened attention

\[
\mathrm{read}(q)=\sum_s w_s(q)v_s,\qquad
w_s(q)=\Lambda_{s+1:t}\beta_sk_s^\top A_t^{-1}q.
\]
This is attention in Mahalanobis key geometry. Weights need not be positive or sum to one.

### Theorem 5.4 — Optimal read

- (Bayes) Under the latent linear-map model, \(\mathrm{read}_t(q)\) is the posterior mean of \(Wq\), minimizing expected squared retrieval error among all measurable functions of history.
- (Gauss–Markov) As \(\varepsilon\to0\), for \(q\) in key span, the read is the minimum-variance unbiased estimator of \(W^*q\) among value-linear estimators.
- (Optimal averaging) For \(n\) repeated noisy writes, risk is \(\sigma^2d_v/n\).

### Proposition 5.5 — Smoothing versus solving

A normalized smoothing read is \(\hat v(q)=\sum_sw_s(q)v_s\), \(w_s\ge0\), \(\sum w_s=1\).

- Collision blindness: smoothing cannot exactly separate overlapping keys; Mamba 4 read does as \(\varepsilon\to0\).
- Convex-hull confinement: smoothing lies in \(\mathrm{conv}\{v_s\}\); linear-functional queries outside the simplex cannot be realized. Mamba 4 read is linear in \(q\), so it realizes them exactly within capacity.
- Suboptimal averaging: smoothing cannot implement inverse-variance weighting across unequal multiplicities.

### Proposition 5.6 — Calibrated introspection

Under the model, per-coordinate predictive variance is
\[
c_t(q)+\beta_q^{-1}.
\]
Model-free, \(c_t(q)\) is monotone in evidence and equals \(\varepsilon^{-1}\) on never-written directions. Thus retrieval confidence is exact in-model and meaningful outside.

### Theorem 5.7 — In-context regression risk

For \(v_i=W^*k_i+\sigma\xi_i\),
\[
\mathbb E\|\mathrm{read}(q)-W^*q\|^2
=
\underbrace{\varepsilon^2\|W^*A^{-1}q\|^2}_{\text{bias}}
+
\underbrace{\sigma^2d_v\,q^\top A^{-1}SA^{-1}q}_{\le\sigma^2d_v c_t(q)}.
\]
Each head performs ridge regression in context; with learned features, kernel ridge regression. The variance term is reported as \(c_t(q)\).

### Theorem 5.8 — Multi-hop inside one layer

With orthonormal codes and \(\le d_k\) edges, \(H\) chained reads
\[
q^{(j+1)}=\mathrm{read}(q^{(j)})
\]
return the \(H\)-th successor exactly as \(\varepsilon\to0\). For finite \(\varepsilon\), error is bounded by
\[
\epsilon_1\sum_{j<H}L^j,
\]
where \(L=\|CA^{-1}\|\le1\) under normalization. Thus one layer realizes \(H\) adaptive memory rounds.

### Theorem 5.9 — Capacity optimality

The Gauss–Markov memory with key dimension \(d_k\) exactly recalls \(K=d_k\) independent associations. No value-linear memory with \(p_v=d_kd_v\) exceeds \(K=d_k\). Thus capacity is met with equality, within factor \(1+d_k/d_v\) of the Shannon floor.

---

## 6. Algorithms and Complexity

### Proposition 6.1 — Scan element

For recurrence (5.1), the per-token monoid element is
\[
(\lambda_t,\;\beta_tk_tk_t^\top,\;\beta_tv_tk_t^\top),
\]
with combine
\[
(\Lambda_2,S_2,C_2)\bullet(\Lambda_1,S_1,C_1)
=(\Lambda_2\Lambda_1,\;S_2+\Lambda_2S_1,\;C_2+\Lambda_2C_1).
\]
Associative, identity \((1,0,0)\), size \(d_k^2+d_kd_v+1\), cost \(O(d_k^2+d_kd_v)\).

### Theorem 6.2 — Training cost

Chunked scan computes all reads with
\[
\Theta(N(d_k^2+d_kd_v))
\]
work, span \(O(c+\log(N/c))\), and linear activation memory.

### Theorem 6.3 — Decode cost

Maintaining Cholesky factor \(R_t^\top R_t\approx S_t+\varepsilon I\), per-token decode costs
\[
O(d_k^2+d_kd_v)
\]
operations and state \(d_k^2/2+d_kd_v+O(d_k)\), independent of context length.

### Lemma 6.4 — Regularization floor

Cycling phantom writes keep the floor diagonal entries within a few percent of \(\varepsilon\) after burn-in whenever \((1-\lambda)d_k\le1\). The prior precision is maintained forever without refactorization.

### Lemma 6.5 — Adjoint scan

The backward recurrence is also a scan:
\[
g_t=\partial\mathcal L/\partial z_t+\lambda_{t+1}g_{t+1}.
\]
Gradients through the read solve use the maintained factor at \(O(d_k^2)\) per token.

### Theorem 6.6 — Stability

With \(\|k_t\|=1\), \(\beta_t\le\beta_{\max}\), \(\lambda_t\le\lambda<1\):
\[
\|S_t\|\le\frac{\beta_{\max}}{1-\lambda},\qquad
\kappa(A_t)\le1+\frac{\beta_{\max}}{\varepsilon(1-\lambda)}.
\]
Gradients cannot explode; vanishing is governed by learned \(\lambda\). Floating-point read error is \(O(\kappa(A_t)u)\) per token.

### Ledger

| | softmax attention | Hebbian fast weights | nonlinear RNN | Mamba 4 | Mamba 4 cascade |
|---|---|---|---|---|---|
| training work | \(\Theta(N^2d)\) | \(\Theta(Nd^2)\) | \(\Theta(Nd^2)\) | \(\Theta(Nd^2)\) | \(\Theta(Nd^2)\) |
| training span | \(O(\log N)\) | \(O(c+\log N)\) | \(\Theta(N)\) | \(O(c+\log N)\) | \(O(c+\log N)\) |
| decode per token | \(\Theta(Nd)\) | \(\Theta(d^2)\) | \(\Theta(d^2)\) | \(\Theta(d^2)\) | \(\Theta(d^2)\) |
| state | \(\Theta(Nd)\) | \(\Theta(d^2)\) | \(\Theta(d)\)–\(\Theta(d^2)\) | \(\Theta(d^2)\) | \(\Theta(d^2\log N)\) |
| exact recall | all stored | 0 non-orthogonal | unprincipled | \(d\) per head, optimal | \(\Theta(d\log N)\) selected |
| confidence | none | none | none | calibrated | calibrated |
| mergeable states | cache concat | additive | no | additive | additive |

---

## 7. Hardware Execution

### Proposition 7.1 — Training is compute-bound

Per chunk/head, arithmetic intensity is \(\Theta(\min(c,d))\) FLOPs/word. With \(c\gtrsim\rho\) (machine balance), the kernel is compute-bound. All heavy operations are dense GEMMs, rank-one updates, and triangular solves; shapes are static; serial depth is \(c+\log(N/c)\).

### Proposition 7.2 — Decode traffic is context-free

Per layer/head, decode touches resident state \(d_k^2/2+d_kd_v\) and does \(O(d_k^2+d_kd_v)\) FLOPs per token. For \(d_k=d_v=64\), 16 heads, 48 layers, state is \(\approx25\) MB. Per-token latency is \(\Theta(\text{model size}/B)\), independent of \(N\).

---

## 8. Dyadic Conjugate Cascade

### Definition 8.1

Fix base span \(c_0\). Maintain a live discounted state over the current block. At each level \(\ell\), keep at most one frozen undiscounted block \((S^{(\ell)},C^{(\ell)})\) spanning \(2^\ell c_0\) tokens. When two blocks meet at level \(\ell\), merge by addition into level \(\ell+1\). Reads address live memory plus levels, gated by recency-aware learned gates.

### Theorem 8.2 — Maintenance is free

Over \(N\) tokens there are at most \(N/c_0\) freezes and \(N/c_0-1\) merges, each \(O(d_k^2+d_kd_v)\). Amortized per-token cost is \(O(d_k)\) for \(c_0\ge d_k\).

### Theorem 8.3 — Telescoped capacity

If the write policy marks at most \(d_k\) independent associations per dyadic block, every marked item is exactly retrievable from its block. Total exact recall:
\[
\Theta(d_k\log(N/c_0)).
\]
This is order-optimal for the state and scale-uniformly Shannon-efficient up to \(1+d_k/d_v\). Unmarked residue is retained as the block’s Bayes-optimal ridge posterior.

### Theorem 8.4 — Power-law forgetting

An item of age \(A\) lies in a block of span at most \(2A\), holding \(\approx2rA\) competitors at write rate \(r\). Exactness holds while \(rA\lesssim d_k/2\). Beyond that, risk grows polynomially in \(rA/d_k\), not exponentially as \(\lambda^A\). Refresh is a learnable rehearsal operation.

### Theorem 8.5 — Mergeability at scale

Partition a corpus into \(P\) segments. Blockwise sums of the \(P\) cascades equal the cascade of the full corpus under block-respecting interleaving. Communication is \(O(Pd^2\log N)\) total, with constant-size messages. Parallel ingestion and knowledge handoff are state addition.

### Proposition 8.6 — Routed reads

Reading all levels costs \(O((d_k^2+d_kd_v)\log N)\). Escalating only when confidence \(c_t(q)\) indicates ignorance gives expected \(O(d_k^2+d_kd_v)\) per token under light-tailed query-age distributions; a budget of \(R\) levels gives worst-case \(O(Rd^2)\).

---

## 9. Adversarial Self-Assessment

- **Adversarial recall:** Fails beyond budget, as all equal-state architectures must (Theorem 3.1). Mamba 4 meets the bound at optimal exchange rate and returns calibrated warning.
- **Key geometry:** Near-dependent keys amplify error, but capped by \(\varepsilon\); degradation converges to pseudo-inverse averaging, not garbage; confidence flags it.
- **Abelian ceiling:** One layer’s state is an order-weighted sum and cannot track non-commutative structure in one pass. This is the price of parallel training, shared by attention. Multi-hop reads add rounds.
- **\(d^2\) state:** Quadratic per head, but smaller than attention cache beyond short context. Sketches interpolate to Hebbian corner with quantifiable error.
- **Distributional fragility:** Optimality and calibration are in-model. Outside, algebraic guarantees remain; failure modes are ridge-regression failure modes.
- **Queries outside hypothesis class:** Universal. Any fixed-size sufficient state declares a class. Mamba 4 makes the class auditable and extensible.
- **No empirical claims:** Training dynamics, gate learning, and key geometry are open.

---

## 10. Adjacent Possible

1. **Conjugate family zoo:** Dirichlet–multinomial, Gamma–Poisson, Wishart, von Mises–Fisher heads inherit the scan, cascade, and mergeability.
2. **Learned conjugacy:** Feature maps define sufficient statistics, so representation learning is hypothesis-class learning.
3. **Memory as commodity:** Additive belief states can be sharded, pooled, shipped, and consolidated.
4. **Theory targets:** Rounds conservation law; tight constants for addressing overhead; cascade Pareto-optimality; beyond squared loss.
5. **Method:** Do not design the state. Declare what future computation must know; find the family with fixed-size sufficient statistics; let Bayes’ rule write the update. All learning belongs to the hypothesis class and the read.

---

## 11. Conclusion

The expressivity–efficiency frontier was three walls and one door. Exact memory costs linear state. Parallel training forbids nonlinear transitions. Fixed budgets cap exact capacity. But a state need not be a vector being mixed; it can be a belief being updated. Mamba 4 does this: state is minimal sufficient statistic; update is Bayes’ rule in natural coordinates, hence an associative scan; read is posterior predictive, hence interpolating, optimal, and self-aware; gates are drift and evidence precision; capacity meets its converse; decode is constant-time with a conditioning floor; the cascade gives logarithmic state with selectable exactness at every scale; and memories add.

What remains beyond Mamba 4 remains beyond every architecture. What lies within now has theorems.