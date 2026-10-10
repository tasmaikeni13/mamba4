# Complete audit of the original monograph

IDs refer to [the preserved original](mamba4-original.md). “Lean” names identify
checked statements with explicit hypotheses; they do not certify stronger
prose or an entire architecture. All named original results are included.

| Original claim | Verdict and correction | Evidence |
|---|---|---|
| 3.1 Exact recall state | Deterministic bit bound valid; finite states and complete decoding required. Randomized entropy bound needs error/coin assumptions. Real dimension is a different resource. | `Capacity.exact_bit_budget` (namespace `Mamba4`); written entropy derivation in mathematical analysis §8. |
| 3.2 No free parallelism | Generic circuit evaluation hardness does not forbid nonlinear special cases or require all efficient transitions to be linear. Specify growing state/input encoding. | Mathematical analysis §3; no Lean complexity-theory claim. |
| 3.3 Value-linear capacity | Valid for fixed keys and a linear encoder/read factoring the complete value identity; excludes extra memory and nonlinear encoders. | `value_linear_capacity`. |
| 3.5 Linearization | Valid sufficient statement on the states separated by a finite invariant function space; “exactly useful scannable transitions” converse is unsupported. | `invariant_space_realization`, `separating_features_injective`. |
| 4.1 Conjugate scan | Valid algebra; distinguish coordinate work from parallel time; discounts do not commute. | `combine_assoc`, `scan_equals_sequential`, `weighted_tempered_log_update`; prefix tests. |
| 4.2 Layer definition | Well-defined inference/operator template, not a trained validated architecture; gate/prior semantics must be explicit. | Mathematical analysis §§1–2; reference code. |
| 4.3 Sufficiency/minimality | Factorization gives sufficiency for specified model. Minimality needs regular full family and removal of redundant/ancillary components. | Mathematical analysis §4; literature, no Lean sufficiency theorem. |
| 4.4 Canonicity | Qualify PKD's regular IID/common-support/statistic assumptions; conjugate prior is not forced. Reject unconditional uniqueness. | Mathematical analysis §4 and cited primary teaching notes. |
| 4.5 Mergeability | Add evidence or subtract one prior. Unordered addition wrong for discounted streams. Exclusivity against all attention/RNNs false. | `prior_aware_merge`, `posterior_kernel_merge`, merge tests. |
| 4.6 Derived gates | Precision interpretation valid for specified Gaussian noise. Tempering and fixed-prior discount are different; no universal maximum-entropy Bayes claim. | `fixed_floor_step`; mathematical analysis §2. |
| 5.1 Gaussian memory | Weighted ridge identity valid; no-discount posterior interpretation specified. Fixed-floor discount requires anchoring. | `ridge_normal_equation`, `ridge_loss_difference`, `ridge_minimizes`, `ridge_posDef`, and end-to-end `weighted_ridge_read_minimizes`. |
| 5.2 Interpolation | Bound valid under independent keys and positive weights; exactness is a limit. Protected QR gives exact real-arithmetic interpolation without ridge. | `positive_ridge_not_exact`, `independent_keys_interpolate`, `exact_interpolation_left_inverse`; weighted-bound Monte Carlo. Spectral bound is derived, not Lean-proved. |
| 5.3 Whitened attention | Valid signed value-linear coefficients, not necessarily probabilities; solving cost counts. | `read_linear_functional`; mathematical analysis §5. |
| 5.4 Optimal read | Bayes under the static model; BLUE at zero ridge for estimable queries and correctly weighted independent noise. Finite ridge adds bias. | `posterior_mean_minimizes` finite-distribution contract; analytic Gaussian/BLUE derivations and repeated-write/risk suites. |
| 5.5 Smoothing versus solving | Convex-hull limitation valid for one operator. Collision and unequal-multiplicity impossibility claims false; sharp softmax distinguishes distinct keys and log precision weights implement optimal averaging. | `smoother_inverse_variance_example`, tests; functional and repeated-write suites. |
| 5.6 Calibrated introspection | Exact under stated model. On unwritten q use norm(q)^2/epsilon. No model-free calibration/conflict warning guarantee. | `confidence_variational`, `confidence_monotone`, `duplicate_confidence_can_be_small`; calibration suite. |
| 5.7 Regression risk | Original S covariance wrong for general weights/discount; replace by sum a_i^2 tau_i^2 k_i k_i^T and retain bias. c is not total frequentist risk. | `ridge_bias_identity`, `squared_error_decomposition`; three conditional-risk experiments. |
| 5.8 Multi-hop | Exact zero-ridge code chase valid. Unit outputs do not imply Lipschitz <=1; keep true L, and count H sequential reads. | `hop_error_bound`, `envelope_geometric_sum`, `noninjective_graph_expands`; graph suite. |
| 5.9 Capacity optimality | Value-linear capacity achieved by unregularized/protected interpolation. Positive ridge not exact; no Shannon bit-efficiency from real-word dimension counts. | `value_linear_capacity`, `independent_keys_interpolate`; mathematical analysis §8. |
| 6.1 Scan element | Valid with chronological combine and explicit initial evidence/prior. | `Scan.lean`; sequential/tree prefix tests. |
| 6.2 Training cost | Include chunk-boundary cubic factorizations: N(d^2+pd)+(N/c)d^3. Quadratic work requires c>=d and a compatible rank-update model. | Algorithm/work derivation §11; no Lean complexity or TPU benchmark claim. |
| 6.3 Decode cost | Quadratic for no discount or repaired cyclic anisotropic prior. Arbitrary fixed isotropic-floor forgetting not established at this cost. | `fixed_floor_step`, `Floor.lean`; Cholesky conformance tests. |
| 6.4 Phantom floor | Original condition permits errors of tens of percent. Replace with stationary initialized cyclic prior with explicit min/max. | `phantom_rule_not_few_percent`, `cyclic_prior_floor`, `cyclic_prior_ceiling`; floor suite. |
| 6.5 Adjoint scan | State adjoint valid; complete gate/encoder/solve/QR gradients need separate treatment. | Scan algebra; `gradients.py` all-input finite-difference test. |
| 6.6 Stability | Evidence bound valid from bounded initial state/keys/weights. State Jacobian nonexpansive; solve/read/parameter gradients can grow as inverse epsilon. | `evidence_invariant`, `decay_product_nonexpansive`; conditioning/gradient diagnostics. |
| 7.1 Compute-bound training | Hardware hypothesis, not theorem. GEMMs, rank updates and solves have different intensity and overhead. | Work ledger §11; later kernel/roofline phase remains pending. |
| 7.2 Decode traffic | Context-independent fixed-head state; correct dtype/packing arithmetic and include buffers. Latency cannot be inferred from a word count. | Exact accounting §11; future device measurements. |
| 8.1 Dyadic definition | Binary counter implementable, but merged blocks do not preserve all child anchors' exactness. Repair: protected redundant banks plus separate backgrounds. | `ProtectedCascade`, cascade tests and Monte Carlo. |
| 8.2 Maintenance free | Additive background merge amortization valid. Protected reselection/QR costs cubic per bank, so new maintenance bound differs. | Count invariant and work derivation §12. |
| 8.3 Telescoped capacity | Original fails at merges and powers of two; residue contaminates exact anchors. Redundant protected banks provide logarithmic retained capacity under full-rank per-block eligibility. | `carry_preserves_mass`, `redundant_counter_mass_bounds`, `occupied_bank_bounds`; retained/evicted IDs and cascade results. |
| 8.4 Power-law forgetting | No general age-only risk bound follows; selector, rank, noise and drift matter. Retract universal claim. | Counterexample/explanation §12; no replacement invented theorem. |
| 8.5 Scale mergeability | Fixed aligned backgrounds add. Anchor selection not generally associative; routing/block metadata and message sizes count. | Merge tests; mathematical analysis §12. |
| 8.6 Routed reads | All-bank work has log factor. Constant expected cost requires proved query/routing/tail assumptions, not just “confidence.” | Conflict experiment and §12; learned routing pending. |
| Abstract, thesis, conclusion | Replace universal dominance/dissolution with conditional operator guarantees and measured limitations. | Corrected paper and full report. |
| Complexity ledger | Attention lookup, learned nonlinearity, confidence variants and mergeable competitors were overstated; distinguish full-cache state from equal-state budgets. | Corrected paper resource table; mathematical analysis §§8,11. |
| Adversarial self-assessment | Low c cannot warn about every overload; order limitations are specific to encoders/no discount, not attention/parallelism. | Conflict, order and overload witnesses. |
| Adjacent possible | Other conjugate heads and learned representations remain proposals; inherited floor/routing/bit guarantees need family-specific proof. | Future phases explicitly retain those requirements. |

`Capacity.exact_bit_budget` above denotes the theorem in Capacity.lean; the
qualified Lean name is `Mamba4.exact_bit_budget`. All remaining listed names
also live in `Mamba4`, except scan names in `Mamba4.AffineSummary`.

## Implementation contracts

| Concrete implementation | Scope and cost | Evidence |
|---|---|---|
| Arbitrary-gate fixed floor | For any nonnegative gates and precisions, `A_t = S_t + diag(floor)` satisfies `x^T A_t x >= min(floor) |x|^2`; latent variance lies in `[0, |q|^2/min(floor)]`. No constant-gate assumption is needed. | `selective_evidence_psd`, `diagonal_floor_bound`, `selective_precision_posDef`, `selective_floor_arbitrary_gates`, `selective_variance_bound`. |
| Retention under the floor | One write of discounted weight `w` is read back as `w / (w + f)` of its value under an isotropic floor `f`; at most half survives once `w <= f`. | `single_write_solve`, `single_write_read`, `single_write_half`. |
| Exact selective read | An exact solve minimizes the discounted weighted ridge loss with per-coordinate penalty `floor_j`; it is a posterior mean only under the static model without discount. | `diag_ridge_minimizes`, `selective_solve_minimizes`; dense-reference forward and all-input gradient tests. |
| Chunked computation | The chunk-boundary affine scan equals the sequential recurrence; chunk sizes 1–16, with padding, agree with sequential dense solves. | `selective_step_is_affine`, `selective_scan_equals_sequential`; `lm/tests/test_mamba4.py`. |
| Factor and reverse pass | The loop, blocked and unrolled schedules factor the same SPD system; the reverse pass `dq = A^-1 g`, `dA = -sym(A^-1 g y^T)` reuses the factor. `O(d^3)` per token-head; no explicit inverse. | Backend parity, custom-derivative and gradient tests; kernel timings in `lm/results/kernels/`. |
| Key alignment | Each value is written under the key of the preceding position, and queries use the current one; no parameters are added. | Causality, gradient and decode-versus-prefill tests with the shift. |
| Cached decode | Exact `O(d^3 + p d)` refactor per token; Mamba-3 layers use the pinned official step recurrence; the cache size does not depend on context length. | Decode-versus-prefill tests; decode timings to 65,536 tokens. |
| Hybrid composition | Four memory layers with fifteen unmodified official Mamba-3 layers, matched within 1% in parameters. Results belong to this composition, not to the memory layer alone. | Parameter ledger, pilots, peer source-identity checks. |
| Protected banks | Not part of the language model; no trained protected-bank claim. The operator contracts above them are unchanged. | Phase-02 operator studies. |

## Trained evaluation (60M, 1B tokens, one seed)

| Claim | Result | Evidence |
|---|---|---|
| Language modelling | Held-out NLL 3.4719, against 3.4758 (Mamba-3) and 3.5352 (Transformer). Paired 95% interval against Mamba-3 is −0.0052 to −0.0025; the fresh holdout gives −0.0059 to −0.0031. The strict screen win holds for this seed. | `lm/results/screen-60m/` |
| Exact retrieval inside the window | Refuted against the Transformer: copying 16 random words, 11.8% against 42.3%. Supported against Mamba-3 (5.7%). Frozen 8-way recall diagnostic: 69/128, against 18 and 30. | `lm/results/claims-60m/`, `docs/trained-recall.md` |
| Long context | Refuted: no passkey retrieved at any length (the Transformer reaches 68.8% at 512 tokens and 0% beyond 1,024). Long-document NLL is the lowest of the three at every position beyond 128, but it does not improve beyond the 1,024-token training length. | `lm/results/claims-60m/` |
| Calibrated confidence | Refuted under the frozen rule (answer-slot ECE is worse on passkey prompts). The memory variance is not a consistent predictor of errors. | `lm/results/claims-60m/` |
| Context-independent decode | Supported: 5.3 ms per token from 1K to 64K context (7.3 MiB cache per sequence), against 4.6 ms rising to 904 ms (22.9 MiB rising to 1.4 GiB) for the Transformer. | `lm/results/claims-60m/` |
| Mechanism under direct training | Supported in small models trained on the tasks: exact associative recall up to the key dimension (87.5% at four times it), exact retention of eight pairs to 16 times the training length, exact two-hop lookups, and in-context regression within 0.04 nats of the exact Bayes predictive NLL. Not supported for absence detection: Mamba-3 separates never-stored keys better. Not shown inside the 60M language model. | `lm/results/synthetic/`, `lm/results/regression/` |
