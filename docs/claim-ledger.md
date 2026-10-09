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

## Phase-03 implementation contracts

| Concrete implementation | Scope and cost | Evidence |
|---|---|---|
| Learned cyclic Gaussian LM head | Constant learned per-head gates and positive learned epsilon, differentiating initialized prior and scheduled injection. Learned neural confidence is not out-of-model calibration. | `lm/models/mamba4.py`, dense forward/gate/epsilon gradient conformance tests. |
| Fixed-floor variable gates | Explicit full-rank prior injection and dense refactor fallback; distinct from cyclic semantics. | `gaussian_memory(..., floor="fixed")`, variable-gate and floor tests. |
| Fused prefill | Two-level chronological evidence scan plus factorization at each token; O(N d cubed) factor work. Does not realize the ideal compatible rank-update training work bound in original claim 6.2. | `lm/kernels/mamba4.py`, multiple-chunk dense-reference tests. |
| Cached Gaussian decode | Two quadratic cyclic rank updates; fixed-floor mode refactors. Full model also counts protected routing/maintenance and order state. | Cyclic reference, factor and full cached/prefill agreement tests. |
| Protected neural branch | Redundant QR banks retain up to configured a<=d independent anchors per eligible bank. All-bank geometry mixture is learned and has no exact-routing guarantee; explicit retained-ID routing has the conditional left-inverse contract. | CPU cascade retained-ID/reselection comparison, isolation and missing-ID tests. |
| Global background composition | All writes enter one discounted Gaussian state. Optional per-bank additive background merging is not implemented by this LM composition. | [Implementation contract](mamba4-implementation.md), separate Gaussian/QR state definitions. |

These implementation entries do not certify trained-model quality, device speed
or a completed phase-03/04 gate. Measured hardware and training evidence must
be recorded separately.

## Screen-60m-v2 contracts

| Claim | Scope and cost | Evidence |
|---|---|---|
| Arbitrary-gate fixed floor | For any nonnegative gates and precisions, `A_t = S_t + diag(floor)` satisfies `x^T A_t x >= min(floor) |x|^2`; latent variance lies in `[0, |q|^2/min(floor)]`. Constant gates are not required. | `selective_evidence_psd`, `diagonal_floor_bound`, `selective_precision_posDef`, `selective_floor_arbitrary_gates`, `selective_variance_bound`. |
| Exact selective read | An exact solve of the discounted design system minimizes the weighted ridge loss with per-coordinate penalty `floor_j`; it is a posterior mean only under the static model (no discount). | `diag_ridge_minimizes`, `selective_solve_minimizes`; dense-reference forward and all-input gradient tests. |
| Chunked computation | The chunk-boundary affine scan equals the sequential recurrence; chunk sizes 1–16 (with padding) agree with sequential dense solves within float32 tolerance. | `selective_step_is_affine`, `selective_scan_equals_sequential`; `lm/tests/test_mamba4_selective.py`. |
| Lane-major factor and reverse pass | Same SPD system for loop, blocked and unrolled schedules; reverse pass `dq = A^-1 g`, `dA = -sym(A^-1 g y^T)` reuses the factor; no explicit inverse. `O(d^3)` per token-head. | Backend parity, custom-derivative and blocked-gradient tests; TPU block timings. |
| Cached decode | Exact `O(d^3 + p d)` refactor per token; hybrid Mamba-3 layers use the pinned step recurrence and official parameter tree. | Decode-versus-prefill tests for pure, hybrid, rotary and wide-head models. |
| Hybrid composition | Four selective memory layers with fifteen unmodified official Mamba-3 layers, parameter-matched within 1%. The measured result belongs to this composition; it does not isolate the conjugate layer from the hybrid effect. | Parameter ledger, pilot table (iteration H), peer-source identity check. |
| Protected banks in v2 | Not used by the v2 language model; no trained protected-bank claim. Operator contracts above are unchanged. | `docs/mamba4-implementation.md`, v1 conformance evidence. |
| Screen outcome | Strict win established for this single seed: held-out NLL 3.4727 vs 3.4758 (Mamba-3) and 3.5352 (Transformer); paired interval vs Mamba-3 -0.0044 to -0.0017. | `lm/results/screen-60m-v2/audit.json`, `REPORT.md`. |
