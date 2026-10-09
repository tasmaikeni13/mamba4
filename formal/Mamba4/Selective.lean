import Mamba4.Evidence
import Mamba4.Scan
import Mathlib

/-! Selective fixed-floor memory used by the screen-60m-v2 language model.

Token drift gates and evidence precisions are arbitrary nonnegative reals. The
prior precision is a diagonal floor that is never discounted, so the precision
is `S_t + diagonal floor`. These results hold for every gate sequence; they do
not depend on the constant-gate cyclic schedule in `Floor.lean`. -/
namespace Mamba4
open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- One write: `S ↦ decay • S + beta • k kᵀ`. -/
def selectiveStep (S : Matrix n n ℝ) (k : n → ℝ) (beta decay : ℝ) : Matrix n n ℝ :=
  decay • S + beta • outer k

/-- Evidence after a chronological list of `(key, precision, decay)` writes. -/
def selectiveEvidence (writes : List ((n → ℝ) × ℝ × ℝ)) (S : Matrix n n ℝ) :
    Matrix n n ℝ :=
  writes.foldl (fun state w => selectiveStep state w.1 w.2.1 w.2.2) S

omit [DecidableEq n] in
theorem selective_evidence_psd (writes : List ((n → ℝ) × ℝ × ℝ))
    (S : Matrix n n ℝ) (hS : S.PosSemidef)
    (hw : ∀ w ∈ writes, 0 ≤ w.2.1 ∧ 0 ≤ w.2.2) :
    (selectiveEvidence writes S).PosSemidef := by
  induction writes generalizing S with
  | nil => simpa [selectiveEvidence] using hS
  | cons w rest ih =>
    have hw0 := hw w (List.mem_cons_self ..)
    have step : (selectiveStep S w.1 w.2.1 w.2.2).PosSemidef :=
      evidence_step_psd S w.1 w.2.1 w.2.2 hS hw0.1 hw0.2
    have tail : ∀ v ∈ rest, 0 ≤ v.2.1 ∧ 0 ≤ v.2.2 :=
      fun v hv => hw v (List.mem_cons_of_mem _ hv)
    simpa [selectiveEvidence, List.foldl_cons] using ih _ step tail

/-- The undiscounted diagonal floor bounds the precision quadratic form. -/
theorem diagonal_floor_bound (S : Matrix n n ℝ) (hS : S.PosSemidef)
    (floor : n → ℝ) (eps : ℝ) (hf : ∀ i, eps ≤ floor i) (x : n → ℝ) :
    eps * (x ⬝ᵥ x) ≤ x ⬝ᵥ ((S + Matrix.diagonal floor) *ᵥ x) := by
  have hpsd : 0 ≤ x ⬝ᵥ (S *ᵥ x) := by
    simpa using hS.dotProduct_mulVec_nonneg x
  have hdiag : eps * (x ⬝ᵥ x) ≤ x ⬝ᵥ (Matrix.diagonal floor *ᵥ x) := by
    simp only [dotProduct, Matrix.mulVec_diagonal, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    nlinarith [hf i, mul_self_nonneg (x i)]
  rw [Matrix.add_mulVec, dotProduct_add]
  linarith

omit [Fintype n] in
theorem selective_precision_posDef (S : Matrix n n ℝ) (hS : S.PosSemidef)
    (floor : n → ℝ) (hf : ∀ i, 0 < floor i) :
    (S + Matrix.diagonal floor).PosDef :=
  Matrix.PosDef.posSemidef_add hS (Matrix.PosDef.diagonal hf)

/-- Arbitrary nonnegative gates keep every precision above the floor. -/
theorem selective_floor_arbitrary_gates (writes : List ((n → ℝ) × ℝ × ℝ))
    (hw : ∀ w ∈ writes, 0 ≤ w.2.1 ∧ 0 ≤ w.2.2)
    (floor : n → ℝ) (eps : ℝ) (hf : ∀ i, eps ≤ floor i) (x : n → ℝ) :
    eps * (x ⬝ᵥ x) ≤
      x ⬝ᵥ ((selectiveEvidence writes 0 + Matrix.diagonal floor) *ᵥ x) :=
  diagonal_floor_bound _ (selective_evidence_psd writes 0 PosSemidef.zero hw)
    floor eps hf x

omit [DecidableEq n] in
/-- If `A y = q` and `A` dominates `eps I`, the latent variance `q·y` lies in
`[0, q·q / eps]`. This is the unwritten-direction value `‖q‖² / eps` bound. -/
theorem selective_variance_bound (A : Matrix n n ℝ) (q y : n → ℝ) (eps : ℝ)
    (he : 0 < eps) (hA : ∀ x, eps * (x ⬝ᵥ x) ≤ x ⬝ᵥ (A *ᵥ x))
    (solves : A *ᵥ y = q) :
    0 ≤ q ⬝ᵥ y ∧ q ⬝ᵥ y ≤ (q ⬝ᵥ q) / eps := by
  have hq : q ⬝ᵥ y = y ⬝ᵥ (A *ᵥ y) := by rw [solves, dotProduct_comm]
  have hy := hA y
  have hyy : 0 ≤ y ⬝ᵥ y := by
    simp only [dotProduct]
    exact Finset.sum_nonneg (fun i _ => mul_self_nonneg (y i))
  have hsq : 0 ≤ (q - eps • y) ⬝ᵥ (q - eps • y) := by
    simp only [dotProduct]
    exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)
  have expand : (q - eps • y) ⬝ᵥ (q - eps • y) =
      q ⬝ᵥ q - 2 * eps * (q ⬝ᵥ y) + eps ^ 2 * (y ⬝ᵥ y) := by
    simp only [sub_dotProduct, dotProduct_sub, smul_dotProduct, dotProduct_smul,
      smul_eq_mul, dotProduct_comm y q]
    ring
  constructor
  · nlinarith
  · rw [le_div_iff₀ he]
    nlinarith

/-- Weighted ridge loss with a per-coordinate (anisotropic) floor penalty. -/
def diagRidgeLoss {ι : Type*} [Fintype ι] (keys : ι → n → ℝ)
    (values weights : ι → ℝ) (floor : n → ℝ) (w : n → ℝ) : ℝ :=
  (∑ i, weights i * ((∑ j, w j * keys i j) - values i) ^ 2) +
    ∑ j, floor j * (w j) ^ 2

omit [DecidableEq n] in
theorem diag_ridge_minimizes {ι : Type*} [Fintype ι] (keys : ι → n → ℝ)
    (values weights : ι → ℝ) (floor : n → ℝ) (w : n → ℝ)
    (hb : ∀ i, 0 ≤ weights i) (hf : ∀ j, 0 ≤ floor j)
    (normal : ∀ j, (∑ i, weights i * ((∑ l, w l * keys i l) - values i) * keys i j)
      + floor j * w j = 0) (candidate : n → ℝ) :
    diagRidgeLoss keys values weights floor w ≤
      diagRidgeLoss keys values weights floor candidate := by
  set h := candidate - w with hh
  have hc : candidate = w + h := by rw [hh]; abel
  have cross :
      (∑ i, weights i * ((∑ l, w l * keys i l) - values i) *
        (∑ j, h j * keys i j)) + ∑ j, floor j * (w j * h j) = 0 := by
    calc
      _ = ∑ j, h j * ((∑ i, weights i * ((∑ l, w l * keys i l) - values i) *
          keys i j) + floor j * w j) := by
        simp only [mul_add, Finset.sum_add_distrib, Finset.mul_sum]
        rw [Finset.sum_comm]
        congr 1 <;> apply Finset.sum_congr rfl <;> intro j _
        · apply Finset.sum_congr rfl; intro i _; ring
        · ring
      _ = 0 := by simp [normal]
  have expand_obs : ∀ i,
      weights i * ((∑ j, (w + h) j * keys i j) - values i) ^ 2 =
      weights i * ((∑ j, w j * keys i j) - values i) ^ 2 +
      weights i * (∑ j, h j * keys i j) ^ 2 +
      2 * (weights i * ((∑ j, w j * keys i j) - values i) *
        (∑ j, h j * keys i j)) := by
    intro i
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
    ring
  have expand_reg : ∀ j, floor j * ((w + h) j) ^ 2 =
      floor j * (w j) ^ 2 + floor j * (h j) ^ 2 + 2 * (floor j * (w j * h j)) := by
    intro j; simp only [Pi.add_apply]; ring
  have ho : 0 ≤ ∑ i, weights i * (∑ j, h j * keys i j) ^ 2 :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hb i) (sq_nonneg _))
  have hr : 0 ≤ ∑ j, floor j * (h j) ^ 2 :=
    Finset.sum_nonneg (fun j _ => mul_nonneg (hf j) (sq_nonneg _))
  rw [hc]
  simp only [diagRidgeLoss, expand_obs, expand_reg, Finset.sum_add_distrib,
    ← Finset.mul_sum]
  nlinarith [cross]

/-- The selective design precision: weighted evidence plus the diagonal floor. -/
noncomputable def selectiveDesign {ι : Type*} [Fintype ι] (keys : ι → n → ℝ)
    (weights : ι → ℝ) (floor : n → ℝ) : Matrix n n ℝ :=
  (∑ t, weights t • outer (keys t)) + Matrix.diagonal floor

/-- An exact solve of the selective design system satisfies the normal
equations of `diagRidgeLoss`, hence minimizes it. -/
theorem selective_solve_minimizes {ι : Type*} [Fintype ι] (keys : ι → n → ℝ)
    (values weights : ι → ℝ) (floor : n → ℝ) (w : n → ℝ)
    (hb : ∀ t, 0 ≤ weights t) (hf : ∀ j, 0 ≤ floor j)
    (solves : selectiveDesign keys weights floor *ᵥ w =
      ∑ t, (weights t * values t) • keys t)
    (candidate : n → ℝ) :
    diagRidgeLoss keys values weights floor w ≤
      diagRidgeLoss keys values weights floor candidate := by
  apply diag_ridge_minimizes keys values weights floor w hb hf _ candidate
  intro j
  have h := congrFun solves j
  have ho : ∀ t, (outer (keys t) *ᵥ w) j =
      keys t j * ∑ l, w l * keys t l := by
    intro t
    simp only [outer, Matrix.mulVec, dotProduct, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro l _
    ring
  simp only [selectiveDesign, Matrix.add_mulVec, Matrix.sum_mulVec,
    Matrix.smul_mulVec, Matrix.mulVec_diagonal, Pi.add_apply, Finset.sum_apply,
    Pi.smul_apply, smul_eq_mul, ho] at h
  have rearrange :
      (∑ t, weights t * ((∑ l, w l * keys t l) - values t) * keys t j) =
      (∑ t, weights t * (keys t j * ∑ l, w l * keys t l)) -
      (∑ t, (weights t * values t) * keys t j) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [rearrange]
  linarith

omit [Fintype n] [DecidableEq n] in
/-- A selective write is the action of the affine summary `(decay, beta k kᵀ)`;
hence the chunked associative scan equals the sequential recurrence. -/
theorem selective_step_is_affine (S : Matrix n n ℝ) (k : n → ℝ) (beta decay : ℝ) :
    selectiveStep S k beta decay =
      AffineSummary.act ⟨decay, beta • outer k⟩ S := by
  simp [selectiveStep, AffineSummary.act]

omit [Fintype n] [DecidableEq n] in
theorem selective_scan_equals_sequential (writes : List ((n → ℝ) × ℝ × ℝ))
    (S : Matrix n n ℝ) :
    AffineSummary.act
        (AffineSummary.summarize
          (writes.map (fun w => (⟨w.2.2, w.2.1 • outer w.1⟩ :
            AffineSummary (Matrix n n ℝ))))) S =
      selectiveEvidence writes S := by
  rw [AffineSummary.scan_equals_sequential]
  induction writes generalizing S with
  | nil => rfl
  | cons w rest ih =>
    simp only [List.map_cons, List.foldl_cons, selectiveEvidence]
    rw [← selective_step_is_affine]
    exact ih _

end Mamba4
