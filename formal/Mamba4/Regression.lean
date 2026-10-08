import Mamba4.Evidence
import Mamba4.Ridge

/-! End-to-end weighted ridge theorem: build evidence from observations, prove
the precision invertible, solve it, derive scalar normal equations, and prove
the actual returned coefficients minimize the declared objective. -/
namespace Mamba4
open Matrix
variable {ι n : Type*} [Fintype ι] [Fintype n] [DecidableEq n]

noncomputable def designPrecision (keys : ι → n → ℝ) (weights : ι → ℝ)
    (eps : ℝ) : Matrix n n ℝ :=
  (∑ t, weights t • outer (keys t)) + eps • (1 : Matrix n n ℝ)

noncomputable def designRhs (keys : ι → n → ℝ) (values weights : ι → ℝ) : n → ℝ :=
  ∑ t, (weights t * values t) • keys t

theorem design_precision_posDef (keys : ι → n → ℝ) (weights : ι → ℝ)
    (eps : ℝ) (hb : ∀ t, 0 ≤ weights t) (he : 0 < eps) :
    (designPrecision keys weights eps).PosDef := by
  have hS : (∑ t, weights t • outer (keys t)).PosSemidef := by
    simpa using Matrix.posSemidef_sum Finset.univ
      (fun t _ => (outer_psd (keys t)).smul (hb t))
  exact ridge_posDef _ eps hS he

theorem solve_implies_normal (keys : ι → n → ℝ) (values weights : ι → ℝ)
    (eps : ℝ) (w : n → ℝ)
    (solves : designPrecision keys weights eps *ᵥ w = designRhs keys values weights) :
    ∀ j, (∑ t, weights t * ((∑ l, w l * keys t l) - values t) * keys t j)
      + eps * w j = 0 := by
  intro j
  have h := congrFun solves j
  have ho : ∀ t, (outer (keys t) *ᵥ w) j =
      keys t j * ∑ l, w l * keys t l := by
    intro t
    simp only [outer, Matrix.mulVec, dotProduct, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro l hl
    ring
  simp only [designPrecision, designRhs, Matrix.add_mulVec, Matrix.sum_mulVec,
    Matrix.smul_mulVec, Matrix.one_mulVec, Pi.add_apply, Finset.sum_apply,
    Pi.smul_apply, smul_eq_mul, ho] at h
  have rearrange :
      (∑ t, weights t * ((∑ l, w l * keys t l) - values t) * keys t j) =
      (∑ t, weights t * (keys t j * ∑ l, w l * keys t l)) -
      (∑ t, (weights t * values t) * keys t j) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro t ht
    ring
  rw [rearrange]
  linarith

theorem weighted_ridge_read_minimizes (keys : ι → n → ℝ)
    (values weights : ι → ℝ) (eps : ℝ) (hb : ∀ t, 0 ≤ weights t)
    (he : 0 < eps) (candidate : n → ℝ) :
    ridgeLoss keys values weights eps
      ((designPrecision keys weights eps)⁻¹ *ᵥ designRhs keys values weights) ≤
        ridgeLoss keys values weights eps candidate := by
  let A := designPrecision keys weights eps
  have hA := design_precision_posDef keys weights eps hb he
  have hd : IsUnit A.det := (Matrix.isUnit_iff_isUnit_det A).mp hA.isUnit
  have solves : A *ᵥ (A⁻¹ *ᵥ designRhs keys values weights) =
      designRhs keys values weights := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv A hd, Matrix.one_mulVec]
  exact ridge_minimizes keys values weights eps _ hb he.le
    (solve_implies_normal keys values weights eps _ solves) candidate

end Mamba4
