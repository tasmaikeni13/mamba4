import Mamba4.Evidence
import Mathlib

/-! Retention of one write under an undiscounted isotropic floor.

A fact written once with precision `beta` and then discounted by `Λ` holds
evidence `w = beta * Λ`. With an isotropic floor `f`, the exact solve for its
unit key is `k / (w + f)`, so the exact read returns `w / (w + f)` times the
stored value: once the discounted weight falls to the floor, at most half of
the value survives. This is the retention mechanism diagnosed in the trained
selective memory on passkey prompts. -/
namespace Mamba4
open Matrix

variable {n p : Type*} [Fintype n] [DecidableEq n] [Fintype p]

omit [DecidableEq n] in
/-- A rank-one evidence matrix acts as `(k ⬝ᵥ v) • k`. -/
theorem outer_mulVec (k v : n → ℝ) : outer k *ᵥ v = (k ⬝ᵥ v) • k := by
  ext i
  simp only [outer, Matrix.mulVec, dotProduct, Pi.smul_apply, smul_eq_mul,
    Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- For a unit key, `(w kkᵀ + f I) y = k` is solved by `y = k / (w + f)`. -/
theorem single_write_solve (k : n → ℝ) (hk : k ⬝ᵥ k = 1) (w f : ℝ)
    (hwf : w + f ≠ 0) :
    (w • outer k + f • (1 : Matrix n n ℝ)) *ᵥ ((w + f)⁻¹ • k) = k := by
  rw [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.smul_mulVec, outer_mulVec,
    Matrix.one_mulVec, dotProduct_smul, hk]
  ext i
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, mul_one]
  field_simp

omit [DecidableEq n] [Fintype p] in
/-- The exact read of the stored value is shrunk by `w / (w + f)`. -/
theorem single_write_read (k : n → ℝ) (hk : k ⬝ᵥ k = 1) (v : p → ℝ) (w f : ℝ) :
    (w • Matrix.vecMulVec v k) *ᵥ ((w + f)⁻¹ • k) = (w / (w + f)) • v := by
  rw [Matrix.smul_mulVec]
  have h : Matrix.vecMulVec v k *ᵥ ((w + f)⁻¹ • k) = ((w + f)⁻¹ * (k ⬝ᵥ k)) • v := by
    ext i
    simp only [Matrix.mulVec, dotProduct, Matrix.vecMulVec_apply, Pi.smul_apply,
      smul_eq_mul, Finset.mul_sum, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro x _
    ring
  rw [h, hk]
  ext i
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

/-- Once the discounted weight is at most the floor, at most half survives. -/
theorem single_write_half (w f : ℝ) (hw : 0 ≤ w) (hf : 0 < f) (hwf : w ≤ f) :
    w / (w + f) ≤ 1 / 2 := by
  rw [div_le_div_iff₀ (by linarith) (by norm_num)]
  linarith

end Mamba4
