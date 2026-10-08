import Mamba4.Scan
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib

/-! Weighted evidence matrices, preservation of PSD and a positive ridge floor. -/
namespace Mamba4
open Matrix

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m]

def outer (k : n → ℝ) : Matrix n n ℝ := fun i j => k i * k j

omit [DecidableEq n] in
theorem outer_psd (k : n → ℝ) : (outer k).PosSemidef := by
  let B : Matrix n Unit ℝ := fun i _ => k i
  have h := Matrix.posSemidef_self_mul_conjTranspose B
  have eq : outer k = B * Bᴴ := by
    ext i j
    change k i * k j = ∑ t : Unit, B i t * star (B j t)
    simp [B]
  rw [eq]
  exact h

omit [DecidableEq n] in
theorem evidence_step_psd (S : Matrix n n ℝ) (k : n → ℝ) (beta decay : ℝ)
    (hS : S.PosSemidef) (hb : 0 ≤ beta) (hl : 0 ≤ decay) :
    (decay • S + beta • outer k).PosSemidef :=
  (hS.smul hl).add ((outer_psd k).smul hb)

omit [Fintype n] in
theorem ridge_posDef (S : Matrix n n ℝ) (eps : ℝ)
    (hS : S.PosSemidef) (he : 0 < eps) : (S + eps • (1 : Matrix n n ℝ)).PosDef :=
  Matrix.PosDef.posSemidef_add hS (Matrix.PosDef.one.smul he)

omit [DecidableEq n] in
theorem confidence_nonnegative (P : Matrix n n ℝ) (q : n → ℝ)
    (hP : P.PosSemidef) : 0 ≤ q ⬝ᵥ (P *ᵥ q) := by
  simpa using hP.dotProduct_mulVec_nonneg q

omit [Fintype n] in
theorem fixed_floor_step (S : Matrix n n ℝ) (k : n → ℝ) (eps beta decay : ℝ) :
    (decay • S + beta • outer k) + eps • (1 : Matrix n n ℝ) =
      decay • (S + eps • (1 : Matrix n n ℝ)) + beta • outer k +
        ((1 - decay) * eps) • (1 : Matrix n n ℝ) := by
  ext i j
  simp [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  ring

theorem prior_aware_merge {V : Type*} [AddCommGroup V]
    (prior evidence₁ evidence₂ : V) :
    (prior + evidence₁) + (prior + evidence₂) - prior =
      prior + (evidence₁ + evidence₂) := by abel

end Mamba4
