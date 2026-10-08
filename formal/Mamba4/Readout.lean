import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Tactic
import Mathlib

/-! Matrix identities underpinning interpolation, ridge bias and linear queries. -/
namespace Mamba4
open Matrix
variable {n m k : Type*} [Fintype n] [DecidableEq n] [Fintype m]
  [Fintype k] [DecidableEq k]

omit [Fintype m] in
theorem ridge_normal_equation (C : Matrix m n ℝ) (A : Matrix n n ℝ)
    (hA : IsUnit A.det) : (C * A⁻¹) * A = C :=
  A.nonsing_inv_mul_cancel_right C hA

omit [DecidableEq n] [Fintype m] in
theorem exact_interpolation_left_inverse (K : Matrix n k ℝ) (V : Matrix m k ℝ)
    (L : Matrix k n ℝ) (hLK : L * K = 1) : (V * L) * K = V := by
  rw [Matrix.mul_assoc, hLK, Matrix.mul_one]

omit [DecidableEq n] [Fintype m] in
theorem gram_interpolation (K : Matrix n k ℝ) (V : Matrix m k ℝ)
    (hG : IsUnit (Kᵀ * K).det) : (V * (Kᵀ * K)⁻¹ * Kᵀ) * K = V := by
  calc
    _ = V * ((Kᵀ * K)⁻¹ * (Kᵀ * K)) := by simp only [Matrix.mul_assoc]
    _ = V := by rw [Matrix.nonsing_inv_mul _ hG, Matrix.mul_one]

omit [Fintype m] in
theorem ridge_bias_identity (W : Matrix m n ℝ) (S : Matrix n n ℝ) (eps : ℝ)
    (hA : IsUnit (S + eps • (1 : Matrix n n ℝ)).det) :
    (W * S) * (S + eps • (1 : Matrix n n ℝ))⁻¹ - W =
      -eps • (W * (S + eps • (1 : Matrix n n ℝ))⁻¹) := by
  let A := S + eps • (1 : Matrix n n ℝ)
  have h : W * A * A⁻¹ = W := Matrix.mul_nonsing_inv_cancel_right A W hA
  dsimp [A] at h
  rw [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_one, Matrix.add_mul,
    Matrix.smul_mul] at h
  rw [neg_smul]
  have hh := eq_sub_of_add_eq h
  calc
    _ = (W - eps • (W * (S + eps • (1 : Matrix n n ℝ))⁻¹)) - W := by rw [hh]
    _ = _ := by abel

omit [DecidableEq n] [Fintype m] in
theorem read_linear_functional (M : Matrix m n ℝ) (q r : n → ℝ) (a b : ℝ) :
    M *ᵥ (a • q + b • r) = a • (M *ᵥ q) + b • (M *ᵥ r) := by
  simp [Matrix.mulVec_add, Matrix.mulVec_smul]

omit [DecidableEq n] in
theorem independent_keys_gram_invertible (K : Matrix n k ℝ)
    (independent : Function.Injective K.mulVec) : IsUnit (Kᵀ * K).det := by
  have hp := Matrix.PosDef.conjTranspose_mul_self K independent
  have he : Kᴴ = Kᵀ := by ext i j; simp
  rw [he] at hp
  exact (Matrix.isUnit_iff_isUnit_det _).mp hp.isUnit

omit [DecidableEq n] [Fintype m] in
theorem independent_keys_interpolate (K : Matrix n k ℝ) (V : Matrix m k ℝ)
    (independent : Function.Injective K.mulVec) :
    (V * (Kᵀ * K)⁻¹ * Kᵀ) * K = V :=
  gram_interpolation K V (independent_keys_gram_invertible K independent)

end Mamba4
