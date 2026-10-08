import Mathlib.Tactic

/-! The bounded state transition does not imply bounded read or gate gradients. -/
namespace Mamba4

theorem evidence_invariant (x beta lambda : ℝ) (hl0 : 0 ≤ lambda)
    (hl1 : lambda < 1) (hx : x ≤ beta / (1 - lambda)) :
    lambda * x + beta ≤ beta / (1 - lambda) := by
  have hd : 0 < 1 - lambda := by linarith
  have hx' : x * (1 - lambda) ≤ beta := (le_div_iff₀ hd).mp hx
  apply (le_div_iff₀ hd).mpr
  nlinarith [mul_nonneg hl0 (sub_nonneg.mpr hx')]

theorem decay_product_nonexpansive (xs : List ℝ)
    (bounds : ∀ x ∈ xs, 0 ≤ x ∧ x ≤ 1) : 0 ≤ xs.prod ∧ xs.prod ≤ 1 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := bounds x (by simp)
    have hs := ih (by intro y hy; exact bounds y (by simp [hy]))
    simp only [List.prod_cons]
    constructor
    · exact mul_nonneg hx.1 hs.1
    · calc x * xs.prod ≤ 1 * 1 := mul_le_mul hx.2 hs.2 hs.1 (by norm_num)
           _ = 1 := by ring

theorem fixed_prior_discount_changes_precision (a eps lambda : ℝ) :
    lambda * (a - eps) + eps = lambda * a + (1 - lambda) * eps := by ring

end Mamba4
