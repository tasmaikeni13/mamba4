import Mathlib.Tactic

/-! Kernel-checked witnesses against unconditional claims in the original paper. -/
namespace Mamba4

theorem positive_ridge_not_exact : (1 : ℚ) / (1 + 1) ≠ 1 := by norm_num

theorem duplicate_confidence_can_be_small :
    (1 : ℚ) / (100 + 1) < 1 / 100 ∧
    (100 * (0 + 2) / 2 : ℚ) / (100 + 1) ≠ 0 := by norm_num

theorem noninjective_graph_expands :
    ((1 : ℚ) ^ 2 + 1 ^ 2) > 1 ^ 2 := by norm_num

theorem phantom_rule_not_few_percent :
    ((64 : ℚ) * (1 - 99 / 100) / (1 - (99 / 100) ^ 64)) > 13 / 10 ∧
    (1 - 99 / 100 : ℚ) * 64 ≤ 1 := by norm_num

theorem discount_combine_noncommutative :
    (1 + (1 / 2 : ℚ) * 2) ≠ (2 + (1 / 2 : ℚ) * 1) := by norm_num

theorem smoother_inverse_variance_example :
    (1 / 4 : ℚ) * 1 + (3 / 4 : ℚ) * 3 = 5 / 2 := by norm_num

theorem duplicate_anchor_inconsistent :
    ¬ ∃ (f : ℚ → ℚ), f 1 = 0 ∧ f 1 = 2 := by
  rintro ⟨f, h₁, h₂⟩
  linarith

theorem merge_overload_witness : (2 : ℕ) + 2 > 2 := by omega

end Mamba4
