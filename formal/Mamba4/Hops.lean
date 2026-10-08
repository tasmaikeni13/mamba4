import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Tactic

/-! Multi-hop perturbation recurrence with the actual Lipschitz constant. -/
namespace Mamba4

def errorEnvelope (L eps : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => eps + L * errorEnvelope L eps n

theorem hop_error_bound (L eps : ℝ) (e : ℕ → ℝ) (hL : 0 ≤ L)
    (initial : e 0 ≤ 0) (step : ∀ n, e (n + 1) ≤ eps + L * e n) :
    ∀ n, e n ≤ errorEnvelope L eps n := by
  intro n
  induction n with
  | zero => exact initial
  | succ n ih =>
    calc e (n + 1) ≤ eps + L * e n := step n
         _ ≤ eps + L * errorEnvelope L eps n :=
           by linarith [mul_le_mul_of_nonneg_left ih hL]

theorem envelope_geometric_sum (L eps : ℝ) (n : ℕ) :
    errorEnvelope L eps n = eps * ∑ j ∈ Finset.range n, L ^ j := by
  induction n with
  | zero => simp [errorEnvelope]
  | succ n ih =>
    rw [errorEnvelope, ih, Finset.sum_range_succ']
    simp only [pow_zero, pow_succ', ← Finset.mul_sum]
    ring

end Mamba4
