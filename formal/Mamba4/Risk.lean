import Mathlib.Tactic

/-! Exact finite-distribution risk identities. Gaussian integration is outside
this file: the hypotheses state the zero first moment/normalization explicitly.
-/
namespace Mamba4
variable {Ω : Type*} [Fintype Ω]

def expectation (p : Ω → ℝ) (x : Ω → ℝ) : ℝ := ∑ ω, p ω * x ω

theorem squared_error_decomposition (p noise : Ω → ℝ) (bias : ℝ)
    (normalized : ∑ ω, p ω = 1) (centered : expectation p noise = 0) :
    expectation p (fun ω => (bias + noise ω) ^ 2) =
      bias ^ 2 + expectation p (fun ω => (noise ω) ^ 2) := by
  have point : ∀ ω, p ω * (bias + noise ω) ^ 2 =
      bias ^ 2 * p ω + 2 * bias * (p ω * noise ω) + p ω * noise ω ^ 2 := by
    intro ω; ring
  simp only [expectation, point, Finset.sum_add_distrib, ← Finset.mul_sum]
  change bias ^ 2 * (∑ ω, p ω) + 2 * bias * expectation p noise +
    expectation p (fun ω => noise ω ^ 2) = _
  rw [normalized, centered]
  simp [expectation]

theorem posterior_mean_risk_difference (p target : Ω → ℝ) (mean candidate : ℝ)
    (normalized : ∑ ω, p ω = 1)
    (isMean : expectation p target = mean) :
    expectation p (fun ω => (candidate - target ω) ^ 2) =
      expectation p (fun ω => (mean - target ω) ^ 2) + (candidate - mean) ^ 2 := by
  have center : expectation p (fun ω => mean - target ω) = 0 := by
    simp only [expectation, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul]
    change (∑ ω, p ω) * mean - expectation p target = 0
    rw [normalized, isMean]; ring
  have h := squared_error_decomposition p (fun ω => mean - target ω)
    (candidate - mean) normalized center
  have point : (fun ω => (candidate - mean + (mean - target ω)) ^ 2) =
      (fun ω => (candidate - target ω) ^ 2) := by funext ω; congr 1; ring
  rw [point] at h
  linarith

theorem posterior_mean_minimizes (p target : Ω → ℝ) (mean candidate : ℝ)
    (normalized : ∑ ω, p ω = 1) (isMean : expectation p target = mean) :
    expectation p (fun ω => (mean - target ω) ^ 2) ≤
      expectation p (fun ω => (candidate - target ω) ^ 2) := by
  rw [posterior_mean_risk_difference p target mean candidate normalized isMean]
  exact le_add_of_nonneg_right (sq_nonneg _)

end Mamba4
