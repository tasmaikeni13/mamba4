import Mathlib.Tactic

/-! Guaranteed cyclic anisotropic prior, initialized on its stationary cycle.
Age 0 is the most recently injected coordinate; age d-1 is next to be injected.
-/
namespace Mamba4

noncomputable def cyclicPrior (eps lambda : ℝ) (d age : ℕ) : ℝ :=
  eps * lambda ^ age / lambda ^ (d - 1)

theorem cyclic_prior_floor (eps lambda : ℝ) (d age : ℕ)
    (he : 0 ≤ eps) (hl : 0 < lambda) (hu : lambda ≤ 1) (ha : age ≤ d - 1) :
    eps ≤ cyclicPrior eps lambda d age := by
  have hp := pow_le_pow_of_le_one hl.le hu ha
  have hd : 0 < lambda ^ (d - 1) := pow_pos hl _
  rw [cyclicPrior, le_div_iff₀ hd]
  nlinarith [mul_nonneg he (sub_nonneg.mpr hp)]

theorem cyclic_prior_ceiling (eps lambda : ℝ) (d age : ℕ)
    (he : 0 ≤ eps) (hl : 0 < lambda) (hu : lambda ≤ 1) :
    cyclicPrior eps lambda d age ≤ eps / lambda ^ (d - 1) := by
  apply div_le_div_of_nonneg_right _ (pow_pos hl _).le
  exact mul_le_of_le_one_right he (pow_le_one₀ hl.le hu)

theorem cyclic_prior_age_step (eps lambda : ℝ) (d age : ℕ) :
    lambda * cyclicPrior eps lambda d age = cyclicPrior eps lambda d (age + 1) := by
  simp only [cyclicPrior, pow_succ]
  ring

theorem cyclic_prior_oldest (eps lambda : ℝ) (d : ℕ) (hl : 0 < lambda) :
    cyclicPrior eps lambda d (d - 1) = eps := by
  simp [cyclicPrior, ne_of_gt (pow_pos hl (d - 1))]

theorem cyclic_prior_reinject (eps lambda : ℝ) (d : ℕ) :
    lambda * eps + (eps / lambda ^ (d - 1) - lambda * eps) =
      cyclicPrior eps lambda d 0 := by
  simp [cyclicPrior]

end Mamba4
