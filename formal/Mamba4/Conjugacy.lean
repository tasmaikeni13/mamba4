import Mathlib.Tactic

/-! Natural-parameter algebra in log coordinates; no normalizer/sufficiency
claim is inferred without a statistical model and its regularity assumptions. -/
namespace Mamba4
variable {ι : Type*} [Fintype ι]

def logKernel (chi eta : ι → ℝ) (nu potential : ℝ) : ℝ :=
  (∑ i, chi i * eta i) - nu * potential

theorem weighted_tempered_log_update (chi eta statistic : ι → ℝ)
    (nu potential beta decay : ℝ) :
    logKernel (decay • chi + beta • statistic) eta (decay * nu + beta) potential =
      decay * logKernel chi eta nu potential +
        beta * ((∑ i, statistic i * eta i) - potential) := by
  simp only [logKernel, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_mul,
    Finset.sum_add_distrib]
  simp_rw [mul_assoc]
  rw [← Finset.mul_sum, ← Finset.mul_sum]
  ring

theorem posterior_kernel_merge (prior evidence₁ evidence₂ : ℝ) :
    Real.exp (prior + evidence₁ + evidence₂) * Real.exp prior =
      Real.exp (prior + evidence₁) * Real.exp (prior + evidence₂) := by
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  ring

end Mamba4
