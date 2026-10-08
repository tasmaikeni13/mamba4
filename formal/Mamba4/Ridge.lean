import Mathlib.Tactic

/-! Completing-square proof of the weighted ridge optimizer for arbitrary d.
The normal equations are explicit finite sums, not an assumed optimizer. -/
namespace Mamba4
open Finset
variable {ι n : Type*} [Fintype ι] [Fintype n]

def ridgeLoss (keys : ι → n → ℝ) (values weights : ι → ℝ) (eps : ℝ)
    (w : n → ℝ) : ℝ :=
  (∑ i, weights i * ((∑ j, w j * keys i j) - values i) ^ 2) +
    eps * ∑ j, (w j) ^ 2

theorem ridge_loss_difference (keys : ι → n → ℝ) (values weights : ι → ℝ)
    (eps : ℝ) (w h : n → ℝ)
    (normal : ∀ j, (∑ i, weights i * ((∑ l, w l * keys i l) - values i) * keys i j)
      + eps * w j = 0) :
    ridgeLoss keys values weights eps (w + h) - ridgeLoss keys values weights eps w =
      (∑ i, weights i * (∑ j, h j * keys i j) ^ 2) + eps * ∑ j, (h j) ^ 2 := by
  have cross :
      (∑ i, weights i * ((∑ l, w l * keys i l) - values i) *
        (∑ j, h j * keys i j)) + eps * ∑ j, w j * h j = 0 := by
    calc
      _ = ∑ j, h j * ((∑ i, weights i * ((∑ l, w l * keys i l) - values i) *
          keys i j) + eps * w j) := by
        simp only [mul_add, Finset.sum_add_distrib, Finset.mul_sum]
        rw [Finset.sum_comm]
        congr 1 <;> apply Finset.sum_congr rfl <;> intro j hj
        · apply Finset.sum_congr rfl; intro i hi; ring
        · ring
      _ = 0 := by simp [normal]
  have expand_obs : ∀ i,
      weights i * ((∑ j, (w + h) j * keys i j) - values i) ^ 2 =
      weights i * ((∑ j, w j * keys i j) - values i) ^ 2 +
      weights i * (∑ j, h j * keys i j) ^ 2 +
      2 * (weights i * ((∑ j, w j * keys i j) - values i) *
        (∑ j, h j * keys i j)) := by
    intro i
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
    ring
  have expand_reg : ∀ j, ((w + h) j) ^ 2 = (w j) ^ 2 + (h j) ^ 2 + 2 * (w j * h j) := by
    intro j; simp only [Pi.add_apply]; ring
  simp only [ridgeLoss, expand_obs, expand_reg, Finset.sum_add_distrib,
    ← Finset.mul_sum]
  nlinarith [cross]

theorem ridge_minimizes (keys : ι → n → ℝ) (values weights : ι → ℝ)
    (eps : ℝ) (w : n → ℝ) (hb : ∀ i, 0 ≤ weights i) (he : 0 ≤ eps)
    (normal : ∀ j, (∑ i, weights i * ((∑ l, w l * keys i l) - values i) * keys i j)
      + eps * w j = 0) (candidate : n → ℝ) :
    ridgeLoss keys values weights eps w ≤ ridgeLoss keys values weights eps candidate := by
  have h := ridge_loss_difference keys values weights eps w (candidate - w) normal
  have hs : w + (candidate - w) = candidate := by ext j; simp
  rw [hs] at h
  have ho : 0 ≤ ∑ i, weights i * (∑ j, (candidate - w) j * keys i j) ^ 2 :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hb i) (sq_nonneg _))
  have hr : 0 ≤ eps * ∑ j, ((candidate - w) j) ^ 2 :=
    mul_nonneg he (Finset.sum_nonneg (fun j _ => sq_nonneg _))
  linarith

end Mamba4
