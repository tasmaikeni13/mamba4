import Mathlib.Tactic

/-! Accounting invariants for a redundant counter. Each occupied scale has one
or two banks; exact anchor reselection is a separate interpolation contract. -/
namespace Mamba4

def dyadicMass (counts : List ℕ) : ℕ :=
  match counts with
  | [] => 0
  | x :: xs => x + 2 * dyadicMass xs

theorem carry_preserves_mass (x y : ℕ) (rest : List ℕ) :
    dyadicMass ((x + 2) :: y :: rest) = dyadicMass (x :: (y + 1) :: rest) := by
  simp [dyadicMass]; omega

theorem redundant_counter_mass_bounds (counts : List ℕ)
    (valid : ∀ x ∈ counts, 1 ≤ x ∧ x ≤ 2) :
    2 ^ counts.length - 1 ≤ dyadicMass counts ∧
      dyadicMass counts ≤ 2 * (2 ^ counts.length - 1) := by
  induction counts with
  | nil => simp [dyadicMass]
  | cons x xs ih =>
    have hx := valid x (by simp)
    have hs := ih (by intro y hy; exact valid y (by simp [hy]))
    have hp : 1 ≤ 2 ^ xs.length := Nat.one_le_two_pow
    simp only [List.length_cons, pow_succ, dyadicMass]
    omega

theorem occupied_bank_bounds (counts : List ℕ)
    (valid : ∀ x ∈ counts, 1 ≤ x ∧ x ≤ 2) :
    counts.length ≤ counts.sum ∧ counts.sum ≤ 2 * counts.length := by
  induction counts with
  | nil => simp
  | cons x xs ih =>
    have hx := valid x (by simp)
    have hs := ih (by intro y hy; exact valid y (by simp [hy]))
    simp only [List.length_cons, List.sum_cons]
    omega

end Mamba4
