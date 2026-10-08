import Mathlib.LinearAlgebra.Dimension.StrongRankCondition
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic

/-! Deterministic counting and the value-linear capacity converse. -/
namespace Mamba4

theorem encoding_injective {Values State : Type*} (encode : Values → State)
    (decode : State → Values) (exactness : ∀ v, decode (encode v) = v) :
    Function.Injective encode := by
  intro x y h
  calc x = decode (encode x) := (exactness x).symm
       _ = decode (encode y) := congrArg decode h
       _ = y := exactness y

theorem deterministic_state_count {Values State : Type*} [Fintype Values] [Fintype State]
    (encode : Values → State) (decode : State → Values)
    (exactness : ∀ v, decode (encode v) = v) :
    Fintype.card Values ≤ Fintype.card State :=
  Fintype.card_le_of_injective encode (encoding_injective encode decode exactness)

theorem exact_bit_budget (K b m : ℕ)
    (encode : (Fin (K * b) → Bool) → (Fin m → Bool))
    (decode : (Fin m → Bool) → (Fin (K * b) → Bool))
    (exactness : ∀ v, decode (encode v) = v) : K * b ≤ m := by
  have h := deterministic_state_count encode decode exactness
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at h
  exact (Nat.pow_le_pow_iff_right (by decide : 1 < (2 : ℕ))).mp h

theorem value_linear_capacity (K dv p : ℕ)
    (encode : (Fin K → Fin dv → ℝ) →ₗ[ℝ] (Fin p → ℝ))
    (decode : (Fin p → ℝ) →ₗ[ℝ] (Fin K → Fin dv → ℝ))
    (exactness : ∀ v, decode (encode v) = v) : K * dv ≤ p := by
  have h := encode.finrank_le_finrank_of_injective
    (encoding_injective encode decode exactness)
  simpa [Module.finrank_pi, Module.finrank_pi_fintype] using h

end Mamba4
