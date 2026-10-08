import Mathlib.LinearAlgebra.Basis.Defs
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic

/-! A finite invariant function space gives a linear realization. Unlike the
original converse wording, this theorem assumes invariance and separation.
-/
namespace Mamba4
open Matrix Module
variable {Z ι : Type*} [Fintype ι]

noncomputable def evaluationFeatures (V : Submodule ℝ (Z → ℝ)) (b : Basis ι ℝ V)
    (z : Z) : ι → ℝ := fun i => (b i : Z → ℝ) z

theorem basis_evaluation (V : Submodule ℝ (Z → ℝ)) (b : Basis ι ℝ V)
    (v : V) (z : Z) :
    (∑ i, b.repr v i * evaluationFeatures V b z i) = (v : Z → ℝ) z := by
  have h := congrArg (fun u : V => (u : Z → ℝ) z) (b.sum_repr v)
  simpa only [evaluationFeatures, Submodule.coe_sum, Submodule.coe_smul,
    Finset.sum_apply, Pi.smul_apply, smul_eq_mul] using h

theorem invariant_space_realization (V : Submodule ℝ (Z → ℝ)) (b : Basis ι ℝ V)
    (f : Z → Z) (invariant : ∀ v : V, (fun z => (v : Z → ℝ) (f z)) ∈ V) :
    ∃ A : Matrix ι ι ℝ, ∀ z,
      evaluationFeatures V b (f z) = A *ᵥ evaluationFeatures V b z := by
  let pulled (i : ι) : V := ⟨fun z => (b i : Z → ℝ) (f z), invariant (b i)⟩
  refine ⟨fun i j => b.repr (pulled i) j, ?_⟩
  intro z
  ext i
  exact (basis_evaluation V b (pulled i) z).symm

theorem separating_features_injective (V : Submodule ℝ (Z → ℝ)) (b : Basis ι ℝ V)
    (separates : ∀ x y : Z, x ≠ y → ∃ v : V, (v : Z → ℝ) x ≠ (v : Z → ℝ) y) :
    Function.Injective (evaluationFeatures V b) := by
  intro x y h
  by_contra hxy
  obtain ⟨v, hv⟩ := separates x y hxy
  apply hv
  rw [← basis_evaluation V b v x, ← basis_evaluation V b v y, h]

end Mamba4
