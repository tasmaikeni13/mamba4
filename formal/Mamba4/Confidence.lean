import Mathlib.LinearAlgebra.BilinearForm.Properties
import Mathlib.Tactic

/-! Variational confidence identity and monotonicity under added PSD evidence.
For matrices, take B(x,y)=x^T A y and q(x)=q^T x; `solution` is A^{-1}q.
-/
namespace Mamba4
open LinearMap (BilinForm)
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem confidence_variational (B : BilinForm ℝ V) (q : V →ₗ[ℝ] ℝ)
    (solution x : V) (symm : B.IsSymm)
    (solves : ∀ y, B solution y = q y) :
    q solution - (2 * q x - B x x) = B (x - solution) (x - solution) := by
  rw [LinearMap.BilinForm.sub_left, LinearMap.BilinForm.sub_right,
    LinearMap.BilinForm.sub_right, symm.eq x solution, solves x, solves solution]
  ring

theorem confidence_dominates (B : BilinForm ℝ V) (q : V →ₗ[ℝ] ℝ)
    (solution x : V) (symm : B.IsSymm) (psd : ∀ y, 0 ≤ B y y)
    (solves : ∀ y, B solution y = q y) : 2 * q x - B x x ≤ q solution := by
  have h := confidence_variational B q solution x symm solves
  have hn := psd (x - solution)
  linarith

theorem confidence_monotone (B D : BilinForm ℝ V) (q : V →ₗ[ℝ] ℝ)
    (oldSolution newSolution : V) (symm : B.IsSymm)
    (psd : ∀ y, 0 ≤ B y y) (added : ∀ y, 0 ≤ D y y)
    (oldSolves : ∀ y, B oldSolution y = q y)
    (newSolves : ∀ y, (B + D) newSolution y = q y) :
    q newSolution ≤ q oldSolution := by
  have bound := confidence_dominates B q oldSolution newSolution symm psd oldSolves
  have equation := newSolves newSolution
  have hn := added newSolution
  change B newSolution newSolution + D newSolution newSolution = q newSolution at equation
  linarith

end Mamba4
