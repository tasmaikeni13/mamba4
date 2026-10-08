import Mathlib.Algebra.Module.Basic
import Mathlib.Tactic

/-! Affine scan summaries. `combine later earlier` applies earlier first. -/
namespace Mamba4

structure AffineSummary (V : Type*) where
  decay : ℝ
  increment : V

namespace AffineSummary
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def combine (later earlier : AffineSummary V) : AffineSummary V :=
  ⟨later.decay * earlier.decay, later.increment + later.decay • earlier.increment⟩

def identity : AffineSummary V := ⟨1, 0⟩

def act (a : AffineSummary V) (z : V) : V := a.decay • z + a.increment

theorem combine_assoc (a b c : AffineSummary V) :
    combine (combine a b) c = combine a (combine b c) := by
  cases a; cases b; cases c
  simp only [combine, smul_add, mul_smul]
  congr 1
  · ring
  · abel

theorem identity_left (a : AffineSummary V) : combine identity a = a := by
  cases a
  simp [combine, identity]

theorem identity_right (a : AffineSummary V) : combine a identity = a := by
  cases a
  simp [combine, identity]

theorem act_combine (a b : AffineSummary V) (z : V) :
    act (combine a b) z = act a (act b z) := by
  simp [act, combine, smul_add, mul_smul]
  abel

def summarize (xs : List (AffineSummary V)) : AffineSummary V :=
  xs.foldl (fun acc x => combine x acc) identity

theorem fold_act (xs : List (AffineSummary V)) (a : AffineSummary V) (z : V) :
    act (xs.foldl (fun acc x => combine x acc) a) z =
      xs.foldl (fun state x => act x state) (act a z) := by
  induction xs generalizing a with
  | nil => rfl
  | cons x xs ih => simpa [List.foldl_cons, act_combine] using ih (combine x a)

theorem scan_equals_sequential (xs : List (AffineSummary V)) (z : V) :
    act (summarize xs) z = xs.foldl (fun state x => act x state) z := by
  simpa [summarize, act, identity] using fold_act xs identity z

theorem undiscounted_commutes (x y : V) :
    combine ⟨1, x⟩ ⟨1, y⟩ = combine ⟨1, y⟩ ⟨1, x⟩ := by
  simp [combine, add_comm]

end AffineSummary
end Mamba4
