-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Equiv
import Dace.Core.MergeLemmas

/-!
# Simulation of fragment outcomes

Building blocks of the congruence proofs (R1 design §2.2): relations between
the intermediate states of two executions and the simulation of one outcome
by the outcome set of a fragment.

* `St.Rel a' a`: two sequential dataflow states agree on the local memory
  and the code-edge values, and `a'`'s log refines `a`'s (`Log.Le`: reads at
  most what `a` reads, same effects up to commuting reorderings);
* `CSt.Rel a' a`: two control-flow states agree on valuation and memory, and
  `a'`'s log refines `a`'s;
* `Sim R X r'`: the outcome `r'` is simulated by the outcome set `X`: either
  `X` contains undefined behavior, or `r'` is normal and `R`-related to a
  normal outcome of `X`.  `Refines` is `Sim obsEq` pointwise
  (`Refines_iff_sim`).

The fragment denotations themselves (one map iteration, one block) are
defined over the execution relations once those are final.
-/

set_option autoImplicit false

namespace Dace

/-! ## Simulation of outcomes -/

/-- `r'` is simulated by the outcome set `X` up to `R`: `X` may be undefined,
or `r'` is a normal outcome related to a normal outcome of `X`. -/
def Sim {α : Type} (R : α → α → Prop) (X : Res α → Prop) (r' : Res α) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ ∃ a', r' = .ok a' ∧ ∃ a, X (.ok a) ∧ R a' a

namespace Sim

variable {α : Type} {R : α → α → Prop} {X Y : Res α → Prop}

/-- A normal outcome of `X` is simulated by `X` itself. -/
theorem of_ok (hR : ∀ a, R a a) {a : α} (h : X (.ok a)) : Sim R X (.ok a) :=
  .inr ⟨a, rfl, a, h, hR a⟩

/-- Undefined behavior in `X` simulates everything. -/
theorem of_ub {e : ErrReason} (he : e.isUB = true) (h : X (.err e)) (r' : Res α) : Sim R X r' :=
  .inl ⟨e, he, h⟩

/-- A non-UB error is never simulated unless `X` may be undefined. -/
theorem err_iff {e : ErrReason} : Sim R X (.err e) ↔ ∃ e', e'.isUB = true ∧ X (.err e') := by
  constructor
  · rintro (h | ⟨a', h, -⟩)
    · exact h
    · cases h
  · exact .inl

theorem mono {R' : α → α → Prop} (hR : ∀ a' a, R a' a → R' a' a) (hX : ∀ r, X r → Y r) {r' : Res α}
    (h : Sim R X r') : Sim R' Y r' := by
  rcases h with ⟨e, he, hx⟩ | ⟨a', h, a, hx, hr⟩
  · exact .inl ⟨e, he, hX _ hx⟩
  · exact .inr ⟨a', h, a, hX _ hx, hR _ _ hr⟩

/-- Simulation composes: if every outcome of `Y` is simulated by `X`, then
whatever `Y` simulates, `X` simulates. -/
theorem trans (hR : ∀ a b c, R a b → R b c → R a c) (hYX : ∀ r, Y r → Sim R X r) {r' : Res α}
    (h : Sim R Y r') : Sim R X r' := by
  rcases h with ⟨e, he, hy⟩ | ⟨a', rfl, b, hy, hab⟩
  · exact .inl (err_iff.mp (hYX _ hy))
  · rcases hYX _ hy with hu | ⟨b', hb, c, hx, hbc⟩
    · exact .inl hu
    · cases hb
      exact .inr ⟨a', rfl, c, hx, hR _ _ _ hab hbc⟩

end Sim

/-- Refinement is pointwise simulation up to observational equality. -/
theorem Refines_iff_sim {Q P : Prog} :
    Refines Q P ↔ ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → ∀ r, Q.exec I ρ₀ m₀ r → Sim (obsEq P.interface) (P.exec I ρ₀ m₀) r := by
  constructor
  · intro h I ρ₀ m₀ hp r hr
    rcases h I ρ₀ m₀ hp with hu | hall
    · exact .inl hu
    · exact .inr (hall r hr)
  · intro h I ρ₀ m₀ hp
    by_cases hu : mayUB P I ρ₀ m₀
    · exact .inl hu
    · refine .inr fun r hr => ?_
      rcases h I ρ₀ m₀ hp r hr with hu' | hok
      · exact absurd hu' hu
      · exact hok

/-! ## Related states -/

/-- Two sequential dataflow states: same local memory and code-edge values,
the log of `a'` refines the log of `a`. -/
def St.Rel (a' a : St) : Prop :=
  a'.mem = a.mem ∧ Log.Le a'.log a.log ∧ ∀ k, a'.val? k = a.val? k

/-- Two control-flow states: same valuation and memory, the log of `a'`
refines the log of `a`. -/
def CSt.Rel (a' a : CSt) : Prop :=
  a'.ρ = a.ρ ∧ a'.mem = a.mem ∧ Log.Le a'.log a.log

theorem St.Rel.refl (a : St) : St.Rel a a := ⟨rfl, Log.Le.refl _, fun _ => rfl⟩

theorem St.Rel.trans {a b c : St} (h₁ : St.Rel a b) (h₂ : St.Rel b c) : St.Rel a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, fun k => (h₁.2.2 k).trans (h₂.2.2 k)⟩

theorem CSt.Rel.refl (a : CSt) : CSt.Rel a a := ⟨rfl, rfl, Log.Le.refl _⟩

theorem CSt.Rel.trans {a b c : CSt} (h₁ : CSt.Rel a b) (h₂ : CSt.Rel b c) : CSt.Rel a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.trans h₂.2.2⟩

/-- The dataflow views of related control-flow states are related. -/
theorem CSt.Rel.st {a' a : CSt} (h : CSt.Rel a' a) : St.Rel a'.st a.st :=
  ⟨h.2.1, h.2.2, fun _ => rfl⟩

end Dace
