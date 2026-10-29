-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Equiv
import Dace.Core.MergeLemmas

/-!
# Matching of fragment outcomes

Building blocks of the congruence proofs: relations between
the intermediate states of two executions and the matching of one outcome
by the outcome set of a fragment.

* `St.Rel a' a`: two sequential dataflow states agree on the local memory
  and the code-edge values, and `a'`'s log refines `a`'s (`Log.Le`: reads at
  most what `a` reads, same effects up to commuting reorderings);
* `CSt.Rel a' a`: two control-flow states agree on valuation and memory, and
  `a'`'s log refines `a`'s;
* `Matched R X r'`: the outcome `r'` is matched by the outcome set `X`: either
  `X` contains undefined behavior, or `r'` is normal and `R`-related to a
  normal outcome of `X`.  `Refines` is `Matched obsEq` pointwise
  (`Refines_iff_matched`).

The fragment denotations themselves (one map iteration, one block) are
defined over the execution relations once those are final.

**Terminology.** "Matched" is a *simulation* in the sense of refinement
proofs (Milner's simulation relations; forward simulations in the sense of
Lynch and Vaandrager), taken over big-step outcomes: the reference program
`X` simulates the outcome `r'` of the program being checked.  The theorems
built on it (`*.matched`, `*.matchedE`, `*.matchedB`) quantify over all
derivations, inputs, symbol values and lawful interpretations; nothing is
executed.  Matching in both directions (`Equiv`) relates outcome *sets*,
which is weaker than *bisimulation*: bisimulation also relates the
branching structure of nondeterministic choices, i.e. at which point an
alternative is still open.  On deterministic programs the two notions
coincide.  The choices the SDFG semantics leaves open are the order of the
nodes in a state (irrelevant for states without races, `Dace.Equiv.Det`;
a race is undefined behavior) and the processing element of a consume task
(`Decisions.md#D38`); control flow is deterministic (several true
interstate edges are undefined behavior, `Decisions.md#D2`).
-/

set_option autoImplicit false

namespace Dace

/-! ## Matching of outcomes -/

/-- `r'` is matched by the outcome set `X` up to `R`: `X` may be undefined,
or `r'` is a normal outcome related to a normal outcome of `X`. -/
def Matched {α : Type} (R : α → α → Prop) (X : Res α → Prop) (r' : Res α) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ ∃ a', r' = .ok a' ∧ ∃ a, X (.ok a) ∧ R a' a

namespace Matched

variable {α : Type} {R : α → α → Prop} {X Y : Res α → Prop}

/-- A normal outcome of `X` is matched by `X` itself. -/
theorem of_ok (hR : ∀ a, R a a) {a : α} (h : X (.ok a)) : Matched R X (.ok a) :=
  .inr ⟨a, rfl, a, h, hR a⟩

/-- Undefined behavior in `X` matches everything. -/
theorem of_ub {e : ErrReason} (he : e.isUB = true) (h : X (.err e)) (r' : Res α) : Matched R X r' :=
  .inl ⟨e, he, h⟩

/-- A non-UB error is never matched unless `X` may be undefined. -/
theorem err_iff {e : ErrReason} : Matched R X (.err e) ↔ ∃ e', e'.isUB = true ∧ X (.err e') := by
  constructor
  · rintro (h | ⟨a', h, -⟩)
    · exact h
    · cases h
  · exact .inl

theorem mono {R' : α → α → Prop} (hR : ∀ a' a, R a' a → R' a' a) (hX : ∀ r, X r → Y r) {r' : Res α}
    (h : Matched R X r') : Matched R' Y r' := by
  rcases h with ⟨e, he, hx⟩ | ⟨a', h, a, hx, hr⟩
  · exact .inl ⟨e, he, hX _ hx⟩
  · exact .inr ⟨a', h, a, hX _ hx, hR _ _ hr⟩

/-- Matching composes: if every outcome of `Y` is matched by `X`, then
whatever `Y` matches, `X` matches. -/
theorem trans (hR : ∀ a b c, R a b → R b c → R a c) (hYX : ∀ r, Y r → Matched R X r) {r' : Res α}
    (h : Matched R Y r') : Matched R X r' := by
  rcases h with ⟨e, he, hy⟩ | ⟨a', rfl, b, hy, hab⟩
  · exact .inl (err_iff.mp (hYX _ hy))
  · rcases hYX _ hy with hu | ⟨b', hb, c, hx, hbc⟩
    · exact .inl hu
    · cases hb
      exact .inr ⟨a', rfl, c, hx, hR _ _ _ hab hbc⟩

end Matched

/-- Refinement is pointwise matching up to observational equality. -/
theorem Refines_iff_matched {Q P : Prog} :
    Refines Q P ↔ ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → ∀ r, Q.exec I ρ₀ m₀ r → Matched (obsEq P.interface) (P.exec I ρ₀ m₀) r := by
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
