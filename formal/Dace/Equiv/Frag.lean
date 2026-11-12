-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Equiv
import Dace.Core.MergeLemmas

/-!
# Matching of fragment outcomes

Building blocks of the congruence proofs: relations between
the intermediate states of two executions and the matching of one outcome
by the outcome set of a fragment.

* `CSt.Rel a' a`: two control-flow states agree on valuation and memory, and
  `a'`'s log refines `a`'s (`Log.Le`: reads at most what `a` reads, same
  effects up to commuting reorderings);
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

/-- Two control-flow states: same valuation and memory, the log of `a'`
refines the log of `a`. -/
def CSt.Rel (a' a : CSt) : Prop :=
  a'.ρ = a.ρ ∧ a'.mem = a.mem ∧ Log.Le a'.log a.log

theorem CSt.Rel.refl (a : CSt) : CSt.Rel a a := ⟨rfl, rfl, Log.Le.refl _⟩

theorem CSt.Rel.trans {a b c : CSt} (h₁ : CSt.Rel a b) (h₂ : CSt.Rel b c) : CSt.Rel a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.trans h₂.2.2⟩

end Dace
