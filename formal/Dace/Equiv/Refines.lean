-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Observable

/-!
# Refinement and equivalence

UB-style refinement of programs (`Decisions.md#D64`; races are undefined
behavior, `doc/sdfg/synchronization.rst`):

* `Refines Q P` ("`Q` refines `P`"): for every initial configuration
  satisfying `P.Pre`, either `P` may end in undefined behavior (a race, a
  blocked pop, an undefined expression, …: `ErrReason.isUB`), or every
  outcome of `Q` is a final memory that some normal outcome of `P` matches
  on `P`'s observable containers (`obsEq`).
* Errors that are not undefined behavior (`malformed`, `unsupported`) make
  **no claim**: in `P` they only remove outcomes `Q` could match, and in `Q`
  they are never allowed (unless `P` may be undefined).
* Refinement is **partial correctness**: divergence is the absence of an
  outcome, so a `Q` without outcomes refines everything.
* `Equiv P Q`: mutual refinement.  It is a partial equivalence relation
  whose domain is `Supported` (`Equiv.refl_iff_supported`): a program is
  equivalent to itself exactly when it never ends in a non-UB error unless it
  may also be undefined.
* `Extends P' P`: `P'` keeps every outcome of `P` except non-UB errors and
  adds outcomes only where `P` has a non-UB error, i.e. it models what `P`
  left unmodeled.  Refinement is conservative over such extensions
  (`Refines.conservative`): modeling more of DaCe never invalidates a proven
  refinement.
-/

set_option autoImplicit false

namespace Dace

/-- `P` may end in undefined behavior from this configuration. -/
def mayUB (P : Prog) (I : Interp) (ρ₀ : Valuation) (m₀ : Mem) : Prop :=
  ∃ e, e.isUB = true ∧ P.exec I ρ₀ m₀ (.err e)

/-- `Q` refines `P` (`Decisions.md#D64`): under `P`'s precondition, `P` may be
undefined, or every outcome of `Q` is normal and matched by a normal outcome
of `P` on `P`'s observable containers. -/
def Refines (Q P : Prog) : Prop :=
  ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ →
    mayUB P I ρ₀ m₀ ∨
      ∀ r, Q.exec I ρ₀ m₀ r → ∃ m, r = .ok m ∧ ∃ m', P.exec I ρ₀ m₀ (.ok m') ∧ obsEq P.interface m m'

/-- Equivalence: mutual refinement. -/
def Equiv (P Q : Prog) : Prop := Refines P Q ∧ Refines Q P

/-- `P` never ends in an error that is not undefined behavior unless it may
also be undefined: the programs about which refinement can make a claim. -/
def Supported (P : Prog) : Prop :=
  ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → mayUB P I ρ₀ m₀ ∨ ∀ e, ¬ P.exec I ρ₀ m₀ (.err e)

/-- `P'` extends `P`: same interface and facts; every outcome of `P` other
than a non-UB error is an outcome of `P'`, and `P'` has new outcomes only
where `P` may end in a non-UB error (`malformed`, `unsupported`). -/
structure Extends (P' P : Prog) : Prop where
  interface : P'.interface = P.interface
  facts : P'.facts = P.facts
  keep : ∀ I ρ₀ m₀ r, P.exec I ρ₀ m₀ r → P'.exec I ρ₀ m₀ r ∨ ∃ e, r = .err e ∧ e.isUB = false
  new : ∀ I ρ₀ m₀ r, P'.exec I ρ₀ m₀ r →
    P.exec I ρ₀ m₀ r ∨ ∃ e, e.isUB = false ∧ P.exec I ρ₀ m₀ (.err e)

/-! ## Basic facts -/

/-- A program that may be undefined is refined by anything. -/
theorem Refines.of_mayUB {Q P : Prog} (h : ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → mayUB P I ρ₀ m₀) : Refines Q P :=
  fun I ρ₀ m₀ hp => .inl (h I ρ₀ m₀ hp)

/-- If `Q` refines `P` and `P` is not undefined, `Q` has no error outcome. -/
theorem Refines.noErr {Q P : Prog} (h : Refines Q P) {I : Interp} {ρ₀ : Valuation} {m₀ : Mem}
    (hp : P.Pre I ρ₀ m₀) (hu : ¬ mayUB P I ρ₀ m₀) (e : ErrReason) : ¬ Q.exec I ρ₀ m₀ (.err e) := by
  intro hq
  rcases h I ρ₀ m₀ hp with hm | hall
  · exact hu hm
  · obtain ⟨m, hm, -⟩ := hall _ hq
    cases hm

/-- A supported program refines itself. -/
theorem Refines.refl_of_supported {P : Prog} (h : Supported P) : Refines P P := by
  intro I ρ₀ m₀ hp
  rcases h I ρ₀ m₀ hp with hu | hne
  · exact .inl hu
  · refine .inr fun r hr => ?_
    cases r with
    | ok m => exact ⟨m, rfl, m, hr, obsEq.refl _ _⟩
    | err e => exact absurd hr (hne e)

/-- A program refining itself is supported. -/
theorem Supported.of_refines_self {P : Prog} (h : Refines P P) : Supported P := by
  intro I ρ₀ m₀ hp
  by_cases hu : mayUB P I ρ₀ m₀
  · exact .inl hu
  · exact .inr (h.noErr hp hu)

theorem Refines.refl_iff_supported {P : Prog} : Refines P P ↔ Supported P :=
  ⟨Supported.of_refines_self, Refines.refl_of_supported⟩

/-- Transitivity.  The middle program `Q` must accept every configuration
`P` accepts (`hpre`, e.g. `P.facts` entail `Q.facts`) and observe at least
what `P` observes (`hif`). -/
theorem Refines.trans {R Q P : Prog} (h₁ : Refines R Q) (h₂ : Refines Q P)
    (hpre : ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → Q.Pre I ρ₀ m₀) (hif : ∀ n ∈ P.interface, n ∈ Q.interface) :
    Refines R P := by
  intro I ρ₀ m₀ hp
  rcases h₂ I ρ₀ m₀ hp with hu | hall₂
  · exact .inl hu
  · refine .inr fun r hr => ?_
    rcases h₁ I ρ₀ m₀ (hpre I ρ₀ m₀ hp) with ⟨e, -, he⟩ | hall₁
    · obtain ⟨m, hm, -⟩ := hall₂ _ he
      cases hm
    · obtain ⟨m, rfl, m', hq, hobs⟩ := hall₁ r hr
      obtain ⟨m₁, hm₁, m'', hp', hobs'⟩ := hall₂ _ hq
      cases hm₁
      exact ⟨m, rfl, m'', hp', (hobs.mono hif).trans hobs'⟩

/-- Programs with the same facts accept the same configurations. -/
theorem Prog.pre_of_facts_eq {P Q : Prog} (h : P.facts = Q.facts) {I : Interp} {ρ₀ : Valuation}
    {m₀ : Mem} (hp : P.Pre I ρ₀ m₀) : Q.Pre I ρ₀ m₀ :=
  hp.of_facts (h ▸ id)

/-- Transitivity for programs with the same facts and interface as the middle
one. -/
theorem Refines.trans_of_eq {R Q P : Prog} (h₁ : Refines R Q) (h₂ : Refines Q P)
    (hf : P.facts = Q.facts) (hi : P.interface = Q.interface) : Refines R P :=
  h₁.trans h₂ (fun _ _ _ hp => Prog.pre_of_facts_eq hf hp) (fun _ hn => hi ▸ hn)

theorem Equiv.symm {P Q : Prog} (h : Equiv P Q) : Equiv Q P := ⟨h.2, h.1⟩

/-- Transitivity: the middle program `Q` accepts every configuration `P` or
`R` accepts and observes at least what they observe. -/
theorem Equiv.trans {P Q R : Prog} (h₁ : Equiv P Q) (h₂ : Equiv Q R)
    (hpreP : ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → Q.Pre I ρ₀ m₀) (hpreR : ∀ I ρ₀ m₀, R.Pre I ρ₀ m₀ → Q.Pre I ρ₀ m₀)
    (hifP : ∀ n ∈ P.interface, n ∈ Q.interface) (hifR : ∀ n ∈ R.interface, n ∈ Q.interface) :
    Equiv P R :=
  ⟨h₁.1.trans h₂.1 hpreR hifR, h₂.2.trans h₁.2 hpreP hifP⟩

/-- Transitivity for programs with equal facts and interfaces. -/
theorem Equiv.trans_of_eq {P Q R : Prog} (h₁ : Equiv P Q) (h₂ : Equiv Q R)
    (hfP : P.facts = Q.facts) (hfR : R.facts = Q.facts) (hiP : P.interface = Q.interface)
    (hiR : R.interface = Q.interface) : Equiv P R :=
  ⟨h₁.1.trans_of_eq h₂.1 hfR hiR, h₂.2.trans_of_eq h₁.2 hfP hiP⟩

/-- Equivalence is a partial equivalence relation with domain `Supported`. -/
theorem Equiv.refl_iff_supported {P : Prog} : Equiv P P ↔ Supported P :=
  ⟨fun h => Refines.refl_iff_supported.mp h.1,
   fun h => ⟨Refines.refl_of_supported h, Refines.refl_of_supported h⟩⟩

/-- Both sides of an equivalence are supported, provided their facts are
interchangeable. -/
theorem Equiv.supported_left {P Q : Prog} (h : Equiv P Q) (hf : P.facts = Q.facts)
    (hi : P.interface = Q.interface) : Supported P :=
  Equiv.refl_iff_supported.mp (h.trans_of_eq h.symm hf hf hi hi)

/-- Refinement is conservative over extensions: replacing non-UB errors
(`malformed`, `unsupported`) of either side by modeled behavior keeps a
refinement valid. -/
theorem Refines.conservative {Q P Q' P' : Prog} (h : Refines Q P) (hQ : Extends Q' Q) (hP : Extends P' P) :
    Refines Q' P' := by
  intro I ρ₀ m₀ hp'
  have hp : P.Pre I ρ₀ m₀ := Prog.pre_of_facts_eq hP.facts hp'
  rcases h I ρ₀ m₀ hp with ⟨e, hub, he⟩ | hall
  · rcases hP.keep I ρ₀ m₀ _ he with he' | ⟨e', he', hnub⟩
    · exact .inl ⟨e, hub, he'⟩
    · cases he'
      rw [hub] at hnub
      cases hnub
  · refine .inr fun r hr => ?_
    rcases hQ.new I ρ₀ m₀ r hr with hq | ⟨e, -, he⟩
    · obtain ⟨m, rfl, m', hpm, hobs⟩ := hall r hq
      rcases hP.keep I ρ₀ m₀ _ hpm with hpm' | ⟨e, he, -⟩
      · exact ⟨m, rfl, m', hpm', hP.interface ▸ hobs⟩
      · cases he
    · obtain ⟨m, hm, -⟩ := hall _ he
      cases hm

/-- Every program extends itself. -/
theorem Extends.refl (P : Prog) : Extends P P :=
  ⟨rfl, rfl, fun _ _ _ _ h => .inl h, fun _ _ _ _ h => .inl h⟩

end Dace
