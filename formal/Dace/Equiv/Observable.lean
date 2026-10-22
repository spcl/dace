-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics
import Dace.STree.Semantics
import Dace.Core.Facts

/-!
# Programs and their observables

The common shape of the two intermediate representations as far as
equivalence is concerned (`Decisions.md#D63`, `#D64`):

* `Prog`: an execution relation `Interp → Valuation → Mem → Res Mem → Prop`
  (`SDFG.Exec`, `STree.Exec`), the observable containers (`interface`) and
  the symbol facts the program may assume (`facts`, `Core/Facts`);
* `Prog.Pre`: the initial configurations an equivalence claim is about: a
  lawful interpretation (`Interp.Lawful`, `Decisions.md#D24`), a valuation
  satisfying the facts, and no alias bound in the initial memory (arguments
  are plain containers addressed by their own names, as `dace-interp` lays
  them out with `Mem.write (Loc.elem name offset)`);
* `interfaceOf`: the observable containers of a descriptor repository
  (`Decisions.md#D63`): non-transient arrays, streams, structures and
  container arrays, and every `__return*` container; non-transient scalars
  are passed by value and are not observable (except `GPU_Global` scalars,
  which DaCe passes by pointer); views and references are aliases, observed
  through the containers they alias;
* `obsEq`: two final memories agree on every location (cells and queues)
  of the observable containers.
-/

set_option autoImplicit false

namespace Dace

/-! ## Observable containers -/

/-- Is container `n` with descriptor `d` observable after the program ends
(`Decisions.md#D63`)?  Return values (`__return`, `__return_<i>`) always are;
otherwise non-transient containers passed by pointer: arrays, streams,
structures and container arrays, and scalars only with `GPU_Global` storage
(`dace/data/core.py`, `Scalar.as_arg` passes the value itself otherwise).
Views and references are aliases (`Decisions.md#D11`) and not observed by
name. -/
def DataDesc.isObservable (n : Name) (d : DataDesc) : Bool :=
  n.startsWith "__return" ||
    (!d.transient &&
      match d.kind with
      | .scalar => d.storage == "GPU_Global"
      | .view _ | .reference _ => false
      | .array | .stream _ | .structure _ _ | .containerArray _ => true)

/-- The observable containers of a descriptor repository, in repository
order. -/
def interfaceOf (arrays : List (Name × DataDesc)) : List Name :=
  (arrays.filter fun (n, d) => d.isObservable n).map Prod.fst

/-- The observable containers of an SDFG (`Decisions.md#D63`). -/
def SDFG.interface (g : SDFG) : List Name := interfaceOf g.arrays

/-- The observable containers of a schedule tree (`Decisions.md#D63`). -/
def STree.interface (t : STree) : List Name := interfaceOf t.containers

theorem mem_interfaceOf {arrays : List (Name × DataDesc)} {n : Name} :
    n ∈ interfaceOf arrays ↔ ∃ d, (n, d) ∈ arrays ∧ d.isObservable n = true := by
  unfold interfaceOf
  simp only [List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨⟨n', d⟩, ⟨h, ho⟩, rfl⟩; exact ⟨d, h, ho⟩
  · rintro ⟨d, h, ho⟩; exact ⟨(n, d), ⟨h, ho⟩, rfl⟩

/-- Non-transient scalars in host memory are not observable. -/
theorem not_mem_interfaceOf_scalar {arrays : List (Name × DataDesc)} {n : Name}
    (hr : n.startsWith "__return" = false)
    (hd : ∀ d, (n, d) ∈ arrays → d.kind = .scalar ∧ d.storage ≠ "GPU_Global") :
    n ∉ interfaceOf arrays := by
  rw [mem_interfaceOf]
  rintro ⟨d, h, ho⟩
  obtain ⟨hk, hs⟩ := hd d h
  simp [DataDesc.isObservable, hr, hk, hs] at ho

/-! ## Programs -/

/-- A program for the purpose of equivalence: its executions, its observable
containers and the symbol facts it may assume. -/
structure Prog where
  /-- `exec I ρ₀ m₀ r`: run with interpretation `I` from valuation `ρ₀` and
  memory `m₀`, the program may end in outcome `r`. -/
  exec : Interp → Valuation → Mem → Res Mem → Prop
  /-- The observable containers (`interfaceOf`). -/
  interface : List Name
  /-- Facts about the symbols (`Core/Facts`) that every claim may assume. -/
  facts : Constraints := {}

/-- The initial configurations an equivalence claim quantifies over: a lawful
interpretation, a valuation satisfying the program's facts, and a memory with
no alias bound (`Decisions.md#D64`). -/
def Prog.Pre (P : Prog) (I : Interp) (ρ₀ : Valuation) (m₀ : Mem) : Prop :=
  I.Lawful ∧ P.facts.satisfies I.expr ρ₀ ∧ ∀ n, m₀.aliases n = none

/-- Preconditions only depend on the facts. -/
theorem Prog.Pre.of_facts {P Q : Prog} {I : Interp} {ρ₀ : Valuation} {m₀ : Mem}
    (h : P.Pre I ρ₀ m₀) (hf : P.facts.satisfies I.expr ρ₀ → Q.facts.satisfies I.expr ρ₀) :
    Q.Pre I ρ₀ m₀ :=
  ⟨h.1, hf h.2.1, h.2.2⟩

/-- The program of an SDFG: `SDFG.Exec`, its interface and the facts of its
symbol repository. -/
def SDFG.prog (g : SDFG) : Prog where
  exec I ρ₀ m₀ r := SDFG.Exec I g ρ₀ m₀ r
  interface := g.interface
  facts := g.facts

/-- The program of a schedule tree: `STree.Exec` and its interface.  The tree
importer does not read a symbol repository, so a tree assumes no facts. -/
def STree.prog (t : STree) : Prog where
  exec I ρ₀ m₀ r := STree.Exec I t ρ₀ m₀ r
  interface := t.interface

/-! ## Observational equality -/

/-- Two final memories are observationally equal on `iface` when every
location of an observable container holds the same value and the same queue
contents.  All locations of the container are compared (not only the points
of its shape): memory is linear (`Dace.Core.Memory`) and an access through a
wrong offset stays within the container's name. -/
def obsEq (iface : List Name) (m m' : Mem) : Prop :=
  ∀ l : Loc, l.name ∈ iface → m.cells l = m'.cells l ∧ m.queues l = m'.queues l

namespace obsEq

theorem refl (iface : List Name) (m : Mem) : obsEq iface m m := fun _ _ => ⟨rfl, rfl⟩

theorem symm {iface : List Name} {m m' : Mem} (h : obsEq iface m m') : obsEq iface m' m :=
  fun l hl => ⟨(h l hl).1.symm, (h l hl).2.symm⟩

theorem trans {iface : List Name} {a b c : Mem} (h₁ : obsEq iface a b) (h₂ : obsEq iface b c) :
    obsEq iface a c :=
  fun l hl => ⟨(h₁ l hl).1.trans (h₂ l hl).1, (h₁ l hl).2.trans (h₂ l hl).2⟩

/-- Observing fewer containers is weaker. -/
theorem mono {iface iface' : List Name} {m m' : Mem} (h : obsEq iface m m')
    (hs : ∀ n ∈ iface', n ∈ iface) : obsEq iface' m m' :=
  fun l hl => h l (hs _ hl)

end obsEq

end Dace
