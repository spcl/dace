-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Sanity checks of the refinement notion (`Dace/Equiv/Equiv.lean`,
`Decisions.md#D64`) on abstract programs, and of the observable interface
(`Decisions.md#D63`) on the corpus.  Run with `lake env lean tests/EquivTest.lean`
from `formal/`; it fails if an `example` does not check or a corpus
interface differs from the expectation. -/

open Dace

/-! ## Abstract programs -/

/-- A lawful interpretation (every WCR keeps the old value). -/
def lawfulI : Interp := { Interp.uninterpreted with wcr := fun _ a _ => a }

theorem lawfulI_lawful : lawfulI.Lawful := ⟨fun _ _ _ _ => rfl, fun _ _ => rfl⟩

/-- The initial memory of the counterexamples: all cells `0`. -/
def m0 : Mem := Mem.empty (.int 0)

theorem pre_any (P : Prog) (hf : P.facts = {}) : P.Pre lawfulI (fun _ => none) m0 :=
  ⟨lawfulI_lawful, hf ▸ Constraints.satisfies_empty _ _, fun _ => rfl⟩

/-- Always a data race. -/
def racy : Prog where
  exec _ _ _ r := r = .err (.race (.race (Loc.whole "A")))
  interface := ["A"]

/-- The identity program. -/
def ident : Prog where
  exec _ _ m r := r = .ok m
  interface := ["A"]

/-- May end in the identity or in `unsupported`. -/
def identOrUnsupported : Prog where
  exec _ _ m r := r = .ok m ∨ r = .err (.unsupported "x")
  interface := ["A"]

/-- Always `unsupported`. -/
def unsup : Prog where
  exec _ _ _ r := r = .err (.unsupported "x")
  interface := ["A"]

/-- Writes `1` to `A[0]`. -/
def writeA : Prog where
  exec _ _ m r := r = .ok { m with cells := Function.update m.cells (Loc.elem "A" 0) (.int 1) }
  interface := ["A"]

/-- Never terminates. -/
def diverge : Prog where
  exec _ _ _ _ := False
  interface := ["A"]

-- A program that is always racy is refined by anything, even by `unsupported`.
example (Q : Prog) : Refines Q racy :=
  Refines.of_mayUB fun _ _ _ _ => ⟨_, rfl, rfl⟩

-- `unsupported` in the refined program only removes outcomes: the identity
-- refines `identOrUnsupported`, matching its real normal outcome.
example : Refines ident identOrUnsupported := by
  intro I ρ m _
  refine .inr fun r hr => ?_
  subst hr
  exact ⟨m, rfl, m, .inl rfl, obsEq.refl _ _⟩

-- ... but a program with a different normal outcome does not.
example : ¬ Refines writeA identOrUnsupported := by
  intro h
  rcases h lawfulI (fun _ => none) m0 (pre_any _ rfl) with ⟨e, hub, he⟩ | hall
  · rcases he with he | he <;> cases he
    exact absurd hub (by decide)
  · obtain ⟨m, hm, m', hp, hobs⟩ := hall _ rfl
    rcases hp with hp | hp
    · cases hm
      cases hp
      have := (hobs (Loc.elem "A" 0) (by simp [Loc.elem, identOrUnsupported])).1
      simp [m0, Mem.empty] at this
    · cases hp

-- `unsupported` is not undefined behavior: it does not refine the identity,
-- and `identOrUnsupported` is not equivalent to itself (`Supported` fails).
example : ¬ Refines unsup ident := by
  intro h
  rcases h lawfulI (fun _ => none) m0 (pre_any _ rfl) with ⟨e, -, he⟩ | hall
  · cases he
  · obtain ⟨m, hm, -⟩ := hall _ rfl
    cases hm

example : ¬ Equiv identOrUnsupported identOrUnsupported := by
  rw [Equiv.refl_iff_supported]
  intro h
  rcases h lawfulI (fun _ => none) m0 (pre_any _ rfl) with ⟨e, hub, he⟩ | hne
  · rcases he with he | he <;> cases he
    exact absurd hub (by decide)
  · exact hne _ (.inr rfl)

-- Partial correctness: a diverging program refines everything.
example (P : Prog) : Refines diverge P := by
  intro I ρ m _
  by_cases hu : mayUB P I ρ m
  · exact .inl hu
  · exact .inr fun _ hr => hr.elim

-- Replacing `unsupported` by real behavior keeps the refinement
-- (`Refines.conservative`).
example : Refines ident ident := by
  have h : Refines ident identOrUnsupported := by
    intro I ρ m _
    exact .inr fun r hr => ⟨m, hr, m, .inl rfl, obsEq.refl _ _⟩
  refine h.conservative (Extends.refl _) ⟨rfl, rfl, ?_, ?_⟩
  · rintro I ρ m r (hr | hr)
    · exact .inl hr
    · exact .inr ⟨_, hr, rfl⟩
  · intro I ρ m r hr
    exact .inl (.inl hr)

/-! ## Transients that outlive a run

`Persistent` and `External` transients keep their contents across calls, so
they are observed like arguments; `Global`, `SDFG` and `Scope` transients are
allocated per run (`Decisions.md#D63`, `#D68`). -/

def arr (transient : Bool) (lt : Lifetime := .scope) : DataDesc :=
  { DataDesc.array (.float 64) [4] transient with lifetime := lt }

def lifetimes : List (Name × DataDesc) :=
  [("A", arr false), ("P", arr true .persistent), ("X", arr true .external),
   ("G", arr true .global), ("S", arr true .sdfg), ("T", arr true)]

#guard interfaceOf lifetimes == ["A", "P", "X"]

/-- An SDFG nesting `lifetimes` in node 3 of its only state.  Nested SDFGs
may own only per-call transients (canonical SDFGs hoist `Persistent`,
`External` and `Global` ones to the top level): the interface is the outer
one, and invoking the nested SDFG is `malformed`. -/
def innerSDFG : SDFG := { name := "inner", arrays := lifetimes, top := {} }

def outer : SDFG where
  name := "outer"
  arrays := [("B", arr false)]
  top := { blocks := [.state { nodes := [{ id := 3, kind := .nested innerSDFG [] false }] }] }

#guard outer.interface == ["B"]
#guard match nestedCheck innerSDFG with
  | .err (.malformed _) => true
  | _ => false
#guard match nestedCheck { innerSDFG with arrays := [("G", arr true .global)] } with
  | .err (.malformed _) => true
  | _ => false
#guard match nestedCheck { innerSDFG with arrays := [("A", arr false), ("S", arr true .sdfg), ("T", arr true)] } with
  | .ok () => true
  | _ => false

/-! ## Observable interfaces of the corpus -/

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

/-- Expected interfaces: non-transient scalars (`axpy.a`) are observable like
any other non-transient container, transients (`two_states.tmp`) are not. -/
def expectedInterfaces : List (String × List Name) :=
  [("axpy", ["a", "x", "y"]), ("two_states", ["A", "B", "C"]), ("matmul", ["A", "B", "C"])]

def main : IO Unit := do
  let dir ← corpusDir
  let entries ← dir.readDir
  let names := (entries.filterMap fun e =>
    let s := e.fileName
    if s.endsWith ".sdfg.json" then some (s.splitOn ".sdfg.json").head! else none).qsort (· < ·)
  let mut failures := 0
  for n in names do
    let g ← match ← SDFG.Json.readSDFG (dir / s!"{n}.sdfg.json") with
      | .ok g => pure g
      | .error e => throw <| IO.userError s!"{n}: {e}"
    let t ← match ← readSTree (dir / s!"{n}.stree.json") with
      | .ok t => pure t
      | .error e => throw <| IO.userError s!"{n}: {e}"
    let ok := g.interface == t.interface &&
      (expectedInterfaces.lookup n).all (· == g.interface)
    if !ok then failures := failures + 1
    IO.println s!"{if ok then "OK  " else "FAIL"} {n}: sdfg {g.interface} tree {t.interface}"
  if failures > 0 then throw <| IO.userError s!"{failures} interface mismatches"

#eval main
