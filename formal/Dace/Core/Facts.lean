-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Expr

/-!
# Symbol facts

Mirror of DaCe's symbol repository (`dace/sdfg/symbol_repo.py`,
`dace/symbolic_facts.py`, serialized under `attributes.symbol_repo`): the
integer-typed parameters of an SDFG, per-symbol sign predicates and relations
between symbolic expressions.  Facts are trusted enablers: a theorem that
uses them is stated for valuations that `satisfy` them
(`Decisions.md#D60`).

Omitted on purpose: the scale facts (`SMALL`, relation kind `MUCH_LT`), which
DaCe checks at run time, runtime assumptions, and the facts of inner scopes
(loops, map entries, interstate edges), which only hold inside their owner.
-/

namespace Dace

set_option autoImplicit false

/-- A sign fact about one symbol (`symbolic_facts.Predicate`). -/
inductive Pred where
  | pos | nonneg | neg | nonpos | nonzero
  deriving DecidableEq, Repr, Inhabited

/-- Relation kinds between two expressions (`symbolic_facts.RelationKind`
without `MUCH_LT`).  `dvd`: `lhs` divides `rhs`, i.e. `rhs` is `lhs` times an
integer. -/
inductive RelKind where
  | lt | le | eq | ne | dvd
  deriving DecidableEq, Repr, Inhabited

/-- `lhs <kind> rhs`. -/
structure Relation where
  kind : RelKind
  lhs : Expr
  rhs : Expr
  deriving DecidableEq, Repr, Inhabited

/-- The facts of an SDFG: integer-typed symbols (all integer types except
`bool`), sign predicates and relations. -/
structure Constraints where
  intSyms : List Sym := []
  sign : List (Sym × Pred) := []
  rels : List Relation := []
  deriving DecidableEq, Repr, Inhabited

namespace Pred

/-- Meaning of a sign predicate on a (rational) value. -/
def holds : Pred → ℚ → Prop
  | pos, v => 0 < v
  | nonneg, v => 0 ≤ v
  | neg, v => v < 0
  | nonpos, v => v ≤ 0
  | nonzero, v => v ≠ 0

/-- Parse a `Predicate` enum name. -/
def ofString? : String → Option Pred
  | "POSITIVE" => some pos
  | "NONNEGATIVE" => some nonneg
  | "NEGATIVE" => some neg
  | "NONPOSITIVE" => some nonpos
  | "NONZERO" => some nonzero
  | _ => none

end Pred

namespace RelKind

/-- Meaning of a relation kind on (rational) values; values of symbolic
expressions are exact rationals (`Decisions.md#D1`). -/
def holds : RelKind → ℚ → ℚ → Prop
  | lt, a, b => a < b
  | le, a, b => a ≤ b
  | eq, a, b => a = b
  | ne, a, b => a ≠ b
  | dvd, a, b => ∃ k : ℤ, b = a * k

/-- Parse a `RelationKind` enum name; `MUCH_LT` (a scale fact) is `none`. -/
def ofString? : String → Option RelKind
  | "LT" => some lt
  | "LE" => some le
  | "EQ" => some eq
  | "NE" => some ne
  | "DIVIDES" => some dvd
  | _ => none

end RelKind

/-- The rational value of an expression (`none` if undefined or a float). -/
def Expr.evalRat (I : ExprInterp) (ρ : Valuation) (e : Expr) : Option ℚ :=
  e.eval I ρ >>= Val.toRat?

/-- A relation holds when both sides evaluate to rationals related by the
kind. -/
def Relation.holds (I : ExprInterp) (ρ : Valuation) (r : Relation) : Prop :=
  ∃ a b, r.lhs.evalRat I ρ = some a ∧ r.rhs.evalRat I ρ = some b ∧ r.kind.holds a b

/-- A valuation satisfies the facts: integer symbols are bound to integers,
sign predicates hold of the symbols' values, and all relations hold. -/
def Constraints.satisfies (I : ExprInterp) (ρ : Valuation) (C : Constraints) : Prop :=
  (∀ s ∈ C.intSyms, ∃ n : Int, ρ s = some (.int n)) ∧
  (∀ sp ∈ C.sign, ∃ q, ((ρ sp.1).bind Val.toRat?) = some q ∧ sp.2.holds q) ∧
  (∀ r ∈ C.rels, r.holds I ρ)

/-- The empty set of facts is satisfied by every valuation. -/
theorem Constraints.satisfies_empty (I : ExprInterp) (ρ : Valuation) :
    ({} : Constraints).satisfies I ρ := by
  refine ⟨?_, ?_, ?_⟩ <;> intro _ h <;> simp at h

end Dace
