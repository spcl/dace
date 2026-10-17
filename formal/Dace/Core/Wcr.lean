-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Memory

/-!
# Write-conflict resolution operators

Concrete application of the built-in `WcrOp`s on values, plus the
`WcrSem` type used by the semantics.  The semantics takes WCR application as a
parameter (`WcrSem`) so that theorems can assume commutativity and
associativity (`Decisions.md#D24`) while the interpreter uses the concrete
IEEE/ℤ instance below.
-/

namespace Dace

/-- A WCR application function: `apply op old new`. -/
abbrev WcrSem := WcrOp → Val → Val → Val

namespace WcrOp

/-- Evaluate a custom lambda body with `a := old`, `b := new`. -/
def evalCustom? (I : ExprInterp) (a b : Sym) (body : Expr) (old new : Val) : Option Val :=
  body.eval I (fun s => if s = a then some old else if s = b then some new else none)

/-- Concrete application of reductions under an expression interpretation.
Custom lambdas are evaluated; if the evaluation is undefined the contribution
degrades to `new` (`Decisions.md#D26`). -/
def applyWith (I : ExprInterp) : WcrSem := fun op old new =>
  let bin (b : BinOp) : Val := (Eval.arith b old new).getD new
  match op with
  | sum => bin .add
  | product => bin .mul
  | min => bin .min
  | max => bin .max
  | logicalAnd => bin .and
  | logicalOr => bin .or
  | logicalXor => .bool ((old.toBool?.getD false) != (new.toBool?.getD false))
  | bitwiseAnd => bin .bitAnd
  | bitwiseOr => bin .bitOr
  | bitwiseXor => bin .bitXor
  | sub => bin .sub
  | div => bin .div
  | exchange => new
  | custom a b body => (evalCustom? I a b body old new).getD new

/-- Concrete application with the pure expression interpretation. -/
def applyConcrete : WcrSem := applyWith ExprInterp.pure

end WcrOp

/-- The algebraic assumptions DaCe makes about WCR operators
(`doc/sdfg/ir.rst:429-443`): contributions can be folded in any order. -/
structure WcrSem.Lawful (w : WcrSem) : Prop where
  comm_assoc : ∀ op (a x y : Val), w op (w op a x) y = w op (w op a y) x

end Dace
