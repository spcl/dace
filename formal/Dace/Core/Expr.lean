-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Prelude

/-!
# Symbolic expressions

DaCe stores most numeric attributes (shapes, strides, subsets, map ranges,
interstate conditions and assignments, symbol mappings) as SymPy expressions
serialized as Python-syntax strings.  This module defines the expression
language used throughout the formalization and its evaluation.

Design (see `Decisions.md#D1`, `#D22`, `#D23`):
* integers are mathematical `Int` (no fixed width);
* floats are IEEE doubles (`Float`) in the executable semantics and are treated
  opaquely by all theorems;
* unknown function applications are kept as `Expr.app` and interpreted by an
  `ExprInterp`; data reads (`A[i]`, as allowed in interstate edges) are resolved
  through a reader callback.

`Expr` and the argument-list type `Exprs` form a mutual inductive (rather than
nesting `List Expr`) so that decidable equality can be derived.
-/

namespace Dace

/-! ## Types and values -/

/-- Scalar data types of DaCe (`dace.dtypes.typeclass`), kept as syntax. -/
inductive Dtype where
  | int (bits : Nat) (signed : Bool)
  | float (bits : Nat)
  | bool
  | opaque (name : String)
  deriving DecidableEq, Repr, Inhabited

/-- Runtime values. -/
inductive Val where
  | int (v : Int)
  | float (f : Float)
  | bool (b : Bool)
  deriving Inhabited, Repr

namespace Val

/-- Floats are compared bitwise so that `Val` has a lawful boolean equality. -/
def beq : Val → Val → Bool
  | int a, int b => a == b
  | float a, float b => a.toBits == b.toBits
  | bool a, bool b => a == b
  | _, _ => false

instance : BEq Val := ⟨beq⟩

/-- Integer coercion used for indices: floats and booleans are rejected. -/
def toInt? : Val → Option Int
  | int v => some v
  | _ => none

def toBool? : Val → Option Bool
  | bool b => some b
  | int v => some (v != 0)
  | float f => some (f != 0.0)

def toFloat : Val → Float
  | int v => Float.ofInt v
  | float f => f
  | bool b => if b then 1.0 else 0.0

def isFloat : Val → Bool
  | float _ => true
  | _ => false

end Val

/-! ## Syntax -/

/-- Literals.  Float literals are stored exactly as a decimal
`mant * 10 ^ exp` so that `Expr` has decidable equality and printing
round-trips; the IEEE value is produced on evaluation. -/
inductive Lit where
  | int (v : Int)
  | float (mant : Int) (exp : Int)
  | bool (b : Bool)
  deriving DecidableEq, Repr, Inhabited

namespace Lit

/-- IEEE double value of a float literal. -/
def floatVal (mant exp : Int) : Float :=
  let m := Float.ofScientific mant.natAbs (exp < 0) exp.natAbs
  if mant < 0 then -m else m

end Lit

inductive UnOp where
  | neg | not | bitNot | abs
  deriving DecidableEq, Repr, Inhabited

inductive BinOp where
  | add | sub | mul | div        -- `div` is true (floating) division
  | intFloor | intCeil | mod     -- `int_floor`, `int_ceil`, `Mod`
  | pow
  | min | max
  | bitAnd | bitOr | bitXor | shl | shr
  | and | or
  | eq | ne | lt | le | gt | ge
  deriving DecidableEq, Repr, Inhabited

mutual
/-- Symbolic expressions. -/
inductive Expr where
  | lit (l : Lit)
  | sym (s : Sym)
  | un (op : UnOp) (a : Expr)
  | bin (op : BinOp) (a b : Expr)
  | ite (c t e : Expr)
  /-- Function application not covered by the primitive operators
      (e.g. `sqrt`, `exp`, user callbacks). -/
  | app (f : String) (args : Exprs)
  /-- Read of a data container element, allowed in interstate edges. -/
  | subscript (data : Name) (idx : Exprs)
  /-- Structure member access `a.b`. -/
  | attr (base : Expr) (field : String)
  deriving Inhabited, Repr

/-- Argument lists. -/
inductive Exprs where
  | nil
  | cons (e : Expr) (es : Exprs)
  deriving Inhabited, Repr
end

deriving instance DecidableEq for Expr, Exprs

namespace Exprs

def ofList : List Expr → Exprs
  | [] => nil
  | e :: es => cons e (ofList es)

def toList : Exprs → List Expr
  | nil => []
  | cons e es => e :: toList es

@[simp] theorem toList_ofList (l : List Expr) : (ofList l).toList = l := by
  induction l with
  | nil => rfl
  | cons e es ih => simp [ofList, toList, ih]

@[simp] theorem ofList_toList : (es : Exprs) → ofList es.toList = es
  | nil => rfl
  | cons e es => by simp [ofList, toList, ofList_toList es]

instance : Coe (List Expr) Exprs := ⟨ofList⟩

end Exprs

namespace Expr

/-- The integer literal `n`. -/
def ofInt (n : Int) : Expr := .lit (.int n)

instance : OfNat Expr n := ⟨ofInt n⟩
instance : Coe Int Expr := ⟨ofInt⟩

def add (a b : Expr) : Expr := .bin .add a b
def sub (a b : Expr) : Expr := .bin .sub a b
def mul (a b : Expr) : Expr := .bin .mul a b
instance : Add Expr := ⟨add⟩
instance : Sub Expr := ⟨sub⟩
instance : Mul Expr := ⟨mul⟩

mutual
/-- Free symbols of an expression (data container names in `subscript` are
*not* symbols). -/
def freeSyms : Expr → List Sym
  | lit _ => []
  | sym s => [s]
  | un _ a => a.freeSyms
  | bin _ a b => a.freeSyms ++ b.freeSyms
  | ite c t e => c.freeSyms ++ t.freeSyms ++ e.freeSyms
  | app _ args => freeSymsList args
  | subscript _ idx => freeSymsList idx
  | attr b _ => b.freeSyms

def freeSymsList : Exprs → List Sym
  | .nil => []
  | .cons e es => e.freeSyms ++ freeSymsList es
end

mutual
/-- Data containers read by an expression. -/
def readData : Expr → List Name
  | lit _ | sym _ => []
  | un _ a => a.readData
  | bin _ a b => a.readData ++ b.readData
  | ite c t e => c.readData ++ t.readData ++ e.readData
  | app _ args => readDataList args
  | subscript d idx => d :: readDataList idx
  | attr b _ => b.readData

def readDataList : Exprs → List Name
  | .nil => []
  | .cons e es => e.readData ++ readDataList es
end

mutual
/-- Capture-free simultaneous substitution of symbols. -/
def subst (σ : Sym → Option Expr) : Expr → Expr
  | lit l => lit l
  | sym s => (σ s).getD (sym s)
  | un op a => un op (a.subst σ)
  | bin op a b => bin op (a.subst σ) (b.subst σ)
  | ite c t e => ite (c.subst σ) (t.subst σ) (e.subst σ)
  | app f args => app f (substList σ args)
  | subscript d idx => subscript d (substList σ idx)
  | attr b f => attr (b.subst σ) f

def substList (σ : Sym → Option Expr) : Exprs → Exprs
  | .nil => .nil
  | .cons e es => .cons (e.subst σ) (substList σ es)
end

end Expr

/-! ## Evaluation -/

/-- Valuation of symbols. -/
abbrev Valuation := Sym → Option Val

/-- Interpretation of the open parts of the expression language. -/
structure ExprInterp where
  /-- Opaque function applications (`sqrt`, `exp`, callbacks, ...). -/
  app : String → List Val → Option Val
  /-- Reads of data container elements (`A[i]` in interstate edges).
      Returns `none` when the read is undefined. -/
  readData : Name → List Int → Option Val
  /-- Structure member reads. -/
  readAttr : Val → String → Option Val := fun _ _ => none

namespace Eval

/-- Floor division (Python `//`), `none` on division by zero. -/
def intFloor (a b : Int) : Option Int := if b = 0 then none else some (Int.fdiv a b)

/-- Ceiling division `int_ceil(a, b) = ⌈a / b⌉`, `none` on division by zero. -/
def intCeil (a b : Int) : Option Int := if b = 0 then none else some (-(Int.fdiv (-a) b))

/-- Python modulo (sign of the divisor), `none` on division by zero. -/
def pyMod (a b : Int) : Option Int := if b = 0 then none else some (Int.fmod a b)

/-- Integer arithmetic. -/
def intArith (op : BinOp) (a b : Int) : Option Val :=
  match op with
  | .add => some (.int (a + b))
  | .sub => some (.int (a - b))
  | .mul => some (.int (a * b))
  | .div => if b = 0 then none else some (.float (Float.ofInt a / Float.ofInt b))
  | .intFloor => (intFloor a b).map .int
  | .intCeil => (intCeil a b).map .int
  | .mod => (pyMod a b).map .int
  | .pow =>
    if b < 0 then (if a = 0 then none else some (.float (Float.ofInt a ^ Float.ofInt b)))
    else some (.int (a ^ b.toNat))
  | .min => some (.int (Min.min a b))
  | .max => some (.int (Max.max a b))
  | .bitAnd => some (.int (Int.land a b))
  | .bitOr => some (.int (Int.lor a b))
  | .bitXor => some (.int (Int.xor a b))
  | .shl => if b < 0 then none else some (.int (a * 2 ^ b.toNat))
  | .shr => if b < 0 then none else some (.int (Int.fdiv a (2 ^ b.toNat)))
  | .eq => some (.bool (a == b))
  | .ne => some (.bool (a != b))
  | .lt => some (.bool (decide (a < b)))
  | .le => some (.bool (decide (a ≤ b)))
  | .gt => some (.bool (decide (a > b)))
  | .ge => some (.bool (decide (a ≥ b)))
  | .and => some (.bool (a != 0 && b != 0))
  | .or => some (.bool (a != 0 || b != 0))

/-- Floating-point (or mixed) arithmetic. -/
def floatArith (op : BinOp) (a b : Float) : Option Val :=
  match op with
  | .add => some (.float (a + b))
  | .sub => some (.float (a - b))
  | .mul => some (.float (a * b))
  | .div => some (.float (a / b))
  | .intFloor => some (.float (Float.floor (a / b)))
  | .intCeil => some (.float (Float.ceil (a / b)))
  | .mod => some (.float (a - b * Float.floor (a / b)))
  | .pow => some (.float (a ^ b))
  | .min => some (.float (if a ≤ b then a else b))
  | .max => some (.float (if a ≤ b then b else a))
  | .eq => some (.bool (a == b))
  | .ne => some (.bool (a != b))
  | .lt => some (.bool (a < b))
  | .le => some (.bool (a ≤ b))
  | .gt => some (.bool (a > b))
  | .ge => some (.bool (a ≥ b))
  | .and => some (.bool (a != 0.0 && b != 0.0))
  | .or => some (.bool (a != 0.0 || b != 0.0))
  | .bitAnd | .bitOr | .bitXor | .shl | .shr => none

def arith (op : BinOp) (x y : Val) : Option Val :=
  match x, y with
  | .int a, .int b => intArith op a b
  | .bool a, .bool b =>
    match op with
    | .and => some (.bool (a && b))
    | .or => some (.bool (a || b))
    | .eq => some (.bool (a == b))
    | .ne => some (.bool (a != b))
    | .bitAnd => some (.bool (a && b))
    | .bitOr => some (.bool (a || b))
    | .bitXor => some (.bool (a != b))
    | _ => intArith op (if a then 1 else 0) (if b then 1 else 0)
  | .bool a, .int b => intArith op (if a then 1 else 0) b
  | .int a, .bool b => intArith op a (if b then 1 else 0)
  | _, _ => floatArith op x.toFloat y.toFloat

def unary (op : UnOp) : Val → Option Val
  | .int a =>
    match op with
    | .neg => some (.int (-a))
    | .not => some (.bool (a == 0))
    | .bitNot => some (.int (Int.not a))
    | .abs => some (.int (Int.natAbs a))
  | .float f =>
    match op with
    | .neg => some (.float (-f))
    | .not => some (.bool (f == 0.0))
    | .bitNot => none
    | .abs => some (.float (Float.abs f))
  | .bool b =>
    match op with
    | .neg => some (.int (if b then -1 else 0))
    | .not => some (.bool (!b))
    | .bitNot => some (.bool (!b))
    | .abs => some (.int (if b then 1 else 0))

end Eval

namespace Expr

mutual
/-- Evaluate an expression.  `none` means the expression is undefined under the
given valuation (unbound symbol, division by zero, type error, failed read). -/
def eval (I : ExprInterp) (ρ : Valuation) : Expr → Option Val
  | lit (.int v) => some (.int v)
  | lit (.float m e) => some (.float (Lit.floatVal m e))
  | lit (.bool b) => some (.bool b)
  | sym s => ρ s
  | un op a => do Eval.unary op (← a.eval I ρ)
  | bin op a b => do Eval.arith op (← a.eval I ρ) (← b.eval I ρ)
  | ite c t e => do
    let cv ← c.eval I ρ
    let b ← cv.toBool?
    if b then t.eval I ρ else e.eval I ρ
  | app f args => do I.app f (← evalList I ρ args)
  | subscript d idx => do
    let vs ← evalList I ρ idx
    let is ← vs.mapM Val.toInt?
    I.readData d is
  | attr b f => do I.readAttr (← b.eval I ρ) f

def evalList (I : ExprInterp) (ρ : Valuation) : Exprs → Option (List Val)
  | .nil => some []
  | .cons e es => do
    let v ← e.eval I ρ
    let vs ← evalList I ρ es
    pure (v :: vs)
end

/-- Evaluate to an integer. -/
def evalInt (I : ExprInterp) (ρ : Valuation) (e : Expr) : Option Int :=
  e.eval I ρ >>= Val.toInt?

/-- Evaluate to a boolean (integers are truthy when non-zero, as in Python). -/
def evalBool (I : ExprInterp) (ρ : Valuation) (e : Expr) : Option Bool :=
  e.eval I ρ >>= Val.toBool?

end Expr

/-- The pure interpretation: no opaque functions, no data reads. -/
def ExprInterp.pure : ExprInterp where
  app _ _ := none
  readData _ _ := none

end Dace
