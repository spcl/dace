-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Data
import Dace.Core.Parse
import Dace.Core.SymArith

/-!
# Memlets

Mirrors `dace/memlet.py` in **normalized** form: instead of DaCe's
`subset` / `other_subset` / `_is_data_src` triple, a memlet carries the source
and destination subsets explicitly.  The JSON importer applies
`Memlet.try_initialize`'s rules (`dace/memlet.py:421-573`) once.

Empty memlets (`Memlet()` with no data) are **not** memlets here: they are a
separate dependency-edge kind in the SDFG syntax.
-/

namespace Dace

/-- Write-conflict resolution operators.  The built-in reductions are the
ones DaCe detects from the lambda (`dace.frontend.operations.detect_reduction_type`);
anything else is kept as a parsed lambda `fun a b => body` so that it can be
evaluated (`Decisions.md#D26`). -/
inductive WcrOp where
  | sum | product | min | max
  | logicalAnd | logicalOr | logicalXor
  | bitwiseAnd | bitwiseOr | bitwiseXor
  | sub | div | exchange
  | custom (a b : Sym) (body : Expr)
  deriving DecidableEq, Repr, Inhabited

namespace WcrOp

/-- Strip surrounding whitespace. -/
private def strip (s : String) : String :=
  String.ofList ((s.toList.dropWhile Char.isWhitespace).reverse.dropWhile Char.isWhitespace).reverse

/-- Structural detection of built-in reductions, mirroring
`detect_reduction_type`: the body is matched against `a ⊕ b` (and the
commuted form for commutative operators). -/
def detect (a b : Sym) (body : Expr) : WcrOp :=
  let isAB (x y : Expr) : Bool := (x = .sym a && y = .sym b)
  let isBA (x y : Expr) : Bool := (x = .sym b && y = .sym a)
  let comm (x y : Expr) : Bool := isAB x y || isBA x y
  match body with
  | .bin .add x y => if comm x y then .sum else .custom a b body
  | .bin .mul x y => if comm x y then .product else .custom a b body
  | .bin .min x y => if comm x y then .min else .custom a b body
  | .bin .max x y => if comm x y then .max else .custom a b body
  | .bin .and x y => if comm x y then .logicalAnd else .custom a b body
  | .bin .or x y => if comm x y then .logicalOr else .custom a b body
  | .bin .bitAnd x y => if comm x y then .bitwiseAnd else .custom a b body
  | .bin .bitOr x y => if comm x y then .bitwiseOr else .custom a b body
  | .bin .bitXor x y => if comm x y then .bitwiseXor else .custom a b body
  | .bin .sub x y => if isAB x y then .sub else .custom a b body
  | .bin .div x y => if isAB x y then .div else .custom a b body
  | .sym s => if s = b then .exchange else .custom a b body
  | _ => .custom a b body

/-- Remove enclosing parentheses: DaCe serializes `(lambda a, b: (a + b))`. -/
private partial def unparen (s : String) : String :=
  let s := strip s
  if s.startsWith "(" && s.endsWith ")" && s.length ≥ 2 then
    let inner := String.ofList (s.toList.drop 1 |>.dropLast)
    if strip inner |>.startsWith "lambda" then unparen inner else s
  else s

/-- Parse a WCR lambda `lambda a, b: body` (optionally parenthesized).
`none` if the text is not a two-argument lambda with a parsable body. -/
def ofLambda? (lam : String) : Option WcrOp := do
  let lam := unparen lam
  let colon ← (lam.toList.findIdx? (· = ':'))
  let header := String.ofList (lam.toList.take colon)
  let body := String.ofList (lam.toList.drop (colon + 1))
  let header := strip header
  guard (header.startsWith "lambda")
  let params := (String.ofList (header.toList.drop "lambda".length)).splitOn "," |>.map strip
  match params with
  | [a, b] =>
    let e ← Expr.parse? body
    pure (detect a b e)
  | _ => none

/-- The lambda text of an operator, as DaCe would write it. -/
def toLambda : WcrOp → String
  | sum => "lambda a, b: a + b"
  | product => "lambda a, b: a * b"
  | min => "lambda a, b: min(a, b)"
  | max => "lambda a, b: max(a, b)"
  | logicalAnd => "lambda a, b: a and b"
  | logicalOr => "lambda a, b: a or b"
  | logicalXor => "lambda a, b: a != b"
  | bitwiseAnd => "lambda a, b: a & b"
  | bitwiseOr => "lambda a, b: a | b"
  | bitwiseXor => "lambda a, b: a ^ b"
  | sub => "lambda a, b: a - b"
  | div => "lambda a, b: a / b"
  | exchange => "lambda a, b: b"
  | custom a b body => s!"lambda {a}, {b}: {body.toPy}"

end WcrOp

/-- A (non-empty) memlet in normalized form. -/
structure Memlet where
  /-- The container the memlet is attached to (`Memlet.data`). -/
  data : Name
  /-- Whether `data` is the source side (`_is_data_src`). -/
  isDataSrc : Bool := true
  srcSubset : Option Subset := none
  dstSubset : Option Subset := none
  volume : Expr := 0
  dynamic : Bool := false
  wcr : Option WcrOp := none
  wcrNonatomic : Bool := false
  allowOob : Bool := false
  deriving DecidableEq, Repr, Inhabited

namespace Memlet

/-- The subset on the `data` side (DaCe's `Memlet.subset`). -/
def dataSubset (m : Memlet) : Option Subset :=
  if m.isDataSrc then m.srcSubset else m.dstSubset

/-- The subset on the other side (DaCe's `Memlet.other_subset`). -/
def otherSubset (m : Memlet) : Option Subset :=
  if m.isDataSrc then m.dstSubset else m.srcSubset

/-- A memlet reading/writing `data[subset]` with the subset on the data side. -/
def simple (data : Name) (subset : Subset) (wcr : Option WcrOp := none) : Memlet :=
  { data, isDataSrc := true, srcSubset := some subset, dstSubset := none,
    volume := (subset.asRange?.map Range.numElementsExpr).getD 1, wcr }

/-- Build from DaCe's triple `(data, subset, other_subset, is_data_src)`. -/
def ofDace (data : Name) (subset otherSubset : Option Subset) (isDataSrc : Bool) : Memlet :=
  if isDataSrc then { data, isDataSrc, srcSubset := subset, dstSubset := otherSubset }
  else { data, isDataSrc, srcSubset := otherSubset, dstSubset := subset }

def hasWcr (m : Memlet) : Bool := m.wcr.isSome

def freeSyms (m : Memlet) : List Sym :=
  (m.srcSubset.map Subset.freeSyms).getD [] ++ (m.dstSubset.map Subset.freeSyms).getD [] ++ m.volume.freeSyms

def subst (σ : Sym → Option Expr) (m : Memlet) : Memlet :=
  { m with srcSubset := m.srcSubset.map (Subset.subst σ), dstSubset := m.dstSubset.map (Subset.subst σ),
           volume := m.volume.subst σ }

end Memlet

/-! ## Omitted sides of copies (`Decisions.md#D9`) -/

/-- The symbolic number of elements of a range with unit steps and tiles,
`∏ (hi - lo + 1)`, as a polynomial (`Range.num_elements`); `none` when a
step or tile is not `1` or a bound is no polynomial. -/
def Range.countPoly? (r : Range) : Option Poly :=
  r.dims.foldlM (fun acc d => do
    if d.step.toPoly? = some (Poly.const 1) ∧ d.tile.toPoly? = some (Poly.const 1) then
      let p ← (d.hi - d.lo + 1).toPoly?
      pure (Poly.mul acc p)
    else none) (Poly.const 1)

/-- Do two subsets provably have different numbers of elements (their
symbolic counts differ by a non-zero constant)? -/
def Subset.countsDiffer (a b : Subset) : Bool :=
  match a, b with
  | .range ra, .range rb =>
    match ra.countPoly?, rb.countPoly? with
    | some p, some q =>
      match (Poly.sub p q).const? with
      | some c => c != 0
      | none => false
    | _, _ => false
  | _, _ => false

/-- The side a copy memlet omits, as `InsertExplicitCopies` derives it
(`_derive_matching_dst_subset`): the full range of the container `desc`
unless the element counts provably differ, then the known side's subset.
"Provably" is decided on the polynomial normal form of the counts (SymPy may
prove more, e.g. with symbol assumptions). -/
def deriveCopySide (known : Subset) (desc : DataDesc) : Subset :=
  let full : Subset := .range desc.fullRange
  if known.countsDiffer full then known else full

/-- Containers between which `InsertExplicitCopies` lifts a copy to a copy
node: arrays or scalars (array views and references included) of the same
element type. -/
def DataDesc.liftableCopy (a b : DataDesc) : Bool :=
  let ok (d : DataDesc) : Bool :=
    match d.kind with
    | .array | .scalar | .view .array | .reference .array => true
    | _ => false
  ok a && ok b && a.dtype == b.dtype

/-- Fill the omitted side of a copy between two access nodes
(`InsertExplicitCopies._replace_direct_copies`): from the known side by
`deriveCopySide`; the `data` side is the full range when absent. -/
def Memlet.fillDirect (m : Memlet) (srcDesc dstDesc : DataDesc) : Memlet :=
  let full (d : DataDesc) : Subset := .range d.fullRange
  match m.srcSubset, m.dstSubset with
  | some _, some _ => m
  | some s, none => { m with dstSubset := some (deriveCopySide s dstDesc) }
  | none, some d => { m with srcSubset := some (deriveCopySide d srcDesc) }
  | none, none =>
    if m.isDataSrc then
      let s := full srcDesc
      { m with srcSubset := some s, dstSubset := some (deriveCopySide s dstDesc) }
    else
      let d := full dstDesc
      { m with srcSubset := some (deriveCopySide d srcDesc), dstSubset := some d }

/-- Fill the subsets of a map staging copy (`InsertExplicitCopies._lift_staging_edge`):
the outer side is the memlet's subset on that side, else its `data` subset;
the inner side is derived from the outer one unless the memlet names both
sides.  `stageIn`: the copy enters the map (outer = source). -/
def Memlet.fillStaging (m : Memlet) (stageIn : Bool) (outerDesc innerDesc : DataDesc) : Memlet :=
  let dataSide := m.dataSubset.getD (.range (if stageIn = m.isDataSrc then outerDesc else innerDesc).fullRange)
  let outerSide := if stageIn then m.srcSubset else m.dstSubset
  let innerSide := if stageIn then m.dstSubset else m.srcSubset
  let outer := outerSide.getD dataSide
  let inner := match innerSide, m.otherSubset with
    | some i, some _ => i
    | _, _ => deriveCopySide outer innerDesc
  if stageIn then { m with srcSubset := some outer, dstSubset := some inner }
  else { m with srcSubset := some inner, dstSubset := some outer }

end Dace
