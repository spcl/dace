-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Subset

/-!
# Data descriptors

Mirrors `dace/data/core.py`.  All descriptors share one record (`DataDesc`)
with a `kind` discriminator; the fields that are meaningless for a kind are
left at their defaults.

Structures and container arrays are represented *flat* (`Decisions.md#D21`):
the descriptor repository of an SDFG contains an entry for every structure
member under its dotted name (`s.m`, as `AccessNode('s.m')` addresses it) and
one entry for the element descriptor of a container array; the structure
descriptor only lists its member names.
-/

namespace Dace

/-- `dace.dtypes.AllocationLifetime`. -/
inductive Lifetime where
  | scope | state | sdfg | global | persistent | external
  deriving DecidableEq, Repr, Inhabited

/-- What a view or reference may point at. -/
inductive BaseKind where
  | array | structure | containerArray
  deriving DecidableEq, Repr, Inhabited

inductive DataKind where
  | array
  | scalar
  | stream (bufferSize : Expr)
  /-- `ArrayView` / `StructureView` / `ContainerView`. -/
  | view (base : BaseKind)
  /-- `ArrayReference` / `StructureReference` / `ContainerArrayReference`. -/
  | reference (base : BaseKind)
  /-- A structure with the given type name and member names; member
      descriptors live under `<name>.<member>` in the repository. -/
  | structure (typeName : String) (members : List String)
  /-- A container array whose element descriptor lives under `elem`. -/
  | containerArray (elem : Name)
  deriving DecidableEq, Repr, Inhabited

/-- A data descriptor. -/
structure DataDesc where
  kind : DataKind := .array
  dtype : Dtype := .float 64
  shape : List Expr := []
  strides : List Expr := []
  offset : List Expr := []
  totalSize : Expr := 0
  startOffset : Expr := 0
  transient : Bool := false
  storage : String := "Default"
  lifetime : Lifetime := .scope
  mayAlias : Bool := false
  allowConflicts : Bool := false
  deriving DecidableEq, Repr, Inhabited

namespace DataDesc

def isTransient (d : DataDesc) : Bool := d.transient

def isScalar (d : DataDesc) : Bool := d.kind matches .scalar

def isStream (d : DataDesc) : Bool := d.kind matches .stream _

def isView (d : DataDesc) : Bool := d.kind matches .view _

def isReference (d : DataDesc) : Bool := d.kind matches .reference _

def isStructure (d : DataDesc) : Bool := d.kind matches .structure _ _

def ndim (d : DataDesc) : Nat := d.shape.length

/-- The full range `0:shape-1` of the descriptor. -/
def fullRange (d : DataDesc) : Range := Range.ofShape d.shape

/-- Symbolic number of elements. -/
def numElements (d : DataDesc) : Expr :=
  d.shape.foldl (fun acc s => acc * s) (1 : Expr)

/-- Row-major strides for a shape (`Data.strides_from_layout` default). -/
def cStrides (shape : List Expr) : List Expr :=
  let rec go : List Expr → List Expr × Expr
    | [] => ([], 1)
    | s :: rest =>
      let (strs, prod) := go rest
      (prod :: strs, prod * s)
  (go shape).1

/-- Effective strides: the declared ones, or row-major if none were given. -/
def effStrides (d : DataDesc) : List Expr :=
  if d.strides.length = d.shape.length then d.strides else cStrides d.shape

/-- A scalar descriptor. -/
def scalar (dtype : Dtype) (transient : Bool := false) : DataDesc :=
  { kind := .scalar, dtype, shape := [1], strides := [1], totalSize := 1, transient }

/-- An array descriptor with row-major strides. -/
def array (dtype : Dtype) (shape : List Expr) (transient : Bool := false) : DataDesc :=
  { kind := .array, dtype, shape, strides := cStrides shape,
    totalSize := shape.foldl (· * ·) (1 : Expr), transient }

/-- Linear element offset of an index vector under the descriptor's strides
and start offset (`cpp_offset_expr` without the per-dimension `offset`, which
DaCe folds into the subset at use sites). -/
def linearOffset (I : ExprInterp) (ρ : Valuation) (d : DataDesc) (idx : Idx) : Option Int := do
  let strides ← d.effStrides.mapM (Expr.evalInt I ρ)
  let start ← d.startOffset.evalInt I ρ
  if strides.length = idx.length then
    pure (start + Range.linearize idx strides)
  else none

/-- Structural equivalence as used for nested-SDFG connectors
(`Data.is_equivalent`): same kind, dtype, shape and strides; `offset`,
`transient`, storage and lifetime are ignored (`Decisions.md#D15`). -/
def isEquivalent (a b : DataDesc) : Bool :=
  a.kind = b.kind && a.dtype = b.dtype && a.shape = b.shape && a.effStrides = b.effStrides

end DataDesc

end Dace
