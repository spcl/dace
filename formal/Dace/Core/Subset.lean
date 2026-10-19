-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Expr

/-!
# Subsets

Mirrors `dace/subsets.py`.  A `Range` is a list of dimensions `lo:hi:step`
with **inclusive** upper bound (DaCe convention) and an optional tile size;
an *index* is a range whose dimensions all have `lo = hi` (DaCe has no
separate `Indices` class any more).  `SubsetUnion` is a list of ranges.

The semantic content of a subset is the list of integer index vectors it
denotes under a valuation, `Range.points`, enumerated in lexicographic
(row-major) order.  This order is what copies and the executable semantics
use (`Decisions.md#D10`).
-/

namespace Dace

/-- An integer index vector. -/
abbrev Idx := List Int

/-- One dimension `lo:hi:step` (inclusive `hi`) with a tile size. -/
structure Dim where
  lo : Expr
  hi : Expr
  step : Expr := 1
  tile : Expr := 1
  deriving DecidableEq, Repr, Inhabited

namespace Dim

/-- The dimension `[i]` selecting a single index. -/
def index (i : Expr) : Dim := { lo := i, hi := i }

/-- Syntactically a single index? -/
def isIndex (d : Dim) : Bool := d.lo = d.hi

/-- Symbolic number of elements `(hi - lo + step) // step` (as `Range.size`
in DaCe). -/
def sizeExpr (d : Dim) : Expr :=
  .bin .intFloor (d.hi - d.lo + d.step) d.step

/-- The elements `lo, lo+step, …, ≤ hi`.  `none` if a bound is undefined or
the step is not positive (DaCe requires positive steps). -/
def elems (I : ExprInterp) (ρ : Valuation) (d : Dim) : Option (List Int) := do
  let lo ← d.lo.evalInt I ρ
  let hi ← d.hi.evalInt I ρ
  let step ← d.step.evalInt I ρ
  if step ≤ 0 then none
  else
    let n : Nat := if hi < lo then 0 else ((hi - lo) / step).toNat + 1
    pure ((List.range n).map fun k => lo + step * k)

def subst (σ : Sym → Option Expr) (d : Dim) : Dim :=
  { lo := d.lo.subst σ, hi := d.hi.subst σ, step := d.step.subst σ, tile := d.tile.subst σ }

def freeSyms (d : Dim) : List Sym :=
  d.lo.freeSyms ++ d.hi.freeSyms ++ d.step.freeSyms ++ d.tile.freeSyms

end Dim

/-- A multidimensional range. -/
structure Range where
  dims : List Dim
  deriving DecidableEq, Repr, Inhabited

namespace Range

def ofDims (ds : List Dim) : Range := ⟨ds⟩

/-- Number of dimensions. -/
def ndim (r : Range) : Nat := r.dims.length

/-- Does any bound or step of the range mention container `n` (`Expr.mentions`)? -/
def mentions (r : Range) (n : Name) : Bool :=
  r.dims.any fun d => d.lo.mentions n || d.hi.mentions n || d.step.mentions n || d.tile.mentions n

/-- The range `[i₀, i₁, …]` of a single point. -/
def ofIndices (is : List Expr) : Range := ⟨is.map Dim.index⟩

/-- The full range `0:shape₀-1, 0:shape₁-1, …` of an array (`Range.from_array`). -/
def ofShape (shape : List Expr) : Range := ⟨shape.map fun s => { lo := 0, hi := s - 1 }⟩

def isIndex (r : Range) : Bool := r.dims.all Dim.isIndex

def isEmpty (r : Range) : Bool := r.dims.isEmpty

/-- Lexicographic Cartesian product of per-dimension element lists. -/
def product : List (List Int) → List Idx
  | [] => [[]]
  | d :: ds => d.flatMap fun i => (product ds).map (i :: ·)

/-- The index vectors denoted by the range, in row-major order. -/
def points (I : ExprInterp) (ρ : Valuation) (r : Range) : Option (List Idx) := do
  let es ← r.dims.mapM (Dim.elems I ρ)
  pure (product es)

/-- Number of points. -/
def size (I : ExprInterp) (ρ : Valuation) (r : Range) : Option Nat :=
  (r.points I ρ).map List.length

/-- Symbolic number of elements (product of per-dimension sizes), as
`Range.num_elements`. -/
def numElementsExpr (r : Range) : Expr :=
  r.dims.foldl (fun acc d => acc * d.sizeExpr) (1 : Expr)

/-- Symbolic per-dimension sizes (`Range.size()`). -/
def sizeExprs (r : Range) : List Expr := r.dims.map Dim.sizeExpr

/-- Symbolic lower corner (`Range.min_element()`). -/
def minElement (r : Range) : List Expr := r.dims.map Dim.lo

/-- `offset(other, negative)`: shift every dimension by the corresponding
offset (subtract when `negative`).  Mirrors `Range.offset`. -/
def offsetBy (r : Range) (off : List Expr) (negative : Bool) : Range :=
  ⟨(r.dims.zip off).map fun (d, o) =>
    let f := fun e => if negative then e - o else e + o
    { d with lo := f d.lo, hi := f d.hi }⟩

/-- Composition `outer ∘ inner` for equal dimensionality: the subset of the
container obtained by selecting `inner` *within* `outer`
(`Range.compose`, same-dimension case). -/
def compose (outer inner : Range) : Option Range :=
  if outer.ndim = inner.ndim then
    some ⟨(outer.dims.zip inner.dims).map fun (o, i) =>
      { lo := o.lo + i.lo * o.step
        hi := o.lo + i.hi * o.step
        step := o.step * i.step
        tile := i.tile }⟩
  else none

/-- Remove the dimensions at the given positions (`Range.squeeze` /
`Range.pop`). -/
def dropDims (r : Range) (positions : List Nat) : Range :=
  ⟨(r.dims.zipIdx).filterMap fun (d, k) => if positions.contains k then none else some d⟩

/-- Positions of syntactic index (size-one) dimensions. -/
def indexDims (r : Range) : List Nat :=
  (r.dims.zipIdx).filterMap fun (d, k) => if d.isIndex then some k else none

/-- `Range.squeeze`: drop all syntactic index dimensions except `keep`. -/
def squeeze (r : Range) (keep : List Nat := []) : Range :=
  r.dropDims (r.indexDims.filter fun k => !keep.contains k)

/-- `Range.unsqueeze`: insert index-`0` dimensions at the given (sorted)
positions. -/
def unsqueeze (r : Range) (positions : List Nat) : Range :=
  let rec go (ds : List Dim) (k : Nat) (ps : List Nat) : List Dim :=
    match ps with
    | [] => ds
    | p :: ps' =>
      if p = k then Dim.index 0 :: go ds (k + 1) ps'
      else match ds with
        | [] => Dim.index 0 :: go [] (k + 1) ps'
        | d :: ds' => d :: go ds' (k + 1) (p :: ps')
  ⟨go r.dims 0 positions⟩

def subst (σ : Sym → Option Expr) (r : Range) : Range := ⟨r.dims.map (Dim.subst σ)⟩

def freeSyms (r : Range) : List Sym := r.dims.flatMap Dim.freeSyms

/-- Linear offset of an index vector under strides (`Subset.at`). -/
def linearize (idx : Idx) (strides : List Int) : Int :=
  ((idx.zip strides).map fun (i, s) => i * s).foldl (· + ·) 0

end Range

/-- Subsets: a single range or a union of ranges (`SubsetUnion`). -/
inductive Subset where
  | range (r : Range)
  | union (rs : List Range)
  deriving DecidableEq, Repr, Inhabited

namespace Subset

/-- Points of a subset: ranges in order, duplicates kept (a union's
enumeration order is not semantically relevant; unions only occur as
footprint annotations). -/
def points (I : ExprInterp) (ρ : Valuation) : Subset → Option (List Idx)
  | range r => r.points I ρ
  | union rs => do
    let ps ← rs.mapM (Range.points I ρ)
    pure ps.flatten

def ndim : Subset → Nat
  | range r => r.ndim
  | union [] => 0
  | union (r :: _) => r.ndim

def subst (σ : Sym → Option Expr) : Subset → Subset
  | range r => range (r.subst σ)
  | union rs => union (rs.map (Range.subst σ))

def freeSyms : Subset → List Sym
  | range r => r.freeSyms
  | union rs => rs.flatMap Range.freeSyms

/-- The underlying range when the subset is a single range. -/
def asRange? : Subset → Option Range
  | range r => some r
  | union [r] => some r
  | union _ => none

end Subset

/-! ## Basic lemmas about enumeration -/

namespace Range

@[simp] theorem product_nil : product [] = [[]] := rfl

@[simp] theorem product_cons (d : List Int) (ds : List (List Int)) :
    product (d :: ds) = d.flatMap fun i => (product ds).map (i :: ·) := rfl

theorem length_product (ds : List (List Int)) :
    (product ds).length = (ds.map List.length).prod := by
  induction ds with
  | nil => simp
  | cons d ds ih =>
    simp [List.length_flatMap, List.length_map, ih, List.map_const', List.sum_replicate]

/-- An index vector is a point of the product iff it picks one element per
dimension. -/
theorem mem_product {ds : List (List Int)} {idx : Idx} :
    idx ∈ product ds ↔ List.Forall₂ (· ∈ ·) idx ds := by
  induction ds generalizing idx with
  | nil => simp [List.forall₂_nil_right_iff]
  | cons d ds ih =>
    simp only [product_cons, List.mem_flatMap, List.mem_map]
    constructor
    · rintro ⟨i, hi, rest, hrest, rfl⟩
      exact List.Forall₂.cons hi (ih.mp hrest)
    · intro h
      cases h with
      | cons hi hrest => exact ⟨_, hi, _, ih.mpr hrest, rfl⟩

theorem length_of_mem_product {ds : List (List Int)} {idx : Idx} (h : idx ∈ product ds) :
    idx.length = ds.length :=
  (mem_product.mp h).length_eq

end Range

end Dace
