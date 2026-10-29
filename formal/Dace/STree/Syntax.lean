-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Syntax

/-!
# Schedule tree syntax

A 1:1 rendering of the node catalog of
`dace/sdfg/analysis/schedule_tree/treenodes.py`: every concrete node class
is a constructor of `SNode` (scopes carry their `children`), and
`ScheduleTreeRoot` is the structure `STree` carrying the unified descriptor
repository (containers, symbols, constants, argument names, callback
mapping).  `StateBoundaryNode` is a conversion artifact of tree→SDFG and
has no constructor (`Decisions.md#D19`, `#D48`); the importer rejects it.

Design notes (`Decisions.md#D48`):

* The four loop classes (`LoopScope`, `ForScope`, `WhileScope`,
  `DoWhileScope`) are separate constructors over the same `LoopMeta` (the
  `LoopRegion` metadata of the serialized tree); `LoopMeta.variant` mirrors
  `treenodes.loop_variant` and drives both printing and the semantics.
* Code blocks that the semantics evaluates (conditions, assignment values,
  loop statements) are parsed at import into `Expr` but keep their source
  text (`PyExpr`, `PyAssign`) so that `prettyPrint` reproduces DaCe's
  `as_string`.
* Memlets reuse `Dace.Memlet`.  In a tree, `Memlet.data` names the container
  the node accesses and `Memlet.dataSubset` / `Memlet.otherSubset` are the
  meaningful subsets (`sdfg_to_tree.normalize_memlet`).
* `prettyPrint` reproduces `ScheduleTreeRoot.as_string()`; symbolic
  expressions are printed with a SymPy-like normal form (`Decisions.md#D56`).
-/

set_option autoImplicit false

namespace Dace

/-- A Python expression kept with its source text (`CodeBlock.as_string`). -/
structure PyExpr where
  code : CodeBlock
  expr : Expr
  deriving DecidableEq, Repr, Inhabited

/-- A Python assignment statement `lhs = rhs` kept with its source text. -/
structure PyAssign where
  code : CodeBlock
  lhs : Sym
  rhs : Expr
  deriving DecidableEq, Repr, Inhabited

/-- Loop metadata of a `LoopScope` (the fields of `LoopRegion` that the
serializer stores, `serialize._loop_to_json`). -/
structure LoopMeta where
  label : String := ""
  loopVar : Option Sym := none
  init : Option PyAssign := none
  cond : PyExpr := { code := { code := "True" }, expr := .lit (.bool true) }
  update : Option PyAssign := none
  inverted : Bool := false
  updateBeforeCond : Bool := false
  deriving DecidableEq, Repr, Inhabited

/-- `treenodes.loop_variant`. -/
inductive LoopVariant where
  | «for» | «while» | doWhile | doFor | doForUncondIncrement
  deriving DecidableEq, Repr, Inhabited

namespace LoopMeta

/-- Exactly `treenodes.loop_variant`: a `for`-family loop needs an update
statement, an init statement and a loop variable. -/
def variant (l : LoopMeta) : LoopVariant :=
  if l.update.isSome && l.init.isSome && l.loopVar.isSome then
    if l.inverted then (if l.updateBeforeCond then .doForUncondIncrement else .doFor) else .for
  else if l.inverted then .doWhile else .while

/-- The init assignment as the semantics consumes it. -/
def initAssign (l : LoopMeta) : Option (Sym × Expr) := l.init.map fun a => (a.lhs, a.rhs)

def updateAssign (l : LoopMeta) : Option (Sym × Expr) := l.update.map fun a => (a.lhs, a.rhs)

end LoopMeta

/-- `in_memlets` / `out_memlets` of a code node: per connector (`dict`) or,
for connector-less library nodes, a plain set (`serialize._memlet_mapping_to_json`). -/
inductive MemletMap where
  | dict (items : List (Conn × Memlet))
  | set (items : List Memlet)
  deriving DecidableEq, Repr, Inhabited

namespace MemletMap

def memlets : MemletMap → List Memlet
  | dict items => items.map Prod.snd
  | set items => items

/-- Memlets attached to connector `c`; set-kind memlets are all attached to
the empty connector name (`Decisions.md#D57`). -/
def on (m : MemletMap) (c : Conn) : List Memlet :=
  match m with
  | dict items => (items.filter (·.1 = c)).map Prod.snd
  | set items => if c = "" then items else []

/-- Connector/memlet pairs (set-kind memlets under the empty connector). -/
def pairs : MemletMap → List (Conn × Memlet)
  | dict items => items
  | set items => items.map fun m => ("", m)

def isEmpty (m : MemletMap) : Bool := m.memlets.isEmpty

end MemletMap

/-- The source of a reference set: a data descriptor, or a code node
elsewhere in the tree (referenced by its guid and class name). -/
inductive RefSetSource where
  | data (desc : DataDesc)
  | codeNode (guid : String) (className : String)
  deriving DecidableEq, Repr, Inhabited

/-- Schedule tree nodes: one constructor per concrete class of
`treenodes.py` (except the root and `StateBoundaryNode`). -/
inductive SNode where
  /-- `NamedRegionScope`: a transparent labeled grouping. -/
  | namedRegion (label : String) (children : List SNode)
  /-- `GBlock`: general (unstructured) control flow with labels and gotos. -/
  | gblock (children : List SNode)
  /-- `StateLabel`: a goto target. -/
  | stateLabel (name : String)
  /-- `GotoNode`; `none` = exit / return. -/
  | goto (target : Option String)
  /-- `AssignNode`: a symbol assignment. -/
  | assign (name : Sym) (value : PyExpr)
  /-- `LoopScope` (general loop). -/
  | loopScope (loop : LoopMeta) (children : List SNode)
  | forScope (loop : LoopMeta) (children : List SNode)
  | whileScope (loop : LoopMeta) (children : List SNode)
  | doWhileScope (loop : LoopMeta) (children : List SNode)
  | ifScope (cond : PyExpr) (children : List SNode)
  /-- `StateIfScope`: a conditional state transition. -/
  | stateIfScope (cond : PyExpr) (children : List SNode)
  | breakNode
  | continueNode
  | elifScope (cond : PyExpr) (children : List SNode)
  | elseScope (children : List SNode)
  | mapScope (map : MapNode) (children : List SNode)
  | consumeScope (consume : ConsumeNode) (children : List SNode)
  | taskletNode (t : Tasklet) (ins outs : MemletMap)
  /-- `LibraryCall`; `props` are the node's own properties as `(name, text)`
      pairs in declaration order (`LibraryCall.own_properties`), for printing
      only (`Decisions.md#D56`). -/
  | libraryCall (l : LibNode) (props : List (String × String)) (ins outs : MemletMap)
  /-- `CopyNode`: `target[other_subset] = memlet.data[memlet.subset]`. -/
  | copyNode (target : Name) (memlet : Memlet)
  /-- `DynScopeCopyNode`: a dynamic input of the next dataflow scope. -/
  | dynScopeCopy (target : Sym) (memlet : Memlet)
  /-- `ViewNode`: `target` views `source[memlet.subset]`. -/
  | viewNode (target source : Name) (memlet : Memlet) (srcDesc viewDesc : DataDesc)
  /-- `RefSetNode`: `target` is set to `memlet.data[memlet.subset]`. -/
  | refSetNode (target : Name) (memlet : Memlet) (src : RefSetSource) (refDesc : DataDesc)
  deriving Inhabited

/-- `ScheduleTreeRoot` with its descriptor repository.  `global_code`,
`init_code` and `exit_code` are not modeled (`Decisions.md#D48`). -/
structure STree where
  name : String
  containers : List (Name × DataDesc) := []
  symbols : List (Sym × Dtype) := []
  constants : List (Name × DataDesc × Option Val) := []
  argNames : List Name := []
  callbackMapping : List (String × String) := []
  children : List SNode := []

namespace SNode

/-- The Python class name. -/
def typeName : SNode → String
  | namedRegion .. => "NamedRegionScope"
  | gblock .. => "GBlock"
  | stateLabel .. => "StateLabel"
  | goto .. => "GotoNode"
  | assign .. => "AssignNode"
  | loopScope .. => "LoopScope"
  | forScope .. => "ForScope"
  | whileScope .. => "WhileScope"
  | doWhileScope .. => "DoWhileScope"
  | ifScope .. => "IfScope"
  | stateIfScope .. => "StateIfScope"
  | breakNode => "BreakNode"
  | continueNode => "ContinueNode"
  | elifScope .. => "ElifScope"
  | elseScope .. => "ElseScope"
  | mapScope .. => "MapScope"
  | consumeScope .. => "ConsumeScope"
  | taskletNode .. => "TaskletNode"
  | libraryCall .. => "LibraryCall"
  | copyNode .. => "CopyNode"
  | dynScopeCopy .. => "DynScopeCopyNode"
  | viewNode .. => "ViewNode"
  | refSetNode .. => "RefSetNode"

/-- Children of a scope node (`none` for leaves). -/
def children? : SNode → Option (List SNode)
  | namedRegion _ cs | gblock cs | loopScope _ cs | forScope _ cs | whileScope _ cs | doWhileScope _ cs
  | ifScope _ cs | stateIfScope _ cs | elifScope _ cs | elseScope cs | mapScope _ cs | consumeScope _ cs => some cs
  | _ => none

def isScope (n : SNode) : Bool := n.children?.isSome

/-- Loop scopes of all four classes. -/
def loop? : SNode → Option (LoopMeta × List SNode)
  | loopScope l cs | forScope l cs | whileScope l cs | doWhileScope l cs => some (l, cs)
  | _ => none

def isLoop (n : SNode) : Bool := n.loop?.isSome

/-- Dataflow scopes (`DataflowScope`). -/
def isDataflowScope : SNode → Bool
  | mapScope .. | consumeScope .. => true
  | _ => false

/-- `ControlFlowScope` subclasses. -/
def isControlFlowScope : SNode → Bool
  | namedRegion .. | gblock .. | loopScope .. | forScope .. | whileScope .. | doWhileScope .. | ifScope ..
  | stateIfScope .. | elifScope .. | elseScope .. => true
  | _ => false

def isDynScopeCopy : SNode → Bool
  | dynScopeCopy .. => true
  | _ => false

def label? : SNode → Option String
  | stateLabel name => some name
  | _ => none

/-- Memlets of a node (not of its children). -/
def memlets : SNode → List Memlet
  | taskletNode _ ins outs | libraryCall _ _ ins outs => ins.memlets ++ outs.memlets
  | copyNode _ m | dynScopeCopy _ m | viewNode _ _ m _ _ | refSetNode _ m _ _ => [m]
  | _ => []

end SNode

/-! ## Traversal -/

mutual

/-- Number of nodes (the node itself and all descendants). -/
def SNode.size : SNode → Nat
  | .namedRegion _ cs | .gblock cs | .loopScope _ cs | .forScope _ cs | .whileScope _ cs | .doWhileScope _ cs
  | .ifScope _ cs | .stateIfScope _ cs | .elifScope _ cs | .elseScope cs | .mapScope _ cs
  | .consumeScope _ cs => 1 + SNodes.size cs
  | _ => 1

def SNodes.size : List SNode → Nat
  | [] => 0
  | c :: cs => c.size + SNodes.size cs

end

mutual

/-- Pre-order list of all nodes. -/
def SNode.preorder : SNode → List SNode
  | n@(.namedRegion _ cs) | n@(.gblock cs) | n@(.loopScope _ cs) | n@(.forScope _ cs) | n@(.whileScope _ cs)
  | n@(.doWhileScope _ cs) | n@(.ifScope _ cs) | n@(.stateIfScope _ cs) | n@(.elifScope _ cs)
  | n@(.elseScope cs) | n@(.mapScope _ cs) | n@(.consumeScope _ cs) => n :: SNodes.preorder cs
  | n => [n]

def SNodes.preorder : List SNode → List SNode
  | [] => []
  | c :: cs => c.preorder ++ SNodes.preorder cs

end

/-- Pre-order list of all nodes of a tree. -/
def STree.preorder (t : STree) : List SNode := SNodes.preorder t.children

/-- Does a node itself (not its children) mention container `t`
(`Decisions.md#D54`, the tree analogue of `#D39`): memlets, copy, view and
reference targets and sources, expressions naming it by subscript or as a
bare name (`Expr.mentions`, `#D33`), or tasklet code naming it other than
through a connector (`Tasklet.namesInCode`). -/
def SNode.mentions (t : Name) : SNode → Bool
  | .assign _ v => v.expr.mentions t
  | .loopScope l _ | .forScope l _ | .whileScope l _ | .doWhileScope l _ =>
    l.cond.expr.mentions t || l.init.any (·.rhs.mentions t) || l.update.any (·.rhs.mentions t)
  | .ifScope c _ | .stateIfScope c _ | .elifScope c _ => c.expr.mentions t
  | .mapScope mp _ => mp.range.mentions t
  | .consumeScope c _ => c.numPes.mentions t || c.condition.any (·.mentions t)
  | .taskletNode tk ins outs => tk.namesInCode t || (ins.memlets ++ outs.memlets).any (·.data = t)
  | .libraryCall _ _ ins outs => (ins.memlets ++ outs.memlets).any (·.data = t)
  | .copyNode target m => m.data = t || target = t
  | .dynScopeCopy _ m => m.data = t
  | .viewNode target source m _ _ => target = t || source = t || m.data = t
  | .refSetNode target m _ _ => target = t || m.data = t
  | _ => false

mutual

/-- Number of nodes mentioning container `t` (`SNode.mentions`) among a node
and its descendants. -/
def SNode.countData (t : Name) : SNode → Nat
  | n@(.namedRegion _ cs) | n@(.gblock cs) | n@(.loopScope _ cs) | n@(.forScope _ cs) | n@(.whileScope _ cs)
  | n@(.doWhileScope _ cs) | n@(.ifScope _ cs) | n@(.stateIfScope _ cs) | n@(.elifScope _ cs)
  | n@(.elseScope cs) | n@(.mapScope _ cs) | n@(.consumeScope _ cs) =>
    (if n.mentions t then 1 else 0) + SNodes.countData t cs
  | n => if n.mentions t then 1 else 0

def SNodes.countData (t : Name) : List SNode → Nat
  | [] => 0
  | c :: cs => c.countData t + SNodes.countData t cs

end

namespace STree

def container? (t : STree) (n : Name) : Option DataDesc := (t.containers.find? (·.1 = n)).map (·.2)

def isTransient (t : STree) (n : Name) : Bool := (t.container? n).map (·.transient) |>.getD false

def countData (t : STree) (n : Name) : Nat := SNodes.countData n t.children

/-- The descriptor repository as an `SDFG` without control flow, so that the
SDFG execution context (`Ctx`) can be reused for trees. -/
def toSDFG (t : STree) : SDFG :=
  { name := t.name, arrays := t.containers, symbols := t.symbols, constants := t.constants,
    argNames := t.argNames, top := { label := t.name } }

end STree

/-! ## Pretty printing (`as_string`)

DaCe prints symbolic expressions through SymPy's `str`, which puts numeric
constants last (`M - 1`, `tile_i + 1`) and simplifies constant arithmetic
(`dim_to_string` prints `0:N` for the inclusive range `0:N-1`).  The
serialized strings use SymPy's internal order (`-1 + M`).  `SymStr` below
normalizes additive expressions into a small polynomial form and prints it
in SymPy's order: terms by decreasing degree, then lexicographically by
generator, constants last (`Decisions.md#D56`). -/

namespace SymStr

/-- A monomial: coefficient and sorted list of atom keys (symbols or the
printed form of opaque subexpressions). -/
abbrev Poly := List (Int × List String)

def const (v : Int) : Poly := if v = 0 then [] else [(v, [])]

private def insertSorted (a : String) : List String → List String
  | [] => [a]
  | b :: bs => if a ≤ b then a :: b :: bs else b :: insertSorted a bs

def sortAtoms (as : List String) : List String := as.foldl (fun acc a => insertSorted a acc) []

/-- Add a monomial, combining equal monomials and dropping zero coefficients. -/
def addMono (p : Poly) (c : Int) (m : List String) : Poly :=
  match p.find? (·.2 = m) with
  | some (c', _) =>
    let c'' := c + c'
    if c'' = 0 then p.filter (·.2 ≠ m) else p.map fun (k, m') => if m' = m then (c'', m') else (k, m')
  | none => if c = 0 then p else p ++ [(c, m)]

def add (p q : Poly) : Poly := q.foldl (fun acc (c, m) => addMono acc c m) p

def scale (k : Int) (p : Poly) : Poly := if k = 0 then [] else p.map fun (c, m) => (k * c, m)

def isConst (p : Poly) : Option Int :=
  match p with
  | [] => some 0
  | [(c, [])] => some c
  | _ => none

/-- Product of two polynomials when one of them is a single monomial (SymPy
does not expand products of sums); otherwise `none`. -/
def mulMono (p q : Poly) : Option Poly :=
  match p, q with
  | [(c, m)], _ => some (q.foldl (fun acc (c', m') => addMono acc (c * c') (sortAtoms (m ++ m'))) [])
  | _, [(c, m)] => some (p.foldl (fun acc (c', m') => addMono acc (c * c') (sortAtoms (m ++ m'))) [])
  | _, _ => none

/-- SymPy's term order: higher degree first, then lexicographic on the
sorted generators; parenthesized factors (sums) sort after symbols. -/
private def atomLt (a b : String) : Bool :=
  let pa := a.startsWith "("
  let pb := b.startsWith "("
  if pa ≠ pb then !pa else a < b

private def monoLt (m₁ m₂ : List String) : Bool :=
  if m₁.length ≠ m₂.length then m₁.length > m₂.length
  else
    let rec go : List String → List String → Bool
      | a :: as, b :: bs => if a = b then go as bs else atomLt a b
      | _, _ => false
    go m₁ m₂

private def insertTerm (t : Int × List String) : List (Int × List String) → List (Int × List String)
  | [] => [t]
  | u :: us => if monoLt t.2 u.2 then t :: u :: us else u :: insertTerm t us

def ordered (p : Poly) : Poly :=
  let nonConst := p.filter (·.2 ≠ [])
  let cst := p.filter (·.2 = [])
  nonConst.foldl (fun acc t => insertTerm t acc) [] ++ cst

/-- Print a monomial with a positive coefficient. -/
private def monoStr (c : Nat) (m : List String) : String :=
  match m with
  | [] => toString c
  | _ => (if c = 1 then "" else s!"{c}*") ++ "*".intercalate m

def polyStr (p : Poly) : String :=
  match ordered p with
  | [] => "0"
  | (c, m) :: rest =>
    let first := (if c < 0 then "-" else "") ++ monoStr c.natAbs m
    rest.foldl (fun acc (c, m) => acc ++ (if c < 0 then " - " else " + ") ++ monoStr c.natAbs m) first

/-- A single-term polynomial is printed without parentheses when used as a factor. -/
def factorStr (p : Poly) : String :=
  match p with
  | [_] => polyStr p
  | _ => s!"({polyStr p})"

private def binName : BinOp → Option String
  | .lt => some "<" | .le => some "<=" | .gt => some ">" | .ge => some ">="
  | .eq => some "==" | .ne => some "!="
  | _ => none

mutual

/-- Normal form of an arithmetic expression. -/
partial def toPoly : Expr → Poly
  | .lit (.int v) => const v
  | .sym s => [(1, [s])]
  | .un .neg a => scale (-1) (toPoly a)
  | .bin .add a b => add (toPoly a) (toPoly b)
  | .bin .sub a b => add (toPoly a) (scale (-1) (toPoly b))
  | .bin .mul a b =>
    let pa := toPoly a
    let pb := toPoly b
    match mulMono pa pb with
    | some p => p
    | none => [(1, [factorStr pa ++ "*" ++ factorStr pb])]
  | .bin .intFloor a (.lit (.int 1)) => toPoly a
  | .bin .intCeil a (.lit (.int 1)) => toPoly a
  | .approx a _ => toPoly a
  | e => [(1, [atom e])]

/-- Printed form of a non-additive expression. -/
partial def atom : Expr → String
  | .lit (.int v) => toString v
  | .lit (.float m e) => Parse.floatLitToString m e
  | .lit (.bool b) => if b then "True" else "False"
  | .sym s => s
  | .un .not a => s!"Not({str a})"
  | .un .bitNot a => s!"~{str a}"
  | .un .abs a => s!"Abs({str a})"
  | .un .floor a => s!"floor({str a})"
  | .un .ceil a => s!"ceiling({str a})"
  | .bin .div a b => s!"{factor a}/{factor b}"
  | .bin .pow a b => s!"{factor a}**{factor b}"
  | .bin .intFloor a b => s!"int_floor({str a}, {str b})"
  | .bin .intCeil a b => s!"int_ceil({str a}, {str b})"
  | .bin .mod a b => s!"Mod({str a}, {str b})"
  | .bin .min a b => s!"Min({str a}, {str b})"
  | .bin .max a b => s!"Max({str a}, {str b})"
  | .bin .and a b => s!"({str a}) & ({str b})"
  | .bin .or a b => s!"({str a}) | ({str b})"
  | .bin .bitAnd a b => s!"{factor a} & {factor b}"
  | .bin .bitOr a b => s!"{factor a} | {factor b}"
  | .bin .bitXor a b => s!"{factor a} ^ {factor b}"
  | .bin .shl a b => s!"{factor a} << {factor b}"
  | .bin .shr a b => s!"{factor a} >> {factor b}"
  | e@(.bin op a b) =>
    match binName op with
    | some s => s!"{str a} {s} {str b}"
    | none => Expr.toPy e
  | .ite c t e => s!"IfExpr({str c}, {str t}, {str e})"
  | .app f args => s!"{f}({", ".intercalate (args.toList.map str)})"
  | .subscript d idx => s!"{d}[{", ".intercalate (idx.toList.map str)}]"
  | .attr b f => s!"{str b}.{f}"
  | .approx a _ => str a
  | .un .neg a => s!"-{factor a}"

/-- SymPy-style string of an expression. -/
partial def str (e : Expr) : String := polyStr (toPoly e)

/-- As a factor of a product: sums are parenthesized. -/
partial def factor (e : Expr) : String := factorStr (toPoly e)

end

/-- Symbolic equality up to the normal form. -/
def polyEq (a b : Expr) : Bool := add (toPoly a) (scale (-1) (toPoly b)) = []

def isZero (e : Expr) : Bool := toPoly e = []

/-- `str(int(val))` when the expression is a constant, else `str(val)`. -/
def simplified (e : Expr) : String := str e

end SymStr

namespace SNode

open SymStr (polyStr toPoly)

/-- `subsets.Range.dim_to_string`. -/
def dimStr (d : Dim) : String :=
  let lo := SymStr.simplified d.lo
  let negStep : Bool := match d.step with
    | .lit (.int k) => decide (k < 0)
    | _ => false
  let s := if SymStr.isZero (d.hi - d.lo) then lo
           else lo ++ ":" ++ polyStr (SymStr.add (toPoly d.hi) (SymStr.const (if negStep then -1 else 1)))
  let stepOne := SymStr.polyEq d.step 1
  let s := if stepOne then s else s ++ ":" ++ SymStr.simplified d.step
  if SymStr.polyEq d.tile 1 then s
  else (if stepOne then s ++ ":" else s) ++ ":" ++ SymStr.simplified d.tile

def rangeStr (r : Range) : String := ", ".intercalate (r.dims.map dimStr)

def subsetStr : Subset → String
  | .range r => rangeStr r
  | .union rs => " ".intercalate (rs.map rangeStr)

/-- `str(ReductionType)` without the class prefix; custom lambdas print their body. -/
def wcrStr : WcrOp → String
  | .sum => "Sum" | .product => "Product" | .min => "Min" | .max => "Max"
  | .logicalAnd => "Logical_And" | .logicalOr => "Logical_Or" | .logicalXor => "Logical_Xor"
  | .bitwiseAnd => "Bitwise_And" | .bitwiseOr => "Bitwise_Or" | .bitwiseXor => "Bitwise_Xor"
  | .sub => "Sub" | .div => "Div" | .exchange => "Exchange"
  | .custom _ _ body => body.toPy

/-- `Memlet.__str__` (`dace/memlet.py:_label`).  With an `other_subset` the
arrow runs from the memlet's source to its destination; `data` (with its
subset) is printed on its own side, decided by `isDataSrc` (DaCe's
`_is_data_src`, `None` read as `True` like `Memlet.src_subset`;
`Decisions.md#D56`). -/
def memletStr (m : Memlet) : String :=
  match m.dataSubset with
  | none => m.data
  | some sub =>
    let numElems : Option Expr := (sub.asRange?).map Range.numElementsExpr
    let volume :=
      if m.dynamic then "(dyn) "
      else match numElems with
        | some n => if SymStr.polyEq m.volume n then "" else s!"({SymStr.str m.volume}) "
        | none => ""
    let base := m.data ++ volume ++ s!"[{subsetStr sub}]"
    let base := match m.wcr with
      | some op => base ++ s!" (CR: {wcrStr op})"
      | none => base
    match m.otherSubset with
    | some o =>
      let other := s!"[{subsetStr o}]"
      let (src, dst) := if m.isDataSrc then (base, other) else (other, base)
      s!"{src} -> {dst}"
    | none => base

/-- The memlets of a mapping, comma-separated; the entries of a set are sorted (`LibraryCall.as_string`). -/
def memletMapStr (mm : MemletMap) : String :=
  match mm with
  | .dict _ => ", ".intercalate (mm.memlets.map memletStr)
  | .set ms => ", ".intercalate ((ms.map memletStr).mergeSort fun a b => !decide (b < a))

def indentation : String := "  "

def ind (n : Nat) : String := String.join (List.replicate n indentation)

/-- Python tuple rendering of a shape. -/
def shapeStr (shape : List Expr) : String :=
  match shape with
  | [s] => s!"({SymStr.str s},)"
  | _ => s!"({", ".intercalate (shape.map SymStr.str)})"

mutual

/-- `ScheduleTreeNode.as_string(indent)`. -/
partial def asString (indent : Nat) : SNode → String
  | .namedRegion label cs => ind indent ++ s!"named region \"{label}\":\n" ++ scopeBody indent cs
  | .gblock cs => ind indent ++ "gblock:\n" ++ scopeBody indent cs
  | .stateLabel name => ind indent ++ s!"label {name}:"
  | .goto target => ind indent ++ s!"goto {target.getD "exit"}"
  | .assign name v => ind indent ++ s!"assign {name} = {v.code.code}"
  | .loopScope l cs =>
    let initS := (l.init.map (·.code.code)).getD ""
    let updateS := (l.update.map (·.code.code)).getD ""
    match l.variant with
    | .doForUncondIncrement =>
      ind indent ++ s!"{initS}\n" ++ ind indent ++ "do:\n" ++ scopeBody indent cs ++ "\n" ++
      ind (indent + 1) ++ s!"{updateS}\n" ++ ind indent ++ s!"while {l.cond.code.code}"
    | .doFor =>
      ind indent ++ s!"{initS}\n" ++ ind indent ++ "while True:\n" ++ scopeBody indent cs ++ "\n" ++
      ind (indent + 1) ++ s!"if (not {l.cond.code.code}):\n" ++ ind (indent + 2) ++ "break\n" ++
      ind (indent + 1) ++ s!"{updateS}\n"
    | _ => scopeBody indent cs
  | .forScope l cs =>
    let initS := (l.init.map (·.code.code)).getD ""
    let updateS := (l.update.map (·.code.code)).getD ""
    ind indent ++ s!"for {initS}; {l.cond.code.code}; {updateS}:\n" ++ scopeBody indent cs
  | .whileScope l cs => ind indent ++ s!"while {l.cond.code.code}:\n" ++ scopeBody indent cs
  | .doWhileScope l cs =>
    ind indent ++ "do:\n" ++ scopeBody indent cs ++ "\n" ++ ind indent ++ s!"while {l.cond.code.code}"
  | .ifScope c cs => ind indent ++ s!"if {c.code.code}:\n" ++ scopeBody indent cs
  | .stateIfScope c cs => ind indent ++ s!"stateif {c.code.code}:\n" ++ scopeBody indent cs
  | .breakNode => ind indent ++ "break"
  | .continueNode => ind indent ++ "continue"
  | .elifScope c cs => ind indent ++ s!"elif {c.code.code}:\n" ++ scopeBody indent cs
  | .elseScope cs => ind indent ++ "else:\n" ++ scopeBody indent cs
  | .mapScope mp cs =>
    ind indent ++ s!"map {", ".intercalate mp.params} in [{rangeStr mp.range}]:\n" ++ scopeBody indent cs
  | .consumeScope c cs =>
    let cond := match c.conditionSrc, c.condition with
      | some src, _ => src
      | none, none => "stream not empty"
      | none, some e => SymStr.str e
    ind indent ++ s!"consume (PE {c.peIndex} out of {SymStr.str c.numPes}) while {cond}:\n" ++ scopeBody indent cs
  | .taskletNode _ ins outs =>
    if outs.isEmpty then ind indent ++ s!"tasklet({memletMapStr ins})"
    else ind indent ++ s!"{memletMapStr outs} = tasklet({memletMapStr ins})"
  | .libraryCall l props ins outs =>
    let libname := (l.classpath.splitOn ".").getLast?.getD l.classpath
    let own := ", ".intercalate (props.map fun (k, v) => s!"{k}={v}")
    ind indent ++ s!"{memletMapStr outs} = library {libname}[{own}]({memletMapStr ins})"
  | .copyNode target m =>
    let offset := match m.otherSubset with
      | some o =>
        let nonzero := match o.asRange? with
          | some r => r.dims.any fun d => !SymStr.isZero d.lo
          | none => true
        if nonzero then s!"[{subsetStr o}]" else ""
      | none => ""
    let wcr := match m.wcr with
      | some op => s!" with {op.toLambda}"
      | none => ""
    let sub := (m.dataSubset.map subsetStr).getD "None"
    ind indent ++ s!"{target}{offset} = copy {m.data}[{sub}]{wcr}"
  | .dynScopeCopy target m =>
    ind indent ++ s!"{target} = dscopy {m.data}[{(m.dataSubset.map subsetStr).getD "None"}]"
  | .viewNode target _ m _ viewDesc => ind indent ++ s!"{target} = view {memletStr m} as {shapeStr viewDesc.shape}"
  | .refSetNode target m src _ =>
    match src with
    | .codeNode _ cls => ind indent ++ s!"{target} = refset from {cls.toLower}"
    | .data _ => ind indent ++ s!"{target} = refset to {memletStr m}"

/-- `ScheduleTreeScope.as_string`: the children one level deeper, or `pass`. -/
partial def scopeBody (indent : Nat) (cs : List SNode) : String :=
  match cs with
  | [] => ind (indent + 1) ++ "pass"
  | _ => "\n".intercalate (cs.map (asString (indent + 1)))

end

end SNode

/-- `ScheduleTreeRoot.as_string()`. -/
def STree.prettyPrint (t : STree) : String := SNode.scopeBody 0 t.children

end Dace
