-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Interp
import Dace.Core.Facts

/-!
# SDFG syntax

A faithful, *flat* rendering of DaCe's SDFG classes (`dace/sdfg/sdfg.py`,
`state.py`, `nodes.py`): states are multigraphs of nodes with connectors,
scopes are given by entry/exit node pairs (not by nesting), control flow is
a hierarchy of regions.  Derived notions (scope tree, memlet paths, memlet
trees) are computed in `Dace.SDFG.Scope`.

Omitted on purpose: debug info, GUIDs, instrumentation, storage locations,
and schedules and other map hints (`unroll`, `collapse`), which do not
affect the semantics
(`Decisions.md#D3`, `#D63`).
-/

namespace Dace

abbrev NodeId := Nat
abbrev BlockId := Nat

/-- `dace.sdfg.nodes.Map` (semantic fields). -/
structure MapNode where
  label : String := ""
  params : List Sym
  range : Range
  deriving DecidableEq, Repr, Inhabited

/-- `dace.sdfg.nodes.Consume`.  The quiescence `condition` is parsed at import
like interstate conditions (`Decisions.md#D41`); `none` = "stream not empty". -/
structure ConsumeNode where
  label : String := ""
  peIndex : Sym
  numPes : Expr := 1
  condition : Option Expr := none
  chunksize : Nat := 1
  deriving DecidableEq, Repr, Inhabited

/-- Properties of the `Reduce` library node, which gets a direct semantics. -/
structure ReduceProps where
  /-- `none` = reduce over all axes. -/
  axes : Option (List Nat)
  wcr : WcrOp
  identity : Option Expr
  deriving DecidableEq, Repr, Inhabited

/-- A library node.  Only `Reduce` is interpreted directly; others go through
`Interp.library` keyed by `classpath`. -/
structure LibNode where
  classpath : String
  label : String := ""
  implementation : Option String := none
  reduce : Option ReduceProps := none
  deriving DecidableEq, Repr, Inhabited

/-- Interstate edge (`dace.sdfg.InterstateEdge`): condition (default true)
and simultaneous symbol assignments. -/
structure IEdge where
  src : BlockId
  dst : BlockId
  cond : Expr := .lit (.bool true)
  assigns : List (Sym × Expr) := []
  deriving DecidableEq, Repr, Inhabited

namespace IEdge
def isUnconditional (e : IEdge) : Bool := e.cond = .lit (.bool true) || e.cond = .lit (.int 1)
end IEdge

/-- Connector table of a node: name ↦ declared type (`none` = inferred). -/
abbrev Conns := List (Conn × Option Dtype)

/-- Payload of a dataflow edge: a memlet, or an empty memlet (a pure
happens-before dependency, `Decisions.md#D27`). -/
inductive EdgeData where
  | memlet (m : Memlet)
  | dep
  deriving DecidableEq, Repr, Inhabited

namespace EdgeData
def memlet? : EdgeData → Option Memlet
  | memlet m => some m
  | dep => none
end EdgeData

/-- A dataflow edge between connectors. -/
structure DEdge where
  src : NodeId
  srcConn : Option Conn
  dst : NodeId
  dstConn : Option Conn
  data : EdgeData
  deriving DecidableEq, Repr, Inhabited

inductive RegionKind where
  | plain
  | named (label : String)
  | functionCall (label : String)
  | unstructured
  deriving DecidableEq, Repr, Inhabited

mutual

/-- Node kinds.  Entry/exit pairing is given by node ids. -/
inductive NodeKind where
  | access (data : Name)
  | tasklet (t : Tasklet)
  | mapEntry (m : MapNode)
  | mapExit
  | consumeEntry (c : ConsumeNode)
  | consumeExit
  /-- Nested SDFG with its symbol mapping (inner symbol ↦ outer expression). -/
  | nested (sdfg : SDFG) (symbolMapping : List (Sym × Expr)) (noInline : Bool)
  | library (l : LibNode)

/-- A node of a state. -/
structure DNode where
  id : NodeId
  label : String := ""
  inConns : Conns := []
  outConns : Conns := []
  /-- The enclosing scope's entry node (`scope_entry`), if any. -/
  scopeEntry : Option NodeId := none
  kind : NodeKind

/-- A state: an acyclic multigraph of nodes and edges. -/
structure State where
  label : String := ""
  nodes : List DNode := []
  edges : List DEdge := []
  nosync : Bool := false

/-- `dace.sdfg.state.LoopRegion`. -/
structure LoopRegion where
  label : String := ""
  loopVar : Option Sym := none
  init : Option (Sym × Expr) := none
  cond : Expr := .lit (.bool true)
  update : Option (Sym × Expr) := none
  inverted : Bool := false
  updateBeforeCond : Bool := false
  body : Region

/-- A control-flow block. -/
inductive Block where
  | state (s : State)
  | region (r : Region)
  | loop (l : LoopRegion)
  /-- `ConditionalBlock`: branches in order, `none` condition = else. -/
  | cond (label : String) (branches : List (Option Expr × Region))
  | breakBlock
  | continueBlock
  | returnBlock

/-- A control-flow region: blocks (indexed by position = `BlockId`), a start
block and interstate edges. -/
structure Region where
  label : String := ""
  kind : RegionKind := .plain
  blocks : List Block := []
  start : BlockId := 0
  edges : List IEdge := []

/-- An SDFG. -/
structure SDFG where
  name : String
  arrays : List (Name × DataDesc) := []
  symbols : List (Sym × Dtype) := []
  /-- Compile-time constants with their (scalar) values when representable. -/
  constants : List (Name × DataDesc × Option Val) := []
  argNames : List Name := []
  top : Region
  /-- Facts about the symbols (`symbol_repo.params`, `Core/Facts`). -/
  facts : Constraints := {}

end

/-! ## Basic accessors -/

namespace DNode

def isAccess (n : DNode) : Bool := n.kind matches .access _
def isTasklet (n : DNode) : Bool := n.kind matches .tasklet _
def isEntry (n : DNode) : Bool := n.kind matches .mapEntry _ | .consumeEntry _
def isExit (n : DNode) : Bool := n.kind matches .mapExit | .consumeExit
def isNested (n : DNode) : Bool := n.kind matches .nested _ _ _
def isLibrary (n : DNode) : Bool := n.kind matches .library _
def isCodeNode (n : DNode) : Bool := n.isTasklet || n.isNested || n.isLibrary

def accessData? (n : DNode) : Option Name :=
  match n.kind with
  | .access d => some d
  | _ => none

def map? (n : DNode) : Option MapNode :=
  match n.kind with
  | .mapEntry m => some m
  | _ => none

/-- Dynamic scope inputs: in-connectors of an entry node not prefixed `IN_`. -/
def dynamicInputs (n : DNode) : List Conn :=
  if n.isEntry then (n.inConns.map Prod.fst).filter (fun c => !c.startsWith "IN_") else []

end DNode

namespace State

def node? (s : State) (id : NodeId) : Option DNode := s.nodes.find? (·.id = id)

def inEdges (s : State) (id : NodeId) : List DEdge := s.edges.filter (·.dst = id)
def outEdges (s : State) (id : NodeId) : List DEdge := s.edges.filter (·.src = id)

def accessNodes (s : State) : List DNode := s.nodes.filter DNode.isAccess

/-- Exit node of an entry node: the unique exit whose `scopeEntry` is `id`
and whose kind matches. -/
def exitOf? (s : State) (id : NodeId) : Option NodeId :=
  (s.nodes.find? fun n => n.isExit && n.scopeEntry = some id).map (·.id)

end State

namespace Region
def block? (r : Region) (id : BlockId) : Option Block := r.blocks[id]?
def outEdges (r : Region) (id : BlockId) : List IEdge := r.edges.filter (·.src = id)
end Region

namespace SDFG

def array? (g : SDFG) (n : Name) : Option DataDesc := (g.arrays.find? (·.1 = n)).map (·.2)

def isTransient (g : SDFG) (n : Name) : Bool := (g.array? n).map (·.transient) |>.getD false

def symbolNames (g : SDFG) : List Sym := g.symbols.map Prod.fst

end SDFG

/-! ## Traversal helpers (mutual recursion over the hierarchy) -/

mutual

/-- All states of a region, depth first. -/
partial def Region.allStates (r : Region) : List State :=
  r.blocks.flatMap Block.allStates

partial def Block.allStates : Block → List State
  | .state s => [s]
  | .region r => r.allStates
  | .loop l => l.body.allStates
  | .cond _ bs => bs.flatMap fun (_, r) => r.allStates
  | _ => []

end

/-- All nested SDFGs reachable from a state (one level). -/
def State.nestedSDFGs (s : State) : List SDFG :=
  s.nodes.filterMap fun n => match n.kind with
    | .nested g _ _ => some g
    | _ => none

end Dace
