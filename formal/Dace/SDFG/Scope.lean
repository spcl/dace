-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Syntax

/-!
# Scopes, memlet trees and concurrency structure of a state

Derived structure over the flat state graph (`dace/sdfg/state.py`:
`scope_dict`, `scope_children`, `memlet_path`, `memlet_tree`;
`dace/sdfg/utils.py`: `concurrent_subgraphs`, topological sorts).

## Leaves and routing edges

Scope entry nodes route data *inward* through connector pairs `IN_x → OUT_x`,
exit nodes route *outward*.  An edge is a **routing edge** if it enters an
entry node through an `IN_` connector or leaves an exit node through an
`OUT_` connector; every other edge is a **leaf**: the place where data
actually moves (DaCe emits copies and tasklet accesses at the leaves,
`doc/sdfg/ir.rst:420-422`).  Dynamic scope inputs (edges into an entry node
through a connector not prefixed `IN_`) are leaves.
-/

namespace Dace

/-- Drop a prefix from a string (as a `String`, not a slice). -/
def String.dropPrefixStr (s : String) (n : Nat) : String := String.ofList (s.toList.drop n)

/-! ## Fuel-bounded breadth-first closure -/

/-- Append the elements of `ys` that are not yet in `acc`, in order. -/
def appendNew {α : Type} [BEq α] (acc ys : List α) : List α :=
  ys.foldl (fun acc y => if acc.contains y then acc else acc ++ [y]) acc

/-- Breadth-first closure under `step`: `seen` is the discovery order, the
entries before index `i` have been expanded.  Each step expands one entry, so
`fuel` must exceed the number of distinct reachable elements. -/
def closureFrom {α : Type} [BEq α] (step : α → List α) : Nat → List α → Nat → List α
  | 0, seen, _ => seen
  | fuel + 1, seen, i =>
    match seen[i]? with
    | some x => closureFrom step fuel (appendNew seen (step x)) (i + 1)
    | none => seen

/-- The elements reachable from `start` through `step`, in breadth-first
discovery order (the order of a FIFO worklist search).  `bound` must be at
least the length of a list containing `start` and every result of `step`
(`Dace/SDFG/ScopeSpec.lean`, `mem_closure`). -/
def closure {α : Type} [BEq α] (step : α → List α) (bound : Nat) (start : α) : List α :=
  closureFrom step (bound + 1) [start] 0

namespace State

def connPrefix (c : Option Conn) (p : String) : Bool :=
  match c with
  | some s => s.startsWith p
  | none => false

/-- The suffix `x` of a passthrough connector `p ++ x`. -/
def passSuffix? (c : Option Conn) (p : String) : Option String :=
  match c with
  | some s => if s.startsWith p then some (String.dropPrefixStr s p.length) else none
  | none => none

def isScopeNode (s : State) (id : NodeId) : Bool :=
  match s.node? id with
  | some n => n.isEntry || n.isExit
  | none => false

def isEntryNode (s : State) (id : NodeId) : Bool := (s.node? id).map DNode.isEntry |>.getD false
def isExitNode (s : State) (id : NodeId) : Bool := (s.node? id).map DNode.isExit |>.getD false
def isAccessNode (s : State) (id : NodeId) : Bool := (s.node? id).map DNode.isAccess |>.getD false
def isCodeNode (s : State) (id : NodeId) : Bool := (s.node? id).map DNode.isCodeNode |>.getD false

/-- Edges feeding `e` through its source scope node's matching `IN_` connector. -/
def prevEdges (s : State) (e : DEdge) : List DEdge :=
  if s.isScopeNode e.src then
    match passSuffix? e.srcConn "OUT_" with
    | some x => (s.inEdges e.src).filter fun f => f.dstConn = some ("IN_" ++ x)
    | none => []
  else []

/-- Edges continuing `e` through its destination scope node's matching `OUT_` connector. -/
def nextEdges (s : State) (e : DEdge) : List DEdge :=
  if s.isScopeNode e.dst then
    match passSuffix? e.dstConn "IN_" with
    | some x => (s.outEdges e.dst).filter fun f => f.srcConn = some ("OUT_" ++ x)
    | none => []
  else []

/-- Routing edge: enters an entry through `IN_` or leaves an exit through `OUT_`. -/
def isRouting (s : State) (e : DEdge) : Bool :=
  (s.isEntryNode e.dst && connPrefix e.dstConn "IN_") ||
  (s.isExitNode e.src && connPrefix e.srcConn "OUT_")

def isLeaf (s : State) (e : DEdge) : Bool := !s.isRouting e

/-- Follow `prevEdges` while the source is a scope node entered through an
`OUT_` connector (`memlet_path(e)[0]`). -/
def walkPrev (s : State) : Nat → DEdge → DEdge
  | 0, e => e
  | fuel + 1, e =>
    if s.isScopeNode e.src && connPrefix e.srcConn "OUT_" then
      match s.prevEdges e with
      | f :: _ => s.walkPrev fuel f
      | [] => e
    else e

/-- Follow `nextEdges` while the destination is a scope node (only exit nodes
when `onlyExits`) entered through an `IN_` connector (`memlet_path(e)[-1]`). -/
def walkNext (s : State) (onlyExits : Bool) : Nat → DEdge → DEdge
  | 0, e => e
  | fuel + 1, e =>
    if (if onlyExits then s.isExitNode e.dst else s.isScopeNode e.dst) && connPrefix e.dstConn "IN_" then
      match s.nextEdges e with
      | f :: _ => s.walkNext onlyExits fuel f
      | [] => e
    else e

/-- First edge of the memlet path through `e`. -/
def pathStart (s : State) (e : DEdge) : DEdge := s.walkPrev (s.edges.length + 1) e

/-- Last edge of the memlet path through `e`. -/
def pathEnd (s : State) (e : DEdge) : DEdge := s.walkNext false (s.edges.length + 1) e

/-- The stream a consume entry pops from: the access node at the start of
the memlet path into its `IN_stream` connector (`Decisions.md#D38`). -/
def consumedStream? (s : State) (entry : NodeId) : Option Name :=
  match (s.inEdges entry).find? (·.dstConn = some "IN_stream") with
  | some e => (s.node? (s.pathStart e).src).bind DNode.accessData?
  | none => none

/-- Dynamic scope input edge. -/
def isDynamicInput (s : State) (e : DEdge) : Bool :=
  s.isEntryNode e.dst && e.dstConn.isSome && !connPrefix e.dstConn "IN_"

/-- All edges of the memlet tree containing `e` (closure under `prev`/`next`),
in breadth-first discovery order.  Every edge reached is `e` or an edge of
the state, so `s.edges.length + 1` bounds the closure. -/
def treeEdges (s : State) (e : DEdge) : List DEdge :=
  closure (fun f => s.prevEdges f ++ s.nextEdges f) (s.edges.length + 1) e

/-- Leaves of the memlet tree containing `e`. -/
def treeLeaves (s : State) (e : DEdge) : List DEdge := (s.treeEdges e).filter (s.isLeaf)

/-- One step of `treeRoot` with fuel: follow `prev` while the source is an
entry node (inward trees) or `next` while the destination is an exit node
(outward trees). -/
def treeRootFuel (s : State) : Nat → DEdge → DEdge
  | 0, e => e
  | fuel + 1, e =>
    if s.isEntryNode e.src && connPrefix e.srcConn "OUT_" then
      match s.prevEdges e with
      | f :: _ => s.treeRootFuel fuel f
      | [] => e
    else if s.isExitNode e.dst && connPrefix e.dstConn "IN_" then
      match s.nextEdges e with
      | f :: _ => s.treeRootFuel fuel f
      | [] => e
    else e

/-- The outermost edge of the tree containing `e`.  The walk visits a path
of distinct edges of an acyclic state, so `s.edges.length + 1` steps suffice. -/
def treeRoot (s : State) (e : DEdge) : DEdge := s.treeRootFuel (s.edges.length + 1) e

/-! ## Scopes -/

/-- Direct children of a scope (`none` = top level).  The exit node of an
entry is among its children, as in DaCe's `scope_children`. -/
def children (s : State) (entry : Option NodeId) : List DNode :=
  s.nodes.filter (·.scopeEntry = entry)

/-- All nodes transitively inside the scope of `entry`, with recursion depth
bounded by `fuel`. -/
def scopeNodesFuel (s : State) : Nat → NodeId → List NodeId
  | 0, _ => []
  | fuel + 1, entry =>
    (s.children (some entry)).flatMap fun n =>
      n.id :: (if n.isEntry then s.scopeNodesFuel fuel n.id else [])

/-- All nodes transitively inside the scope of `entry`.  Scope nesting is at
most `s.nodes.length` deep when it is acyclic (`ScopeWF`), so the fuel
`s.nodes.length + 1` is never exhausted on well-formed states. -/
def scopeNodes (s : State) (entry : NodeId) : List NodeId :=
  s.scopeNodesFuel (s.nodes.length + 1) entry

/-- The parent scope of a node (`scope_dict`): its `scopeEntry`, except for
the malformed case of an exit node naming itself. -/
def parent? (s : State) (id : NodeId) : Option NodeId :=
  match s.node? id with
  | some n =>
    match n.scopeEntry with
    | some p => if n.isExit && p = id then none else some p
    | none => none
  | none => none

/-- Parent scope chain of a node, innermost first, computed with `fuel`
steps; `none` when the fuel runs out (cyclic scope nesting). -/
def scopeChainFuel (s : State) : Nat → NodeId → Option (List NodeId)
  | 0, _ => none
  | fuel + 1, id =>
    match s.parent? id with
    | some p => (s.scopeChainFuel fuel p).map (p :: ·)
    | none => some []

/-- Parent scope chain of a node, innermost first (`[]` for cyclic scope
nesting, which `ScopeWF` and the WF checker exclude). -/
def scopeChain (s : State) (id : NodeId) : List NodeId :=
  (s.scopeChainFuel (s.nodes.length + 1) id).getD []

/-- Structural facts about scopes that the scope lemmas (`Dace/SDFG/ScopeSpec.lean`)
need: unique node ids, every `scopeEntry` names an entry node, and acyclic
scope nesting (every scope chain ends within the fuel).  Checked by
`scopeWF?`, and part of the WF checker (`WF.checkState`). -/
structure ScopeWF (s : State) : Prop where
  nodup : (s.nodes.map (·.id)).Nodup
  parentEntry : ∀ n ∈ s.nodes, ∀ p, n.scopeEntry = some p → s.isEntryNode p = true
  acyclic : ∀ n ∈ s.nodes, (s.scopeChainFuel (s.nodes.length + 1) n.id).isSome = true

/-- Decision procedure for `ScopeWF` (soundness: `State.scopeWF?_sound`). -/
def scopeWF? (s : State) : Bool :=
  decide (s.nodes.map (·.id)).Nodup &&
  s.nodes.all (fun n => match n.scopeEntry with
    | some p => s.isEntryNode p
    | none => true) &&
  s.nodes.all (fun n => (s.scopeChainFuel (s.nodes.length + 1) n.id).isSome)

/-- Is `inner` inside the scope of `entry` (transitively)? -/
def inScope (s : State) (entry : NodeId) (inner : NodeId) : Bool :=
  (s.scopeChain inner).contains entry

/-- Entry node of an exit node. -/
def entryOfExit? (s : State) (id : NodeId) : Option NodeId :=
  match s.node? id with
  | some n => if n.isExit then n.scopeEntry else none
  | none => none

/-! ## Concurrency structure -/

/-- Undirected neighbours (dependency edges count). -/
def neighbors (s : State) (id : NodeId) : List NodeId :=
  s.edges.filterMap fun e =>
    if e.src = id then some e.dst else if e.dst = id then some e.src else none

/-- The weakly connected component of `id`, in breadth-first discovery order.
Every node reached is `id` or an edge endpoint, which bounds the closure. -/
def component (s : State) (id : NodeId) : List NodeId :=
  closure s.neighbors (2 * s.edges.length + 1) id

/-- Components of the nodes `ns` not already covered by `seen`. -/
def componentsFrom (s : State) (seen : List NodeId) : List DNode → List (List NodeId)
  | [] => []
  | n :: rest =>
    if seen.contains n.id then s.componentsFrom seen rest
    else
      let comp := s.component n.id
      comp :: s.componentsFrom (comp ++ seen) rest

/-- Weakly connected components in node order (a coarsening of DaCe's
`concurrent_subgraphs`, `Decisions.md#D4`). -/
def components (s : State) : List (List NodeId) := s.componentsFrom [] s.nodes

/-- Is `order` a topological order of (a subset of) the nodes: every edge
between listed nodes goes forward? -/
def respectsEdges (s : State) (order : List NodeId) : Bool :=
  s.edges.all fun e =>
    match order.idxOf? e.src, order.idxOf? e.dst with
    | some i, some j => i < j
    | _, _ => true

/-- `order` is a topological order of exactly the nodes `ids`. -/
def isTopologicalOrder (s : State) (ids : List NodeId) (order : List NodeId) : Bool :=
  order.length = ids.length && ids.all order.contains && order.Nodup && s.respectsEdges order

/-- A deterministic topological order of the given nodes (Kahn's algorithm,
ties broken by node order).  Returns `none` if the subgraph is cyclic. -/
def topoOrder (s : State) (ids : List NodeId) : Option (List NodeId) :=
  let rec go (fuel : Nat) (placed : List NodeId) (remaining : List NodeId) : Option (List NodeId) :=
    match fuel with
    | 0 => if remaining.isEmpty then some placed.reverse else none
    | fuel + 1 =>
      match remaining with
      | [] => some placed.reverse
      | _ =>
        let ready := remaining.filter fun n =>
          (s.inEdges n).all fun e => !remaining.contains e.src || e.src = n
        match ready with
        | [] => none
        | r :: _ => go fuel (r :: placed) (remaining.erase r)
  go (ids.length + 1) [] ids

end State

end Dace
