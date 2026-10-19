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

/-- All edges of the memlet tree containing `e` (closure under `prev`/`next`). -/
partial def treeEdges (s : State) (e : DEdge) : List DEdge :=
  let rec go (seen : List DEdge) : List DEdge → List DEdge
    | [] => seen.reverse
    | f :: rest =>
      if seen.contains f then go seen rest
      else go (f :: seen) (rest ++ s.prevEdges f ++ s.nextEdges f)
  go [] [e]

/-- Leaves of the memlet tree containing `e`. -/
def treeLeaves (s : State) (e : DEdge) : List DEdge := (s.treeEdges e).filter (s.isLeaf)

/-- The outermost edge of the tree containing `e`: follow `prev` while the
source is an entry node (inward trees) or `next` while the destination is
an exit node (outward trees). -/
partial def treeRoot (s : State) (e : DEdge) : DEdge :=
  if s.isEntryNode e.src && connPrefix e.srcConn "OUT_" then
    match s.prevEdges e with
    | f :: _ => s.treeRoot f
    | [] => e
  else if s.isExitNode e.dst && connPrefix e.dstConn "IN_" then
    match s.nextEdges e with
    | f :: _ => s.treeRoot f
    | [] => e
  else e

/-! ## Scopes -/

/-- Direct children of a scope (`none` = top level).  The exit node of an
entry is among its children, as in DaCe's `scope_children`. -/
def children (s : State) (entry : Option NodeId) : List DNode :=
  s.nodes.filter (·.scopeEntry = entry)

/-- All nodes transitively inside the scope of `entry`. -/
partial def scopeNodes (s : State) (entry : NodeId) : List NodeId :=
  let direct := s.children (some entry)
  direct.flatMap fun n => n.id :: (if n.isEntry then s.scopeNodes n.id else [])

/-- Parent scope chain of a node, innermost first. -/
partial def scopeChain (s : State) (id : NodeId) : List NodeId :=
  match s.node? id with
  | some n =>
    match n.scopeEntry with
    | some p => if n.isExit && p = id then [] else p :: s.scopeChain p
    | none => []
  | none => []

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

/-- Weakly connected components in node order (a coarsening of DaCe's
`concurrent_subgraphs`, `Decisions.md#D4`). -/
partial def components (s : State) : List (List NodeId) :=
  let rec bfs (seen : List NodeId) : List NodeId → List NodeId
    | [] => seen.reverse
    | n :: rest =>
      if seen.contains n then bfs seen rest else bfs (n :: seen) (rest ++ s.neighbors n)
  let rec go (seen : List NodeId) (acc : List (List NodeId)) : List DNode → List (List NodeId)
    | [] => acc.reverse
    | n :: rest =>
      if seen.contains n.id then go seen acc rest
      else
        let comp := bfs [] [n.id]
        go (comp ++ seen) (comp :: acc) rest
  go [] [] s.nodes

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
