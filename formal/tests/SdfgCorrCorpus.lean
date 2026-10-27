-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! The SDFG–SDFG checker (`Dace.Equiv.SdfgCorr`) over the corpus.  Run with
`lake env lean tests/SdfgCorrCorpus.lean` from `formal/` after generating the
corpus (`scripts/dump_corpus.py`).

For every corpus SDFG, a variant with the nodes of every state renumbered
(reversed ids) must be certified `EQUIVALENT`, a variant
with the node list of every state reordered (within components) must be
certified, and a variant with one edge reversed must not be. -/

open Dace

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

/-- Reverse the ids of all nodes (nested SDFG nodes included). -/
def reversePairs (s : State) : List (NodeId × NodeId) :=
  let ids := s.nodes.map (·.id)
  ids.zip ids.reverse

mutual
partial def relabelRegion (r : Region) : Region :=
  { r with blocks := r.blocks.map relabelBlock }
partial def relabelBlock : Block → Block
  | .state s => .state (s.relabel (permOfPairs (reversePairs s)))
  | .region r => .region (relabelRegion r)
  | .loop l => .loop { l with body := relabelRegion l.body }
  | .cond lb brs => .cond lb (brs.map fun (c, r) => (c, relabelRegion r))
  | b => b
end

/-- Reverse the node list of every state, keeping each component's position
(components are discovered in node order): the nodes of each component are
reversed in place. -/
def reorderState (s : State) : State :=
  let comps := s.components
  let pos := fun (id : NodeId) => comps.findIdx (·.contains id)
  let groups := (List.range comps.length).map fun k => (s.nodes.filter fun n => pos n.id = k).reverse
  -- interleave: keep the first node of every component first
  { s with nodes := groups.flatten }

mutual
partial def reorderRegion (r : Region) : Region :=
  { r with blocks := r.blocks.map reorderBlock }
partial def reorderBlock : Block → Block
  | .state s => .state (reorderState s)
  | .region r => .region (reorderRegion r)
  | .loop l => .loop { l with body := reorderRegion l.body }
  | .cond lb brs => .cond lb (brs.map fun (c, r) => (c, reorderRegion r))
  | b => b
end

/-- Reverse the first edge of the first top-level state with edges (the
nodes stay the same, the dataflow changes). -/
partial def flipFirstEdge (r : Region) : Option Region :=
  let rec go : List Block → Option (List Block)
    | [] => none
    | .state s :: bs =>
      match s.edges with
      | e :: es => some (.state { s with edges := { e with src := e.dst, dst := e.src } :: es } :: bs)
      | [] => (go bs).map (.state s :: ·)
    | b :: bs => (go bs).map (b :: ·)
  (go r.blocks).map fun bs => { r with blocks := bs }

/-- Reordered variants the rules do not certify: two independent views
bound in the other order (ties in `State.viewOrder` follow the node list;
when both bindings fail the reported error would differ). -/
def knownReorder : List String := ["empty_view", "rt.view_of_view"]

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  let mut ok := 0
  let mut steps := 0
  let mut rok := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    match ← SDFG.Json.readSDFG f with
    | .error e => IO.println s!"IMPORT FAIL {name}: {e}"; bad := bad + 1
    | .ok g =>
      let v := { g with top := relabelRegion g.top }
      match sdfgCheck g v with
      | .ok sts => ok := ok + 1; steps := steps + sts.length
      | .error m => IO.println s!"UNKNOWN  {name}: {m}"; bad := bad + 1
      let w := { g with top := reorderRegion g.top }
      match sdfgCheck g w with
      | .ok _ => rok := rok + 1
      | .error m =>
        if knownReorder.contains name then IO.println s!"known UNKNOWN {name} (reordered): {m}"
        else IO.println s!"UNKNOWN  {name} (reordered): {m}"; bad := bad + 1
      match flipFirstEdge g.top with
      | some top =>
        if (sdfgCheck g { g with top }).toBool then
          IO.println s!"WRONGLY EQUIVALENT {name} (flipped edge)"
          bad := bad + 1
      | none => pure ()
  IO.println s!"{ok} relabeled variants certified ({steps} rule applications), {rok} reordered, {bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} failures"

#eval main
