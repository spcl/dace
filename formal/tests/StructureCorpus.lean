-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Structural checks over the corpus: scope/memlet-tree computations and the
well-formedness checker.  Run with `lake env lean tests/StructureCorpus.lean`
from `formal/`. -/

open Dace

mutual
partial def statesOfRegion (r : Region) : List State := r.blocks.flatMap statesOfBlock
partial def statesOfBlock : Block → List State
  | .state s => s :: (s.nestedSDFGs.flatMap fun g => statesOfRegion g.top)
  | .region r => statesOfRegion r
  | .loop l => statesOfRegion l.body
  | .cond _ bs => bs.flatMap fun (_, r) => statesOfRegion r
  | _ => []
end

/-- Per-state facts: components, leaves, roots, topological order. -/
def stateFacts (s : State) : List String :=
  let ids := s.nodes.map (·.id)
  let comps := s.components
  let leaves := s.edges.filter s.isLeaf
  let memletEdges := s.edges.filter fun e => e.data matches .memlet _
  let roots := (memletEdges.map s.treeRoot).eraseDups
  let topo := s.topoOrder ids
  let topoOk := match topo with
    | some o => s.isTopologicalOrder ids o
    | none => false
  let rootedOk := roots.all fun r =>
    s.isAccessNode r.src || s.isAccessNode r.dst || (s.isCodeNode r.src && s.isCodeNode r.dst)
  let dyn := s.edges.filter s.isDynamicInput
  [s!"  state '{s.label}': {s.nodes.length} nodes, {s.edges.length} edges, {comps.length} components, " ++
   s!"{leaves.length} leaves, {roots.length} trees, {dyn.length} dynamic inputs, topo={topoOk}, rooted={rootedOk}"]

def main : IO Unit := do
  let dir : System.FilePath := "tests/corpus"
  let entries ← dir.readDir
  let files := (entries.map (·.path)).qsort (fun a b => a.toString < b.toString)
  let mut bad := 0
  for f in files do
    if f.toString.endsWith ".sdfg.json" then
      match ← SDFG.Json.readSDFG f with
      | .error e => IO.println s!"IMPORT FAIL {f.fileName.getD ""}: {e}"; bad := bad + 1
      | .ok g =>
        let r := g.check
        IO.println s!"{f.fileName.getD ""}: {r.errors.length} errors, {r.warnings.length} warnings"
        for e in r.errors do IO.println s!"  ERROR {e}"
        for w in r.warnings do IO.println s!"  warn  {w}"
        for s in statesOfRegion g.top do
          for l in stateFacts s do IO.println l
        if !r.ok then bad := bad + 1
  IO.println s!"{bad} files with problems"

#eval main
