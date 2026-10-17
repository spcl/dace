-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Import every `.sdfg.json` of the corpus and print a structural summary.
Run with `lake env lean tests/ImportCorpus.lean` from `formal/`. -/

open Dace

mutual
partial def countRegion (r : Region) : Nat × Nat × Nat :=
  r.blocks.foldl (fun (s, n, e) b => let (s', n', e') := countBlock b; (s + s', n + n', e + e')) (0, 0, 0)

partial def countBlock : Block → Nat × Nat × Nat
  | .state s => (1, s.nodes.length, s.edges.length)
  | .region r => countRegion r
  | .loop l => countRegion l.body
  | .cond _ bs => bs.foldl (fun (s, n, e) (_, r) => let (s', n', e') := countRegion r; (s + s', n + n', e + e')) (0, 0, 0)
  | _ => (0, 0, 0)
end

def summary (g : SDFG) : String :=
  let (states, nodes, edges) := countRegion g.top
  s!"{g.name}: {g.arrays.length} arrays, {g.symbols.length} symbols, {states} states, {nodes} nodes, {edges} edges, top blocks {g.top.blocks.length}"

def main : IO Unit := do
  let dir : System.FilePath := "tests/corpus"
  let entries ← dir.readDir
  let files := (entries.map (·.path)).qsort (fun a b => a.toString < b.toString)
  let mut failures := 0
  for f in files do
    if f.toString.endsWith ".sdfg.json" then
      match ← SDFG.Json.readSDFG f with
      | .ok g => IO.println s!"OK   {f.fileName.getD ""}: {summary g}"
      | .error e =>
        failures := failures + 1
        IO.println s!"FAIL {f.fileName.getD ""}: {e}"
  IO.println s!"{failures} failures"

#eval main
