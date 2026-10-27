-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! The SDFG–SDFG checker (`Dace.Equiv.SdfgCorr`) over the corpus.  Run with
`lake env lean tests/SdfgCorrCorpus.lean` from `formal/` after generating the
corpus (`scripts/dump_corpus.py`).

For every corpus SDFG, a variant with the nodes of every state renumbered
(reversed ids, nested SDFG nodes fixed) must be certified `EQUIVALENT`, and a
variant with one state label changed must not be. -/

open Dace

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

/-- Reverse the ids of the nodes that are not nested SDFGs. -/
def reversePairs (s : State) : List (NodeId × NodeId) :=
  let ids := (s.nodes.filter (!·.isNested)).map (·.id)
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

/-- Change the label of the first state found. -/
partial def relabelFirstLabel (r : Region) : Option Region :=
  let rec go : List Block → Option (List Block)
    | [] => none
    | .state s :: bs => some (.state { s with label := s.label ++ "_x" } :: bs)
    | b :: bs => (go bs).map (b :: ·)
  (go r.blocks).map fun bs => { r with blocks := bs }

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  let mut ok := 0
  let mut steps := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    match ← SDFG.Json.readSDFG f with
    | .error e => IO.println s!"IMPORT FAIL {name}: {e}"; bad := bad + 1
    | .ok g =>
      let v := { g with top := relabelRegion g.top }
      match sdfgCheck g v with
      | .ok sts => ok := ok + 1; steps := steps + sts.length
      | .error m => IO.println s!"UNKNOWN  {name}: {m}"; bad := bad + 1
      match relabelFirstLabel g.top with
      | some top =>
        if (sdfgCheck g { g with top }).toBool then
          IO.println s!"WRONGLY EQUIVALENT {name} (changed state label)"
          bad := bad + 1
      | none => pure ()
  IO.println s!"{ok} relabeled variants certified ({steps} rule applications), {bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} failures"

#eval main
