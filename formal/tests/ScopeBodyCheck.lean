-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! The check of the map-body congruence (`Dace.Equiv.ScopeBody`) over the
corpus.  Run with `lake env lean tests/ScopeBodyCheck.lean` from `formal/`
after generating the corpus (`scripts/dump_corpus.py`).

For every map of every top-level state:

* the unchanged state passes `sameOutside? g s s M` (the check accepts the
  identity replacement), and
* the state rebuilt with its own body (`State.withBody`, which moves the body
  nodes to the end of the node list) passes `bodyFits?` and `sameOutside?` in
  both directions;
* a body cut off from its entry (no edge from `M`) is rejected by
  `sameOutside?`, and an extra edge from a body node to an outer node
  changes the outer edges (the second hypothesis of `Congr.scopeBody`). -/

open Dace

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

def mapEntries (s : State) : List NodeId :=
  s.nodes.filterMap fun n => match n.kind with
    | .mapEntry _ => some n.id
    | _ => none

/-- The body of map `M`: its nodes and the edges ending in it. -/
def body (s : State) (M : NodeId) : List DNode × List DEdge :=
  let B := s.scopeNodes M
  (s.nodes.filter fun n => B.contains n.id, s.edges.filter fun e => B.contains e.dst)

/-- An extra edge from a body node (other than the exit) to an outer access
node, if both exist. -/
def leak (s : State) (M : NodeId) : Option DEdge := do
  let B := s.scopeNodes M
  let src ← (s.nodes.find? fun n => B.contains n.id && !n.isExit)
  let dst ← (s.nodes.find? fun n => !B.contains n.id && n.isAccess)
  pure { src := src.id, srcConn := none, dst := dst.id, dstConn := none, data := .dep }

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  let mut maps := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    match ← SDFG.Json.readSDFG f with
    | .error e => IO.println s!"IMPORT FAIL {name}: {e}"; bad := bad + 1
    | .ok g =>
      for s in g.top.allStates do
        for M in mapEntries s do
          maps := maps + 1
          let (bn, be) := body s M
          let s' := s.withBody M bn be
          let idOk := sameOutside? g s s M
          let fits := s.bodyFits? M bn be
          let fwd := sameOutside? g s' s M
          let bwd := sameOutside? g s s' M
          let cut := s.withBody M bn (be.filter (·.src != M))
          let cutOk := !sameOutside? g cut s M
          let leakOk := match leak s M with
            | some e =>
              let s₂ := { s with edges := s.edges ++ [e] }
              !(s₂.outerEdges (s₂.scopeNodes M) == s.outerEdges (s.scopeNodes M))
            | none => true
          if idOk && fits && fwd && bwd && cutOk && leakOk then
            IO.println s!"ok     {name} [{s.label}] map {M}"
          else
            IO.println s!"FAIL   {name} [{s.label}] map {M}: identity {idOk}, fits {fits}, rebuilt {fwd}/{bwd}, cut rejected {cutOk}, leak rejected {leakOk}"
            bad := bad + 1
  IO.println s!"{maps} maps, {bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} scope-body checks failed"

#eval main
