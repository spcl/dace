-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! The hypotheses of the determinism theorems (`Dace.Equiv.Det`) over the
corpus.  Run with `lake env lean tests/DetCheck.lean` from `formal/` after
generating the corpus (`scripts/dump_corpus.py`).

* Every state passes `scopesOk?` (map scopes list their nodes once and are
  convex; the corpus states are well formed), and the states in the
  fragment `seqPure?` are counted.
* A map scope made non-convex by an extra dependency edge from its exit
  back to its entry (a path leaving the scope and re-entering it) is
  rejected.
* The programs passing the whole-program check `SDFG.det?`
  (`Dace.Equiv.DetProg`: every state in the fragment with convex scopes)
  are counted; a program passes exactly when all its states do. -/

open Dace

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

def mapEntries (s : State) : List NodeId :=
  s.nodes.filterMap fun n => match n.kind with
    | .mapEntry _ => some n.id
    | _ => none

/-- The state with an extra edge from the exit of map `M` back to `M`. -/
def loopBack (s : State) (M : NodeId) : Option State := do
  let x ← s.nodes.find? fun n => (s.scopeNodes M).contains n.id && n.isExit
  pure { s with edges := s.edges ++ [{ src := x.id, srcConn := none, dst := M, dstConn := none, data := .dep }] }

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  let mut states := 0
  let mut inFrag := 0
  let mut maps := 0
  let mut progs := 0
  let mut detProgs := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    match ← SDFG.Json.readSDFG f with
    | .error e => IO.println s!"IMPORT FAIL {name}: {e}"; bad := bad + 1
    | .ok g =>
      let ctx := checkCtx g
      progs := progs + 1
      let d := g.det?
      if d then detProgs := detProgs + 1
      let all := g.top.allStates.all fun s => seqPure? ctx s && scopesOk? s
      if d != all then
        IO.println s!"FAIL   {name}: SDFG.det? {d} but all states pass: {all}"
        bad := bad + 1
      IO.println s!"prog   {name} SDFG.det? {d}"
      for s in g.top.allStates do
        states := states + 1
        let sp := seqPure? ctx s
        if sp then inFrag := inFrag + 1
        if !scopesOk? s then
          IO.println s!"FAIL   {name} [{s.label}]: scopesOk? rejects a corpus state"
          bad := bad + 1
        for M in mapEntries s do
          maps := maps + 1
          match loopBack s M with
          | some s' =>
            if s'.convex? (s'.scopeNodes M) then
              IO.println s!"FAIL   {name} [{s.label}] map {M}: non-convex scope accepted"
              bad := bad + 1
          | none => pure ()
        IO.println s!"ok     {name} [{s.label}] seqPure? {sp}"
  IO.println s!"{states} states ({inFrag} in the fragment), {maps} maps, {progs} programs ({detProgs} pass SDFG.det?), {bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} determinism checks failed"

#eval main
