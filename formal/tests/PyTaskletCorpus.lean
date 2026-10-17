-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Load every `.sdfg.json` of the corpus, collect all Python tasklets (across
nested SDFGs and loop/conditional regions), and check that the executable
statement parser (`Dace.Core.PyTasklet`) accepts each one.  Run with
`lake env lean tests/PyTaskletCorpus.lean` from `formal/`. -/

open Dace

/-- All tasklets of an SDFG, descending into states, regions, loops,
conditionals (via `Region.allStates`) and nested SDFGs. -/
partial def collectTasklets (g : SDFG) : List Tasklet :=
  let states := g.top.allStates
  let here := states.flatMap fun s => s.nodes.filterMap fun n =>
    match n.kind with
    | .tasklet t => some t
    | _ => none
  let nested := states.flatMap State.nestedSDFGs
  here ++ nested.flatMap collectTasklets

def isPython (t : Tasklet) : Bool :=
  match t.code.language with
  | .python => true
  | _ => false

def main : IO Unit := do
  let dir : System.FilePath := "tests/corpus"
  let entries ← dir.readDir
  let files := (entries.map (·.path)).qsort (fun a b => a.toString < b.toString)
  let mut totalParsed := 0
  let mut totalUnparsed := 0
  for f in files do
    if f.toString.endsWith ".sdfg.json" then
      match ← SDFG.Json.readSDFG f with
      | .error e => IO.println s!"FAIL {f.fileName.getD ""}: {e}"
      | .ok g =>
        let ts := (collectTasklets g).filter isPython
        let unparsed := ts.filter fun t => (PyTasklet.parseStmts t.code.code).isNone
        let parsed := ts.length - unparsed.length
        totalParsed := totalParsed + parsed
        totalUnparsed := totalUnparsed + unparsed.length
        IO.println s!"{f.fileName.getD ""}: {parsed}/{ts.length} tasklets parsed"
        for t in unparsed do
          IO.println s!"    UNPARSED: {t.code.code}"
  IO.println s!"total: {totalParsed} parsed, {totalUnparsed} unparsed"

#eval main
