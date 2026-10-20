-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Verdicts of the static race check `raceFree?` (`Dace.Equiv.RaceFree`)
over the corpus.  Run with `lake env lean tests/RaceFreeCorpus.lean`
from `formal/` after generating the corpus (`scripts/dump_corpus.py`).

Every program with a pinned verdict must get exactly that verdict; programs
that race by construction (`unordered_writes`, `map_same_write`) must never be
`RACE-FREE`.  `UNKNOWN` verdicts name the reason (outside the fragment). -/

open Dace

/-- Pinned verdicts: `true` = `RACE-FREE`, `false` = `UNKNOWN`. -/
def expected : List (String × Bool) :=
  [ ("augassign_wcr", true), ("axpy", true), ("loop_with_branch", true), ("matmul", true),
    ("matmul.tiled", true), ("matmul_lib", true), ("nested_call", true), ("oob_allowed", true),
    ("oob_read", true), ("private_stage_out", true), ("reduce_rows", true), ("slice_view", true),
    ("stencil2d", true), ("strided_sum", true), ("two_states", true), ("while_loop", true),
    ("write_outside", true), ("loop_control", true), ("early_return", true), ("do_while", true),
    ("do_while_cond_first", true), ("regions", true),
    -- consume scope, a view, nested SDFGs, dynamic map ranges: outside the fragment
    ("consume_sum", false), ("empty_view", false), ("nested_maps_transient", false),
    ("reduce_rows.expanded", false), ("spmv", false),
    -- races
    ("unordered_writes", false), ("map_same_write", false) ]

/-- Programs that race: `RACE-FREE` would contradict `raceFree?_sound`. -/
def racy : List String := ["unordered_writes", "map_same_write"]

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    match ← SDFG.Json.readSDFG f with
    | .error e => IO.println s!"IMPORT FAIL {name}: {e}"; bad := bad + 1
    | .ok g =>
      let v := raceFree? g
      let shown := if v then "RACE-FREE" else "UNKNOWN"
      if v && racy.contains name then
        IO.println s!"UNSOUND {name}: RACE-FREE for a racing program"; bad := bad + 1
      else match expected.lookup name with
        | some e =>
          if e == v then IO.println s!"ok     {name}: {shown}"
          else IO.println s!"FAIL   {name}: {shown}, expected {if e then "RACE-FREE" else "UNKNOWN"}"; bad := bad + 1
        | none => IO.println s!"new    {name}: {shown} (no pinned verdict)"
  IO.println s!"{bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} race-check verdicts differ"

#eval main
