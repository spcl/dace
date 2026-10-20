-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! The verified SDFG / schedule-tree pair checker (`Dace.Equiv.CorrCheck`)
over the corpus.  Run with `lake env lean tests/EquivCorpus.lean` from
`formal/` after generating the corpus (`scripts/dump_corpus.py`).

For every `<name>.sdfg.json` with a `<name>.stree.json`, prints `REFINES`
(`corr?` accepts the pair: every run of the tree is a run of the SDFG or the
SDFG may be undefined, `Corr.corr?_exec`) or `UNKNOWN: <reason>`, and fails
if a program listed in `expected` is not certified. -/

open Dace

def corpusDir : IO System.FilePath :=
  return (← IO.getEnv "DACE_FORMAL_CORPUS").getD "tests/corpus"

/-- Programs the checker must certify. -/
def expected : List String :=
  ["axpy", "matmul", "matmul.tiled", "stencil2d", "strided_sum", "two_states", "reduce_rows", "matmul_lib",
   "map_same_write", "unordered_writes", "oob_read", "oob_allowed", "write_outside", "slice_view",
   "loop_with_branch", "while_loop", "nested_call", "loop_control", "early_return", "do_while", "do_while_cond_first"]

def main : IO Unit := do
  let dir ← corpusDir
  let files := ((← dir.readDir).map (·.path)).filter (·.toString.endsWith ".sdfg.json")
  if files.isEmpty then
    throw <| IO.userError s!"no *.sdfg.json files in {dir}; run scripts/dump_corpus.py first"
  let mut bad := 0
  let mut certified := 0
  for f in files.qsort (fun a b => a.toString < b.toString) do
    let name := (f.fileName.getD "").replace ".sdfg.json" ""
    let tf := dir / s!"{name}.stree.json"
    if !(← tf.pathExists) then continue
    match ← SDFG.Json.readSDFG f, ← readSTree tf with
    | .ok g, .ok t =>
      match Corr.corrCheck g t with
      | .ok () =>
        certified := certified + 1
        IO.println s!"REFINES  {name}"
      | .error m =>
        IO.println s!"UNKNOWN  {name}: {m}"
        if expected.contains name then bad := bad + 1
    | _, _ => IO.println s!"IMPORT FAIL {name}"; bad := bad + 1
  IO.println s!"{certified} pairs certified, {bad} failures"
  if bad > 0 then throw <| IO.userError s!"{bad} expected pairs not certified"

#eval main
