-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Parse every line of a text file of DaCe expression strings and report the
ones that fail to parse or to round-trip.  Run with
`DACE_EXPRS=path lake env lean tests/ParseCorpus.lean`. -/

open Dace

def checkLine (s : String) : Option String :=
  match Expr.parse? s with
  | none =>
    match Expr.parseAssign? s with
    | some _ => none
    | none => some s!"PARSE FAIL: {s}"
  | some e =>
    if Expr.parse? e.toPy == some e then none
    else some s!"ROUNDTRIP FAIL: {s}  =>  {e.toPy}"

def runCorpus : IO Unit := do
  let path ← IO.getEnv "DACE_EXPRS"
  match path with
  | none => IO.println "DACE_EXPRS not set"
  | some p =>
    let content ← IO.FS.readFile p
    let lines := content.splitOn "\n" |>.filter (· ≠ "")
    let fails := lines.filterMap checkLine
    for f in fails.take 50 do IO.println f
    IO.println s!"{lines.length} expressions, {fails.length} failures"

#eval runCorpus
