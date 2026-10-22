-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
# `lake exe dace-equiv`

Verified checks on SDFGs (and, later, equivalence of IR pairs):

```
lake exe dace-equiv racefree [--emit-obligations DIR] <sdfg.json>...
lake exe dace-equiv stree <sdfg.json> <stree.json>
```

`stree` runs the pair checker `Corr.corr?` (`Dace.Equiv.CorrCheck`) on an
SDFG and its schedule tree and prints `REFINES` (every run of the tree is a
run of the SDFG, or the SDFG may end in undefined behavior: theorem
`Dace.Corr.corr?_exec`; `Refines t g` for a supported SDFG:
`Dace.Corr.corr?_refines`, exit `0`) or `UNKNOWN: <reason>` (outside the
checker's fragment, exit `1`).

`racefree` runs the static race check `raceFree?` (`Dace.Equiv.RaceFree`),
whose `true` answer is proven to imply `RaceFree` (`raceFree?_sound`), and
prints one line per file: `RACE-FREE <file>` or `UNKNOWN <file>` (the check
is incomplete: `UNKNOWN` is not a race).  Exit code `0` if every file is
race-free, `1` if some answer is `UNKNOWN` or a file cannot be imported, `2`
on usage errors.

With `--emit-obligations DIR`, a program that `raceFree?` cannot decide but
whose remaining disjointness queries (`raceObligations`) the assumed-obligations
check `raceFreeWith` accepts is reported as `OBLIGATIONS <k> <file>` (race-free only if they hold), and
the `k` queries are written to `DIR/<name>.obligations.lean` as closed theorems
over ℤ (`Query.toLeanFile`).  When that file checks (`lake env lean`), the
queries are valid and `raceFreeWith_sound` gives race freedom (the link from a
query to its printed statement is the trusted printer `Query.toLean`).
-/

open Dace

def raceFreeMain (emit : Option String) (paths : List String) : IO UInt32 := do
  let mut code : UInt32 := 0
  for p in paths do
    match ← SDFG.Json.readSDFG p with
    | .error e =>
      IO.println s!"ERROR {p}: cannot import SDFG: {e}"
      code := 1
    | .ok g =>
      if raceFree? g then IO.println s!"RACE-FREE {p}"
      else
        let obl := raceObligations g
        match emit with
        | some dir =>
          if !obl.isEmpty && raceFreeWith (fun q => obl.contains q) g then
            let base := ((System.FilePath.mk p).fileName.getD "sdfg").replace ".sdfg.json" ""
            let file := System.FilePath.mk dir / s!"{base}.obligations.lean"
            IO.FS.createDirAll dir
            IO.FS.writeFile file (Query.toLeanFile (obl.zipIdx.map fun (q, i) => (s!"obligation_{i}", q)))
            IO.println s!"OBLIGATIONS {obl.length} {p} ({file}): race-free if this file checks"
          else
            IO.println s!"UNKNOWN {p}"
          code := 1
        | none =>
          IO.println s!"UNKNOWN {p}"
          code := 1
  return code

def streeMain (sp tp : String) : IO UInt32 := do
  match ← SDFG.Json.readSDFG sp, ← readSTree tp with
  | .error e, _ => IO.println s!"ERROR {sp}: cannot import SDFG: {e}"; return 1
  | _, .error e => IO.println s!"ERROR {tp}: cannot import schedule tree: {e}"; return 1
  | .ok g, .ok t =>
    match Corr.corrCheck g t with
    | .ok () =>
      IO.println s!"REFINES {tp} {sp} (theorem Dace.Corr.corr?_exec)"
      return 0
    | .error m =>
      IO.println s!"UNKNOWN {tp} {sp}: {m}"
      return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | "racefree" :: "--emit-obligations" :: dir :: paths@(_ :: _) => raceFreeMain (some dir) paths
  | "racefree" :: paths@(_ :: _) => raceFreeMain none paths
  | ["stree", sp, tp] => streeMain sp tp
  | _ =>
    IO.eprintln "usage: dace-equiv racefree [--emit-obligations DIR] <sdfg.json>...\n       dace-equiv stree <sdfg.json> <stree.json>"
    return 2
