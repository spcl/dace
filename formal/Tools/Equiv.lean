-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
# `lake exe dace-equiv`

Verified checks on SDFGs (and, later, equivalence of IR pairs):

```
lake exe dace-equiv racefree <sdfg.json>...
```

`racefree` runs the static race check `raceFree?` (`Dace.Equiv.RaceFree`),
whose `true` answer is proven to imply `RaceFree` (`raceFree?_sound`), and
prints one line per file: `RACE-FREE <file>` or `UNKNOWN <file>` (the check
is incomplete: `UNKNOWN` is not a race).  Exit code `0` if every file is
race-free, `1` if some answer is `UNKNOWN` or a file cannot be imported, `2`
on usage errors.
-/

open Dace

def raceFreeMain (paths : List String) : IO UInt32 := do
  let mut code : UInt32 := 0
  for p in paths do
    match ← SDFG.Json.readSDFG p with
    | .error e =>
      IO.println s!"ERROR {p}: cannot import SDFG: {e}"
      code := 1
    | .ok g =>
      if raceFree? g then IO.println s!"RACE-FREE {p}"
      else
        IO.println s!"UNKNOWN {p}"
        code := 1
  return code

def main (args : List String) : IO UInt32 := do
  match args with
  | "racefree" :: paths@(_ :: _) => raceFreeMain paths
  | _ =>
    IO.eprintln "usage: dace-equiv racefree <sdfg.json>..."
    return 2
