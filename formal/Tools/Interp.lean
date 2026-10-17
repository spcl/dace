-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! `lake exe dace-interp`: run an SDFG or schedule tree on concrete inputs. -/

def main (args : List String) : IO UInt32 := do
  IO.eprintln s!"dace-interp: not implemented yet (args: {args})"
  return 2
