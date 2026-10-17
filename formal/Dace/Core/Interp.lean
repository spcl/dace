-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Tasklet
import Dace.Core.Wcr

/-!
# Interpretations

Everything the semantics leaves open is bundled in an `Interp`: opaque
functions in symbolic expressions, WCR operator application, tasklet bodies
and library nodes.  Theorems are stated for every (lawful) interpretation;
the interpreter instantiates it concretely.
-/

namespace Dace

/-- Semantics of a library node call: input connector values to output
connector values (`none` = unsupported node). -/
abbrev LibrarySem := String → (Conn → Option ConnIn) → Option TaskletOut

structure Interp where
  expr : ExprInterp
  wcr : WcrSem
  tasklet : TaskletSem
  /-- Keyed by the library node's class path. -/
  library : LibrarySem

namespace Interp

/-- The purely uninterpreted instance: everything open is undefined. -/
def uninterpreted : Interp where
  expr := ExprInterp.pure
  wcr := WcrOp.applyConcrete
  tasklet := fun _ _ => none
  library := fun _ _ => none

/-- Lawfulness: the WCR semantics is commutative and associative. -/
abbrev Lawful (I : Interp) : Prop := I.wcr.Lawful

end Interp

end Dace
