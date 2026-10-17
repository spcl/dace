-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Memlet

/-!
# Tasklets

Syntax of tasklets and the interface through which the semantics treats them
as **uninterpreted functions** of their connector values.  A tasklet's code is
kept verbatim; `Interp.tasklet` (see `Core/Interp`) gives it meaning.  The
executable Python-expression subset is provided in `Core/PyTasklet`.
-/

namespace Dace

/-- `dace.dtypes.Language`. -/
inductive Language where
  | python | cpp | mlir | other (name : String)
  deriving DecidableEq, Repr, Inhabited

structure CodeBlock where
  code : String
  language : Language := .python
  deriving DecidableEq, Repr, Inhabited

/-- A tasklet node. -/
structure Tasklet where
  label : String := ""
  code : CodeBlock
  /-- Input connectors with their (optional) declared types. -/
  inConns : List (Conn × Option Dtype) := []
  outConns : List (Conn × Option Dtype) := []
  /-- `Tasklet.side_effects` (`none` = auto-detect in DaCe). -/
  sideEffects : Option Bool := none
  deriving DecidableEq, Repr, Inhabited

/-- Value bound to an input connector when a tasklet runs. -/
inductive ConnIn where
  /-- A scalar connector (single-element memlet). -/
  | scalar (v : Val)
  /-- A pointer connector: elements of the memlet's subset, addressed by
      indices relative to the subset's lower corner. -/
  | array (get : Idx → Option Val)
  /-- A stream connector: the values currently available for popping. -/
  | stream (available : List Val)

/-- Effects of one tasklet execution. -/
structure TaskletOut where
  /-- Scalar output connectors that were assigned (unassigned dynamic
      outputs are absent, `Decisions.md#D13`). -/
  scalars : List (Conn × Val) := []
  /-- Element writes through pointer output connectors (relative indices). -/
  arrayWrites : List (Conn × Idx × Val) := []
  /-- Values pushed to stream output connectors, in order. -/
  pushes : List (Conn × List Val) := []
  /-- Number of values popped from stream input connectors. -/
  pops : List (Conn × Nat) := []
  deriving Inhabited

/-- Tasklet semantics: deterministic, `none` = undefined behaviour. -/
abbrev TaskletSem := Tasklet → (Conn → Option ConnIn) → Option TaskletOut

end Dace
