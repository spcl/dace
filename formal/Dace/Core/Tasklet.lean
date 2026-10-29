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

/-- Identifier tokens of a code string: the maximal runs of letters, digits
and `_` (a conservative over-approximation of the names it uses: names in
comments and string literals count as well). -/
def codeIdents (code : String) : List String :=
  let (toks, cur) := code.toList.foldl (fun (acc : List String × List Char) c =>
    if c.isAlphanum || c == '_' then (acc.1, c :: acc.2)
    else if acc.2.isEmpty then acc
    else (String.ofList acc.2.reverse :: acc.1, [])) ([], [])
  (if cur.isEmpty then toks else String.ofList cur.reverse :: toks).reverse

/-- Does the tasklet's code name container `n` other than through one of its
connectors (`Tasklet.free_symbols` ignores connector names,
`dace/sdfg/nodes.py:602-606`)?  Used for the private transients of a scope
(`Decisions.md#D39`). -/
def Tasklet.namesInCode (t : Tasklet) (n : Name) : Bool :=
  !(t.inConns.any (·.1 = n)) && !(t.outConns.any (·.1 = n)) && (codeIdents t.code.code).contains n

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

/-- Tasklet semantics: deterministic, `none` = undefined behaviour.  Besides
its connector bindings a tasklet sees the current symbol valuation (SDFG
symbols, map parameters, dynamic scope inputs), as its code does in the
generated program (`Decisions.md#D46`). -/
abbrev TaskletSem := Tasklet → Valuation → (Conn → Option ConnIn) → Option TaskletOut

end Dace
