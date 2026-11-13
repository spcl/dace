-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics.Basic
import Dace.STree.Semantics.Leaf
import Dace.STree.Semantics.Exec

/-!
# Schedule tree semantics

Big-step **operational** semantics of schedule trees as a relation
`STree.Exec I t ρ₀ m₀ : Res Mem → Prop`, with the same outcomes
(`Res`, `ErrReason`) and the same observable convention as `SDFG.Exec`:
"tree `t`, under interpretation `I`, started with symbol valuation `ρ₀` and
memory `m₀`, may end in outcome `r`".  Divergence is the absence of a
derivation.

## Style

* **Sequential siblings.**  The children of every scope run in list order
  (`Decisions.md#D20`); the sibling-list relation `ListExec` threads the
  control-flow state `CSt` (valuation, memory, log) together with a sibling
  context `SibCtx`: the dynamic scope inputs pending for the next dataflow
  scope (`#D50`) and the state of an open `if`/`elif`/`else` chain (`#D52`).
* **Function-shaped rules** over the leaf functions of
  `Dace.STree.Semantics.Leaf`, which reuse the container-level primitives
  of the SDFG semantics (`Dace.SDFG.Semantics.Leaf`): a tree node and the
  SDFG node it came from execute the same code.
* **Parallel scopes denote logs.**  Map iterations and consume tasks start
  from the pre-scope memory and their logs are merged by `merge` (`#D3`,
  `#D38`); the private transients of a scope (`#D54`) are dropped first.
* **Control signals** `TCtl` (`#D51`): `break`/`continue` for the innermost
  loop, `ret` (= `GotoNode none`) for the whole program, and `goto l`, which
  unwinds scopes until a sibling list resolves the label (forward gotos
  anywhere, backward gotos inside a `GBlock`).
* **Total over syntax.**  Situations well-formedness excludes yield
  `malformed` outcomes.

## Relations

| Relation | Meaning |
|----------|---------|
| `NodeExec tc sc n cst r` | one node |
| `ListExec tc full gb sc nodes cst r` | the remaining siblings `nodes` of the list `full` |
| `IterExec tc params body priv pre pts r` | the map iterations `pts` |
| `ConsumeTasks tc ρ c d body l np priv pre q r` | the consume tasks on queue `q` |
| `LoopExec` / `LoopGuard` / `LoopIter` / `LoopTail` | a loop scope and its phases (`#D36`) |
| `STree.Exec I t ρ₀ m₀ r` | the program-level relation |

Semantic decisions are recorded in `Dace/Decisions.md` (D48–D58 for this
module).
-/
