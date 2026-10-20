-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Basic
import Dace.SDFG.Semantics.Leaf
import Dace.SDFG.Semantics.Exec

/-!
# SDFG semantics

Big-step **operational** semantics of SDFGs as a relation
`SDFG.Exec I g ρ₀ m₀ : Res Mem → Prop`: "program `g`, under interpretation
`I` (opaque functions, WCR operators, tasklet and library bodies), started
with symbol valuation `ρ₀` and memory `m₀`, may end in outcome `r`".
Outcomes are a final memory or an explicit error (`ErrReason`: data race,
blocked pop, undefined expression, unset reference, out-of-bounds access
(`Decisions.md#D65`), undefined tasklet, ...),
a `malformed` program, or an `unsupported` construct (valid DaCe that is not
modeled, `Decisions.md#D59`).
Divergence is the absence of any derivation.

## Style

* **Relational and nondeterministic where DaCe is.**  The relations are
  mutual inductive predicates over the mutual syntax
  (`Dace.SDFG.Semantics.Exec`).  Nondeterminism appears only as explicit
  choices in constructors: the topological order of a scope body or state
  component (`Decisions.md#D20`, `#D43`), the interstate edge taken among the
  true ones (`#D2`), and the processing element of a consume task (`#D38`).
* **Function-shaped rules.**  Each constructor computes its intermediate
  data with ordinary functions (`Dace.SDFG.Semantics.Leaf`) and has premises
  `f x = .ok y` / `f x = .err e`, so that a fuel-based interpreter can be
  proven sound by mirroring the constructors.
* **Dataflow denotes logs.**  Within a state, every sequential execution (a
  component, a map iteration, a consume task, a nested invocation) works on
  a local memory view and produces a `Log` of its reads, writes, WCR
  contributions, alias bindings and queue effects (`Dace.Core.Merge`).  The
  logs of parallel executions are merged commutatively onto the pre-scope
  memory by `merge`; a conflict is the error `race`.  Map schedules are
  inert (`#D3`), components of a state run in parallel (`#D4`).
* **Memory is linear.**  Containers are addressed by linear element offsets
  through their strides (`Ctx.loc`); views, references and nested-SDFG
  connectors are aliases resolved on every access (`Dace.Core.Memory`), so
  codegen's pointer arithmetic is reproduced exactly.
* **Total over syntax.**  Situations well-formedness excludes yield
  `malformed` outcomes rather than missing cases.

## Relations

| Relation | Meaning |
|----------|---------|
| `NodeExec ctx ρ s n st r` | one node `n` of state `s`, from `st` to `r : Res St` |
| `SeqExec ctx ρ s ids st r` | the nodes `ids` in this order |
| `OrderedExec ctx ρ s all direct st r` | view prologue (`#D31`), then the `direct` nodes in some topological order of `all` |
| `IterExec ctx ρ s entry params priv pre pts r` | the map iterations `pts`, each from `pre`, yielding their logs |
| `ConsumeTasks ctx ρ s entry c l np priv pre q r` | the consume tasks on queue `q`: logs, remaining queue, deferred pushes |
| `CompsExec ctx ρ s m comps r` | the concurrent components of a state, each from memory `m` |
| `StateExec ctx cst s r` | a state: components merged, `r : Res (Ctl × CSt)` |
| `BlockExec ctx cst b r` | a control-flow block |
| `RegionAt ctx rg cst bid r` / `RegionExec ctx cst rg r` | a region from block `bid` / from its start |
| `LoopExec` / `LoopGuard` / `LoopIter` / `LoopTail` | a loop region and its phases (`#D36`) |
| `CondExec ctx cst branches r` | a conditional block |
| `SDFGExec ctx ρ m g r` | an (outer or nested) SDFG invocation, `r : Res (Valuation × Mem × Log)` |
| `SDFG.Exec I g ρ₀ m₀ r` | the program-level relation |

`Ctx` carries the interpretation, the current SDFG and the container-name
prefix of the current nested invocation; `St` is the state of a sequential
dataflow execution (local memory, log, code-edge values); `CSt` the state
threaded through control flow (valuation, memory, log).

Semantic decisions are recorded in `Dace/Decisions.md` (D29–D43 for this
module).
-/
