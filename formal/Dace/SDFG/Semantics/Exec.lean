-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Leaf
import Dace.Core.MergeLemmas

/-!
# SDFG semantics: the execution relations

Big-step relational semantics of SDFGs as one mutual family of inductive
predicates.  Every constructor is *function-shaped*: intermediate data is
computed by the functions of `Dace.SDFG.Semantics.Leaf`, premises are
equations `f x = .ok y` / `f x = .err e`, and the only nondeterminism is in
explicit choices (topological order of a scope body or component, the
interstate edge taken among the true ones, the processing element of a
consume task).  Divergence (an endless loop, a never-quiescent consume) is
the absence of a derivation.

## Relations

| Relation | Meaning |
|----------|---------|
| `NodeExec ctx ρ s n st r` | node `n` of state `s`, from `st` to `r` |
| `SeqExec ctx ρ s ids st r` | the nodes `ids` in this order, each from an empty log; `r` carries the trace of their deltas |
| `OrderedExec ctx ρ s all direct st r` | view prologue, then the `direct` nodes in some topological order of `all`, free of happens-before races |
| `IterExec ctx ρ s entry params priv pre pts r` | the map iterations `pts`, each from `pre`, yielding their logs |
| `ConsumeTasks ctx ρ s entry c l np priv pre q r` | the consume tasks on queue `q`: logs, remaining queue, deferred pushes |
| `CompsExec ctx ρ s m comps r` | the concurrent components of a state, each from `m` |
| `StateExec ctx cst s r` | a state |
| `BlockExec ctx cst b r` | a control-flow block |
| `RegionAt ctx r cst bid res` | a region from block `bid` |
| `RegionExec ctx cst r res` | a region from its start block |
| `LoopExec` / `LoopGuard` / `LoopIter` / `LoopTail` | a loop region (`Decisions.md#D36`) |
| `CondExec ctx cst branches r` | a conditional block |
| `SDFGExec ctx ρ m g r` | an (outer or nested) SDFG invocation |

Results are `Res St` for dataflow, `Res (Ctl × CSt)` for control flow and
`Res (Valuation × Mem × Log)` for SDFG invocations.  Dataflow results carry
the *local* memory view and a log; the logs of parallel executions (map
iterations, consume tasks, state components) are merged commutatively by
`merge` (`Decisions.md#D3`, `#D4`), which is where races between parallel
executions surface.  Within one sequential execution, every node's own log
delta is recorded in a trace (`SeqExec`); two nodes that no path of the
state orders must not have conflicting deltas (`State.hbFree`), otherwise
the execution is a race (`OrderedExec.hbRace`, `Decisions.md#D61`).
-/

set_option autoImplicit false

namespace Dace

mutual

/-- Execution of one node in the current sequential execution. -/
inductive NodeExec : Ctx → Valuation → State → DNode → St → Res St → Prop where
  /-- Access, tasklet, library and exit nodes (`execSimple`). -/
  | simple {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St}
      (h : n.isSimple = true) :
      NodeExec ctx ρ s n st (execSimple ctx ρ s n st)
  /-- Map entry: dynamic inputs or iteration space undefined. -/
  | mapSetupErr {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St} {mp : MapNode} {e : ErrReason}
      (hk : n.kind = .mapEntry mp) (hs : mapSetup ctx ρ s n mp st = .err e) :
      NodeExec ctx ρ s n st (.err e)
  /-- Map entry: an iteration fails. -/
  | mapIterErr {ctx : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {st st₁ : St} {mp : MapNode}
      {pts : List Idx} {e : ErrReason}
      (hk : n.kind = .mapEntry mp) (hs : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts))
      (hi : IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e)) :
      NodeExec ctx ρ s n st (.err e)
  /-- Map entry: all iterations run from the pre-map state, their logs are
      merged (`Decisions.md#D3`, `#D39`). -/
  | mapScope {ctx : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {st st₁ : St} {mp : MapNode}
      {pts : List Idx} {logs : List Log}
      (hk : n.kind = .mapEntry mp) (hs : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts))
      (hi : IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs)) :
      NodeExec ctx ρ s n st (scopeFinish ctx st₁ logs)
  /-- Consume entry: set-up fails. -/
  | consumeSetupErr {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St} {c : ConsumeNode} {e : ErrReason}
      (hk : n.kind = .consumeEntry c) (hs : consumeSetup ctx ρ s n c st = .err e) :
      NodeExec ctx ρ s n st (.err e)
  /-- Consume entry: a task fails. -/
  | consumeTasksErr {ctx : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {st st₁ : St} {c : ConsumeNode}
      {l : Loc} {np : Int} {e : ErrReason}
      (hk : n.kind = .consumeEntry c) (hs : consumeSetup ctx ρ s n c st = .ok (ρ', st₁, l, np))
      (ht : ConsumeTasks ctx ρ' s n.id c l np (ctx.privateTransients s n.id) st₁
              (st₁.mem.queues l) (.err e)) :
      NodeExec ctx ρ s n st (.err e)
  /-- Consume entry (`Decisions.md#D38`): the tasks consume the queue, their
      logs are merged, the consumed stream ends with the remaining elements
      followed by the body's deferred pushes. -/
  | consumeScope {ctx : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {st st₁ : St} {c : ConsumeNode}
      {l : Loc} {np : Int} {logs : List Log} {rest pushed : List Val}
      (hk : n.kind = .consumeEntry c) (hs : consumeSetup ctx ρ s n c st = .ok (ρ', st₁, l, np))
      (ht : ConsumeTasks ctx ρ' s n.id c l np (ctx.privateTransients s n.id) st₁
              (st₁.mem.queues l) (.ok (logs, rest, pushed))) :
      NodeExec ctx ρ s n st (consumeFinish ctx st₁ l logs rest pushed)
  /-- Nested SDFG: symbol mapping or connector binding fails. -/
  | nestedSetupErr {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St} {inner : SDFG}
      {mapping : List (Sym × Expr)} {noInline : Bool} {e : ErrReason}
      (hk : n.kind = .nested inner mapping noInline) (hs : nestedSetup ctx ρ s n inner mapping st = .err e) :
      NodeExec ctx ρ s n st (.err e)
  /-- Nested SDFG: the inner execution fails. -/
  | nestedInnerErr {ctx ctx' : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {st st₁ : St} {inner : SDFG}
      {mapping : List (Sym × Expr)} {noInline : Bool} {aliased : List Name} {e : ErrReason}
      (hk : n.kind = .nested inner mapping noInline)
      (hs : nestedSetup ctx ρ s n inner mapping st = .ok (ctx', ρ', st₁, aliased))
      (hr : SDFGExec ctx' ρ' st₁.mem inner (.err e)) :
      NodeExec ctx ρ s n st (.err e)
  /-- Nested SDFG (`Decisions.md#D15`, `#D34`): run the inner SDFG on the
      connector-aliased memory; symbols never flow out. -/
  | nested {ctx ctx' : Ctx} {ρ ρ' ρ₂ : Valuation} {s : State} {n : DNode} {st st₁ : St} {inner : SDFG}
      {mapping : List (Sym × Expr)} {noInline : Bool} {aliased : List Name} {m₂ : Mem} {g₂ : Log}
      (hk : n.kind = .nested inner mapping noInline)
      (hs : nestedSetup ctx ρ s n inner mapping st = .ok (ctx', ρ', st₁, aliased))
      (hr : SDFGExec ctx' ρ' st₁.mem inner (.ok (ρ₂, m₂, g₂))) :
      NodeExec ctx ρ s n st (nestedFinish ctx' ρ' s n aliased st₁ m₂ g₂)

/-- Sequential execution of a list of nodes in the given order.  Each node
runs from the current state with an empty log (`St.forNode`); the result is
the final state, whose log accumulates the nodes' deltas (`St.afterNode`),
and the *trace*: every node with its own delta, in execution order
(`Decisions.md#D61`). -/
inductive SeqExec : Ctx → Valuation → State → List NodeId → St → Res (St × List (NodeId × Log)) → Prop where
  | nil {ctx : Ctx} {ρ : Valuation} {s : State} {st : St} :
      SeqExec ctx ρ s [] st (.ok (st, []))
  | missing {ctx : Ctx} {ρ : Valuation} {s : State} {id : NodeId} {rest : List NodeId} {st : St}
      (h : s.node? id = none) :
      SeqExec ctx ρ s (id :: rest) st (.err (.malformed "unknown node in an execution order"))
  | headErr {ctx : Ctx} {ρ : Valuation} {s : State} {id : NodeId} {rest : List NodeId} {n : DNode} {st : St}
      {e : ErrReason}
      (h : s.node? id = some n) (hn : NodeExec ctx ρ s n st.forNode (.err e)) :
      SeqExec ctx ρ s (id :: rest) st (.err e)
  | tailErr {ctx : Ctx} {ρ : Valuation} {s : State} {id : NodeId} {rest : List NodeId} {n : DNode} {st st' : St}
      {e : ErrReason}
      (h : s.node? id = some n) (hn : NodeExec ctx ρ s n st.forNode (.ok st'))
      (hr : SeqExec ctx ρ s rest (st.afterNode st') (.err e)) :
      SeqExec ctx ρ s (id :: rest) st (.err e)
  | cons {ctx : Ctx} {ρ : Valuation} {s : State} {id : NodeId} {rest : List NodeId} {n : DNode} {st st' st₂ : St}
      {tr : List (NodeId × Log)}
      (h : s.node? id = some n) (hn : NodeExec ctx ρ s n st.forNode (.ok st'))
      (hr : SeqExec ctx ρ s rest (st.afterNode st') (.ok (st₂, tr))) :
      SeqExec ctx ρ s (id :: rest) st (.ok (st₂, (id, st'.log) :: tr))

/-- Execution of the `direct` nodes of a scope (or component) in some
topological order of `all` (the scope's nodes including nested ones, so
that paths through nested scopes order their endpoints), after the view
prologue (`Decisions.md#D31`).  The execution is defined only if no two of
its nodes that the state leaves unordered have conflicting deltas
(`State.hbFree`); otherwise it is a race (`Decisions.md#D61`). -/
inductive OrderedExec : Ctx → Valuation → State → List NodeId → List NodeId → St → Res St → Prop where
  | prologueErr {ctx : Ctx} {ρ : Valuation} {s : State} {all direct : List NodeId} {st : St} {e : ErrReason}
      (hp : bindViews ctx ρ s (s.viewOrder ctx direct) st = .err e) :
      OrderedExec ctx ρ s all direct st (.err e)
  | seqErr {ctx : Ctx} {ρ : Valuation} {s : State} {all direct order : List NodeId} {st st₁ : St} {e : ErrReason}
      (ho : s.isTopologicalOrder all order = true)
      (hp : bindViews ctx ρ s (s.viewOrder ctx direct) st = .ok st₁)
      (hs : SeqExec ctx ρ s (order.filter direct.contains) st₁ (.err e)) :
      OrderedExec ctx ρ s all direct st (.err e)
  | hbRace {ctx : Ctx} {ρ : Valuation} {s : State} {all direct order : List NodeId} {st st₁ st₂ : St}
      {tr : List (NodeId × Log)}
      (ho : s.isTopologicalOrder all order = true)
      (hp : bindViews ctx ρ s (s.viewOrder ctx direct) st = .ok st₁)
      (hs : SeqExec ctx ρ s (order.filter direct.contains) st₁ (.ok (st₂, tr)))
      (hf : s.hbFree tr = false) :
      OrderedExec ctx ρ s all direct st (.err (.race (s.hbWitness tr)))
  | run {ctx : Ctx} {ρ : Valuation} {s : State} {all direct order : List NodeId} {st st₁ st₂ : St}
      {tr : List (NodeId × Log)}
      (ho : s.isTopologicalOrder all order = true)
      (hp : bindViews ctx ρ s (s.viewOrder ctx direct) st = .ok st₁)
      (hs : SeqExec ctx ρ s (order.filter direct.contains) st₁ (.ok (st₂, tr)))
      (hf : s.hbFree tr = true) :
      OrderedExec ctx ρ s all direct st (.ok st₂)

/-- The iterations of a map, each from the pre-map state `pre`, yielding
their logs without the scope's private transients. -/
inductive IterExec : Ctx → Valuation → State → NodeId → List Sym → List Name → St → List Idx → Res (List Log) → Prop where
  | nil {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name} {pre : St} :
      IterExec ctx ρ s entry params priv pre [] (.ok [])
  | headErr {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name}
      {pre : St} {i : Idx} {rest : List Idx} {e : ErrReason}
      (hb : OrderedExec ctx (ρ.bindIdx params i) s (s.scopeNodes entry) (s.directChildren (some entry))
              (iterStart priv pre) (.err e)) :
      IterExec ctx ρ s entry params priv pre (i :: rest) (.err e)
  | tailErr {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name}
      {pre st : St} {i : Idx} {rest : List Idx} {e : ErrReason}
      (hb : OrderedExec ctx (ρ.bindIdx params i) s (s.scopeNodes entry) (s.directChildren (some entry))
              (iterStart priv pre) (.ok st))
      (hr : IterExec ctx ρ s entry params priv pre rest (.err e)) :
      IterExec ctx ρ s entry params priv pre (i :: rest) (.err e)
  | cons {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name}
      {pre st : St} {i : Idx} {rest : List Idx} {logs : List Log}
      (hb : OrderedExec ctx (ρ.bindIdx params i) s (s.scopeNodes entry) (s.directChildren (some entry))
              (iterStart priv pre) (.ok st))
      (hr : IterExec ctx ρ s entry params priv pre rest (.ok logs)) :
      IterExec ctx ρ s entry params priv pre (i :: rest) (.ok (st.log.dropNames priv :: logs))

/-- The tasks of a consume scope on the queue `q` (`Decisions.md#D38`):
task `k` sees the queue left by tasks `< k`, pops up to `chunksize`
elements, runs the body with a nondeterministic processing element, and the
scope stops when the quiescence condition fails.  The result is the task
logs (without effects on the consumed stream), the remaining queue and the
values the bodies pushed to the consumed stream. -/
inductive ConsumeTasks : Ctx → Valuation → State → NodeId → ConsumeNode → Loc → Int → List Name → St →
    List Val → Res (List Log × List Val × List Val) → Prop where
  | condErr {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : St} {q : List Val} {e : ErrReason}
      (hc : consumeCond ctx ρ c pre.mem l q = .err e) :
      ConsumeTasks ctx ρ s entry c l np priv pre q (.err e)
  | stop {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : St} {q : List Val}
      (hc : consumeCond ctx ρ c pre.mem l q = .ok false) :
      ConsumeTasks ctx ρ s entry c l np priv pre q (.ok ([], q, []))
  | bodyErr {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : St} {q : List Val} {p : Int} {e : ErrReason}
      (hc : consumeCond ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes entry)
              (s.directChildren (some entry)) (taskStart ctx s entry c l priv pre q) (.err e)) :
      ConsumeTasks ctx ρ s entry c l np priv pre q (.err e)
  | restErr {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
      {priv : List Name} {pre st : St} {q : List Val} {p : Int} {e : ErrReason}
      (hc : consumeCond ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes entry)
              (s.directChildren (some entry)) (taskStart ctx s entry c l priv pre q) (.ok st))
      (hr : ConsumeTasks ctx ρ s entry c l np priv pre (q.drop c.chunksize) (.err e)) :
      ConsumeTasks ctx ρ s entry c l np priv pre q (.err e)
  | step {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
      {priv : List Name} {pre st : St} {q : List Val} {p : Int} {logs : List Log} {rest pushed : List Val}
      (hc : consumeCond ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes entry)
              (s.directChildren (some entry)) (taskStart ctx s entry c l priv pre q) (.ok st))
      (hr : ConsumeTasks ctx ρ s entry c l np priv pre (q.drop c.chunksize) (.ok (logs, rest, pushed))) :
      ConsumeTasks ctx ρ s entry c l np priv pre q
        (.ok (taskLog st priv l :: logs, rest, taskPushed st priv l ++ pushed))

/-- The concurrent components of a state, each from the state's initial
memory with an empty log (`Decisions.md#D4`). -/
inductive CompsExec : Ctx → Valuation → State → Mem → List (List NodeId) → Res (List Log) → Prop where
  | nil {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} :
      CompsExec ctx ρ s m [] (.ok [])
  | headErr {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {comp : List NodeId} {rest : List (List NodeId)}
      {e : ErrReason}
      (hc : OrderedExec ctx ρ s comp (s.topLevelOf comp) { mem := m } (.err e)) :
      CompsExec ctx ρ s m (comp :: rest) (.err e)
  | tailErr {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {comp : List NodeId} {rest : List (List NodeId)}
      {st : St} {e : ErrReason}
      (hc : OrderedExec ctx ρ s comp (s.topLevelOf comp) { mem := m } (.ok st))
      (hr : CompsExec ctx ρ s m rest (.err e)) :
      CompsExec ctx ρ s m (comp :: rest) (.err e)
  | cons {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {comp : List NodeId} {rest : List (List NodeId)}
      {st : St} {logs : List Log}
      (hc : OrderedExec ctx ρ s comp (s.topLevelOf comp) { mem := m } (.ok st))
      (hr : CompsExec ctx ρ s m rest (.ok logs)) :
      CompsExec ctx ρ s m (comp :: rest) (.ok (st.log :: logs))

/-- A state: its components run concurrently from the state's memory with
the state's transients allocated (`Ctx.stateTransients`,
`Decisions.md#D68`), and their logs are merged onto the state's memory. -/
inductive StateExec : Ctx → CSt → State → Res (Ctl × CSt) → Prop where
  | compsErr {ctx : Ctx} {cst : CSt} {s : State} {e : ErrReason}
      (hc : CompsExec ctx cst.ρ s (ctx.allocate (ctx.stateTransients s) cst.mem) s.components (.err e)) :
      StateExec ctx cst s (.err e)
  | run {ctx : Ctx} {cst : CSt} {s : State} {logs : List Log}
      (hc : CompsExec ctx cst.ρ s (ctx.allocate (ctx.stateTransients s) cst.mem) s.components (.ok logs)) :
      StateExec ctx cst s (stateFinish ctx cst logs)

/-- A control-flow block. -/
inductive BlockExec : Ctx → CSt → Block → Res (Ctl × CSt) → Prop where
  | state {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)}
      (h : StateExec ctx cst s r) : BlockExec ctx cst (.state s) r
  | region {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)}
      (h : RegionExec ctx cst rg r) : BlockExec ctx cst (.region rg) r
  | loop {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (h : LoopExec ctx cst l r) : BlockExec ctx cst (.loop l) r
  | cond {ctx : Ctx} {cst : CSt} {label : String} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)}
      (h : CondExec ctx cst brs r) : BlockExec ctx cst (.cond label brs) r
  | brk {ctx : Ctx} {cst : CSt} : BlockExec ctx cst .breakBlock (.ok (.brk, cst))
  | cont {ctx : Ctx} {cst : CSt} : BlockExec ctx cst .continueBlock (.ok (.cont, cst))
  | ret {ctx : Ctx} {cst : CSt} : BlockExec ctx cst .returnBlock (.ok (.ret, cst))

/-- A region from block `bid`: run the block; a non-`normal` control signal
ends the region; otherwise take the true out-edge (`Decisions.md#D35`),
apply its assignments and continue; no true edge ends the region normally,
several true edges are undefined behavior (`Decisions.md#D2`). -/
inductive RegionAt : Ctx → Region → CSt → BlockId → Res (Ctl × CSt) → Prop where
  | missing {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId}
      (hb : rg.block? bid = none) :
      RegionAt ctx rg cst bid (.err (.malformed s!"block {bid} out of range"))
  | blockErr {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {b : Block} {e : ErrReason}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.err e)) :
      RegionAt ctx rg cst bid (.err e)
  | exit {ctx : Ctx} {rg : Region} {cst cst₁ : CSt} {bid : BlockId} {b : Block} {ctl : Ctl}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (ctl, cst₁))) (hc : ctl ≠ .normal) :
      RegionAt ctx rg cst bid (.ok (ctl, cst₁))
  | edgesErr {ctx : Ctx} {rg : Region} {cst cst₁ : CSt} {bid : BlockId} {b : Block} {e : ErrReason}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (.normal, cst₁)))
      (he : trueEdges ctx rg cst₁ bid = .err e) :
      RegionAt ctx rg cst bid (.err e)
  | stop {ctx : Ctx} {rg : Region} {cst cst₁ : CSt} {bid : BlockId} {b : Block}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (.normal, cst₁)))
      (he : trueEdges ctx rg cst₁ bid = .ok []) :
      RegionAt ctx rg cst bid (.ok (.normal, cst₁))
  | ambiguous {ctx : Ctx} {rg : Region} {cst cst₁ : CSt} {bid : BlockId} {b : Block} {ie₁ ie₂ : IEdge}
      {es : List IEdge}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (.normal, cst₁)))
      (he : trueEdges ctx rg cst₁ bid = .ok (ie₁ :: ie₂ :: es)) :
      RegionAt ctx rg cst bid (.err (.ambiguousTransition bid))
  | assignErr {ctx : Ctx} {rg : Region} {cst cst₁ : CSt} {bid : BlockId} {b : Block}
      {ie : IEdge} {e : ErrReason}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (.normal, cst₁)))
      (he : trueEdges ctx rg cst₁ bid = .ok [ie])
      (ha : applyAssigns ctx cst₁ ie.assigns = .err e) :
      RegionAt ctx rg cst bid (.err e)
  | step {ctx : Ctx} {rg : Region} {cst cst₁ cst₂ : CSt} {bid : BlockId} {b : Block}
      {ie : IEdge} {r : Res (Ctl × CSt)}
      (hb : rg.block? bid = some b) (h : BlockExec ctx cst b (.ok (.normal, cst₁)))
      (he : trueEdges ctx rg cst₁ bid = .ok [ie])
      (ha : applyAssigns ctx cst₁ ie.assigns = .ok cst₂) (hn : RegionAt ctx rg cst₂ ie.dst r) :
      RegionAt ctx rg cst bid r

/-- A region from its start block.  `kind` is semantically irrelevant. -/
inductive RegionExec : Ctx → CSt → Region → Res (Ctl × CSt) → Prop where
  | start {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)}
      (h : RegionAt ctx rg cst rg.start r) : RegionExec ctx cst rg r

/-- A loop region (`LoopRegion.inline`, `dace/sdfg/state.py:3609-3747`, and
`control_flow.py:139-167`; `Decisions.md#D36`): initialization, then the
guard (or the body first when `inverted`). -/
inductive LoopExec : Ctx → CSt → LoopRegion → Res (Ctl × CSt) → Prop where
  | initErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hi : applyAssign? ctx cst l.init = .err e) : LoopExec ctx cst l (.err e)
  | guardFirst {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (hi : applyAssign? ctx cst l.init = .ok cst₁) (hv : l.inverted = false)
      (hg : LoopGuard ctx cst₁ l r) : LoopExec ctx cst l r
  | bodyFirst {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (hi : applyAssign? ctx cst l.init = .ok cst₁) (hv : l.inverted = true)
      (hb : LoopIter ctx cst₁ l r) : LoopExec ctx cst l r

/-- The loop guard: evaluate the condition, enter the body or exit. -/
inductive LoopGuard : Ctx → CSt → LoopRegion → Res (Ctl × CSt) → Prop where
  | condErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .err e) : LoopGuard ctx cst l (.err e)
  | exit {ctx : Ctx} {cst : CSt} {l : LoopRegion}
      (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .ok false) : LoopGuard ctx cst l (.ok (.normal, cst))
  | enter {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .ok true) (hb : LoopIter ctx cst l r) : LoopGuard ctx cst l r

/-- One execution of the loop body: `ret` propagates, `brk` exits the loop
normally, `normal`/`cont` proceed to the tail. -/
inductive LoopIter : Ctx → CSt → LoopRegion → Res (Ctl × CSt) → Prop where
  | bodyErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hb : RegionExec ctx cst l.body (.err e)) : LoopIter ctx cst l (.err e)
  | ret {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion}
      (hb : RegionExec ctx cst l.body (.ok (.ret, cst₁))) : LoopIter ctx cst l (.ok (.ret, cst₁))
  | brk {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion}
      (hb : RegionExec ctx cst l.body (.ok (.brk, cst₁))) : LoopIter ctx cst l (.ok (.normal, cst₁))
  | tail {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion} {ctl : Ctl} {r : Res (Ctl × CSt)}
      (hb : RegionExec ctx cst l.body (.ok (ctl, cst₁))) (hc : ctl.continuesLoop = true)
      (ht : LoopTail ctx cst₁ l r) : LoopIter ctx cst l r

/-- The loop tail after a body execution: update then guard, except for
inverted loops with `update_before_condition = false`, which test the
condition before updating (`control_flow.py:150-158`). -/
inductive LoopTail : Ctx → CSt → LoopRegion → Res (Ctl × CSt) → Prop where
  | updateErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = false) (hu : applyAssign? ctx cst l.update = .err e) :
      LoopTail ctx cst l (.err e)
  | updateGuard {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (hv : (l.inverted && !l.updateBeforeCond) = false) (hu : applyAssign? ctx cst l.update = .ok cst₁)
      (hg : LoopGuard ctx cst₁ l r) : LoopTail ctx cst l r
  | condFirstErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = true) (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .err e) :
      LoopTail ctx cst l (.err e)
  | condFirstExit {ctx : Ctx} {cst : CSt} {l : LoopRegion}
      (hv : (l.inverted && !l.updateBeforeCond) = true) (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .ok false) :
      LoopTail ctx cst l (.ok (.normal, cst))
  | condFirstUpdateErr {ctx : Ctx} {cst : CSt} {l : LoopRegion} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = true) (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .ok true)
      (hu : applyAssign? ctx cst l.update = .err e) : LoopTail ctx cst l (.err e)
  | condFirstNext {ctx : Ctx} {cst cst₁ : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
      (hv : (l.inverted && !l.updateBeforeCond) = true) (hc : ctx.evalBoolR cst.ρ cst.mem l.cond = .ok true)
      (hu : applyAssign? ctx cst l.update = .ok cst₁) (hb : LoopIter ctx cst₁ l r) : LoopTail ctx cst l r

/-- A conditional block: the first branch whose condition holds, in list
order; `none` is the else branch; no branch taken is a no-op. -/
inductive CondExec : Ctx → CSt → List (Option Expr × Region) → Res (Ctl × CSt) → Prop where
  | nil {ctx : Ctx} {cst : CSt} : CondExec ctx cst [] (.ok (.normal, cst))
  | els {ctx : Ctx} {cst : CSt} {rg : Region} {rest : List (Option Expr × Region)} {r : Res (Ctl × CSt)}
      (h : RegionExec ctx cst rg r) : CondExec ctx cst ((none, rg) :: rest) r
  | condErr {ctx : Ctx} {cst : CSt} {c : Expr} {rg : Region} {rest : List (Option Expr × Region)} {e : ErrReason}
      (hc : ctx.evalBoolR cst.ρ cst.mem c = .err e) : CondExec ctx cst ((some c, rg) :: rest) (.err e)
  | taken {ctx : Ctx} {cst : CSt} {c : Expr} {rg : Region} {rest : List (Option Expr × Region)} {r : Res (Ctl × CSt)}
      (hc : ctx.evalBoolR cst.ρ cst.mem c = .ok true) (h : RegionExec ctx cst rg r) :
      CondExec ctx cst ((some c, rg) :: rest) r
  | skipped {ctx : Ctx} {cst : CSt} {c : Expr} {rg : Region} {rest : List (Option Expr × Region)} {r : Res (Ctl × CSt)}
      (hc : ctx.evalBoolR cst.ρ cst.mem c = .ok false) (h : CondExec ctx cst rest r) :
      CondExec ctx cst ((some c, rg) :: rest) r

/-- An SDFG invocation: the top region from the valuation extended by the
constants; `ret` is normal termination, `brk`/`cont` escaping are malformed. -/
inductive SDFGExec : Ctx → Valuation → Mem → SDFG → Res (Valuation × Mem × Log) → Prop where
  | err {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG} {e : ErrReason}
      (h : RegionExec ctx { ρ := g.withConstants ρ, mem := m } g.top (.err e)) :
      SDFGExec ctx ρ m g (.err e)
  | escape {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG} {ctl : Ctl} {cst : CSt}
      (h : RegionExec ctx { ρ := g.withConstants ρ, mem := m } g.top (.ok (ctl, cst)))
      (hc : (ctl == .brk || ctl == .cont) = true) :
      SDFGExec ctx ρ m g (.err (.malformed "break/continue escaping the program"))
  | run {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG} {ctl : Ctl} {cst : CSt}
      (h : RegionExec ctx { ρ := g.withConstants ρ, mem := m } g.top (.ok (ctl, cst)))
      (hc : (ctl == .normal || ctl == .ret) = true) :
      SDFGExec ctx ρ m g (.ok (cst.ρ, cst.mem, cst.log))

end

/-- Program-level semantics: `g` run by interpretation `I` from the symbol
valuation `ρ₀` and memory `m₀` may produce outcome `r`. -/
def SDFG.Exec (I : Interp) (g : SDFG) (ρ₀ : Valuation) (m₀ : Mem) (r : Res Mem) : Prop :=
  ∃ res, SDFGExec { I, g } ρ₀ (g.startMem m₀) g res ∧ r = res.map (fun x => x.2.1)

/-! ## Sanity lemmas -/

/-- Sequential composition with the empty log is the identity. -/
theorem Log.seq_empty (g : Log) : g.seq Log.empty = g := by
  cases g; simp [Log.seq, Log.empty]

/-- Sequential composition of a list of logs, in order. -/
def Log.seqAll : List Log → Log
  | [] => Log.empty
  | g :: gs => g.seq (Log.seqAll gs)

theorem SeqExec_nil {ctx : Ctx} {ρ : Valuation} {s : State} {st : St} {r : Res (St × List (NodeId × Log))}
    (h : SeqExec ctx ρ s [] st r) : r = .ok (st, []) := by
  cases h; rfl

/-- The log of a sequential execution is the initial log followed by the
deltas of its trace, in order. -/
theorem SeqExec.log_eq {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} {st st₂ : St}
    {tr : List (NodeId × Log)} (h : SeqExec ctx ρ s ids st (.ok (st₂, tr))) :
    st₂.log = tr.foldl (fun g x => g.seq x.2) st.log := by
  induction ids generalizing st st₂ tr with
  | nil => cases h; rfl
  | cons id rest ih =>
    cases h with
    | cons _ _ hr =>
      have h' := ih hr
      rw [h']
      rfl

/-- The same, with the trace's deltas composed first: the log is the
initial log followed by `Log.seqAll` of the trace (by `Log.seq_assoc`), so
that, for an execution starting with an empty log, it is exactly the
sequential composition of the nodes' deltas. -/
theorem SeqExec.log_eq_seqAll {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} {st st₂ : St}
    {tr : List (NodeId × Log)} (h : SeqExec ctx ρ s ids st (.ok (st₂, tr))) :
    st₂.log = st.log.seq (Log.seqAll (tr.map (·.2))) := by
  rw [h.log_eq]
  clear h
  generalize st.log = L
  induction tr generalizing L with
  | nil => simp [Log.seqAll, Log.seq_empty]
  | cons x xs ih =>
    simp only [List.foldl_cons, List.map_cons, Log.seqAll]
    rw [ih, Log.seq_assoc]

theorem SeqExec.log_eq_seqAll_empty {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} {st st₂ : St}
    {tr : List (NodeId × Log)} (h : SeqExec ctx ρ s ids st.forNode (.ok (st₂, tr))) :
    st₂.log = Log.seqAll (tr.map (·.2)) := by
  rw [h.log_eq_seqAll]
  exact Log.empty_seq _

theorem IterExec_nil {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym}
    {priv : List Name} {pre : St} {r : Res (List Log)}
    (h : IterExec ctx ρ s entry params priv pre [] r) : r = .ok [] := by
  cases h; rfl

theorem CompsExec_nil {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {r : Res (List Log)}
    (h : CompsExec ctx ρ s m [] r) : r = .ok [] := by
  cases h; rfl

theorem CondExec_nil {ctx : Ctx} {cst : CSt} {r : Res (Ctl × CSt)}
    (h : CondExec ctx cst [] r) : r = .ok (.normal, cst) := by
  cases h; rfl


theorem Log.unionAll_nil : Log.unionAll [] = Log.empty := rfl

/-- Merging no logs leaves the memory unchanged. -/
theorem merge_nil (w : WcrSem) (m : Mem) : merge w m [] = .ok m := rfl

/-- A scope without iterations leaves the state unchanged. -/
theorem scopeFinish_nil (ctx : Ctx) (pre : St) : scopeFinish ctx pre [] = .ok pre := by
  unfold scopeFinish
  rw [merge_nil]
  show Res.ok { mem := pre.mem, log := pre.log.seq (Log.unionAll []), vals := pre.vals } = Res.ok pre
  rw [Log.unionAll_nil, Log.seq_empty]

/-- A state without components (no nodes) leaves memory and valuation
unchanged. -/
theorem StateExec_noComponents {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)}
    (hs : s.components = []) (h : StateExec ctx cst s r) : r = .ok (.normal, cst) := by
  cases h with
  | compsErr hc =>
    rw [hs] at hc
    exact absurd (CompsExec_nil hc) (by simp)
  | run hc =>
    rw [hs] at hc
    have hl := CompsExec_nil hc
    simp only [Res.ok_inj] at hl
    subst hl
    unfold stateFinish
    rw [merge_nil]
    show Res.ok (Ctl.normal, ({ ρ := cst.ρ, mem := cst.mem, log := cst.log.seq (Log.unionAll []) } : CSt))
      = Res.ok (Ctl.normal, cst)
    rw [Log.unionAll_nil, Log.seq_empty]

end Dace
