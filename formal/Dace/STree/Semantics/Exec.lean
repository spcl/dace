-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Exec
import Dace.STree.Semantics.Leaf

/-!
# Schedule tree semantics: the execution relations

Big-step relational semantics of schedule trees as one mutual family of
inductive predicates, in the style of `Dace.SDFG.Semantics.Exec`: every
constructor is function-shaped (intermediate data computed by the functions
of `Dace.STree.Semantics.Leaf`, premises `f x = .ok y` / `f x = .err e`),
and the only nondeterminism is the processing element of a consume task
(`Decisions.md#D38`).  Children of a scope run **sequentially** in list
order (`#D20`); map iterations and consume tasks run from the pre-scope
memory and their logs are merged (`#D3`).

## Relations

| Relation | Meaning |
|----------|---------|
| `NodeExec tc sc n cst r` | node `n` with sibling context `sc`, `r : Res NodeOut` (next sibling context, control, state) |
| `ListExec tc full gb sc nodes cst r` | the remaining siblings `nodes` of the list `full` (`gb`: inside a `GBlock`), with label-indexed resumption of gotos |
| `IterExec tc params body priv pre pts r` | the map iterations `pts`, each from `pre`, yielding their logs |
| `ConsumeTasks tc ρ c d body l np priv pre q r` | the consume tasks on queue `q` of stream `d` |
| `LoopExec` / `LoopGuard` / `LoopIter` / `LoopTail` | a loop scope and its phases (`#D36`) |

Control signals (`TCtl`, `#D51`): `brk`/`cont` reach the innermost loop,
`ret` (an exit goto) ends the program, `goto l` unwinds scopes until a
sibling list resolves the label (`resumeAt`); control leaving a map
iteration or consume task is `malformed`.
-/

set_option autoImplicit false

namespace Dace.STree

mutual

/-- Execution of one node. -/
inductive NodeExec : TCtx → SibCtx → SNode → CSt → Res NodeOut → Prop where
  /-- Pending dynamic scope inputs reach a node that cannot take them (`Decisions.md#D50`). -/
  | pendingErr {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt}
      (hn : n.consumesPending = false) (hp : sc.pending.isEmpty = false) :
      NodeExec tc sc n cst (.err pendingError)
  /-- Leaves (`stepLeaf`): everything but scopes. -/
  | leaf {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut}
      (hp : (n.isDynScopeCopy || sc.pending.isEmpty) = true) (h : stepLeaf tc sc n cst = some r) :
      NodeExec tc sc n cst r
  /-- `NamedRegionScope`: transparent. -/
  | namedRegion {tc : TCtx} {sc : SibCtx} {label : String} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (h : ListExec tc cs false {} cs cst r) :
      NodeExec tc sc (.namedRegion label cs) cst (scopeOut r)
  /-- `GBlock`: a sibling list in which gotos may also jump backward. -/
  | gblock {tc : TCtx} {sc : SibCtx} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (h : ListExec tc cs true {} cs cst r) :
      NodeExec tc sc (.gblock cs) cst (scopeOut r)
  /-- `IfScope`: condition undefined. -/
  | ifCondErr {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {e : ErrReason}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .err e) :
      NodeExec tc sc (.ifScope c cs) cst (.err e)
  /-- `IfScope`: not taken, the chain opens untaken (`Decisions.md#D52`). -/
  | ifSkip {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok false) :
      NodeExec tc sc (.ifScope c cs) cst (.ok ({ chain := some false }, .normal, cst))
  /-- `IfScope`: taken, the chain opens taken. -/
  | ifTaken {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok true)
      (h : ListExec tc cs false {} cs cst r) :
      NodeExec tc sc (.ifScope c cs) cst (chainOut r)
  /-- `StateIfScope`: a conditional state transition; never opens a chain. -/
  | stateIfCondErr {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {e : ErrReason}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .err e) :
      NodeExec tc sc (.stateIfScope c cs) cst (.err e)
  | stateIfSkip {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok false) :
      NodeExec tc sc (.stateIfScope c cs) cst (.ok ({}, .normal, cst))
  | stateIfTaken {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (hc : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok true)
      (h : ListExec tc cs false {} cs cst r) :
      NodeExec tc sc (.stateIfScope c cs) cst (scopeOut r)
  /-- `ElifScope` without an open chain. -/
  | elifNoChain {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = none) :
      NodeExec tc sc (.elifScope c cs) cst (.err (.malformed "elif without a preceding if"))
  /-- `ElifScope` after a taken branch: skipped, chain stays taken. -/
  | elifAfterTaken {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some true) :
      NodeExec tc sc (.elifScope c cs) cst (.ok ({ chain := some true }, .normal, cst))
  | elifCondErr {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {e : ErrReason}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some false)
      (he : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .err e) :
      NodeExec tc sc (.elifScope c cs) cst (.err e)
  | elifSkip {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some false)
      (he : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok false) :
      NodeExec tc sc (.elifScope c cs) cst (.ok ({ chain := some false }, .normal, cst))
  | elifTaken {tc : TCtx} {sc : SibCtx} {c : PyExpr} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some false)
      (he : tc.ctx.evalBoolR cst.ρ cst.mem c.expr = .ok true) (h : ListExec tc cs false {} cs cst r) :
      NodeExec tc sc (.elifScope c cs) cst (chainOut r)
  | elseNoChain {tc : TCtx} {sc : SibCtx} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = none) :
      NodeExec tc sc (.elseScope cs) cst (.err (.malformed "else without a preceding if"))
  | elseAfterTaken {tc : TCtx} {sc : SibCtx} {cs : List SNode} {cst : CSt}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some true) :
      NodeExec tc sc (.elseScope cs) cst (.ok ({}, .normal, cst))
  | elseTaken {tc : TCtx} {sc : SibCtx} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (hc : sc.chain = some false) (h : ListExec tc cs false {} cs cst r) :
      NodeExec tc sc (.elseScope cs) cst (scopeOut r)
  /-- Loop scopes of all four classes (`SNode.loop?`). -/
  | loop {tc : TCtx} {sc : SibCtx} {n : SNode} {l : LoopMeta} {cs : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hp : sc.pending.isEmpty = true) (hl : n.loop? = some (l, cs)) (h : LoopExec tc l cs cst r) :
      NodeExec tc sc n cst (scopeOut r)
  /-- `MapScope`: dynamic inputs or iteration space undefined. -/
  | mapSetupErr {tc : TCtx} {sc : SibCtx} {mp : MapNode} {cs : List SNode} {cst : CSt} {e : ErrReason}
      (hs : mapSetupT tc sc mp cst = .err e) :
      NodeExec tc sc (.mapScope mp cs) cst (.err e)
  /-- `MapScope`: an iteration fails. -/
  | mapIterErr {tc : TCtx} {sc : SibCtx} {mp : MapNode} {cs : List SNode} {cst : CSt} {ρ' : Valuation}
      {pts : List Idx} {e : ErrReason}
      (hs : mapSetupT tc sc mp cst = .ok (ρ', pts))
      (hi : IterExec tc mp.params cs (tc.privateTransients cs) { cst with ρ := ρ' } pts (.err e)) :
      NodeExec tc sc (.mapScope mp cs) cst (.err e)
  /-- `MapScope`: all iterations run from the pre-map state, their logs are
      merged (`Decisions.md#D3`, `#D54`); the dynamic inputs are bound in the
      body only (`#D12`). -/
  | mapScope {tc : TCtx} {sc : SibCtx} {mp : MapNode} {cs : List SNode} {cst : CSt} {ρ' : Valuation}
      {pts : List Idx} {logs : List Log}
      (hs : mapSetupT tc sc mp cst = .ok (ρ', pts))
      (hi : IterExec tc mp.params cs (tc.privateTransients cs) { cst with ρ := ρ' } pts (.ok logs)) :
      NodeExec tc sc (.mapScope mp cs) cst (dataflowOut (scopeFinishT tc cst logs))
  /-- `ConsumeScope`: set-up fails. -/
  | consumeSetupErr {tc : TCtx} {sc : SibCtx} {c : ConsumeNode} {cs : List SNode} {cst : CSt} {e : ErrReason}
      (hs : consumeSetupT tc sc c cst = .err e) :
      NodeExec tc sc (.consumeScope c cs) cst (.err e)
  /-- `ConsumeScope`: a task fails. -/
  | consumeTasksErr {tc : TCtx} {sc : SibCtx} {c : ConsumeNode} {cs : List SNode} {cst : CSt} {ρ' : Valuation}
      {d : Name} {l : Loc} {np : Int} {e : ErrReason}
      (hs : consumeSetupT tc sc c cst = .ok (ρ', d, l, np))
      (ht : ConsumeTasks tc ρ' c d cs l np (tc.privateTransients cs) cst (cst.mem.queues l) (.err e)) :
      NodeExec tc sc (.consumeScope c cs) cst (.err e)
  /-- `ConsumeScope` (`Decisions.md#D38`, `#D50`): the tasks consume the
      stream given by the `IN_stream` input. -/
  | consumeScope {tc : TCtx} {sc : SibCtx} {c : ConsumeNode} {cs : List SNode} {cst : CSt} {ρ' : Valuation}
      {d : Name} {l : Loc} {np : Int} {logs : List Log} {rest pushed : List Val}
      (hs : consumeSetupT tc sc c cst = .ok (ρ', d, l, np))
      (ht : ConsumeTasks tc ρ' c d cs l np (tc.privateTransients cs) cst (cst.mem.queues l)
              (.ok (logs, rest, pushed))) :
      NodeExec tc sc (.consumeScope c cs) cst (dataflowOut (consumeFinishT tc cst l logs rest pushed))

/-- Sequential execution of the remaining siblings `nodes` of the list
`full`; `gb` marks the children of a `GBlock`. -/
inductive ListExec : TCtx → List SNode → Bool → SibCtx → List SNode → CSt → Res (TCtl × CSt) → Prop where
  | nil {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cst : CSt}
      (hp : sc.pending.isEmpty = true) :
      ListExec tc full gb sc [] cst (.ok (.normal, cst))
  /-- Dynamic scope inputs left over at the end of a list (`Decisions.md#D50`). -/
  | nilPending {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cst : CSt}
      (hp : sc.pending.isEmpty = false) :
      ListExec tc full gb sc [] cst (.err pendingError)
  | headErr {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {n : SNode} {rest : List SNode} {cst : CSt}
      {e : ErrReason}
      (h : NodeExec tc sc n cst (.err e)) :
      ListExec tc full gb sc (n :: rest) cst (.err e)
  | step {tc : TCtx} {full : List SNode} {gb : Bool} {sc sc' : SibCtx} {n : SNode} {rest : List SNode} {cst cst' : CSt}
      {r : Res (TCtl × CSt)}
      (h : NodeExec tc sc n cst (.ok (sc', .normal, cst'))) (hr : ListExec tc full gb sc' rest cst' r) :
      ListExec tc full gb sc (n :: rest) cst r
  /-- `break`, `continue` and `return` end the list. -/
  | jump {tc : TCtx} {full : List SNode} {gb : Bool} {sc sc' : SibCtx} {n : SNode} {rest : List SNode} {cst cst' : CSt}
      {ctl : TCtl}
      (h : NodeExec tc sc n cst (.ok (sc', ctl, cst'))) (hc : ctl.isJump = true) :
      ListExec tc full gb sc (n :: rest) cst (.ok (ctl, cst'))
  /-- A goto whose label this list resolves resumes after the label
      (`Decisions.md#D51`). -/
  | gotoResume {tc : TCtx} {full : List SNode} {gb : Bool} {sc sc' : SibCtx} {n : SNode} {rest suffix : List SNode}
      {cst cst' : CSt} {l : String} {r : Res (TCtl × CSt)}
      (h : NodeExec tc sc n cst (.ok (sc', .goto l, cst'))) (hl : resumeAt gb full rest l = some suffix)
      (hr : ListExec tc full gb {} suffix cst' r) :
      ListExec tc full gb sc (n :: rest) cst r
  /-- A goto this list cannot resolve escapes to the enclosing scope. -/
  | gotoEscape {tc : TCtx} {full : List SNode} {gb : Bool} {sc sc' : SibCtx} {n : SNode} {rest : List SNode}
      {cst cst' : CSt} {l : String}
      (h : NodeExec tc sc n cst (.ok (sc', .goto l, cst'))) (hl : resumeAt gb full rest l = none) :
      ListExec tc full gb sc (n :: rest) cst (.ok (.goto l, cst'))

/-- The iterations of a map, each from the pre-map state `pre` with the
parameters bound, yielding their logs without the private transients. -/
inductive IterExec : TCtx → List Sym → List SNode → List Name → CSt → List Idx → Res (List Log) → Prop where
  | nil {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} :
      IterExec tc params body priv pre [] (.ok [])
  | headErr {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {i : Idx}
      {rest : List Idx} {res : Res (TCtl × CSt)} {e : ErrReason}
      (hb : ListExec tc body false {} body (iterStartT priv pre (pre.ρ.bindIdx params i)) res)
      (he : iterLog priv res = .err e) :
      IterExec tc params body priv pre (i :: rest) (.err e)
  | tailErr {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {i : Idx}
      {rest : List Idx} {res : Res (TCtl × CSt)} {g : Log} {e : ErrReason}
      (hb : ListExec tc body false {} body (iterStartT priv pre (pre.ρ.bindIdx params i)) res)
      (hl : iterLog priv res = .ok g) (hr : IterExec tc params body priv pre rest (.err e)) :
      IterExec tc params body priv pre (i :: rest) (.err e)
  | cons {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {i : Idx}
      {rest : List Idx} {res : Res (TCtl × CSt)} {g : Log} {logs : List Log}
      (hb : ListExec tc body false {} body (iterStartT priv pre (pre.ρ.bindIdx params i)) res)
      (hl : iterLog priv res = .ok g) (hr : IterExec tc params body priv pre rest (.ok logs)) :
      IterExec tc params body priv pre (i :: rest) (.ok (g :: logs))

/-- The tasks of a consume scope on the queue `q` of stream `d`
(`Decisions.md#D38`): task `k` sees the queue left by tasks `< k`, pops up
to `chunksize` elements, runs the body (which reads the chunk under the
stream's name) with a nondeterministic processing element, and the scope
stops when the quiescence condition fails. -/
inductive ConsumeTasks : TCtx → Valuation → ConsumeNode → Name → List SNode → Loc → Int → List Name → CSt →
    List Val → Res (List Log × List Val × List Val) → Prop where
  | condErr {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : CSt} {q : List Val} {e : ErrReason}
      (hc : consumeCond tc.ctx ρ c pre.mem l q = .err e) :
      ConsumeTasks tc ρ c d body l np priv pre q (.err e)
  | stop {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : CSt} {q : List Val}
      (hc : consumeCond tc.ctx ρ c pre.mem l q = .ok false) :
      ConsumeTasks tc ρ c d body l np priv pre q (.ok ([], q, []))
  | bodyErr {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : CSt} {q : List Val} {p : Int} {res : Res (TCtl × CSt)} {e : ErrReason}
      (hc : consumeCond tc.ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : ListExec (tc.consuming d) body false {} body
              (taskStartT tc d c l priv pre (Valuation.set ρ c.peIndex (.int p)) q) res)
      (he : taskOutcome priv l res = .err e) :
      ConsumeTasks tc ρ c d body l np priv pre q (.err e)
  | restErr {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : CSt} {q : List Val} {p : Int} {res : Res (TCtl × CSt)} {g : Log}
      {pushed : List Val} {e : ErrReason}
      (hc : consumeCond tc.ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : ListExec (tc.consuming d) body false {} body
              (taskStartT tc d c l priv pre (Valuation.set ρ c.peIndex (.int p)) q) res)
      (ho : taskOutcome priv l res = .ok (g, pushed))
      (hr : ConsumeTasks tc ρ c d body l np priv pre (q.drop c.chunksize) (.err e)) :
      ConsumeTasks tc ρ c d body l np priv pre q (.err e)
  | step {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
      {priv : List Name} {pre : CSt} {q : List Val} {p : Int} {res : Res (TCtl × CSt)} {g : Log}
      {pushed : List Val} {logs : List Log} {rest pushed' : List Val}
      (hc : consumeCond tc.ctx ρ c pre.mem l q = .ok true) (hq : q.isEmpty = false)
      (hp : (0 ≤ p && p < np) = true)
      (hb : ListExec (tc.consuming d) body false {} body
              (taskStartT tc d c l priv pre (Valuation.set ρ c.peIndex (.int p)) q) res)
      (ho : taskOutcome priv l res = .ok (g, pushed))
      (hr : ConsumeTasks tc ρ c d body l np priv pre (q.drop c.chunksize) (.ok (logs, rest, pushed'))) :
      ConsumeTasks tc ρ c d body l np priv pre q (.ok (g :: logs, rest, pushed ++ pushed'))

/-- A loop scope (`Decisions.md#D36`): initialization, then the guard (or
the body first when `inverted`). -/
inductive LoopExec : TCtx → LoopMeta → List SNode → CSt → Res (TCtl × CSt) → Prop where
  | initErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hi : applyAssign? tc.ctx cst l.initAssign = .err e) : LoopExec tc l body cst (.err e)
  | guardFirst {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {r : Res (TCtl × CSt)}
      (hi : applyAssign? tc.ctx cst l.initAssign = .ok cst₁) (hv : l.inverted = false)
      (hg : LoopGuard tc l body cst₁ r) : LoopExec tc l body cst r
  | bodyFirst {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {r : Res (TCtl × CSt)}
      (hi : applyAssign? tc.ctx cst l.initAssign = .ok cst₁) (hv : l.inverted = true)
      (hb : LoopIter tc l body cst₁ r) : LoopExec tc l body cst r

/-- The loop guard: evaluate the condition, enter the body or exit. -/
inductive LoopGuard : TCtx → LoopMeta → List SNode → CSt → Res (TCtl × CSt) → Prop where
  | condErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .err e) : LoopGuard tc l body cst (.err e)
  | exit {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .ok false) : LoopGuard tc l body cst (.ok (.normal, cst))
  | enter {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .ok true) (hb : LoopIter tc l body cst r) :
      LoopGuard tc l body cst r

/-- One execution of the loop body: `ret` and `goto` propagate, `brk` exits
the loop normally, `normal`/`cont` proceed to the tail. -/
inductive LoopIter : TCtx → LoopMeta → List SNode → CSt → Res (TCtl × CSt) → Prop where
  | bodyErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hb : ListExec tc body false {} body cst (.err e)) : LoopIter tc l body cst (.err e)
  | ret {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt}
      (hb : ListExec tc body false {} body cst (.ok (.ret, cst₁))) : LoopIter tc l body cst (.ok (.ret, cst₁))
  | brk {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt}
      (hb : ListExec tc body false {} body cst (.ok (.brk, cst₁))) : LoopIter tc l body cst (.ok (.normal, cst₁))
  | gotoOut {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {lbl : String}
      (hb : ListExec tc body false {} body cst (.ok (.goto lbl, cst₁))) :
      LoopIter tc l body cst (.ok (.goto lbl, cst₁))
  | tail {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {ctl : TCtl} {r : Res (TCtl × CSt)}
      (hb : ListExec tc body false {} body cst (.ok (ctl, cst₁))) (hc : ctl.continuesLoop = true)
      (ht : LoopTail tc l body cst₁ r) : LoopIter tc l body cst r

/-- The loop tail after a body execution: update then guard, except for
inverted loops with `update_before_condition = false`, which test the
condition before updating (`Decisions.md#D36`). -/
inductive LoopTail : TCtx → LoopMeta → List SNode → CSt → Res (TCtl × CSt) → Prop where
  | updateErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = false) (hu : applyAssign? tc.ctx cst l.updateAssign = .err e) :
      LoopTail tc l body cst (.err e)
  | updateGuard {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {r : Res (TCtl × CSt)}
      (hv : (l.inverted && !l.updateBeforeCond) = false) (hu : applyAssign? tc.ctx cst l.updateAssign = .ok cst₁)
      (hg : LoopGuard tc l body cst₁ r) : LoopTail tc l body cst r
  | condFirstErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = true)
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .err e) :
      LoopTail tc l body cst (.err e)
  | condFirstExit {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
      (hv : (l.inverted && !l.updateBeforeCond) = true)
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .ok false) :
      LoopTail tc l body cst (.ok (.normal, cst))
  | condFirstUpdateErr {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {e : ErrReason}
      (hv : (l.inverted && !l.updateBeforeCond) = true)
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .ok true)
      (hu : applyAssign? tc.ctx cst l.updateAssign = .err e) : LoopTail tc l body cst (.err e)
  | condFirstNext {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst cst₁ : CSt} {r : Res (TCtl × CSt)}
      (hv : (l.inverted && !l.updateBeforeCond) = true)
      (hc : tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr = .ok true)
      (hu : applyAssign? tc.ctx cst l.updateAssign = .ok cst₁) (hb : LoopIter tc l body cst₁ r) :
      LoopTail tc l body cst r

end

/-- Program-level semantics: the tree `t` run by interpretation `I` from the
symbol valuation `ρ₀` and memory `m₀` may produce outcome `r` (the same
observable as `SDFG.Exec`). -/
def Exec (I : Interp) (t : STree) (ρ₀ : Valuation) (m₀ : Mem) (r : Res Mem) : Prop :=
  ∃ res, ListExec (TCtx.ofTree I t) t.children false {} t.children (rootStart t ρ₀ m₀) res ∧ r = programOutcome res

/-! ## Sanity lemmas -/

theorem ListExec_nil {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cst : CSt} {r : Res (TCtl × CSt)}
    (hp : sc.pending.isEmpty = true) (h : ListExec tc full gb sc [] cst r) : r = .ok (.normal, cst) := by
  cases h with
  | nil => rfl
  | nilPending hp' => rw [hp] at hp'; cases hp'

theorem IterExec_nil {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt}
    {r : Res (List Log)} (h : IterExec tc params body priv pre [] r) : r = .ok [] := by
  cases h; rfl

/-- The dataflow view round-trips. -/
theorem withSt_st_eq (cst : CSt) : cst.withSt cst.st = cst := CSt.withSt_st cst

/-- A map over no points leaves the state unchanged. -/
theorem scopeFinishT_nil (tc : TCtx) (pre : CSt) : scopeFinishT tc pre [] = .ok pre := by
  unfold scopeFinishT
  rw [scopeFinish_nil]
  show Res.ok (pre.withSt pre.st) = Res.ok pre
  rw [CSt.withSt_st]

/-- Scopes are not leaves. -/
theorem stepLeaf_mapScope (tc : TCtx) (sc : SibCtx) (mp : MapNode) (cs : List SNode) (cst : CSt) :
    stepLeaf tc sc (.mapScope mp cs) cst = none := rfl

/-- A map scope whose iteration space is empty leaves memory, valuation and
log unchanged (and consumes the pending dynamic inputs). -/
theorem NodeExec_mapScope_noPoints {tc : TCtx} {sc : SibCtx} {mp : MapNode} {cs : List SNode} {cst : CSt}
    {ρ' : Valuation} {r : Res NodeOut}
    (hs : mapSetupT tc sc mp cst = .ok (ρ', [])) (h : NodeExec tc sc (.mapScope mp cs) cst r) :
    r = .ok ({}, .normal, cst) := by
  cases h with
  | pendingErr hn _ => simp [SNode.consumesPending, SNode.isDataflowScope] at hn
  | leaf _ hl => rw [stepLeaf_mapScope] at hl; cases hl
  | loop _ hl _ => simp [SNode.loop?] at hl
  | mapSetupErr hs' => rw [hs] at hs'; cases hs'
  | mapIterErr hs' hi =>
    rw [hs] at hs'
    cases hs'
    exact absurd (IterExec_nil hi) (by simp)
  | mapScope hs' hi =>
    rw [hs] at hs'
    cases hs'
    have hl := IterExec_nil hi
    simp only [Res.ok_inj] at hl
    subst hl
    rw [scopeFinishT_nil]
    rfl

end Dace.STree
