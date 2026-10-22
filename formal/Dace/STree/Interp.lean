-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics

/-!
# Schedule tree interpreter

A fuel-based executable mirror of the relational semantics of schedule trees
(`Dace.STree.Semantics.Exec`), in the style of `Dace.SDFG.Interp`: one total
function per relation, with the same arguments, the same helper functions
(`Dace.STree.Semantics.Leaf`) and the same premises, so that the soundness
statement `run … = some r → Exec … r` (`Dace.STree.Soundness`) can be proven
constructor by constructor.  `none` means that the fuel ran out or that the
relation has no derivation (a consume scope whose quiescence condition holds
on an empty queue, `Decisions.md#D8`).

| Function | Relation |
|----------|----------|
| `execNode` | `NodeExec` |
| `execList` | `ListExec` |
| `execIter` | `IterExec` |
| `execConsume` | `ConsumeTasks` |
| `execLoop` / `loopGuard` / `loopIter` / `loopTail` | `LoopExec` / `LoopGuard` / `LoopIter` / `LoopTail` |
| `STree.run` | `STree.Exec` |

## Determinism

The tree relations are deterministic except for the processing element of a
consume task (`Decisions.md#D38`); as in the SDFG interpreter it is `0`
whenever `num_pes ≥ 1`, and otherwise no task can run and the result is
`none`.  In particular the two node-level rules that do not inspect the node
class, `NodeExec.pendingErr` (pending dynamic scope inputs reach a node that
cannot take them, `#D50`) and `NodeExec.leaf`, are disjoint: `pendingErr`
needs `consumesPending = false` and a non-empty pending set, `leaf` needs the
node to be a `DynScopeCopyNode` (which consumes pending inputs) or the pending
set to be empty.  `execNode` therefore tests
`n.consumesPending || sc.pending.isEmpty` first, exactly as the premises
state, and dispatches on the node class only when it holds.

Every function takes the fuel first and recurses structurally on it; every
recursive call passes the predecessor.  The result type is `Option (Res α)`.
-/

set_option autoImplicit false

namespace Dace.STree

mutual

/-- Mirror of `NodeExec`. -/
def execNode (fuel : Nat) (tc : TCtx) (sc : SibCtx) (n : SNode) (cst : CSt) : Option (Res NodeOut) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match n.consumesPending || sc.pending.isEmpty with
    | false => some (.err pendingError)
    | true =>
      match n.loop? with
      | some (l, cs) =>
        match execLoop fuel tc l cs cst with
        | none => none
        | some r => some (scopeOut r)
      | none =>
        match n with
        | .namedRegion _ cs =>
          match execList fuel tc cs false {} cs cst with
          | none => none
          | some r => some (scopeOut r)
        | .gblock cs =>
          match execList fuel tc cs true {} cs cst with
          | none => none
          | some r => some (scopeOut r)
        | .ifScope c cs =>
          match tc.ctx.evalBoolR cst.ρ cst.mem c.expr with
          | .err e => some (.err e)
          | .ok false => some (.ok ({ chain := some false }, .normal, cst))
          | .ok true =>
            match execList fuel tc cs false {} cs cst with
            | none => none
            | some r => some (chainOut r)
        | .stateIfScope c cs =>
          match tc.ctx.evalBoolR cst.ρ cst.mem c.expr with
          | .err e => some (.err e)
          | .ok false => some (.ok ({}, .normal, cst))
          | .ok true =>
            match execList fuel tc cs false {} cs cst with
            | none => none
            | some r => some (scopeOut r)
        | .elifScope c cs =>
          match sc.chain with
          | none => some (.err (.malformed "elif without a preceding if"))
          | some true => some (.ok ({ chain := some true }, .normal, cst))
          | some false =>
            match tc.ctx.evalBoolR cst.ρ cst.mem c.expr with
            | .err e => some (.err e)
            | .ok false => some (.ok ({ chain := some false }, .normal, cst))
            | .ok true =>
              match execList fuel tc cs false {} cs cst with
              | none => none
              | some r => some (chainOut r)
        | .elseScope cs =>
          match sc.chain with
          | none => some (.err (.malformed "else without a preceding if"))
          | some true => some (.ok ({}, .normal, cst))
          | some false =>
            match execList fuel tc cs false {} cs cst with
            | none => none
            | some r => some (scopeOut r)
        | .mapScope mp cs =>
          match mapSetupT tc sc mp cst with
          | .err e => some (.err e)
          | .ok (ρ', pts) =>
            match execIter fuel tc mp.params cs (tc.privateTransients cs) { cst with ρ := ρ' } pts with
            | none => none
            | some (.err e) => some (.err e)
            | some (.ok logs) => some (dataflowOut (scopeFinishT tc cst logs))
        | .consumeScope c cs =>
          match consumeSetupT tc sc c cst with
          | .err e => some (.err e)
          | .ok (ρ', d, l, np) =>
            match execConsume fuel tc ρ' c d cs l np (tc.privateTransients cs) cst (cst.mem.queues l) with
            | none => none
            | some (.err e) => some (.err e)
            | some (.ok (logs, rest, pushed)) => some (dataflowOut (consumeFinishT tc cst l logs rest pushed))
        | n => stepLeaf tc sc n cst

/-- Mirror of `ListExec`. -/
def execList (fuel : Nat) (tc : TCtx) (full : List SNode) (gb : Bool) (sc : SibCtx) (nodes : List SNode)
    (cst : CSt) : Option (Res (TCtl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match nodes with
    | [] =>
      match sc.pending.isEmpty with
      | true => some (.ok (.normal, cst))
      | false => some (.err pendingError)
    | n :: rest =>
      match execNode fuel tc sc n cst with
      | none => none
      | some (.err e) => some (.err e)
      | some (.ok (sc', .normal, cst')) => execList fuel tc full gb sc' rest cst'
      | some (.ok (_, .brk, cst')) => some (.ok (.brk, cst'))
      | some (.ok (_, .cont, cst')) => some (.ok (.cont, cst'))
      | some (.ok (_, .ret, cst')) => some (.ok (.ret, cst'))
      | some (.ok (_, .goto l, cst')) =>
        match resumeAt gb full rest l with
        | some suffix => execList fuel tc full gb {} suffix cst'
        | none => some (.ok (.goto l, cst'))

/-- Mirror of `IterExec`. -/
def execIter (fuel : Nat) (tc : TCtx) (params : List Sym) (body : List SNode) (priv : List Name) (pre : CSt)
    (pts : List Idx) : Option (Res (List Log)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match pts with
    | [] => some (.ok [])
    | i :: rest =>
      match execList fuel tc body false {} body (iterStartT pre (pre.ρ.bindIdx params i)) with
      | none => none
      | some res =>
        match iterLog priv res with
        | .err e => some (.err e)
        | .ok g =>
          match execIter fuel tc params body priv pre rest with
          | none => none
          | some (.err e) => some (.err e)
          | some (.ok logs) => some (.ok (g :: logs))

/-- Mirror of `ConsumeTasks`; the processing element is `0`. -/
def execConsume (fuel : Nat) (tc : TCtx) (ρ : Valuation) (c : ConsumeNode) (d : Name) (body : List SNode)
    (l : Loc) (np : Int) (priv : List Name) (pre : CSt) (q : List Val) :
    Option (Res (List Log × List Val × List Val)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match consumeCond tc.ctx ρ c pre.mem l q with
    | .err e => some (.err e)
    | .ok false => some (.ok ([], q, []))
    | .ok true =>
      match q.isEmpty with
      | true => none
      | false =>
        match (0 ≤ (0 : Int) && (0 : Int) < np) with
        | true =>
          match execList fuel (tc.consuming d) body false {} body
                  (taskStartT tc d c l pre (Valuation.set ρ c.peIndex (.int 0)) q) with
          | none => none
          | some res =>
            match taskOutcome priv l res with
            | .err e => some (.err e)
            | .ok (g, pushed) =>
              match execConsume fuel tc ρ c d body l np priv pre (q.drop c.chunksize) with
              | none => none
              | some (.err e) => some (.err e)
              | some (.ok (logs, rest, pushed')) => some (.ok (g :: logs, rest, pushed ++ pushed'))
        | false => none

/-- Mirror of `LoopExec`. -/
def execLoop (fuel : Nat) (tc : TCtx) (l : LoopMeta) (body : List SNode) (cst : CSt) : Option (Res (TCtl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match applyAssign? tc.ctx cst l.initAssign with
    | .err e => some (.err e)
    | .ok cst₁ =>
      match l.inverted with
      | false => loopGuard fuel tc l body cst₁
      | true => loopIter fuel tc l body cst₁

/-- Mirror of `LoopGuard`. -/
def loopGuard (fuel : Nat) (tc : TCtx) (l : LoopMeta) (body : List SNode) (cst : CSt) : Option (Res (TCtl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr with
    | .err e => some (.err e)
    | .ok false => some (.ok (.normal, cst))
    | .ok true => loopIter fuel tc l body cst

/-- Mirror of `LoopIter`. -/
def loopIter (fuel : Nat) (tc : TCtx) (l : LoopMeta) (body : List SNode) (cst : CSt) : Option (Res (TCtl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match execList fuel tc body false {} body cst with
    | none => none
    | some (.err e) => some (.err e)
    | some (.ok (.ret, cst₁)) => some (.ok (.ret, cst₁))
    | some (.ok (.brk, cst₁)) => some (.ok (.normal, cst₁))
    | some (.ok (.goto lbl, cst₁)) => some (.ok (.goto lbl, cst₁))
    | some (.ok (.normal, cst₁)) => loopTail fuel tc l body cst₁
    | some (.ok (.cont, cst₁)) => loopTail fuel tc l body cst₁

/-- Mirror of `LoopTail`. -/
def loopTail (fuel : Nat) (tc : TCtx) (l : LoopMeta) (body : List SNode) (cst : CSt) : Option (Res (TCtl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match (l.inverted && !l.updateBeforeCond) with
    | false =>
      match applyAssign? tc.ctx cst l.updateAssign with
      | .err e => some (.err e)
      | .ok cst₁ => loopGuard fuel tc l body cst₁
    | true =>
      match tc.ctx.evalBoolR cst.ρ cst.mem l.cond.expr with
      | .err e => some (.err e)
      | .ok false => some (.ok (.normal, cst))
      | .ok true =>
        match applyAssign? tc.ctx cst l.updateAssign with
        | .err e => some (.err e)
        | .ok cst₁ => loopIter fuel tc l body cst₁

end

/-- Program-level interpreter, mirroring `STree.Exec`: run `t` under `I` from
the symbol valuation `ρ₀` and memory `m₀` with the given fuel.  `none` = fuel
exhausted (or no outcome exists). -/
def run (I : Interp) (t : STree) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) : Option (Res Mem) :=
  (execList fuel (TCtx.ofTree I t) t.children false {} t.children (rootStart t ρ₀ m₀)).map programOutcome

/-! ## Sanity lemmas -/

theorem execList_zero (tc : TCtx) (full : List SNode) (gb : Bool) (sc : SibCtx) (nodes : List SNode) (cst : CSt) :
    execList 0 tc full gb sc nodes cst = none := by
  simp [execList]

theorem execList_nil (fuel : Nat) (tc : TCtx) (full : List SNode) (gb : Bool) (sc : SibCtx) (cst : CSt)
    (hp : sc.pending.isEmpty = true) : execList (fuel + 1) tc full gb sc [] cst = some (.ok (.normal, cst)) := by
  simp [execList, hp]

theorem execIter_nil (fuel : Nat) (tc : TCtx) (params : List Sym) (body : List SNode) (priv : List Name)
    (pre : CSt) : execIter (fuel + 1) tc params body priv pre [] = some (.ok []) := by
  simp [execIter]

end Dace.STree
