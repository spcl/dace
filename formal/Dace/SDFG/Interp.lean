-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics

/-!
# SDFG interpreter

A fuel-based executable mirror of the relational semantics
(`Dace.SDFG.Semantics.Exec`).  There is one total function per relation, with
the same arguments, the same helper functions and the same premises, so that
the soundness statement `run … = some r → Exec … r` can be proven constructor
by constructor.  `none` means the fuel ran out (or, in the situations the
relation leaves without any derivation, that no outcome exists:
`Decisions.md#D8`).

The relations are nondeterministic in three places; the interpreter resolves
each deterministically, as a refinement:

* **Scope body / component order** (`OrderedExec.run`, `Decisions.md#D43`):
  `State.topoOrder all` (Kahn's algorithm, ties broken by node order).  The
  computed order is re-checked with `State.isTopologicalOrder` (the premise
  of `OrderedExec.run`) so that soundness does not depend on the internals
  of `topoOrder`.  A cyclic body (self-loop edges included), or an `all` the
  check rejects (duplicate node ids), has no derivation in the relation and
  yields `none`.  The trace of the chosen order is checked for
  happens-before races (`State.hbFree`, `Decisions.md#D61`); the check does
  not depend on the order chosen except through the nodes' deltas.
* **Interstate edge** (`RegionAt.step`, `Decisions.md#D2`): the first true
  edge in `Region.outEdges` order, as the code generator does.
* **Consume processing element** (`ConsumeTasks.step`, `Decisions.md#D38`):
  `0` whenever `num_pes ≥ 1`; otherwise no task can run and the result is
  `none`.

Every function takes the fuel first and recurses structurally on it; every
recursive call passes the predecessor.  The result type is `Option (Res α)`.
-/

set_option autoImplicit false

namespace Dace

mutual

/-- Mirror of `NodeExec`. -/
def execNode (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) : Option (Res St) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match n.kind with
    | .access _ | .tasklet _ | .library _ | .mapExit | .consumeExit => some (execSimple ctx ρ s n st)
    | .mapEntry mp =>
      match mapSetup ctx ρ s n mp st with
      | .err e => some (.err e)
      | .ok (ρ', st₁, pts) =>
        match execIter fuel ctx ρ' s n.id mp.params (ctx.privateTransients s n.id) st₁ pts with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok logs) => some (scopeFinish ctx st₁ logs)
    | .consumeEntry c =>
      match consumeSetup ctx ρ s n c st with
      | .err e => some (.err e)
      | .ok (ρ', st₁, l, np) =>
        match execConsume fuel ctx ρ' s n.id c l np (ctx.privateTransients s n.id) st₁ (st₁.mem.queues l) with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok (logs, rest, pushed)) => some (consumeFinish ctx st₁ l logs rest pushed)
    | .nested inner mapping _ =>
      match nestedSetup ctx ρ s n inner mapping st with
      | .err e => some (.err e)
      | .ok (ctx', ρ', st₁, aliased) =>
        match execSDFG fuel ctx' ρ' st₁.mem inner with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok (_, m₂, g₂)) => some (nestedFinish ctx' ρ' s n aliased st₁ m₂ g₂)

/-- Mirror of `SeqExec`. -/
def execSeq (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (ids : List NodeId) (st : St) :
    Option (Res (St × List (NodeId × Log))) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match ids with
    | [] => some (.ok (st, []))
    | id :: rest =>
      match s.node? id with
      | none => some (.err (.malformed s!"unknown node {id}"))
      | some n =>
        match execNode fuel ctx ρ s n st.forNode with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok st') =>
          match execSeq fuel ctx ρ s rest (st.afterNode st') with
          | none => none
          | some (.err e) => some (.err e)
          | some (.ok (st₂, tr)) => some (.ok (st₂, (id, st'.log) :: tr))

/-- Mirror of `OrderedExec`; the topological order is `State.topoOrder all`,
and the trace of the sequential execution is checked with `State.hbFree`. -/
def execOrdered (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (all direct : List NodeId) (st : St) :
    Option (Res St) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match bindViews ctx ρ s direct st with
    | .err e => some (.err e)
    | .ok st₁ =>
      match s.topoOrder all with
      | none => none
      | some order =>
        if s.isTopologicalOrder all order then
          match execSeq fuel ctx ρ s (order.filter direct.contains) st₁ with
          | none => none
          | some (.err e) => some (.err e)
          | some (.ok (st₂, tr)) =>
            if s.hbFree tr then some (.ok st₂) else some (.err (.race (s.hbWitness tr)))
        else none

/-- Mirror of `IterExec`. -/
def execIter (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (entry : NodeId) (params : List Sym)
    (priv : List Name) (pre : St) (pts : List Idx) : Option (Res (List Log)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match pts with
    | [] => some (.ok [])
    | i :: rest =>
      match execOrdered fuel ctx (ρ.bindIdx params i) s (s.scopeNodes entry) (s.directChildren (some entry))
              (iterStart priv pre) with
      | none => none
      | some (.err e) => some (.err e)
      | some (.ok st) =>
        match execIter fuel ctx ρ s entry params priv pre rest with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok logs) => some (.ok (st.log.dropNames priv :: logs))

/-- Mirror of `ConsumeTasks`; the processing element is `0`. -/
def execConsume (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (entry : NodeId) (c : ConsumeNode) (l : Loc)
    (np : Int) (priv : List Name) (pre : St) (q : List Val) : Option (Res (List Log × List Val × List Val)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match consumeCond ctx ρ c pre.mem l q with
    | .err e => some (.err e)
    | .ok false => some (.ok ([], q, []))
    | .ok true =>
      if q.isEmpty then none
      else if (0 ≤ (0 : Int) && (0 : Int) < np) then
        match execOrdered fuel ctx (Valuation.set ρ c.peIndex (.int 0)) s (s.scopeNodes entry)
                (s.directChildren (some entry)) (taskStart ctx s entry c l priv pre q) with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok st) =>
          match execConsume fuel ctx ρ s entry c l np priv pre (q.drop c.chunksize) with
          | none => none
          | some (.err e) => some (.err e)
          | some (.ok (logs, rest, pushed)) =>
            some (.ok (taskLog st priv l :: logs, rest, taskPushed st priv l ++ pushed))
      else none

/-- Mirror of `CompsExec`. -/
def execComps (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (m : Mem) (comps : List (List NodeId)) :
    Option (Res (List Log)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match comps with
    | [] => some (.ok [])
    | comp :: rest =>
      match execOrdered fuel ctx ρ s comp (s.topLevelOf comp) { mem := m } with
      | none => none
      | some (.err e) => some (.err e)
      | some (.ok st) =>
        match execComps fuel ctx ρ s m rest with
        | none => none
        | some (.err e) => some (.err e)
        | some (.ok logs) => some (.ok (st.log :: logs))

/-- Mirror of `StateExec`. -/
def execState (fuel : Nat) (ctx : Ctx) (cst : CSt) (s : State) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match execComps fuel ctx cst.ρ s cst.mem s.components with
    | none => none
    | some (.err e) => some (.err e)
    | some (.ok logs) => some (stateFinish ctx cst logs)

/-- Mirror of `BlockExec`. -/
def execBlock (fuel : Nat) (ctx : Ctx) (cst : CSt) (b : Block) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match b with
    | .state s => execState fuel ctx cst s
    | .region rg => execRegion fuel ctx cst rg
    | .loop l => execLoop fuel ctx cst l
    | .cond _ brs => execCond fuel ctx cst brs
    | .breakBlock => some (.ok (.brk, cst))
    | .continueBlock => some (.ok (.cont, cst))
    | .returnBlock => some (.ok (.ret, cst))

/-- Mirror of `RegionAt`; the interstate edge taken is the first true one. -/
def regionAt (fuel : Nat) (ctx : Ctx) (rg : Region) (cst : CSt) (bid : BlockId) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match rg.block? bid with
    | none => some (.err (.malformed s!"block {bid} out of range"))
    | some b =>
      match execBlock fuel ctx cst b with
      | none => none
      | some (.err e) => some (.err e)
      | some (.ok (.brk, cst₁)) => some (.ok (.brk, cst₁))
      | some (.ok (.cont, cst₁)) => some (.ok (.cont, cst₁))
      | some (.ok (.ret, cst₁)) => some (.ok (.ret, cst₁))
      | some (.ok (.normal, cst₁)) =>
        match trueEdges ctx rg cst₁ bid with
        | .err e => some (.err e)
        | .ok [] => some (.ok (.normal, cst₁))
        | .ok [ie] =>
          match applyAssigns ctx cst₁ ie.assigns with
          | .err e => some (.err e)
          | .ok cst₂ => regionAt fuel ctx rg cst₂ ie.dst
        | .ok (_ :: _ :: _) => some (.err (.ambiguousTransition bid))

/-- Mirror of `RegionExec`. -/
def execRegion (fuel : Nat) (ctx : Ctx) (cst : CSt) (rg : Region) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 => regionAt fuel ctx rg cst rg.start

/-- Mirror of `LoopExec`. -/
def execLoop (fuel : Nat) (ctx : Ctx) (cst : CSt) (l : LoopRegion) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match applyAssign? ctx cst l.init with
    | .err e => some (.err e)
    | .ok cst₁ =>
      match l.inverted with
      | false => loopGuard fuel ctx cst₁ l
      | true => loopIter fuel ctx cst₁ l

/-- Mirror of `LoopGuard`. -/
def loopGuard (fuel : Nat) (ctx : Ctx) (cst : CSt) (l : LoopRegion) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match ctx.evalBoolR cst.ρ cst.mem l.cond with
    | .err e => some (.err e)
    | .ok false => some (.ok (.normal, cst))
    | .ok true => loopIter fuel ctx cst l

/-- Mirror of `LoopIter`. -/
def loopIter (fuel : Nat) (ctx : Ctx) (cst : CSt) (l : LoopRegion) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match execRegion fuel ctx cst l.body with
    | none => none
    | some (.err e) => some (.err e)
    | some (.ok (.ret, cst₁)) => some (.ok (.ret, cst₁))
    | some (.ok (.brk, cst₁)) => some (.ok (.normal, cst₁))
    | some (.ok (.normal, cst₁)) => loopTail fuel ctx cst₁ l
    | some (.ok (.cont, cst₁)) => loopTail fuel ctx cst₁ l

/-- Mirror of `LoopTail`. -/
def loopTail (fuel : Nat) (ctx : Ctx) (cst : CSt) (l : LoopRegion) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match (l.inverted && !l.updateBeforeCond) with
    | false =>
      match applyAssign? ctx cst l.update with
      | .err e => some (.err e)
      | .ok cst₁ => loopGuard fuel ctx cst₁ l
    | true =>
      match ctx.evalBoolR cst.ρ cst.mem l.cond with
      | .err e => some (.err e)
      | .ok false => some (.ok (.normal, cst))
      | .ok true =>
        match applyAssign? ctx cst l.update with
        | .err e => some (.err e)
        | .ok cst₁ => loopIter fuel ctx cst₁ l

/-- Mirror of `CondExec`. -/
def execCond (fuel : Nat) (ctx : Ctx) (cst : CSt) (brs : List (Option Expr × Region)) : Option (Res (Ctl × CSt)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match brs with
    | [] => some (.ok (.normal, cst))
    | (none, rg) :: _ => execRegion fuel ctx cst rg
    | (some c, rg) :: rest =>
      match ctx.evalBoolR cst.ρ cst.mem c with
      | .err e => some (.err e)
      | .ok true => execRegion fuel ctx cst rg
      | .ok false => execCond fuel ctx cst rest

/-- Mirror of `SDFGExec`. -/
def execSDFG (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (m : Mem) (g : SDFG) :
    Option (Res (Valuation × Mem × Log)) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match execRegion fuel ctx { ρ := g.withConstants ρ, mem := m } g.top with
    | none => none
    | some (.err e) => some (.err e)
    | some (.ok (.brk, _)) => some (.err (.malformed "break/continue escaping the program"))
    | some (.ok (.cont, _)) => some (.err (.malformed "break/continue escaping the program"))
    | some (.ok (.normal, cst)) => some (.ok (cst.ρ, cst.mem, cst.log))
    | some (.ok (.ret, cst)) => some (.ok (cst.ρ, cst.mem, cst.log))

end

/-- Program-level interpreter, mirroring `SDFG.Exec`: run `g` under `I` from
the symbol valuation `ρ₀` and memory `m₀` with the given fuel.  `none` =
fuel exhausted (or no outcome exists). -/
def SDFG.run (I : Interp) (g : SDFG) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) : Option (Res Mem) :=
  (execSDFG fuel { I, g } ρ₀ (g.startMem m₀) g).map fun res => res.map fun x => x.2.1

/-! ## Sanity lemmas -/

theorem execSeq_zero (ctx : Ctx) (ρ : Valuation) (s : State) (ids : List NodeId) (st : St) :
    execSeq 0 ctx ρ s ids st = none := by
  simp [execSeq]

theorem execSeq_nil (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (st : St) :
    execSeq (fuel + 1) ctx ρ s [] st = some (.ok (st, [])) := by
  simp [execSeq]

theorem execIter_nil (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (entry : NodeId) (params : List Sym)
    (priv : List Name) (pre : St) :
    execIter (fuel + 1) ctx ρ s entry params priv pre [] = some (.ok []) := by
  simp [execIter]

theorem execComps_nil (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (m : Mem) :
    execComps (fuel + 1) ctx ρ s m [] = some (.ok []) := by
  simp [execComps]

theorem execCond_nil (fuel : Nat) (ctx : Ctx) (cst : CSt) :
    execCond (fuel + 1) ctx cst [] = some (.ok (.normal, cst)) := by
  simp [execCond]

/-- The empty node list is executed as `SeqExec.nil` derives it. -/
theorem execSeq_nil_sound (fuel : Nat) (ctx : Ctx) (ρ : Valuation) (s : State) (st : St)
    {r : Res (St × List (NodeId × Log))}
    (h : execSeq (fuel + 1) ctx ρ s [] st = some r) : SeqExec ctx ρ s [] st r := by
  rw [execSeq_nil] at h
  cases h
  exact SeqExec.nil

end Dace
