-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics
import Dace.SDFG.LogShift

/-!
# Logs are write-only (schedule trees)

The schedule-tree analogue of `Dace.SDFG.LogShift`: no execution relation
of the tree semantics reads the log of the control-flow state it starts
from.  With `CSt.shift L cst` (prefix `L` to the log):

* `NodeExec` (`Res NodeOut`), `ListExec`, `LoopExec`, `LoopGuard`,
  `LoopIter`, `LoopTail` (`Res (TCtl × CSt)`):
  `R … (cst.shift L) r ↔ ∃ r₀, R … cst r₀ ∧ r = r₀.map (shift L)`
  (`STree.*.shift_iff`; `STree.NodeExec.log_iff`, `STree.ListExec.log_iff`
  for `{cst with log := L}` versus `{cst with log := .empty}`);
* `IterExec` and `ConsumeTasks` do not depend on the log of the pre-scope
  state at all (`STree.*.shift_iff`).

The leaf functions of `Dace.STree.Semantics.Leaf` commute with the shift
(through the SDFG leaf lemmas); the relations go by mutual recursion on the
derivation.  Corollaries (`STree.*.rel`): runs from `CSt.Rel`-related start
states have related outcomes, in both directions.
-/

set_option autoImplicit false

namespace Dace

open St (shiftR)

/-! ## Shifting control-flow states and outcomes -/

namespace CSt

@[simp] theorem shift_st (L : Log) (cst : CSt) : (cst.shift L).st = cst.st.shift L := rfl

theorem shift_withSt (L : Log) (cst : CSt) (s : St) : (cst.shift L).withSt (s.shift L) = (cst.withSt s).shift L := rfl

theorem lift_shift (L : Log) (cst : CSt) (r : Res St) :
    (cst.shift L).lift (r.map (St.shift L)) = (cst.lift r).map (CSt.shift L) := by
  cases r <;> rfl

end CSt

/-- Shift the state of a tree control outcome. -/
def shiftT (L : Log) (x : TCtl × CSt) : TCtl × CSt := (x.1, x.2.shift L)

/-- Shift the state of a tree node outcome. -/
def shiftOut (L : Log) (x : NodeOut) : NodeOut := (x.1, x.2.1, x.2.2.shift L)

theorem shiftT_eq_ok {L : Log} {r₀ : Res (TCtl × CSt)} {c : TCtl} {x : CSt}
    (h : Res.ok (c, x) = r₀.map (shiftT L)) : ∃ y, r₀ = .ok (c, y) ∧ x = y.shift L := by
  cases r₀ with
  | err e => cases h
  | ok a =>
    obtain ⟨c', y⟩ := a
    simp only [Res.map_ok, shiftT, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨y, rfl, rfl⟩

theorem shiftOut_eq_ok {L : Log} {r₀ : Res NodeOut} {sc : SibCtx} {c : TCtl} {x : CSt}
    (h : Res.ok (sc, c, x) = r₀.map (shiftOut L)) : ∃ y, r₀ = .ok (sc, c, y) ∧ x = y.shift L := by
  cases r₀ with
  | err e => cases h
  | ok a =>
    obtain ⟨sc', c', y⟩ := a
    simp only [Res.map_ok, shiftOut, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨y, rfl, rfl⟩

theorem leafOut_shift (L : Log) (r : Res CSt) : leafOut (r.map (CSt.shift L)) = (leafOut r).map (shiftOut L) := by
  cases r <;> rfl

theorem scopeOut_shift (L : Log) (r : Res (TCtl × CSt)) :
    scopeOut (r.map (shiftT L)) = (scopeOut r).map (shiftOut L) := by
  cases r <;> rfl

theorem chainOut_shift (L : Log) (r : Res (TCtl × CSt)) :
    chainOut (r.map (shiftT L)) = (chainOut r).map (shiftOut L) := by
  cases r <;> rfl

theorem dataflowOut_shift (L : Log) (r : Res CSt) :
    dataflowOut (r.map (CSt.shift L)) = (dataflowOut r).map (shiftOut L) := by
  cases r <;> rfl

/-! ## Leaf functions -/

section Leaf

variable (tc : TCtx) (ρ : Valuation) (L : Log)

theorem bindMemletIn_shift (m : Memlet) (st : St) :
    bindMemletIn tc ρ m (st.shift L) = (bindMemletIn tc ρ m st).map (shiftR L) := by
  unfold bindMemletIn
  cases tc.inData m with
  | mk d sub => exact bindInData_shift ..

theorem bindInputsT_shift (ins : MemletMap) (st : St) :
    bindInputsT tc ρ ins (st.shift L) = (bindInputsT tc ρ ins st).map (shiftR L) := by
  unfold bindInputsT
  rw [show (([] : List (Conn × ConnIn)), st.shift L) = shiftR L ([], st) from rfl,
    List.foldlM_res_comm (shiftR L) _ ?hstep]
  · exact Res.bind_shift _ fun ⟨bs, st'⟩ => rfl
  case hstep =>
    intro ⟨acc, st⟩ ⟨c, m⟩
    simp only [Prod.map_apply, id_eq]
    rw [bindMemletIn_shift]
    exact Res.bind_shift _ fun ⟨ci, st'⟩ => rfl

theorem applyOutT_shift (ins outs : MemletMap) (st : St) (out : TaskletOut) :
    applyOutT tc ρ ins outs (st.shift L) out = (applyOutT tc ρ ins outs st out).map (St.shift L) := by
  unfold applyOutT
  rw [List.foldlM_res_comm (St.shift L) _ ?hpop]
  · refine Res.bind_shift _ fun st => ?_
    rw [List.foldlM_res_comm (St.shift L) _ ?hsc]
    · refine Res.bind_shift _ fun st => ?_
      rw [List.foldlM_res_comm (St.shift L) _ ?haw]
      · refine Res.bind_shift _ fun st => ?_
        exact List.foldlM_res_comm (St.shift L) _ (fun st ⟨c, vs⟩ =>
          List.foldlM_res_comm (St.shift L) _ (fun st m => pushTo_shift ..) _ _) _ _
      case haw =>
        exact fun st ⟨c, rel, v⟩ =>
          List.foldlM_res_comm (St.shift L) _ (fun st m => storeArrayWrite_shift ..) _ _
    case hsc =>
      exact fun st ⟨c, v⟩ => List.foldlM_res_comm (St.shift L) _ (fun st m => storeScalar_shift ..) _ _
  case hpop =>
    intro st ⟨c, k⟩
    simp only
    split
    · rfl
    · refine List.foldlM_res_comm (St.shift L) _ (fun st m => ?_) _ _
      cases tc.inData m with
      | mk d sub => exact popFrom_shift ..

theorem execCodeT_shift (label : String) (ins outs : MemletMap) (st : St)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (ue : ErrReason) :
    execCodeT tc ρ label ins outs (st.shift L) run ue = (execCodeT tc ρ label ins outs st run ue).map (St.shift L) := by
  unfold execCodeT
  split
  · rfl
  · rw [bindInputsT_shift]
    refine Res.bind_shift _ fun ⟨b, st'⟩ => ?_
    simp only [Prod.map_apply, id_eq]
    split
    · rfl
    · exact applyOutT_shift ..

theorem execReduceT_shift (props : ReduceProps) (ins outs : MemletMap) (st : St) :
    execReduceT tc ρ props ins outs (st.shift L) = (execReduceT tc ρ props ins outs st).map (St.shift L) := by
  unfold execReduceT
  split
  · cases tc.inData _ with
    | mk d sub => exact reduceData_shift ..
  · rfl

theorem execLibraryT_shift (l : LibNode) (ins outs : MemletMap) (st : St) :
    execLibraryT tc ρ l ins outs (st.shift L) = (execLibraryT tc ρ l ins outs st).map (St.shift L) := by
  unfold execLibraryT
  split
  · exact execReduceT_shift ..
  · exact execCodeT_shift ..

theorem execCopyT_shift (target : Name) (m : Memlet) (st : St) :
    execCopyT tc ρ target m (st.shift L) = (execCopyT tc ρ target m st).map (St.shift L) := by
  unfold execCopyT
  cases tc.inData m with
  | mk d sub => exact copyData_shift ..

theorem execViewT_shift (target source : Name) (m : Memlet) (st : St) :
    execViewT tc ρ target source m (st.shift L) = (execViewT tc ρ target source m st).map (St.shift L) := by
  unfold execViewT
  split
  · cases tc.inData m with
    | mk d sub => exact aliasTo_shift ..
  · rfl

theorem execRefSetT_shift (target : Name) (m : Memlet) (src : RefSetSource) (st : St) :
    execRefSetT tc ρ target m src (st.shift L) = (execRefSetT tc ρ target m src st).map (St.shift L) := by
  unfold execRefSetT
  split
  · exact aliasTo_shift ..
  · rfl

theorem execDynScopeCopy_shift (p : Pending) (target : Sym) (m : Memlet) (st : St) :
    execDynScopeCopy tc ρ p target m (st.shift L) = (execDynScopeCopy tc ρ p target m st).map (shiftR L) := by
  unfold execDynScopeCopy
  simp only [St.shift_mem]
  split
  · split
    · exact Res.bind_shift' _ fun pt => rfl
    · rfl
  · cases tc.inData m with
    | mk d sub =>
      simp only
      rw [readScalarData_shift]
      refine Res.bind_shift _ fun ⟨v?, st'⟩ => ?_
      cases v? <;> rfl

theorem stepLeaf_shift (sc : SibCtx) (n : SNode) (cst : CSt) :
    stepLeaf tc sc n (cst.shift L) = (stepLeaf tc sc n cst).map (Res.map (shiftOut L)) := by
  unfold stepLeaf
  split <;> simp only [Option.map_some, Option.map_none, CSt.shift_ρ, CSt.shift_st]
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rw [applyAssigns_shift, leafOut_shift]
  · rw [execCodeT_shift, CSt.lift_shift, leafOut_shift]
  · rw [execLibraryT_shift, CSt.lift_shift, leafOut_shift]
  · rw [execCopyT_shift, CSt.lift_shift, leafOut_shift]
  · rw [execDynScopeCopy_shift]
    cases execDynScopeCopy tc cst.ρ sc.pending _ _ cst.st <;> rfl
  · rw [execViewT_shift, CSt.lift_shift, leafOut_shift]
  · rw [execRefSetT_shift, CSt.lift_shift, leafOut_shift]

theorem scopeFinishT_shift (pre : CSt) (logs : List Log) :
    scopeFinishT tc (pre.shift L) logs = (scopeFinishT tc pre logs).map (CSt.shift L) := by
  unfold scopeFinishT
  rw [CSt.shift_st, scopeFinish_shift]
  cases scopeFinish tc.ctx pre.st logs <;> rfl

theorem consumeFinishT_shift (pre : CSt) (l : Loc) (logs : List Log) (rest pushed : List Val) :
    consumeFinishT tc (pre.shift L) l logs rest pushed =
      (consumeFinishT tc pre l logs rest pushed).map (CSt.shift L) := by
  unfold consumeFinishT
  rw [CSt.shift_st, consumeFinish_shift]
  cases consumeFinish tc.ctx pre.st l logs rest pushed <;> rfl

end Leaf

/-! ## Map iterations and consume tasks ignore the pre-scope log -/

section Scopes

variable {tc : TCtx}

/-- The map iterations depend on the pre-map state only through its
valuation and memory. -/
theorem STree.IterExec.of_eq {params : List Sym} {body : List SNode} {priv : List Name} {pre pre' : CSt}
    {pts : List Idx} {r : Res (List Log)} (hρ : pre'.ρ = pre.ρ) (hm : pre'.mem = pre.mem) :
    STree.IterExec tc params body priv pre pts r → STree.IterExec tc params body priv pre' pts r
  | .nil => .nil
  | .headErr hb he => .headErr (by unfold iterStartT at hb ⊢; rw [hρ, hm]; exact hb) he
  | .tailErr hr => .tailErr (STree.IterExec.of_eq hρ hm hr)
  | .cons hb hl hr =>
    .cons (by unfold iterStartT at hb ⊢; rw [hρ, hm]; exact hb) hl (STree.IterExec.of_eq hρ hm hr)

theorem STree.IterExec.shift_iff {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt}
    {pts : List Idx} {r : Res (List Log)} (L : Log) :
    STree.IterExec tc params body priv (pre.shift L) pts r ↔ STree.IterExec tc params body priv pre pts r :=
  ⟨STree.IterExec.of_eq rfl rfl, STree.IterExec.of_eq rfl rfl⟩

/-- The consume tasks depend on the pre-scope state only through its memory. -/
theorem STree.ConsumeTasks.of_eq {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc}
    {np : Int} {priv : List Name} {pre pre' : CSt} {q : List Val} {r : Res (List Log × List Val × List Val)}
    (hm : pre'.mem = pre.mem) :
    STree.ConsumeTasks tc ρ c d body l np priv pre q r → STree.ConsumeTasks tc ρ c d body l np priv pre' q r
  | .condErr hc => .condErr (hm ▸ hc)
  | .stop hc => .stop (hm ▸ hc)
  | .bodyErr hc hq hp hb he => .bodyErr (hm ▸ hc) hq hp (by unfold taskStartT at hb ⊢; rw [hm]; exact hb) he
  | .restErr hc hq hp hb ho hr =>
    .restErr (hm ▸ hc) hq hp (by unfold taskStartT at hb ⊢; rw [hm]; exact hb) ho (STree.ConsumeTasks.of_eq hm hr)
  | .step hc hq hp hb ho hr =>
    .step (hm ▸ hc) hq hp (by unfold taskStartT at hb ⊢; rw [hm]; exact hb) ho (STree.ConsumeTasks.of_eq hm hr)

theorem STree.ConsumeTasks.shift_iff {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc}
    {np : Int} {priv : List Name} {pre : CSt} {q : List Val} {r : Res (List Log × List Val × List Val)} (L : Log) :
    STree.ConsumeTasks tc ρ c d body l np priv (pre.shift L) q r ↔ STree.ConsumeTasks tc ρ c d body l np priv pre q r :=
  ⟨STree.ConsumeTasks.of_eq rfl, STree.ConsumeTasks.of_eq rfl⟩

end Scopes

/-! ## The mutual family -/

section Rel

variable {tc : TCtx}

mutual

theorem STree.NodeExec.shift {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} (L : Log) :
    STree.NodeExec tc sc n cst r → STree.NodeExec tc sc n (cst.shift L) (r.map (shiftOut L))
  | .pendingErr hn hp => .pendingErr hn hp
  | .leaf hp h => .leaf hp (by rw [stepLeaf_shift, h]; rfl)
  | .namedRegion hp h => by
    rw [← scopeOut_shift]
    exact .namedRegion hp (STree.ListExec.shift L h)
  | .gblock hp h => by
    rw [← scopeOut_shift]
    exact .gblock hp (STree.ListExec.shift L h)
  | .ifCondErr hp hc => .ifCondErr hp hc
  | .ifSkip hp hc => .ifSkip hp hc
  | .ifTaken hp hc h => by
    rw [← chainOut_shift]
    exact .ifTaken hp hc (STree.ListExec.shift L h)
  | .stateIfCondErr hp hc => .stateIfCondErr hp hc
  | .stateIfSkip hp hc => .stateIfSkip hp hc
  | .stateIfTaken hp hc h => by
    rw [← scopeOut_shift]
    exact .stateIfTaken hp hc (STree.ListExec.shift L h)
  | .elifNoChain hp hc => .elifNoChain hp hc
  | .elifAfterTaken hp hc => .elifAfterTaken hp hc
  | .elifCondErr hp hc he => .elifCondErr hp hc he
  | .elifSkip hp hc he => .elifSkip hp hc he
  | .elifTaken hp hc he h => by
    rw [← chainOut_shift]
    exact .elifTaken hp hc he (STree.ListExec.shift L h)
  | .elseNoChain hp hc => .elseNoChain hp hc
  | .elseAfterTaken hp hc => .elseAfterTaken hp hc
  | .elseTaken hp hc h => by
    rw [← scopeOut_shift]
    exact .elseTaken hp hc (STree.ListExec.shift L h)
  | .loop hp hl h => by
    rw [← scopeOut_shift]
    exact .loop hp hl (STree.LoopExec.shift L h)
  | .mapSetupErr hs => .mapSetupErr hs
  | .mapIterErr hs hi => .mapIterErr hs (STree.IterExec.of_eq (by rfl) (by rfl) hi)
  | .mapScope hs hi => by
    rw [← dataflowOut_shift, ← scopeFinishT_shift]
    exact .mapScope hs (STree.IterExec.of_eq (by rfl) (by rfl) hi)
  | .consumeSetupErr hs => .consumeSetupErr hs
  | .consumeTasksErr hs ht => .consumeTasksErr hs (STree.ConsumeTasks.of_eq (by rfl) ht)
  | .consumeScope hs ht => by
    rw [← dataflowOut_shift, ← consumeFinishT_shift]
    exact .consumeScope hs (STree.ConsumeTasks.of_eq (by rfl) ht)

theorem STree.ListExec.shift {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (L : Log) :
    STree.ListExec tc full gb sc nodes cst r → STree.ListExec tc full gb sc nodes (cst.shift L) (r.map (shiftT L))
  | .nil hp => .nil hp
  | .nilPending hp => .nilPending hp
  | .headErr h => .headErr (STree.NodeExec.shift L h)
  | .step h hr => .step (STree.NodeExec.shift L h) (STree.ListExec.shift L hr)
  | .jump h hc => .jump (STree.NodeExec.shift L h) hc
  | .gotoResume h hl hr => .gotoResume (STree.NodeExec.shift L h) hl (STree.ListExec.shift L hr)
  | .gotoEscape h hl => .gotoEscape (STree.NodeExec.shift L h) hl

theorem STree.LoopExec.shift {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopExec tc l body cst r → STree.LoopExec tc l body (cst.shift L) (r.map (shiftT L))
  | .initErr hi => .initErr ((applyAssign?_shift ..).trans (congrArg _ hi))
  | .guardFirst hi hv hg =>
    .guardFirst ((applyAssign?_shift ..).trans (congrArg _ hi)) hv (STree.LoopGuard.shift L hg)
  | .bodyFirst hi hv hb =>
    .bodyFirst ((applyAssign?_shift ..).trans (congrArg _ hi)) hv (STree.LoopIter.shift L hb)

theorem STree.LoopGuard.shift {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopGuard tc l body cst r → STree.LoopGuard tc l body (cst.shift L) (r.map (shiftT L))
  | .condErr hc => .condErr hc
  | .exit hc => .exit hc
  | .enter hc hb => .enter hc (STree.LoopIter.shift L hb)

theorem STree.LoopIter.shift {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopIter tc l body cst r → STree.LoopIter tc l body (cst.shift L) (r.map (shiftT L))
  | .bodyErr hb => .bodyErr (STree.ListExec.shift L hb)
  | .ret hb => .ret (STree.ListExec.shift L hb)
  | .brk hb => .brk (STree.ListExec.shift L hb)
  | .gotoOut hb => .gotoOut (STree.ListExec.shift L hb)
  | .tail hb hc ht => .tail (STree.ListExec.shift L hb) hc (STree.LoopTail.shift L ht)

theorem STree.LoopTail.shift {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopTail tc l body cst r → STree.LoopTail tc l body (cst.shift L) (r.map (shiftT L))
  | .updateErr hv hu => .updateErr hv ((applyAssign?_shift ..).trans (congrArg _ hu))
  | .updateGuard hv hu hg =>
    .updateGuard hv ((applyAssign?_shift ..).trans (congrArg _ hu)) (STree.LoopGuard.shift L hg)
  | .condFirstErr hv hc => .condFirstErr hv hc
  | .condFirstExit hv hc => .condFirstExit hv hc
  | .condFirstUpdateErr hv hc hu => .condFirstUpdateErr hv hc ((applyAssign?_shift ..).trans (congrArg _ hu))
  | .condFirstNext hv hc hu hb =>
    .condFirstNext hv hc ((applyAssign?_shift ..).trans (congrArg _ hu)) (STree.LoopIter.shift L hb)

end

mutual

theorem STree.NodeExec.unshift {sc : SibCtx} {n : SNode} {cst' cst : CSt} {L : Log} {r : Res NodeOut}
    (heq : cst' = cst.shift L) :
    STree.NodeExec tc sc n cst' r → ∃ r₀, STree.NodeExec tc sc n cst r₀ ∧ r = r₀.map (shiftOut L)
  | .pendingErr hn hp => ⟨_, .pendingErr hn hp, rfl⟩
  | .leaf hp h => by
    rw [heq, stepLeaf_shift, Option.map_eq_some_iff] at h
    obtain ⟨r₀, h₀, he⟩ := h
    exact ⟨r₀, .leaf hp h₀, he.symm⟩
  | .namedRegion hp h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    exact ⟨_, .namedRegion hp h₀, scopeOut_shift ..⟩
  | .gblock hp h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    exact ⟨_, .gblock hp h₀, scopeOut_shift ..⟩
  | .ifCondErr hp hc => by
    subst heq
    exact ⟨_, .ifCondErr hp hc, rfl⟩
  | .ifSkip hp hc => by
    subst heq
    exact ⟨_, .ifSkip hp hc, rfl⟩
  | .ifTaken hp hc h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    subst heq
    exact ⟨_, .ifTaken hp hc h₀, chainOut_shift ..⟩
  | .stateIfCondErr hp hc => by
    subst heq
    exact ⟨_, .stateIfCondErr hp hc, rfl⟩
  | .stateIfSkip hp hc => by
    subst heq
    exact ⟨_, .stateIfSkip hp hc, rfl⟩
  | .stateIfTaken hp hc h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    subst heq
    exact ⟨_, .stateIfTaken hp hc h₀, scopeOut_shift ..⟩
  | .elifNoChain hp hc => ⟨_, .elifNoChain hp hc, rfl⟩
  | .elifAfterTaken hp hc => by
    subst heq
    exact ⟨_, .elifAfterTaken hp hc, rfl⟩
  | .elifCondErr hp hc he => by
    subst heq
    exact ⟨_, .elifCondErr hp hc he, rfl⟩
  | .elifSkip hp hc he => by
    subst heq
    exact ⟨_, .elifSkip hp hc he, rfl⟩
  | .elifTaken hp hc he h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    subst heq
    exact ⟨_, .elifTaken hp hc he h₀, chainOut_shift ..⟩
  | .elseNoChain hp hc => ⟨_, .elseNoChain hp hc, rfl⟩
  | .elseAfterTaken hp hc => by
    subst heq
    exact ⟨_, .elseAfterTaken hp hc, rfl⟩
  | .elseTaken hp hc h => by
    obtain ⟨_, h₀, rfl⟩ := STree.ListExec.unshift heq h
    exact ⟨_, .elseTaken hp hc h₀, scopeOut_shift ..⟩
  | .loop hp hl h => by
    obtain ⟨_, h₀, rfl⟩ := STree.LoopExec.unshift heq h
    exact ⟨_, .loop hp hl h₀, scopeOut_shift ..⟩
  | .mapSetupErr hs => by
    subst heq
    exact ⟨_, .mapSetupErr hs, rfl⟩
  | .mapIterErr hs hi => by
    subst heq
    exact ⟨_, .mapIterErr hs (STree.IterExec.of_eq (by rfl) (by rfl) hi), rfl⟩
  | .mapScope hs hi => by
    subst heq
    refine ⟨_, .mapScope hs (STree.IterExec.of_eq (by rfl) (by rfl) hi), ?_⟩
    rw [scopeFinishT_shift, dataflowOut_shift]
  | .consumeSetupErr hs => by
    subst heq
    exact ⟨_, .consumeSetupErr hs, rfl⟩
  | .consumeTasksErr hs ht => by
    subst heq
    exact ⟨_, .consumeTasksErr hs (STree.ConsumeTasks.of_eq (by rfl) ht), rfl⟩
  | .consumeScope hs ht => by
    subst heq
    refine ⟨_, .consumeScope hs (STree.ConsumeTasks.of_eq (by rfl) ht), ?_⟩
    rw [consumeFinishT_shift, dataflowOut_shift]

theorem STree.ListExec.unshift {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode} {cst' cst : CSt}
    {L : Log} {r : Res (TCtl × CSt)} (heq : cst' = cst.shift L) :
    STree.ListExec tc full gb sc nodes cst' r → ∃ r₀, STree.ListExec tc full gb sc nodes cst r₀ ∧ r = r₀.map (shiftT L)
  | .nil hp => ⟨_, .nil hp, by rw [heq]; rfl⟩
  | .nilPending hp => ⟨_, .nilPending hp, rfl⟩
  | .headErr h => by
    obtain ⟨r₀, h₀, he⟩ := STree.NodeExec.unshift heq h
    rw [eq_comm, Res.map_eq_err] at he
    subst he
    exact ⟨_, .headErr h₀, rfl⟩
  | .step h hr => by
    obtain ⟨r₀, h₀, he⟩ := STree.NodeExec.unshift heq h
    obtain ⟨y, rfl, hy⟩ := shiftOut_eq_ok he
    obtain ⟨_, h₁, he₁⟩ := STree.ListExec.unshift hy hr
    exact ⟨_, .step h₀ h₁, he₁⟩
  | .jump h hc => by
    obtain ⟨r₀, h₀, he⟩ := STree.NodeExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftOut_eq_ok he
    exact ⟨_, .jump h₀ hc, rfl⟩
  | .gotoResume h hl hr => by
    obtain ⟨r₀, h₀, he⟩ := STree.NodeExec.unshift heq h
    obtain ⟨y, rfl, hy⟩ := shiftOut_eq_ok he
    obtain ⟨_, h₁, he₁⟩ := STree.ListExec.unshift hy hr
    exact ⟨_, .gotoResume h₀ hl h₁, he₁⟩
  | .gotoEscape h hl => by
    obtain ⟨r₀, h₀, he⟩ := STree.NodeExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftOut_eq_ok he
    exact ⟨_, .gotoEscape h₀ hl, rfl⟩

theorem STree.LoopExec.unshift {l : LoopMeta} {body : List SNode} {cst' cst : CSt} {L : Log} {r : Res (TCtl × CSt)}
    (heq : cst' = cst.shift L) :
    STree.LoopExec tc l body cst' r → ∃ r₀, STree.LoopExec tc l body cst r₀ ∧ r = r₀.map (shiftT L)
  | .initErr hi => by
    rw [heq, applyAssign?_shift, Res.map_eq_err] at hi
    exact ⟨_, .initErr hi, rfl⟩
  | .guardFirst hi hv hg => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hi
    obtain ⟨cst₁, hi, h₁⟩ := hi
    obtain ⟨_, h₀, he⟩ := STree.LoopGuard.unshift h₁.symm hg
    exact ⟨_, .guardFirst hi hv h₀, he⟩
  | .bodyFirst hi hv hb => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hi
    obtain ⟨cst₁, hi, h₁⟩ := hi
    obtain ⟨_, h₀, he⟩ := STree.LoopIter.unshift h₁.symm hb
    exact ⟨_, .bodyFirst hi hv h₀, he⟩

theorem STree.LoopGuard.unshift {l : LoopMeta} {body : List SNode} {cst' cst : CSt} {L : Log} {r : Res (TCtl × CSt)}
    (heq : cst' = cst.shift L) :
    STree.LoopGuard tc l body cst' r → ∃ r₀, STree.LoopGuard tc l body cst r₀ ∧ r = r₀.map (shiftT L)
  | .condErr hc => by
    subst heq
    exact ⟨_, .condErr hc, rfl⟩
  | .exit hc => by
    subst heq
    exact ⟨_, .exit hc, rfl⟩
  | .enter hc hb => by
    obtain ⟨_, h₀, he⟩ := STree.LoopIter.unshift heq hb
    subst heq
    exact ⟨_, .enter hc h₀, he⟩

theorem STree.LoopIter.unshift {l : LoopMeta} {body : List SNode} {cst' cst : CSt} {L : Log} {r : Res (TCtl × CSt)}
    (heq : cst' = cst.shift L) :
    STree.LoopIter tc l body cst' r → ∃ r₀, STree.LoopIter tc l body cst r₀ ∧ r = r₀.map (shiftT L)
  | .bodyErr hb => by
    obtain ⟨r₀, h₀, he⟩ := STree.ListExec.unshift heq hb
    rw [eq_comm, Res.map_eq_err] at he
    subst he
    exact ⟨_, .bodyErr h₀, rfl⟩
  | .ret hb => by
    obtain ⟨r₀, h₀, he⟩ := STree.ListExec.unshift heq hb
    obtain ⟨y, rfl, rfl⟩ := shiftT_eq_ok he
    exact ⟨_, .ret h₀, rfl⟩
  | .brk hb => by
    obtain ⟨r₀, h₀, he⟩ := STree.ListExec.unshift heq hb
    obtain ⟨y, rfl, rfl⟩ := shiftT_eq_ok he
    exact ⟨_, .brk h₀, rfl⟩
  | .gotoOut hb => by
    obtain ⟨r₀, h₀, he⟩ := STree.ListExec.unshift heq hb
    obtain ⟨y, rfl, rfl⟩ := shiftT_eq_ok he
    exact ⟨_, .gotoOut h₀, rfl⟩
  | .tail hb hc ht => by
    obtain ⟨r₀, h₀, he⟩ := STree.ListExec.unshift heq hb
    obtain ⟨y, rfl, hy⟩ := shiftT_eq_ok he
    obtain ⟨_, h₁, he₁⟩ := STree.LoopTail.unshift hy ht
    exact ⟨_, .tail h₀ hc h₁, he₁⟩

theorem STree.LoopTail.unshift {l : LoopMeta} {body : List SNode} {cst' cst : CSt} {L : Log} {r : Res (TCtl × CSt)}
    (heq : cst' = cst.shift L) :
    STree.LoopTail tc l body cst' r → ∃ r₀, STree.LoopTail tc l body cst r₀ ∧ r = r₀.map (shiftT L)
  | .updateErr hv hu => by
    rw [heq, applyAssign?_shift, Res.map_eq_err] at hu
    exact ⟨_, .updateErr hv hu, rfl⟩
  | .updateGuard hv hu hg => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hu
    obtain ⟨cst₁, hu, h₁⟩ := hu
    obtain ⟨_, h₀, he⟩ := STree.LoopGuard.unshift h₁.symm hg
    exact ⟨_, .updateGuard hv hu h₀, he⟩
  | .condFirstErr hv hc => by
    subst heq
    exact ⟨_, .condFirstErr hv hc, rfl⟩
  | .condFirstExit hv hc => by
    subst heq
    exact ⟨_, .condFirstExit hv hc, rfl⟩
  | .condFirstUpdateErr hv hc hu => by
    subst heq
    rw [applyAssign?_shift, Res.map_eq_err] at hu
    exact ⟨_, .condFirstUpdateErr hv hc hu, rfl⟩
  | .condFirstNext hv hc hu hb => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hu
    obtain ⟨cst₁, hu, h₁⟩ := hu
    obtain ⟨_, h₀, he⟩ := STree.LoopIter.unshift h₁.symm hb
    subst heq
    exact ⟨_, .condFirstNext hv hc hu h₀, he⟩

end

/-! ### Logs are write-only (schedule trees) -/

variable {cst : CSt}

theorem STree.NodeExec.shift_iff {sc : SibCtx} {n : SNode} {r : Res NodeOut} (L : Log) :
    STree.NodeExec tc sc n (cst.shift L) r ↔ ∃ r₀, STree.NodeExec tc sc n cst r₀ ∧ r = r₀.map (shiftOut L) :=
  ⟨STree.NodeExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem STree.ListExec.shift_iff {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode}
    {r : Res (TCtl × CSt)} (L : Log) :
    STree.ListExec tc full gb sc nodes (cst.shift L) r ↔
      ∃ r₀, STree.ListExec tc full gb sc nodes cst r₀ ∧ r = r₀.map (shiftT L) :=
  ⟨STree.ListExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem STree.LoopExec.shift_iff {l : LoopMeta} {body : List SNode} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopExec tc l body (cst.shift L) r ↔ ∃ r₀, STree.LoopExec tc l body cst r₀ ∧ r = r₀.map (shiftT L) :=
  ⟨STree.LoopExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem STree.LoopGuard.shift_iff {l : LoopMeta} {body : List SNode} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopGuard tc l body (cst.shift L) r ↔ ∃ r₀, STree.LoopGuard tc l body cst r₀ ∧ r = r₀.map (shiftT L) :=
  ⟨STree.LoopGuard.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem STree.LoopIter.shift_iff {l : LoopMeta} {body : List SNode} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopIter tc l body (cst.shift L) r ↔ ∃ r₀, STree.LoopIter tc l body cst r₀ ∧ r = r₀.map (shiftT L) :=
  ⟨STree.LoopIter.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem STree.LoopTail.shift_iff {l : LoopMeta} {body : List SNode} {r : Res (TCtl × CSt)} (L : Log) :
    STree.LoopTail tc l body (cst.shift L) r ↔ ∃ r₀, STree.LoopTail tc l body cst r₀ ∧ r = r₀.map (shiftT L) :=
  ⟨STree.LoopTail.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

/-- The form used by congruence proofs: a node run from a state with log `L`
is a run from the same state with the empty log, whose final log `δ`
becomes `L.seq δ`. -/
theorem STree.NodeExec.log_iff {sc : SibCtx} {n : SNode} {r : Res NodeOut} (L : Log) :
    STree.NodeExec tc sc n { cst with log := L } r ↔
      ∃ r₀, STree.NodeExec tc sc n { cst with log := Log.empty } r₀ ∧ r = r₀.map (shiftOut L) := by
  rw [CSt.withLog_eq_shift]
  exact STree.NodeExec.shift_iff L

theorem STree.ListExec.log_iff {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode}
    {r : Res (TCtl × CSt)} (L : Log) :
    STree.ListExec tc full gb sc nodes { cst with log := L } r ↔
      ∃ r₀, STree.ListExec tc full gb sc nodes { cst with log := Log.empty } r₀ ∧ r = r₀.map (shiftT L) := by
  rw [CSt.withLog_eq_shift]
  exact STree.ListExec.shift_iff L

/-! ### Related start states (schedule trees) -/

/-- Related tree node outcomes: same sibling context and signal, `CSt.Rel` states. -/
def NodeOutRel (x' x : NodeOut) : Prop := x'.1 = x.1 ∧ x'.2.1 = x.2.1 ∧ CSt.Rel x'.2.2 x.2.2

/-- Related tree control outcomes: same signal, `CSt.Rel` states. -/
def TCtlRel (x' x : TCtl × CSt) : Prop := x'.1 = x.1 ∧ CSt.Rel x'.2 x.2

theorem NodeOutRel.shiftOut {g' g : Log} (h : Log.Le g' g) (x : NodeOut) : NodeOutRel (shiftOut g' x) (shiftOut g x) :=
  ⟨rfl, rfl, CSt.Rel.shift h x.2.2⟩

theorem TCtlRel.shiftT {g' g : Log} (h : Log.Le g' g) (x : TCtl × CSt) : TCtlRel (shiftT g' x) (shiftT g x) :=
  ⟨rfl, CSt.Rel.shift h x.2⟩

theorem STree.NodeExec.rel {sc : SibCtx} {n : SNode} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, STree.NodeExec tc sc n a r → ∃ r', STree.NodeExec tc sc n a' r' ∧ Res.Rel NodeOutRel r' r) ∧
    (∀ r', STree.NodeExec tc sc n a' r' → ∃ r, STree.NodeExec tc sc n a r ∧ Res.Rel NodeOutRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.NodeExec tc sc n a r) (fun _ L _ => STree.NodeExec.shift_iff L)
    NodeOutRel.shiftOut h

theorem STree.ListExec.rel {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode} {a' a : CSt}
    (h : CSt.Rel a' a) :
    (∀ r, STree.ListExec tc full gb sc nodes a r →
      ∃ r', STree.ListExec tc full gb sc nodes a' r' ∧ Res.Rel TCtlRel r' r) ∧
    (∀ r', STree.ListExec tc full gb sc nodes a' r' →
      ∃ r, STree.ListExec tc full gb sc nodes a r ∧ Res.Rel TCtlRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.ListExec tc full gb sc nodes a r)
    (fun _ L _ => STree.ListExec.shift_iff L) TCtlRel.shiftT h

theorem STree.LoopExec.rel {l : LoopMeta} {body : List SNode} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, STree.LoopExec tc l body a r → ∃ r', STree.LoopExec tc l body a' r' ∧ Res.Rel TCtlRel r' r) ∧
    (∀ r', STree.LoopExec tc l body a' r' → ∃ r, STree.LoopExec tc l body a r ∧ Res.Rel TCtlRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.LoopExec tc l body a r) (fun _ L _ => STree.LoopExec.shift_iff L)
    TCtlRel.shiftT h

theorem STree.LoopGuard.rel {l : LoopMeta} {body : List SNode} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, STree.LoopGuard tc l body a r → ∃ r', STree.LoopGuard tc l body a' r' ∧ Res.Rel TCtlRel r' r) ∧
    (∀ r', STree.LoopGuard tc l body a' r' → ∃ r, STree.LoopGuard tc l body a r ∧ Res.Rel TCtlRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.LoopGuard tc l body a r) (fun _ L _ => STree.LoopGuard.shift_iff L)
    TCtlRel.shiftT h

theorem STree.LoopIter.rel {l : LoopMeta} {body : List SNode} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, STree.LoopIter tc l body a r → ∃ r', STree.LoopIter tc l body a' r' ∧ Res.Rel TCtlRel r' r) ∧
    (∀ r', STree.LoopIter tc l body a' r' → ∃ r, STree.LoopIter tc l body a r ∧ Res.Rel TCtlRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.LoopIter tc l body a r) (fun _ L _ => STree.LoopIter.shift_iff L)
    TCtlRel.shiftT h

theorem STree.LoopTail.rel {l : LoopMeta} {body : List SNode} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, STree.LoopTail tc l body a r → ∃ r', STree.LoopTail tc l body a' r' ∧ Res.Rel TCtlRel r' r) ∧
    (∀ r', STree.LoopTail tc l body a' r' → ∃ r, STree.LoopTail tc l body a r ∧ Res.Rel TCtlRel r' r) :=
  CSt.Rel.transferWith (X := fun a r => STree.LoopTail tc l body a r) (fun _ L _ => STree.LoopTail.shift_iff L)
    TCtlRel.shiftT h

end Rel

end Dace
