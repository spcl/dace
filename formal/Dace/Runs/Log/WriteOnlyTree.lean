-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics
import Dace.Runs.Log.WriteOnly

/-!
# Logs are write-only (schedule trees)

The schedule-tree analogue of `Dace.Runs.Log.WriteOnly`: no execution relation
of the tree semantics reads the log of the control-flow state it starts
from.  With `CSt.shift L cst` (prefix `L` to the log),
a run of `NodeExec` (`Res NodeOut`), `ListExec`, `LoopExec`, `LoopGuard`,
`LoopIter` or `LoopTail` (`Res (TCtl × CSt)`) from `cst.shift L` is a run
from `cst` whose outcome is shifted by `L` (`STree.*.unshift`;
`STree.ListExec.of_log` for `{cst with log := L}` versus
`{cst with log := .empty}`).  `IterExec` and `ConsumeTasks` do not depend on
the log of the pre-scope state at all (`STree.IterExec.of_eq`,
`STree.ConsumeTasks.of_eq`).

The leaf functions of `Dace.STree.Semantics.Leaf` commute with the shift
(through the SDFG leaf lemmas); the relations go by mutual recursion on the
derivation.
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

end Scopes

/-! ## The mutual family -/

section Rel

variable {tc : TCtx}

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

/-- **Logs are write-only**: a run from a state whose log is `L` is a run
from the same state with an empty log, the outcome's log prefixed by `L`. -/
theorem STree.ListExec.of_log {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (L : Log) (h : STree.ListExec tc full gb sc nodes { cst with log := L } r) :
    ∃ r₀, STree.ListExec tc full gb sc nodes { cst with log := Log.empty } r₀ ∧ r = r₀.map (shiftT L) :=
  STree.ListExec.unshift (CSt.withLog_eq_shift cst L) h

end Rel

end Dace
