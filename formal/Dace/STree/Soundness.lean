-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Interp

/-!
# Soundness of the schedule tree interpreter

`Dace.STree.Interp` mirrors the relational semantics
`Dace.STree.Semantics.Exec` function by function.  This module proves that
every outcome the interpreter produces is derivable:
`execX fuel … = some r → XExec … r` for each of the eight functions,
culminating in `STree.run_sound`.

The proof follows `Dace.SDFG.Soundness`: the eight statements are bundled
into `STree.Sound fuel` and proven together by induction on the fuel; at
fuel `0` every function returns `none`, at `fuel + 1` each function is
unfolded once, its case distinctions are replayed on the hypothesis
(`split at h`), recursive calls are discharged by the induction hypothesis
and the matching constructor closes the goal.

The only nondeterminism of the relation is the processing element of a
consume task; the interpreter's choice `0` satisfies the premise
`0 ≤ p ∧ p < num_pes` exactly when a task can run.  The node-level test
`n.consumesPending || sc.pending.isEmpty` of `execNode` yields the pending
premises of the constructors (`pending_true`, `leaf_pending`).
-/

set_option autoImplicit false

namespace Dace.STree

/-- Soundness of every interpreter function at one fuel value. -/
structure Sound (fuel : Nat) : Prop where
  node : ∀ {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut},
    STree.execNode fuel tc sc n cst = some r → STree.NodeExec tc sc n cst r
  list : ∀ {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)},
    STree.execList fuel tc full gb sc nodes cst = some r → STree.ListExec tc full gb sc nodes cst r
  iter : ∀ {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {pts : List Idx}
    {r : Res (List Log)},
    STree.execIter fuel tc params body priv pre pts = some r → STree.IterExec tc params body priv pre pts r
  consume : ∀ {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
    {priv : List Name} {pre : CSt} {q : List Val} {r : Res (List Log × List Val × List Val)},
    STree.execConsume fuel tc ρ c d body l np priv pre q = some r →
      STree.ConsumeTasks tc ρ c d body l np priv pre q r
  loop : ∀ {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
    STree.execLoop fuel tc l body cst = some r → STree.LoopExec tc l body cst r
  guard : ∀ {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
    STree.loopGuard fuel tc l body cst = some r → STree.LoopGuard tc l body cst r
  liter : ∀ {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
    STree.loopIter fuel tc l body cst = some r → STree.LoopIter tc l body cst r
  tail : ∀ {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
    STree.loopTail fuel tc l body cst = some r → STree.LoopTail tc l body cst r

/-! ## Fuel `0`: no outcome -/

theorem sound_zero : STree.Sound 0 where
  node h := by simp only [STree.execNode] at h; cases h
  list h := by simp only [STree.execList] at h; cases h
  iter h := by simp only [STree.execIter] at h; cases h
  consume h := by simp only [STree.execConsume] at h; cases h
  loop h := by simp only [STree.execLoop] at h; cases h
  guard h := by simp only [STree.loopGuard] at h; cases h
  liter h := by simp only [STree.loopIter] at h; cases h
  tail h := by simp only [STree.loopTail] at h; cases h

/-! ## Pending dynamic scope inputs -/

/-- A node that does not consume pending inputs ran only because none were pending. -/
private theorem pending_true {n : SNode} {sc : SibCtx} (h : (n.consumesPending || sc.pending.isEmpty) = true)
    (hn : n.consumesPending = false) : sc.pending.isEmpty = true := by
  rw [hn] at h; exact h

/-- Loop scopes do not consume pending inputs. -/
private theorem consumesPending_of_loop {n : SNode} {l : LoopMeta} {cs : List SNode} (hl : n.loop? = some (l, cs)) :
    n.consumesPending = false := by
  cases n <;> simp_all [SNode.loop?, SNode.consumesPending, SNode.isDataflowScope, SNode.isDynScopeCopy]

/-- The premise of `NodeExec.leaf` for a node `stepLeaf` accepts. -/
private theorem leaf_pending {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut}
    (hp : (n.consumesPending || sc.pending.isEmpty) = true) (h : stepLeaf tc sc n cst = some r) :
    (n.isDynScopeCopy || sc.pending.isEmpty) = true := by
  cases n <;> simp_all [SNode.consumesPending, SNode.isDataflowScope, SNode.isDynScopeCopy, stepLeaf]

/-! ## Fuel `fuel + 1`: one step per function -/

theorem execNode_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt}
    {r : Res NodeOut} (h : STree.execNode (fuel + 1) tc sc n cst = some r) : STree.NodeExec tc sc n cst r := by
  simp only [STree.execNode] at h
  split at h
  next hp =>
    cases h
    have hp' := Bool.or_eq_false_iff.mp hp
    exact .pendingErr hp'.1 hp'.2
  next hp =>
    split at h
    next l cs hl =>
      split at h
      · cases h
      next r' he => cases h; exact .loop (pending_true hp (consumesPending_of_loop hl)) hl (ih.loop he)
    next _ =>
      split at h
      next label cs =>
        split at h
        · cases h
        next r' he => cases h; exact .namedRegion (pending_true hp rfl) (ih.list he)
      next cs =>
        split at h
        · cases h
        next r' he => cases h; exact .gblock (pending_true hp rfl) (ih.list he)
      next c cs =>
        split at h
        next e hc => cases h; exact .ifCondErr (pending_true hp rfl) hc
        next hc => cases h; exact .ifSkip (pending_true hp rfl) hc
        next hc =>
          split at h
          · cases h
          next r' he => cases h; exact .ifTaken (pending_true hp rfl) hc (ih.list he)
      next c cs =>
        split at h
        next e hc => cases h; exact .stateIfCondErr (pending_true hp rfl) hc
        next hc => cases h; exact .stateIfSkip (pending_true hp rfl) hc
        next hc =>
          split at h
          · cases h
          next r' he => cases h; exact .stateIfTaken (pending_true hp rfl) hc (ih.list he)
      next c cs =>
        split at h
        next hc => cases h; exact .elifNoChain (pending_true hp rfl) hc
        next hc => cases h; exact .elifAfterTaken (pending_true hp rfl) hc
        next hc =>
          split at h
          next e he => cases h; exact .elifCondErr (pending_true hp rfl) hc he
          next he => cases h; exact .elifSkip (pending_true hp rfl) hc he
          next he =>
            split at h
            · cases h
            next r' hl => cases h; exact .elifTaken (pending_true hp rfl) hc he (ih.list hl)
      next cs =>
        split at h
        next hc => cases h; exact .elseNoChain (pending_true hp rfl) hc
        next hc => cases h; exact .elseAfterTaken (pending_true hp rfl) hc
        next hc =>
          split at h
          · cases h
          next r' hl => cases h; exact .elseTaken (pending_true hp rfl) hc (ih.list hl)
      next mp cs =>
        split at h
        next e hs => cases h; exact .mapSetupErr hs
        next ρ' pts hs =>
          split at h
          · cases h
          next e hi => cases h; exact .mapIterErr hs (ih.iter hi)
          next logs hi => cases h; exact .mapScope hs (ih.iter hi)
      next c cs =>
        split at h
        next e hs => cases h; exact .consumeSetupErr hs
        next ρ' d l np hs =>
          split at h
          · cases h
          next e ht => cases h; exact .consumeTasksErr hs (ih.consume ht)
          next logs rest pushed ht => cases h; exact .consumeScope hs (ih.consume ht)
      next n _ _ _ _ _ _ _ _ => exact .leaf (leaf_pending hp h) h

theorem execList_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {full : List SNode} {gb : Bool}
    {sc : SibCtx} {nodes : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
    (h : STree.execList (fuel + 1) tc full gb sc nodes cst = some r) : STree.ListExec tc full gb sc nodes cst r := by
  simp only [STree.execList] at h
  split at h
  · split at h
    next hp => cases h; exact .nil hp
    next hp => cases h; exact .nilPending hp
  · split at h
    · cases h
    next e he => cases h; exact .headErr (ih.node he)
    next sc' cst' he => exact .step (ih.node he) (ih.list h)
    next sc' cst' he => cases h; exact .jump (ih.node he) rfl
    next sc' cst' he => cases h; exact .jump (ih.node he) rfl
    next sc' cst' he => cases h; exact .jump (ih.node he) rfl
    next sc' l cst' he =>
      split at h
      next suffix hl => exact .gotoResume (ih.node he) hl (ih.list h)
      next hl => cases h; exact .gotoEscape (ih.node he) hl

theorem execIter_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {params : List Sym} {body : List SNode}
    {priv : List Name} {pre : CSt} {pts : List Idx} {r : Res (List Log)}
    (h : STree.execIter (fuel + 1) tc params body priv pre pts = some r) :
    STree.IterExec tc params body priv pre pts r := by
  simp only [STree.execIter] at h
  split at h
  · cases h; exact .nil
  · split at h
    · cases h
    next res hb =>
      split at h
      next e he => cases h; exact .headErr (ih.list hb) he
      next g hl =>
        split at h
        · cases h
        next e hr => cases h; exact .tailErr (ih.iter hr)
        next logs hr => cases h; exact .cons (ih.list hb) hl (ih.iter hr)

theorem execConsume_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {ρ : Valuation} {c : ConsumeNode}
    {d : Name} {body : List SNode} {l : Loc} {np : Int} {priv : List Name} {pre : CSt} {q : List Val}
    {r : Res (List Log × List Val × List Val)}
    (h : STree.execConsume (fuel + 1) tc ρ c d body l np priv pre q = some r) :
    STree.ConsumeTasks tc ρ c d body l np priv pre q r := by
  simp only [STree.execConsume] at h
  split at h
  next e hc => cases h; exact .condErr hc
  next hc => cases h; exact .stop hc
  next hc =>
    split at h
    · cases h
    next hq =>
      split at h
      next hp =>
        split at h
        · cases h
        next res hb =>
          split at h
          next e he => cases h; exact .bodyErr hc hq hp (ih.list hb) he
          next g pushed ho =>
            split at h
            · cases h
            next e hr => cases h; exact .restErr hc hq hp (ih.list hb) ho (ih.consume hr)
            next logs rest pushed' hr => cases h; exact .step hc hq hp (ih.list hb) ho (ih.consume hr)
      · cases h

theorem execLoop_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {l : LoopMeta} {body : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.execLoop (fuel + 1) tc l body cst = some r) :
    STree.LoopExec tc l body cst r := by
  simp only [STree.execLoop] at h
  split at h
  next e hi => cases h; exact .initErr hi
  next cst₁ hi =>
    split at h
    next hv => exact .guardFirst hi hv (ih.guard h)
    next hv => exact .bodyFirst hi hv (ih.liter h)

theorem loopGuard_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {l : LoopMeta} {body : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.loopGuard (fuel + 1) tc l body cst = some r) :
    STree.LoopGuard tc l body cst r := by
  simp only [STree.loopGuard] at h
  split at h
  next e hc => cases h; exact .condErr hc
  next hc => cases h; exact .exit hc
  next hc => exact .enter hc (ih.liter h)

theorem loopIter_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {l : LoopMeta} {body : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.loopIter (fuel + 1) tc l body cst = some r) :
    STree.LoopIter tc l body cst r := by
  simp only [STree.loopIter] at h
  split at h
  · cases h
  next e hb => cases h; exact .bodyErr (ih.list hb)
  next cst₁ hb => cases h; exact .ret (ih.list hb)
  next cst₁ hb => cases h; exact .brk (ih.list hb)
  next lbl cst₁ hb => cases h; exact .gotoOut (ih.list hb)
  next cst₁ hb => exact .tail (ih.list hb) rfl (ih.tail h)
  next cst₁ hb => exact .tail (ih.list hb) rfl (ih.tail h)

theorem loopTail_succ {fuel : Nat} (ih : STree.Sound fuel) {tc : TCtx} {l : LoopMeta} {body : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.loopTail (fuel + 1) tc l body cst = some r) :
    STree.LoopTail tc l body cst r := by
  simp only [STree.loopTail] at h
  split at h
  next hv =>
    split at h
    next e hu => cases h; exact .updateErr hv hu
    next cst₁ hu => exact .updateGuard hv hu (ih.guard h)
  next hv =>
    split at h
    next e hc => cases h; exact .condFirstErr hv hc
    next hc => cases h; exact .condFirstExit hv hc
    next hc =>
      split at h
      next e hu => cases h; exact .condFirstUpdateErr hv hc hu
      next cst₁ hu => exact .condFirstNext hv hc hu (ih.liter h)

theorem sound_succ {fuel : Nat} (ih : STree.Sound fuel) : STree.Sound (fuel + 1) where
  node h := STree.execNode_succ ih h
  list h := STree.execList_succ ih h
  iter h := STree.execIter_succ ih h
  consume h := STree.execConsume_succ ih h
  loop h := STree.execLoop_succ ih h
  guard h := STree.loopGuard_succ ih h
  liter h := STree.loopIter_succ ih h
  tail h := STree.loopTail_succ ih h

/-! ## All fuel values -/

theorem sound : ∀ fuel : Nat, STree.Sound fuel
  | 0 => STree.sound_zero
  | fuel + 1 => STree.sound_succ (STree.sound fuel)

theorem execNode_sound (fuel : Nat) {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut}
    (h : STree.execNode fuel tc sc n cst = some r) : STree.NodeExec tc sc n cst r :=
  (STree.sound fuel).node h

theorem execList_sound (fuel : Nat) {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {nodes : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.execList fuel tc full gb sc nodes cst = some r) :
    STree.ListExec tc full gb sc nodes cst r :=
  (STree.sound fuel).list h

theorem execIter_sound (fuel : Nat) {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name}
    {pre : CSt} {pts : List Idx} {r : Res (List Log)}
    (h : STree.execIter fuel tc params body priv pre pts = some r) : STree.IterExec tc params body priv pre pts r :=
  (STree.sound fuel).iter h

theorem execConsume_sound (fuel : Nat) {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode}
    {l : Loc} {np : Int} {priv : List Name} {pre : CSt} {q : List Val} {r : Res (List Log × List Val × List Val)}
    (h : STree.execConsume fuel tc ρ c d body l np priv pre q = some r) :
    STree.ConsumeTasks tc ρ c d body l np priv pre q r :=
  (STree.sound fuel).consume h

theorem execLoop_sound (fuel : Nat) {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (h : STree.execLoop fuel tc l body cst = some r) : STree.LoopExec tc l body cst r :=
  (STree.sound fuel).loop h

theorem loopGuard_sound (fuel : Nat) {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (h : STree.loopGuard fuel tc l body cst = some r) : STree.LoopGuard tc l body cst r :=
  (STree.sound fuel).guard h

theorem loopIter_sound (fuel : Nat) {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (h : STree.loopIter fuel tc l body cst = some r) : STree.LoopIter tc l body cst r :=
  (STree.sound fuel).liter h

theorem loopTail_sound (fuel : Nat) {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (h : STree.loopTail fuel tc l body cst = some r) : STree.LoopTail tc l body cst r :=
  (STree.sound fuel).tail h

/-! ## Program level -/

/-- Every outcome of `STree.run` is an outcome of the semantics `STree.Exec`. -/
theorem run_sound (I : Interp) (t : STree) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) {r : Res Mem}
    (h : STree.run I t ρ₀ m₀ fuel = some r) : STree.Exec I t ρ₀ m₀ r := by
  unfold STree.run at h
  cases hx : STree.execList fuel (TCtx.ofTree I t) t.children false {} t.children (rootStart t ρ₀ m₀) with
  | none => rw [hx] at h; cases h
  | some res =>
    rw [hx] at h
    exact Exists.intro res ⟨STree.execList_sound fuel hx, (Option.some.inj h).symm⟩

/-- A successful run is a successful execution. -/
theorem run_ok_sound (I : Interp) (t : STree) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) {m : Mem}
    (h : STree.run I t ρ₀ m₀ fuel = some (.ok m)) : STree.Exec I t ρ₀ m₀ (.ok m) :=
  STree.run_sound I t ρ₀ m₀ fuel h

end Dace.STree
