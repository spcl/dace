-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Interp

/-!
# Soundness of the SDFG interpreter

`Dace.SDFG.Interp` mirrors the relational semantics
`Dace.SDFG.Semantics.Exec` function by function.  This module proves that
every outcome the interpreter produces is derivable:
`execX fuel … = some r → XExec … r` for each of the sixteen functions,
culminating in `SDFG.run_sound`.

The functions are mutually recursive on the fuel, so the sixteen statements
are bundled into `Sound fuel` and proven together by induction on the fuel:
at fuel `0` every function returns `none`; at `fuel + 1` each function is
unfolded once, the case distinctions of its definition are replayed on the
hypothesis (`split at h`), the recursive calls are discharged by the
induction hypothesis and the matching constructor closes the goal.

The interpreter resolves the nondeterminism of the relation (topological
order, interstate edge, processing element) deterministically; the proofs
only need the witnesses it chose to satisfy the constructors' premises:
the computed order passes `State.isTopologicalOrder` (checked by
`execOrdered`), the first true edge is a member of the true edges, and the
processing element `0` lies in `[0, num_pes)`.
-/

set_option autoImplicit false

namespace Dace

/-- Soundness of every interpreter function at one fuel value. -/
structure Sound (fuel : Nat) : Prop where
  node : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St} {r : Res St},
    execNode fuel ctx ρ s n st = some r → NodeExec ctx ρ s n st r
  seq : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))},
    execSeq fuel ctx ρ s ids st = some r → SeqExec ctx ρ s ids st r
  ordered : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {all direct : List NodeId} {st : St} {r : Res St},
    execOrdered fuel ctx ρ s all direct st = some r → OrderedExec ctx ρ s all direct st r
  iter : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name}
    {pre : St} {pts : List Idx} {r : Res (List Log)},
    execIter fuel ctx ρ s entry params priv pre pts = some r → IterExec ctx ρ s entry params priv pre pts r
  consume : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
    {priv : List Name} {pre : St} {q : List Val} {r : Res (List Log × List Val × List Val)},
    execConsume fuel ctx ρ s entry c l np priv pre q = some r → ConsumeTasks ctx ρ s entry c l np priv pre q r
  comps : ∀ {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {comps : List (List NodeId)} {r : Res (List Log)},
    execComps fuel ctx ρ s m comps = some r → CompsExec ctx ρ s m comps r
  state : ∀ {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)},
    execState fuel ctx cst s = some r → StateExec ctx cst s r
  block : ∀ {ctx : Ctx} {cst : CSt} {b : Block} {r : Res (Ctl × CSt)},
    execBlock fuel ctx cst b = some r → BlockExec ctx cst b r
  regAt : ∀ {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)},
    regionAt fuel ctx rg cst bid = some r → RegionAt ctx rg cst bid r
  region : ∀ {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)},
    execRegion fuel ctx cst rg = some r → RegionExec ctx cst rg r
  loop : ∀ {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)},
    execLoop fuel ctx cst l = some r → LoopExec ctx cst l r
  guard : ∀ {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)},
    loopGuard fuel ctx cst l = some r → LoopGuard ctx cst l r
  liter : ∀ {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)},
    loopIter fuel ctx cst l = some r → LoopIter ctx cst l r
  tail : ∀ {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)},
    loopTail fuel ctx cst l = some r → LoopTail ctx cst l r
  cond : ∀ {ctx : Ctx} {cst : CSt} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)},
    execCond fuel ctx cst brs = some r → CondExec ctx cst brs r
  sdfg : ∀ {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG} {r : Res (Valuation × Mem × Log)},
    execSDFG fuel ctx ρ m g = some r → SDFGExec ctx ρ m g r

/-! ## Fuel `0`: no outcome -/

theorem sound_zero : Sound 0 where
  node h := by simp only [execNode] at h; cases h
  seq h := by simp only [execSeq] at h; cases h
  ordered h := by simp only [execOrdered] at h; cases h
  iter h := by simp only [execIter] at h; cases h
  consume h := by simp only [execConsume] at h; cases h
  comps h := by simp only [execComps] at h; cases h
  state h := by simp only [execState] at h; cases h
  block h := by simp only [execBlock] at h; cases h
  regAt h := by simp only [regionAt] at h; cases h
  region h := by simp only [execRegion] at h; cases h
  loop h := by simp only [execLoop] at h; cases h
  guard h := by simp only [loopGuard] at h; cases h
  liter h := by simp only [loopIter] at h; cases h
  tail h := by simp only [loopTail] at h; cases h
  cond h := by simp only [execCond] at h; cases h
  sdfg h := by simp only [execSDFG] at h; cases h

/-! ## Fuel `fuel + 1`: one step per function -/

private theorem bool_eq_false_of_not {b : Bool} (h : ¬ b = true) : b = false := by
  cases b <;> simp_all

theorem execNode_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode}
    {st : St} {r : Res St} (h : execNode (fuel + 1) ctx ρ s n st = some r) : NodeExec ctx ρ s n st r := by
  simp only [execNode] at h
  split at h
  next hk => cases h; exact .simple (by simp only [DNode.isSimple, hk])
  next hk => cases h; exact .simple (by simp only [DNode.isSimple, hk])
  next hk => cases h; exact .simple (by simp only [DNode.isSimple, hk])
  next hk => cases h; exact .simple (by simp only [DNode.isSimple, hk])
  next hk => cases h; exact .simple (by simp only [DNode.isSimple, hk])
  next mp hk =>
    split at h
    next e hs => cases h; exact .mapSetupErr hk hs
    next ρ' st₁ pts hs =>
      split at h
      · cases h
      next e hi => cases h; exact .mapIterErr hk hs (ih.iter hi)
      next logs hi => cases h; exact .mapScope hk hs (ih.iter hi)
  next c hk =>
    split at h
    next e hs => cases h; exact .consumeSetupErr hk hs
    next ρ' st₁ l np hs =>
      split at h
      · cases h
      next e ht => cases h; exact .consumeTasksErr hk hs (ih.consume ht)
      next logs rest pushed ht => cases h; exact .consumeScope hk hs (ih.consume ht)
  next inner mapping noInline hk =>
    split at h
    next e hs => cases h; exact .nestedSetupErr hk hs
    next ctx' ρ' st₁ aliased hs =>
      split at h
      · cases h
      next e hr => cases h; exact .nestedInnerErr hk hs (ih.sdfg hr)
      next ρ₂ m₂ g₂ hr => cases h; exact .nested hk hs (ih.sdfg hr)

theorem execSeq_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId}
    {st : St} {r : Res (St × List (NodeId × Log))} (h : execSeq (fuel + 1) ctx ρ s ids st = some r) :
    SeqExec ctx ρ s ids st r := by
  simp only [execSeq] at h
  split at h
  · cases h; exact .nil
  · split at h
    next hn => cases h; exact .missing hn
    next n hn =>
      split at h
      · cases h
      next e he => cases h; exact .headErr hn (ih.node he)
      next st' he =>
        split at h
        · cases h
        next e hr => cases h; exact .tailErr hn (ih.node he) (ih.seq hr)
        next st₂ tr hr => cases h; exact .cons hn (ih.node he) (ih.seq hr)

theorem execOrdered_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State}
    {all direct : List NodeId} {st : St} {r : Res St}
    (h : execOrdered (fuel + 1) ctx ρ s all direct st = some r) : OrderedExec ctx ρ s all direct st r := by
  simp only [execOrdered] at h
  split at h
  next e hp => cases h; exact .prologueErr hp
  next st₁ hp =>
    split at h
    · cases h
    next order _ =>
      split at h
      next ho =>
        split at h
        · cases h
        next e hs => cases h; exact .seqErr ho hp (ih.seq hs)
        next st₂ tr hs =>
          split at h
          next hf => cases h; exact .run ho hp (ih.seq hs) hf
          next hf => cases h; exact .hbRace ho hp (ih.seq hs) (bool_eq_false_of_not hf)
      · cases h

theorem execIter_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId}
    {params : List Sym} {priv : List Name} {pre : St} {pts : List Idx} {r : Res (List Log)}
    (h : execIter (fuel + 1) ctx ρ s entry params priv pre pts = some r) :
    IterExec ctx ρ s entry params priv pre pts r := by
  simp only [execIter] at h
  split at h
  · cases h; exact .nil
  · split at h
    · cases h
    next e hb => cases h; exact .headErr (ih.ordered hb)
    next st hb =>
      split at h
      · cases h
      next e hr => cases h; exact .tailErr (ih.ordered hb) (ih.iter hr)
      next logs hr => cases h; exact .cons (ih.ordered hb) (ih.iter hr)

theorem execConsume_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId}
    {c : ConsumeNode} {l : Loc} {np : Int} {priv : List Name} {pre : St} {q : List Val}
    {r : Res (List Log × List Val × List Val)}
    (h : execConsume (fuel + 1) ctx ρ s entry c l np priv pre q = some r) :
    ConsumeTasks ctx ρ s entry c l np priv pre q r := by
  simp only [execConsume] at h
  split at h
  next e hc => cases h; exact .condErr hc
  next hc => cases h; exact .stop hc
  next hc =>
    split at h
    · cases h
    next hq =>
      have hq' := bool_eq_false_of_not hq
      split at h
      next hp =>
        split at h
        · cases h
        next e hb => cases h; exact .bodyErr hc hq' hp (ih.ordered hb)
        next st hb =>
          split at h
          · cases h
          next e hr => cases h; exact .restErr hc hq' hp (ih.ordered hb) (ih.consume hr)
          next logs rest pushed hr =>
            cases h; exact .step hc hq' hp (ih.ordered hb) (ih.consume hr)
      · cases h

theorem execComps_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem}
    {comps : List (List NodeId)} {r : Res (List Log)}
    (h : execComps (fuel + 1) ctx ρ s m comps = some r) : CompsExec ctx ρ s m comps r := by
  simp only [execComps] at h
  split at h
  · cases h; exact .nil
  · split at h
    · cases h
    next e hc => cases h; exact .headErr (ih.ordered hc)
    next st hc =>
      split at h
      · cases h
      next e hr => cases h; exact .tailErr (ih.ordered hc) (ih.comps hr)
      next logs hr => cases h; exact .cons (ih.ordered hc) (ih.comps hr)

theorem execState_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)}
    (h : execState (fuel + 1) ctx cst s = some r) : StateExec ctx cst s r := by
  simp only [execState] at h
  split at h
  · cases h
  next e hc => cases h; exact .compsErr (ih.comps hc)
  next logs hc => cases h; exact .run (ih.comps hc)

theorem execBlock_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {b : Block} {r : Res (Ctl × CSt)}
    (h : execBlock (fuel + 1) ctx cst b = some r) : BlockExec ctx cst b r := by
  simp only [execBlock] at h
  split at h
  · exact .state (ih.state h)
  · exact .region (ih.region h)
  · exact .loop (ih.loop h)
  · exact .cond (ih.cond h)
  · cases h; exact .brk
  · cases h; exact .cont
  · cases h; exact .ret

theorem regionAt_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId}
    {r : Res (Ctl × CSt)} (h : regionAt (fuel + 1) ctx rg cst bid = some r) : RegionAt ctx rg cst bid r := by
  simp only [regionAt] at h
  split at h
  next hb => cases h; exact .missing hb
  next b hb =>
    split at h
    · cases h
    next e he => cases h; exact .blockErr hb (ih.block he)
    next cst₁ he => cases h; exact .exit hb (ih.block he) nofun
    next cst₁ he => cases h; exact .exit hb (ih.block he) nofun
    next cst₁ he => cases h; exact .exit hb (ih.block he) nofun
    next cst₁ he =>
      split at h
      next e ht => cases h; exact .edgesErr hb (ih.block he) ht
      next ht => cases h; exact .stop hb (ih.block he) ht
      next ie tl ht =>
        split at h
        next e ha => cases h; exact .assignErr hb (ih.block he) ht (List.mem_cons_self ..) ha
        next cst₂ ha => exact .step hb (ih.block he) ht (List.mem_cons_self ..) ha (ih.regAt h)

theorem execRegion_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)}
    (h : execRegion (fuel + 1) ctx cst rg = some r) : RegionExec ctx cst rg r := by
  simp only [execRegion] at h
  exact .start (ih.regAt h)

theorem execLoop_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : execLoop (fuel + 1) ctx cst l = some r) : LoopExec ctx cst l r := by
  simp only [execLoop] at h
  split at h
  next e hi => cases h; exact .initErr hi
  next cst₁ hi =>
    split at h
    next hv => exact .guardFirst hi hv (ih.guard h)
    next hv => exact .bodyFirst hi hv (ih.liter h)

theorem loopGuard_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopGuard (fuel + 1) ctx cst l = some r) : LoopGuard ctx cst l r := by
  simp only [loopGuard] at h
  split at h
  next e hc => cases h; exact .condErr hc
  next hc => cases h; exact .exit hc
  next hc => exact .enter hc (ih.liter h)

theorem loopIter_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopIter (fuel + 1) ctx cst l = some r) : LoopIter ctx cst l r := by
  simp only [loopIter] at h
  split at h
  · cases h
  next e hb => cases h; exact .bodyErr (ih.region hb)
  next cst₁ hb => cases h; exact .ret (ih.region hb)
  next cst₁ hb => cases h; exact .brk (ih.region hb)
  next cst₁ hb => exact .tail (ih.region hb) rfl (ih.tail h)
  next cst₁ hb => exact .tail (ih.region hb) rfl (ih.tail h)

theorem loopTail_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopTail (fuel + 1) ctx cst l = some r) : LoopTail ctx cst l r := by
  simp only [loopTail] at h
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

theorem execCond_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {cst : CSt} {brs : List (Option Expr × Region)}
    {r : Res (Ctl × CSt)} (h : execCond (fuel + 1) ctx cst brs = some r) : CondExec ctx cst brs r := by
  simp only [execCond] at h
  split at h
  · cases h; exact .nil
  · exact .els (ih.region h)
  · split at h
    next e hc => cases h; exact .condErr hc
    next hc => exact .taken hc (ih.region h)
    next hc => exact .skipped hc (ih.cond h)

theorem execSDFG_succ {fuel : Nat} (ih : Sound fuel) {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG}
    {r : Res (Valuation × Mem × Log)} (h : execSDFG (fuel + 1) ctx ρ m g = some r) : SDFGExec ctx ρ m g r := by
  simp only [execSDFG] at h
  split at h
  · cases h
  next e hr => cases h; exact .err (ih.region hr)
  next cst hr => cases h; exact .escape (ih.region hr) rfl
  next cst hr => cases h; exact .escape (ih.region hr) rfl
  next cst hr => cases h; exact .run (ih.region hr) rfl
  next cst hr => cases h; exact .run (ih.region hr) rfl

theorem sound_succ {fuel : Nat} (ih : Sound fuel) : Sound (fuel + 1) where
  node h := execNode_succ ih h
  seq h := execSeq_succ ih h
  ordered h := execOrdered_succ ih h
  iter h := execIter_succ ih h
  consume h := execConsume_succ ih h
  comps h := execComps_succ ih h
  state h := execState_succ ih h
  block h := execBlock_succ ih h
  regAt h := regionAt_succ ih h
  region h := execRegion_succ ih h
  loop h := execLoop_succ ih h
  guard h := loopGuard_succ ih h
  liter h := loopIter_succ ih h
  tail h := loopTail_succ ih h
  cond h := execCond_succ ih h
  sdfg h := execSDFG_succ ih h

/-! ## All fuel values -/

theorem sound : ∀ fuel : Nat, Sound fuel
  | 0 => sound_zero
  | fuel + 1 => sound_succ (sound fuel)

theorem execNode_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {st : St} {r : Res St}
    (h : execNode fuel ctx ρ s n st = some r) : NodeExec ctx ρ s n st r :=
  (sound fuel).node h

theorem execSeq_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} {st : St}
    {r : Res (St × List (NodeId × Log))} (h : execSeq fuel ctx ρ s ids st = some r) : SeqExec ctx ρ s ids st r :=
  (sound fuel).seq h

theorem execOrdered_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {all direct : List NodeId}
    {st : St} {r : Res St} (h : execOrdered fuel ctx ρ s all direct st = some r) :
    OrderedExec ctx ρ s all direct st r :=
  (sound fuel).ordered h

theorem execIter_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym}
    {priv : List Name} {pre : St} {pts : List Idx} {r : Res (List Log)}
    (h : execIter fuel ctx ρ s entry params priv pre pts = some r) : IterExec ctx ρ s entry params priv pre pts r :=
  (sound fuel).iter h

theorem execConsume_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {entry : NodeId} {c : ConsumeNode}
    {l : Loc} {np : Int} {priv : List Name} {pre : St} {q : List Val} {r : Res (List Log × List Val × List Val)}
    (h : execConsume fuel ctx ρ s entry c l np priv pre q = some r) :
    ConsumeTasks ctx ρ s entry c l np priv pre q r :=
  (sound fuel).consume h

theorem execComps_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {comps : List (List NodeId)}
    {r : Res (List Log)} (h : execComps fuel ctx ρ s m comps = some r) : CompsExec ctx ρ s m comps r :=
  (sound fuel).comps h

theorem execState_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)}
    (h : execState fuel ctx cst s = some r) : StateExec ctx cst s r :=
  (sound fuel).state h

theorem execBlock_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {b : Block} {r : Res (Ctl × CSt)}
    (h : execBlock fuel ctx cst b = some r) : BlockExec ctx cst b r :=
  (sound fuel).block h

theorem regionAt_sound (fuel : Nat) {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)}
    (h : regionAt fuel ctx rg cst bid = some r) : RegionAt ctx rg cst bid r :=
  (sound fuel).regAt h

theorem execRegion_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)}
    (h : execRegion fuel ctx cst rg = some r) : RegionExec ctx cst rg r :=
  (sound fuel).region h

theorem execLoop_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : execLoop fuel ctx cst l = some r) : LoopExec ctx cst l r :=
  (sound fuel).loop h

theorem loopGuard_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopGuard fuel ctx cst l = some r) : LoopGuard ctx cst l r :=
  (sound fuel).guard h

theorem loopIter_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopIter fuel ctx cst l = some r) : LoopIter ctx cst l r :=
  (sound fuel).liter h

theorem loopTail_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)}
    (h : loopTail fuel ctx cst l = some r) : LoopTail ctx cst l r :=
  (sound fuel).tail h

theorem execCond_sound (fuel : Nat) {ctx : Ctx} {cst : CSt} {brs : List (Option Expr × Region)}
    {r : Res (Ctl × CSt)} (h : execCond fuel ctx cst brs = some r) : CondExec ctx cst brs r :=
  (sound fuel).cond h

theorem execSDFG_sound (fuel : Nat) {ctx : Ctx} {ρ : Valuation} {m : Mem} {g : SDFG}
    {r : Res (Valuation × Mem × Log)} (h : execSDFG fuel ctx ρ m g = some r) : SDFGExec ctx ρ m g r :=
  (sound fuel).sdfg h

/-! ## Program level -/

/-- Every outcome of `SDFG.run` is an outcome of the semantics `SDFG.Exec`. -/
theorem SDFG.run_sound (I : Interp) (g : SDFG) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) {r : Res Mem}
    (h : SDFG.run I g ρ₀ m₀ fuel = some r) : SDFG.Exec I g ρ₀ m₀ r := by
  unfold SDFG.run at h
  cases hx : execSDFG fuel { I, g, pfx := "" } ρ₀ m₀ g with
  | none => rw [hx] at h; cases h
  | some res =>
    rw [hx] at h
    exact Exists.intro res ⟨execSDFG_sound fuel hx, (Option.some.inj h).symm⟩

/-- A successful run is a successful execution. -/
theorem SDFG.run_ok_sound (I : Interp) (g : SDFG) (ρ₀ : Valuation) (m₀ : Mem) (fuel : Nat) {m : Mem}
    (h : SDFG.run I g ρ₀ m₀ fuel = some (.ok m)) : SDFG.Exec I g ρ₀ m₀ (.ok m) :=
  SDFG.run_sound I g ρ₀ m₀ fuel h

end Dace
