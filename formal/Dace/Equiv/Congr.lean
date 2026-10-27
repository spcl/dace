-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.LogShift

/-!
# Congruence: replacing a control-flow block

The first congruence principle of the equivalence framework: replacing one control-flow block of an SDFG (a state, a loop, a
conditional, a sub-region, at any depth of the control-flow hierarchy) by a
block whose runs it matches yields a refinement of the whole SDFG.

## Context congruence (`ctx_congr`)

The execution relations depend on the context `ctx` only through

* `ctx.erase`: the interpretation `ctx.I`, the container-name prefix
  memory names `ctx.cname`, `ctx.scope` and the descriptor repository `ctx.g.arrays` (every leaf function
  of `Dace.SDFG.Semantics.Leaf` sees nothing else: `f ctx = f ctx.erase`
  holds by `rfl`, `Ctx.Agree.execSimple` etc.), and
* `Ctx.privateTransients`, which counts the mentions of each transient over
  the whole SDFG (`SDFG.countMentions`).

Constants, symbols, facts, the SDFG name and the control-flow hierarchy
`ctx.g.top` are not read by any relation (constants enter through the
`SDFG` argument of `SDFGExec`; nested invocations build their own context
from `ctx.I` and `ctx.scope` alone, `Ctx.enter`).  `Ctx.Agree ctx' ctx` states
the two conditions; under it every relation holds in `ctx'` iff it holds in
`ctx` (`NodeExec.congr`, …, `SDFGExec.congr`, `BlockExec.congr_iff`).

## Paths and replacement

A block is addressed by a list of `PathStep`s, each descending from a region
into a sub-region of one of its blocks (a `.region` block, a loop body, a
branch of a conditional), followed by the index of the block in the region
reached (`Region.blockAt?`, `Region.setBlockAt`, `SDFG.setBlockAt`).  Paths
stop at nested SDFGs (v1).

## Private transients

Replacing a block changes the mention counts of the SDFG, hence possibly the
private transients of every scope (`Decisions.md#D39`).
`sameTransientMentions? g b' b` checks that every transient of `g` with
`Scope` lifetime is mentioned equally often by both blocks; then every
mention count of the SDFG is unchanged (`Region.countMentions_setBlockAt`),
so `privateTransients` agrees on **every** scope (inside the block or not)
and the contexts of the old and the new SDFG `Agree`
(`Ctx.agree_setBlockAt`).  This is the weakest condition that is local to
the two blocks: a differing count of a transient `t` leaves the private sets
unchanged only if no scope of the SDFG has all mentions of `t` before or
after the replacement, a whole-SDFG property.

## Matching up to equal errors

("Matching" is a simulation relation over big-step outcomes; see the
terminology note in `Dace.Equiv.Frag`.)

`MatchedE R X r'`: some outcome `r` of `X` approximates `r'` (`Res.Below`): `r`
is undefined behavior, or `r'` and `r` are `R`-related normal outcomes or
the *same* error.  Unlike `Matched`, this relation is reflexive (an unchanged
block matches itself even when it may end in `malformed` or
`unsupported`), so it is the motive of the congruence proofs.  At the end,
`Supported` (the old program has no non-UB error unless it may be undefined)
turns `MatchedE` into `Refines` (`Refines.of_matchedE`).

## Main results

* `Congr.region_setBlockAt`: in one context, if the new block is matched by
  the old one (`BlockMatchedE`), the region with the block replaced is matched
  by the original region.  Proof: induction on the path; at each level, a
  recursion on the derivation of the new region (`RegionAt.matchedE`), loop
  (`LoopGuard.matchedE`, `LoopIter.matchedE`, `LoopTail.matchedE`) or conditional
  (`CondExec.matchedE`), with the motive "matched by the old relation from a
  `CSt.Rel`-related start"; unchanged blocks match themselves by the
  log-shift lemmas (`BlockExec.rel`).
* `Congr.block`: `Refines (g.setBlockAt p i b').prog g.prog` if the block at
  the path is `b`, `sameTransientMentions? g b' b`, every run of `b'` is
  matched (`MatchedE`) by a run of `b` from the same state in the context of `g`
  (for every lawful interpretation), and `g.prog` is `Supported`.
* `Congr.blockEquiv`: both directions give `Equiv` (the new SDFG is then
  supported as well).
* Instances: `Congr.emptyState` (any state without nodes may replace any
  other, e.g. a renamed one), `Congr.splitEmpty` / `Congr.mergeEmpty` (a
  state is equivalent to the sub-region "the state, then an empty state").
  Merging two states of the *same* region needs region-level rewiring
  (inlining a sub-region into its parent), which is not a block replacement
  and not covered here.
-/

set_option autoImplicit false

namespace Dace

/-! ## Context congruence -/

namespace Ctx

/-- The part of a context the leaf functions read: interpretation, prefix and
descriptor repository. -/
def erase (ctx : Ctx) : Ctx :=
  { I := ctx.I, g := { name := "", arrays := ctx.g.arrays, top := {} }, cname := ctx.cname, scope := ctx.scope }

theorem erase_eq_of {ctx' ctx : Ctx} (hI : ctx'.I = ctx.I) (hc : ctx'.cname = ctx.cname) (hs : ctx'.scope = ctx.scope)
    (ha : ctx'.g.arrays = ctx.g.arrays) : ctx'.erase = ctx.erase := by
  simp only [erase, hI, hc, hs, ha]

/-- Two contexts in which the execution relations coincide (`ctx_congr`):
same interpretation, prefix and descriptors, and the same private transients
for every scope of every state. -/
structure Agree (ctx' ctx : Ctx) : Prop where
  erase : ctx'.erase = ctx.erase
  priv : ∀ (s : State) (entry : NodeId), ctx'.privateTransients s entry = ctx.privateTransients s entry
  stateT : ∀ s : State, ctx'.stateTransients s = ctx.stateTransients s

theorem Agree.refl (ctx : Ctx) : Agree ctx ctx := ⟨rfl, fun _ _ => rfl, fun _ => rfl⟩

theorem Agree.symm {ctx' ctx : Ctx} (h : Agree ctx' ctx) : Agree ctx ctx' :=
  ⟨h.erase.symm, fun s e => (h.priv s e).symm, fun s => (h.stateT s).symm⟩

theorem Agree.trans {a b c : Ctx} (h₁ : Agree a b) (h₂ : Agree b c) : Agree a c :=
  ⟨h₁.erase.trans h₂.erase, fun s e => (h₁.priv s e).trans (h₂.priv s e), fun s => (h₁.stateT s).trans (h₂.stateT s)⟩


/-- A function of the context that only reads its erased part agrees on
agreeing contexts. -/
theorem Agree.leaf {α : Sort _} {ctx' ctx : Ctx} (h : Agree ctx' ctx) (f : Ctx → α)
    (hf : ∀ c, f c = f c.erase) : f ctx' = f ctx := by
  rw [hf ctx', h.erase, ← hf ctx]

theorem Agree.allocate {ctx' ctx : Ctx} (h : Agree ctx' ctx) (s : State) (m : Mem) :
    ctx'.allocate (ctx'.stateTransients s) m = ctx.allocate (ctx.stateTransients s) m := by
  rw [h.stateT s]
  exact h.leaf (fun c => c.allocate (ctx.stateTransients s) m) (fun _ => rfl)

variable {ctx' ctx : Ctx}

theorem Agree.execSimple (h : Agree ctx' ctx) : Dace.execSimple ctx' = Dace.execSimple ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.mapSetup (h : Agree ctx' ctx) : Dace.mapSetup ctx' = Dace.mapSetup ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.scopeFinish (h : Agree ctx' ctx) : Dace.scopeFinish ctx' = Dace.scopeFinish ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.consumeSetup (h : Agree ctx' ctx) : Dace.consumeSetup ctx' = Dace.consumeSetup ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.consumeCond (h : Agree ctx' ctx) : Dace.consumeCond ctx' = Dace.consumeCond ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.taskStart (h : Agree ctx' ctx) : Dace.taskStart ctx' = Dace.taskStart ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.consumeFinish (h : Agree ctx' ctx) : Dace.consumeFinish ctx' = Dace.consumeFinish ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.nestedSetup (h : Agree ctx' ctx) : Dace.nestedSetup ctx' = Dace.nestedSetup ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.bindViews (h : Agree ctx' ctx) : Dace.bindViews ctx' = Dace.bindViews ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.viewOrder (h : Agree ctx' ctx) (s : State) (ids : List NodeId) :
    s.viewOrder ctx' ids = s.viewOrder ctx ids :=
  h.leaf (fun c => s.viewOrder c ids) fun _ => rfl
theorem Agree.stateFinish (h : Agree ctx' ctx) : Dace.stateFinish ctx' = Dace.stateFinish ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.trueEdges (h : Agree ctx' ctx) : Dace.trueEdges ctx' = Dace.trueEdges ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.applyAssigns (h : Agree ctx' ctx) : Dace.applyAssigns ctx' = Dace.applyAssigns ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.applyAssign? (h : Agree ctx' ctx) : Dace.applyAssign? ctx' = Dace.applyAssign? ctx :=
  h.leaf _ fun _ => rfl
theorem Agree.evalBoolR (h : Agree ctx' ctx) : ctx'.evalBoolR = ctx.evalBoolR :=
  h.leaf Ctx.evalBoolR fun _ => rfl

end Ctx

section CtxCongr

open Ctx (Agree)

mutual

theorem NodeExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State} {n : DNode} {st : St}
    {r : Res St} : NodeExec ctx' ρ s n st r → NodeExec ctx ρ s n st r
  | .simple hs => by rw [h.execSimple]; exact .simple hs
  | .mapSetupErr hk hs => .mapSetupErr hk (by rw [← h.mapSetup]; exact hs)
  | .mapIterErr hk hs hi =>
    .mapIterErr hk (by rw [← h.mapSetup]; exact hs) (by rw [← h.priv]; exact IterExec.congr h hi)
  | .mapScope hk hs hi => by
    rw [h.scopeFinish]
    exact .mapScope hk (by rw [← h.mapSetup]; exact hs) (by rw [← h.priv]; exact IterExec.congr h hi)
  | .consumeSetupErr hk hs => .consumeSetupErr hk (by rw [← h.consumeSetup]; exact hs)
  | .consumeTasksErr hk hs ht =>
    .consumeTasksErr hk (by rw [← h.consumeSetup]; exact hs) (by rw [← h.priv]; exact ConsumeTasks.congr h ht)
  | .consumeScope hk hs ht => by
    rw [h.consumeFinish]
    exact .consumeScope hk (by rw [← h.consumeSetup]; exact hs) (by rw [← h.priv]; exact ConsumeTasks.congr h ht)
  | .nestedSetupErr hk hs => .nestedSetupErr hk (by rw [← h.nestedSetup]; exact hs)
  | .nestedInnerErr hk hs hr => .nestedInnerErr hk (by rw [← h.nestedSetup]; exact hs) hr
  | .nested hk hs hr => .nested hk (by rw [← h.nestedSetup]; exact hs) hr

theorem SeqExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State} {ids : List NodeId}
    {st : St} {r : Res (St × List (NodeId × Log))} : SeqExec ctx' ρ s ids st r → SeqExec ctx ρ s ids st r
  | .nil => .nil
  | .missing hm => .missing hm
  | .headErr hm hn => .headErr hm (NodeExec.congr h hn)
  | .tailErr hm hn hr => .tailErr hm (NodeExec.congr h hn) (SeqExec.congr h hr)
  | .cons hm hn hr => .cons hm (NodeExec.congr h hn) (SeqExec.congr h hr)

theorem OrderedExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State}
    {all direct : List NodeId} {st : St} {r : Res St} :
    OrderedExec ctx' ρ s all direct st r → OrderedExec ctx ρ s all direct st r
  | .prologueErr hp => .prologueErr (by rw [← h.bindViews, ← h.viewOrder]; exact hp)
  | .seqErr ho hp hs => .seqErr ho (by rw [← h.bindViews, ← h.viewOrder]; exact hp) (SeqExec.congr h hs)
  | .hbRace ho hp hs hf => .hbRace ho (by rw [← h.bindViews, ← h.viewOrder]; exact hp) (SeqExec.congr h hs) hf
  | .run ho hp hs hf => .run ho (by rw [← h.bindViews, ← h.viewOrder]; exact hp) (SeqExec.congr h hs) hf

theorem IterExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State} {entry : NodeId}
    {params : List Sym} {priv : List Name} {pre : St} {pts : List Idx} {r : Res (List Log)} :
    IterExec ctx' ρ s entry params priv pre pts r → IterExec ctx ρ s entry params priv pre pts r
  | .nil => .nil
  | .headErr hb => .headErr (OrderedExec.congr h hb)
  | .tailErr hr => .tailErr (IterExec.congr h hr)
  | .cons hb hr => .cons (OrderedExec.congr h hb) (IterExec.congr h hr)

theorem ConsumeTasks.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State} {entry : NodeId}
    {c : ConsumeNode} {l : Loc} {np : Int} {priv : List Name} {pre : St} {q : List Val}
    {r : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx' ρ s entry c l np priv pre q r → ConsumeTasks ctx ρ s entry c l np priv pre q r
  | .condErr hc => .condErr (by rw [← h.consumeCond]; exact hc)
  | .stop hc => .stop (by rw [← h.consumeCond]; exact hc)
  | .bodyErr hc hq hp hb =>
    .bodyErr (by rw [← h.consumeCond]; exact hc) hq hp (by rw [← h.taskStart]; exact OrderedExec.congr h hb)
  | .restErr hc hq hp hb hr =>
    .restErr (by rw [← h.consumeCond]; exact hc) hq hp (by rw [← h.taskStart]; exact OrderedExec.congr h hb)
      (ConsumeTasks.congr h hr)
  | .step hc hq hp hb hr =>
    .step (by rw [← h.consumeCond]; exact hc) hq hp (by rw [← h.taskStart]; exact OrderedExec.congr h hb)
      (ConsumeTasks.congr h hr)

end

theorem CompsExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {s : State} {m : Mem}
    {comps : List (List NodeId)} {r : Res (List Log)} : CompsExec ctx' ρ s m comps r → CompsExec ctx ρ s m comps r
  | .nil => .nil
  | .headErr hc => .headErr (OrderedExec.congr h hc)
  | .tailErr hr => .tailErr (CompsExec.congr h hr)
  | .cons hc hr => .cons (OrderedExec.congr h hc) (CompsExec.congr h hr)

theorem StateExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {s : State} {r : Res (Ctl × CSt)} :
    StateExec ctx' cst s r → StateExec ctx cst s r
  | .compsErr hc => .compsErr (h.allocate s _ ▸ CompsExec.congr h hc)
  | .run hc => by rw [h.stateFinish]; exact .run (h.allocate s _ ▸ CompsExec.congr h hc)

mutual

theorem BlockExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} :
    BlockExec ctx' cst b r → BlockExec ctx cst b r
  | .state hs => .state (StateExec.congr h hs)
  | .region hr => .region (RegionExec.congr h hr)
  | .loop hl => .loop (LoopExec.congr h hl)
  | .cond hc => .cond (CondExec.congr h hc)
  | .brk => .brk
  | .cont => .cont
  | .ret => .ret

theorem RegionAt.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {rg : Region} {cst : CSt} {bid : BlockId}
    {r : Res (Ctl × CSt)} : RegionAt ctx' rg cst bid r → RegionAt ctx rg cst bid r
  | .missing hb => .missing hb
  | .blockErr hb hB => .blockErr hb (BlockExec.congr h hB)
  | .exit hb hB hc => .exit hb (BlockExec.congr h hB) hc
  | .edgesErr hb hB he => .edgesErr hb (BlockExec.congr h hB) (by rw [← h.trueEdges]; exact he)
  | .stop hb hB he => .stop hb (BlockExec.congr h hB) (by rw [← h.trueEdges]; exact he)
  | .ambiguous hb hB he => .ambiguous hb (BlockExec.congr h hB) (by rw [← h.trueEdges]; exact he)
  | .assignErr hb hB he ha =>
    .assignErr hb (BlockExec.congr h hB) (by rw [← h.trueEdges]; exact he) (by rw [← h.applyAssigns]; exact ha)
  | .step hb hB he ha hn =>
    .step hb (BlockExec.congr h hB) (by rw [← h.trueEdges]; exact he) (by rw [← h.applyAssigns]; exact ha)
      (RegionAt.congr h hn)

theorem RegionExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)} :
    RegionExec ctx' cst rg r → RegionExec ctx cst rg r
  | .start hr => .start (RegionAt.congr h hr)

theorem LoopExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopExec ctx' cst l r → LoopExec ctx cst l r
  | .initErr hi => .initErr (by rw [← h.applyAssign?]; exact hi)
  | .guardFirst hi hv hg => .guardFirst (by rw [← h.applyAssign?]; exact hi) hv (LoopGuard.congr h hg)
  | .bodyFirst hi hv hb => .bodyFirst (by rw [← h.applyAssign?]; exact hi) hv (LoopIter.congr h hb)

theorem LoopGuard.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopGuard ctx' cst l r → LoopGuard ctx cst l r
  | .condErr hc => .condErr (by rw [← h.evalBoolR]; exact hc)
  | .exit hc => .exit (by rw [← h.evalBoolR]; exact hc)
  | .enter hc hb => .enter (by rw [← h.evalBoolR]; exact hc) (LoopIter.congr h hb)

theorem LoopIter.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopIter ctx' cst l r → LoopIter ctx cst l r
  | .bodyErr hb => .bodyErr (RegionExec.congr h hb)
  | .ret hb => .ret (RegionExec.congr h hb)
  | .brk hb => .brk (RegionExec.congr h hb)
  | .tail hb hc ht => .tail (RegionExec.congr h hb) hc (LoopTail.congr h ht)

theorem LoopTail.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopTail ctx' cst l r → LoopTail ctx cst l r
  | .updateErr hv hu => .updateErr hv (by rw [← h.applyAssign?]; exact hu)
  | .updateGuard hv hu hg => .updateGuard hv (by rw [← h.applyAssign?]; exact hu) (LoopGuard.congr h hg)
  | .condFirstErr hv hc => .condFirstErr hv (by rw [← h.evalBoolR]; exact hc)
  | .condFirstExit hv hc => .condFirstExit hv (by rw [← h.evalBoolR]; exact hc)
  | .condFirstUpdateErr hv hc hu =>
    .condFirstUpdateErr hv (by rw [← h.evalBoolR]; exact hc) (by rw [← h.applyAssign?]; exact hu)
  | .condFirstNext hv hc hu hb =>
    .condFirstNext hv (by rw [← h.evalBoolR]; exact hc) (by rw [← h.applyAssign?]; exact hu) (LoopIter.congr h hb)

theorem CondExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {brs : List (Option Expr × Region)}
    {r : Res (Ctl × CSt)} : CondExec ctx' cst brs r → CondExec ctx cst brs r
  | .nil => .nil
  | .els hr => .els (RegionExec.congr h hr)
  | .condErr hc => .condErr (by rw [← h.evalBoolR]; exact hc)
  | .taken hc hr => .taken (by rw [← h.evalBoolR]; exact hc) (RegionExec.congr h hr)
  | .skipped hc hr => .skipped (by rw [← h.evalBoolR]; exact hc) (CondExec.congr h hr)

end

theorem SDFGExec.congr {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {m : Mem} {g : SDFG}
    {r : Res (Valuation × Mem × Log)} : SDFGExec ctx' ρ m g r → SDFGExec ctx ρ m g r
  | .err hr => .err (RegionExec.congr h hr)
  | .escape hr hc => .escape (RegionExec.congr h hr) hc
  | .run hr hc => .run (RegionExec.congr h hr) hc

/-- **Context congruence** (control flow): agreeing contexts have the same
block executions; likewise `RegionExec.congr_iff`, `StateExec.congr_iff`,
`SDFGExec.congr_iff`, and one direction for every other relation
(`NodeExec.congr`, …), the other following from `Agree.symm`. -/
theorem BlockExec.congr_iff {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} :
    BlockExec ctx' cst b r ↔ BlockExec ctx cst b r :=
  ⟨BlockExec.congr h, BlockExec.congr h.symm⟩

theorem RegionExec.congr_iff {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {rg : Region}
    {r : Res (Ctl × CSt)} : RegionExec ctx' cst rg r ↔ RegionExec ctx cst rg r :=
  ⟨RegionExec.congr h, RegionExec.congr h.symm⟩

theorem StateExec.congr_iff {ctx' ctx : Ctx} (h : Agree ctx' ctx) {cst : CSt} {s : State} {r : Res (Ctl × CSt)} :
    StateExec ctx' cst s r ↔ StateExec ctx cst s r :=
  ⟨StateExec.congr h, StateExec.congr h.symm⟩

theorem SDFGExec.congr_iff {ctx' ctx : Ctx} (h : Agree ctx' ctx) {ρ : Valuation} {m : Mem} {g : SDFG}
    {r : Res (Valuation × Mem × Log)} : SDFGExec ctx' ρ m g r ↔ SDFGExec ctx ρ m g r :=
  ⟨SDFGExec.congr h, SDFGExec.congr h.symm⟩

end CtxCongr

/-! ## Matching up to equal errors -/

namespace Res

/-- `r'` is approximated by `r`: `r` is undefined behavior, or both are
`R`-related normal outcomes or the same error. -/
def Below {α : Type} (R : α → α → Prop) (r' r : Res α) : Prop :=
  (∃ e, r = .err e ∧ e.isUB = true) ∨ Rel R r' r

variable {α : Type} {R : α → α → Prop}

theorem Rel.refl (hR : ∀ a, R a a) : ∀ r : Res α, Rel R r r
  | .ok a => hR a
  | .err _ => rfl

theorem Rel.trans (hR : ∀ a b c, R a b → R b c → R a c) :
    ∀ {r₁ r₂ r₃ : Res α}, Rel R r₁ r₂ → Rel R r₂ r₃ → Rel R r₁ r₃
  | .ok _, .ok _, .ok _, h₁, h₂ => hR _ _ _ h₁ h₂
  | .err _, .err _, .err _, h₁, h₂ => Eq.trans h₁ h₂
  | .ok _, .err _, _, h₁, _ => (h₁ : False).elim
  | .err _, .ok _, _, h₁, _ => (h₁ : False).elim
  | .ok _, .ok _, .err _, _, h₂ => (h₂ : False).elim
  | .err _, .err _, .ok _, _, h₂ => (h₂ : False).elim

theorem Rel.err_inv {e : ErrReason} {r : Res α} (h : Rel R (.err e) r) : r = .err e := by
  cases r with
  | ok _ => exact (h : False).elim
  | err e₀ => exact congrArg Res.err (h : e = e₀).symm

theorem Rel.ok_inv {a' : α} {r : Res α} (h : Rel R (.ok a') r) : ∃ a, r = .ok a ∧ R a' a := by
  cases r with
  | ok a => exact ⟨a, rfl, h⟩
  | err _ => exact (h : False).elim

theorem Below.err_inv {e : ErrReason} {r : Res α} (h : Below R (.err e) r) : ∃ e₀, r = .err e₀ := by
  rcases h with ⟨e₀, rfl, -⟩ | h
  · exact ⟨_, rfl⟩
  · exact ⟨e, h.err_inv⟩

theorem Below.ok_inv {a' : α} {r : Res α} (h : Below R (.ok a') r) :
    (∃ e, r = .err e ∧ e.isUB = true) ∨ ∃ a, r = .ok a ∧ R a' a :=
  h.imp id Rel.ok_inv

/-- Approximation between errors does not depend on the relation. -/
theorem Below.err_err {β : Type} {S : β → β → Prop} {e e₀ : ErrReason} (h : Below R (.err e) (.err e₀)) :
    Below S (.err e) (.err e₀) := by
  rcases h with ⟨e₁, he, hu⟩ | h
  · exact .inl ⟨e₁, by cases he; rfl, hu⟩
  · exact .inr (h : e = e₀)

theorem Below.trans (hR : ∀ a b c, R a b → R b c → R a c) {r₁ r₂ r₃ : Res α}
    (h₁ : Below R r₁ r₂) (h₂ : Below R r₂ r₃) : Below R r₁ r₃ := by
  rcases h₂ with h₂ | h₂
  · exact .inl h₂
  · rcases h₁ with ⟨e, rfl, he⟩ | h₁
    · obtain rfl := h₂.err_inv
      exact .inl ⟨e, rfl, he⟩
    · exact .inr (Res.Rel.trans hR h₁ h₂)

end Res

/-- `r'` is matched by the outcome set `X` up to `R` and equal errors: some
outcome of `X` approximates it (`Res.Below`).  Weaker than `Matched` only in that
an error of `r'` may be matched by the same error of `X`. -/
def MatchedE {α : Type} (R : α → α → Prop) (X : Res α → Prop) (r' : Res α) : Prop :=
  ∃ r, X r ∧ Res.Below R r' r

namespace MatchedE

variable {α : Type} {R : α → α → Prop} {X Y : Res α → Prop}

theorem of_rel {r' r : Res α} (hX : X r) (h : Res.Rel R r' r) : MatchedE R X r' := ⟨r, hX, .inr h⟩

theorem of_ub {r' : Res α} {e : ErrReason} (hX : X (.err e)) (he : e.isUB = true) : MatchedE R X r' :=
  ⟨_, hX, .inl ⟨e, rfl, he⟩⟩

theorem of_matched {r' : Res α} (h : Matched R X r') : MatchedE R X r' := by
  rcases h with ⟨e, he, hx⟩ | ⟨a', rfl, a, hx, hr⟩
  · exact of_ub hx he
  · exact of_rel hx hr

/-- A `MatchedE` match is a `Matched` match unless `r'` is an error that `X`
has as well. -/
theorem matched_or_err {r' : Res α} (h : MatchedE R X r') : Matched R X r' ∨ ∃ e, r' = .err e ∧ X (.err e) := by
  obtain ⟨r, hx, hb⟩ := h
  rcases hb with ⟨e, rfl, he⟩ | hb
  · exact .inl (.inl ⟨e, he, hx⟩)
  · cases r' with
    | ok a' =>
      obtain ⟨a, rfl, ha⟩ := hb.ok_inv
      exact .inl (.inr ⟨a', rfl, a, hx, ha⟩)
    | err e =>
      obtain rfl := hb.err_inv
      exact .inr ⟨e, rfl, hx⟩

theorem mono {R' : α → α → Prop} (hR : ∀ a' a, R a' a → R' a' a) (hX : ∀ r, X r → Y r) {r' : Res α}
    (h : MatchedE R X r') : MatchedE R' Y r' := by
  obtain ⟨r, hx, hb⟩ := h
  refine ⟨r, hX r hx, hb.imp id ?_⟩
  cases r' <;> cases r <;> first | exact id | exact hR _ _

/-- Compose with a related outcome on the left. -/
theorem of_rel_left (hR : ∀ a b c, R a b → R b c → R a c) {r'' r' : Res α} (h₁ : Res.Rel R r'' r')
    (h₂ : MatchedE R X r') : MatchedE R X r'' :=
  let ⟨r, hx, hb⟩ := h₂
  ⟨r, hx, Res.Below.trans hR (.inr h₁) hb⟩

end MatchedE

/-- Refinement from `MatchedE` matching: a supported program has no error
outcome to match (unless it may be undefined). -/
theorem Refines.of_matchedE {Q P : Prog} (hs : Supported P)
    (h : ∀ I ρ₀ m₀, P.Pre I ρ₀ m₀ → ∀ r, Q.exec I ρ₀ m₀ r → MatchedE (obsEq P.interface) (P.exec I ρ₀ m₀) r) :
    Refines Q P := by
  intro I ρ₀ m₀ hp
  rcases hs I ρ₀ m₀ hp with hu | hne
  · exact .inl hu
  · refine .inr fun r hr => ?_
    obtain ⟨y, hy, hb⟩ := h I ρ₀ m₀ hp r hr
    rcases hb with ⟨e, rfl, -⟩ | hb
    · exact absurd hy (hne e)
    · cases y with
      | err e₀ => exact absurd hy (hne e₀)
      | ok m' =>
        cases r with
        | err _ => exact (hb : False).elim
        | ok m => exact ⟨m, rfl, m', hy, hb⟩

/-- Supportedness transfers along `MatchedE` matching in both directions. -/
theorem Supported.of_matchedE {Q P : Prog} {R R' : Mem → Mem → Prop} (hP : Supported P)
    (hpre : ∀ I ρ₀ m₀, Q.Pre I ρ₀ m₀ → P.Pre I ρ₀ m₀)
    (hQP : ∀ I ρ₀ m₀, Q.Pre I ρ₀ m₀ → ∀ r, Q.exec I ρ₀ m₀ r → MatchedE R (P.exec I ρ₀ m₀) r)
    (hPQ : ∀ I ρ₀ m₀, Q.Pre I ρ₀ m₀ → ∀ r, P.exec I ρ₀ m₀ r → MatchedE R' (Q.exec I ρ₀ m₀) r) : Supported Q := by
  intro I ρ₀ m₀ hq
  by_cases hu : mayUB Q I ρ₀ m₀
  · exact .inl hu
  · refine .inr fun e he => hu ?_
    obtain ⟨y, hy, hb⟩ := hQP I ρ₀ m₀ hq _ he
    obtain ⟨e₀, rfl⟩ := hb.err_inv
    obtain ⟨e₁, he₁, hy₁⟩ : mayUB P I ρ₀ m₀ := by
      rcases hP I ρ₀ m₀ (hpre _ _ _ hq) with h | h
      · exact h
      · exact absurd hy (h e₀)
    obtain ⟨z, hz, hbz⟩ := hPQ I ρ₀ m₀ hq _ hy₁
    obtain ⟨e₂, rfl⟩ := hbz.err_inv
    refine ⟨e₂, ?_, hz⟩
    rcases hbz with ⟨e₃, he₃, hub⟩ | hrel
    · cases he₃
      exact hub
    · rw [← (hrel : e₁ = e₂)]
      exact he₁

/-! ## Block and region matching -/

theorem CtlRel.refl (x : Ctl × CSt) : CtlRel x x := ⟨rfl, CSt.Rel.refl _⟩

theorem CtlRel.trans {x y z : Ctl × CSt} (h₁ : CtlRel x y) (h₂ : CtlRel y z) : CtlRel x z :=
  ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩

theorem CtlRel.trans' : ∀ x y z : Ctl × CSt, CtlRel x y → CtlRel y z → CtlRel x z :=
  fun _ _ _ => CtlRel.trans

/-- Every run of `b'` from a state is matched (`MatchedE`, `CtlRel`) by a run of
`b` from any related state, in context `ctx`. -/
def BlockMatchedE (ctx : Ctx) (b' b : Block) : Prop :=
  ∀ (a' a : CSt), CSt.Rel a' a → ∀ (r' : Res (Ctl × CSt)), BlockExec ctx a' b' r' →
    MatchedE CtlRel (BlockExec ctx a b) r'

/-- The same for regions. -/
def RegionMatchedE (ctx : Ctx) (rg' rg : Region) : Prop :=
  ∀ (a' a : CSt), CSt.Rel a' a → ∀ (r' : Res (Ctl × CSt)), RegionExec ctx a' rg' r' →
    MatchedE CtlRel (RegionExec ctx a rg) r'

/-- From matching from equal start states (the form atoms prove), by the
log-shift lemmas. -/
theorem BlockMatchedE.of_same {ctx : Ctx} {b' b : Block}
    (h : ∀ cst r', BlockExec ctx cst b' r' → MatchedE CtlRel (BlockExec ctx cst b) r') : BlockMatchedE ctx b' b := by
  intro a' a ha r' hr'
  obtain ⟨r, hr, hrel⟩ := (BlockExec.rel ha).2 r' hr'
  exact MatchedE.of_rel_left CtlRel.trans' hrel (h a r hr)

theorem BlockMatchedE.refl (ctx : Ctx) (b : Block) : BlockMatchedE ctx b b := by
  intro a' a ha r' hr'
  obtain ⟨r, hr, hrel⟩ := (BlockExec.rel ha).2 r' hr'
  exact MatchedE.of_rel hr hrel

theorem RegionMatchedE.refl (ctx : Ctx) (rg : Region) : RegionMatchedE ctx rg rg := by
  intro a' a ha r' hr'
  obtain ⟨r, hr, hrel⟩ := (RegionExec.rel ha).2 r' hr'
  exact MatchedE.of_rel hr hrel

/-- Case split on the old outcome that matches a normal new outcome. -/
theorem MatchedE.okCases {β : Type} {S : β → β → Prop} {X : Res (Ctl × CSt) → Prop} {Y : Res β → Prop} {c : Ctl}
    {x' : CSt} {r' : Res β}
    (h : MatchedE CtlRel X (.ok (c, x')))
    (hub : ∀ e, e.isUB = true → X (.err e) → MatchedE S Y r')
    (hok : ∀ x, CSt.Rel x' x → X (.ok (c, x)) → MatchedE S Y r') : MatchedE S Y r' := by
  obtain ⟨y, hy, hb⟩ := h
  rcases hb.ok_inv with ⟨e, rfl, he⟩ | ⟨⟨c₀, x⟩, rfl, h₁, h₂⟩
  · exact hub e he hy
  · obtain rfl : c = c₀ := h₁
    exact hok x h₂ hy

section Leaf

variable {ctx : Ctx} {a' a : CSt}

theorem evalBoolR_rel (h : CSt.Rel a' a) (c : Expr) : ctx.evalBoolR a'.ρ a'.mem c = ctx.evalBoolR a.ρ a.mem c := by
  rw [h.1, h.2.1]

theorem trueEdges_rel {rg' rg : Region} (he : rg'.edges = rg.edges) (h : CSt.Rel a' a) {bid : BlockId} :
    trueEdges ctx rg' a' bid = trueEdges ctx rg a bid := by
  unfold trueEdges Region.outEdges
  rw [he, h.1, h.2.1]

theorem applyAssigns_rel (h : CSt.Rel a' a) (as : List (Sym × Expr)) :
    Res.Rel CSt.Rel (applyAssigns ctx a' as) (applyAssigns ctx a as) := by
  obtain ⟨h₁, h₂⟩ := h.eq_shift
  rw [h₁, h₂, applyAssigns_shift, applyAssigns_shift]
  exact Res.rel_map (fun x => CSt.Rel.shift h.2.2 x) _

theorem applyAssign?_rel (h : CSt.Rel a' a) (as : Option (Sym × Expr)) :
    Res.Rel CSt.Rel (applyAssign? ctx a' as) (applyAssign? ctx a as) := by
  cases as with
  | none => exact h
  | some x => exact applyAssigns_rel h [x]

theorem applyAssigns_rel_ok {as : List (Sym × Expr)} {c' : CSt} (h : CSt.Rel a' a)
    (hx : applyAssigns ctx a' as = .ok c') : ∃ c, applyAssigns ctx a as = .ok c ∧ CSt.Rel c' c := by
  have := applyAssigns_rel (ctx := ctx) h as
  rw [hx] at this
  exact this.ok_inv

theorem applyAssigns_rel_err {as : List (Sym × Expr)} {e : ErrReason} (h : CSt.Rel a' a)
    (hx : applyAssigns ctx a' as = .err e) : applyAssigns ctx a as = .err e := by
  have := applyAssigns_rel (ctx := ctx) h as
  rw [hx] at this
  exact this.err_inv

theorem applyAssign?_rel_ok {as : Option (Sym × Expr)} {c' : CSt} (h : CSt.Rel a' a)
    (hx : applyAssign? ctx a' as = .ok c') : ∃ c, applyAssign? ctx a as = .ok c ∧ CSt.Rel c' c := by
  have := applyAssign?_rel (ctx := ctx) h as
  rw [hx] at this
  exact this.ok_inv

theorem applyAssign?_rel_err {as : Option (Sym × Expr)} {e : ErrReason} (h : CSt.Rel a' a)
    (hx : applyAssign? ctx a' as = .err e) : applyAssign? ctx a as = .err e := by
  have := applyAssign?_rel (ctx := ctx) h as
  rw [hx] at this
  exact this.err_inv

end Leaf

/-! ### Lists related pointwise -/

theorem forall₂_getElem?_none {α β : Type} {R : α → β → Prop} :
    ∀ {l' : List α} {l : List β}, List.Forall₂ R l' l → ∀ {j : Nat}, l'[j]? = none → l[j]? = none
  | [], [], .nil, _, _ => rfl
  | _ :: _, _ :: _, .cons _ _, 0, h => by simp at h
  | _ :: _, _ :: _, .cons _ hr, j + 1, h => by simpa using forall₂_getElem?_none hr (by simpa using h)

theorem forall₂_getElem?_some {α β : Type} {R : α → β → Prop} :
    ∀ {l' : List α} {l : List β}, List.Forall₂ R l' l → ∀ {j : Nat} {x' : α}, l'[j]? = some x' →
      ∃ x, l[j]? = some x ∧ R x' x
  | [], [], .nil, _, _, h => by simp at h
  | _ :: _, x :: _, .cons hx _, 0, _, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨x, rfl, hx⟩
  | _ :: _, _ :: _, .cons _ hr, j + 1, _, h => by
    simpa using forall₂_getElem?_some hr (by simpa using h)

theorem forall₂_refl {α : Type} {R : α → α → Prop} : ∀ {l : List α}, (∀ y ∈ l, R y y) → List.Forall₂ R l l
  | [], _ => .nil
  | y :: _, hr => .cons (hr y List.mem_cons_self) (forall₂_refl fun z hz => hr z (List.mem_cons_of_mem _ hz))

/-- Replacing one element by a related one, all others related to themselves. -/
theorem forall₂_set {α : Type} {R : α → α → Prop} :
    ∀ {l : List α} {j : Nat} {x x' : α}, (∀ y ∈ l, R y y) → l[j]? = some x → R x' x →
      List.Forall₂ R (l.set j x') l
  | [], _, _, _, _, h, _ => by simp at h
  | y :: l, 0, x, x', hr, h, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact .cons hx (forall₂_refl fun z hz => hr z (List.mem_cons_of_mem _ hz))
  | y :: l, j + 1, x, x', hr, h, hx => by
    simp only [List.getElem?_cons_succ] at h
    exact .cons (hr y List.mem_cons_self) (forall₂_set (fun z hz => hr z (List.mem_cons_of_mem _ hz)) h hx)

theorem forall₂_cons_inv {α β : Type} {R : α → β → Prop} {x' : α} {l' : List α} {l : List β}
    (h : List.Forall₂ R (x' :: l') l) : ∃ x l₀, l = x :: l₀ ∧ R x' x ∧ List.Forall₂ R l' l₀ := by
  cases h with
  | cons hx hr => exact ⟨_, _, rfl, hx, hr⟩

theorem list_set_of_getElem? {α : Type} {l : List α} {j : Nat} {x : α} (h : l[j]? = some x) : l.set j x = l := by
  obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 h
  exact List.set_getElem_self hj

/-! ### Regions -/

/-- A region whose blocks are matched pointwise, with the same edges, runs
from any block are matched by the old region's (recursion on the new
derivation). -/
theorem RegionAt.matchedE {ctx : Ctx} {rg' rg : Region} (he : rg'.edges = rg.edges)
    (hb : List.Forall₂ (BlockMatchedE ctx) rg'.blocks rg.blocks) {a' a : CSt} {bid : BlockId}
    {r' : Res (Ctl × CSt)} (ha : CSt.Rel a' a) :
    RegionAt ctx rg' a' bid r' → MatchedE CtlRel (RegionAt ctx rg a bid) r'
  | .missing hm => MatchedE.of_rel (.missing (forall₂_getElem?_none hb hm)) rfl
  | .blockErr hm hB => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    obtain ⟨y, hy, hby⟩ := hsim _ _ ha _ hB
    obtain ⟨e₀, rfl⟩ := hby.err_inv
    exact ⟨_, .blockErr hb0 hy, hby⟩
  | .exit hm hB hc => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    exact (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu)
      (fun c₁ hrel hy => MatchedE.of_rel (.exit hb0 hy hc) ⟨rfl, hrel⟩)
  | .edgesErr hm hB hte => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    exact (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu)
      (fun c₁ hrel hy => MatchedE.of_rel (.edgesErr hb0 hy ((trueEdges_rel he hrel).symm.trans hte)) rfl)
  | .stop hm hB hte => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    exact (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu)
      (fun c₁ hrel hy => MatchedE.of_rel (.stop hb0 hy ((trueEdges_rel he hrel).symm.trans hte)) ⟨rfl, hrel⟩)
  | .ambiguous hm hB hte => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    exact (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu)
      (fun c₁ hrel hy => MatchedE.of_rel (.ambiguous hb0 hy ((trueEdges_rel he hrel).symm.trans hte)) rfl)
  | .assignErr hm hB hte hA => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    exact (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu)
      (fun c₁ hrel hy => MatchedE.of_rel (.assignErr hb0 hy ((trueEdges_rel he hrel).symm.trans hte)
        (applyAssigns_rel_err hrel hA)) rfl)
  | .step hm hB hte hA hn => by
    obtain ⟨b, hb0, hsim⟩ := forall₂_getElem?_some hb hm
    refine (hsim _ _ ha _ hB).okCases (fun e hu hy => MatchedE.of_ub (.blockErr hb0 hy) hu) (fun c₁ hrel hy => ?_)
    obtain ⟨c₂, hc₂, hrel₂⟩ := applyAssigns_rel_ok hrel hA
    obtain ⟨z, hz, hbz⟩ := RegionAt.matchedE he hb hrel₂ hn
    exact ⟨z, .step hb0 hy ((trueEdges_rel he hrel).symm.trans hte) hc₂ hz, hbz⟩

theorem RegionMatchedE.of_blocks {ctx : Ctx} {rg' rg : Region} (he : rg'.edges = rg.edges) (hs : rg'.start = rg.start)
    (hb : List.Forall₂ (BlockMatchedE ctx) rg'.blocks rg.blocks) : RegionMatchedE ctx rg' rg := by
  intro a' a ha r' hr
  cases hr with
  | start h =>
    obtain ⟨z, hz, hbz⟩ := RegionAt.matchedE he hb ha h
    exact ⟨z, .start (hs ▸ hz), hbz⟩

/-- Replacing one block of a region by a block whose runs it matches. -/
theorem RegionMatchedE.of_set {ctx : Ctx} {rg : Region} {j : BlockId} {x x' : Block} (hj : rg.block? j = some x)
    (hx : BlockMatchedE ctx x' x) : RegionMatchedE ctx { rg with blocks := rg.blocks.set j x' } rg := by
  intro a' a ha r' hr
  exact RegionMatchedE.of_blocks (rg' := { rg with blocks := rg.blocks.set j x' }) (rg := rg) rfl rfl
    (forall₂_set (fun y _ => BlockMatchedE.refl ctx y) hj hx) _ _ ha _ hr

theorem BlockMatchedE.region {ctx : Ctx} {rg' rg : Region} (h : RegionMatchedE ctx rg' rg) :
    BlockMatchedE ctx (.region rg') (.region rg) := by
  intro a' a ha r' hr
  cases hr with
  | region h' =>
    obtain ⟨z, hz, hbz⟩ := h _ _ ha _ h'
    exact ⟨z, .region hz, hbz⟩

/-! ### Loops -/

/-- Two loops with the same header (initialization, condition, update and
flags) and matched bodies. -/
structure LoopMatchedE (ctx : Ctx) (l' l : LoopRegion) : Prop where
  init : l'.init = l.init
  cond : l'.cond = l.cond
  update : l'.update = l.update
  inverted : l'.inverted = l.inverted
  updateBeforeCond : l'.updateBeforeCond = l.updateBeforeCond
  body : RegionMatchedE ctx l'.body l.body

section Loop

variable {ctx : Ctx} {l' l : LoopRegion}

namespace LoopMatchedE

theorem condEq (hL : LoopMatchedE ctx l' l) {a' a : CSt} (ha : CSt.Rel a' a) {x : Res Bool}
    (hc : ctx.evalBoolR a'.ρ a'.mem l'.cond = x) : ctx.evalBoolR a.ρ a.mem l.cond = x := by
  rw [← hL.cond, ← evalBoolR_rel ha]
  exact hc

theorem flag (hL : LoopMatchedE ctx l' l) {b : Bool} (hv : (l'.inverted && !l'.updateBeforeCond) = b) :
    (l.inverted && !l.updateBeforeCond) = b := by
  rw [← hL.inverted, ← hL.updateBeforeCond]
  exact hv

theorem update_ok (hL : LoopMatchedE ctx l' l) {a' a c' : CSt} (ha : CSt.Rel a' a)
    (hu : applyAssign? ctx a' l'.update = .ok c') : ∃ c, applyAssign? ctx a l.update = .ok c ∧ CSt.Rel c' c :=
  applyAssign?_rel_ok ha (hL.update ▸ hu)

theorem update_err (hL : LoopMatchedE ctx l' l) {a' a : CSt} {e : ErrReason} (ha : CSt.Rel a' a)
    (hu : applyAssign? ctx a' l'.update = .err e) : applyAssign? ctx a l.update = .err e :=
  applyAssign?_rel_err ha (hL.update ▸ hu)

theorem init_ok (hL : LoopMatchedE ctx l' l) {a' a c' : CSt} (ha : CSt.Rel a' a)
    (hu : applyAssign? ctx a' l'.init = .ok c') : ∃ c, applyAssign? ctx a l.init = .ok c ∧ CSt.Rel c' c :=
  applyAssign?_rel_ok ha (hL.init ▸ hu)

theorem init_err (hL : LoopMatchedE ctx l' l) {a' a : CSt} {e : ErrReason} (ha : CSt.Rel a' a)
    (hu : applyAssign? ctx a' l'.init = .err e) : applyAssign? ctx a l.init = .err e :=
  applyAssign?_rel_err ha (hL.init ▸ hu)

/-- The loop with its body replaced. -/
theorem of_body {body' : Region} (h : RegionMatchedE ctx body' l.body) : LoopMatchedE ctx { l with body := body' } l :=
  ⟨rfl, rfl, rfl, rfl, rfl, h⟩

end LoopMatchedE

mutual

theorem LoopGuard.matchedE (hL : LoopMatchedE ctx l' l) {a' a : CSt} {r' : Res (Ctl × CSt)}
    (ha : CSt.Rel a' a) : LoopGuard ctx a' l' r' → MatchedE CtlRel (LoopGuard ctx a l) r'
  | .condErr hc => MatchedE.of_rel (.condErr (hL.condEq ha hc)) rfl
  | .exit hc => MatchedE.of_rel (.exit (hL.condEq ha hc)) ⟨rfl, ha⟩
  | .enter hc hb =>
    let ⟨z, hz, hbz⟩ := LoopIter.matchedE hL ha hb
    ⟨z, .enter (hL.condEq ha hc) hz, hbz⟩

theorem LoopIter.matchedE (hL : LoopMatchedE ctx l' l) {a' a : CSt} {r' : Res (Ctl × CSt)}
    (ha : CSt.Rel a' a) : LoopIter ctx a' l' r' → MatchedE CtlRel (LoopIter ctx a l) r'
  | .bodyErr hb => by
    obtain ⟨y, hy, hby⟩ := hL.body _ _ ha _ hb
    obtain ⟨e₀, rfl⟩ := hby.err_inv
    exact ⟨_, .bodyErr hy, hby⟩
  | .ret hb => (hL.body _ _ ha _ hb).okCases (fun _ hu hy => MatchedE.of_ub (.bodyErr hy) hu)
      (fun _ hrel hy => MatchedE.of_rel (.ret hy) ⟨rfl, hrel⟩)
  | .brk hb => (hL.body _ _ ha _ hb).okCases (fun _ hu hy => MatchedE.of_ub (.bodyErr hy) hu)
      (fun _ hrel hy => MatchedE.of_rel (.brk hy) ⟨rfl, hrel⟩)
  | .tail hb hc ht => (hL.body _ _ ha _ hb).okCases (fun _ hu hy => MatchedE.of_ub (.bodyErr hy) hu)
      (fun _ hrel hy =>
        let ⟨z, hz, hbz⟩ := LoopTail.matchedE hL hrel ht
        ⟨z, .tail hy hc hz, hbz⟩)

theorem LoopTail.matchedE (hL : LoopMatchedE ctx l' l) {a' a : CSt} {r' : Res (Ctl × CSt)}
    (ha : CSt.Rel a' a) : LoopTail ctx a' l' r' → MatchedE CtlRel (LoopTail ctx a l) r'
  | .updateErr hv hu => MatchedE.of_rel (.updateErr (hL.flag hv) (hL.update_err ha hu)) rfl
  | .updateGuard hv hu hg => by
    obtain ⟨c₁, hc₁, hrel⟩ := hL.update_ok ha hu
    obtain ⟨z, hz, hbz⟩ := LoopGuard.matchedE hL hrel hg
    exact ⟨z, .updateGuard (hL.flag hv) hc₁ hz, hbz⟩
  | .condFirstErr hv hc => MatchedE.of_rel (.condFirstErr (hL.flag hv) (hL.condEq ha hc)) rfl
  | .condFirstExit hv hc => MatchedE.of_rel (.condFirstExit (hL.flag hv) (hL.condEq ha hc)) ⟨rfl, ha⟩
  | .condFirstUpdateErr hv hc hu =>
    MatchedE.of_rel (.condFirstUpdateErr (hL.flag hv) (hL.condEq ha hc) (hL.update_err ha hu)) rfl
  | .condFirstNext hv hc hu hb => by
    obtain ⟨c₁, hc₁, hrel⟩ := hL.update_ok ha hu
    obtain ⟨z, hz, hbz⟩ := LoopIter.matchedE hL hrel hb
    exact ⟨z, .condFirstNext (hL.flag hv) (hL.condEq ha hc) hc₁ hz, hbz⟩

end

theorem LoopExec.matchedE (hL : LoopMatchedE ctx l' l) {a' a : CSt} {r' : Res (Ctl × CSt)}
    (ha : CSt.Rel a' a) : LoopExec ctx a' l' r' → MatchedE CtlRel (LoopExec ctx a l) r'
  | .initErr hi => MatchedE.of_rel (.initErr (hL.init_err ha hi)) rfl
  | .guardFirst hi hv hg => by
    obtain ⟨c₁, hc₁, hrel⟩ := hL.init_ok ha hi
    obtain ⟨z, hz, hbz⟩ := LoopGuard.matchedE hL hrel hg
    exact ⟨z, .guardFirst hc₁ (hL.inverted ▸ hv) hz, hbz⟩
  | .bodyFirst hi hv hb => by
    obtain ⟨c₁, hc₁, hrel⟩ := hL.init_ok ha hi
    obtain ⟨z, hz, hbz⟩ := LoopIter.matchedE hL hrel hb
    exact ⟨z, .bodyFirst hc₁ (hL.inverted ▸ hv) hz, hbz⟩

theorem BlockMatchedE.loop (hL : LoopMatchedE ctx l' l) : BlockMatchedE ctx (.loop l') (.loop l) := by
  intro a' a ha r' hr
  cases hr with
  | loop h =>
    obtain ⟨z, hz, hbz⟩ := LoopExec.matchedE hL ha h
    exact ⟨z, .loop hz, hbz⟩

end Loop

/-! ### Conditionals -/

/-- Two branches: same condition, matched regions. -/
def BranchMatchedE (ctx : Ctx) (x' x : Option Expr × Region) : Prop := x'.1 = x.1 ∧ RegionMatchedE ctx x'.2 x.2

theorem CondExec.matchedE {ctx : Ctx} {brs' brs : List (Option Expr × Region)}
    (hb : List.Forall₂ (BranchMatchedE ctx) brs' brs) {a' a : CSt} {r' : Res (Ctl × CSt)} (ha : CSt.Rel a' a) :
    CondExec ctx a' brs' r' → MatchedE CtlRel (CondExec ctx a brs) r'
  | .nil => by
    cases hb
    exact MatchedE.of_rel .nil ⟨rfl, ha⟩
  | .els h => by
    obtain ⟨⟨c, rg⟩, rest, rfl, ⟨hc, hr⟩, -⟩ := forall₂_cons_inv hb
    obtain rfl : none = c := hc
    obtain ⟨z, hz, hbz⟩ := hr _ _ ha _ h
    exact ⟨z, .els hz, hbz⟩
  | .condErr hc => by
    obtain ⟨⟨c, rg⟩, rest, rfl, ⟨hc', -⟩, -⟩ := forall₂_cons_inv hb
    obtain rfl := hc'
    exact MatchedE.of_rel (.condErr ((evalBoolR_rel ha _).symm.trans hc)) rfl
  | .taken hc h => by
    obtain ⟨⟨c, rg⟩, rest, rfl, ⟨hc', hr⟩, -⟩ := forall₂_cons_inv hb
    obtain rfl := hc'
    obtain ⟨z, hz, hbz⟩ := hr _ _ ha _ h
    exact ⟨z, .taken ((evalBoolR_rel ha _).symm.trans hc) hz, hbz⟩
  | .skipped hc h => by
    obtain ⟨⟨c, rg⟩, rest, rfl, ⟨hc', -⟩, hrest⟩ := forall₂_cons_inv hb
    obtain rfl := hc'
    obtain ⟨z, hz, hbz⟩ := CondExec.matchedE hrest ha h
    exact ⟨z, .skipped ((evalBoolR_rel ha _).symm.trans hc) hz, hbz⟩

theorem BlockMatchedE.cond {ctx : Ctx} {label : String} {brs' brs : List (Option Expr × Region)}
    (hb : List.Forall₂ (BranchMatchedE ctx) brs' brs) : BlockMatchedE ctx (.cond label brs') (.cond label brs) := by
  intro a' a ha r' hr
  cases hr with
  | cond h =>
    obtain ⟨z, hz, hbz⟩ := CondExec.matchedE hb ha h
    exact ⟨z, .cond hz, hbz⟩

/-! ## Paths to blocks -/

/-- One step from a region into a sub-region of its block `i`. -/
inductive PathStep where
  /-- Block `i` is a `.region`: continue in it. -/
  | sub (i : BlockId)
  /-- Block `i` is a loop: continue in its body. -/
  | body (i : BlockId)
  /-- Block `i` is a conditional: continue in the region of its branch `k`. -/
  | branch (i : BlockId) (k : Nat)
  deriving DecidableEq, Repr, Inhabited

namespace Region

/-- The sub-region a step leads to. -/
def child? (r : Region) : PathStep → Option Region
  | .sub i => match r.block? i with
    | some (.region c) => some c
    | _ => none
  | .body i => match r.block? i with
    | some (.loop l) => some l.body
    | _ => none
  | .branch i k => match r.block? i with
    | some (.cond _ brs) => (brs[k]?).map (·.2)
    | _ => none

/-- Replace the sub-region a step leads to (unchanged if there is none). -/
def setChild (r : Region) (c : Region) : PathStep → Region
  | .sub i => match r.block? i with
    | some (.region _) => { r with blocks := r.blocks.set i (.region c) }
    | _ => r
  | .body i => match r.block? i with
    | some (.loop l) => { r with blocks := r.blocks.set i (.loop { l with body := c }) }
    | _ => r
  | .branch i k => match r.block? i with
    | some (.cond lb brs) =>
      match brs[k]? with
      | some (e, _) => { r with blocks := r.blocks.set i (.cond lb (brs.set k (e, c))) }
      | none => r
    | _ => r

/-- The block `i` of the region reached by the path `p`. -/
def blockAt? (r : Region) : List PathStep → BlockId → Option Block
  | [], i => r.block? i
  | st :: p, i => match r.child? st with
    | some c => c.blockAt? p i
    | none => none

/-- Replace the block `i` of the region reached by the path `p`. -/
def setBlockAt (r : Region) : List PathStep → BlockId → Block → Region
  | [], i, b => { r with blocks := r.blocks.set i b }
  | st :: p, i, b => match r.child? st with
    | some c => r.setChild (c.setBlockAt p i b) st
    | none => r

end Region

/-- Replace a block of the control-flow hierarchy of an SDFG. -/
def SDFG.setBlockAt (g : SDFG) (p : List PathStep) (i : BlockId) (b : Block) : SDFG :=
  { g with top := g.top.setBlockAt p i b }

/-- One step: the region with a sub-region replaced is matched if the
sub-region is. -/
theorem RegionMatchedE.setChild {ctx : Ctx} {r : Region} {st : PathStep} {c c' : Region}
    (hc : r.child? st = some c) (h : RegionMatchedE ctx c' c) : RegionMatchedE ctx (r.setChild c' st) r := by
  cases st with
  | sub j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i c₀ hj
      cases hc
      simp only [Region.setChild, hj]
      exact RegionMatchedE.of_set hj (BlockMatchedE.region h)
    · cases hc
  | body j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i l hj
      cases hc
      simp only [Region.setChild, hj]
      exact RegionMatchedE.of_set hj (BlockMatchedE.loop (LoopMatchedE.of_body h))
    · cases hc
  | branch j k =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i lb brs hj
      obtain ⟨⟨e, c₀⟩, hk, rfl⟩ := Option.map_eq_some_iff.1 hc
      simp only [Region.setChild, hj, hk]
      have hb : List.Forall₂ (BranchMatchedE ctx) (brs.set k (e, c')) brs :=
        forall₂_set (x' := (e, c')) (fun y _ => ⟨rfl, RegionMatchedE.refl ctx y.2⟩) hk ⟨rfl, h⟩
      exact RegionMatchedE.of_set hj (BlockMatchedE.cond hb)
    · cases hc

/-- **Congruence in one context**: replacing the block at a path by a block
whose runs it matches gives a region matched by the original one. -/
theorem Congr.region_setBlockAt {ctx : Ctx} {b' b : Block} {i : BlockId} (hs : BlockMatchedE ctx b' b) :
    ∀ (p : List PathStep) (r : Region), r.blockAt? p i = some b → RegionMatchedE ctx (r.setBlockAt p i b') r
  | [], r, h => RegionMatchedE.of_set h hs
  | st :: p, r, h => by
    simp only [Region.blockAt?] at h
    split at h
    · rename_i c hc
      simp only [Region.setBlockAt, hc]
      exact RegionMatchedE.setChild hc (Congr.region_setBlockAt hs p c h)
    · cases h

/-! ### Replacing twice -/

namespace Region

theorem block?_set {r : Region} {j : BlockId} {x x' : Block} (hj : r.block? j = some x) :
    ({ r with blocks := r.blocks.set j x' } : Region).block? j = some x' :=
  List.getElem?_set_self (List.getElem?_eq_some_iff.1 hj).1

theorem with_blocks_self {r : Region} {bs : List Block} (h : bs = r.blocks) : { r with blocks := bs } = r := by
  subst h
  cases r
  rfl

theorem with_set_set {r : Region} {j : BlockId} {x x' : Block} (hj : r.block? j = some x) :
    { r with blocks := (r.blocks.set j x').set j x } = r :=
  with_blocks_self (by rw [List.set_set, list_set_of_getElem? hj])

theorem with_set_self {r : Region} {j : BlockId} {x : Block} (hj : r.block? j = some x) :
    { r with blocks := r.blocks.set j x } = r :=
  with_blocks_self (list_set_of_getElem? hj)

theorem child?_setChild {r : Region} {st : PathStep} {c c' : Region} (hc : r.child? st = some c) :
    (r.setChild c' st).child? st = some c' := by
  cases st with
  | sub j =>
    simp only [child?] at hc
    split at hc
    · rename_i c₀ hj
      simp only [setChild, hj, child?, block?_set hj]
    · cases hc
  | body j =>
    simp only [child?] at hc
    split at hc
    · rename_i l hj
      simp only [setChild, hj, child?, block?_set hj]
    · cases hc
  | branch j k =>
    simp only [child?] at hc
    split at hc
    · rename_i lb brs hj
      obtain ⟨⟨e, c₀⟩, hk, rfl⟩ := Option.map_eq_some_iff.1 hc
      simp only [setChild, hj, hk, child?, block?_set hj,
        List.getElem?_set_self (List.getElem?_eq_some_iff.1 hk).1, Option.map_some]
    · cases hc

theorem setChild_setChild {r : Region} {st : PathStep} {c c' : Region} (hc : r.child? st = some c) :
    (r.setChild c' st).setChild c st = r := by
  cases st with
  | sub j =>
    simp only [child?] at hc
    split at hc
    · rename_i c₀ hj
      cases hc
      simp only [setChild, hj, block?_set hj]
      exact with_set_set hj
    · cases hc
  | body j =>
    simp only [child?] at hc
    split at hc
    · rename_i l hj
      cases hc
      obtain ⟨_, _, _, _, _, _, _, _⟩ := l
      simp only [setChild, hj, block?_set hj]
      exact with_set_set hj
    · cases hc
  | branch j k =>
    simp only [child?] at hc
    split at hc
    · rename_i lb brs hj
      obtain ⟨⟨e, c₀⟩, hk, rfl⟩ := Option.map_eq_some_iff.1 hc
      have hk' := List.getElem?_set_self (a := (e, c')) (List.getElem?_eq_some_iff.1 hk).1
      simp only [setChild, hj, hk, block?_set hj, hk', List.set_set, list_set_of_getElem? hk]
      exact with_set_self hj
    · cases hc

theorem blockAt?_setBlockAt {i : BlockId} {b b' : Block} :
    ∀ (p : List PathStep) (r : Region), r.blockAt? p i = some b → (r.setBlockAt p i b').blockAt? p i = some b'
  | [], r, h => block?_set h
  | st :: p, r, h => by
    simp only [blockAt?] at h
    split at h
    · rename_i c hc
      simp only [setBlockAt, hc, blockAt?, child?_setChild hc]
      exact blockAt?_setBlockAt p c h
    · cases h

theorem setBlockAt_setBlockAt {i : BlockId} {b b' : Block} :
    ∀ (p : List PathStep) (r : Region), r.blockAt? p i = some b → (r.setBlockAt p i b').setBlockAt p i b = r
  | [], r, h => with_set_set h
  | st :: p, r, h => by
    simp only [blockAt?] at h
    split at h
    · rename_i c hc
      simp only [setBlockAt, hc, child?_setChild hc, setBlockAt_setBlockAt p c h]
      exact setChild_setChild hc
    · cases h

end Region

/-! ## Mention counts and private transients -/

theorem Blocks.countMentions_set (t : Name) :
    ∀ {bs : List Block} {j : Nat} {x x' : Block}, bs[j]? = some x →
      Block.countMentions t x' = Block.countMentions t x →
        Blocks.countMentions t (bs.set j x') = Blocks.countMentions t bs
  | [], _, _, _, h, _ => by simp at h
  | y :: bs, 0, x, x', h, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [Blocks.countMentions, hx]
  | y :: bs, j + 1, x, x', h, hx => by
    simp only [List.getElem?_cons_succ] at h
    simp [Blocks.countMentions, Blocks.countMentions_set t h hx]

theorem Branches.countMentions_set (t : Name) :
    ∀ {brs : List (Option Expr × Region)} {k : Nat} {e : Option Expr} {c c' : Region}, brs[k]? = some (e, c) →
      Region.countMentions t c' = Region.countMentions t c →
        Branches.countMentions t (brs.set k (e, c')) = Branches.countMentions t brs
  | [], _, _, _, _, h, _ => by simp at h
  | y :: brs, 0, e, c, c', h, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [Branches.countMentions, hx]
  | y :: brs, k + 1, e, c, c', h, hx => by
    obtain ⟨c₀, r₀⟩ := y
    simp only [List.getElem?_cons_succ] at h
    simp [Branches.countMentions, Branches.countMentions_set t h hx]

theorem Region.countMentions_setBlocks (t : Name) {r : Region} {j : BlockId} {x x' : Block}
    (hj : r.block? j = some x) (hx : Block.countMentions t x' = Block.countMentions t x) :
    Region.countMentions t { r with blocks := r.blocks.set j x' } = Region.countMentions t r := by
  obtain ⟨_, _, blocks, _, edges⟩ := r
  simp only [Region.countMentions]
  rw [Blocks.countMentions_set t hj hx]

theorem Region.countMentions_setChild (t : Name) {r : Region} {st : PathStep} {c c' : Region}
    (hc : r.child? st = some c) (h : Region.countMentions t c' = Region.countMentions t c) :
    Region.countMentions t (r.setChild c' st) = Region.countMentions t r := by
  cases st with
  | sub j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i c₀ hj
      cases hc
      simp only [Region.setChild, hj]
      exact Region.countMentions_setBlocks t hj (by simp [Block.countMentions, h])
    · cases hc
  | body j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i l hj
      cases hc
      simp only [Region.setChild, hj]
      refine Region.countMentions_setBlocks t hj ?_
      obtain ⟨_, _, _, _, _, _, _, _⟩ := l
      simp only [Block.countMentions] at h ⊢
      rw [h]
    · cases hc
  | branch j k =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i lb brs hj
      obtain ⟨⟨e, c₀⟩, hk, rfl⟩ := Option.map_eq_some_iff.1 hc
      simp only [Region.setChild, hj, hk]
      exact Region.countMentions_setBlocks t hj (by simp [Block.countMentions, Branches.countMentions_set t hk h])
    · cases hc

/-- Replacing a block by one with the same mention count of `t` keeps the
mention count of the whole region. -/
theorem Region.countMentions_setBlockAt (t : Name) {i : BlockId} {b b' : Block}
    (hx : Block.countMentions t b' = Block.countMentions t b) :
    ∀ (p : List PathStep) (r : Region), r.blockAt? p i = some b →
      Region.countMentions t (r.setBlockAt p i b') = Region.countMentions t r
  | [], r, h => Region.countMentions_setBlocks t h hx
  | st :: p, r, h => by
    simp only [Region.blockAt?] at h
    split at h
    · rename_i c hc
      simp only [Region.setBlockAt, hc]
      exact Region.countMentions_setChild t hc (Region.countMentions_setBlockAt t hx p c h)
    · cases h

/-! ## Start memory and mention counts -/

/-- Replacing a block keeps the start memory: it depends only on the
descriptors (`Decisions.md#D68`). -/
theorem SDFG.startMem_setBlockAt (g : SDFG) (p : List PathStep) (i : BlockId) (b' : Block) :
    (g.setBlockAt p i b').startMem = g.startMem := rfl

/-- Is every transient of `g` with `Scope` or `State` lifetime mentioned
equally often by `b'` and `b` (`Region.countMentions`)?  Then replacing `b`
by `b'` keeps every private-transient set of `g` (`Ctx.agree_setBlockAt`). -/
def sameTransientMentions? (g : SDFG) (b' b : Block) : Bool :=
  g.arrays.all fun (t, d) =>
    !(d.transient && (d.lifetime == .scope || d.lifetime == .state)) ||
      Block.countMentions t b' == Block.countMentions t b

theorem sameTransientMentions?_count {g : SDFG} {b' b : Block} (h : sameTransientMentions? g b' b = true)
    {t : Name} {d : DataDesc} (hmem : (t, d) ∈ g.arrays)
    (hd : (d.transient && (d.lifetime == .scope || d.lifetime == .state)) = true) :
    Block.countMentions t b' = Block.countMentions t b := by
  unfold sameTransientMentions? at h
  have := List.all_eq_true.1 h (t, d) hmem
  simp only [hd, Bool.not_true, Bool.false_or, beq_iff_eq] at this
  exact this

theorem sameTransientMentions?_symm {g g' : SDFG} {b' b : Block} (ha : g'.arrays = g.arrays)
    (h : sameTransientMentions? g b' b = true) : sameTransientMentions? g' b b' = true := by
  unfold sameTransientMentions? at h ⊢
  rw [ha]
  refine List.all_eq_true.2 fun x hx => ?_
  have := List.all_eq_true.1 h x hx
  simp only [Bool.or_eq_true, beq_iff_eq] at this ⊢
  exact this.imp id Eq.symm

/-- **The contexts of an SDFG and of the SDFG with a block replaced agree**
(same descriptors, and same private transients for every scope) when the
two blocks mention every `Scope`-lifetime transient equally often. -/
theorem Ctx.agree_setBlockAt {I : Interp} {cn sc : Name → Name} {g : SDFG} {p : List PathStep} {i : BlockId}
    {b' b : Block} (hat : g.top.blockAt? p i = some b) (hm : sameTransientMentions? g b' b = true) :
    Ctx.Agree { I, g := g.setBlockAt p i b', cname := cn, scope := sc } { I, g, cname := cn, scope := sc } := by
  have hmem : ∀ {t : Name} {d : DataDesc}, ({ I, g, cname := cn, scope := sc } : Ctx).desc? t = some d →
      (t, d) ∈ g.arrays := by
    intro t d h
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at h
    obtain ⟨⟨t', d'⟩, hf, rfl⟩ := h
    have := List.find?_some hf
    simp only [decide_eq_true_eq] at this
    subst this
    exact List.mem_of_find?_eq_some hf
  refine ⟨rfl, fun s entry => ?_, fun s => ?_⟩
  · unfold Ctx.privateTransients
    refine List.filterMap_congr fun t _ => ?_
    simp only
    rw [show ({ I, g := g.setBlockAt p i b', cname := cn, scope := sc } : Ctx).desc? t =
      ({ I, g, cname := cn, scope := sc } : Ctx).desc? t from rfl]
    cases hdt : ({ I, g, cname := cn, scope := sc } : Ctx).desc? t with
    | none => rfl
    | some d =>
      simp only
      split
      · rename_i hsc
        have hd : (d.transient && (d.lifetime == .scope || d.lifetime == .state)) = true := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hsc
          simp [hsc.1, hsc.2]
        have hc : SDFG.countMentions (g.setBlockAt p i b') t = g.countMentions t :=
          Region.countMentions_setBlockAt t (sameTransientMentions?_count hm (hmem hdt) hd) p g.top hat
        rw [hc]
      · rfl
  · unfold Ctx.stateTransients
    refine List.filterMap_congr fun t _ => ?_
    simp only
    rw [show ({ I, g := g.setBlockAt p i b', cname := cn, scope := sc } : Ctx).desc? t =
      ({ I, g, cname := cn, scope := sc } : Ctx).desc? t from rfl]
    cases hdt : ({ I, g, cname := cn, scope := sc } : Ctx).desc? t with
    | none => rfl
    | some d =>
      simp only
      split
      · rename_i hd
        have hc : SDFG.countMentions (g.setBlockAt p i b') t = g.countMentions t :=
          Region.countMentions_setBlockAt t (sameTransientMentions?_count hm (hmem hdt) hd) p g.top hat
        rw [hc]
      · rfl

/-! ## Congruence for SDFGs -/

theorem SDFG.Exec.intro {I : Interp} {g : SDFG} {ρ₀ : Valuation} {m₀ : Mem} {res : Res (Valuation × Mem × Log)}
    (h : SDFGExec { I, g } ρ₀ (g.startMem m₀) g res) : SDFG.Exec I g ρ₀ m₀ (res.map fun x => x.2.1) :=
  ⟨res, h, rfl⟩

/-- Every run of the SDFG with the block replaced is matched (`MatchedE`) (with
equal final memories) by a run of the original SDFG. -/
theorem Congr.exec_matchedE {g : SDFG} {p : List PathStep} {i : BlockId} {b' b : Block}
    (hat : g.top.blockAt? p i = some b) (hm : sameTransientMentions? g b' b = true) {I : Interp}
    (hs : BlockMatchedE { I, g } b' b) {ρ₀ : Valuation} {m₀ : Mem} {r : Res Mem}
    (hr : SDFG.Exec I (g.setBlockAt p i b') ρ₀ m₀ r) : MatchedE Eq (SDFG.Exec I g ρ₀ m₀) r := by
  obtain ⟨res, hres, rfl⟩ := hr
  have hres := SDFGExec.congr (Ctx.agree_setBlockAt (I := I) (cn := id) (sc := id) hat hm) hres
  have hreg := Congr.region_setBlockAt hs p g.top hat
  cases hres with
  | err h =>
    obtain ⟨y, hy, hby⟩ := hreg _ _ (CSt.Rel.refl _) _ h
    obtain ⟨e₀, rfl⟩ := hby.err_inv
    exact ⟨_, SDFG.Exec.intro (.err hy), hby.err_err⟩
  | escape h hc =>
    exact (hreg _ _ (CSt.Rel.refl _) _ h).okCases (fun _ hu hy => MatchedE.of_ub (SDFG.Exec.intro (.err hy)) hu)
      (fun _ _ hy => MatchedE.of_rel (SDFG.Exec.intro (.escape hy hc)) rfl)
  | run h hc =>
    exact (hreg _ _ (CSt.Rel.refl _) _ h).okCases (fun _ hu hy => MatchedE.of_ub (SDFG.Exec.intro (.err hy)) hu)
      (fun _ hrel hy => MatchedE.of_rel (SDFG.Exec.intro (.run hy hc)) hrel.2.1)

/-- **Block congruence**: replacing the block `b` at path
`p`, index `i` of `g` by `b'` refines `g` if

* `b'` mentions every `Scope`-lifetime transient as often as `b` does
  (`sameTransientMentions?`; the private transients of every scope are kept);
* for every lawful interpretation, every run of `b'` from a control state is
  matched (`MatchedE`) (`CtlRel`: same signal, same valuation and memory, log
  refined by `Log.Le`; or the same error; or undefined behavior of `b`) by a
  run of `b` from the same state, both in the context of `g` (equivalently
  of the new SDFG, `Ctx.agree_setBlockAt`);
* `g` is `Supported` (necessary: `Refines g g` alone needs it). -/
theorem Congr.block {g : SDFG} {p : List PathStep} {i : BlockId} {b' b : Block}
    (hat : g.top.blockAt? p i = some b) (hm : sameTransientMentions? g b' b = true)
    (hs : ∀ I : Interp, I.Lawful → ∀ cst r', BlockExec { I, g } cst b' r' →
      MatchedE CtlRel (BlockExec { I, g } cst b) r')
    (hsup : Supported g.prog) : Refines (g.setBlockAt p i b').prog g.prog :=
  Refines.of_matchedE hsup fun I _ _ hp _ hr =>
    (Congr.exec_matchedE hat hm (BlockMatchedE.of_same (hs I hp.1)) hr).mono
      (fun _ _ h => h ▸ obsEq.refl _ _) (fun _ h => h)

/-- **Block congruence, both directions**: blocks that match each other's runs
give equivalent SDFGs (the new SDFG is then supported as well). -/
theorem Congr.blockEquiv {g : SDFG} {p : List PathStep} {i : BlockId} {b' b : Block}
    (hat : g.top.blockAt? p i = some b) (hm : sameTransientMentions? g b' b = true)
    (hs : ∀ I : Interp, I.Lawful → ∀ cst r', BlockExec { I, g } cst b' r' →
      MatchedE CtlRel (BlockExec { I, g } cst b) r')
    (hs' : ∀ I : Interp, I.Lawful → ∀ cst r, BlockExec { I, g } cst b r →
      MatchedE CtlRel (BlockExec { I, g } cst b') r)
    (hsup : Supported g.prog) : Equiv (g.setBlockAt p i b').prog g.prog := by
  have hat' : (g.setBlockAt p i b').top.blockAt? p i = some b' := Region.blockAt?_setBlockAt p g.top hat
  have hm' := sameTransientMentions?_symm (g' := g.setBlockAt p i b') rfl hm
  have hg : (g.setBlockAt p i b').setBlockAt p i b = g := by
    obtain ⟨_, _, _, _, _, top, _⟩ := g
    simp only [SDFG.setBlockAt, Region.setBlockAt_setBlockAt p top hat]
  have fwd : ∀ I ρ₀ m₀, (g.setBlockAt p i b').prog.Pre I ρ₀ m₀ → ∀ r,
      (g.setBlockAt p i b').prog.exec I ρ₀ m₀ r → MatchedE Eq (g.prog.exec I ρ₀ m₀) r :=
    fun I _ _ hp _ hr => Congr.exec_matchedE hat hm (BlockMatchedE.of_same (hs I hp.1)) hr
  have bwd : ∀ I ρ₀ m₀, (g.setBlockAt p i b').prog.Pre I ρ₀ m₀ → ∀ r,
      g.prog.exec I ρ₀ m₀ r → MatchedE Eq ((g.setBlockAt p i b').prog.exec I ρ₀ m₀) r := by
    intro I ρ₀ m₀ hp r hr
    have hA := Ctx.agree_setBlockAt (I := I) (cn := id) (sc := id) hat hm
    have hsI : BlockMatchedE { I, g := g.setBlockAt p i b' } b b' :=
      BlockMatchedE.of_same fun cst r hr =>
        (hs' I hp.1 cst r (BlockExec.congr hA hr)).mono (fun _ _ h => h) (fun _ h => BlockExec.congr hA.symm h)
    have hr' : SDFG.Exec I ((g.setBlockAt p i b').setBlockAt p i b) ρ₀ m₀ r := by rw [hg]; exact hr
    exact Congr.exec_matchedE hat' hm' hsI hr'
  have hsup' : Supported (g.setBlockAt p i b').prog :=
    Supported.of_matchedE hsup (fun _ _ _ h => h) fwd bwd
  refine ⟨Refines.of_matchedE hsup fun I ρ₀ m₀ hp r hr => (fwd I ρ₀ m₀ hp r hr).mono
      (fun _ _ h => h ▸ obsEq.refl _ _) (fun _ h => h),
    Refines.of_matchedE hsup' fun I ρ₀ m₀ hp r hr => (bwd I ρ₀ m₀ hp r hr).mono
      (fun _ _ h => h ▸ obsEq.refl _ _) (fun _ h => h)⟩

/-! ## Instances -/

/-- A state ends with the `normal` signal. -/
theorem StateExec.ok_normal {ctx : Ctx} {cst : CSt} {s : State} {r : Res (Ctl × CSt)} (h : StateExec ctx cst s r) :
    ∀ c x, r = .ok (c, x) → c = .normal := by
  cases h with
  | compsErr => intro c x h; cases h
  | run hc =>
    intro c x h
    unfold stateFinish at h
    split at h
    · simp only [Res.ok_inj, Prod.mk.injEq] at h
      exact h.1.symm
    · cases h

theorem State.components_of_nodes {s : State} (hs : s.nodes = []) : s.components = [] := by
  simp [State.components, hs, State.componentsFrom]

/-- A state without nodes is a no-op. -/
theorem StateExec.of_noNodes {ctx : Ctx} {cst : CSt} {s : State} (hs : s.nodes = []) :
    StateExec ctx cst s (.ok (.normal, cst)) := by
  have hc := State.components_of_nodes hs
  have h := StateExec.run (ctx := ctx) (cst := cst) (s := s) (logs := []) (hc ▸ CompsExec.nil)
  rwa [StateExec_noComponents hc h] at h

/-- Any state without nodes matches any other (in every context). -/
theorem emptyState_matchedE {ctx : Ctx} {s' s : State} (hs' : s'.nodes = []) (hs : s.nodes = []) :
    ∀ cst r', BlockExec ctx cst (.state s') r' → MatchedE CtlRel (BlockExec ctx cst (.state s)) r' := by
  intro cst r' h
  cases h with
  | state h =>
    rw [StateExec_noComponents (State.components_of_nodes hs') h]
    exact MatchedE.of_rel (.state (StateExec.of_noNodes hs)) (CtlRel.refl _)

/-- **Instance 1**: a state without nodes may be replaced by any other state
without nodes (e.g. the same state renamed), at any path. -/
theorem Congr.emptyState {g : SDFG} {p : List PathStep} {i : BlockId} {s s' : State}
    (hat : g.top.blockAt? p i = some (.state s)) (hs : s.nodes = []) (hs' : s'.nodes = [])
    (hsup : Supported g.prog) : Equiv (g.setBlockAt p i (.state s')).prog g.prog :=
  Congr.blockEquiv hat (by simp [sameTransientMentions?, Block.countMentions, hs, hs'])
    (fun _ _ => emptyState_matchedE hs' hs) (fun _ _ => emptyState_matchedE hs hs') hsup

/-- The sub-region "`s`, then an empty state" (an unconditional edge without
assignments between them). -/
def State.thenEmpty (s : State) : Region :=
  { blocks := [.state s, .state {}], edges := [{ src := 0, dst := 1 }] }

section ThenEmpty

variable {ctx : Ctx} {s : State} {cst : CSt}

theorem State.thenEmpty_trueEdges₀ : trueEdges ctx s.thenEmpty cst 0 = .ok [{ src := 0, dst := 1 }] := rfl

theorem State.thenEmpty_trueEdges₁ : trueEdges ctx s.thenEmpty cst 1 = .ok [] := rfl

theorem applyAssigns_nil : applyAssigns ctx cst [] = .ok { cst with ρ := Valuation.setAll cst.ρ [] } := rfl

theorem CSt.Rel.setAll_nil (c : CSt) : CSt.Rel c { c with ρ := Valuation.setAll c.ρ [] } :=
  ⟨rfl, rfl, Log.Le.refl _⟩

/-- The sub-region `s.thenEmpty` is matched by the state `s`. -/
theorem thenEmpty_matchedE : ∀ cst r', BlockExec ctx cst (.region s.thenEmpty) r' →
    MatchedE CtlRel (BlockExec ctx cst (.state s)) r' := by
  intro cst r' h
  cases h with
  | region h =>
  cases h with
  | start h =>
  cases h with
  | missing hm => cases hm
  | blockErr hb hB =>
    cases hb
    exact MatchedE.of_rel hB rfl
  | exit hb hB hc =>
    cases hb
    cases hB with
    | state hB => exact absurd (StateExec.ok_normal hB _ _ rfl) hc
  | edgesErr hb hB he =>
    cases State.thenEmpty_trueEdges₀.symm.trans he
  | stop hb hB he =>
    cases State.thenEmpty_trueEdges₀.symm.trans he
  | ambiguous hb hB he =>
    cases State.thenEmpty_trueEdges₀.symm.trans he
  | assignErr hb hB he ha =>
    cases State.thenEmpty_trueEdges₀.symm.trans he
    cases applyAssigns_nil.symm.trans ha
  | step hb hB he ha hn =>
    cases hb
    cases State.thenEmpty_trueEdges₀.symm.trans he
    cases applyAssigns_nil.symm.trans ha
    cases hn with
    | missing hm => cases hm
    | blockErr hb hB' =>
      cases hb
      cases hB' with
      | state h => cases StateExec_noComponents rfl h
    | exit hb hB' hc =>
      cases hb
      cases hB' with
      | state h => exact absurd (StateExec.ok_normal h _ _ rfl) hc
    | edgesErr hb hB' he' => cases State.thenEmpty_trueEdges₁.symm.trans he'
    | stop hb hB' he' =>
      cases hb
      cases hB' with
      | state h =>
        cases StateExec_noComponents rfl h
        exact MatchedE.of_rel hB ⟨rfl, rfl, rfl, Log.Le.refl _⟩
    | ambiguous hb hB' he' => cases State.thenEmpty_trueEdges₁.symm.trans he'
    | assignErr hb hB' he' => cases State.thenEmpty_trueEdges₁.symm.trans he'
    | step hb hB' he' => cases State.thenEmpty_trueEdges₁.symm.trans he'

/-- The state `s` is matched by the sub-region `s.thenEmpty`. -/
theorem state_thenEmpty_matchedE : ∀ cst r', BlockExec ctx cst (.state s) r' →
    MatchedE CtlRel (BlockExec ctx cst (.region s.thenEmpty)) r' := by
  intro cst r' h
  cases r' with
  | err e => exact MatchedE.of_rel (.region (.start (.blockErr rfl h))) rfl
  | ok x =>
    obtain ⟨c, c₁⟩ := x
    have hn : c = .normal := by
      cases h with
      | state h => exact StateExec.ok_normal h _ _ rfl
    subst hn
    exact MatchedE.of_rel (.region (.start (.step rfl h State.thenEmpty_trueEdges₀
      applyAssigns_nil (.stop rfl (.state (StateExec.of_noNodes rfl)) State.thenEmpty_trueEdges₁))))
      ⟨rfl, CSt.Rel.setAll_nil _⟩

end ThenEmpty

theorem Block.countMentions_thenEmpty (t : Name) (s : State) :
    Block.countMentions t (.region s.thenEmpty) = Block.countMentions t (.state s) := by
  have : (Expr.lit (.bool true)).mentions t = false := rfl
  simp [State.thenEmpty, Block.countMentions, Region.countMentions, Blocks.countMentions, IEdge.mentions, this]

/-- **Instance 2 (state split)**: a state is equivalent to the sub-region
"the state, then an empty state". -/
theorem Congr.splitEmpty {g : SDFG} {p : List PathStep} {i : BlockId} {s : State}
    (hat : g.top.blockAt? p i = some (.state s)) (hsup : Supported g.prog) :
    Equiv (g.setBlockAt p i (.region s.thenEmpty)).prog g.prog :=
  Congr.blockEquiv hat (by simp [sameTransientMentions?, Block.countMentions_thenEmpty])
    (fun _ _ => thenEmpty_matchedE) (fun _ _ => state_thenEmpty_matchedE) hsup

/-- **Instance 2 (state merge)**: the sub-region "a state, then an empty
state" is equivalent to the state alone. -/
theorem Congr.mergeEmpty {g : SDFG} {p : List PathStep} {i : BlockId} {s : State}
    (hat : g.top.blockAt? p i = some (.region s.thenEmpty)) (hsup : Supported g.prog) :
    Equiv (g.setBlockAt p i (.state s)).prog g.prog :=
  Congr.blockEquiv hat (by simp [sameTransientMentions?, Block.countMentions_thenEmpty])
    (fun _ _ => state_thenEmpty_matchedE) (fun _ _ => thenEmpty_matchedE) hsup

end Dace
