-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrMatch
import Dace.Equiv.Equiv

/-!
# SDFG and schedule tree: states, the state chain and the program (M5a)

* **States** (`StateCorr`, `StateCorr.matched`): a state of the fragment
  `SeqPure` with at most one weakly connected component whose tree segment
  is the image of the component's top-level nodes in a topological order of
  the component (`SegCorr`).  The SDFG runs the component from the state's
  memory with an empty log and merges its log; the tree runs the segment on
  the current state.  Coherence (`OrderedExec.coherent`: the memory after a
  run is the initial memory with its log applied) makes the two agree.
* **The chain** (`ChainCorr`, `ChainCorr.matched`): the top region is a chain of
  states joined by unconditional edges without assignments; the tree's root
  list is the concatenation of their segments.
* **Programs** (`Corr.exec`): every outcome of the tree is an outcome of the
  SDFG, or the SDFG may end in undefined behavior; hence the tree refines
  the SDFG when the SDFG is `Supported` (`Corr.refines`).
-/

set_option autoImplicit false

namespace Dace

namespace Corr

/-! ## States -/

/-- The tree segment `cs` corresponds to state `s`. -/
structure StateCorr (ctx : Ctx) (tc : TCtx) (s : State) (cs : List SNode) : Prop where
  pure : SeqPure ctx s
  comps : (s.components = [] ∧ cs = []) ∨
    ∃ c o full, s.components = [c] ∧ SegCorr ctx tc s o cs ∧ s.isTopologicalOrder c full = true ∧
      full.filter (s.topLevelOf c).contains = o

theorem unionAll_single (g : Log) : Log.unionAll [g] = g := by
  cases g; simp [Log.unionAll]

section State

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.erase = ctx.erase) (hcons : tc.consumed = none)

include he hcons in
/-- **Matching of a state**: the tree runs the segment, the SDFG the state;
undefined behavior, the same error, or the same control-flow state. -/
theorem StateCorr.matched {s : State} {cs : List SNode} (hsc : StateCorr ctx tc s cs) {full : List SNode} {gb : Bool}
    {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
    (hx : STree.ListExec tc full gb {} (cs ++ rest) cst r) :
    (∃ e, e.isUB = true ∧ StateExec ctx cst s (.err e)) ∨
    (∃ e, r = .err e ∧ StateExec ctx cst s (.err e)) ∨
    ∃ cst', STree.ListExec tc full gb {} rest cst' r ∧ StateExec ctx cst s (.ok (.normal, cst')) := by
  have h := hsc.pure
  rcases hsc.comps with ⟨hc, rfl⟩ | ⟨c, o, ofull, hc, hseg, hto, hdir⟩
  · -- no component: the state does nothing
    refine .inr (.inr ⟨cst, hx, ?_⟩)
    have := StateExec.run (ctx := ctx) (cst := cst) (s := s) (logs := []) (hc ▸ .nil)
    have hf : stateFinish ctx cst [] = .ok (.normal, cst) := by
      unfold stateFinish
      rw [merge_eq_apply ctx.I.wcr cst.mem (rfl : Log.pairwiseFree [] = true)]
      have hu : Log.unionAll [] = Log.empty := rfl
      rw [hu, Log.apply_empty, Log.seq_empty]
    rwa [hf] at this
  · let st₀ : St := { mem := cst.mem, log := cst.log }
    have hrel : StRel st₀ cst := ⟨rfl, rfl, rfl⟩
    let base : St := { mem := cst.mem }
    have hst₀ : st₀ = { base with log := cst.log } := rfl
    have hp : bindViews ctx cst.ρ s (s.topLevelOf c) base = .ok base := bindViews_eq cst.ρ h _ _
    -- one SDFG component run from the state's memory, as an outcome of the state
    have hstate : ∀ {rc : Res St}, OrderedExec ctx cst.ρ s c (s.topLevelOf c) base rc →
        (∀ e, rc = .err e → StateExec ctx cst s (.err e)) ∧
        (∀ st'', rc = .ok st'' → StateExec ctx cst s
          (.ok (.normal, { cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log }))) := by
      intro rc hrc
      refine ⟨fun e he' => ?_, fun st'' hok => ?_⟩
      · subst he'
        exact .compsErr (by rw [hc]; exact CompsExec.headErr hrc)
      · subst hok
        have := StateExec.run (ctx := ctx) (cst := cst) (s := s) (logs := [st''.log])
          (by rw [hc]; exact CompsExec.cons hrc .nil)
        have hf : stateFinish ctx cst [st''.log] =
            .ok (.normal, { cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log }) := by
          simp only [stateFinish]
          rw [merge_eq_apply ctx.I.wcr cst.mem (by simp [Log.pairwiseFree]), unionAll_single]
        rwa [hf] at this
    rcases SegCorr.matched h he hcons hseg hx hrel with ⟨e, hu, hs⟩ | ⟨e, hr, hs⟩ | ⟨cst', st', tr, hl, hs, hρ, hrel'⟩
    · rw [hst₀] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.log_iff cst.log).1 hs
      cases r₀ with
      | ok _ => cases he₀
      | err e' =>
        cases he₀
        rw [← hdir] at hs₀
        exact .inl ⟨_, hu, (hstate (.seqErr hto hp hs₀)).1 _ rfl⟩
    · rw [hst₀] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.log_iff cst.log).1 hs
      cases r₀ with
      | ok _ => cases he₀
      | err e' =>
        cases he₀
        rw [← hdir] at hs₀
        exact .inr (.inl ⟨_, hr, (hstate (.seqErr hto hp hs₀)).1 _ rfl⟩)
    · rw [hst₀] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.log_iff cst.log).1 hs
      cases r₀ with
      | err _ => cases he₀
      | ok p =>
        obtain ⟨st'', tr''⟩ := p
        simp only [Res.map, Prod.map, Res.ok.injEq, Prod.mk.injEq, id] at he₀
        obtain ⟨rfl, rfl⟩ := he₀
        rw [← hdir] at hs₀
        cases hf : s.hbFree tr with
        | false => exact .inl ⟨_, rfl, (hstate (.hbRace hto hp hs₀ hf)).1 _ rfl⟩
        | true =>
          have hrun : OrderedExec ctx cst.ρ s c (s.topLevelOf c) base (.ok st'') := .run hto hp hs₀ hf
          have hcoh := OrderedExec.coherent h (st := base) hrun
          refine .inr (.inr ⟨cst', hl, ?_⟩)
          have := (hstate hrun).2 _ rfl
          have hcst : ({ cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log } : CSt) =
              cst' := by
            obtain ⟨hm, hlg, -⟩ := hrel'
            cases cst'
            simp only [St.shift_mem, St.shift_log] at hm hlg
            simp only at hρ
            simp only [CSt.mk.injEq]
            exact ⟨hρ.symm, by rw [← hm, hcoh], hlg⟩
          rwa [hcst] at this

end State

/-! ## The state chain -/

/-- The top region is a chain of states `blocks` joined by unconditional
edges without assignments, and the tree list is the concatenation of their
segments. -/
inductive ChainCorr (ctx : Ctx) (tc : TCtx) (rg : Region) : BlockId → List SNode → Prop where
  | last {bid : BlockId} {s : State} {cs : List SNode}
      (hb : rg.block? bid = some (.state s)) (hs : StateCorr ctx tc s cs) (he : rg.outEdges bid = []) :
      ChainCorr ctx tc rg bid cs
  | step {bid : BlockId} {s : State} {cs rest : List SNode} {e : IEdge}
      (hb : rg.block? bid = some (.state s)) (hs : StateCorr ctx tc s cs) (he : rg.outEdges bid = [e])
      (hc : e.cond = .lit (.bool true)) (ha : e.assigns = []) (h : ChainCorr ctx tc rg e.dst rest) :
      ChainCorr ctx tc rg bid (cs ++ rest)

section Chain

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.erase = ctx.erase) (hcons : tc.consumed = none)

theorem evalBoolR_true (ctx : Ctx) (ρ : Valuation) (m : Mem) : ctx.evalBoolR ρ m (.lit (.bool true)) = .ok true := by
  simp [Ctx.evalBoolR, Ctx.evalBool, Expr.evalBool, Expr.eval, Val.toBool?, Res.ofOption]

theorem trueEdges_uncond {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {e : IEdge}
    (he : rg.outEdges bid = [e]) (hc : e.cond = .lit (.bool true)) : trueEdges ctx rg cst bid = .ok [e] := by
  simp [trueEdges, he, List.filterMapM, List.filterMapM.loop, hc, evalBoolR_true]

theorem trueEdges_nil {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} (he : rg.outEdges bid = []) :
    trueEdges ctx rg cst bid = .ok [] := by
  simp only [trueEdges, he]
  rfl

include he hcons in
/-- **Matching of the chain**: the tree's list from the segment of block
`bid` on, against the SDFG region from `bid`. -/
theorem ChainCorr.matched {rg : Region} {bid : BlockId} {cs : List SNode} (hch : ChainCorr ctx tc rg bid cs) :
    ∀ {full : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
      STree.ListExec tc full false {} cs cst r →
      (∃ e, e.isUB = true ∧ RegionAt ctx rg cst bid (.err e)) ∨
      (∃ e, r = .err e ∧ RegionAt ctx rg cst bid (.err e)) ∨
      ∃ c, r = .ok (.normal, c) ∧ RegionAt ctx rg cst bid (.ok (.normal, c)) := by
  induction hch with
  | @last bid s cs hb hs hout =>
    intro full cst r hx
    have hx₀ : STree.ListExec tc full false {} (cs ++ []) cst r := by rw [List.append_nil]; exact hx
    rcases hs.matched he hcons hx₀ with ⟨e, hu, hx'⟩ | ⟨e, hr, hx'⟩ | ⟨cst', hl, hx'⟩
    · exact .inl ⟨e, hu, .blockErr hb (.state hx')⟩
    · exact .inr (.inl ⟨e, hr, .blockErr hb (.state hx')⟩)
    · have := STree.ListExec_nil rfl hl
      subst this
      exact .inr (.inr ⟨_, rfl, .stop hb (.state hx') (trueEdges_nil hout)⟩)
  | step hb hs hout hc ha _ ih =>
    intro full cst r hx
    rcases hs.matched he hcons hx with ⟨e, hu, hx'⟩ | ⟨e, hr, hx'⟩ | ⟨cst', hl, hx'⟩
    · exact .inl ⟨e, hu, .blockErr hb (.state hx')⟩
    · exact .inr (.inl ⟨e, hr, .blockErr hb (.state hx')⟩)
    · have hassign : applyAssigns ctx cst' [] = .ok cst' := by
        simp [applyAssigns]
        rfl
      rw [← ha] at hassign
      have hstep := fun {r'} (h' : RegionAt ctx rg cst' _ r') =>
        RegionAt.step hb (.state hx') (trueEdges_uncond hout hc) (List.mem_singleton_self _) hassign h'
      rcases ih hl with ⟨e, hu, hy⟩ | ⟨e, hr, hy⟩ | ⟨c, hr, hy⟩
      · exact .inl ⟨e, hu, hstep hy⟩
      · exact .inr (.inl ⟨e, hr, hstep hy⟩)
      · exact .inr (.inr ⟨c, hr, hstep hy⟩)

end Chain

/-! ## Programs -/

/-- The contexts in which an SDFG and a schedule tree run. -/
def sctx (I : Interp) (g : SDFG) : Ctx := { I, g, pfx := "" }

/-- The tree's root list corresponds to the SDFG's top region. -/
structure ProgCorr (I : Interp) (g : SDFG) (t : STree) : Prop where
  arrays : t.containers = g.arrays
  consts : t.toSDFG.withConstants = g.withConstants
  chain : ChainCorr (sctx I g) (TCtx.ofTree I t) g.top g.top.start t.children

/-- **Programs**: every outcome of the tree is an outcome of the SDFG, or the
SDFG may end in undefined behavior. -/
theorem ProgCorr.exec {I : Interp} {g : SDFG} {t : STree} (hc : ProgCorr I g t) {ρ₀ : Valuation} {m₀ : Mem}
    {r : Res Mem} (ht : STree.Exec I t ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨ SDFG.Exec I g ρ₀ m₀ r := by
  obtain ⟨res, hx, rfl⟩ := ht
  have he : (TCtx.ofTree I t).ctx.erase = (sctx I g).erase :=
    Ctx.erase_eq_of rfl rfl (by simp [TCtx.ofTree, STree.toSDFG, sctx, hc.arrays])
  have hstart : rootStart t ρ₀ m₀ = { ρ := g.withConstants ρ₀, mem := m₀ } := by
    simp [rootStart, hc.consts]
  rw [hstart] at hx
  rcases hc.chain.matched he rfl hx with ⟨e, hu, hy⟩ | ⟨e, rfl, hy⟩ | ⟨c, rfl, hy⟩
  · exact .inl ⟨e, hu, _, .err (.start hy), rfl⟩
  · exact .inr ⟨_, .err (.start hy), rfl⟩
  · exact .inr ⟨_, .run (.start hy) rfl, rfl⟩

/-- **Refinement**: if the SDFG is supported, the tree refines it. -/
theorem ProgCorr.refines {g : SDFG} {t : STree} (hc : ∀ I, ProgCorr I g t) (hs : Supported g.prog) :
    Refines t.prog g.prog := by
  intro I ρ₀ m₀ hpre
  by_cases hmu : mayUB g.prog I ρ₀ m₀
  · exact .inl hmu
  · refine .inr fun r hr => ?_
    rcases (hc I).exec hr with ⟨e, hu, hy⟩ | hy
    · exact absurd ⟨e, hu, hy⟩ hmu
    · cases r with
      | ok m => exact ⟨m, rfl, m, hy, obsEq.refl _ _⟩
      | err e =>
        rcases hs I ρ₀ m₀ hpre with h | h
        · exact absurd h hmu
        · exact absurd hy (h e)

end Corr

end Dace
