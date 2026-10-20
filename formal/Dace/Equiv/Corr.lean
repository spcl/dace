-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrMatch
import Dace.Equiv.Equiv

/-!
# SDFG and schedule tree: states (M5a)

* **States** (`StateCorr`, `StateCorr.matched`): a state of the fragment
  `SeqPure` with at most one weakly connected component whose tree segment
  is the image of the component's top-level nodes in a topological order of
  the component (`SegCorr`).  The SDFG runs the component from the state's
  memory with an empty log and merges its log; the tree runs the segment on
  the current state.  Coherence (`OrderedExec.coherent`: the memory after a
  run is the initial memory with its log applied) makes the two agree.

Control flow and whole programs: `Dace.Equiv.CorrCtl`.
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

end Corr

end Dace
