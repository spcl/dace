-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrMatch
import Dace.SDFG.Sim
import Dace.Equiv.Equiv

/-!
# SDFG and schedule tree: states (M5a)

* **States** (`StateCorr`, `StateCorr.matched`): a state of the fragment
  `SeqPlain` (`SeqPure` with private transients allowed, any node kind inside
  map scopes) with at most one
  weakly connected component whose tree segment
  is the image of the component's top-level nodes in a topological order of
  the component (`SegCorr`).  The SDFG runs the component from the state's
  memory with an empty log and merges its log; the tree runs the segment on
  the current state.  Coherence (`OrderedExec.coh`: the memory after a
  run is the initial memory with its log applied) makes the two agree.

Control flow and whole programs: `Dace.Equiv.CorrCtl`.
-/

set_option autoImplicit false

namespace Dace

namespace Corr

/-! ## States -/

/-- The tree segment `cs` corresponds to state `s`. -/
structure StateCorr (ctx : Ctx) (tc : TCtx) (s : State) (cs : List SNode) : Prop where
  /-- Scope bodies bind no views (views inside map scopes are not matched
  yet). -/
  scopeViews : s.scopeViewsFree? ctx = true
  /-- The transients the state allocates are no streams (a fresh stream is
  empty, which the tree, allocating less, does not reproduce). -/
  noStreams : ∀ t ∈ ctx.stateTransients s, ctx.isStream t = false
  /-- The single component's tree segment: its views (`ViewNode`s, the
  prologue `bindViews`), then the image of its top-level nodes. -/
  comps : (s.components = [] ∧ cs = []) ∨
    ∃ c o full ks vs seg, s.components = [c] ∧ s.viewKeys ctx (s.viewOrder ctx (s.topLevelOf c)) = some ks ∧ ViewsCorr ks vs ∧
      cs = vs ++ seg ∧ SegCorr ctx tc s o seg ∧ s.isTopologicalOrder c full = true ∧
      full.filter (s.topLevelOf c).contains = o

theorem unionAll_single (g : Log) : Log.unionAll [g] = g := by
  cases g; simp [Log.unionAll]

/-- Allocating transients that are no streams makes the memory less
initialized (`Decisions.md#D68`). -/
theorem allocate_left_one (ctx : Ctx) (ts : List Name) (hs : ∀ t ∈ ts, ctx.isStream t = false) {m : Mem} :
    Mem.Sim 1 (fun _ => True) (fun _ => True) (ctx.allocate ts m) m :=
  allocate_left ctx (Mem.Sim.refl m) ts hs fun _ _ => trivial

section State

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL) (hcons : tc.consumed = none)
  (hw : ∀ op v, ctx.I.wcr op .undef v = .undef)

include he hcons hw in
/-- **Matching of a state**: the tree runs the segment, the SDFG the state;
undefined behavior, an error when the tree fails, or the same control-flow
state. -/
theorem StateCorr.matched {s : State} {cs : List SNode} (hsc : StateCorr ctx tc s cs) {full : List SNode} {gb : Bool}
    {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
    (hx : STree.ListExec tc full gb {} (cs ++ rest) cst r) :
    (∃ e, e.isUB = true ∧ StateExec ctx cst s (.err e)) ∨
    (∃ e e', r = .err e ∧ StateExec ctx cst s (.err e')) ∨
    ∃ cst', STree.ListExec tc full gb {} rest cst' r ∧ StateExec ctx cst s (.ok (.normal, cst')) := by
  have h := State.scopeViewsFree?_sound hsc.scopeViews
  rcases hsc.comps with ⟨hc, rfl⟩ | ⟨c, o, ofull, ks, vs, seg, hc, hks, hvs, rfl, hseg, hto, hdir⟩
  · -- no component: the state does nothing
    refine .inr (.inr ⟨cst, hx, ?_⟩)
    have := StateExec.run (ctx := ctx) (cst := cst) (s := s) (logs := []) (hc ▸ .nil)
    have hf : stateFinish ctx cst [] = .ok (.normal, cst) := by
      unfold stateFinish
      rw [merge_eq_apply ctx.I.wcr cst.mem (rfl : Log.pairwiseFree [] = true)]
      have hu : Log.unionAll [] = Log.empty := rfl
      rw [hu, Log.apply_empty, Log.seq_empty]
    rwa [hf] at this
  · let base : St := { mem := cst.mem }
    let abase : St := { mem := ctx.allocate (ctx.stateTransients s) cst.mem }
    have hcst : cst.st = base.shift cst.log := St.withLog_eq_shift base cst.log
    have hfold : ks.foldlM (viewStep ctx cst.ρ) cst.st = (ks.foldlM (viewStep ctx cst.ρ) base).map (St.shift cst.log) := by
      rw [hcst]
      exact List.foldlM_res_comm _ _ (fun b k => aliasTo_shift ctx cst.ρ cst.log b k.d k.base k.sub) _ _
    have hpv : bindViews ctx cst.ρ s (s.viewOrder ctx (s.topLevelOf c)) base = ks.foldlM (viewStep ctx cst.ρ) base :=
      bindViews_keys hks base
    -- the component run from the allocated memory simulates the one from the state's memory
    have hab : St.Sim 1 (fun _ => True) (fun _ => True) abase base :=
      ⟨Corr.allocate_left_one ctx (ctx.stateTransients s) hsc.noStreams, (Log.ren_one _).symm, rfl, Log.NamesP.true _⟩
    have hstate : ∀ {rc : Res St}, OrderedExec ctx cst.ρ s c (s.topLevelOf c) base rc →
        (∀ e, rc = .err e → ∃ e', StateExec ctx cst s (.err e') ∧ (e'.isUB = true ∨ e' = e)) ∧
        (∀ st'', rc = .ok st'' → (∃ e', e'.isUB = true ∧ StateExec ctx cst s (.err e')) ∨ StateExec ctx cst s
          (.ok (.normal, { cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log }))) := by
      intro rc hrc
      obtain ⟨r₂, hr₂, hsim⟩ := OrderedExec.sim (CtxSim.refl ctx) hw (SNames.refl ctx s) hrc abase hab
      refine ⟨fun e he' => ?_, fun st'' hok => ?_⟩
      · subst he'
        obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hsim
        exact ⟨e', .compsErr (by rw [hc]; exact CompsExec.headErr hr₂), hu⟩
      · subst hok
        rcases ResSim.ok_cases hsim with ⟨e, rfl, hu⟩ | ⟨st₂, rfl, hs₂⟩
        · exact .inl ⟨e, hu, .compsErr (by rw [hc]; exact CompsExec.headErr hr₂)⟩
        · refine .inr ?_
          have := StateExec.run (ctx := ctx) (cst := cst) (s := s) (logs := [st₂.log])
            (by rw [hc]; exact CompsExec.cons hr₂ .nil)
          have hlog : st₂.log = st''.log := by rw [hs₂.log, Log.ren_one]
          have hf : stateFinish ctx cst [st₂.log] =
              .ok (.normal, { cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log }) := by
            simp only [stateFinish, hlog]
            rw [merge_eq_apply ctx.I.wcr cst.mem (by simp [Log.pairwiseFree]), unionAll_single]
          rwa [hf] at this
    rw [List.append_assoc] at hx
    rcases views_run he hvs hx with ⟨e, rfl, hf⟩ | ⟨cst₁, hx₁, hf, hρ₁⟩
    · -- a view binding fails
      rw [hfold] at hf
      cases hb : ks.foldlM (viewStep ctx cst.ρ) base with
      | ok _ => rw [hb] at hf; cases hf
      | err e' =>
        rw [hb] at hf
        cases hf
        obtain ⟨e'', hx', hu'⟩ := (hstate (.prologueErr (hpv.trans hb))).1 _ rfl
        rcases hu' with hu' | rfl
        · exact .inl ⟨_, hu', hx'⟩
        · exact .inr (.inl ⟨_, _, rfl, hx'⟩)
    rw [hfold] at hf
    cases hb : ks.foldlM (viewStep ctx cst.ρ) base with
    | err _ => rw [hb] at hf; cases hf
    | ok base₁ =>
    rw [hb] at hf
    simp only [Res.map, Res.ok.injEq] at hf
    have hp : bindViews ctx cst.ρ s (s.viewOrder ctx (s.topLevelOf c)) base = .ok base₁ := hpv.trans hb
    have hrel : StRel cst₁.st cst₁ := ⟨rfl, rfl, rfl⟩
    have hst₁ : cst₁.st = base₁.shift cst.log := hf.symm
    rcases SegCorr.matched h he hw hseg hx₁ hrel with ⟨e, hu, hs⟩ | ⟨e, e₀, hr, hs⟩ | ⟨cst', st', tr, hl, hs, hρ, hrel'⟩
    · rw [hst₁, hρ₁] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.shift_iff cst.log).1 hs
      cases r₀ with
      | ok _ => cases he₀
      | err e' =>
        cases he₀
        rw [← hdir] at hs₀
        obtain ⟨e'', hx', hu'⟩ := (hstate (.seqErr hto hp hs₀)).1 _ rfl
        rcases hu' with hu' | rfl
        · exact .inl ⟨_, hu', hx'⟩
        · exact .inl ⟨_, hu, hx'⟩
    · rw [hst₁, hρ₁] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.shift_iff cst.log).1 hs
      cases r₀ with
      | ok _ => cases he₀
      | err e' =>
        cases he₀
        rw [← hdir] at hs₀
        obtain ⟨e'', hx', hu'⟩ := (hstate (.seqErr hto hp hs₀)).1 _ rfl
        rcases hu' with hu' | rfl
        · exact .inl ⟨_, hu', hx'⟩
        · exact .inr (.inl ⟨_, _, hr, hx'⟩)
    · rw [hst₁, hρ₁] at hs
      obtain ⟨r₀, hs₀, he₀⟩ := (SeqExec.shift_iff cst.log).1 hs
      cases r₀ with
      | err _ => cases he₀
      | ok p =>
        obtain ⟨st'', tr''⟩ := p
        simp only [Res.map, Prod.map, Res.ok.injEq, Prod.mk.injEq, id] at he₀
        obtain ⟨rfl, rfl⟩ := he₀
        rw [← hdir] at hs₀
        cases hf : s.hbFree tr with
        | false =>
          obtain ⟨e'', hx', hu'⟩ := (hstate (.hbRace hto hp hs₀ hf)).1 _ rfl
          rcases hu' with hu' | rfl
          · exact .inl ⟨_, hu', hx'⟩
          · exact .inl ⟨_, rfl, hx'⟩
        | true =>
          have hrun : OrderedExec ctx cst.ρ s c (s.topLevelOf c) base (.ok st'') := .run hto hp hs₀ hf
          have hflat : ∀ id ∈ s.topLevelOf c, ∀ n, s.node? id = some n → n.FlatIn s := by
            intro id hid n hn
            have hc' : id ∈ c := (List.mem_filter.1 hid).1
            have hfull : id ∈ ofull := by
              simp only [State.isTopologicalOrder, Bool.and_eq_true, List.all_eq_true] at hto
              simpa using hto.1.1.2 id hc'
            exact hseg.flat id (hdir ▸ List.mem_filter.2 ⟨hfull, by simpa using hid⟩) n hn
          have hcoh := OrderedExec.coherent' (fun id hid n hn => .inl (hflat id hid n hn)) (st := base) hrun
          rcases (hstate hrun).2 _ rfl with ⟨e, hu, hx'⟩ | this
          · exact .inl ⟨e, hu, hx'⟩
          refine .inr (.inr ⟨cst', hl, ?_⟩)
          have hcst : ({ cst with mem := st''.log.apply ctx.I.wcr cst.mem, log := cst.log.seq st''.log } : CSt) =
              cst' := by
            obtain ⟨hm, hlg, -⟩ := hrel'
            cases cst'
            simp only [St.shift_mem, St.shift_log] at hm hlg
            simp only at hρ
            simp only [CSt.mk.injEq]
            exact ⟨(hρ.trans hρ₁).symm, by rw [← hm, hcoh], hlg⟩
          rwa [hcst] at this

end State

end Corr

end Dace
