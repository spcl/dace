-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Congr
import Dace.Equiv.Relabel

/-!
# Rules on control-flow regions

* **Region congruence** (`Congr.regionEquiv`): replacing the region reached
  by a path (the top-level region, or the region of a nested region block,
  loop body or branch) by a region whose runs match the old one's in both
  directions gives an equivalent SDFG, provided every transient with `Scope`
  or `State` lifetime keeps its mention count (`sameRegionMentions?`).
* **Block renumbering** (`Congr.relabelBlocks`): the blocks of a region may
  be renumbered by any permutation `φ` of block ids that maps the region's
  ids onto themselves (`Region.relabelBlocks`, side condition
  `Region.blocksPermOk`: every id in range, start and edges in range).  Runs
  correspond with equal outcomes, except that an ambiguous transition names
  the block (both undefined behavior).
-/

set_option autoImplicit false

namespace Dace

/-! ## Paths to regions -/

namespace Region

/-- The region reached by a path (`[]`: the region itself). -/
def regionAt? (r : Region) : List PathStep → Option Region
  | [] => some r
  | st :: p => match r.child? st with
    | some c => c.regionAt? p
    | none => none

/-- Replace the region reached by a path. -/
def setRegionAt (r : Region) : List PathStep → Region → Region
  | [], c => c
  | st :: p, c => match r.child? st with
    | some c₀ => r.setChild (c₀.setRegionAt p c) st
    | none => r

theorem regionAt?_setRegionAt {c c' : Region} :
    ∀ (p : List PathStep) (r : Region), r.regionAt? p = some c → (r.setRegionAt p c').regionAt? p = some c'
  | [], _, _ => rfl
  | st :: p, r, h => by
    simp only [regionAt?] at h
    split at h
    · rename_i c₀ hc
      simp only [setRegionAt, hc, regionAt?, child?_setChild hc]
      exact regionAt?_setRegionAt p c₀ h
    · cases h

theorem setRegionAt_setRegionAt {c c' : Region} :
    ∀ (p : List PathStep) (r : Region), r.regionAt? p = some c → (r.setRegionAt p c').setRegionAt p c = r
  | [], _, h => by cases h; rfl
  | st :: p, r, h => by
    simp only [regionAt?] at h
    split at h
    · rename_i c₀ hc
      simp only [setRegionAt, hc, child?_setChild hc, setRegionAt_setRegionAt p c₀ h]
      exact setChild_setChild hc
    · cases h

end Region

/-- Replace the region of an SDFG reached by a path. -/
def SDFG.setRegionAt (g : SDFG) (p : List PathStep) (c : Region) : SDFG := { g with top := g.top.setRegionAt p c }

theorem SDFG.interface_setRegionAt (g : SDFG) (p : List PathStep) (c : Region) :
    (g.setRegionAt p c).interface = g.interface := rfl

theorem SDFG.facts_setRegionAt (g : SDFG) (p : List PathStep) (c : Region) :
    (g.setRegionAt p c).facts = g.facts := rfl

theorem Congr.region_setRegionAt {ctx : Ctx} {c' c : Region} (hs : RegionMatchedE ctx c' c) :
    ∀ (p : List PathStep) (r : Region), r.regionAt? p = some c → RegionMatchedE ctx (r.setRegionAt p c') r
  | [], _, h => by cases h; exact hs
  | st :: p, r, h => by
    simp only [Region.regionAt?] at h
    split at h
    · rename_i c₀ hc
      simp only [Region.setRegionAt, hc]
      exact RegionMatchedE.setChild hc (Congr.region_setRegionAt hs p c₀ h)
    · cases h

theorem Region.countMentions_setRegionAt (t : Name) {c c' : Region}
    (hx : Region.countMentions t c' = Region.countMentions t c) :
    ∀ (p : List PathStep) (r : Region), r.regionAt? p = some c →
      Region.countMentions t (r.setRegionAt p c') = Region.countMentions t r
  | [], _, h => by cases h; exact hx
  | st :: p, r, h => by
    simp only [Region.regionAt?] at h
    split at h
    · rename_i c₀ hc
      simp only [Region.setRegionAt, hc]
      exact Region.countMentions_setChild t hc (Region.countMentions_setRegionAt t hx p c₀ h)
    · cases h

/-! ## Contexts agreeing on mention counts -/

/-- Contexts over SDFGs with the same descriptors and the same mention counts
of every `Scope`/`State` transient agree. -/
theorem Ctx.agree_of_counts {I : Interp} {cn sc : Name → Name} {g' g : SDFG} (ha : g'.arrays = g.arrays)
    (hc : ∀ t d, (t, d) ∈ g.arrays → (d.transient && (d.lifetime == .scope || d.lifetime == .state)) = true →
      g'.countMentions t = g.countMentions t) :
    Ctx.Agree { I, g := g', cname := cn, scope := sc } { I, g, cname := cn, scope := sc } := by
  have hdesc : ∀ t, ({ I, g := g', cname := cn, scope := sc } : Ctx).desc? t =
      ({ I, g, cname := cn, scope := sc } : Ctx).desc? t := fun t => by
    simp only [Ctx.desc?, SDFG.array?, ha]
  have hmem : ∀ {t : Name} {d : DataDesc}, ({ I, g, cname := cn, scope := sc } : Ctx).desc? t = some d →
      (t, d) ∈ g.arrays := by
    intro t d h
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at h
    obtain ⟨⟨t', d'⟩, hf, rfl⟩ := h
    have := List.find?_some hf
    simp only [decide_eq_true_eq] at this
    subst this
    exact List.mem_of_find?_eq_some hf
  have hnames : ({ I, g := g', cname := cn, scope := sc } : Ctx).containerNames =
      ({ I, g, cname := cn, scope := sc } : Ctx).containerNames := by
    simp only [Ctx.containerNames, ha]
  refine ⟨Ctx.erase_eq_of rfl rfl rfl ha, fun s entry => ?_, fun s => ?_⟩
  · unfold Ctx.privateTransients
    rw [hnames]
    refine List.filterMap_congr fun t _ => ?_
    simp only
    rw [hdesc t]
    cases hdt : ({ I, g, cname := cn, scope := sc } : Ctx).desc? t with
    | none => rfl
    | some d =>
      simp only
      split
      · rename_i hsc
        have hd : (d.transient && (d.lifetime == .scope || d.lifetime == .state)) = true := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hsc
          simp [hsc.1, hsc.2]
        rw [hc t d (hmem hdt) hd]
      · rfl
  · unfold Ctx.stateTransients
    rw [hnames]
    refine List.filterMap_congr fun t _ => ?_
    simp only
    rw [hdesc t]
    cases hdt : ({ I, g, cname := cn, scope := sc } : Ctx).desc? t with
    | none => rfl
    | some d =>
      simp only
      split
      · rename_i hd
        rw [hc t d (hmem hdt) hd]
      · rfl

/-- Is every `Scope`/`State` transient of `g` mentioned equally often by the
two regions? -/
def sameRegionMentions? (g : SDFG) (c' c : Region) : Bool :=
  g.arrays.all fun (t, d) =>
    !(d.transient && (d.lifetime == .scope || d.lifetime == .state)) ||
      Region.countMentions t c' == Region.countMentions t c

theorem sameRegionMentions?_count {g : SDFG} {c' c : Region} (h : sameRegionMentions? g c' c = true)
    {t : Name} {d : DataDesc} (hmem : (t, d) ∈ g.arrays)
    (hd : (d.transient && (d.lifetime == .scope || d.lifetime == .state)) = true) :
    Region.countMentions t c' = Region.countMentions t c := by
  unfold sameRegionMentions? at h
  have := List.all_eq_true.1 h (t, d) hmem
  simp only [hd, Bool.not_true, Bool.false_or, beq_iff_eq] at this
  exact this

theorem Ctx.agree_setRegionAt {I : Interp} {cn sc : Name → Name} {g : SDFG} {p : List PathStep} {c' c : Region}
    (hat : g.top.regionAt? p = some c) (hm : sameRegionMentions? g c' c = true) :
    Ctx.Agree { I, g := g.setRegionAt p c', cname := cn, scope := sc } { I, g, cname := cn, scope := sc } :=
  Ctx.agree_of_counts rfl fun t _ hmem hd =>
    Region.countMentions_setRegionAt t (sameRegionMentions?_count hm hmem hd) p g.top hat

theorem sameRegionMentions?_symm {g g' : SDFG} {c' c : Region} (ha : g'.arrays = g.arrays)
    (h : sameRegionMentions? g c' c = true) : sameRegionMentions? g' c c' = true := by
  unfold sameRegionMentions? at h ⊢
  rw [ha]
  refine List.all_eq_true.2 fun x hx => ?_
  have := List.all_eq_true.1 h x hx
  simp only [Bool.or_eq_true, beq_iff_eq] at this ⊢
  exact this.imp id Eq.symm

/-! ## Region congruence -/

/-- From matching from equal start states, by the log-shift lemmas. -/
theorem RegionMatchedE.of_same {ctx : Ctx} {c' c : Region}
    (h : ∀ cst r', RegionExec ctx cst c' r' → MatchedE CtlRel (RegionExec ctx cst c) r') : RegionMatchedE ctx c' c := by
  intro a' a ha r' hr'
  obtain ⟨r, hr, hrel⟩ := (RegionExec.rel ha).2 r' hr'
  exact MatchedE.of_rel_left CtlRel.trans' hrel (h a r hr)

theorem Congr.exec_matchedE_region {g : SDFG} {p : List PathStep} {c' c : Region}
    (hat : g.top.regionAt? p = some c) (hm : sameRegionMentions? g c' c = true) {I : Interp}
    (hs : RegionMatchedE { I, g } c' c) {ρ₀ : Valuation} {m₀ : Mem} {r : Res Mem}
    (hr : SDFG.Exec I (g.setRegionAt p c') ρ₀ m₀ r) : MatchedE Eq (SDFG.Exec I g ρ₀ m₀) r := by
  obtain ⟨res, hres, rfl⟩ := hr
  have hres := SDFGExec.congr (Ctx.agree_setRegionAt (I := I) (cn := id) (sc := id) hat hm) hres
  have hreg := Congr.region_setRegionAt hs p g.top hat
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

/-- **Region congruence, both directions**: regions that match each other's
runs give equivalent SDFGs. -/
theorem Congr.regionEquiv {g : SDFG} {p : List PathStep} {c' c : Region}
    (hat : g.top.regionAt? p = some c) (hm : sameRegionMentions? g c' c = true)
    (hs : ∀ I : Interp, I.Lawful → ∀ cst r', RegionExec { I, g } cst c' r' →
      MatchedE CtlRel (RegionExec { I, g } cst c) r')
    (hs' : ∀ I : Interp, I.Lawful → ∀ cst r, RegionExec { I, g } cst c r →
      MatchedE CtlRel (RegionExec { I, g } cst c') r)
    (hsup : Supported g.prog) : Equiv (g.setRegionAt p c').prog g.prog := by
  have hat' : (g.setRegionAt p c').top.regionAt? p = some c' := Region.regionAt?_setRegionAt p g.top hat
  have hm' := sameRegionMentions?_symm (g' := g.setRegionAt p c') rfl hm
  have hg : (g.setRegionAt p c').setRegionAt p c = g := by
    obtain ⟨_, _, _, _, _, top, _⟩ := g
    simp only [SDFG.setRegionAt, Region.setRegionAt_setRegionAt p top hat]
  have fwd : ∀ I ρ₀ m₀, (g.setRegionAt p c').prog.Pre I ρ₀ m₀ → ∀ r,
      (g.setRegionAt p c').prog.exec I ρ₀ m₀ r → MatchedE Eq (g.prog.exec I ρ₀ m₀) r :=
    fun I _ _ hp _ hr => Congr.exec_matchedE_region hat hm (RegionMatchedE.of_same (hs I hp.1)) hr
  have bwd : ∀ I ρ₀ m₀, (g.setRegionAt p c').prog.Pre I ρ₀ m₀ → ∀ r,
      g.prog.exec I ρ₀ m₀ r → MatchedE Eq ((g.setRegionAt p c').prog.exec I ρ₀ m₀) r := by
    intro I ρ₀ m₀ hp r hr
    have hA := Ctx.agree_setRegionAt (I := I) (cn := id) (sc := id) hat hm
    have hsI : RegionMatchedE { I, g := g.setRegionAt p c' } c c' :=
      RegionMatchedE.of_same fun cst r hr =>
        (hs' I hp.1 cst r (RegionExec.congr hA hr)).mono (fun _ _ h => h) (fun _ h => RegionExec.congr hA.symm h)
    have hr' : SDFG.Exec I ((g.setRegionAt p c').setRegionAt p c) ρ₀ m₀ r := by rw [hg]; exact hr
    exact Congr.exec_matchedE_region hat' hm' hsI hr'
  have hsup' : Supported (g.setRegionAt p c').prog :=
    Supported.of_matchedE hsup (fun _ _ _ h => h) fwd bwd
  refine ⟨Refines.of_matchedE hsup fun I ρ₀ m₀ hp r hr => (fwd I ρ₀ m₀ hp r hr).mono
      (fun _ _ h => h ▸ obsEq.refl _ _) (fun _ h => h),
    Refines.of_matchedE hsup' fun I ρ₀ m₀ hp r hr => (bwd I ρ₀ m₀ hp r hr).mono
      (fun _ _ h => h ▸ obsEq.refl _ _) (fun _ h => h)⟩

/-! ## Block renumbering -/

namespace Region

/-- An interstate edge with its endpoints renumbered. -/
def relabelEdge (φ : Equiv.Perm Nat) (e : IEdge) : IEdge := { e with src := φ e.src, dst := φ e.dst }

/-- The region with block `i` moved to `φ i`. -/
def relabelBlocks (r : Region) (φ : Equiv.Perm Nat) : Region :=
  { r with
    blocks := (List.range r.blocks.length).map fun k => (r.blocks[φ.symm k]?).getD .returnBlock
    start := φ r.start
    edges := r.edges.map (relabelEdge φ) }

/-- The side condition of block renumbering: `φ` maps the block ids of the
region onto themselves, and the start block and every edge are in range. -/
def blocksPermOk (r : Region) (φ : Equiv.Perm Nat) : Bool :=
  (List.range r.blocks.length).all (fun k => φ k < r.blocks.length && φ.symm k < r.blocks.length) &&
  decide (r.start < r.blocks.length) && r.edges.all (fun e => decide (e.src < r.blocks.length) && decide (e.dst < r.blocks.length))

variable {r : Region} {φ : Equiv.Perm Nat}

theorem relabelBlocks_length : (r.relabelBlocks φ).blocks.length = r.blocks.length := by
  simp [relabelBlocks]

structure PermOk (r : Region) (φ : Equiv.Perm Nat) : Prop where
  fwd : ∀ k < r.blocks.length, φ k < r.blocks.length
  bwd : ∀ k < r.blocks.length, φ.symm k < r.blocks.length
  start : r.start < r.blocks.length
  edges : ∀ e ∈ r.edges, e.src < r.blocks.length ∧ e.dst < r.blocks.length

theorem blocksPermOk_sound (h : r.blocksPermOk φ = true) : r.PermOk φ := by
  unfold blocksPermOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  obtain ⟨⟨h₁, h₂⟩, h₃⟩ := h
  exact ⟨fun k hk => (h₁ k hk).1, fun k hk => (h₁ k hk).2, h₂, h₃⟩

theorem block?_relabel (hok : r.PermOk φ) {i : BlockId} (hi : i < r.blocks.length) :
    (r.relabelBlocks φ).block? (φ i) = r.block? i := by
  simp only [block?, relabelBlocks, List.getElem?_map, List.getElem?_range (hok.fwd i hi), Option.map_some,
    Equiv.symm_apply_apply]
  rw [List.getElem?_eq_getElem hi]
  rfl

theorem outEdges_relabel (i : BlockId) : (r.relabelBlocks φ).outEdges (φ i) = (r.outEdges i).map (relabelEdge φ) := by
  simp only [outEdges, relabelBlocks, List.filter_map]
  congr 1
  refine List.filter_congr fun e _ => ?_
  simp [relabelEdge, φ.injective.eq_iff]

end Region

/-- One out-edge, kept when its condition holds (`trueEdges`). -/
def edgeIfTrue (ctx : Ctx) (cst : CSt) (e : IEdge) : Res (Option IEdge) := do
  let b ← ctx.evalBoolR cst.ρ cst.mem e.cond
  pure (if b then some e else none)

theorem trueEdges_eq (ctx : Ctx) (r : Region) (cst : CSt) (bid : BlockId) :
    trueEdges ctx r cst bid = (r.outEdges bid).filterMapM (edgeIfTrue ctx cst) := rfl

theorem trueEdges_map_relabel (ctx : Ctx) (cst : CSt) (φ : Equiv.Perm Nat) :
    ∀ l : List IEdge, (l.map (Region.relabelEdge φ)).filterMapM (edgeIfTrue ctx cst) =
      (l.filterMapM (edgeIfTrue ctx cst)).map (List.map (Region.relabelEdge φ))
  | [] => rfl
  | e :: es => by
    simp only [List.map_cons, List.filterMapM_cons, edgeIfTrue]
    have ih := trueEdges_map_relabel ctx cst φ es
    simp only [Region.relabelEdge] at ih ⊢
    cases ctx.evalBoolR cst.ρ cst.mem e.cond with
    | err _ => rfl
    | ok b =>
      cases b
      · exact ih
      · show (Res.bind _ _) = Res.map _ (Res.bind _ _)
        rw [ih]
        cases List.filterMapM (edgeIfTrue ctx cst) es <;> rfl

theorem trueEdges_relabel (ctx : Ctx) (r : Region) (cst : CSt) (φ : Equiv.Perm Nat) (i : BlockId) :
    trueEdges ctx (r.relabelBlocks φ) cst (φ i) = (trueEdges ctx r cst i).map (List.map (Region.relabelEdge φ)) := by
  rw [trueEdges_eq, trueEdges_eq, Region.outEdges_relabel]
  exact trueEdges_map_relabel ctx cst φ _

theorem filterMapM_cond_sub (ctx : Ctx) (cst : CSt) :
    ∀ (l es : List IEdge), l.filterMapM (edgeIfTrue ctx cst) = .ok es → ∀ e ∈ es, e ∈ l
  | [], es, h, e, he => by cases h; cases he
  | x :: xs, es, h, e, he => by
    simp only [List.filterMapM_cons, edgeIfTrue] at h
    cases hb : ctx.evalBoolR cst.ρ cst.mem x.cond with
    | err _ => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      cases b
      · exact List.mem_cons_of_mem _ (filterMapM_cond_sub ctx cst xs es h e he)
      · change Res.bind _ _ = _ at h
        cases hr : List.filterMapM (edgeIfTrue ctx cst) xs with
        | err _ => rw [hr] at h; cases h
        | ok rest =>
          rw [hr] at h
          cases h
          rcases List.mem_cons.1 he with rfl | he
          · exact List.mem_cons_self ..
          · exact List.mem_cons_of_mem _ (filterMapM_cond_sub ctx cst xs rest hr e he)

theorem trueEdges_mem_edges {ctx : Ctx} {r : Region} {cst : CSt} {i : BlockId} {es : List IEdge}
    (h : trueEdges ctx r cst i = .ok es) : ∀ e ∈ es, e ∈ r.edges := fun e he =>
  (List.mem_filter.1 (filterMapM_cond_sub ctx cst _ es (trueEdges_eq .. ▸ h) e he)).1

/-- **Renumbered regions run alike** (an ambiguous transition names the
renumbered block: both are undefined behavior). -/
theorem RegionAt.relabelBlocks {ctx : Ctx} {r : Region} {φ : Equiv.Perm Nat} (hok : r.PermOk φ) {cst : CSt}
    {i : BlockId} {res : Res (Ctl × CSt)} :
    RegionAt ctx r cst i res → i < r.blocks.length →
      ∃ res₂, RegionAt ctx (r.relabelBlocks φ) cst (φ i) res₂ ∧ ResSim Eq res₂ res
  | .missing hb, hi => by simp [Region.block?, List.getElem?_eq_getElem hi] at hb
  | .blockErr hb h, hi => ⟨_, .blockErr (by rw [Region.block?_relabel hok hi]; exact hb) h, ResSim.err _⟩
  | .exit hb h hc, hi => ⟨_, .exit (by rw [Region.block?_relabel hok hi]; exact hb) h hc, ResSim.of_eq fun _ => rfl⟩
  | .edgesErr hb h he, hi =>
    ⟨_, .edgesErr (by rw [Region.block?_relabel hok hi]; exact hb) h
      (by rw [trueEdges_relabel, he]; rfl), ResSim.err _⟩
  | .stop hb h he, hi =>
    ⟨_, .stop (by rw [Region.block?_relabel hok hi]; exact hb) h
      (by rw [trueEdges_relabel, he]; rfl), ResSim.of_eq fun _ => rfl⟩
  | .ambiguous hb h he, hi =>
    ⟨_, .ambiguous (by rw [Region.block?_relabel hok hi]; exact hb) h
      (by rw [trueEdges_relabel, he]; rfl), ResSim.ub rfl _⟩
  | .assignErr (ie := ie) hb h he ha, hi =>
    ⟨_, .assignErr (ie := Region.relabelEdge φ ie) (by rw [Region.block?_relabel hok hi]; exact hb) h
      (by rw [trueEdges_relabel, he]; rfl) ha, ResSim.err _⟩
  | .step (ie := ie) hb h he ha hn, hi => by
    have hd : ie.dst < r.blocks.length := (hok.edges ie (trueEdges_mem_edges he ie (List.mem_singleton_self _))).2
    obtain ⟨res₂, hr₂, hs⟩ := RegionAt.relabelBlocks hok hn hd
    exact ⟨res₂, .step (ie := Region.relabelEdge φ ie) (by rw [Region.block?_relabel hok hi]; exact hb) h
      (by rw [trueEdges_relabel, he]; rfl) ha hr₂, hs⟩

namespace Region

variable {r : Region} {φ : Equiv.Perm Nat}

theorem PermOk.relabel (hok : r.PermOk φ) : (r.relabelBlocks φ).PermOk φ.symm := by
  refine ⟨fun k hk => ?_, fun k hk => ?_, ?_, fun e he => ?_⟩
  · rw [relabelBlocks_length] at hk ⊢; exact hok.bwd k hk
  · rw [relabelBlocks_length] at hk ⊢; simpa using hok.fwd k hk
  · rw [relabelBlocks_length]; exact hok.fwd _ hok.start
  · rw [relabelBlocks_length]
    simp only [relabelBlocks, List.mem_map] at he
    obtain ⟨e₀, he₀, rfl⟩ := he
    obtain ⟨h₁, h₂⟩ := hok.edges e₀ he₀
    exact ⟨hok.fwd _ h₁, hok.fwd _ h₂⟩

theorem relabelBlocks_symm (hok : r.PermOk φ) : (r.relabelBlocks φ).relabelBlocks φ.symm = r := by
  obtain ⟨label, kind, blocks, start, edges⟩ := r
  simp only [relabelBlocks, List.length_map, List.length_range, Equiv.symm_symm, Equiv.symm_apply_apply,
    List.map_map, mk.injEq, true_and]
  refine ⟨List.ext_getElem? fun k => ?_, ?_⟩
  · by_cases hk : k < blocks.length
    · have hφ := hok.fwd k hk
      simp only [List.getElem?_map, List.getElem?_range hk, List.getElem?_range hφ, Option.map_some,
        Equiv.symm_apply_apply, Option.getD_some]
      rw [List.getElem?_eq_getElem hk]
      rfl
    · simp [List.getElem?_map, hk]
  · conv_rhs => rw [← List.map_id edges]
    refine List.map_congr_left fun e _ => ?_
    simp [relabelEdge]

end Region

/-- Runs of the renumbered region are matched by runs of the original. -/
theorem RegionExec.relabelBlocks_matched {ctx : Ctx} {r : Region} {φ : Equiv.Perm Nat} (hok : r.PermOk φ)
    {cst : CSt} {res : Res (Ctl × CSt)} (h : RegionExec ctx cst (r.relabelBlocks φ) res) :
    MatchedE CtlRel (RegionExec ctx cst r) res := by
  cases h with
  | start h =>
    have hok' := hok.relabel
    obtain ⟨res₂, hr, hs⟩ := RegionAt.relabelBlocks hok' h hok'.start
    rw [Region.relabelBlocks_symm hok] at hr
    have hst : φ.symm (r.relabelBlocks φ).start = r.start := by simp [Region.relabelBlocks]
    rw [hst] at hr
    refine ⟨res₂, .start hr, ?_⟩
    rcases hs with ⟨e, rfl, hu⟩ | ⟨e, rfl, rfl⟩ | ⟨a, b, rfl, rfl, rfl⟩
    · exact .inl ⟨e, rfl, hu⟩
    · exact .inr rfl
    · exact .inr (CtlRel.refl _)

/-- Runs of the original region are matched by runs of the renumbered one. -/
theorem RegionExec.relabelBlocks_matched' {ctx : Ctx} {r : Region} {φ : Equiv.Perm Nat} (hok : r.PermOk φ)
    {cst : CSt} {res : Res (Ctl × CSt)} (h : RegionExec ctx cst r res) :
    MatchedE CtlRel (RegionExec ctx cst (r.relabelBlocks φ)) res := by
  have hok' := hok.relabel
  have h' : RegionExec ctx cst ((r.relabelBlocks φ).relabelBlocks φ.symm) res := by
    rw [Region.relabelBlocks_symm hok]; exact h
  exact RegionExec.relabelBlocks_matched hok' h'

/-- **Block renumbering rule**: renumbering the blocks of the region at path
`p` by a permutation that maps its block ids onto themselves gives an
equivalent SDFG. -/
theorem Congr.relabelBlocks {g : SDFG} {p : List PathStep} {c : Region} {φ : Equiv.Perm Nat}
    (hat : g.top.regionAt? p = some c) (hok : c.blocksPermOk φ = true)
    (hm : sameRegionMentions? g (c.relabelBlocks φ) c = true) (hsup : Supported g.prog) :
    Equiv (g.setRegionAt p (c.relabelBlocks φ)).prog g.prog :=
  Congr.regionEquiv hat hm (fun _ _ _ _ h => RegionExec.relabelBlocks_matched (Region.blocksPermOk_sound hok) h)
    (fun _ _ _ _ h => RegionExec.relabelBlocks_matched' (Region.blocksPermOk_sound hok) h) hsup

/-! ## Inserting and removing an empty state -/

/-- Out-edges kept when their condition holds, after a map that keeps
conditions. -/
theorem filterMapM_edgeIfTrue_map (ctx : Ctx) (cst : CSt) (f : IEdge → IEdge) (hf : ∀ e, (f e).cond = e.cond) :
    ∀ l : List IEdge, (l.map f).filterMapM (edgeIfTrue ctx cst) =
      (l.filterMapM (edgeIfTrue ctx cst)).map (List.map f)
  | [] => rfl
  | e :: es => by
    simp only [List.map_cons, List.filterMapM_cons, edgeIfTrue, hf]
    have ih := filterMapM_edgeIfTrue_map ctx cst f hf es
    cases ctx.evalBoolR cst.ρ cst.mem e.cond with
    | err _ => rfl
    | ok b =>
      cases b
      · exact ih
      · show (Res.bind _ _) = Res.map _ (Res.bind _ _)
        rw [ih]
        cases List.filterMapM (edgeIfTrue ctx cst) es <;> rfl

namespace Region

/-- Move an edge leaving `j` to leave `n` instead. -/
def moveSrc (j n : BlockId) (e : IEdge) : IEdge := if e.src = j then { e with src := n } else e

/-- Insert the empty state `e` after block `j`: it becomes the last block,
takes over the out-edges of `j`, and `j` flows into it unconditionally. -/
def insertEmpty (r : Region) (j : BlockId) (e : State) : Region :=
  { r with
    blocks := r.blocks ++ [.state e]
    edges := r.edges.map (moveSrc j r.blocks.length) ++ [{ src := j, dst := r.blocks.length }] }

/-- The side condition of `insertEmpty`. -/
structure InsertOk (r : Region) (j : BlockId) (e : State) : Prop where
  j : j < r.blocks.length
  start : r.start < r.blocks.length
  edges : ∀ x ∈ r.edges, x.src < r.blocks.length ∧ x.dst < r.blocks.length
  empty : e.nodes = []

def insertOk? (r : Region) (j : BlockId) (e : State) : Bool :=
  decide (j < r.blocks.length) && decide (r.start < r.blocks.length) &&
  r.edges.all (fun x => decide (x.src < r.blocks.length) && decide (x.dst < r.blocks.length)) && e.nodes.isEmpty

theorem insertOk?_sound {r : Region} {j : BlockId} {e : State} (h : r.insertOk? j e = true) : r.InsertOk j e := by
  unfold insertOk? at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.isEmpty_iff] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

variable {r : Region} {j : BlockId} {e : State}

theorem insert_block?_lt {b : BlockId} (hb : b < r.blocks.length) : (r.insertEmpty j e).block? b = r.block? b := by
  simp [insertEmpty, block?, List.getElem?_append_left hb]

theorem insert_block?_n : (r.insertEmpty j e).block? r.blocks.length = some (.state e) := by
  simp [insertEmpty, block?]

theorem insert_block?_j (hok : r.InsertOk j e) : ∃ bj, r.block? j = some bj :=
  ⟨_, List.getElem?_eq_getElem hok.j⟩

theorem moveSrc_src (n : BlockId) (x : IEdge) : (moveSrc j n x).src = if x.src = j then n else x.src := by
  unfold moveSrc; split <;> rfl

theorem moveSrc_of_ne {n : BlockId} {x : IEdge} (h : x.src ≠ j) : moveSrc j n x = x := by
  simp [moveSrc, h]

theorem insert_outEdges_j (hok : r.InsertOk j e) :
    (r.insertEmpty j e).outEdges j = [{ src := j, dst := r.blocks.length }] := by
  simp only [insertEmpty, outEdges, List.filter_append, List.filter_map]
  have : List.filter ((fun x => decide (x.src = j)) ∘ moveSrc j r.blocks.length) r.edges = [] :=
    List.filter_eq_nil_iff.2 fun x hx => by
      have := (hok.edges x hx).1
      have hj := hok.j
      simp only [Function.comp_def, moveSrc_src, decide_eq_true_eq]
      by_cases hs : x.src = j
      · rw [if_pos hs]; exact Nat.ne_of_gt hj
      · rw [if_neg hs]; exact hs
  rw [this]
  simp

theorem insert_outEdges_lt (_hok : r.InsertOk j e) {b : BlockId} (hbj : b ≠ j) (hb : b < r.blocks.length) :
    (r.insertEmpty j e).outEdges b = r.outEdges b := by
  simp only [insertEmpty, outEdges, List.filter_append, List.filter_map]
  have h₁ : List.filter (fun x => decide (x.src = b)) [({ src := j, dst := r.blocks.length } : IEdge)] = [] := by
    simp [Ne.symm hbj]
  rw [h₁, List.append_nil]
  have : ∀ x ∈ r.edges, ((fun x => decide (x.src = b)) ∘ moveSrc j r.blocks.length) x = decide (x.src = b) := by
    intro x hxm
    simp only [Function.comp_def, moveSrc_src]
    by_cases hs : x.src = j
    · rw [if_pos hs, hs]
      have : ¬ r.blocks.length = b := fun h => Nat.lt_irrefl _ (h ▸ hb)
      have : ¬ j = b := fun h => hbj h.symm
      simp [*]
    · rw [if_neg hs]
  rw [List.filter_congr this]
  conv_rhs => rw [← List.map_id (List.filter _ r.edges)]
  refine List.map_congr_left fun x hx => ?_
  simp only [List.mem_filter, decide_eq_true_eq] at hx
  exact moveSrc_of_ne (hx.2 ▸ hbj)

theorem insert_outEdges_n (hok : r.InsertOk j e) :
    (r.insertEmpty j e).outEdges r.blocks.length = (r.outEdges j).map (moveSrc j r.blocks.length) := by
  simp only [insertEmpty, outEdges, List.filter_append, List.filter_map]
  have h₁ : List.filter (fun x => decide (x.src = r.blocks.length)) [({ src := j, dst := r.blocks.length } : IEdge)] = [] := by
    simp [Nat.ne_of_lt hok.j]
  rw [h₁, List.append_nil]
  congr 1
  refine List.filter_congr fun x hx => ?_
  have := (hok.edges x hx).1
  simp only [Function.comp_def, moveSrc_src]
  by_cases hs : x.src = j
  · rw [if_pos hs]; simp [hs]
  · rw [if_neg hs]
    have : ¬ x.src = r.blocks.length := Nat.ne_of_lt this
    simp [*]

end Region

theorem moveSrc_cond (j n : BlockId) (e : IEdge) : (Region.moveSrc j n e).cond = e.cond := by
  unfold Region.moveSrc; split <;> rfl

section Insert

variable {ctx : Ctx} {r : Region} {j : BlockId} {e : State}

theorem insert_trueEdges_j (hok : r.InsertOk j e) (a : CSt) :
    trueEdges ctx (r.insertEmpty j e) a j = .ok [{ src := j, dst := r.blocks.length }] := by
  rw [trueEdges_eq, Region.insert_outEdges_j hok]
  rfl

theorem insert_trueEdges_lt (hok : r.InsertOk j e) {b : BlockId} (hbj : b ≠ j) (hb : b < r.blocks.length) (a : CSt) :
    trueEdges ctx (r.insertEmpty j e) a b = trueEdges ctx r a b := by
  rw [trueEdges_eq, trueEdges_eq, Region.insert_outEdges_lt hok hbj hb]

theorem insert_trueEdges_n (hok : r.InsertOk j e) (a : CSt) :
    trueEdges ctx (r.insertEmpty j e) a r.blocks.length =
      (trueEdges ctx r a j).map (List.map (Region.moveSrc j r.blocks.length)) := by
  rw [trueEdges_eq, trueEdges_eq, Region.insert_outEdges_n hok]
  exact filterMapM_edgeIfTrue_map ctx a _ (moveSrc_cond _ _) _

theorem moveSrc_dst_lt (hok : r.InsertOk j e) {a : CSt} {es : List IEdge} (h : trueEdges ctx r a j = .ok es) :
    ∀ x ∈ es, x.dst < r.blocks.length := fun x hx => (hok.edges x (trueEdges_mem_edges h x hx)).2

theorem empty_run (hok : r.InsertOk j e) (a : CSt) : BlockExec ctx a (.state e) (.ok (.normal, a)) :=
  .state (StateExec.of_noNodes hok.empty)

theorem empty_only (hok : r.InsertOk j e) {a : CSt} {res : Res (Ctl × CSt)} (h : BlockExec ctx a (.state e) res) :
    res = .ok (.normal, a) := by
  cases h with
  | state h => exact StateExec_noComponents (State.components_of_nodes hok.empty) h

theorem map_moveSrc_eq {es : List IEdge} {ie : IEdge}
    (h : es.map (Region.moveSrc j r.blocks.length) = [ie]) : ∃ x, es = [x] ∧ ie = Region.moveSrc j r.blocks.length x := by
  match es, h with
  | [x], h => exact ⟨x, rfl, (List.cons.inj h).1.symm⟩

theorem moveSrc_assigns (n : BlockId) (x : IEdge) : (Region.moveSrc j n x).assigns = x.assigns := by
  unfold Region.moveSrc; split <;> rfl

theorem moveSrc_dst (n : BlockId) (x : IEdge) : (Region.moveSrc j n x).dst = x.dst := by
  unfold Region.moveSrc; split <;> rfl

/-- Runs of the region with the empty state inserted are runs of the
original region (up to the block named by an ambiguous transition). -/
theorem RegionAt.insert_back' (hok : r.InsertOk j e) : ∀ {r' : Region} {a : CSt} {b : BlockId}
    {res' : Res (Ctl × CSt)}, RegionAt ctx r' a b res' → r' = r.insertEmpty j e →
      (b < r.blocks.length → ∃ res, RegionAt ctx r a b res ∧ ResSim Eq res res') ∧
      (b = r.blocks.length → ∀ a₀ bj, r.block? j = some bj → BlockExec ctx a₀ bj (.ok (.normal, a)) →
        ∃ res, RegionAt ctx r a₀ j res ∧ ResSim Eq res res')
  | _, _, _, _, .missing hb, rfl => by
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · rw [Region.insert_block?_lt hlt] at hb; simp [Region.block?, List.getElem?_eq_getElem hlt] at hb
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb
  | _, _, _, _, .blockErr hb h, rfl => by
    refine ⟨fun hlt => ⟨_, .blockErr (by rw [← Region.insert_block?_lt hlt]; exact hb) h, ResSim.err _⟩, fun heq => ?_⟩
    subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
  | _, _, _, _, .exit hb h hc, rfl => by
    refine ⟨fun hlt => ⟨_, .exit (by rw [← Region.insert_block?_lt hlt]; exact hb) h hc, ResSim.of_eq fun _ => rfl⟩,
      fun heq => ?_⟩
    subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h; exact absurd rfl hc
  | _, _, b, _, .edgesErr hb h he, rfl => by
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · by_cases hbj : b = j
      · subst hbj; rw [insert_trueEdges_j hok] at he; cases he
      · rw [insert_trueEdges_lt hok hbj hlt] at he
        exact ⟨_, .edgesErr (by rw [← Region.insert_block?_lt hlt]; exact hb) h he, ResSim.err _⟩
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
      intro a₀ bj hbj hj
      rw [insert_trueEdges_n hok] at he
      cases ht : trueEdges ctx r _ j with
      | err e' => rw [ht] at he; cases he; exact ⟨_, .edgesErr hbj hj ht, ResSim.err _⟩
      | ok es => rw [ht] at he; cases he
  | _, _, b, _, .stop hb h he, rfl => by
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · by_cases hbj : b = j
      · subst hbj; rw [insert_trueEdges_j hok] at he; cases he
      · rw [insert_trueEdges_lt hok hbj hlt] at he
        exact ⟨_, .stop (by rw [← Region.insert_block?_lt hlt]; exact hb) h he, ResSim.of_eq fun _ => rfl⟩
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
      intro a₀ bj hbj hj
      rw [insert_trueEdges_n hok] at he
      cases ht : trueEdges ctx r _ j with
      | err e' => rw [ht] at he; cases he
      | ok es =>
        rw [ht] at he
        have : es = [] := List.map_eq_nil_iff.1 (Res.ok_inj.1 he)
        subst this
        exact ⟨_, .stop hbj hj ht, ResSim.of_eq fun _ => rfl⟩
  | _, _, b, _, .ambiguous hb h he, rfl => by
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · by_cases hbj : b = j
      · subst hbj; rw [insert_trueEdges_j hok] at he; cases he
      · rw [insert_trueEdges_lt hok hbj hlt] at he
        exact ⟨_, .ambiguous (by rw [← Region.insert_block?_lt hlt]; exact hb) h he, ResSim.err _⟩
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
      intro a₀ bj hbj hj
      rw [insert_trueEdges_n hok] at he
      cases ht : trueEdges ctx r _ j with
      | err e' => rw [ht] at he; cases he
      | ok es =>
        rw [ht] at he
        match es, he with
        | x₁ :: x₂ :: xs, _ => exact ⟨_, .ambiguous hbj hj ht, ResSim.ub rfl _⟩
  | _, _, b, _, .assignErr hb h he ha, rfl => by
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · by_cases hbj : b = j
      · subst hbj; rw [insert_trueEdges_j hok] at he; cases he; cases ha
      · rw [insert_trueEdges_lt hok hbj hlt] at he
        exact ⟨_, .assignErr (by rw [← Region.insert_block?_lt hlt]; exact hb) h he ha, ResSim.err _⟩
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
      intro a₀ bj hbj hj
      rw [insert_trueEdges_n hok] at he
      cases ht : trueEdges ctx r _ j with
      | err e' => rw [ht] at he; cases he
      | ok es =>
        rw [ht] at he
        obtain ⟨x, rfl, rfl⟩ := map_moveSrc_eq (Res.ok_inj.1 he)
        rw [moveSrc_assigns] at ha
        exact ⟨_, .assignErr hbj hj ht ha, ResSim.err _⟩
  | _, _, b, _, .step (ie := ie) hb h he ha hn, rfl => by
    have IH := RegionAt.insert_back' hok hn rfl
    refine ⟨fun hlt => ?_, fun heq => ?_⟩
    · by_cases hbj : b = j
      · subst hbj
        rw [insert_trueEdges_j hok] at he
        cases he
        cases ha
        obtain ⟨bj, hbj'⟩ := Region.insert_block?_j hok
        have hb' := hb
        rw [Region.insert_block?_lt hok.j, hbj'] at hb'
        cases hb'
        exact IH.2 rfl _ _ hbj' h
      · rw [insert_trueEdges_lt hok hbj hlt] at he
        have hd : ie.dst < r.blocks.length := (hok.edges ie (trueEdges_mem_edges he ie (List.mem_singleton_self _))).2
        obtain ⟨res, hr, hs⟩ := IH.1 hd
        exact ⟨res, .step (by rw [← Region.insert_block?_lt hlt]; exact hb) h he ha hr, hs⟩
    · subst heq; rw [Region.insert_block?_n] at hb; cases hb; cases empty_only hok h
      intro a₀ bj hbj hj
      rw [insert_trueEdges_n hok] at he
      cases ht : trueEdges ctx r _ j with
      | err e' => rw [ht] at he; cases he
      | ok es =>
        rw [ht] at he
        obtain ⟨x, rfl, rfl⟩ := map_moveSrc_eq (Res.ok_inj.1 he)
        rw [moveSrc_assigns] at ha
        have hd : (Region.moveSrc j r.blocks.length x).dst < r.blocks.length := by
          rw [moveSrc_dst]; exact moveSrc_dst_lt hok ht x (List.mem_singleton_self _)
        obtain ⟨res, hr, hs⟩ := IH.1 hd
        rw [moveSrc_dst] at hr
        exact ⟨res, .step hbj hj ht ha hr, hs⟩

theorem RegionAt.insert_back (hok : r.InsertOk j e) {a : CSt} {b : BlockId} {res' : Res (Ctl × CSt)}
    (h : RegionAt ctx (r.insertEmpty j e) a b res') :
      (b < r.blocks.length → ∃ res, RegionAt ctx r a b res ∧ ResSim Eq res res') ∧
      (b = r.blocks.length → ∀ a₀ bj, r.block? j = some bj → BlockExec ctx a₀ bj (.ok (.normal, a)) →
        ∃ res, RegionAt ctx r a₀ j res ∧ ResSim Eq res res') :=
  RegionAt.insert_back' hok h rfl

/-- Runs of the original region are runs of the region with the empty state
inserted (up to the block named by an ambiguous transition). -/
theorem RegionAt.insert_fwd (hok : r.InsertOk j e) : ∀ {a : CSt} {b : BlockId} {res : Res (Ctl × CSt)},
    RegionAt ctx r a b res → b < r.blocks.length →
      ∃ res', RegionAt ctx (r.insertEmpty j e) a b res' ∧ ResSim Eq res' res
  | _, _, _, .missing hb, hlt => by simp [Region.block?, List.getElem?_eq_getElem hlt] at hb
  | _, _, _, .blockErr hb h, hlt => ⟨_, .blockErr (by rw [Region.insert_block?_lt hlt]; exact hb) h, ResSim.err _⟩
  | _, _, _, .exit hb h hc, hlt =>
    ⟨_, .exit (by rw [Region.insert_block?_lt hlt]; exact hb) h hc, ResSim.of_eq fun _ => rfl⟩
  | _, b, _, .edgesErr (cst₁ := a₁) (e := e') (b := blk) hb h he, hlt => by
    have hb' : (r.insertEmpty j e).block? b = some blk := by rw [Region.insert_block?_lt hlt]; exact hb
    by_cases hbj : b = j
    · subst hbj
      have hn : RegionAt ctx (r.insertEmpty b e) a₁ r.blocks.length (.err e') :=
        .edgesErr Region.insert_block?_n (empty_run hok a₁) (by rw [insert_trueEdges_n hok, he]; rfl)
      exact ⟨_, .step hb' h (insert_trueEdges_j hok a₁) rfl hn, ResSim.err _⟩
    · exact ⟨_, .edgesErr hb' h (by rw [insert_trueEdges_lt hok hbj hlt]; exact he), ResSim.err _⟩
  | _, b, _, .stop (cst₁ := a₁) (b := blk) hb h he, hlt => by
    have hb' : (r.insertEmpty j e).block? b = some blk := by rw [Region.insert_block?_lt hlt]; exact hb
    by_cases hbj : b = j
    · subst hbj
      have hn : RegionAt ctx (r.insertEmpty b e) a₁ r.blocks.length (.ok (.normal, a₁)) :=
        .stop Region.insert_block?_n (empty_run hok a₁) (by rw [insert_trueEdges_n hok, he]; rfl)
      exact ⟨_, .step hb' h (insert_trueEdges_j hok a₁) rfl hn, ResSim.of_eq fun _ => rfl⟩
    · exact ⟨_, .stop hb' h (by rw [insert_trueEdges_lt hok hbj hlt]; exact he), ResSim.of_eq fun _ => rfl⟩
  | _, b, _, .ambiguous (cst₁ := a₁) (b := blk) hb h he, hlt => by
    have hb' : (r.insertEmpty j e).block? b = some blk := by rw [Region.insert_block?_lt hlt]; exact hb
    by_cases hbj : b = j
    · subst hbj
      have hn : RegionAt ctx (r.insertEmpty b e) a₁ r.blocks.length (.err (.ambiguousTransition r.blocks.length)) :=
        .ambiguous Region.insert_block?_n (empty_run hok a₁) (by rw [insert_trueEdges_n hok, he]; rfl)
      exact ⟨_, .step hb' h (insert_trueEdges_j hok a₁) rfl hn, ResSim.ub rfl _⟩
    · exact ⟨_, .ambiguous hb' h (by rw [insert_trueEdges_lt hok hbj hlt]; exact he), ResSim.err _⟩
  | _, b, _, .assignErr (cst₁ := a₁) (ie := ie) (e := e') (b := blk) hb h he ha, hlt => by
    have hb' : (r.insertEmpty j e).block? b = some blk := by rw [Region.insert_block?_lt hlt]; exact hb
    by_cases hbj : b = j
    · subst hbj
      have hn : RegionAt ctx (r.insertEmpty b e) a₁ r.blocks.length (.err e') :=
        .assignErr (ie := Region.moveSrc b r.blocks.length ie) Region.insert_block?_n (empty_run hok a₁)
          (by rw [insert_trueEdges_n hok, he]; rfl) (by rw [moveSrc_assigns]; exact ha)
      exact ⟨_, .step hb' h (insert_trueEdges_j hok a₁) rfl hn, ResSim.err _⟩
    · exact ⟨_, .assignErr hb' h (by rw [insert_trueEdges_lt hok hbj hlt]; exact he) ha, ResSim.err _⟩
  | _, b, _, .step (cst₁ := a₁) (ie := ie) (b := blk) hb h he ha hn, hlt => by
    have hb' : (r.insertEmpty j e).block? b = some blk := by rw [Region.insert_block?_lt hlt]; exact hb
    have hd : ie.dst < r.blocks.length := (hok.edges ie (trueEdges_mem_edges he ie (List.mem_singleton_self _))).2
    obtain ⟨res', hr', hs⟩ := RegionAt.insert_fwd hok hn hd
    by_cases hbj : b = j
    · subst hbj
      have hm : RegionAt ctx (r.insertEmpty b e) a₁ r.blocks.length res' :=
        .step (ie := Region.moveSrc b r.blocks.length ie) Region.insert_block?_n (empty_run hok a₁)
          (by rw [insert_trueEdges_n hok, he]; rfl) (by rw [moveSrc_assigns]; exact ha) (by rw [moveSrc_dst]; exact hr')
      exact ⟨_, .step hb' h (insert_trueEdges_j hok a₁) rfl hm, hs⟩
    · exact ⟨_, .step hb' h (by rw [insert_trueEdges_lt hok hbj hlt]; exact he) ha hr', hs⟩

end Insert

section InsertRules

variable {r : Region} {j : BlockId} {e : State}

theorem RegionExec.insert_back {ctx : Ctx} (hok : r.InsertOk j e) {a : CSt} {res' : Res (Ctl × CSt)}
    (h : RegionExec ctx a (r.insertEmpty j e) res') : MatchedE CtlRel (RegionExec ctx a r) res' := by
  cases h with
  | start h =>
    obtain ⟨res, hr, hs⟩ := (RegionAt.insert_back hok h).1 hok.start
    exact ⟨res, .start hr, Res.Below.of_resSim hs⟩

theorem RegionExec.insert_fwd {ctx : Ctx} (hok : r.InsertOk j e) {a : CSt} {res : Res (Ctl × CSt)}
    (h : RegionExec ctx a r res) : MatchedE CtlRel (RegionExec ctx a (r.insertEmpty j e)) res := by
  cases h with
  | start h =>
    obtain ⟨res', hr, hs⟩ := RegionAt.insert_fwd hok h hok.start
    exact ⟨res', .start hr, Res.Below.of_resSim hs⟩

/-- **Empty-state insertion rule**: inserting an empty state after block `j`
of the region at path `p` (it takes over `j`'s out-edges) gives an
equivalent SDFG. -/
theorem Congr.insertEmpty {g : SDFG} {p : List PathStep} {c : Region}
    (hat : g.top.regionAt? p = some c) (hok : c.insertOk? j e = true)
    (hm : sameRegionMentions? g (c.insertEmpty j e) c = true) (hsup : Supported g.prog) :
    Equiv (g.setRegionAt p (c.insertEmpty j e)).prog g.prog :=
  Congr.regionEquiv hat hm (fun _ _ _ _ h => RegionExec.insert_back (Region.insertOk?_sound hok) h)
    (fun _ _ _ _ h => RegionExec.insert_fwd (Region.insertOk?_sound hok) h) hsup

/-- **Empty-state removal rule**: the inverse of `Congr.insertEmpty`. -/
theorem Congr.removeEmpty {g : SDFG} {p : List PathStep} {c c' : Region}
    (hat : g.top.regionAt? p = some c) (hc : c = c'.insertEmpty j e) (hok : c'.insertOk? j e = true)
    (hm : sameRegionMentions? g c' c = true) (hsup : Supported g.prog) :
    Equiv (g.setRegionAt p c').prog g.prog := by
  subst hc
  exact Congr.regionEquiv hat hm (fun _ _ _ _ h => RegionExec.insert_fwd (Region.insertOk?_sound hok) h)
    (fun _ _ _ _ h => RegionExec.insert_back (Region.insertOk?_sound hok) h) hsup

end InsertRules

/-- Undo `insertEmpty` on the last block (an empty state reached from `j`):
the region without it and the removed state (candidates; `Congr.removeEmpty`
checks that inserting it back gives the region). -/
def Region.removeLast? (c : Region) (j : BlockId) : Option (Region × State) :=
  match c.blocks.getLast? with
  | some (.state e) =>
    let n := c.blocks.length - 1
    some ({ c with
      blocks := c.blocks.dropLast
      edges := (c.edges.filter fun x => !(x.src == j && x.dst == n)).map fun x =>
        if x.src = n then { x with src := j } else x }, e)
  | _ => none

end Dace
