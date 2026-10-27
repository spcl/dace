-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Footprinted

/-!
# Frame property of sequential dataflow (SDFG)

On the fragment `SeqPure` (plain containers: no streams, references or
views; no consume scopes and no nested SDFGs; pure subsets, ranges and
reduction identities; map scopes without private transients; code nodes that
pass no values to other code nodes; decided by `seqPure?`,
`seqPure?_sound`), a sequential execution is a
*footprinted action* (`Dace.SDFG.Footprinted`):

* every normal run of a node (`NodeExec`), a sequence of nodes (`SeqExec`),
  a scope body or component (`OrderedExec`) or the iterations of a map
  (`IterExec`) changes the state by a plain delta `δ`: the final local view
  is `δ` applied to the initial one (`St.step`);
* from any other state with the same alias table and code-edge values that
  agrees with the initial one on the cells `δ` reads, the same derivation
  shape yields a normal run with the same delta (and the same trace)
  (**`NodeExec.frame`**, `SeqExec.frame`, `OrderedExec.frame`,
  `IterExec.frame`).

WCR contributions need no read: the delta applies them to whatever the cell
holds (`Log.apply_wcr`).  As a corollary, along a run from an empty log the
local view is the initial memory with the log applied (**St-coherence**,
`NodeExec.coherent`, `OrderedExec.coherent`).

Why the fragment:
* streams, views and references act on queues and alias tables (outside the
  plain delta); nested SDFGs and consume scopes are out of scope for now;
* `Ctx.eval` reads memory without logging (`Decisions.md#D33`), so
  evaluated expressions must be pure (`Ctx.exprPure`);
* iteration logs drop private transients, so a read of a private transient
  before it is written would be a dependence the map's delta does not
  record: map scopes must have none (`Ctx.privateTransients = []`);
* code→code values are not logged; code nodes must not produce them.

Without the restriction on private transients (`SeqPureP`, `seqPureP?`),
runs are still coherent (`OrderedExec.coh`, end of the file); the pair
checker of `Dace.Equiv.CorrCheck` only needs coherence.
-/

set_option autoImplicit false

namespace Dace

/-- Tasklets and library nodes. -/
def DNode.isCode (n : DNode) : Bool :=
  match n.kind with
  | .tasklet _ | .library _ => true
  | _ => false

/-- The fragment of state-level dataflow whose sequential executions are
coherent (the local view is the initial memory with the log applied,
`NodeExec.coh`): `SeqPure` without the restriction on private transients. -/
structure SeqPlain (ctx : Ctx) (s : State) : Prop where
  /-- No streams, references or views. -/
  plain : ∀ n d, ctx.desc? n = some d → d.isStream = false ∧ d.isReference = false ∧ d.isView = false
  /-- Shapes are pure (full ranges of missing subsets). -/
  shapes : ∀ n d, ctx.desc? n = some d → ctx.rangePure d.fullRange = true
  /-- Memlet subsets are pure. -/
  memlets : ∀ e ∈ s.edges, ∀ m, e.data = .memlet m →
    m.srcSubset.all ctx.subsetPure = true ∧ m.dstSubset.all ctx.subsetPure = true
  /-- Map scopes have pure ranges. -/
  ranges : ∀ n ∈ s.nodes, ∀ mp, n.kind = .mapEntry mp → ctx.rangePure mp.range = true
  /-- Reduction identities are pure. -/
  reduces : ∀ n ∈ s.nodes, ∀ l props, n.kind = .library l → l.reduce = some props →
    props.identity.all ctx.exprPure = true
  /-- Code nodes write containers only. -/
  codeOut : ∀ n ∈ s.nodes, n.isCode = true → ∀ e ∈ s.outEdges n.id, ∀ m, e.data = .memlet m →
    (s.dstEndpoint e).isData = true

/-- `SeqPlain` with only simple nodes and map entries (no consume scopes, no
nested SDFGs). -/
structure SeqPureP (ctx : Ctx) (s : State) : Prop extends SeqPlain ctx s where
  kinds : ∀ n ∈ s.nodes, n.isSimple = true ∨ ∃ mp, n.kind = .mapEntry mp

instance {ctx : Ctx} {s : State} : Coe (SeqPureP ctx s) (SeqPlain ctx s) := ⟨SeqPureP.toSeqPlain⟩

/-- The fragment of state-level dataflow on which sequential executions are
footprinted actions: `SeqPureP` and map scopes without private transients. -/
structure SeqPure (ctx : Ctx) (s : State) : Prop extends SeqPureP ctx s where
  /-- Map scopes have no private transients. -/
  noPriv : ∀ n ∈ s.nodes, ∀ mp, n.kind = .mapEntry mp → ctx.privateTransients s n.id = []

instance {ctx : Ctx} {s : State} : Coe (SeqPure ctx s) (SeqPureP ctx s) := ⟨SeqPure.toSeqPureP⟩

theorem SeqPure.maps {ctx : Ctx} {s : State} (h : SeqPure ctx s) :
    ∀ n ∈ s.nodes, ∀ mp, n.kind = .mapEntry mp →
      ctx.rangePure mp.range = true ∧ ctx.privateTransients s n.id = [] :=
  fun n hn mp hk => ⟨h.ranges n hn mp hk, h.noPriv n hn mp hk⟩

/-- Decision procedure for `SeqPureP` (`seqPureP?_sound`). -/
def seqPureP? (ctx : Ctx) (s : State) : Bool :=
  ctx.g.arrays.all (fun (_, d) => !d.isStream && !d.isReference && !d.isView && ctx.rangePure d.fullRange) &&
  s.edges.all (fun e => match e.data with
    | .memlet m => m.srcSubset.all ctx.subsetPure && m.dstSubset.all ctx.subsetPure
    | .dep => true) &&
  s.nodes.all (fun n =>
    (n.isSimple || (match n.kind with | .mapEntry _ => true | _ => false)) &&
    (match n.kind with
      | .mapEntry mp => ctx.rangePure mp.range
      | .library l => (l.reduce.map fun p => p.identity.all ctx.exprPure).getD true
      | _ => true) &&
    (!n.isCode || (s.outEdges n.id).all fun e => match e.data with
      | .memlet _ => (s.dstEndpoint e).isData
      | .dep => true))

/-- Decision procedure for `SeqPlain` (`seqPlain?_sound`). -/
def seqPlain? (ctx : Ctx) (s : State) : Bool :=
  ctx.g.arrays.all (fun (_, d) => !d.isStream && !d.isReference && !d.isView && ctx.rangePure d.fullRange) &&
  s.edges.all (fun e => match e.data with
    | .memlet m => m.srcSubset.all ctx.subsetPure && m.dstSubset.all ctx.subsetPure
    | .dep => true) &&
  s.nodes.all (fun n =>
    (match n.kind with
      | .mapEntry mp => ctx.rangePure mp.range
      | .library l => (l.reduce.map fun p => p.identity.all ctx.exprPure).getD true
      | _ => true) &&
    (!n.isCode || (s.outEdges n.id).all fun e => match e.data with
      | .memlet _ => (s.dstEndpoint e).isData
      | .dep => true))

/-- Decision procedure for `SeqPure` (`seqPure?_sound`). -/
def seqPure? (ctx : Ctx) (s : State) : Bool :=
  ctx.g.arrays.all (fun (_, d) => !d.isStream && !d.isReference && !d.isView && ctx.rangePure d.fullRange) &&
  s.edges.all (fun e => match e.data with
    | .memlet m => m.srcSubset.all ctx.subsetPure && m.dstSubset.all ctx.subsetPure
    | .dep => true) &&
  s.nodes.all (fun n =>
    (n.isSimple || (match n.kind with | .mapEntry _ => true | _ => false)) &&
    (match n.kind with
      | .mapEntry mp => ctx.rangePure mp.range && (ctx.privateTransients s n.id).isEmpty
      | .library l => (l.reduce.map fun p => p.identity.all ctx.exprPure).getD true
      | _ => true) &&
    (!n.isCode || (s.outEdges n.id).all fun e => match e.data with
      | .memlet _ => (s.dstEndpoint e).isData
      | .dep => true))

theorem seqPlain?_sound {ctx : Ctx} {s : State} (h : seqPlain? ctx s = true) : SeqPlain ctx s := by
  simp only [seqPlain?, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨⟨harr, hedges⟩, hnodes⟩ := h
  have hdesc : ∀ n d, ctx.desc? n = some d → ∃ x ∈ ctx.g.arrays, x.2 = d := by
    intro n d hd
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    exact ⟨x, List.mem_of_find?_eq_some hx, rfl⟩
  have harr' : ∀ n d, ctx.desc? n = some d →
      d.isStream = false ∧ d.isReference = false ∧ d.isView = false ∧ ctx.rangePure d.fullRange = true := by
    intro n d hd
    obtain ⟨x, hx, rfl⟩ := hdesc n d hd
    have := harr x hx
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
    exact ⟨this.1.1.1, this.1.1.2, this.1.2, this.2⟩
  refine ⟨fun n d hd => ⟨(harr' n d hd).1, (harr' n d hd).2.1, (harr' n d hd).2.2.1⟩,
    fun n d hd => (harr' n d hd).2.2.2, fun e he m hm => ?_, fun n hn mp hk => ?_,
    fun n hn l props hk hr => ?_, fun n hn hc e he m hm => ?_⟩
  · have := hedges e he
    simp only [hm, Bool.and_eq_true] at this
    exact this
  · have := (hnodes n hn).1
    simp only [hk] at this
    exact this
  · have := (hnodes n hn).1
    simp only [hk, hr, Option.map_some, Option.getD_some] at this
    exact this
  · have := (hnodes n hn).2
    simp only [hc, Bool.not_true, Bool.false_or, List.all_eq_true] at this
    have := this e he
    simp only [hm] at this
    exact this

theorem seqPureP?_sound {ctx : Ctx} {s : State} (h : seqPureP? ctx s = true) : SeqPureP ctx s := by
  simp only [seqPureP?, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨⟨harr, hedges⟩, hnodes⟩ := h
  have hdesc : ∀ n d, ctx.desc? n = some d → ∃ x ∈ ctx.g.arrays, x.2 = d := by
    intro n d hd
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    exact ⟨x, List.mem_of_find?_eq_some hx, rfl⟩
  have harr' : ∀ n d, ctx.desc? n = some d →
      d.isStream = false ∧ d.isReference = false ∧ d.isView = false ∧ ctx.rangePure d.fullRange = true := by
    intro n d hd
    obtain ⟨x, hx, rfl⟩ := hdesc n d hd
    have := harr x hx
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
    exact ⟨this.1.1.1, this.1.1.2, this.1.2, this.2⟩
  refine ⟨⟨fun n d hd => ⟨(harr' n d hd).1, (harr' n d hd).2.1, (harr' n d hd).2.2.1⟩,
    fun n d hd => (harr' n d hd).2.2.2, fun e he m hm => ?_, fun n hn mp hk => ?_,
    fun n hn l props hk hr => ?_, fun n hn hc e he m hm => ?_⟩, fun n hn => ?_⟩
  · have := hedges e he
    simp only [hm, Bool.and_eq_true] at this
    exact this
  rotate_left 3
  · have := (hnodes n hn).1.1
    simp only [Bool.or_eq_true] at this
    rcases this with h₁ | h₁
    · exact .inl h₁
    · right
      cases hk : n.kind <;> simp [hk] at h₁
      exact ⟨_, rfl⟩
  · have := (hnodes n hn).1.2
    simp only [hk] at this
    exact this
  · have := (hnodes n hn).1.2
    simp only [hk, hr, Option.map_some, Option.getD_some] at this
    exact this
  · have := (hnodes n hn).2
    simp only [hc, Bool.not_true, Bool.false_or, List.all_eq_true] at this
    have := this e he
    simp only [hm] at this
    exact this

theorem seqPure?_sound {ctx : Ctx} {s : State} (h : seqPure? ctx s = true) : SeqPure ctx s := by
  simp only [seqPure?, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨⟨harr, hedges⟩, hnodes⟩ := h
  have hdesc : ∀ n d, ctx.desc? n = some d → ∃ x ∈ ctx.g.arrays, x.2 = d := by
    intro n d hd
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    exact ⟨x, List.mem_of_find?_eq_some hx, rfl⟩
  have harr' : ∀ n d, ctx.desc? n = some d →
      d.isStream = false ∧ d.isReference = false ∧ d.isView = false ∧ ctx.rangePure d.fullRange = true := by
    intro n d hd
    obtain ⟨x, hx, rfl⟩ := hdesc n d hd
    have := harr x hx
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
    exact ⟨this.1.1.1, this.1.1.2, this.1.2, this.2⟩
  have hmaps : ∀ n ∈ s.nodes, ∀ mp, n.kind = .mapEntry mp →
      ctx.rangePure mp.range = true ∧ ctx.privateTransients s n.id = [] := by
    intro n hn mp hk
    have := (hnodes n hn).1.2
    simp only [hk, Bool.and_eq_true, List.isEmpty_iff] at this
    exact this
  refine ⟨⟨⟨fun n d hd => ⟨(harr' n d hd).1, (harr' n d hd).2.1, (harr' n d hd).2.2.1⟩,
    fun n d hd => (harr' n d hd).2.2.2, fun e he m hm => ?_, fun n hn mp hk => (hmaps n hn mp hk).1,
    fun n hn l props hk hr => ?_, fun n hn hc e he m hm => ?_⟩, fun n hn => ?_⟩, fun n hn mp hk => (hmaps n hn mp hk).2⟩
  · have := hedges e he
    simp only [hm, Bool.and_eq_true] at this
    exact this
  rotate_left 2
  · have := (hnodes n hn).1.1
    simp only [Bool.or_eq_true] at this
    rcases this with h₁ | h₁
    · exact .inl h₁
    · right
      cases hk : n.kind <;> simp [hk] at h₁
      exact ⟨_, rfl⟩
  · have := (hnodes n hn).1.2
    simp only [hk, hr, Option.map_some, Option.getD_some] at this
    exact this
  · have := (hnodes n hn).2
    simp only [hc, Bool.not_true, Bool.false_or, List.all_eq_true] at this
    have := this e he
    simp only [hm] at this
    exact this

namespace SeqPlain

variable {ctx : Ctx} {s : State} (h : SeqPlain ctx s)
include h

theorem isStream (n : Name) : ctx.isStream n = false := by
  unfold Ctx.isStream
  cases hd : ctx.desc? n with
  | none => rfl
  | some d => simp [(h.plain n d hd).1]

theorem isReference (n : Name) : ctx.isReference n = false := by
  unfold Ctx.isReference
  cases hd : ctx.desc? n with
  | none => rfl
  | some d => simp [(h.plain n d hd).2.1]

theorem isView (n : Name) : ctx.isView n = false := by
  unfold Ctx.isView
  cases hd : ctx.desc? n with
  | none => rfl
  | some d => simp [(h.plain n d hd).2.2]

theorem pointsPure_none (n : Name) : ctx.pointsPure n none = true := by
  simp only [Ctx.pointsPure]
  cases hd : ctx.desc? n with
  | none => rfl
  | some d => simpa using h.shapes n d hd

theorem src_pure {e : DEdge} (he : e ∈ s.edges) (n : Name) : ctx.pointsPure n (State.leafSrcSubset e) = true := by
  unfold State.leafSrcSubset
  cases hd : e.data with
  | dep => exact h.pointsPure_none n
  | memlet m =>
    have := (h.memlets e he m hd).1
    simp only [EdgeData.memlet?, Option.bind_some]
    cases hs : m.srcSubset with
    | none => exact h.pointsPure_none n
    | some sub => simpa [hs, Ctx.pointsPure] using this

theorem dst_pure {e : DEdge} (he : e ∈ s.edges) (n : Name) : ctx.pointsPure n (State.leafDstSubset e) = true := by
  unfold State.leafDstSubset
  cases hd : e.data with
  | dep => exact h.pointsPure_none n
  | memlet m =>
    have := (h.memlets e he m hd).2
    simp only [EdgeData.memlet?, Option.bind_some]
    cases hs : m.dstSubset with
    | none => exact h.pointsPure_none n
    | some sub => simpa [hs, Ctx.pointsPure] using this

/-- The container endpoints of edges have pure subsets. -/
theorem srcEndpoint_pure {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset}
    (hs : s.srcEndpoint e = .data d sub) : ctx.pointsPure d sub = true := by
  unfold State.srcEndpoint at hs
  repeat' ((try dsimp only at hs); split at hs)
  all_goals first
    | (cases hs; exact h.pointsPure_none _)
    | (cases hs; exact h.src_pure he _)
    | cases hs

theorem dstEndpoint_pure {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset}
    (hs : s.dstEndpoint e = .data d sub) : ctx.pointsPure d sub = true := by
  unfold State.dstEndpoint at hs
  repeat' ((try dsimp only at hs); split at hs)
  all_goals first
    | (cases hs; exact h.dst_pure he _)
    | cases hs

end SeqPlain

/-! ## Leaf functions are footprinted -/

section Leaf

variable {ctx : Ctx} (ρ : Valuation) {s : State} (h : SeqPlain ctx s)
include h

omit h in
theorem mem_edges_of_in {id : NodeId} {e : DEdge} (he : e ∈ s.inEdges id) : e ∈ s.edges :=
  (List.mem_filter.1 he).1

omit h in
theorem mem_edges_of_out {id : NodeId} {e : DEdge} (he : e ∈ s.outEdges id) : e ∈ s.edges :=
  (List.mem_filter.1 he).1

omit h in
theorem mem_outEdges_of_on {n : DNode} {c : Conn} {e : DEdge} (he : e ∈ s.outEdgesOn n c) :
    e ∈ s.outEdges n.id :=
  (List.mem_filter.1 he).1

theorem readScalarData_foot (d : Name) (sub : Option Subset) (oob : Bool) (hp : ctx.pointsPure d sub = true) :
    FootA ctx.I.wcr (fun st => readScalarData ctx ρ d sub oob st) := by
  unfold readScalarData
  refine FootA.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hp) (fun pts => ?_)
  rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
  · exact FootA.err _
  · simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
    exact FootA.bind (St.read_foot ctx ρ _ d p oob) (fun v => FootA.pure (some v))
  · exact FootA.err _

theorem readEdgeScalar_foot {e : DEdge} (he : e ∈ s.edges) :
    FootA ctx.I.wcr (fun st => readEdgeScalar ctx ρ s e st) := by
  unfold readEdgeScalar
  cases hse : s.srcEndpoint e with
  | code key => exact FootA.ofVals (fun vs => (vs.find? (·.1 = key)).map (·.2))
  | data d sub => exact readScalarData_foot ρ h d sub _ (h.srcEndpoint_pure he hse)
  | invalid => exact FootA.err _

theorem copyData_foot (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (wcr : Option WcrOp) (oob : Bool) (hs : ctx.pointsPure sd ssub = true) (hd : ctx.pointsPure dd dsub = true) :
    Foot ctx.I.wcr (fun st => copyData ctx ρ st sd ssub dd dsub wcr oob) := by
  unfold copyData
  simp only [h.isStream]
  refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hs) (fun spts => ?_)
  refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hd) (fun dpts => ?_)
  refine Foot.foldlM _ (fun ⟨p, q⟩ _ => ?_)
  exact Foot.bindA (St.read_foot ctx ρ _ sd p oob) (fun v => St.store_foot ctx ρ dd q oob wcr v)

theorem execCopy_foot {e : DEdge} (he : e ∈ s.edges) : Foot ctx.I.wcr (fun st => execCopy ctx ρ s e st) := by
  unfold execCopy
  cases hd : e.data with
  | dep => exact Foot.pure
  | memlet m =>
    cases hs : s.srcEndpoint e <;> cases ht : s.dstEndpoint e <;> first
      | exact Foot.err _
      | exact copyData_foot ρ h _ _ _ _ _ _ (h.srcEndpoint_pure he hs) (h.dstEndpoint_pure he ht)

omit h in
theorem forall₂_loc_mem {m : Mem} {d : Name} {oob : Bool} {pts : List Idx} {ls : List Loc}
    (hf : List.Forall₂ (fun p l => ctx.loc ρ m d p oob = .ok l) pts ls) {p : Idx} (hp : p ∈ pts) :
    ∃ l ∈ ls, ctx.loc ρ m d p oob = .ok l := by
  induction hf with
  | nil => cases hp
  | cons hl _ ih =>
    rcases List.mem_cons.1 hp with rfl | hp
    · exact ⟨_, List.mem_cons_self, hl⟩
    · obtain ⟨l, hl', he⟩ := ih hp
      exact ⟨l, List.mem_cons_of_mem _ hl', he⟩

omit h in
/-- The pointer-connector branch of `bindInData`: the connector reads the
cells that `readMany` logged. -/
theorem bindInData_array_foot (d : Name) (sub : Option Subset) (oob : Bool) (pts : List Idx)
    (hp : ctx.pointsPure d sub = true) :
    FootA ctx.I.wcr (fun st => do
      let lo ← ctx.lowerCorner ρ st.mem d sub
      let (_, st) ← st.readMany ctx ρ d pts oob
      let m := st.mem
      let get : Idx → Option Val := fun rel =>
        if rel.length = lo.length then
          let abs := List.zipWith (· + ·) lo rel
          if pts.contains abs then
            match ctx.loc ρ m d abs oob with
            | .ok l => some (m.read l)
            | .err _ => none
          else none
        else none
      pure (ConnIn.array get, st)) := by
  refine FootA.bind_det (fun st st₂ _ => ctx.lowerCorner_pure ρ _ _ hp) (fun lo => ?_)
  intro st a st' hr
  dsimp only at hr
  cases hrm : st.readMany ctx ρ d pts oob with
  | err e => rw [hrm] at hr; cases hr
  | ok r =>
    obtain ⟨vs, st₁⟩ := r
    rw [hrm] at hr
    simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    unfold St.readMany at hrm
    obtain ⟨ls, hf, -, rfl, hd⟩ := St.readMany_ok ctx ρ pts [] st vs st₁ hrm
    refine ⟨{ reads := ls.map st.mem.resolve }, ⟨rfl, rfl⟩, rfl, fun st₂ hag hv => ?_⟩
    have hf₂ : List.Forall₂ (fun p l => ctx.loc ρ st₂.mem d p oob = .ok l) pts ls :=
      hf.imp fun p l hl => (ctx.loc_congr ρ hag.1 d p oob).trans hl
    have hd₂ : ∀ l ∈ ls, (st₂.mem.read l).isUndef = false := fun l hl => by
      simp only [Mem.read, Mem.resolve_congr hag.1 l]
      rw [hag.2 _ (List.mem_map_of_mem hl)]
      exact hd l hl
    have hrm₂ := St.readMany_of_forall₂ ctx ρ pts ls [] st₂ hf₂ hd₂
    have hres : ls.map st₂.mem.resolve = ls.map st.mem.resolve :=
      List.map_congr_left fun l _ => Mem.resolve_congr hag.1 l
    dsimp only
    unfold St.readMany
    rw [hrm₂]
    simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq, hres]
    refine ⟨?_, rfl⟩
    congr 1
    funext rel
    split
    · split
      · next hc =>
        have habs : List.zipWith (· + ·) lo rel ∈ pts := by simpa using hc
        obtain ⟨l, hl, hloc⟩ := forall₂_loc_mem ρ hf habs
        have hloc₂ : ctx.loc ρ (st₂.logged { reads := ls.map st.mem.resolve }).mem d
            (List.zipWith (· + ·) lo rel) oob = .ok l := (ctx.loc_congr ρ hag.1 _ _ _).trans hloc
        have hloc₁ : ctx.loc ρ (st.logged { reads := ls.map st.mem.resolve }).mem d
            (List.zipWith (· + ·) lo rel) oob = .ok l := hloc
        rw [hloc₁, hloc₂]
        simp only [St.logged, Mem.read, Mem.resolve_congr hag.1 l]
        rw [hag.2 _ (List.mem_map_of_mem hl)]
      · rfl
    · rfl

theorem bindInData_foot (d : Name) (sub : Option Subset) (oob : Bool) (hp : ctx.pointsPure d sub = true) :
    FootA ctx.I.wcr (fun st => bindInData ctx ρ d sub oob st) := by
  unfold bindInData
  simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
  refine FootA.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hp) (fun pts => ?_)
  rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
  · exact bindInData_array_foot ρ d sub oob [] hp
  · exact FootA.bind (St.read_foot ctx ρ _ d p oob) (fun v => FootA.pure (ConnIn.scalar v))
  · exact bindInData_array_foot ρ d sub oob _ hp

theorem bindIn_foot {e : DEdge} (he : e ∈ s.edges) : FootA ctx.I.wcr (fun st => bindIn ctx ρ s e st) := by
  unfold bindIn
  cases hse : s.srcEndpoint e with
  | code key => exact FootA.ofVals (fun vs => ((vs.find? (·.1 = key)).map (·.2)).map ConnIn.scalar)
  | data d sub =>
    exact FootA.bind (bindInData_foot ρ h d sub _ (h.srcEndpoint_pure he hse)) (fun c => FootA.pure (some c))
  | invalid => exact FootA.err _

theorem bindInputs_foot (n : DNode) : FootA ctx.I.wcr (fun st => bindInputs ctx ρ s n st) := by
  unfold bindInputs
  refine FootA.bind (FootA.foldlM _ (fun e he acc => ?_) []) (fun bs => FootA.pure _)
  cases hd : e.data with
  | dep => exact FootA.pure _
  | memlet m =>
    refine FootA.bind (bindIn_foot ρ h (mem_edges_of_in he)) (fun ci => ?_)
    cases ci with
    | none => exact FootA.pure _
    | some c => exact FootA.pure _

theorem storeScalar_foot (d : Name) (sub : Option Subset) (wcr : Option WcrOp) (oob : Bool) (v : Val)
    (hp : ctx.pointsPure d sub = true) : Foot ctx.I.wcr (fun st => storeScalar ctx ρ st d sub wcr oob v) := by
  unfold storeScalar
  simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
  refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hp) (fun pts => ?_)
  rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
  · exact Foot.err _
  · exact St.store_foot ctx ρ d p oob wcr v
  · exact Foot.err _

theorem applyScalar_foot {n : DNode} (hn : n ∈ s.nodes) (hc : n.isCode = true) (c : Conn) (v : Val) :
    Foot ctx.I.wcr (fun st => applyScalar ctx ρ s n st c v) := by
  unfold applyScalar
  refine Foot.foldlM _ (fun e he => ?_)
  have he' := mem_outEdges_of_on he
  cases hd : e.data with
  | dep => exact Foot.pure
  | memlet m =>
    have hdata := h.codeOut n hn hc e he' m hd
    cases hde : s.dstEndpoint e with
    | code key => rw [hde] at hdata; cases hdata
    | data d sub => exact storeScalar_foot ρ h d sub _ _ v (h.dstEndpoint_pure (mem_edges_of_out he') hde)
    | invalid => exact Foot.err _

theorem storeArrayWrite_foot (d : Name) (sub : Option Subset) (wcr : Option WcrOp) (oob : Bool) (rel : Idx)
    (v : Val) (hp : ctx.pointsPure d sub = true) :
    Foot ctx.I.wcr (fun st => storeArrayWrite ctx ρ st d sub wcr oob rel v) := by
  unfold storeArrayWrite
  simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
  refine Foot.bind_det (fun st st₂ _ => ctx.lowerCorner_pure ρ _ _ hp) (fun lo => ?_)
  by_cases hl : rel.length = lo.length
  · simp only [hl, ↓reduceIte]
    refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hp) (fun pts => ?_)
    split
    · exact St.store_foot ctx ρ d _ oob wcr v
    · exact Foot.err _
  · simp only [hl, ↓reduceIte]
    exact Foot.err _

theorem applyArrayWrite_foot (n : DNode) (c : Conn) (rel : Idx) (v : Val) :
    Foot ctx.I.wcr (fun st => applyArrayWrite ctx ρ s n st c rel v) := by
  unfold applyArrayWrite
  refine Foot.foldlM _ (fun e he => ?_)
  have he' := mem_edges_of_out (mem_outEdges_of_on he)
  cases hd : e.data with
  | dep => exact Foot.pure
  | memlet m =>
    cases hde : s.dstEndpoint e with
    | data d sub => exact storeArrayWrite_foot ρ h d sub _ _ rel v (h.dstEndpoint_pure he' hde)
    | code key => exact Foot.err _
    | invalid => exact Foot.err _

theorem applyPush_foot (n : DNode) (c : Conn) (vs : List Val) :
    Foot ctx.I.wcr (fun st => applyPush ctx ρ s n st c vs) := by
  unfold applyPush
  refine Foot.foldlM _ (fun e _ => ?_)
  cases hde : s.dstEndpoint e with
  | data d sub =>
    unfold pushTo
    simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
    exact Foot.err _
  | code key => exact Foot.err _
  | invalid => exact Foot.err _

theorem applyPop_foot (n : DNode) (c : Conn) (k : Nat) :
    Foot ctx.I.wcr (fun st => applyPop ctx ρ s n st c k) := by
  unfold applyPop
  cases hi : s.inEdgeOn n c with
  | none => exact Foot.err _
  | some e =>
    dsimp only
    cases hse : s.srcEndpoint e with
    | data d sub =>
      unfold popFrom
      simp only [h.isStream, Bool.false_eq_true, ↓reduceIte]
      exact Foot.err _
    | code key => exact Foot.err _
    | invalid => exact Foot.err _

theorem applyOut_foot {n : DNode} (hn : n ∈ s.nodes) (hc : n.isCode = true) (out : TaskletOut) :
    Foot ctx.I.wcr (fun st => applyOut ctx ρ s n st out) := by
  unfold applyOut
  refine Foot.bind (Foot.foldlM _ fun ⟨c, k⟩ _ => applyPop_foot ρ h n c k) ?_
  refine Foot.bind (Foot.foldlM _ fun ⟨c, v⟩ _ => applyScalar_foot ρ h hn hc c v) ?_
  refine Foot.bind (Foot.foldlM _ fun ⟨c, rel, v⟩ _ => applyArrayWrite_foot ρ h n c rel v) ?_
  exact Foot.foldlM _ fun ⟨c, vs⟩ _ => applyPush_foot ρ h n c vs

theorem execCode_foot {n : DNode} (hn : n ∈ s.nodes) (hc : n.isCode = true)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (ue : ErrReason) :
    Foot ctx.I.wcr (fun st => execCode ctx ρ s n st run ue) := by
  unfold execCode
  by_cases hw : s.hasDynamicWcrOut n = true
  · simp only [hw, ↓reduceIte]
    exact Foot.err _
  · simp only [hw, Bool.false_eq_true, ↓reduceIte]
    refine Foot.bindA (bindInputs_foot ρ h n) (fun ins => ?_)
    dsimp only
    cases hr : run ins with
    | none => exact Foot.err _
    | some out => exact applyOut_foot ρ h hn hc out

theorem reduceData_foot (props : ReduceProps) (hid : props.identity.all ctx.exprPure = true) (sd : Name)
    (ssub : Option Subset) (dd : Name) (dsub : Option Subset) (outWcr : Option WcrOp) (soob doob : Bool)
    (hs : ctx.pointsPure sd ssub = true) (hd : ctx.pointsPure dd dsub = true) :
    Foot ctx.I.wcr (fun st => reduceData ctx ρ st props sd ssub dd dsub outWcr soob doob) := by
  unfold reduceData
  simp only [h.isStream, Bool.or_false, Bool.false_eq_true, ↓reduceIte]
  refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hs) (fun spts => ?_)
  refine Foot.bind_det (fun st st₂ _ => ctx.pointsOf_pure ρ _ _ hd) (fun dpts => ?_)
  split
  · exact Foot.err _
  · refine Foot.bindA (St.readMany_foot ctx ρ _ sd spts soob) (fun vals => ?_)
    dsimp only
    refine Foot.foldlM _ (fun x _ => ?_)
    split
    · next ide hide =>
      have hp : ctx.exprPure ide = true := by simpa [hide] using hid
      exact Foot.bind_det (fun st st₂ _ => ctx.evalR_pure ρ _ _ hp)
        (fun i => St.store_foot ctx ρ dd x.2 doob outWcr _)
    · split
      · exact Foot.pure
      · exact St.wcr_foot ctx ρ dd x.2 doob _ _

theorem execReduce_foot {n : DNode} (hn : n ∈ s.nodes) {l : LibNode} (hk : n.kind = .library l)
    (props : ReduceProps) (hr : l.reduce = some props) :
    Foot ctx.I.wcr (fun st => execReduce ctx ρ s n props st) := by
  have hid := h.reduces n hn l props hk hr
  unfold execReduce
  dsimp only
  split
  · next ie oe hins houts =>
    have hie : ie ∈ s.edges := mem_edges_of_in (List.mem_of_mem_filter (hins ▸ List.mem_cons_self))
    have hoe : oe ∈ s.edges := mem_edges_of_out (List.mem_of_mem_filter (houts ▸ List.mem_cons_self))
    split
    · next sd ssub dd dsub om hse hde _ =>
      exact reduceData_foot ρ h props hid sd ssub dd dsub _ _ _ (h.srcEndpoint_pure hie hse)
        (h.dstEndpoint_pure hoe hde)
    · exact Foot.err _
  · exact Foot.err _

theorem refSetTarget_none (e : DEdge) : refSetTarget? ctx s e = none := by
  unfold refSetTarget?
  dsimp only
  split
  · cases (s.node? (s.dstRootEdge e).dst).bind DNode.accessData? with
    | none => rfl
    | some d => simp [Option.filter, h.isReference]
  · rfl

theorem execCopiesInto_foot (n : DNode) : Foot ctx.I.wcr (fun st => execCopiesInto ctx ρ s n st) := by
  unfold execCopiesInto
  refine Foot.foldlM _ (fun e he => ?_)
  rw [refSetTarget_none h e]
  cases hd : e.data with
  | dep => exact Foot.pure
  | memlet m =>
    dsimp only
    split
    · exact Foot.pure
    · split
      · exact execCopy_foot ρ h (mem_edges_of_in he)
      · exact Foot.pure

theorem execSimple_foot {n : DNode} (hn : n ∈ s.nodes) : Foot ctx.I.wcr (fun st => execSimple ctx ρ s n st) := by
  unfold execSimple
  have hcode : ∀ {t : Tasklet}, n.kind = .tasklet t → n.isCode = true := fun hk => by simp [DNode.isCode, hk]
  have hcode' : ∀ {l : LibNode}, n.kind = .library l → n.isCode = true := fun hk => by simp [DNode.isCode, hk]
  cases hk : n.kind with
  | access d => exact execCopiesInto_foot ρ h n
  | mapExit => exact execCopiesInto_foot ρ h n
  | consumeExit => exact execCopiesInto_foot ρ h n
  | tasklet t => exact execCode_foot ρ h hn (hcode hk) _ _
  | library l =>
    dsimp only
    cases hr : l.reduce with
    | some props => exact execReduce_foot ρ h hn hk props hr
    | none => exact execCode_foot ρ h hn (hcode' hk) _ _
  | mapEntry _ => exact Foot.err _
  | consumeEntry _ => exact Foot.err _
  | nested _ _ _ => exact Foot.err _

theorem bindDynamicInputs_foot (n : DNode) (ρ₀ : Valuation) :
    FootA ctx.I.wcr (fun st => bindDynamicInputs ctx ρ₀ s n st) := by
  unfold bindDynamicInputs
  refine FootA.foldlM _ (fun e he ρ₁ => ?_) ρ₀
  dsimp only
  split
  · split
    · refine FootA.bind (readEdgeScalar_foot ρ₁ h (mem_edges_of_in he)) (fun v? => ?_)
      cases v? with
      | none => exact FootA.pure _
      | some v => exact FootA.pure _
    · exact FootA.pure _
  · exact FootA.pure _

/-- The views of a sequential execution: none in the fragment. -/
theorem bindViews_eq (ids : List NodeId) (st : St) : bindViews ctx ρ s ids st = .ok st := by
  unfold bindViews
  simp only [h.isView, Bool.false_eq_true, ↓reduceIte]
  induction ids generalizing st with
  | nil => rfl
  | cons id ids ih =>
    simp only [List.foldlM_cons]
    split
    · split <;> exact ih st
    · exact ih st

theorem mapSetup_frame {n : DNode} {mp : MapNode} (hr : ctx.rangePure mp.range = true) {st st₁ : St}
    {ρ' : Valuation} {pts : List Idx} (hs : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts)) :
    ∃ δ : Log, δ.Plain ∧ st₁ = st.step ctx.I.wcr δ ∧ ∀ st₂ : St, Mem.Agree δ.reads st₂.mem st.mem →
      st₂.vals = st.vals → mapSetup ctx ρ s n mp st₂ = .ok (ρ', st₂.step ctx.I.wcr δ, pts) := by
  unfold mapSetup at hs ⊢
  cases hb : bindDynamicInputs ctx ρ s n st with
  | err e => rw [hb] at hs; cases hs
  | ok p =>
    obtain ⟨ρ₁, st₁'⟩ := p
    rw [hb] at hs
    obtain ⟨δ, hp, rfl, hδ⟩ := bindDynamicInputs_foot h n ρ st ρ₁ st₁' hb
    refine ⟨δ, hp, ?_, fun st₂ hag hv => ?_⟩
    · cases hrp : ctx.rangePoints ρ₁ (st.step ctx.I.wcr δ).mem mp.range with
      | err e => simp only [Res.ok_bind, hrp] at hs; cases hs
      | ok q =>
        simp only [Res.ok_bind, hrp, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs
        exact hs.2.1.symm
    · rw [show bindDynamicInputs ctx ρ s n st₂ = _ from hδ st₂ hag hv]
      simp only [Res.ok_bind] at hs ⊢
      rw [ctx.rangePoints_pure ρ₁ _ (st.step ctx.I.wcr δ).mem hr]
      cases hrp : ctx.rangePoints ρ₁ (st.step ctx.I.wcr δ).mem mp.range with
      | err e => rw [hrp] at hs; cases hs
      | ok q =>
        simp only [hrp, Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs ⊢
        obtain ⟨h1, -, h3⟩ := hs
        simp [h1, h3]

end Leaf

theorem scopeFinish_ok {ctx : Ctx} {pre st' : St} {logs : List Log} (h : scopeFinish ctx pre logs = .ok st') :
    Log.pairwiseFree logs = true ∧ st' = pre.step ctx.I.wcr (Log.unionAll logs) := by
  unfold scopeFinish at h
  by_cases hf : Log.pairwiseFree logs = true
  · rw [merge_eq_apply _ _ hf] at h
    simp only [Res.ok_inj] at h
    exact ⟨hf, h.symm⟩
  · cases hm : merge ctx.I.wcr pre.mem logs with
    | ok m => exact absurd ((merge_isOk_iff _ _ _).1 ⟨m, hm⟩) hf
    | error e => rw [hm] at h; cases h

theorem scopeFinish_eq_step (ctx : Ctx) (pre : St) {logs : List Log} (hf : Log.pairwiseFree logs = true) :
    scopeFinish ctx pre logs = .ok (pre.step ctx.I.wcr (Log.unionAll logs)) := by
  unfold scopeFinish
  rw [merge_eq_apply _ _ hf]
  rfl

theorem St.afterNode_step (w : WcrSem) (st : St) (δ : Log) :
    st.afterNode (st.forNode.step w δ) = st.step w (Log.empty.seq δ) := by
  simp [St.afterNode, St.forNode, St.step, Log.empty_seq]

theorem Log.mem_unionAll_reads {gs : List Log} {g : Log} (hg : g ∈ gs) {l : Loc} (hl : l ∈ g.reads) :
    l ∈ (Log.unionAll gs).reads := by
  simp only [Log.unionAll, List.mem_flatMap]
  exact ⟨g, hg, hl⟩

/-! ## Frame theorems -/

/-- `st'` is reached from `st` by a plain delta, and every state agreeing
with `st` on its reads (same alias table and code-edge values) has a normal
`X`-outcome with the same delta. -/
def St.Footed (w : WcrSem) (X : St → Res St → Prop) (st st' : St) : Prop :=
  ∃ δ : Log, δ.Plain ∧ st' = st.step w δ ∧
    ∀ st₂ : St, Mem.Agree δ.reads st₂.mem st.mem → st₂.vals = st.vals → X st₂ (.ok (st₂.step w δ))

section FrameThm

variable {ctx : Ctx} {s : State} (h : SeqPure ctx s)
include h

set_option linter.unusedSectionVars false

mutual

/-- **Frame property of one node.** -/
theorem NodeExec.frame {ρ : Valuation} {n : DNode} {st : St} {r : Res St} (hn : n ∈ s.nodes) :
    NodeExec ctx ρ s n st r → ∀ st', r = .ok st' → St.Footed ctx.I.wcr (NodeExec ctx ρ s n) st st'
  | .simple hsim, st', hr => by
    obtain ⟨δ, hp, rfl, hδ⟩ := execSimple_foot ρ h hn st st' hr
    refine ⟨δ, hp, rfl, fun st₂ hag hv => ?_⟩
    rw [← hδ st₂ hag hv]
    exact .simple hsim
  | .mapSetupErr _ _, _, hr => by cases hr
  | .mapIterErr _ _ _, _, hr => by cases hr
  | .mapScope (mp := mp) hk hs hi, st', hr => by
    have ih := IterExec.frame (h.maps _ hn mp hk).2 hi _ rfl
    obtain ⟨hrange, -⟩ := h.maps _ hn mp hk
    obtain ⟨hf, rfl⟩ := scopeFinish_ok hr
    obtain ⟨δ₁, hp₁, rfl, hδ₁⟩ := mapSetup_frame _ h hrange hs
    obtain ⟨hpl, hit⟩ := ih
    refine ⟨δ₁.seq (Log.unionAll _), hp₁.seq (Log.Plain.unionAll hpl), St.step_step _ _ _ _,
      fun st₂ hag hv => ?_⟩
    rw [← St.step_step, ← scopeFinish_eq_step ctx _ hf]
    exact .mapScope hk (hδ₁ st₂ hag.seq_left hv) (hit _ (hag.seq_right _) hv)
  | .consumeSetupErr _ _, _, hr => by cases hr
  | .consumeTasksErr _ _ _, _, hr => by cases hr
  | .consumeScope hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .nestedSetupErr _ _, _, hr => by cases hr
  | .nestedInnerErr _ _ _, _, hr => by cases hr
  | .nested hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm

/-- **Frame property of a sequential execution** (same trace). -/
theorem SeqExec.frame {ρ : Valuation} {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s ids st r → ∀ st' tr, r = .ok (st', tr) →
      ∃ δ : Log, δ.Plain ∧ st' = st.step ctx.I.wcr δ ∧ ∀ st₂ : St, Mem.Agree δ.reads st₂.mem st.mem →
        st₂.vals = st.vals → SeqExec ctx ρ s ids st₂ (.ok (st₂.step ctx.I.wcr δ, tr))
  | .nil, st', tr, hr => by
    simp only [Res.ok_inj, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact ⟨Log.empty, Log.Plain.empty, (St.step_empty _ _).symm, fun st₂ _ _ => by
      rw [St.step_empty]; exact .nil⟩
  | .missing _, _, _, hr => by cases hr
  | .headErr _ _, _, _, hr => by cases hr
  | .tailErr _ _ _, _, _, hr => by cases hr
  | .cons (rest := rest) (tr := tr) hnode hn hrest, st', tr', hr => by
    have ihn := NodeExec.frame (List.mem_of_find?_eq_some hnode) hn _ rfl
    have ihr := SeqExec.frame hrest _ _ rfl
    simp only [Res.ok_inj, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    obtain ⟨δn, hpn, rfl, hδn⟩ := ihn
    rw [St.afterNode_step] at ihr
    obtain ⟨δr, hpr, rfl, hδr⟩ := ihr
    refine ⟨(Log.empty.seq δn).seq δr, (Log.Plain.empty.seq hpn).seq hpr, St.step_step _ _ _ _,
      fun st₂ hag hv => ?_⟩
    rw [← St.step_step]
    have hn₂ := hδn st₂.forNode (by have := hag.seq_left; rwa [Log.empty_seq] at this) hv
    have hrest₂ : SeqExec ctx ρ s rest (st₂.afterNode (st₂.forNode.step ctx.I.wcr δn))
        (.ok ((st₂.step ctx.I.wcr (Log.empty.seq δn)).step ctx.I.wcr δr, tr)) := by
      rw [St.afterNode_step]
      exact hδr _ (hag.seq_right _) hv
    have hc := SeqExec.cons hnode hn₂ hrest₂
    exact hc

/-- **Frame property of a scope body or component.** -/
theorem OrderedExec.frame {ρ : Valuation} {all direct : List NodeId} {st : St} {r : Res St} :
    OrderedExec ctx ρ s all direct st r → ∀ st', r = .ok st' →
      St.Footed ctx.I.wcr (OrderedExec ctx ρ s all direct) st st'
  | .prologueErr _, _, hr => by cases hr
  | .seqErr _ _ _, _, hr => by cases hr
  | .hbRace _ _ _ _, _, hr => by cases hr
  | .run ho hp hs hf, st', hr => by
    have ih := SeqExec.frame hs _ _ rfl
    simp only [Res.ok_inj] at hr
    subst hr
    rw [bindViews_eq ρ h] at hp
    simp only [Res.ok_inj] at hp
    subst hp
    obtain ⟨δ, hpl, rfl, hδ⟩ := ih
    exact ⟨δ, hpl, rfl, fun st₂ hag hv => .run ho (bindViews_eq ρ h _ _) (hδ st₂ hag hv) hf⟩

/-- **Frame property of the iterations of a map** without private
transients: the same logs from every pre-scope state agreeing on their
reads. -/
theorem IterExec.frame {ρ : Valuation} {entry : NodeId} {params : List Sym} {priv : List Name} {pre : St}
    {pts : List Idx} {r : Res (List Log)} (hpriv : priv = []) :
    IterExec ctx ρ s entry params priv pre pts r → ∀ logs, r = .ok logs →
      (∀ g ∈ logs, g.Plain) ∧ ∀ pre₂ : St, Mem.Agree (Log.unionAll logs).reads pre₂.mem pre.mem →
        pre₂.vals = pre.vals → IterExec ctx ρ s entry params priv pre₂ pts (.ok logs)
  | .nil, logs, hr => by
    simp only [Res.ok_inj] at hr
    subst hr
    exact ⟨fun _ hg => (by cases hg), fun _ _ _ => .nil⟩
  | .headErr _, _, hr => by cases hr
  | .tailErr _, _, hr => by cases hr
  | .cons hb hrest, logs, hr => by
    have ihb := OrderedExec.frame hb _ rfl
    have ihr := IterExec.frame hpriv hrest _ rfl
    simp only [Res.ok_inj] at hr
    subst hr
    subst hpriv
    obtain ⟨δ, hpl, rfl, hδ⟩ := ihb
    obtain ⟨hpls, hit⟩ := ihr
    refine ⟨?_, fun pre₂ hag hv => ?_⟩
    · intro g hg
      rcases List.mem_cons.1 hg with rfl | hg
      · rw [Log.dropNames_nil]; exact Log.Plain.empty.seq hpl
      · exact hpls g hg
    · have h₁ : Mem.Agree δ.reads (iterStart [] pre₂).mem (iterStart [] pre).mem := by
        refine ⟨hag.1, fun l hl => hag.2 l (Log.mem_unionAll_reads List.mem_cons_self ?_)⟩
        simpa [Log.dropNames_nil, St.step, iterStart, Log.empty_seq] using hl
      have h₂ : Mem.Agree (Log.unionAll _).reads pre₂.mem pre.mem := ⟨hag.1, fun l hl => hag.2 l (by
          simp only [Log.unionAll, List.flatMap_cons, List.mem_append] at hl ⊢; exact .inr hl)⟩
      have hc := IterExec.cons (hδ (iterStart [] pre₂) h₁ hv) (hit pre₂ h₂ hv)
      exact hc

end

omit h in
/-- A state with the empty log, stepped by `δ`, has log `δ`. -/
theorem St.step_emptyLog (w : WcrSem) (δ : Log) (st : St) :
    ({ st with log := Log.empty } : St).step w δ = { mem := δ.apply w st.mem, log := δ, vals := st.vals } := by
  simp [St.step, Log.empty_seq]

omit h in
/-- The `Footed` form specialized to a run from the empty log: the delta is
the final log. -/
theorem St.Footed.emptyLog {X : St → Res St → Prop} {st st' : St}
    (hf : St.Footed ctx.I.wcr X { st with log := Log.empty } st') :
    st'.mem = st'.log.apply ctx.I.wcr st.mem ∧ st'.vals = st.vals ∧ st'.log.Plain ∧
      ∀ st₂ : St, st₂.mem.aliases = st.mem.aliases → st₂.vals = st.vals →
        (∀ l ∈ st'.log.reads, st₂.mem.cells l = st.mem.cells l) →
        X { st₂ with log := Log.empty }
          (.ok { mem := st'.log.apply ctx.I.wcr st₂.mem, log := st'.log, vals := st₂.vals }) := by
  obtain ⟨δ, hp, rfl, hδ⟩ := hf
  rw [St.step_emptyLog]
  refine ⟨rfl, rfl, hp, fun st₂ ha hv hr => ?_⟩
  have := hδ { st₂ with log := Log.empty } ⟨ha, hr⟩ hv
  rwa [St.step_emptyLog] at this

/-- **Frame property of a node (`frame_ok`)**: a normal run from `st` with
an empty log ends in the local view `δ` applied to `st.mem`, where `δ` is
its log (*St-coherence*); from any `st₂` with the same alias table and
code-edge values that agrees with `st` on the cells `δ` reads, the node
runs normally with the same delta, ending in `δ` applied to `st₂.mem`. -/
theorem NodeExec.frame_ok {ρ : Valuation} {n : DNode} {st st' : St} (hn : n ∈ s.nodes)
    (hx : NodeExec ctx ρ s n { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem ∧ st'.vals = st.vals ∧ st'.log.Plain ∧
      ∀ st₂ : St, st₂.mem.aliases = st.mem.aliases → st₂.vals = st.vals →
        (∀ l ∈ st'.log.reads, st₂.mem.cells l = st.mem.cells l) →
        NodeExec ctx ρ s n { st₂ with log := Log.empty }
          (.ok { mem := st'.log.apply ctx.I.wcr st₂.mem, log := st'.log, vals := st₂.vals }) :=
  St.Footed.emptyLog (NodeExec.frame h hn hx _ rfl)

/-- **Frame property of a scope body or component (`frame_ok`)**, as
`NodeExec.frame_ok`. -/
theorem OrderedExec.frame_ok {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hx : OrderedExec ctx ρ s all direct { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem ∧ st'.vals = st.vals ∧ st'.log.Plain ∧
      ∀ st₂ : St, st₂.mem.aliases = st.mem.aliases → st₂.vals = st.vals →
        (∀ l ∈ st'.log.reads, st₂.mem.cells l = st.mem.cells l) →
        OrderedExec ctx ρ s all direct { st₂ with log := Log.empty }
          (.ok { mem := st'.log.apply ctx.I.wcr st₂.mem, log := st'.log, vals := st₂.vals }) :=
  St.Footed.emptyLog (OrderedExec.frame h hx _ rfl)

/-- **St-coherence**: along a run from an empty log, the local view is the
initial memory with the log applied. -/
theorem NodeExec.coherent {ρ : Valuation} {n : DNode} {st st' : St} (hn : n ∈ s.nodes)
    (hx : NodeExec ctx ρ s n { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem :=
  (NodeExec.frame_ok h hn hx).1

theorem OrderedExec.coherent {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hx : OrderedExec ctx ρ s all direct { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem :=
  (OrderedExec.frame_ok h hx).1

theorem SeqExec.coherent {ρ : Valuation} {ids : List NodeId} {st st' : St} {tr : List (NodeId × Log)}
    (hx : SeqExec ctx ρ s ids { st with log := Log.empty } (.ok (st', tr))) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem := by
  obtain ⟨δ, -, rfl, -⟩ := SeqExec.frame h hx _ _ rfl
  rw [St.step_emptyLog]

end FrameThm

/-! ## Coherence on `SeqPureP`

Coherence does not need the frame property: a normal run of a node steps
the state by a delta (a leaf by `execSimple_foot`, a map scope by its
set-up delta followed by the union of its iteration logs, private
transients dropped or not), so a run from an empty log ends in its log
applied to the initial memory (`OrderedExec.coh`). -/

section Coherence

/-- The node runs by itself: a simple node or a map scope. -/
def DNode.IsFlat (n : DNode) : Prop := n.isSimple = true ∨ ∃ mp, n.kind = .mapEntry mp

/-- Do all blocks of a region hold states? -/
def Region.statesOnly (rg : Region) : Bool :=
  rg.blocks.all fun b => match b with
    | .state _ => true
    | _ => false

/-- Every memlet out-edge of `n` ends at a container (not at a code node);
empty memlets only order. -/
def State.DataOuts (s : State) (n : DNode) : Prop :=
  ∀ e ∈ s.outEdges n.id, (∃ m, e.data = .memlet m) → (s.dstEndpoint e).isData = true

/-- A nested SDFG node whose call steps the state by the inner log: the call
allocates no transients and passes no scalars to or from code nodes. -/
def DNode.NestedFlat (s : State) (n : DNode) : Prop :=
  ∃ inner mapping ni, n.kind = .nested inner mapping ni ∧ inner.callTransients = [] ∧
    (∀ c ∈ nestedConns n, (nestedEndpoint s n c).isData = true) ∧ s.DataOuts n

/-- A node whose normal runs step the state by a delta (`NodeExec.steps`). -/
def DNode.FlatIn (s : State) (n : DNode) : Prop := n.IsFlat ∨ n.NestedFlat s

theorem DNode.FlatIn.of_flat {s : State} {n : DNode} (h : n.IsFlat) : n.FlatIn s := .inl h

/-! ### Regions of states step the state by their log -/

theorem applyAssigns_mem_log {ctx : Ctx} {cst c : CSt} {as : List (Sym × Expr)} (h : applyAssigns ctx cst as = .ok c) :
    c.mem = cst.mem ∧ c.log = cst.log := by
  unfold applyAssigns at h
  cases hv : as.mapM (fun (x, e) => do let v ← ctx.evalR cst.ρ cst.mem e; pure (x, v)) with
  | err e => rw [hv] at h; cases h
  | ok vs =>
    rw [hv] at h
    cases h
    exact ⟨rfl, rfl⟩

theorem merge_ok_free' {w : WcrSem} {m m' : Mem} {gs : List Log} (h : merge w m gs = .ok m') :
    Log.pairwiseFree gs = true :=
  (merge_isOk_iff _ _ _).1 ⟨m', h⟩

theorem StateExec.steps {ctx : Ctx} {cst : CSt} {s : State} {ctl : Ctl} {c : CSt}
    (h : StateExec ctx cst s (.ok (ctl, c))) : ∃ δ, c.mem = δ.apply ctx.I.wcr cst.mem ∧ c.log = cst.log.seq δ := by
  generalize hr : (Res.ok (ctl, c) : Res (Ctl × CSt)) = r at h
  cases h with
  | compsErr => cases hr
  | @run _ _ _ logs _ =>
    unfold stateFinish at hr
    split at hr
    · rename_i m' hm
      have hm' : m' = (Log.unionAll logs).apply ctx.I.wcr cst.mem := by
        rw [merge_eq_apply _ _ (merge_ok_free' hm)] at hm
        injection hm with h
        exact h.symm
      simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      obtain ⟨-, rfl⟩ := hr
      exact ⟨_, hm', rfl⟩
    · cases hr

theorem Region.statesOnly_block {rg : Region} (hs : rg.statesOnly = true) {bid : BlockId} {b : Block}
    (hb : rg.block? bid = some b) : ∃ st, b = .state st := by
  have hmem : b ∈ rg.blocks := List.mem_of_getElem? hb
  have := List.all_eq_true.1 hs b hmem
  cases b with
  | state st => exact ⟨st, rfl⟩
  | _ => simp at this

/-- `c` is `cst` stepped by a delta. -/
def CSt.Steps (ctx : Ctx) (cst c : CSt) : Prop := ∃ δ, c.mem = δ.apply ctx.I.wcr cst.mem ∧ c.log = cst.log.seq δ

theorem CSt.Steps.refl (ctx : Ctx) (cst : CSt) : cst.Steps ctx cst := ⟨Log.empty, rfl, (Log.seq_empty _).symm⟩

theorem CSt.Steps.trans {ctx : Ctx} {a b c : CSt} (h₁ : a.Steps ctx b) (h₂ : b.Steps ctx c) : a.Steps ctx c := by
  obtain ⟨δ₁, hm₁, hl₁⟩ := h₁
  obtain ⟨δ₂, hm₂, hl₂⟩ := h₂
  exact ⟨δ₁.seq δ₂, by rw [hm₂, hm₁, Log.apply_seq], by rw [hl₂, hl₁, Log.seq_assoc]⟩

theorem CSt.Steps.of_mem_log {ctx : Ctx} {a b : CSt} (hm : b.mem = a.mem) (hl : b.log = a.log) : a.Steps ctx b :=
  ⟨Log.empty, hm, by rw [hl, Log.seq_empty]⟩

theorem applyAssign?_steps {ctx : Ctx} {cst c : CSt} {a : Option (Sym × Expr)} (h : applyAssign? ctx cst a = .ok c) :
    cst.Steps ctx c := by
  cases a with
  | none => cases h; exact CSt.Steps.refl ctx cst
  | some a => obtain ⟨hm, hl⟩ := applyAssigns_mem_log h; exact CSt.Steps.of_mem_log hm hl

mutual

/-- A run of a region steps the state by its log. -/
theorem RegionAt.steps {ctx : Ctx} {rg : Region} :
    ∀ {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)}, RegionAt ctx rg cst bid r →
      ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | _, _, _, .exit _ hbx _, ctl, c, hr => BlockExec.steps hbx ctl c hr
  | _, _, _, .stop _ hbx _, ctl, c, hr => by cases hr; exact BlockExec.steps hbx _ _ rfl
  | _, _, _, .step _ hbx _ ha hn, ctl, c, hr => by
    obtain ⟨hm₂, hl₂⟩ := applyAssigns_mem_log ha
    exact ((BlockExec.steps hbx _ _ rfl).trans (CSt.Steps.of_mem_log hm₂ hl₂)).trans (RegionAt.steps hn ctl c hr)
  | _, _, _, .missing _, _, _, hr => by cases hr
  | _, _, _, .blockErr .., _, _, hr => by cases hr
  | _, _, _, .edgesErr .., _, _, hr => by cases hr
  | _, _, _, .ambiguous .., _, _, hr => by cases hr
  | _, _, _, .assignErr .., _, _, hr => by cases hr

theorem BlockExec.steps {ctx : Ctx} {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} :
    BlockExec ctx cst b r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .state h, ctl, c, hr => by
    subst hr
    obtain ⟨δ, hm, hl⟩ := StateExec.steps h
    exact ⟨δ, hm, hl⟩
  | .region h, ctl, c, hr => RegionExec.steps h ctl c hr
  | .loop h, ctl, c, hr => LoopExec.steps h ctl c hr
  | .cond h, ctl, c, hr => CondExec.steps h ctl c hr
  | .brk, _, _, hr => by cases hr; exact CSt.Steps.refl _ _
  | .cont, _, _, hr => by cases hr; exact CSt.Steps.refl _ _
  | .ret, _, _, hr => by cases hr; exact CSt.Steps.refl _ _

theorem RegionExec.steps {ctx : Ctx} {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)} :
    RegionExec ctx cst rg r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .start h, ctl, c, hr => RegionAt.steps h ctl c hr

theorem LoopExec.steps {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopExec ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .initErr _, _, _, hr => by cases hr
  | .guardFirst hi _ hg, ctl, c, hr => (applyAssign?_steps hi).trans (LoopGuard.steps hg ctl c hr)
  | .bodyFirst hi _ hb, ctl, c, hr => (applyAssign?_steps hi).trans (LoopIter.steps hb ctl c hr)

theorem LoopGuard.steps {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopGuard ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .condErr _, _, _, hr => by cases hr
  | .exit _, _, _, hr => by cases hr; exact CSt.Steps.refl _ _
  | .enter _ hb, ctl, c, hr => LoopIter.steps hb ctl c hr

theorem LoopIter.steps {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopIter ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .bodyErr _, _, _, hr => by cases hr
  | .ret hb, ctl, c, hr => RegionExec.steps hb ctl c hr
  | .brk hb, _, _, hr => by cases hr; exact RegionExec.steps hb _ _ rfl
  | .tail hb _ ht, ctl, c, hr => (RegionExec.steps hb _ _ rfl).trans (LoopTail.steps ht ctl c hr)

theorem LoopTail.steps {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopTail ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .updateErr .., _, _, hr => by cases hr
  | .updateGuard _ hu hg, ctl, c, hr => (applyAssign?_steps hu).trans (LoopGuard.steps hg ctl c hr)
  | .condFirstErr .., _, _, hr => by cases hr
  | .condFirstExit .., _, _, hr => by cases hr; exact CSt.Steps.refl _ _
  | .condFirstUpdateErr .., _, _, hr => by cases hr
  | .condFirstNext _ _ hu hb, ctl, c, hr => (applyAssign?_steps hu).trans (LoopIter.steps hb ctl c hr)

theorem CondExec.steps {ctx : Ctx} {cst : CSt} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)} :
    CondExec ctx cst brs r → ∀ ctl c, r = .ok (ctl, c) → cst.Steps ctx c
  | .nil, _, _, hr => by cases hr; exact CSt.Steps.refl _ _
  | .els h, ctl, c, hr => RegionExec.steps h ctl c hr
  | .condErr _, _, _, hr => by cases hr
  | .taken _ h, ctl, c, hr => RegionExec.steps h ctl c hr
  | .skipped _ h, ctl, c, hr => CondExec.steps h ctl c hr

end

theorem bindCodeInputs_data {ctx' : Ctx} {ρ' : Valuation} {s : State} {n : DNode}
    (h : ∀ c ∈ nestedConns n, (nestedEndpoint s n c).isData = true) (st : St) :
    bindCodeInputs ctx' ρ' s n st = .ok (st, []) := by
  unfold bindCodeInputs
  generalize nestedConns n = cs at h
  suffices H : ∀ (acc : St × List Name), (∀ c ∈ cs, (nestedEndpoint s n c).isData = true) →
      cs.foldlM (fun (p : St × List Name) c =>
        match nestedEndpoint s n c with
        | .code key =>
          match p.1.val? key with
          | some v => do
            let l ← ctx'.loc ρ' p.1.mem c []
            pure ({ p.1 with mem := p.1.mem.write l v }, p.2 ++ [ctx'.cname c])
          | none => pure (p.1, p.2 ++ [ctx'.cname c])
        | _ => pure (p.1, p.2)) acc = .ok acc from H (st, []) h
  intro acc hcs
  induction cs generalizing acc with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.foldlM_cons]
    have hc := hcs c List.mem_cons_self
    cases he : nestedEndpoint s n c with
    | code key => rw [he] at hc; cases hc
    | data _ _ =>
      have := ih (fun c' hc' => hcs c' (List.mem_cons_of_mem _ hc')) acc (fun c' hc' => hcs c' (List.mem_cons_of_mem _ hc'))
      simpa [he, Bind.bind, Res.bind, Pure.pure] using this
    | invalid =>
      have := ih (fun c' hc' => hcs c' (List.mem_cons_of_mem _ hc')) acc (fun c' hc' => hcs c' (List.mem_cons_of_mem _ hc'))
      simpa [he, Bind.bind, Res.bind, Pure.pure] using this

/-- Without code consumers of the outputs, a nested call ends with the inner
memory and the inner log without the dropped names. -/
theorem nestedFinish_data' {ctx' : Ctx} {ρ' : Valuation} {s : State} {n : DNode} {dropped : List Name} {pre : St}
    {m₂ : Mem} {g₂ : Log} (h : s.DataOuts n) :
    nestedFinish ctx' ρ' s n dropped pre m₂ g₂ =
      .ok { mem := m₂, log := pre.log.seq (g₂.dropNames dropped), vals := pre.vals } := by
  unfold nestedFinish
  generalize ({ mem := m₂, log := pre.log.seq (g₂.dropNames dropped), vals := pre.vals } : St) = acc
  unfold State.DataOuts at h
  generalize s.outEdges n.id = es at h ⊢
  induction es generalizing acc with
  | nil => rfl
  | cons e es ih =>
    rw [List.foldlM_cons]
    have hd := h e List.mem_cons_self
    have ih' := fun acc => ih acc (fun e' he' => h e' (List.mem_cons_of_mem _ he'))
    cases hdat : e.data with
    | dep => simp only; exact ih' acc
    | memlet mm =>
      cases hc : e.srcConn with
      | none => simp only; exact ih' acc
      | some c =>
        cases hE : s.dstEndpoint e with
        | data d sub => simp only; exact ih' acc
        | code k => rw [hE] at hd; cases hd ⟨mm, hdat⟩
        | invalid => simp only; exact ih' acc

theorem Log.dropNames_nil' (g : Log) : g.dropNames [] = g := by
  cases g; simp [Log.dropNames]

variable {ctx : Ctx} {s : State} (h : SeqPlain ctx s)
include h

/-- A normal run of a simple node or a map scope steps the state by a delta
(the body of a map scope is unrestricted). -/
theorem NodeExec.steps {ρ : Valuation} {n : DNode} {st a : St} (hn : n ∈ s.nodes) (hk : n.FlatIn s)
    (hx : NodeExec ctx ρ s n st (.ok a)) : ∃ δ, a = st.step ctx.I.wcr δ := by
  suffices H : ∀ r, NodeExec ctx ρ s n st r → r = .ok a → ∃ δ, a = st.step ctx.I.wcr δ from H _ hx rfl
  intro r hx hr
  cases hx with
  | simple _ =>
    obtain ⟨δ, -, hδ, -⟩ := execSimple_foot ρ h hn st a hr
    exact ⟨δ, hδ⟩
  | mapScope hk' hs _ =>
    obtain ⟨δ₁, -, rfl, -⟩ := mapSetup_frame ρ h (h.ranges _ hn _ hk') hs
    obtain ⟨-, rfl⟩ := scopeFinish_ok hr
    exact ⟨_, St.step_step _ _ _ _⟩
  | mapSetupErr => cases hr
  | mapIterErr => cases hr
  | consumeSetupErr => cases hr
  | consumeTasksErr => cases hr
  | nestedSetupErr => cases hr
  | nestedInnerErr => cases hr
  | consumeScope hk' _ _ =>
    exfalso
    rcases hk with (hs | ⟨mp, hm⟩) | ⟨_, _, _, hm, -⟩
    · simp [DNode.isSimple, hk'] at hs
    · rw [hk'] at hm; cases hm
    · rw [hk'] at hm; cases hm
  | @nested _ ctx' _ ρ' ρ₂ _ _ _ st₁ inner mapping _ aliased m₂ g₂ hk' hs hx =>
    rcases hk with (hs' | ⟨mp, hm⟩) | ⟨inner', mapping', ni', hkn, hct, hcode, hout⟩
    · simp [DNode.isSimple, hk'] at hs'
    · rw [hk'] at hm; cases hm
    rw [hk'] at hkn
    cases hkn
    unfold nestedSetup at hs
    cases hb : nestedBindings ctx s n with
    | err _ => rw [hb] at hs; cases hs
    | ok conns =>
      cases hv : nestedValuation ctx ρ st.mem mapping with
      | err _ => rw [hb] at hs; simp only [Res.ok_bind] at hs; rw [hv] at hs; cases hs
      | ok ρ'' =>
        rw [hb] at hs
        simp only [Res.ok_bind] at hs
        rw [hv] at hs
        simp only [Res.ok_bind, hct, List.map_nil, bindCodeInputs_data hcode, List.append_nil] at hs
        simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl, rfl, rfl⟩ := hs
        cases hx with
        | run hR _ =>
          cases hR with
          | start hR =>
            obtain ⟨δ, hm, hl⟩ := RegionAt.steps hR _ _ rfl
            rw [nestedFinish_data' hout] at hr
            simp only [Res.ok_inj] at hr
            subst hr
            refine ⟨δ, ?_⟩
            simp only [St.step, hm, hl, Log.empty_seq, Log.dropNames_nil']
            rfl

/-- A normal sequential run of simple nodes and map scopes steps the state
by a delta. -/
theorem SeqExec.steps {ρ : Valuation} :
    ∀ {ids : List NodeId} {st st₂ : St} {tr : List (NodeId × Log)},
      (∀ id ∈ ids, ∀ n, s.node? id = some n → n.FlatIn s) →
      SeqExec ctx ρ s ids st (.ok (st₂, tr)) → ∃ δ, st₂ = st.step ctx.I.wcr δ
  | [], st, _, _, _, hx => by cases hx; exact ⟨Log.empty, (St.step_empty _ _).symm⟩
  | id :: _, st, _, _, hk, hx => by
    cases hx with
    | cons hnode hn hr =>
      obtain ⟨δ, rfl⟩ := NodeExec.steps h (List.mem_of_find?_eq_some hnode) (hk id List.mem_cons_self _ hnode) hn
      obtain ⟨δ', rfl⟩ := SeqExec.steps (fun i hi => hk i (List.mem_cons_of_mem _ hi)) hr
      rw [St.afterNode_step, St.step_step]
      exact ⟨_, rfl⟩

/-- A normal run of a scope body or component whose direct nodes are simple
nodes and map scopes steps the state by a delta. -/
theorem OrderedExec.steps {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hk : ∀ id ∈ direct, ∀ n, s.node? id = some n → n.FlatIn s)
    (hx : OrderedExec ctx ρ s all direct st (.ok st')) : ∃ δ, st' = st.step ctx.I.wcr δ := by
  suffices H : ∀ r, OrderedExec ctx ρ s all direct st r → r = .ok st' → ∃ δ, st' = st.step ctx.I.wcr δ from
    H _ hx rfl
  intro r hx hr
  cases hx with
  | prologueErr => cases hr
  | seqErr => cases hr
  | hbRace => cases hr
  | run _ hp hs _ =>
    cases hr
    rw [bindViews_eq ρ h] at hp
    cases hp
    exact SeqExec.steps h (fun i hi => hk i (by simpa using (List.mem_filter.1 hi).2)) hs

/-- **St-coherence**: along a run from an empty log of a scope body or
component whose direct nodes are simple nodes and map scopes, the local view
is the initial memory with the log applied, and the code-edge values are
unchanged. -/
theorem OrderedExec.coh' {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hk : ∀ id ∈ direct, ∀ n, s.node? id = some n → n.FlatIn s)
    (hx : OrderedExec ctx ρ s all direct { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem ∧ st'.vals = st.vals := by
  obtain ⟨δ, rfl⟩ := OrderedExec.steps h hk hx
  rw [St.step_emptyLog]
  exact ⟨rfl, rfl⟩

end Coherence

/-- **St-coherence on `SeqPureP`**: along a run from an empty log, the local
view is the initial memory with the log applied, and the code-edge values
are unchanged. -/
theorem OrderedExec.coh {ctx : Ctx} {s : State} (h : SeqPureP ctx s) {ρ : Valuation} {all direct : List NodeId}
    {st st' : St} (hx : OrderedExec ctx ρ s all direct { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem ∧ st'.vals = st.vals :=
  OrderedExec.coh' h.toSeqPlain (fun _ _ n hn => .inl (h.kinds n (List.mem_of_find?_eq_some hn))) hx

end Dace
