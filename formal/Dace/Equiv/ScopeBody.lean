-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Congr
import Dace.Equiv.Order

/-!
# Congruence: replacing the body of a map scope

The second congruence principle of the equivalence framework (R1.5, R1
design §2.3): replacing the body of one map scope of a state by a body whose
iterations are matched by the old body's iterations refines the SDFG
(`Congr.scopeBody`; both directions give `Equiv`, `Congr.scopeBodyEquiv`).
An atom proves only `BodyMatchedE`: one iteration of the new body, from any
pre-map state and valuation, is matched by one iteration of the old body,
comparing the logs without the map's private transients.

## The side condition

The new state `s'` and the old state `s` must agree on everything outside the
bodies `B = s.scopeNodes M`, `B' = s'.scopeNodes M` of the map with entry
`M`.  Equal outer nodes and edges (`State.outerNodes`, `State.outerEdges`)
are hypotheses (`State.withBody` builds `s'` so that they hold, checked by
`State.bodyFits?`); everything else is the decidable check `sameOutside?`
(soundness: `sameOutside?_spec`, giving `OuterFacts`):

* **locality** (`NodeLocal`, `EdgeSame`): every leaf function executes an
  outer node the same way in both states when its adjacent edges have the
  same memlet-tree endpoints, routing/dynamic flags, reference targets and
  view-definition flags (`edgeFp`), and outer view nodes see the same view
  target (`State.viewBase?`);
* **body shape** (`bodyShape?`): the body is entered only from `M`, every
  new body node has an in-edge, old body nodes reach the outside only
  through nodes of the new body (the shared exit), and the old body has a
  topological order;
* **orders** (`State.isTopologicalOrder_splice`): a topological order of a
  scope list of `s'` becomes one of the corresponding list of `s` by dropping
  `B'` and inserting an order of `B` right after `M`; the outer nodes keep
  their relative order, so the sequential executions run the same outer
  nodes in the same order;
* **happens-before**: reachability between outer nodes in `s` implies
  reachability in `s'`, so a race of the new run is a race of the old one;
* **components and enclosing scopes** (`scopesSame?`): the components
  correspond index-wise (their top-level lists equal, or permuted when they
  contain no view node: components are discovered in node order, and
  binding views is the only order-dependent use of the list); every
  enclosing map scope has corresponding node lists, equal direct children
  and equal private transients; `M`'s private transients are equal;
* `sameTransientMentions?`, so that `Congr.block` keeps every private set.

## Fragment

Outer consume scopes are rejected (their tasks are not matched); the entry
`M` must be a map entry.  Nested SDFGs outside the body are supported (their
invocation is local).  The matching (`StateExec.matchedB`) is a mutual
induction over the new run (`NodeExec.matchedB`, `SeqExec.matchedB`,
`OrderedExec.matchedB`, `IterExec.matchedB`); the replaced map's iterations come
from `BodyMatchedE` (`IterExec.matchedBody`), merges respect `Log.Le` (`merge_le`),
and a merge that fails in the old run is a race (undefined behavior).
`tests/ScopeBodyCheck.lean` runs the check on every map of the corpus
(unchanged and rebuilt bodies pass; a body cut off from its entry fails).
-/

set_option autoImplicit false

namespace Dace

theorem List.foldlM_congr_mem {α β : Type} {l : List α} {f g : β → α → Res β}
    (h : ∀ x ∈ l, ∀ b, f b x = g b x) (b : β) : l.foldlM f b = l.foldlM g b := by
  induction l generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    rw [h x (List.mem_cons_self ..) b]
    congr 1
    funext b'
    exact ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) b'

namespace State

/-- The nodes outside the body `B` (a list of node ids). -/
def outerNodes (s : State) (B : List NodeId) : List DNode := s.nodes.filter fun n => !B.contains n.id

/-- The edges that do not end in the body `B`. -/
def outerEdges (s : State) (B : List NodeId) : List DEdge := s.edges.filter fun e => !B.contains e.dst

/-- What `bindView` reads from the state: the viewed container and subset. -/
def viewBase? (s : State) (v : DNode) : Option (Option Name × Option Subset) :=
  (s.viewEdge? v).map fun (e, isIn) =>
    (if isIn then (s.node? (s.pathStart e).src).bind DNode.accessData?
     else (s.node? (s.pathEnd e).dst).bind DNode.accessData?,
     if isIn then State.leafSrcSubset e else State.leafDstSubset e)

end State

theorem bindView_eq (ctx : Ctx) (ρ : Valuation) (s : State) (v : DNode) (d : Name) (st : St) :
    bindView ctx ρ s v d st =
      match s.viewBase? v with
      | none => .err (.malformed s!"ambiguous or invalid view edge for '{d}'")
      | some (none, _) => .err (.malformed s!"view edge of '{d}' does not reach an access node")
      | some (some base, sub) => aliasTo ctx ρ st d base sub := by
  unfold bindView State.viewBase?
  rcases s.viewEdge? v with _ | ⟨e, isIn⟩
  · rfl
  · simp only [Option.map_some]
    split <;> simp_all

/-- What the leaf functions read about one edge. -/
structure EdgeSame (ctx : Ctx) (s' s : State) (e : DEdge) : Prop where
  src : s'.srcEndpoint e = s.srcEndpoint e
  dst : s'.dstEndpoint e = s.dstEndpoint e
  routing : s'.isRouting e = s.isRouting e
  dynamic : s'.isDynamicInput e = s.isDynamicInput e
  refTarget : refSetTarget? ctx s' e = refSetTarget? ctx s e
  viewDef : ctx.isViewDef s' e = ctx.isViewDef s e

/-- A node sees the same adjacent edges, with the same endpoints, in two
states. -/
structure NodeLocal (ctx : Ctx) (s' s : State) (n : DNode) : Prop where
  inE : s'.inEdges n.id = s.inEdges n.id
  outE : s'.outEdges n.id = s.outEdges n.id
  edgeIn : ∀ e ∈ s.inEdges n.id, EdgeSame ctx s' s e
  edgeOut : ∀ e ∈ s.outEdges n.id, EdgeSame ctx s' s e

section Local

variable {ctx : Ctx} {s' s : State} {n : DNode} (h : NodeLocal ctx s' s n) {ρ : Valuation}
include h

omit h in
theorem State.mem_inEdgeOn {c : Conn} {e : DEdge} (he : s.inEdgeOn n c = some e) : e ∈ s.inEdges n.id :=
  List.mem_of_find?_eq_some he

theorem NodeLocal.inEdgeOn (c : Conn) : s'.inEdgeOn n c = s.inEdgeOn n c := by
  simp only [State.inEdgeOn, h.inE]

theorem NodeLocal.outEdgesOn (c : Conn) : s'.outEdgesOn n c = s.outEdgesOn n c := by
  simp only [State.outEdgesOn, h.outE]

omit h in
theorem State.mem_outEdgesOn {c : Conn} {e : DEdge} (he : e ∈ s.outEdgesOn n c) : e ∈ s.outEdges n.id :=
  (List.mem_filter.1 he).1

theorem NodeLocal.execCopiesInto (st : St) : execCopiesInto ctx ρ s' n st = execCopiesInto ctx ρ s n st := by
  unfold Dace.execCopiesInto
  rw [h.inE]
  refine List.foldlM_congr_mem (fun e he st => ?_) st
  have hs := h.edgeIn e he
  simp only [refSet, execCopy, hs.src, hs.dst, hs.routing, hs.refTarget, hs.viewDef]

theorem NodeLocal.bindInputs (st : St) : Dace.bindInputs ctx ρ s' n st = Dace.bindInputs ctx ρ s n st := by
  unfold Dace.bindInputs
  rw [h.inE]
  congr 1
  refine List.foldlM_congr_mem (fun e he x => ?_) _
  have hs := h.edgeIn e he
  simp only [bindIn, hs.src]

theorem NodeLocal.applyPop (st : St) (c : Conn) (k : Nat) :
    Dace.applyPop ctx ρ s' n st c k = Dace.applyPop ctx ρ s n st c k := by
  unfold Dace.applyPop
  rw [h.inEdgeOn]
  split
  · rfl
  · rename_i e he
    rw [(h.edgeIn e (State.mem_inEdgeOn he)).src]

theorem NodeLocal.applyScalar (st : St) (c : Conn) (v : Val) :
    Dace.applyScalar ctx ρ s' n st c v = Dace.applyScalar ctx ρ s n st c v := by
  unfold Dace.applyScalar
  rw [h.outEdgesOn]
  refine List.foldlM_congr_mem (fun e he st => ?_) st
  simp only [(h.edgeOut e (State.mem_outEdgesOn he)).dst]

theorem NodeLocal.applyArrayWrite (st : St) (c : Conn) (rel : Idx) (v : Val) :
    Dace.applyArrayWrite ctx ρ s' n st c rel v = Dace.applyArrayWrite ctx ρ s n st c rel v := by
  unfold Dace.applyArrayWrite
  rw [h.outEdgesOn]
  refine List.foldlM_congr_mem (fun e he st => ?_) st
  simp only [(h.edgeOut e (State.mem_outEdgesOn he)).dst]

theorem NodeLocal.applyPush (st : St) (c : Conn) (vs : List Val) :
    Dace.applyPush ctx ρ s' n st c vs = Dace.applyPush ctx ρ s n st c vs := by
  unfold Dace.applyPush
  rw [h.outEdgesOn]
  refine List.foldlM_congr_mem (fun e he st => ?_) st
  simp only [(h.edgeOut e (State.mem_outEdgesOn he)).dst]

theorem NodeLocal.applyOut (st : St) (out : TaskletOut) :
    Dace.applyOut ctx ρ s' n st out = Dace.applyOut ctx ρ s n st out := by
  unfold Dace.applyOut
  simp only [h.applyPop, h.applyScalar, h.applyArrayWrite, h.applyPush]

theorem NodeLocal.execCode (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut) (err : ErrReason) :
    Dace.execCode ctx ρ s' n st run err = Dace.execCode ctx ρ s n st run err := by
  unfold Dace.execCode State.hasDynamicWcrOut
  simp only [h.outE, h.bindInputs, h.applyOut]

theorem NodeLocal.execReduce (props : ReduceProps) (st : St) :
    Dace.execReduce ctx ρ s' n props st = Dace.execReduce ctx ρ s n props st := by
  unfold Dace.execReduce
  rw [h.inE, h.outE]
  dsimp only
  split
  · rename_i ie oe hi ho
    have hie : ie ∈ s.inEdges n.id := by
      have := List.mem_singleton_self ie
      rw [← hi] at this
      exact List.mem_of_mem_filter this
    have hoe : oe ∈ s.outEdges n.id := by
      have := List.mem_singleton_self oe
      rw [← ho] at this
      exact List.mem_of_mem_filter this
    rw [(h.edgeIn ie hie).src, (h.edgeOut oe hoe).dst]
  · rfl

theorem NodeLocal.execSimple (st : St) : Dace.execSimple ctx ρ s' n st = Dace.execSimple ctx ρ s n st := by
  unfold Dace.execSimple
  split
  · exact h.execCopiesInto st
  · exact h.execCopiesInto st
  · exact h.execCopiesInto st
  · exact h.execCode st _ _
  · split
    · exact h.execReduce _ st
    · exact h.execCode st _ _
  · rfl
  · rfl
  · rfl

theorem NodeLocal.bindDynamicInputs (st : St) :
    Dace.bindDynamicInputs ctx ρ s' n st = Dace.bindDynamicInputs ctx ρ s n st := by
  unfold Dace.bindDynamicInputs
  rw [h.inE]
  refine List.foldlM_congr_mem (fun e he x => ?_) _
  have hs := h.edgeIn e he
  simp only [readEdgeScalar, hs.src, hs.dynamic]

theorem NodeLocal.mapSetup (mp : MapNode) (st : St) :
    Dace.mapSetup ctx ρ s' n mp st = Dace.mapSetup ctx ρ s n mp st := by
  unfold Dace.mapSetup
  rw [h.bindDynamicInputs]

theorem NodeLocal.nestedEndpoint (c : Conn) : Dace.nestedEndpoint s' n c = Dace.nestedEndpoint s n c := by
  unfold Dace.nestedEndpoint
  rw [h.inEdgeOn, h.outEdgesOn]
  cases hi : s.inEdgeOn n c with
  | some e => simp only [(h.edgeIn e (State.mem_inEdgeOn hi)).src]
  | none =>
    cases ho : s.outEdgesOn n c with
    | nil => rfl
    | cons e rest =>
      simp only [(h.edgeOut e (State.mem_outEdgesOn (by rw [ho]; exact List.mem_cons_self ..))).dst]

theorem NodeLocal.nestedSetup (inner : SDFG) (mapping : List (Sym × Expr)) (st : St) :
    Dace.nestedSetup ctx ρ s' n inner mapping st = Dace.nestedSetup ctx ρ s n inner mapping st := by
  unfold Dace.nestedSetup Dace.nestedBindings Dace.bindCodeInputs
  simp only [h.nestedEndpoint]

theorem NodeLocal.nestedFinish (ctx₂ : Ctx) (ρ' : Valuation) (aliased : List Name) (pre : St) (m₂ : Mem)
    (g₂ : Log) :
    Dace.nestedFinish ctx₂ ρ' s' n aliased pre m₂ g₂ = Dace.nestedFinish ctx₂ ρ' s n aliased pre m₂ g₂ := by
  unfold Dace.nestedFinish
  rw [h.outE]
  refine List.foldlM_congr_mem (fun e he x => ?_) _
  simp only [(h.edgeOut e he).dst]

end Local

/-! ## Transporting a topological order to the old body -/

namespace State

/-- `a` comes before `b` in `o`. -/
def Before (o : List NodeId) (a b : NodeId) : Prop := ∃ l₁ l₂, o = l₁ ++ l₂ ∧ a ∈ l₁ ∧ b ∈ l₂

theorem idxOf?_append_of_mem {l₁ l₂ : List NodeId} {a : NodeId} (h : a ∈ l₁) :
    (l₁ ++ l₂).idxOf? a = l₁.idxOf? a := by
  obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr h)
  unfold List.idxOf? at hi ⊢
  rw [List.findIdx?_append, hi]
  rfl

theorem idxOf?_append_of_not_mem {l₁ l₂ : List NodeId} {a : NodeId} (h : a ∉ l₁) :
    (l₁ ++ l₂).idxOf? a = (l₂.idxOf? a).map (· + l₁.length) := by
  have hn := List.idxOf?_eq_none_iff.mpr h
  unfold List.idxOf? at hn ⊢
  rw [List.findIdx?_append, hn]
  rfl

theorem mem_of_idxOf? {o : List NodeId} {a : NodeId} {i : Nat} (h : o.idxOf? a = some i) : a ∈ o := by
  obtain ⟨hi, ha, -⟩ := List.idxOf?_eq_some_iff.mp h
  exact ha ▸ List.getElem_mem hi

theorem before_iff {o : List NodeId} (hnd : o.Nodup) {a b : NodeId} :
    Before o a b ↔ ∃ i j, o.idxOf? a = some i ∧ o.idxOf? b = some j ∧ i < j := by
  constructor
  · rintro ⟨l₁, l₂, rfl, ha, hb⟩
    have hb₁ : b ∉ l₁ := fun hb₁ => (List.nodup_append.1 hnd).2.2 b hb₁ b hb rfl
    obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr ha)
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr hb)
    refine ⟨i, j + l₁.length, by rw [idxOf?_append_of_mem ha, hi], by rw [idxOf?_append_of_not_mem hb₁, hj]; rfl, ?_⟩
    obtain ⟨hlt, -⟩ := List.idxOf?_eq_some_iff.mp hi
    omega
  · rintro ⟨i, j, hi, hj, hij⟩
    obtain ⟨hi', hai, -⟩ := List.idxOf?_eq_some_iff.mp hi
    obtain ⟨hj', hbj, -⟩ := List.idxOf?_eq_some_iff.mp hj
    refine ⟨o.take j, o.drop j, (List.take_append_drop j o).symm, ?_, ?_⟩
    · exact List.mem_take_iff_getElem.2 ⟨i, by omega, hai⟩
    · exact List.mem_drop_iff_getElem.2 ⟨0, by omega, by simpa using hbj⟩

theorem Before.trans {o : List NodeId} (hnd : o.Nodup) {a b c : NodeId} (h₁ : Before o a b) (h₂ : Before o b c) :
    Before o a c := by
  obtain ⟨i, j, hi, hj, hij⟩ := (before_iff hnd).1 h₁
  obtain ⟨j', k, hj', hk, hjk⟩ := (before_iff hnd).1 h₂
  rw [hj] at hj'
  cases hj'
  exact (before_iff hnd).2 ⟨i, k, hi, hk, by omega⟩

theorem Before.append_left {o : List NodeId} {a b : NodeId} (h : Before o a b) (p : List NodeId) :
    Before (p ++ o) a b := by
  obtain ⟨l₁, l₂, rfl, ha, hb⟩ := h
  exact ⟨p ++ l₁, l₂, by simp, List.mem_append_right _ ha, hb⟩

theorem Before.append_right {o : List NodeId} {a b : NodeId} (h : Before o a b) (q : List NodeId) :
    Before (o ++ q) a b := by
  obtain ⟨l₁, l₂, rfl, ha, hb⟩ := h
  exact ⟨l₁, l₂ ++ q, by simp, ha, List.mem_append_left _ hb⟩

theorem respectsEdges_iff {s : State} {o : List NodeId} (hnd : o.Nodup) :
    s.respectsEdges o = true ↔ ∀ e ∈ s.edges, e.src ∈ o → e.dst ∈ o → Before o e.src e.dst := by
  simp only [respectsEdges, List.all_eq_true]
  constructor
  · intro h e he hs hd
    obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr hs)
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr hd)
    have := h e he
    rw [hi, hj] at this
    exact (before_iff hnd).2 ⟨i, j, hi, hj, by simpa using this⟩
  · intro h e he
    split
    · rename_i i j hi hj
      obtain ⟨i', j', hi', hj', hlt⟩ := (before_iff hnd).1 (h e he (mem_of_idxOf? hi) (mem_of_idxOf? hj))
      rw [hi] at hi'
      rw [hj] at hj'
      cases hi'
      cases hj'
      simpa using hlt
    · rfl

/-- In a topological order of a state where every body node has an in-edge
from the body or the entry, the entry precedes every body node. -/
theorem entry_before {s : State} {o B : List NodeId} {M : NodeId} (hnd : o.Nodup)
    (hresp : s.respectsEdges o = true) (hM : M ∈ o) (hBo : ∀ x ∈ B, x ∈ o)
    (hin : ∀ x ∈ B, ∃ e ∈ s.edges, e.dst = x) (hsrc : ∀ e ∈ s.edges, e.dst ∈ B → e.src ∈ B ∨ e.src = M) :
    ∀ x ∈ B, Before o M x := by
  have key : ∀ k x, x ∈ B → o.idxOf? x = some k → Before o M x := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
      intro x hx hk
      obtain ⟨e, he, rfl⟩ := hin _ hx
      rcases hsrc e he hx with hs | hs
      · have hb := (respectsEdges_iff hnd).1 hresp e he (hBo _ hs) (hBo _ hx)
        obtain ⟨i, j, hi, hj, hij⟩ := (before_iff hnd).1 hb
        rw [hk] at hj
        cases hj
        exact (ih i hij _ hs hi).trans hnd hb
      · rw [← hs]
        exact (respectsEdges_iff hnd).1 hresp e he (hs ▸ hM) (hBo _ hx)
  intro x hx
  obtain ⟨k, hk⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr (hBo x hx))
  exact key k x hx hk

/-- Replace the body nodes `B'` of an order by the order `Bo`, placed right
after the entry `M`. -/
def splice (B' : List NodeId) (M : NodeId) (Bo : List NodeId) (o : List NodeId) : List NodeId :=
  (o.filter fun x => !B'.contains x).flatMap fun x => if x = M then M :: Bo else [x]

section Splice

variable {B' Bo : List NodeId} {M : NodeId}

theorem splice_append (l₁ l₂ : List NodeId) : splice B' M Bo (l₁ ++ l₂) = splice B' M Bo l₁ ++ splice B' M Bo l₂ := by
  simp [splice, List.filter_append, List.flatMap_append]

theorem splice_cons_entry (hM : M ∉ B') (l : List NodeId) : splice B' M Bo (M :: l) = M :: Bo ++ splice B' M Bo l := by
  simp [splice, hM]

theorem mem_splice {o : List NodeId} {x : NodeId} :
    x ∈ splice B' M Bo o ↔ (x ∈ o ∧ x ∉ B') ∨ (M ∈ o ∧ M ∉ B' ∧ x ∈ Bo) := by
  simp only [splice, List.mem_flatMap, List.mem_filter, Bool.not_eq_true', List.contains_eq_mem,
    decide_eq_false_iff_not]
  constructor
  · rintro ⟨y, ⟨hy, hyB⟩, hx⟩
    split at hx
    · rename_i hyM
      subst hyM
      rcases List.mem_cons.1 hx with rfl | hx
      · exact .inl ⟨hy, hyB⟩
      · exact .inr ⟨hy, hyB, hx⟩
    · rcases List.mem_singleton.1 hx with rfl
      exact .inl ⟨hy, hyB⟩
  · rintro (⟨hx, hxB⟩ | ⟨hM, hMB, hx⟩)
    · refine ⟨x, ⟨hx, hxB⟩, ?_⟩
      split
      · rename_i h; subst h; exact List.mem_cons_self ..
      · exact List.mem_singleton_self _
    · exact ⟨M, ⟨hM, hMB⟩, by simp [hx]⟩

theorem filter_splice {o : List NodeId} {p : NodeId → Bool} (hp : ∀ x ∈ Bo, p x = false) :
    (splice B' M Bo o).filter p = (o.filter fun x => !B'.contains x).filter p := by
  unfold splice
  rw [List.filter_flatMap]
  generalize o.filter (fun x => !B'.contains x) = l
  have hBo : Bo.filter p = [] := List.filter_eq_nil_iff.2 fun x hx => by simp [hp x hx]
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [List.flatMap_cons, ih]
    by_cases hx : x = M
    · subst hx
      cases hpx : p x <;> simp [hpx, hBo]
    · cases hpx : p x <;> simp [hpx, hx]

theorem nodup_splice {o : List NodeId} (ho : o.Nodup) (hBo : Bo.Nodup) (hM : M ∉ Bo)
    (hdisj : ∀ x ∈ o, x ∉ B' → x ∉ Bo) : (splice B' M Bo o).Nodup := by
  unfold splice
  have hf := ho.filter (fun x => !B'.contains x)
  rw [List.nodup_flatMap]
  refine ⟨fun x _ => ?_, ?_⟩
  · split
    · exact List.nodup_cons.2 ⟨hM, hBo⟩
    · exact List.nodup_singleton _
  · refine (List.nodup_iff_pairwise_ne.1 hf).imp_of_mem fun {a b} ha hb hab => ?_
    simp only [List.mem_filter, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at ha hb
    have hna := hdisj a ha.1 ha.2
    have hnb := hdisj b hb.1 hb.2
    intro x hxa hxb
    by_cases haM : a = M
    · subst haM
      have hbM : b ≠ a := Ne.symm hab
      simp only [hbM, ite_false, ite_true, List.mem_cons, List.not_mem_nil, or_false] at hxa hxb
      subst hxb
      rcases hxa with h | h
      · exact hab h.symm
      · exact hnb h
    · by_cases hbM : b = M
      · subst hbM
        simp only [haM, ite_false, ite_true, List.mem_cons, List.not_mem_nil, or_false] at hxa hxb
        subst hxa
        rcases hxb with h | h
        · exact hab h
        · exact hna h
      · simp only [haM, hbM, ite_false, List.mem_singleton] at hxa hxb
        exact hab (hxa.symm.trans hxb)

end Splice

/-- **Order transport.**  A topological order `o'` of the scope list `all'`
of the new state gives one of the corresponding list `all` of the old state:
drop the new body `B'` and insert a topological order `Bo` of the old body
`B` right after the entry `M`.  Every node outside both bodies keeps its
relative position (`filter_splice`). -/
theorem isTopologicalOrder_splice {s' s : State} {B' B Bo all' all o' : List NodeId} {M : NodeId}
    (ho' : s'.isTopologicalOrder all' o' = true) (hall' : all'.Nodup) (hall : all.Nodup)
    (hperm : (all.filter fun x => !B.contains x).Perm (all'.filter fun x => !B'.contains x))
    (hMB : M ∉ B) (hMB' : M ∉ B')
    (hin : M ∈ all → (∀ x ∈ B, x ∈ all) ∧ ∀ x ∈ B', x ∈ all')
    (hout : M ∉ all → ∀ x ∈ all, x ∉ B)
    (hBo : s.isTopologicalOrder B Bo = true) (hB : B.Nodup)
    (houter : ∀ e ∈ s.edges, e.dst ∉ B → e ∈ s'.edges)
    (hsrc : ∀ e ∈ s.edges, e.dst ∈ B → e.src ∈ B ∨ e.src = M)
    (hsrc' : ∀ e ∈ s'.edges, e.dst ∈ B' → e.src ∈ B' ∨ e.src = M)
    (hin' : ∀ x ∈ B', ∃ e ∈ s'.edges, e.dst = x)
    (hexit : ∀ e ∈ s.edges, e.src ∈ B → e.dst ∉ B → e.src ∈ B') :
    s.isTopologicalOrder all (splice B' M Bo o') = true := by
  have hp' := isTopologicalOrder_perm hall' ho'
  have hresp' : s'.respectsEdges o' = true := by
    simp only [isTopologicalOrder, Bool.and_eq_true] at ho'
    exact ho'.2
  have hnd' : o'.Nodup := hp'.nodup_iff.2 hall'
  have hpB := isTopologicalOrder_perm hB hBo
  have hndBo : Bo.Nodup := hpB.nodup_iff.2 hB
  have hrespB : s.respectsEdges Bo = true := by
    simp only [isTopologicalOrder, Bool.and_eq_true] at hBo
    exact hBo.2
  -- membership in the outer parts
  have hout_iff : ∀ x, (x ∈ all' ∧ x ∉ B') ↔ (x ∈ all ∧ x ∉ B) := by
    intro x
    have := hperm.mem_iff (a := x)
    simp only [List.mem_filter, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at this
    exact this.symm
  have hM_iff : M ∈ all' ↔ M ∈ all := by
    have := hout_iff M
    simp only [hMB, hMB', not_false_eq_true, and_true] at this
    exact this
  have hmem : ∀ x, x ∈ splice B' M Bo o' ↔ x ∈ all := by
    intro x
    rw [mem_splice]
    simp only [hp'.mem_iff, hpB.mem_iff]
    rw [hout_iff, hM_iff]
    constructor
    · rintro (⟨h, -⟩ | ⟨hM, -, hx⟩)
      · exact h
      · exact (hin hM).1 x hx
    · intro hx
      by_cases hxB : x ∈ B
      · by_cases hM : M ∈ all
        · exact .inr ⟨hM, hMB', hxB⟩
        · exact absurd hxB (hout hM x hx)
      · exact .inl ⟨hx, hxB⟩
  have hnd : (splice B' M Bo o').Nodup := by
    refine nodup_splice hnd' hndBo (fun h => hMB (hpB.mem_iff.1 h)) fun x hx hxB' hxBo => ?_
    have := ((hout_iff x).1 ⟨hp'.mem_iff.1 hx, hxB'⟩).2
    exact this (hpB.mem_iff.1 hxBo)
  have hperm' : (splice B' M Bo o').Perm all := (List.perm_ext_iff_of_nodup hnd hall).2 hmem
  have houter_mem : ∀ x, x ∈ splice B' M Bo o' → x ∉ B → x ∈ o' ∧ x ∉ B' := by
    intro x hx hxB
    rcases mem_splice.1 hx with h | ⟨-, -, hxBo⟩
    · exact h
    · exact absurd (hpB.mem_iff.1 hxBo) hxB
  have hbody_mem : ∀ x, x ∈ splice B' M Bo o' → x ∈ B → M ∈ o' ∧ x ∈ Bo := by
    intro x hx hxB
    rcases mem_splice.1 hx with ⟨h, hxB'⟩ | ⟨hM, -, hxBo⟩
    · have := ((hout_iff x).1 ⟨hp'.mem_iff.1 h, hxB'⟩).2
      exact absurd hxB this
    · exact ⟨hM, hxBo⟩
  have hresp : s.respectsEdges (splice B' M Bo o') = true := by
    refine (respectsEdges_iff hnd).2 fun e he hs hd => ?_
    by_cases hdB : e.dst ∈ B
    · obtain ⟨hM, hdBo⟩ := hbody_mem _ hd hdB
      have hsrc_e := hsrc e he hdB
      have hsBo : e.src ∈ B → e.src ∈ Bo := fun hsB => (hbody_mem _ hs hsB).2
      obtain ⟨l₁, l₂, rfl⟩ := List.mem_iff_append.1 hM
      rw [splice_append, splice_cons_entry hMB']
      rcases hsrc_e with hsB | hsM
      · obtain ⟨m₁, m₂, hm, ha, hb⟩ := (respectsEdges_iff hndBo).1 hrespB e he (hsBo hsB) hdBo
        generalize splice B' M Bo l₁ = p₁
        generalize splice B' M Bo l₂ = p₂
        rw [hm]
        exact ⟨p₁ ++ M :: m₁, m₂ ++ p₂, by simp, by simp [ha], by simp [hb]⟩
      · rw [hsM]
        exact ⟨splice B' M Bo l₁ ++ [M], Bo ++ splice B' M Bo l₂, by simp, by simp, by simp [hdBo]⟩
    · obtain ⟨hdo, hdB'⟩ := houter_mem _ hd hdB
      have he' := houter e he hdB
      by_cases hsB : e.src ∈ B
      · obtain ⟨hM, hsBo⟩ := hbody_mem _ hs hsB
        have hsB' := hexit e he hsB hdB
        have hMall : M ∈ all := hM_iff.1 (hp'.mem_iff.1 hM)
        have hB'o : ∀ x ∈ B', x ∈ o' := fun x hx => hp'.mem_iff.2 ((hin hMall).2 x hx)
        have h1 := entry_before hnd' hresp' hM hB'o hin' hsrc' _ hsB'
        have h2 := (respectsEdges_iff hnd').1 hresp' e he' (hB'o _ hsB') hdo
        obtain ⟨l₁, l₂, rfl, hMl, hdl⟩ := h1.trans hnd' h2
        exact ⟨splice B' M Bo l₁, splice B' M Bo l₂, splice_append .., mem_splice.2 (.inr ⟨hMl, hMB', hsBo⟩),
          mem_splice.2 (.inl ⟨hdl, hdB'⟩)⟩
      · obtain ⟨hso, hsB'⟩ := houter_mem _ hs hsB
        obtain ⟨l₁, l₂, rfl, hsl, hdl⟩ := (respectsEdges_iff hnd').1 hresp' e he' hso hdo
        exact ⟨splice B' M Bo l₁, splice B' M Bo l₂, splice_append .., mem_splice.2 (.inl ⟨hsl, hsB'⟩),
          mem_splice.2 (.inl ⟨hdl, hdB'⟩)⟩
  simp only [isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
  exact ⟨⟨⟨hperm'.length_eq, fun x hx => by simpa using hperm'.symm.subset hx⟩, hnd⟩, hresp⟩

end State

/-! ## Facts about the unchanged outside -/

/-- No node among `ids` is a view access node: binding views is a no-op. -/
def ViewFree (ctx : Ctx) (s : State) (ids : List NodeId) : Prop :=
  ∀ id ∈ ids, ∀ n, s.node? id = some n → ∀ d, n.kind = .access d → ctx.isView d = false

theorem List.foldlM_fixed {α β : Type} {f : β → α → Res β} {b : β} :
    ∀ {l : List α}, (∀ x ∈ l, f b x = .ok b) → l.foldlM f b = .ok b
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.foldlM_cons, h x (List.mem_cons_self ..)]
    exact List.foldlM_fixed fun y hy => h y (List.mem_cons_of_mem _ hy)

theorem bindViews_viewFree {ctx : Ctx} {ρ : Valuation} {s : State} {ids : List NodeId} (h : ViewFree ctx s ids)
    (st : St) : bindViews ctx ρ s ids st = .ok st := by
  unfold bindViews
  refine List.foldlM_fixed fun id hid => ?_
  cases hn : s.node? id with
  | none => rfl
  | some n =>
    dsimp only
    split
    · rename_i d hk
      rw [h id hid n hn d hk]
      rfl
    · rfl

/-- A scope list of the new state (a component, or the nodes of an enclosing
scope) corresponds to one of the old state: the same nodes outside the
bodies, and either both bodies inside (with their entry `M`) or neither. -/
structure ScopeCorr (B' B : List NodeId) (M : NodeId) (all' all : List NodeId) : Prop where
  nd : all.Nodup
  nd' : all'.Nodup
  perm : (all.filter fun x => !B.contains x).Perm (all'.filter fun x => !B'.contains x)
  hin : M ∈ all → (∀ x ∈ B, x ∈ all) ∧ ∀ x ∈ B', x ∈ all'
  hout : M ∉ all → ∀ x ∈ all, x ∉ B

/-- What the matching of `s'` by `s` (the body of the map `M` replaced)
needs about the part of the states outside the bodies `B = s.scopeNodes M`,
`B' = s'.scopeNodes M`.  `sameOutside?_spec` derives it from a check. -/
structure OuterFacts (ctx : Ctx) (s' s : State) (M : NodeId) : Prop where
  nodes : s'.outerNodes (s'.scopeNodes M) = s.outerNodes (s.scopeNodes M)
  edges : s'.outerEdges (s'.scopeNodes M) = s.outerEdges (s.scopeNodes M)
  entry : M ∉ s.scopeNodes M
  entry' : M ∉ s'.scopeNodes M
  src : ∀ e ∈ s.edges, e.dst ∈ s.scopeNodes M → e.src ∈ s.scopeNodes M ∨ e.src = M
  src' : ∀ e ∈ s'.edges, e.dst ∈ s'.scopeNodes M → e.src ∈ s'.scopeNodes M ∨ e.src = M
  fed' : ∀ x ∈ s'.scopeNodes M, ∃ e ∈ s'.edges, e.dst = x
  exit : ∀ e ∈ s.edges, e.src ∈ s.scopeNodes M → e.dst ∉ s.scopeNodes M → e.src ∈ s'.scopeNodes M
  bodyNodup : (s.scopeNodes M).Nodup
  bodyOrder : ∃ Bo, s.isTopologicalOrder (s.scopeNodes M) Bo = true
  noConsume : ∀ n ∈ s.outerNodes (s.scopeNodes M), ∀ c, n.kind ≠ .consumeEntry c
  entryKind : ∀ n ∈ s.outerNodes (s.scopeNodes M), n.id = M → ∃ mp, n.kind = .mapEntry mp
  edge : ∀ e ∈ s.outerEdges (s.scopeNodes M), EdgeSame ctx s' s e
  view : ∀ n ∈ s.outerNodes (s.scopeNodes M), ∀ d, n.kind = .access d → s'.viewBase? n = s.viewBase? n
  reach : ∀ a b, a ∈ (s.outerNodes (s.scopeNodes M)).map (·.id) → b ∈ (s.outerNodes (s.scopeNodes M)).map (·.id) →
    s.reaches a b = true → s'.reaches a b = true
  comps : List.Forall₂ (fun c' c => ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M c' c ∧
      (s'.topLevelOf c' = s.topLevelOf c ∨
        (s'.topLevelOf c').Perm (s.topLevelOf c) ∧ ViewFree ctx s (s.topLevelOf c)) ∧
      ∀ id ∈ s.topLevelOf c, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M)
    s'.components s.components
  priv : ∀ n ∈ s.outerNodes (s.scopeNodes M), ∀ mp, n.kind = .mapEntry mp →
    ctx.privateTransients s' n.id = ctx.privateTransients s n.id
  /-- The two states allocate the same transients (`Decisions.md#D68`). -/
  stateT : ctx.stateTransients s' = ctx.stateTransients s
  scopes : ∀ n ∈ s.outerNodes (s.scopeNodes M), ∀ mp, n.kind = .mapEntry mp → n.id ≠ M →
    ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M (s'.scopeNodes n.id) (s.scopeNodes n.id) ∧
    s'.directChildren (some n.id) = s.directChildren (some n.id) ∧
    ∀ id ∈ s.directChildren (some n.id), id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M

namespace OuterFacts

variable {ctx : Ctx} {s' s : State} {M : NodeId} (hF : OuterFacts ctx s' s M)
include hF

omit hF in
theorem find?_outer (l : List DNode) (B : List NodeId) {id : NodeId} (hid : id ∉ B) :
    (l.filter fun n => !B.contains n.id).find? (fun n => n.id = id) = l.find? (fun n => n.id = id) := by
  rw [List.find?_filter]
  congr 1
  funext n
  by_cases hn : n.id = id
  · subst hn; simp [hid]
  · simp [hn]

theorem node? {id : NodeId} (h : id ∉ s.scopeNodes M) (h' : id ∉ s'.scopeNodes M) : s'.node? id = s.node? id := by
  unfold State.node?
  rw [← find?_outer s.nodes _ h, ← find?_outer s'.nodes _ h']
  exact congrArg (List.find? _) hF.nodes

theorem mem_outer {id : NodeId} {n : DNode} (h : s'.node? id = some n) (h' : id ∉ s'.scopeNodes M) :
    n ∈ s.outerNodes (s.scopeNodes M) := by
  obtain ⟨hn, rfl⟩ := State.node?_some h
  rw [← hF.nodes]
  exact List.mem_filter.2 ⟨hn, by simpa using h'⟩

theorem outer_out {n : DNode} (h : n ∈ s.outerNodes (s.scopeNodes M)) :
    n.id ∉ s.scopeNodes M ∧ n.id ∉ s'.scopeNodes M := by
  have h' := h
  rw [← hF.nodes] at h'
  exact ⟨by simpa using (List.mem_filter.1 h).2, by simpa using (List.mem_filter.1 h').2⟩

omit hF in
theorem inEdges_filter (t : State) (B : List NodeId) {id : NodeId} (h : id ∉ B) :
    t.inEdges id = (t.outerEdges B).filter (·.dst = id) := by
  unfold State.inEdges State.outerEdges
  rw [List.filter_filter]
  refine List.filter_congr fun e _ => ?_
  by_cases he : e.dst = id
  · subst he; simp [h]
  · simp [he]

theorem inEdges {id : NodeId} (h : id ∉ s.scopeNodes M) (h' : id ∉ s'.scopeNodes M) :
    s'.inEdges id = s.inEdges id := by
  rw [inEdges_filter s _ h, inEdges_filter s' _ h', hF.edges]

theorem outEdges {id : NodeId} (h : id ∉ s.scopeNodes M) (h' : id ∉ s'.scopeNodes M) (hM : id ≠ M) :
    s'.outEdges id = s.outEdges id := by
  have key : ∀ (t : State) (B : List NodeId), id ∉ B →
      (∀ e ∈ t.edges, e.dst ∈ B → e.src ∈ B ∨ e.src = M) → t.outEdges id = (t.outerEdges B).filter (·.src = id) := by
    intro t B hB hsrc
    unfold State.outEdges State.outerEdges
    rw [List.filter_filter]
    refine List.filter_congr fun e he => ?_
    by_cases hs : e.src = id
    · subst hs
      have : e.dst ∉ B := fun hd => (hsrc e he hd).elim hB hM
      simp [this]
    · simp [hs]
  rw [key s _ h hF.src, key s' _ h' hF.src', hF.edges]

omit hF in
theorem mem_outerEdges_in {id : NodeId} {e : DEdge} (h : id ∉ s.scopeNodes M) (he : e ∈ s.inEdges id) :
    e ∈ s.outerEdges (s.scopeNodes M) := by
  rw [inEdges_filter s _ h] at he
  exact (List.mem_filter.1 he).1

theorem mem_outerEdges_out {id : NodeId} {e : DEdge} (h : id ∉ s.scopeNodes M) (hM : id ≠ M)
    (he : e ∈ s.outEdges id) : e ∈ s.outerEdges (s.scopeNodes M) := by
  unfold State.outEdges at he
  obtain ⟨he, hs⟩ := List.mem_filter.1 he
  simp only [decide_eq_true_eq] at hs
  refine List.mem_filter.2 ⟨he, ?_⟩
  have : e.dst ∉ s.scopeNodes M := fun hd => (hF.src e he hd).elim (hs ▸ h) (hs ▸ hM)
  simpa using this

/-- An outer node other than the entry is local. -/
theorem nodeLocal {n : DNode} (hn : n ∈ s.outerNodes (s.scopeNodes M)) (hM : n.id ≠ M) : NodeLocal ctx s' s n := by
  obtain ⟨h, h'⟩ := hF.outer_out hn
  exact ⟨hF.inEdges h h', hF.outEdges h h' hM, fun e he => hF.edge e (mem_outerEdges_in h he),
    fun e he => hF.edge e (hF.mem_outerEdges_out h hM he)⟩

theorem bindDynamicInputs {n : DNode} (hn : n ∈ s.outerNodes (s.scopeNodes M)) (ρ : Valuation) (st : St) :
    Dace.bindDynamicInputs ctx ρ s' n st = Dace.bindDynamicInputs ctx ρ s n st := by
  obtain ⟨h, h'⟩ := hF.outer_out hn
  unfold Dace.bindDynamicInputs
  rw [hF.inEdges h h']
  refine List.foldlM_congr_mem (fun e he x => ?_) _
  have hs := hF.edge e (mem_outerEdges_in h he)
  simp only [readEdgeScalar, hs.src, hs.dynamic]

theorem mapSetup {n : DNode} (hn : n ∈ s.outerNodes (s.scopeNodes M)) (ρ : Valuation) (mp : MapNode) (st : St) :
    Dace.mapSetup ctx ρ s' n mp st = Dace.mapSetup ctx ρ s n mp st := by
  unfold Dace.mapSetup
  rw [hF.bindDynamicInputs hn]

theorem bindViews {ρ : Valuation} {ids : List NodeId}
    (hids : ∀ id ∈ ids, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) (st : St) :
    Dace.bindViews ctx ρ s' ids st = Dace.bindViews ctx ρ s ids st := by
  unfold Dace.bindViews
  refine List.foldlM_congr_mem (fun id hid x => ?_) st
  obtain ⟨h, h'⟩ := hids id hid
  rw [hF.node? h h']
  cases hn : s.node? id with
  | none => rfl
  | some n =>
    have hmem : n ∈ s.outerNodes (s.scopeNodes M) := hF.mem_outer (by rw [hF.node? h h']; exact hn) h'
    dsimp only
    split
    · rename_i d hk
      rw [bindView_eq, bindView_eq, hF.view n hmem d hk]
    · rfl

theorem viewedData {id : NodeId} (h : id ∉ s.scopeNodes M) (h' : id ∉ s'.scopeNodes M)
    (hv : s.isViewNode ctx id = true) :
    (s'.node? id).bind s'.viewedData? = (s.node? id).bind s.viewedData? := by
  rw [hF.node? h h']
  unfold State.isViewNode at hv
  cases hn : s.node? id with
  | none => rfl
  | some n =>
    rw [hn] at hv
    simp only [Option.bind_some]
    cases hk : n.kind with
    | access d =>
      have hmem : n ∈ s.outerNodes (s.scopeNodes M) := hF.mem_outer (by rw [hF.node? h h']; exact hn) h'
      have := hF.view n hmem d hk
      unfold State.viewBase? at this
      unfold State.viewedData?
      cases he' : s'.viewEdge? n <;> cases he : s.viewEdge? n <;> rw [he', he] at this <;> simp at this ⊢
      · rename_i p' p
        obtain ⟨e', i'⟩ := p'
        obtain ⟨e, i⟩ := p
        simp only at this ⊢
        exact this.1
    | _ => simp [hk] at hv

theorem viewGens_congr (f : Nat) (pending : List NodeId)
    (h : ∀ id ∈ pending, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M ∧ s.isViewNode ctx id = true) :
    s'.viewGens f pending = s.viewGens f pending :=
  viewGensWith_congr f pending fun id hid =>
    ⟨by rw [hF.node? (h id hid).1 (h id hid).2.1], hF.viewedData (h id hid).1 (h id hid).2.1 (h id hid).2.2⟩

theorem viewSig {id : NodeId} (h : id ∉ s.scopeNodes M) (h' : id ∉ s'.scopeNodes M) :
    s'.viewSig id = s.viewSig id := by
  unfold State.viewSig
  rw [hF.node? h h']
  cases hn : s.node? id with
  | none => rfl
  | some n =>
    simp only [Option.bind_some]
    cases hk : n.kind with
    | access d =>
      have hmem : n ∈ s.outerNodes (s.scopeNodes M) := hF.mem_outer (by rw [hF.node? h h']; exact hn) h'
      have := hF.view n hmem d hk
      have hb : s'.viewBinding? n = s.viewBinding? n := this
      rw [hb]
    | _ => simp [DNode.accessData?, hk]

theorem viewOrder {ids : List NodeId} (hids : ∀ id ∈ ids, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) :
    s'.viewOrder ctx ids = s.viewOrder ctx ids := by
  unfold State.viewOrder
  have hf : ids.filter (s'.isViewNode ctx) = ids.filter (s.isViewNode ctx) := List.filter_congr fun id hid => by
    unfold State.isViewNode
    rw [hF.node? (hids id hid).1 (hids id hid).2]
  simp only
  rw [hf, keepFirstSig_congr fun id hid => by
    have hm := List.mem_filter.1 hid
    exact hF.viewSig (hids id hm.1).1 (hids id hm.1).2]
  exact hF.viewGens_congr _ _ fun id hid => by
    have hm := List.mem_filter.1 (keepFirstSig_sub hid)
    exact ⟨(hids id hm.1).1, (hids id hm.1).2, hm.2⟩

end OuterFacts

theorem State.viewOrder_viewFree {ctx : Ctx} {s : State} {ids : List NodeId} (h : ViewFree ctx s ids) :
    s.viewOrder ctx ids = [] := by
  unfold State.viewOrder
  have : ids.filter (s.isViewNode ctx) = [] := List.filter_eq_nil_iff.2 fun id hid => by
    unfold State.isViewNode
    cases hn : s.node? id with
    | none => simp
    | some n =>
      cases hk : n.kind with
      | access d => simp [hk, h id hid n hn d hk]
      | _ => simp [hk]
  simp only
  rw [this]
  rfl

/-! ## Relations and helpers -/

/-- Related dataflow states: same memory and code values, refined log. -/
def St.RelV (a' a : St) : Prop := a'.mem = a.mem ∧ a'.vals = a.vals ∧ Log.Le a'.log a.log

theorem St.RelV.refl (a : St) : St.RelV a a := ⟨rfl, rfl, Log.Le.refl _⟩

theorem St.RelV.trans {a b c : St} (h₁ : St.RelV a b) (h₂ : St.RelV b c) : St.RelV a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.trans h₂.2.2⟩

/-- Related trace entries: same node, refined delta. -/
def TrRel (x' x : NodeId × Log) : Prop := x'.1 = x.1 ∧ Log.Le x'.2 x.2

/-- Related sequential outcomes. -/
def SeqRelV (x' x : St × List (NodeId × Log)) : Prop := St.RelV x'.1 x.1 ∧ List.Forall₂ TrRel x'.2 x.2

theorem forall₂_trRel_refl : ∀ (t : List (NodeId × Log)), List.Forall₂ TrRel t t
  | [] => .nil
  | _ :: t => .cons ⟨rfl, Log.Le.refl _⟩ (forall₂_trRel_refl t)

theorem forall₂_trRel_trans : ∀ {a b c : List (NodeId × Log)}, List.Forall₂ TrRel a b → List.Forall₂ TrRel b c →
    List.Forall₂ TrRel a c
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons h₁ t₁, .cons h₂ t₂ => .cons ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩ (forall₂_trRel_trans t₁ t₂)

theorem SeqRelV.trans' : ∀ a b c : St × List (NodeId × Log), SeqRelV a b → SeqRelV b c → SeqRelV a c :=
  fun _ _ _ h₁ h₂ => ⟨h₁.1.trans h₂.1, forall₂_trRel_trans h₁.2 h₂.2⟩

theorem forall₂_le_refl : ∀ (t : List Log), List.Forall₂ Log.Le t t
  | [] => .nil
  | _ :: t => .cons (Log.Le.refl _) (forall₂_le_refl t)

namespace MatchedE

variable {α : Type} {R : α → α → Prop} {X : Res α → Prop}

/-- An error matched by `X`: `X` has the same error or undefined behavior. -/
theorem err_cases {e : ErrReason} (h : MatchedE R X (.err e)) :
    ∃ e₀, X (.err e₀) ∧ ∀ {β : Type} (S : β → β → Prop), Res.Below S (.err e) (.err e₀) := by
  obtain ⟨r, hr, hb⟩ := h
  obtain ⟨e₀, rfl⟩ := hb.err_inv
  exact ⟨e₀, hr, fun S => hb.err_err⟩

/-- A normal outcome matched by `X`: undefined behavior, or a related normal outcome. -/
theorem ok_cases {a' : α} (h : MatchedE R X (.ok a')) :
    (∃ e, e.isUB = true ∧ X (.err e)) ∨ ∃ a, X (.ok a) ∧ R a' a := by
  obtain ⟨r, hr, hb⟩ := h
  rcases hb.ok_inv with ⟨e, rfl, he⟩ | ⟨a, rfl, ha⟩
  · exact .inl ⟨e, he, hr⟩
  · exact .inr ⟨a, hr, ha⟩

end MatchedE

section Helpers

variable {ctx : Ctx} {ρ : Valuation} {s : State}

/-- Sequential executions from related start states have related outcomes. -/
theorem SeqExec.transferV {ids : List NodeId} {a' a : St} (hm : a'.mem = a.mem) (hv : a'.vals = a.vals)
    (hle : Log.Le a'.log a.log) {r : Res (St × List (NodeId × Log))} (h : SeqExec ctx ρ s ids a' r) :
    ∃ r₀, SeqExec ctx ρ s ids a r₀ ∧ Res.Rel SeqRelV r r₀ := by
  obtain ⟨h₁, h₂⟩ := St.eq_shift_of hm hv
  rw [h₁] at h
  obtain ⟨r₁, hr₁, rfl⟩ := (SeqExec.shift_iff _).1 h
  have h₃ := hr₁.shift a.log
  rw [← h₂] at h₃
  refine ⟨_, h₃, ?_⟩
  cases r₁ with
  | ok x => exact ⟨⟨rfl, rfl, Log.Le.seq hle (Log.Le.refl _)⟩, forall₂_trRel_refl _⟩
  | err e => rfl

theorem St.RelV.afterNode (st : St) {a' a : St} (h : St.RelV a' a) :
    (st.afterNode a').mem = (st.afterNode a).mem ∧ (st.afterNode a').vals = (st.afterNode a).vals ∧
      Log.Le (st.afterNode a').log (st.afterNode a).log :=
  ⟨h.1, h.2.1, Log.Le.seq (Log.Le.refl _) h.2.2⟩

theorem merge_ok_free {w : WcrSem} {m m' : Mem} {gs : List Log} (h : merge w m gs = .ok m') :
    Log.pairwiseFree gs = true := by
  unfold merge at h
  split at h
  · assumption
  · split at h <;> cases h

theorem scopeFinish_below (hw : ctx.I.wcr.Lawful) (pre : St) {logs' logs : List Log}
    (h : List.Forall₂ Log.Le logs' logs) : Res.Below St.RelV (scopeFinish ctx pre logs') (scopeFinish ctx pre logs) := by
  unfold scopeFinish
  cases hm : merge ctx.I.wcr pre.mem logs with
  | error e => exact .inl ⟨_, rfl, rfl⟩
  | ok m' =>
    obtain ⟨-, heq⟩ := merge_le ctx.I.wcr hw pre.mem h (merge_ok_free hm)
    rw [heq, hm]
    exact .inr ⟨rfl, rfl, Log.Le.seq (Log.Le.refl _) (Log.Le.unionAll h)⟩

theorem stateFinish_below (hw : ctx.I.wcr.Lawful) (cst : CSt) {logs' logs : List Log}
    (h : List.Forall₂ Log.Le logs' logs) : Res.Below CtlRel (stateFinish ctx cst logs') (stateFinish ctx cst logs) := by
  unfold stateFinish
  cases hm : merge ctx.I.wcr cst.mem logs with
  | error e => exact .inl ⟨_, rfl, rfl⟩
  | ok m' =>
    obtain ⟨-, heq⟩ := merge_le ctx.I.wcr hw cst.mem h (merge_ok_free hm)
    rw [heq, hm]
    exact .inr ⟨rfl, rfl, rfl, Log.Le.seq (Log.Le.refl _) (Log.Le.unionAll h)⟩

end Helpers

/-- The body matching the congruence assumes: every run of one iteration
of the new body (from the pre-map state, under any valuation) is matched
by a run of an iteration of the old body, comparing the logs without the
map's private transients. -/
def BodyMatchedE (ctx : Ctx) (s' s : State) (M : NodeId) : Prop :=
  ∀ (ρ : Valuation) (pre : St) (r' : Res St),
    OrderedExec ctx ρ s' (s'.scopeNodes M) (s'.directChildren (some M)) (iterStart (ctx.privateTransients s M) pre) r' →
      MatchedE (fun a' a => Log.Le (a'.log.dropNames (ctx.privateTransients s M)) (a.log.dropNames (ctx.privateTransients s M)))
        (OrderedExec ctx ρ s (s.scopeNodes M) (s.directChildren (some M)) (iterStart (ctx.privateTransients s M) pre)) r'

/-! ## The matching -/

theorem SeqExec.trace_mem {ctx : Ctx} {ρ : Valuation} {s : State} :
    ∀ {ids : List NodeId} {st st₂ : St} {tr : List (NodeId × Log)},
      SeqExec ctx ρ s ids st (.ok (st₂, tr)) → ∀ x ∈ tr, x.1 ∈ ids ∧ ∃ n, s.node? x.1 = some n
  | _, _, _, _, .nil => by simp
  | _, _, _, _, .cons h _ hr => by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨List.mem_cons_self .., _, h⟩
    · obtain ⟨h₁, h₂⟩ := SeqExec.trace_mem hr x hx
      exact ⟨List.mem_cons_of_mem _ h₁, h₂⟩

namespace OuterFacts

variable {ctx : Ctx} {s' s : State} {M : NodeId} (hF : OuterFacts ctx s' s M)
include hF

theorem hbFree_false {tr' tr : List (NodeId × Log)} (h : List.Forall₂ TrRel tr' tr)
    (hout : ∀ x ∈ tr', x.1 ∈ (s.outerNodes (s.scopeNodes M)).map (·.id)) (hf : s'.hbFree tr' = false) :
    s.hbFree tr = false := by
  have h' : List.Forall₂ (fun x' x => TrRel x' x ∧ x'.1 ∈ (s.outerNodes (s.scopeNodes M)).map (·.id)) tr' tr := by
    clear hf
    induction h with
    | nil => exact .nil
    | cons hab _ ih =>
      exact .cons ⟨hab, hout _ (List.mem_cons_self ..)⟩ (ih fun x hx => hout x (List.mem_cons_of_mem _ hx))
  cases hc : s.hbFree tr with
  | false => rfl
  | true =>
    rw [State.hbFree_iff] at hc
    have := ListLemmas.pairwise_of_forall₂ (R' := fun x y => s'.hbConflict x y = false) (fun a' a b' b ha hb hab => ?_) h' hc
    · rw [← State.hbFree_iff] at this
      rw [this] at hf
      cases hf
    · obtain ⟨⟨ha₁, ha₂⟩, haO⟩ := ha
      obtain ⟨⟨hb₁, hb₂⟩, hbO⟩ := hb
      have key : Log.conflicts a.2 b.2 = false ∨ s.reaches a.1 b.1 = true ∨ s.reaches b.1 a.1 = true := by
        revert hab
        unfold State.hbConflict
        cases Log.conflicts a.2 b.2 <;> cases s.reaches a.1 b.1 <;> cases s.reaches b.1 a.1 <;> simp
      unfold State.hbConflict
      rw [ha₁] at haO
      rw [hb₁] at hbO
      rcases key with hk | hk | hk
      · rw [Log.conflicts_mono ha₂.sub hb₂.sub hk]
        rfl
      · rw [ha₁, hb₁, hF.reach _ _ haO hbO hk]
        simp
      · rw [ha₁, hb₁, hF.reach _ _ hbO haO hk]
        simp

theorem transport {all' all o' direct : List NodeId} (ho : s'.isTopologicalOrder all' o' = true)
    (hc : ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M all' all)
    (hd : ∀ id ∈ direct, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) :
    ∃ o, s.isTopologicalOrder all o = true ∧ o.filter direct.contains = o'.filter direct.contains := by
  obtain ⟨Bo, hBo⟩ := hF.bodyOrder
  refine ⟨State.splice (s'.scopeNodes M) M Bo o', State.isTopologicalOrder_splice ho hc.nd' hc.nd hc.perm hF.entry
    hF.entry' hc.hin hc.hout hBo hF.bodyNodup (fun e he hd => ?_) hF.src hF.src' hF.fed' hF.exit, ?_⟩
  · have : e ∈ s.outerEdges (s.scopeNodes M) := List.mem_filter.2 ⟨he, by simpa using hd⟩
    rw [← hF.edges] at this
    exact (List.mem_filter.1 this).1
  · have hpB := State.isTopologicalOrder_perm hF.bodyNodup hBo
    rw [State.filter_splice fun x hx => ?_, List.filter_filter]
    · refine List.filter_congr fun x _ => ?_
      cases hx : direct.contains x
      · simp
      · have := (hd x (by simpa using hx)).2
        simp [this]
    · cases hx' : direct.contains x
      · rfl
      · exact absurd (hpB.mem_iff.1 hx) (hd x (by simpa using hx')).1

theorem ne_entry_of_kind {n : DNode} (hn : n ∈ s.outerNodes (s.scopeNodes M))
    (hk : ∀ mp, n.kind ≠ .mapEntry mp) : n.id ≠ M := fun h => by
  obtain ⟨mp, hmp⟩ := hF.entryKind n hn h
  exact hk mp hmp

omit hF in
theorem topPerm {l' l : List NodeId} (h : l' = l ∨ l'.Perm l ∧ ViewFree ctx s l) : l'.Perm l := by
  rcases h with rfl | ⟨h, -⟩
  · exact List.Perm.refl _
  · exact h

theorem topViews {l' l : List NodeId} (h : l' = l ∨ l'.Perm l ∧ ViewFree ctx s l)
    (hout : ∀ id ∈ l, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) (ρ : Valuation) (st : St) :
    Dace.bindViews ctx ρ s' (s'.viewOrder ctx l') st = Dace.bindViews ctx ρ s (s.viewOrder ctx l) st := by
  rcases h with rfl | ⟨hp, hv⟩
  · rw [hF.viewOrder hout]
    exact hF.bindViews (fun id hid => hout id (State.viewOrder_sub hid)) st
  · have hv' : ViewFree ctx s' l' := by
      intro id hid n hn d hk
      have hid' := hp.mem_iff.1 hid
      obtain ⟨h₁, h₂⟩ := hout id hid'
      exact hv id hid' n ((hF.node? h₁ h₂).symm.trans hn) d hk
    rw [State.viewOrder_viewFree hv, State.viewOrder_viewFree hv']
    rfl

end OuterFacts

section Matched

variable {ctx : Ctx} {s' s : State} {M : NodeId}

/-- The iterations of the replaced map, from the body matching. -/
theorem IterExec.matchedBody (hbody : BodyMatchedE ctx s' s M) {ρ : Valuation} {params : List Sym}
    {pre : St} {pts : List Idx} {r' : Res (List Log)} :
    IterExec ctx ρ s' M params (ctx.privateTransients s M) pre pts r' →
      MatchedE (List.Forall₂ Log.Le) (IterExec ctx ρ s M params (ctx.privateTransients s M) pre pts) r'
  | .nil => MatchedE.of_rel .nil List.Forall₂.nil
  | .headErr hb => by
    obtain ⟨e₀, hr, hbl⟩ := (hbody _ _ _ hb).err_cases
    exact ⟨_, .headErr hr, hbl _⟩
  | .tailErr hb hr => by
    rcases (hbody _ _ _ hb).ok_cases with ⟨e, he, hx⟩ | ⟨a, hx, -⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · obtain ⟨e₀, hr₀, hb₀⟩ := MatchedE.err_cases (IterExec.matchedBody hbody hr)
      exact ⟨_, .tailErr hx hr₀, hb₀ _⟩
  | .cons hb hr => by
    rcases (hbody _ _ _ hb).ok_cases with ⟨e, he, hx⟩ | ⟨a, hx, hle⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · rcases MatchedE.ok_cases (IterExec.matchedBody hbody hr) with ⟨e, he, hx'⟩ | ⟨logs, hx', hl⟩
      · exact MatchedE.of_ub (.tailErr hx hx') he
      · exact MatchedE.of_rel (.cons hx hx') (List.Forall₂.cons hle hl)

variable (hF : OuterFacts ctx s' s M) (hw : ctx.I.wcr.Lawful) (hbody : BodyMatchedE ctx s' s M)
include hF hw hbody

set_option linter.unusedSectionVars false

mutual

theorem NodeExec.matchedB {ρ : Valuation} {n : DNode} {st : St} {r' : Res St} :
    NodeExec ctx ρ s' n st r' → n ∈ s.outerNodes (s.scopeNodes M) → MatchedE St.RelV (NodeExec ctx ρ s n st) r'
  | .simple hs, hn => by
    have hM : n.id ≠ M := hF.ne_entry_of_kind hn fun mp hk => by simp [DNode.isSimple, hk] at hs
    rw [(hF.nodeLocal hn hM).execSimple]
    exact MatchedE.of_rel (.simple hs) (Res.Rel.refl St.RelV.refl _)
  | .mapSetupErr hk hs, hn => MatchedE.of_rel (.mapSetupErr hk (by rw [← hF.mapSetup hn]; exact hs)) rfl
  | .mapIterErr (ρ' := ρ') (st₁ := st₁) (pts := pts) (mp := mp) (e := e) hk hs hi, hn => by
    have hs' : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts) := by rw [← hF.mapSetup hn]; exact hs
    have hsim : MatchedE (List.Forall₂ Log.Le)
        (IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s' n.id) st₁ pts) (.err e) := by
      by_cases hM : n.id = M
      · rw [hF.priv n hn mp hk] at hi ⊢
        rw [hM] at hi ⊢
        exact IterExec.matchedBody hbody hi
      · obtain ⟨hc, hd, ho⟩ := hF.scopes n hn mp hk hM
        exact IterExec.matchedB hi hc hd ho
    rw [hF.priv n hn mp hk] at hsim
    obtain ⟨e₀, hr, hb⟩ := hsim.err_cases
    exact ⟨_, .mapIterErr hk hs' hr, hb _⟩
  | .mapScope (ρ' := ρ') (st₁ := st₁) (pts := pts) (mp := mp) (logs := logs') hk hs hi, hn => by
    have hs' : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts) := by rw [← hF.mapSetup hn]; exact hs
    have hsim : MatchedE (List.Forall₂ Log.Le)
        (IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s' n.id) st₁ pts) (.ok logs') := by
      by_cases hM : n.id = M
      · rw [hF.priv n hn mp hk] at hi ⊢
        rw [hM] at hi ⊢
        exact IterExec.matchedBody hbody hi
      · obtain ⟨hc, hd, ho⟩ := hF.scopes n hn mp hk hM
        exact IterExec.matchedB hi hc hd ho
    rw [hF.priv n hn mp hk] at hsim
    rcases hsim.ok_cases with ⟨e, he, hx⟩ | ⟨logs, hx, hl⟩
    · exact MatchedE.of_ub (.mapIterErr hk hs' hx) he
    · exact ⟨_, .mapScope hk hs' hx, scopeFinish_below hw st₁ hl⟩
  | .consumeSetupErr hk _, hn => absurd hk (hF.noConsume n hn _)
  | .consumeTasksErr hk _ _, hn => absurd hk (hF.noConsume n hn _)
  | .consumeScope hk _ _, hn => absurd hk (hF.noConsume n hn _)
  | .nestedSetupErr hk hs, hn => by
    have hL := hF.nodeLocal hn (hF.ne_entry_of_kind hn fun mp h => by rw [hk] at h; cases h)
    exact MatchedE.of_rel (.nestedSetupErr hk (by rw [← hL.nestedSetup]; exact hs)) rfl
  | .nestedInnerErr hk hs hr, hn => by
    have hL := hF.nodeLocal hn (hF.ne_entry_of_kind hn fun mp h => by rw [hk] at h; cases h)
    exact MatchedE.of_rel (.nestedInnerErr hk (by rw [← hL.nestedSetup]; exact hs) hr) rfl
  | .nested hk hs hr, hn => by
    have hL := hF.nodeLocal hn (hF.ne_entry_of_kind hn fun mp h => by rw [hk] at h; cases h)
    rw [hL.nestedFinish]
    exact MatchedE.of_rel (.nested hk (by rw [← hL.nestedSetup]; exact hs) hr) (Res.Rel.refl St.RelV.refl _)

theorem SeqExec.matchedB {ρ : Valuation} {ids : List NodeId} {st : St} {r' : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s' ids st r' → (∀ id ∈ ids, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) →
      MatchedE SeqRelV (SeqExec ctx ρ s ids st) r'
  | .nil, _ => MatchedE.of_rel .nil ⟨St.RelV.refl _, .nil⟩
  | .missing h, hids => by
    obtain ⟨h₁, h₂⟩ := hids _ (List.mem_cons_self ..)
    exact MatchedE.of_rel (.missing (by rw [← hF.node? h₁ h₂]; exact h)) rfl
  | .headErr h hn, hids => by
    obtain ⟨h₁, h₂⟩ := hids _ (List.mem_cons_self ..)
    obtain ⟨e₀, hr, hb⟩ := MatchedE.err_cases (NodeExec.matchedB hn (hF.mem_outer h h₂))
    exact ⟨_, .headErr (by rw [← hF.node? h₁ h₂]; exact h) hr, hb _⟩
  | .tailErr h hn hr, hids => by
    obtain ⟨h₁, h₂⟩ := hids _ (List.mem_cons_self ..)
    have hnode := (hF.node? h₁ h₂).symm.trans h
    rcases MatchedE.ok_cases (NodeExec.matchedB hn (hF.mem_outer h h₂)) with ⟨e, he, hx⟩ | ⟨a, hx, ha⟩
    · exact MatchedE.of_ub (.headErr hnode hx) he
    · obtain ⟨e₀, hr₀, hb₀⟩ := MatchedE.err_cases (SeqExec.matchedB hr fun id hid => hids id (List.mem_cons_of_mem _ hid))
      obtain ⟨hm, hv, hle⟩ := ha.afterNode st
      obtain ⟨r₁, hr₁, hrel⟩ := SeqExec.transferV hm hv hle hr₀
      obtain rfl := hrel.err_inv
      exact ⟨_, .tailErr hnode hx hr₁, hb₀ _⟩
  | .cons h hn hr, hids => by
    obtain ⟨h₁, h₂⟩ := hids _ (List.mem_cons_self ..)
    have hnode := (hF.node? h₁ h₂).symm.trans h
    rcases MatchedE.ok_cases (NodeExec.matchedB hn (hF.mem_outer h h₂)) with ⟨e, he, hx⟩ | ⟨a, hx, ha⟩
    · exact MatchedE.of_ub (.headErr hnode hx) he
    · obtain ⟨hm, hv, hle⟩ := ha.afterNode st
      rcases MatchedE.ok_cases (SeqExec.matchedB hr fun id hid => hids id (List.mem_cons_of_mem _ hid)) with
        ⟨e, he, hx'⟩ | ⟨y, hx', hrel⟩
      · obtain ⟨r₁, hr₁, hrel₁⟩ := SeqExec.transferV hm hv hle hx'
        obtain rfl := hrel₁.err_inv
        exact MatchedE.of_ub (.tailErr hnode hx hr₁) he
      · obtain ⟨r₁, hr₁, hrel₁⟩ := SeqExec.transferV hm hv hle hx'
        obtain ⟨z, rfl, hz⟩ := hrel₁.ok_inv
        exact MatchedE.of_rel (.cons hnode hx hr₁)
          ⟨hrel.1.trans hz.1, .cons ⟨rfl, ha.2.2⟩ (forall₂_trRel_trans hrel.2 hz.2)⟩

theorem OrderedExec.matchedB {ρ : Valuation} {all' direct' : List NodeId} {st : St} {r' : Res St} :
    OrderedExec ctx ρ s' all' direct' st r' → ∀ all direct,
      ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M all' all → direct'.Perm direct →
      bindViews ctx ρ s' (s'.viewOrder ctx direct') st = bindViews ctx ρ s (s.viewOrder ctx direct) st →
      (∀ id ∈ direct, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) →
      MatchedE St.RelV (OrderedExec ctx ρ s all direct st) r'
  | .prologueErr hp, _, _, _, _, hbv, _ => MatchedE.of_rel (.prologueErr (hbv ▸ hp)) rfl
  | .seqErr (order := o') ho hp hs, all, direct, hc, hpd, hbv, hd => by
    obtain ⟨o, ho₁, hfo⟩ := hF.transport ho hc hd
    have hp' := hbv.symm.trans hp
    have hfl : o.filter direct.contains = o'.filter direct'.contains :=
      hfo.trans (List.filter_congr fun x _ => by simp only [List.contains_eq_mem, hpd.mem_iff])
    have hids : ∀ id ∈ o'.filter direct'.contains, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M :=
      fun id hid => hd id (hpd.mem_iff.1 (by simpa using (List.mem_filter.1 hid).2))
    obtain ⟨e₀, hr, hb⟩ := MatchedE.err_cases (SeqExec.matchedB hs hids)
    exact ⟨_, .seqErr ho₁ hp' (hfl ▸ hr), hb _⟩
  | .hbRace (order := o') ho hp hs hf, all, direct, hc, hpd, hbv, hd => by
    obtain ⟨o, ho₁, hfo⟩ := hF.transport ho hc hd
    have hp' := hbv.symm.trans hp
    have hfl : o.filter direct.contains = o'.filter direct'.contains :=
      hfo.trans (List.filter_congr fun x _ => by simp only [List.contains_eq_mem, hpd.mem_iff])
    have hids : ∀ id ∈ o'.filter direct'.contains, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M :=
      fun id hid => hd id (hpd.mem_iff.1 (by simpa using (List.mem_filter.1 hid).2))
    rcases MatchedE.ok_cases (SeqExec.matchedB hs hids) with ⟨e, he, hx⟩ | ⟨⟨st₂, tr⟩, hx, hrel⟩
    · exact MatchedE.of_ub (.seqErr ho₁ hp' (hfl ▸ hx)) he
    · have hf' := hF.hbFree_false hrel.2 (fun x hx => by
        obtain ⟨h₁, n, h₂⟩ := SeqExec.trace_mem hs x hx
        have hn := hF.mem_outer h₂ (hids _ h₁).2
        obtain ⟨-, hid⟩ := State.node?_some h₂
        exact List.mem_map.2 ⟨n, hn, hid⟩) hf
      exact MatchedE.of_ub (.hbRace ho₁ hp' (hfl ▸ hx) hf') rfl
  | .run (order := o') ho hp hs hf, all, direct, hc, hpd, hbv, hd => by
    obtain ⟨o, ho₁, hfo⟩ := hF.transport ho hc hd
    have hp' := hbv.symm.trans hp
    have hfl : o.filter direct.contains = o'.filter direct'.contains :=
      hfo.trans (List.filter_congr fun x _ => by simp only [List.contains_eq_mem, hpd.mem_iff])
    have hids : ∀ id ∈ o'.filter direct'.contains, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M :=
      fun id hid => hd id (hpd.mem_iff.1 (by simpa using (List.mem_filter.1 hid).2))
    rcases MatchedE.ok_cases (SeqExec.matchedB hs hids) with ⟨e, he, hx⟩ | ⟨⟨st₂, tr⟩, hx, hrel⟩
    · exact MatchedE.of_ub (.seqErr ho₁ hp' (hfl ▸ hx)) he
    · cases hft : s.hbFree tr with
      | false => exact MatchedE.of_ub (.hbRace ho₁ hp' (hfl ▸ hx) hft) rfl
      | true => exact MatchedE.of_rel (.run ho₁ hp' (hfl ▸ hx) hft) hrel.1

theorem IterExec.matchedB {ρ : Valuation} {P : NodeId} {params : List Sym} {priv : List Name} {pre : St}
    {pts : List Idx} {r' : Res (List Log)} :
    IterExec ctx ρ s' P params priv pre pts r' →
      ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M (s'.scopeNodes P) (s.scopeNodes P) →
      s'.directChildren (some P) = s.directChildren (some P) →
      (∀ id ∈ s.directChildren (some P), id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M) →
      MatchedE (List.Forall₂ Log.Le) (IterExec ctx ρ s P params priv pre pts) r'
  | .nil, _, _, _ => MatchedE.of_rel .nil List.Forall₂.nil
  | .headErr hb, hc, hdir, ho => by
    have h₁ := OrderedExec.matchedB hb _ _ hc (by rw [hdir])
      (by rw [hdir, hF.viewOrder ho]; exact hF.bindViews (fun id hid => ho id (State.viewOrder_sub hid)) _) ho
    obtain ⟨e₀, hr, hbl⟩ := h₁.err_cases
    exact ⟨_, .headErr hr, hbl _⟩
  | .tailErr hb hr, hc, hdir, ho => by
    have h₁ := OrderedExec.matchedB hb _ _ hc (by rw [hdir])
      (by rw [hdir, hF.viewOrder ho]; exact hF.bindViews (fun id hid => ho id (State.viewOrder_sub hid)) _) ho
    rcases h₁.ok_cases with ⟨e, he, hx⟩ | ⟨a, hx, -⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · obtain ⟨e₀, hr₀, hb₀⟩ := MatchedE.err_cases (IterExec.matchedB hr hc hdir ho)
      exact ⟨_, .tailErr hx hr₀, hb₀ _⟩
  | .cons hb hr, hc, hdir, ho => by
    have h₁ := OrderedExec.matchedB hb _ _ hc (by rw [hdir])
      (by rw [hdir, hF.viewOrder ho]; exact hF.bindViews (fun id hid => ho id (State.viewOrder_sub hid)) _) ho
    rcases h₁.ok_cases with ⟨e, he, hx⟩ | ⟨a, hx, ha⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · rcases MatchedE.ok_cases (IterExec.matchedB hr hc hdir ho) with ⟨e, he, hx'⟩ | ⟨logs, hx', hl⟩
      · exact MatchedE.of_ub (.tailErr hx hx') he
      · exact MatchedE.of_rel (.cons hx hx') (List.Forall₂.cons (ha.2.2.dropNames _) hl)

end

theorem CompsExec.matchedB {ρ : Valuation} {m : Mem} :
    ∀ {comps' comps : List (List NodeId)} {r' : Res (List Log)}, CompsExec ctx ρ s' m comps' r' →
      List.Forall₂ (fun c' c => ScopeCorr (s'.scopeNodes M) (s.scopeNodes M) M c' c ∧
        (s'.topLevelOf c' = s.topLevelOf c ∨
          (s'.topLevelOf c').Perm (s.topLevelOf c) ∧ ViewFree ctx s (s.topLevelOf c)) ∧
        ∀ id ∈ s.topLevelOf c, id ∉ s.scopeNodes M ∧ id ∉ s'.scopeNodes M)
        comps' comps →
      MatchedE (List.Forall₂ Log.Le) (CompsExec ctx ρ s m comps) r'
  | _, _, _, .nil, .nil => MatchedE.of_rel .nil List.Forall₂.nil
  | _, _, _, .headErr hc, .cons ⟨hcorr, htop, hout⟩ _ => by
    have h₁ := OrderedExec.matchedB hF hw hbody hc _ _ hcorr (OuterFacts.topPerm htop) (hF.topViews htop hout _ _) hout
    obtain ⟨e₀, hr, hb⟩ := MatchedE.err_cases h₁
    exact ⟨_, .headErr hr, hb _⟩
  | _, _, _, .tailErr hc hr, .cons ⟨hcorr, htop, hout⟩ hrest => by
    have h₁ := OrderedExec.matchedB hF hw hbody hc _ _ hcorr (OuterFacts.topPerm htop) (hF.topViews htop hout _ _) hout
    rcases MatchedE.ok_cases h₁ with ⟨e, he, hx⟩ | ⟨a, hx, -⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · obtain ⟨e₀, hr₀, hb₀⟩ := MatchedE.err_cases (CompsExec.matchedB hr hrest)
      exact ⟨_, .tailErr hx hr₀, hb₀ _⟩
  | _, _, _, .cons hc hr, .cons ⟨hcorr, htop, hout⟩ hrest => by
    have h₁ := OrderedExec.matchedB hF hw hbody hc _ _ hcorr (OuterFacts.topPerm htop) (hF.topViews htop hout _ _) hout
    rcases MatchedE.ok_cases h₁ with ⟨e, he, hx⟩ | ⟨a, hx, ha⟩
    · exact MatchedE.of_ub (.headErr hx) he
    · rcases MatchedE.ok_cases (CompsExec.matchedB hr hrest) with ⟨e, he, hx'⟩ | ⟨logs, hx', hl⟩
      · exact MatchedE.of_ub (.tailErr hx hx') he
      · exact MatchedE.of_rel (.cons hx hx') (List.Forall₂.cons ha.2.2 hl)

/-- **The state with the body replaced is matched by the original state.** -/
theorem StateExec.matchedB {cst : CSt} {r' : Res (Ctl × CSt)} (h : StateExec ctx cst s' r') :
    MatchedE CtlRel (StateExec ctx cst s) r' := by
  cases h with
  | compsErr hc =>
    rw [hF.stateT] at hc
    obtain ⟨e₀, hr, hb⟩ := MatchedE.err_cases (CompsExec.matchedB hF hw hbody hc hF.comps)
    exact ⟨_, .compsErr hr, hb _⟩
  | run hc =>
    rw [hF.stateT] at hc
    rcases MatchedE.ok_cases (CompsExec.matchedB hF hw hbody hc hF.comps) with ⟨e, he, hx⟩ | ⟨logs, hx, hl⟩
    · exact MatchedE.of_ub (.compsErr hx) he
    · exact ⟨_, .run hx, stateFinish_below hw cst hl⟩

end Matched

/-! ## The check -/

/-- The context in which the check evaluates the context-dependent queries
(descriptor kinds, private transients): they read only the SDFG and the
prefix, not the interpretation. -/
def checkCtx (g : SDFG) : Ctx := { I := Interp.uninterpreted, g }

/-- What the leaf functions read about an edge (`EdgeSame`). -/
def edgeFp (ctx : Ctx) (s : State) (e : DEdge) :
    State.Endpoint × State.Endpoint × Bool × Bool × Option Name × Bool :=
  (s.srcEndpoint e, s.dstEndpoint e, s.isRouting e, s.isDynamicInput e, refSetTarget? ctx s e, ctx.isViewDef s e)

/-- Decision procedure for `ScopeCorr`. -/
def scopeCorr? (B' B : List NodeId) (M : NodeId) (all' all : List NodeId) : Bool :=
  decide all.Nodup && decide all'.Nodup &&
  (all.filter fun x => !B.contains x).isPerm (all'.filter fun x => !B'.contains x) &&
  (if all.contains M then B.all all.contains && B'.all all'.contains else all.all fun x => !B.contains x)

theorem scopeCorr?_sound {B' B : List NodeId} {M : NodeId} {all' all : List NodeId}
    (h : scopeCorr? B' B M all' all = true) : ScopeCorr B' B M all' all := by
  unfold scopeCorr? at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.isPerm_iff] at h
  obtain ⟨⟨⟨hnd, hnd'⟩, hp⟩, hc⟩ := h
  refine ⟨hnd, hnd', hp, fun hM => ?_, fun hM => ?_⟩
  · have : all.contains M = true := by simpa using hM
    rw [this] at hc
    simp only [↓reduceIte, Bool.and_eq_true, List.all_eq_true, List.contains_eq_mem, decide_eq_true_eq] at hc
    exact hc
  · have : all.contains M = false := by simpa using hM
    rw [this] at hc
    simp only [Bool.false_eq_true, ↓reduceIte, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not] at hc
    exact hc

/-- The body of `M` is entered only through `M`, every new body node has an
in-edge, old body nodes reach the outside only through nodes of the new body
(the shared exit), and the old body has a topological order. -/
def bodyShape? (s' s : State) (M : NodeId) : Bool :=
  let B := s.scopeNodes M
  let B' := s'.scopeNodes M
  !B.contains M && !B'.contains M &&
  s.edges.all (fun e => !B.contains e.dst || B.contains e.src || e.src == M) &&
  s'.edges.all (fun e => !B'.contains e.dst || B'.contains e.src || e.src == M) &&
  B'.all (fun x => s'.edges.any (·.dst == x)) &&
  s.edges.all (fun e => !B.contains e.src || B.contains e.dst || B'.contains e.src) &&
  decide B.Nodup && (s.topoOrder B).isSome

/-- Outer nodes and edges are seen the same way in both states. -/
def outerSame? (g : SDFG) (s' s : State) (M : NodeId) : Bool :=
  let ctx := checkCtx g
  let B := s.scopeNodes M
  let O := s.outerNodes B
  let Oids := O.map (·.id)
  O.all (fun n => (match n.kind with | .consumeEntry _ => false | _ => true) &&
    (n.id != M || match n.kind with | .mapEntry _ => true | _ => false)) &&
  (s.outerEdges B).all (fun e => edgeFp ctx s' e == edgeFp ctx s e) &&
  O.all (fun n => match n.kind with | .access _ => s'.viewBase? n == s.viewBase? n | _ => true) &&
  Oids.all (fun a => Oids.all fun b => !s.reaches a b || s'.reaches a b)

/-- Decision procedure for `ViewFree`. -/
def viewFree? (ctx : Ctx) (s : State) (ids : List NodeId) : Bool :=
  ids.all fun id => match s.node? id with
    | some n => match n.kind with
      | .access d => !ctx.isView d
      | _ => true
    | none => true

theorem viewFree?_sound {ctx : Ctx} {s : State} {ids : List NodeId} (h : viewFree? ctx s ids = true) :
    ViewFree ctx s ids := by
  intro id hid n hn d hk
  have := List.all_eq_true.1 h id hid
  rw [hn] at this
  dsimp only at this
  rw [hk] at this
  simpa using this

/-- The components and the enclosing map scopes correspond. -/
def scopesSame? (g : SDFG) (s' s : State) (M : NodeId) : Bool :=
  let ctx := checkCtx g
  let B := s.scopeNodes M
  let B' := s'.scopeNodes M
  let out := fun x => !B.contains x && !B'.contains x
  s'.components.length == s.components.length &&
  (s'.components.zip s.components).all (fun (c', c) =>
    scopeCorr? B' B M c' c &&
    (s'.topLevelOf c' == s.topLevelOf c ||
      (s'.topLevelOf c').isPerm (s.topLevelOf c) && viewFree? ctx s (s.topLevelOf c)) &&
    (s.topLevelOf c).all out) &&
  (s.outerNodes B).all (fun n => match n.kind with
    | .mapEntry _ => ctx.privateTransients s' n.id == ctx.privateTransients s n.id &&
        (n.id == M || (scopeCorr? B' B M (s'.scopeNodes n.id) (s.scopeNodes n.id) &&
          s'.directChildren (some n.id) == s.directChildren (some n.id) &&
          (s.directChildren (some n.id)).all out))
    | _ => true)

/-- **The check of the map-body congruence**: replacing the body of the map
with entry `M` turns `s` into `s'` without changing anything the rest of the
state depends on.  Together with equal outer nodes and edges
(`State.outerNodes`, `State.outerEdges`; `State.withBody` gives them by
construction) it yields `OuterFacts` (`sameOutside?_spec`). -/
def sameOutside? (g : SDFG) (s' s : State) (M : NodeId) : Bool :=
  bodyShape? s' s M && outerSame? g s' s M && scopesSame? g s' s M &&
  sameTransientMentions? g (.state s') (.state s)

theorem sameOutside?_spec {g : SDFG} {s' s : State} {M : NodeId} (h : sameOutside? g s' s M = true)
    (hN : s'.outerNodes (s'.scopeNodes M) = s.outerNodes (s.scopeNodes M))
    (hE : s'.outerEdges (s'.scopeNodes M) = s.outerEdges (s.scopeNodes M)) (I : Interp) :
    OuterFacts { I, g } s' s M := by
  unfold sameOutside? at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨hb, ho⟩, hs⟩, hm⟩ := h
  unfold bodyShape? at hb
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not,
    List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, List.any_eq_true] at hb
  obtain ⟨⟨⟨⟨⟨⟨⟨hM, hM'⟩, hsrc⟩, hsrc'⟩, hfed⟩, hexit⟩, hnd⟩, htopo⟩ := hb
  obtain ⟨Bo, hBo⟩ := Option.isSome_iff_exists.1 htopo
  unfold outerSame? at ho
  simp only [Bool.and_eq_true, List.all_eq_true] at ho
  obtain ⟨⟨⟨hkind, hedge⟩, hview⟩, hreach⟩ := ho
  unfold scopesSame? at hs
  simp only [Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hs
  obtain ⟨⟨hlen, hcomps⟩, hscopes⟩ := hs
  refine
    { nodes := hN
      edges := hE
      entry := hM
      entry' := hM'
      src := fun e he hd => ?_
      src' := fun e he hd => ?_
      fed' := fun x hx => ?_
      exit := fun e he hs hd => ?_
      bodyNodup := hnd
      bodyOrder := ⟨Bo, State.topoOrder_spec hnd hBo⟩
      noConsume := fun n hn c hc => ?_
      entryKind := fun n hn hid => ?_
      edge := fun e he => ?_
      view := fun n hn d hk => ?_
      reach := fun a b ha hb hr => ?_
      comps := ?_
      priv := fun n hn mp hk => ?_
      stateT := ?_
      scopes := fun n hn mp hk hnM => ?_ }
  · rcases hsrc e he with (h | h) | h
    · exact absurd hd h
    · exact .inl h
    · exact .inr h
  · rcases hsrc' e he with (h | h) | h
    · exact absurd hd h
    · exact .inl h
    · exact .inr h
  · obtain ⟨e, he, hd⟩ := hfed x hx
    exact ⟨e, he, hd⟩
  · rcases hexit e he with (h | h) | h
    · exact absurd hs h
    · exact absurd h hd
    · exact h
  · have := hkind n hn
    rw [hc] at this
    simp at this
  · obtain ⟨-, h₂⟩ := hkind n hn
    rw [hid] at h₂
    simp only [bne_self_eq_false, Bool.false_or] at h₂
    split at h₂
    · rename_i mp hmp
      exact ⟨mp, hmp⟩
    · cases h₂
  · have := hedge e he
    simp only [edgeFp, beq_iff_eq, Prod.mk.injEq] at this
    obtain ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩ := this
    exact ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩
  · have := hview n hn
    rw [hk] at this
    simpa using this
  · have := hreach a ha b hb
    rw [hr] at this
    simpa using this
  · refine List.forall₂_iff_zip.2 ⟨hlen, fun {c' c} hz => ?_⟩
    have := hcomps (c', c) hz
    simp only [Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at this
    obtain ⟨⟨hc, ht⟩, hout⟩ := this
    refine ⟨scopeCorr?_sound hc, ?_, fun id hid => hout id hid⟩
    simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, List.isPerm_iff] at ht
    rcases ht with ht | ⟨hp, hv⟩
    · exact .inl ht
    · exact .inr ⟨hp, viewFree?_sound hv⟩
  · have := hscopes n hn
    rw [hk] at this
    simp only [Bool.and_eq_true, beq_iff_eq] at this
    exact this.1
  · unfold Ctx.stateTransients
    refine List.filterMap_congr fun t _ => ?_
    cases hdt : ({ I, g } : Ctx).desc? t with
    | none => rfl
    | some d =>
      simp only
      split
      · rename_i hd
        have hmem : (t, d) ∈ g.arrays := by
          simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hdt
          obtain ⟨⟨t', d'⟩, hf, rfl⟩ := hdt
          have := List.find?_some hf
          simp only [decide_eq_true_eq] at this
          subst this
          exact List.mem_of_find?_eq_some hf
        have hc := sameTransientMentions?_count hm hmem hd
        simp only [Block.countMentions] at hc
        rw [hc]
      · rfl
  · have := hscopes n hn
    rw [hk] at this
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, hnM, false_or, List.all_eq_true,
      Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at this
    obtain ⟨-, ⟨hc, hd⟩, hout⟩ := this
    exact ⟨scopeCorr?_sound hc, hd, fun id hid => hout id hid⟩

theorem sameOutside?_mentions {g : SDFG} {s' s : State} {M : NodeId} (h : sameOutside? g s' s M = true) :
    sameTransientMentions? g (.state s') (.state s) = true := by
  unfold sameOutside? at h
  simp only [Bool.and_eq_true] at h
  exact h.2

/-! ## Congruence -/

/-- The block matching `Congr.block` needs, from the body matching. -/
theorem scopeBody_blockMatchedE {g : SDFG} {s' s : State} {M : NodeId} (hchk : sameOutside? g s' s M = true)
    (hN : s'.outerNodes (s'.scopeNodes M) = s.outerNodes (s.scopeNodes M))
    (hE : s'.outerEdges (s'.scopeNodes M) = s.outerEdges (s.scopeNodes M)) {I : Interp} (hI : I.Lawful)
    (hbody : BodyMatchedE { I, g } s' s M) :
    ∀ cst r', BlockExec { I, g } cst (.state s') r' →
      MatchedE CtlRel (BlockExec { I, g } cst (.state s)) r' := by
  intro cst r' h
  cases h with
  | state h =>
    exact (StateExec.matchedB (sameOutside?_spec hchk hN hE I) hI hbody h).mono (fun _ _ h => h) fun _ h => .state h

/-- **Map-body congruence** (R1.5, R1 design §2.3): replacing the body of the
map with entry `M` in the state at path `p`, index `i` of `g` refines `g` if

* `sameOutside? g s' s M`: everything outside the bodies is seen the same way
  by both states (nodes, edges, memlet-tree endpoints of the edges they use,
  view targets, reachability, components, enclosing map scopes and their
  private transients, mention counts of `Scope` transients), the new body is
  entered only through `M` and fed from it, and both bodies leave only
  through the shared exit;
* the nodes and edges outside the bodies are equal (`State.withBody` gives
  this by construction, `State.withBody_outer`);
* for every lawful interpretation, every run of one iteration of the new body
  is matched by a run of an iteration of the old body from the same state,
  with the same outcome up to the log outside the map's private transients
  (`BodyMatchedE`, which is all an atom has to prove);
* `g` is `Supported`. -/
theorem Congr.scopeBody {g : SDFG} {p : List PathStep} {i : BlockId} {s s' : State} {M : NodeId}
    (hat : g.top.blockAt? p i = some (.state s)) (hchk : sameOutside? g s' s M = true)
    (hN : s'.outerNodes (s'.scopeNodes M) = s.outerNodes (s.scopeNodes M))
    (hE : s'.outerEdges (s'.scopeNodes M) = s.outerEdges (s.scopeNodes M))
    (hbody : ∀ I : Interp, I.Lawful → BodyMatchedE { I, g } s' s M)
    (hsup : Supported g.prog) : Refines (g.setBlockAt p i (.state s')).prog g.prog :=
  Congr.block hat (sameOutside?_mentions hchk)
    (fun I hI => scopeBody_blockMatchedE hchk hN hE hI (hbody I hI)) hsup

/-- **Map-body congruence, both directions**: bodies that match each other's runs
(with the check in both directions) give equivalent SDFGs. -/
theorem Congr.scopeBodyEquiv {g : SDFG} {p : List PathStep} {i : BlockId} {s s' : State} {M : NodeId}
    (hat : g.top.blockAt? p i = some (.state s)) (hchk : sameOutside? g s' s M = true)
    (hchk' : sameOutside? g s s' M = true)
    (hN : s'.outerNodes (s'.scopeNodes M) = s.outerNodes (s.scopeNodes M))
    (hE : s'.outerEdges (s'.scopeNodes M) = s.outerEdges (s.scopeNodes M))
    (hbody : ∀ I : Interp, I.Lawful → BodyMatchedE { I, g } s' s M)
    (hbody' : ∀ I : Interp, I.Lawful → BodyMatchedE { I, g } s s' M)
    (hsup : Supported g.prog) : Equiv (g.setBlockAt p i (.state s')).prog g.prog :=
  Congr.blockEquiv hat (sameOutside?_mentions hchk)
    (fun I hI => scopeBody_blockMatchedE hchk hN hE hI (hbody I hI))
    (fun I hI => scopeBody_blockMatchedE hchk' hN.symm hE.symm hI (hbody' I hI)) hsup

/-! ## Building the new state -/

/-- The state `s` with the body of the map `M` replaced by the nodes `bn` and
the edges `be` (the edges ending in the body: from `M`, inside it, and into
the exit). -/
def State.withBody (s : State) (M : NodeId) (bn : List DNode) (be : List DEdge) : State :=
  { s with nodes := s.outerNodes (s.scopeNodes M) ++ bn, edges := s.outerEdges (s.scopeNodes M) ++ be }

/-- Do the new nodes form exactly the new body, and do the old outer nodes and
edges stay outside it? -/
def State.bodyFits? (s : State) (M : NodeId) (bn : List DNode) (be : List DEdge) : Bool :=
  let B' := (s.withBody M bn be).scopeNodes M
  (s.outerNodes (s.scopeNodes M)).all (fun n => !B'.contains n.id) && bn.all (fun n => B'.contains n.id) &&
  (s.outerEdges (s.scopeNodes M)).all (fun e => !B'.contains e.dst) && be.all (fun e => B'.contains e.dst)

theorem State.withBody_outer {s : State} {M : NodeId} {bn : List DNode} {be : List DEdge}
    (h : s.bodyFits? M bn be = true) :
    (s.withBody M bn be).outerNodes ((s.withBody M bn be).scopeNodes M) = s.outerNodes (s.scopeNodes M) ∧
      (s.withBody M bn be).outerEdges ((s.withBody M bn be).scopeNodes M) = s.outerEdges (s.scopeNodes M) := by
  unfold bodyFits? at h
  simp only [Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩ := h
  constructor
  · show List.filter _ (s.outerNodes (s.scopeNodes M) ++ bn) = _
    rw [List.filter_append, List.filter_eq_self.2 h₁, List.filter_eq_nil_iff.2 fun n hn => by simpa using h₂ n hn,
      List.append_nil]
  · show List.filter _ (s.outerEdges (s.scopeNodes M) ++ be) = _
    rw [List.filter_append, List.filter_eq_self.2 h₃, List.filter_eq_nil_iff.2 fun e he => by simpa using h₄ e he,
      List.append_nil]

/-- **Map-body congruence for `State.withBody`**: the outer-equality
hypotheses of `Congr.scopeBody` become the check `bodyFits?`. -/
theorem Congr.mapBody {g : SDFG} {p : List PathStep} {i : BlockId} {s : State} {M : NodeId} {bn : List DNode}
    {be : List DEdge} (hat : g.top.blockAt? p i = some (.state s)) (hfit : s.bodyFits? M bn be = true)
    (hchk : sameOutside? g (s.withBody M bn be) s M = true)
    (hbody : ∀ I : Interp, I.Lawful → BodyMatchedE { I, g } (s.withBody M bn be) s M)
    (hsup : Supported g.prog) : Refines (g.setBlockAt p i (.state (s.withBody M bn be))).prog g.prog :=
  Congr.scopeBody hat hchk (State.withBody_outer hfit).1 (State.withBody_outer hfit).2 hbody hsup

/-- An unchanged body matches itself (the trivial instance). -/
theorem BodyMatchedE.refl (ctx : Ctx) (s : State) (M : NodeId) : BodyMatchedE ctx s s M :=
  fun _ _ _ h => MatchedE.of_rel h (by
    rename_i r'
    cases r' with
    | ok a => exact Log.Le.refl _
    | err e => rfl)

end Dace
