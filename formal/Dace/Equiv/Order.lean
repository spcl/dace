-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Frame
import Dace.SDFG.ScopeSpec

/-!
# Order independence of sequential dataflow (race-free ⇒ deterministic)

On the fragment `SeqPure` (`Dace.SDFG.Frame`) and for a lawful WCR
semantics:

* **adjacent swap** (`SeqExec.swap`): two neighbouring nodes of a
  sequential execution whose deltas do not conflict can run in the other
  order, with the same deltas, the same memory and code-edge values, and
  logs equal up to commuting reorderings (`St.Eqv`: `Log.Le` both ways);
* **reordering** (`SeqExec.reorder`): a normal sequential execution whose
  trace is free of happens-before races (`State.hbFree`) can be replayed in
  any other order of its nodes that is consistent with reachability; the
  trace is permuted, the result `St.Eqv`-equal.  The proof moves the first
  node of the target order to the front by adjacent swaps (`SeqExec.bubble`)
  with nodes it is unordered with (so `hbFree` makes their deltas free);
* **order independence** (`OrderedExec.order_indep`): a normal outcome of a
  scope body or component (some topological order) gives a normal run along
  *every* topological order of the scope (`State.isTopologicalOrder`),
  race-free and with an `St.Eqv`-equal result, provided paths between the
  scope's nodes stay inside the scope (`State.Convex`; acyclic well-formed
  scopes and weakly connected components are convex);
* **components** (`CompsExec.sequential`, `CompsExec.merge_perm`): the merge of
  the component logs of a state equals running the components one after the
  other (each from the memory left by the previous ones), and the merge is
  independent of the order of the components.

The full determinism theorem (*every* derivation, including the orders
chosen inside nested map scopes, is normal and `St.Eqv`-equal) needs a
mutual induction over the derivation whose per-node step is the frame
property plus uniqueness of node outcomes; it is not stated here.
-/

set_option autoImplicit false

namespace Dace

/-! ## Equal results -/

/-- Two sequential states with equal memory and code-edge values whose logs
are equal up to commuting reorderings. -/
def St.Eqv (a b : St) : Prop := a.mem = b.mem ∧ a.vals = b.vals ∧ Log.Le a.log b.log ∧ Log.Le b.log a.log

namespace St.Eqv

theorem refl (a : St) : St.Eqv a a := ⟨rfl, rfl, Log.Le.refl _, Log.Le.refl _⟩

theorem symm {a b : St} (h : St.Eqv a b) : St.Eqv b a := ⟨h.1.symm, h.2.1.symm, h.2.2.2, h.2.2.1⟩

theorem trans {a b c : St} (h₁ : St.Eqv a b) (h₂ : St.Eqv b c) : St.Eqv a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.1.trans h₂.2.2.1, h₂.2.2.2.trans h₁.2.2.2⟩

theorem rel {a b : St} (h : St.Eqv a b) : St.Rel a b :=
  ⟨h.1, h.2.2.1, fun k => by simp only [St.val?, h.2.1]⟩

end St.Eqv

theorem St.afterNode_step' (w : WcrSem) (st : St) (δ : Log) :
    st.afterNode (st.forNode.step w δ) = st.step w δ := by
  rw [St.afterNode_step, Log.empty_seq]

/-! ## Traces -/

theorem SeqExec.trace_ids {ctx : Ctx} {ρ : Valuation} {s : State} :
    ∀ {ids : List NodeId} {st a : St} {tr : List (NodeId × Log)},
      SeqExec ctx ρ s ids st (.ok (a, tr)) → tr.map Prod.fst = ids
  | [], _, _, _, h => by cases h; rfl
  | _ :: _, _, _, _, h => by
    cases h with
    | cons _ _ hr => simp [SeqExec.trace_ids hr]

namespace State

theorem hbConflict_comm (s : State) (x y : NodeId × Log) : s.hbConflict x y = s.hbConflict y x := by
  simp only [hbConflict, Log.conflicts_comm x.2 y.2]
  cases Log.conflicts y.2 x.2 <;> simp [Bool.and_comm]

theorem hbFree_iff (s : State) (tr : List (NodeId × Log)) :
    s.hbFree tr = true ↔ tr.Pairwise (fun x y => s.hbConflict x y = false) := by
  induction tr with
  | nil => simp [hbFree]
  | cons x xs ih =>
    simp only [hbFree, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, ih,
      List.pairwise_cons]

theorem hbFree_perm (s : State) {tr tr' : List (NodeId × Log)} (hp : tr.Perm tr') :
    s.hbFree tr = s.hbFree tr' := by
  rw [Bool.eq_iff_iff, hbFree_iff, hbFree_iff]
  exact hp.pairwise_iff fun {a b} (h : s.hbConflict a b = false) => (s.hbConflict_comm b a).trans h

/-- Unordered entries of a race-free trace have non-conflicting deltas. -/
theorem conflicts_of_hbFree (s : State) {x y : NodeId × Log} (hxy : s.hbConflict x y = false)
    (h₁ : s.reaches x.1 y.1 = false) (h₂ : s.reaches y.1 x.1 = false) : Log.conflicts x.2 y.2 = false := by
  simpa [hbConflict, h₁, h₂] using hxy

end State

/-! ## Adjacent swap -/

/-- Applying a log that neither writes nor updates the reads of another
keeps agreement on those reads. -/
theorem Mem.Agree.of_free {a b : Log} (ha : a.Plain) (hf : Log.HalfFree a b) (w : WcrSem) (m : Mem) :
    Mem.Agree b.reads (a.apply w m) m := by
  refine ⟨ha.apply_aliases w m, fun l hl => Log.apply_cells_untouched a w m ?_ ?_⟩
  · exact fun hw => (hf.write l hw).2.1 hl
  · exact fun hc => hf.wcrRead l hc hl

section Swap

variable {ctx : Ctx} {ρ : Valuation} {s : State} (h : SeqPure ctx s) (hl : ctx.I.wcr.Lawful)
include h

include hl in
/-- **Adjacent swap**: two neighbouring nodes with non-conflicting deltas
can be run in the other order, with the same deltas and an `St.Eqv`-equal
result. -/
theorem SeqExec.swap {x y : NodeId} {rest : List NodeId} {st a : St} {δx δy : Log} {tr : List (NodeId × Log)}
    (hx : SeqExec ctx ρ s (x :: y :: rest) st (.ok (a, (x, δx) :: (y, δy) :: tr)))
    (hc : Log.conflicts δx δy = false) :
    ∃ a', SeqExec ctx ρ s (y :: x :: rest) st (.ok (a', (y, δy) :: (x, δx) :: tr)) ∧ St.Eqv a' a := by
  cases hx with
  | cons hnx hnodex hr₁ =>
  cases hr₁ with
  | cons hny hnodey hr₂ =>
  obtain ⟨dx, hpx, rfl, hdx⟩ := NodeExec.frame h (List.mem_of_find?_eq_some hnx) hnodex _ rfl
  rw [St.afterNode_step'] at hnodey hr₂
  obtain ⟨dy, hpy, rfl, hdy⟩ := NodeExec.frame h (List.mem_of_find?_eq_some hny) hnodey _ rfl
  rw [St.afterNode_step'] at hr₂
  obtain ⟨dr, -, rfl, hdr⟩ := SeqExec.frame h hr₂ _ _ rfl
  replace hc : Log.conflicts dx dy = false := by simpa [St.step, St.forNode, Log.empty_seq] using hc
  have hfree : Log.Free dx dy := (Log.conflicts_eq_false_iff _ _).1 hc
  -- `y` first, from the original state
  have hny₂ := hdy st.forNode (Mem.Agree.of_free hpx hfree.1 _ st.mem).symm rfl
  -- then `x`
  have hnx₂ := hdx (st.step ctx.I.wcr dy).forNode (Mem.Agree.of_free hpy hfree.2 _ st.mem) rfl
  -- the memories after both nodes coincide
  have hcomm : Log.Le (dy.seq dx) (dx.seq dy) :=
    Log.Le.seq_comm ((Log.conflicts_comm _ _).trans hc) (fun l hl => by
      simp [Log.pushedQueues, hpy.queues] at hl)
  have hmem : ((st.step ctx.I.wcr dy).step ctx.I.wcr dx).mem = ((st.step ctx.I.wcr dx).step ctx.I.wcr dy).mem := by
    simp only [St.step_step, St.step_mem]
    exact hcomm.apply_eq _ hl _
  have hr₃ := hdr ((st.step ctx.I.wcr dy).step ctx.I.wcr dx) ⟨by rw [hmem], fun l _ => by rw [hmem]⟩ rfl
  have hr₄ : SeqExec ctx ρ s rest ((st.step ctx.I.wcr dy).afterNode ((st.step ctx.I.wcr dy).forNode.step ctx.I.wcr dx))
      (.ok (((st.step ctx.I.wcr dy).step ctx.I.wcr dx).step ctx.I.wcr dr, tr)) := by
    rw [St.afterNode_step']; exact hr₃
  have hr₅ : SeqExec ctx ρ s (x :: rest) (st.afterNode (st.forNode.step ctx.I.wcr dy))
      (.ok (((st.step ctx.I.wcr dy).step ctx.I.wcr dx).step ctx.I.wcr dr,
        (x, (st.forNode.step ctx.I.wcr dx).log) :: tr)) := by
    rw [St.afterNode_step']
    have := SeqExec.cons hnx hnx₂ hr₄
    exact this
  have hr₆ := SeqExec.cons hny hny₂ hr₅
  refine ⟨_, hr₆, ?_⟩
  refine ⟨by rw [St.step_mem, hmem]; rfl, rfl, ?_, ?_⟩
  · simp only [St.step_log]
    rw [Log.seq_assoc st.log dy dx, Log.seq_assoc st.log dx dy]
    exact Log.Le.seq (Log.Le.seq (Log.Le.refl _) hcomm) (Log.Le.refl _)
  · simp only [St.step_log]
    rw [Log.seq_assoc st.log dy dx, Log.seq_assoc st.log dx dy]
    exact Log.Le.seq (Log.Le.seq (Log.Le.refl _) (Log.Le.seq_comm hc (fun l hl => by
      simp [Log.pushedQueues, hpx.queues] at hl))) (Log.Le.refl _)

include hl in
/-- Move a node to the front past nodes whose deltas do not conflict with
its own. -/
theorem SeqExec.bubble : ∀ (pre : List (NodeId × Log)) {z : NodeId} {δz : Log} {post : List NodeId}
    {trpost : List (NodeId × Log)} {st a : St},
    SeqExec ctx ρ s (pre.map Prod.fst ++ z :: post) st (.ok (a, pre ++ (z, δz) :: trpost)) →
    (∀ p ∈ pre, Log.conflicts p.2 δz = false) →
    ∃ a' tr', SeqExec ctx ρ s (z :: (pre.map Prod.fst ++ post)) st (.ok (a', (z, δz) :: tr')) ∧
      St.Eqv a' a ∧ tr'.Perm (pre ++ trpost)
  | [], _, _, _, _, _, a, hx, _ => ⟨a, _, hx, St.Eqv.refl _, List.Perm.refl _⟩
  | (p, δp) :: pre', z, δz, post, trpost, st, a, hx, hc => by
    cases hx with
    | cons hnp hnodep hrest =>
      obtain ⟨a', tr', hr', he, hp⟩ :=
        SeqExec.bubble pre' hrest (fun q hq => hc q (List.mem_cons_of_mem _ hq))
      have h2 := SeqExec.cons hnp hnodep hr'
      obtain ⟨a'', h3, he'⟩ := SeqExec.swap h hl h2 (hc _ List.mem_cons_self)
      exact ⟨a'', _ :: tr', h3, he'.trans he, hp.cons _⟩

include hl in
/-- **Reordering**: a normal, race-free sequential execution can be
replayed along any reordering of its nodes that, like its own order, never
puts a node after a node it reaches. -/
theorem SeqExec.reorder : ∀ (f₂ : List NodeId) {st a : St} {tr : List (NodeId × Log)},
    SeqExec ctx ρ s (tr.map Prod.fst) st (.ok (a, tr)) → s.hbFree tr = true → (tr.map Prod.fst).Nodup →
    (tr.map Prod.fst).Pairwise (fun x y => s.reaches y x = false) →
    f₂.Pairwise (fun x y => s.reaches y x = false) → (tr.map Prod.fst).Perm f₂ →
    ∃ a' tr', SeqExec ctx ρ s f₂ st (.ok (a', tr')) ∧ St.Eqv a' a ∧ tr'.Perm tr
  | [], st, a, tr, hx, _, _, _, _, hp => by
    have : tr = [] := List.map_eq_nil_iff.1 (List.Perm.eq_nil hp)
    subst this
    exact ⟨a, [], hx, St.Eqv.refl _, List.Perm.refl _⟩
  | z :: f₂', st, a, tr, hx, hf, hnd, hpw, hpw₂, hp => by
    have hz : z ∈ tr.map Prod.fst := hp.symm.subset List.mem_cons_self
    obtain ⟨⟨z', δz⟩, hzmem, rfl⟩ := List.mem_map.1 hz
    obtain ⟨pre, post, rfl⟩ := List.append_of_mem hzmem
    simp only [List.map_append, List.map_cons] at hx hnd hpw hp
    have hpwtr := (State.hbFree_iff s _).1 hf
    have hc : ∀ p ∈ pre, Log.conflicts p.2 δz = false := by
      intro p hpm
      have hpz : p.1 ∈ pre.map Prod.fst := List.mem_map_of_mem hpm
      have h1 : s.hbConflict p (z', δz) = false :=
        (List.pairwise_append.1 hpwtr).2.2 p hpm _ List.mem_cons_self
      have h2 : s.reaches z' p.1 = false :=
        (List.pairwise_append.1 hpw).2.2 _ hpz _ List.mem_cons_self
      have hne : p.1 ≠ z' := fun he => by
        have := (List.nodup_append.1 hnd).2.2 _ hpz _ List.mem_cons_self
        exact this he
      have hin : p.1 ∈ f₂' := by
        have := hp.subset (List.mem_append_left _ hpz)
        rcases List.mem_cons.1 this with he | he
        · exact absurd he hne
        · exact he
      have h3 : s.reaches p.1 z' = false := (List.pairwise_cons.1 hpw₂).1 _ hin
      exact State.conflicts_of_hbFree s h1 h3 h2
    obtain ⟨a₁, tr₁, hb, he₁, hp₁⟩ := SeqExec.bubble h hl pre hx hc
    have hperm₁ : ((z', δz) :: tr₁).Perm (pre ++ (z', δz) :: post) :=
      (hp₁.cons _).trans List.perm_middle.symm
    have hf₁ : s.hbFree tr₁ = true := by
      have := (State.hbFree_perm s hperm₁).trans hf
      simp only [State.hbFree, Bool.and_eq_true] at this
      exact this.2
    cases hb with
    | cons hnz hnodez hrest =>
      have hids := SeqExec.trace_ids hrest
      have hsub : List.Sublist (pre.map Prod.fst ++ post.map Prod.fst) (pre.map Prod.fst ++ z' :: post.map Prod.fst) :=
        List.Sublist.append_left (List.sublist_cons_self _ _) _
      have hp₂ : (tr₁.map Prod.fst).Perm f₂' := by
        rw [hids]
        exact List.Perm.cons_inv (List.perm_middle.symm.trans hp)
      rw [← hids] at hrest hsub
      obtain ⟨a₂, tr₂, hr₂, he₂, hp₂'⟩ := SeqExec.reorder f₂' hrest hf₁ (hnd.sublist hsub) (hpw.sublist hsub)
        (List.pairwise_cons.1 hpw₂).2 hp₂
      have hc' := SeqExec.cons hnz hnodez hr₂
      exact ⟨a₂, _, hc', he₂.trans he₁, (hp₂'.cons _).trans hperm₁⟩

end Swap

/-! ## Topological orders and reachability -/

/-- Paths between nodes of `U` stay inside `U`. -/
def State.Convex (s : State) (U : List NodeId) : Prop :=
  ∀ x ∈ U, ∀ y ∈ U, s.reaches x y = true → Relation.TransGen (fun a b => s.Edge a b ∧ a ∈ U ∧ b ∈ U) x y

/-- In a topological order of a convex node set, no node is reached from a
later one. -/
theorem State.topo_pairwise {s : State} {all o : List NodeId} (hnd : all.Nodup) (hconv : s.Convex all)
    (ho : s.isTopologicalOrder all o = true) : o.Pairwise (fun x y => s.reaches y x = false) := by
  have hperm := s.isTopologicalOrder_perm hnd ho
  simp only [State.isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq] at ho
  obtain ⟨⟨⟨_, _⟩, hond⟩, hresp⟩ := ho
  have key : ∀ a b, Relation.TransGen (fun a b => s.Edge a b ∧ a ∈ all ∧ b ∈ all) a b →
      ∃ i j, o.idxOf? a = some i ∧ o.idxOf? b = some j ∧ i < j := by
    have hedge : ∀ x y, (s.Edge x y ∧ x ∈ all ∧ y ∈ all) →
        ∃ i j, o.idxOf? x = some i ∧ o.idxOf? y = some j ∧ i < j := by
      rintro x y ⟨⟨e, he, rfl, rfl⟩, hx, hy⟩
      obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr (hperm.symm.subset hx))
      obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr (hperm.symm.subset hy))
      have := List.all_eq_true.mp hresp e he
      simp only [hi, hj, decide_eq_true_eq] at this
      exact ⟨i, j, hi, hj, this⟩
    intro a b hab
    induction hab with
    | single hab => exact hedge _ _ hab
    | tail _ hbc ih =>
      obtain ⟨i, j, hi, hj, hij⟩ := ih
      obtain ⟨j', k, hj', hk, hjk⟩ := hedge _ _ hbc
      rw [hj] at hj'
      cases hj'
      exact ⟨i, k, hi, hk, by omega⟩
  have hidx : ∀ i (hi : i < o.length), o.idxOf? o[i] = some i := fun i hi =>
    List.idxOf?_eq_some_iff.2 ⟨hi, rfl, fun j hj heq => by
      have := (hond.getElem_inj_iff).1 heq; omega⟩
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  cases hr : s.reaches o[j] o[i] with
  | false => rfl
  | true =>
    exfalso
    obtain ⟨i', j', hi', hj', hlt⟩ :=
      key _ _ (hconv _ (hperm.subset (List.getElem_mem hj)) _ (hperm.subset (List.getElem_mem hi)) hr)
    rw [hidx j hj] at hi'
    rw [hidx i hi] at hj'
    cases hi'; cases hj'
    omega

/-- Weakly connected components are convex: a path from a node of a
component stays in it. -/
theorem State.components_convex {s : State} {c : List NodeId} (hc : c ∈ s.components) : s.Convex c := by
  intro x hx y _ hr
  have hpath := State.reaches_iff.1 hr
  have key : ∀ a b, Relation.TransGen s.Edge a b → a ∈ c →
      Relation.TransGen (fun a b => s.Edge a b ∧ a ∈ c ∧ b ∈ c) a b ∧ b ∈ c := by
    intro a b hab
    induction hab with
    | single hab =>
      intro ha
      obtain ⟨e, he, rfl, rfl⟩ := hab
      have hb := (State.components_edge hc he).1 ha
      exact ⟨.single ⟨⟨e, he, rfl, rfl⟩, ha, hb⟩, hb⟩
    | tail _ hbc ih =>
      intro ha
      obtain ⟨ih₁, hb⟩ := ih ha
      obtain ⟨e, he, hs, hd⟩ := hbc
      have hd' := (State.components_edge hc he).1 (hs ▸ hb)
      rw [hd] at hd'
      exact ⟨.tail ih₁ ⟨⟨e, he, hs, hd⟩, hb, hd'⟩, hd'⟩
  exact (key _ _ hpath hx).1

theorem State.components_nodup {s : State} {c : List NodeId} (hc : c ∈ s.components) : c.Nodup :=
  (List.nodup_flatten.mp (State.components_flatten_nodup (s := s))).1 c hc

/-! ## Order independence -/

section Indep

variable {ctx : Ctx} {ρ : Valuation} {s : State} (h : SeqPure ctx s) (hl : ctx.I.wcr.Lawful)
include h hl

/-- **Order independence (race-free ⇒ deterministic)**: a normal outcome of
a scope body or component gives, for *every* topological order of the
scope's nodes, a normal race-free sequential run along that order with an
`St.Eqv`-equal result (equal memory and code-edge values, logs equal up to
commuting reorderings). -/
theorem OrderedExec.order_indep {all direct : List NodeId} {st a : St} (hnd : all.Nodup)
    (hconv : s.Convex all) (hx : OrderedExec ctx ρ s all direct st (.ok a)) {o : List NodeId}
    (ho : s.isTopologicalOrder all o = true) :
    ∃ a' tr', SeqExec ctx ρ s (o.filter direct.contains) st (.ok (a', tr')) ∧ s.hbFree tr' = true ∧
      St.Eqv a' a := by
  cases hx with
  | run ho₁ hp hs hf =>
    rename_i o₁ st₁ tr
    rw [bindViews_eq ρ h] at hp
    simp only [Res.ok_inj] at hp
    subst hp
    have hids := SeqExec.trace_ids hs
    have hnd₁ : (o₁.filter direct.contains).Nodup := by
      simp only [State.isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq] at ho₁
      exact ho₁.1.2.filter _
    have hpw₁ := (State.topo_pairwise hnd hconv ho₁).filter (p := direct.contains)
    have hpw₂ := (State.topo_pairwise hnd hconv ho).filter (p := direct.contains)
    have hperm : (o₁.filter direct.contains).Perm (o.filter direct.contains) :=
      ((s.isTopologicalOrder_perm hnd ho₁).trans (s.isTopologicalOrder_perm hnd ho).symm).filter _
    rw [← hids] at hs hnd₁ hpw₁ hperm
    obtain ⟨a', tr', hr, he, hp'⟩ := SeqExec.reorder h hl _ hs hf hnd₁ hpw₁ hpw₂ hperm
    exact ⟨a', tr', hr, (State.hbFree_perm s hp').trans hf, he⟩

/-- The run along any other topological order is an `OrderedExec` outcome. -/
theorem OrderedExec.order_indep_exec {all direct : List NodeId} {st a : St} (hnd : all.Nodup)
    (hconv : s.Convex all) (hx : OrderedExec ctx ρ s all direct st (.ok a)) {o : List NodeId}
    (ho : s.isTopologicalOrder all o = true) :
    ∃ a', OrderedExec ctx ρ s all direct st (.ok a') ∧ St.Eqv a' a ∧
      ∃ tr', SeqExec ctx ρ s (o.filter direct.contains) st (.ok (a', tr')) := by
  obtain ⟨a', tr', hr, hf, he⟩ := OrderedExec.order_indep h hl hnd hconv hx ho
  exact ⟨a', .run ho (bindViews_eq ρ h _ _) hr hf, he, tr', hr⟩

/-- **Components (M5a)**: the direct nodes of a state component, run in any
topological order of the component (e.g. the schedule tree's sibling order),
give the same memory as the SDFG semantics' run of the component. -/
theorem OrderedExec.order_indep_component {c : List NodeId} (hc : c ∈ s.components) {st a : St}
    (hx : OrderedExec ctx ρ s c (s.topLevelOf c) st (.ok a)) {o : List NodeId}
    (ho : s.isTopologicalOrder c o = true) :
    ∃ a' tr', SeqExec ctx ρ s (o.filter (s.topLevelOf c).contains) st (.ok (a', tr')) ∧
      s.hbFree tr' = true ∧ St.Eqv a' a :=
  OrderedExec.order_indep h hl (State.components_nodup hc) (State.components_convex hc) hx ho

end Indep

/-! ## Components -/

/-- The components of a state run one after the other, each from the memory
left by the previous ones. -/
inductive CompsSeq (ctx : Ctx) (ρ : Valuation) (s : State) : Mem → List (List NodeId) → Mem → Prop where
  | nil {m : Mem} : CompsSeq ctx ρ s m [] m
  | cons {m m' : Mem} {comp : List NodeId} {rest : List (List NodeId)} {st : St}
      (hc : OrderedExec ctx ρ s comp (s.topLevelOf comp) { mem := m } (.ok st))
      (hr : CompsSeq ctx ρ s st.mem rest m') :
      CompsSeq ctx ρ s m (comp :: rest) m'

section Comps

variable {ctx : Ctx} {ρ : Valuation} {s : State} (h : SeqPure ctx s)
include h

/-- Logs of normal component runs are plain. -/
theorem CompsExec.plain {m : Mem} :
    ∀ {comps : List (List NodeId)} {logs : List Log}, CompsExec ctx ρ s m comps (.ok logs) → ∀ g ∈ logs, g.Plain
  | [], _, hx => by cases hx; intro g hg; cases hg
  | _ :: _, _, hx => by
    cases hx with
    | cons hc hr =>
      intro g hg
      rcases List.mem_cons.1 hg with rfl | hg
      · exact (OrderedExec.frame_ok h (st := { mem := m }) hc).2.2.1
      · exact CompsExec.plain hr g hg

/-- **Component merge is sequential**: the merge of the component logs is
running the components one after the other, from every memory that agrees
with the state's initial memory on what the components read. -/
theorem CompsExec.sequential {m : Mem} :
    ∀ {comps : List (List NodeId)} {logs : List Log}, CompsExec ctx ρ s m comps (.ok logs) →
      Log.pairwiseFree logs = true → ∀ m₀ : Mem, Mem.Agree (Log.unionAll logs).reads m₀ m →
        CompsSeq ctx ρ s m₀ comps (logs.foldl (fun m g => g.apply ctx.I.wcr m) m₀)
  | [], _, hx, _, m₀, _ => by cases hx; exact .nil
  | _ :: _, _, hx, hf, m₀, hag => by
    have hpl := CompsExec.plain h hx
    cases hx with
    | cons hc hrest =>
      obtain ⟨-, -, -, hfr⟩ := OrderedExec.frame_ok h (st := { mem := m }) hc
      have hc₀ := hfr { mem := m₀ } hag.1 rfl
        (fun l hl => hag.2 l (Log.mem_unionAll_reads List.mem_cons_self hl))
      obtain ⟨hcg, hf'⟩ := Log.pairwiseFree_cons.1 hf
      have hfree := (Log.conflicts_eq_false_iff _ _).1 hcg
      refine .cons hc₀ ?_
      refine CompsExec.sequential hrest hf' _ ⟨?_, fun l hl => ?_⟩
      · exact ((hpl _ List.mem_cons_self).apply_aliases _ _).trans hag.1
      · have hl' : l ∈ (Log.unionAll _).reads := hl
        rw [Log.apply_cells_untouched _ _ _ (fun hw => (hfree.1.write l hw).2.1 hl')
          (fun hw => hfree.1.wcrRead l hw hl')]
        exact hag.2 l (by simp only [Log.unionAll, List.flatMap_cons, List.mem_append] at hl' ⊢; exact .inr hl')

/-- **Component merge = sequential, in any order**: the merge of a state's
component logs (`stateFinish`) applies them one after the other, in any
order (queues are untouched in the fragment), and that is running the
components one after the other (`CompsExec.sequential`). -/
theorem CompsExec.merge_perm (hl : ctx.I.wcr.Lawful) {m : Mem} {comps : List (List NodeId)}
    {logs logs' : List Log} (hx : CompsExec ctx ρ s m comps (.ok logs)) (hf : Log.pairwiseFree logs = true)
    (hp : logs.Perm logs') :
    merge ctx.I.wcr m logs = .ok (logs'.foldl (fun m g => g.apply ctx.I.wcr m) m) := by
  have hpl := CompsExec.plain h hx
  rw [Dace.merge_perm _ hl m hp hf ?_, merge_eq_foldl _ _ ((Log.pairwiseFree_perm hp) ▸ hf)]
  refine List.Pairwise.imp_of_mem (R := fun _ _ => True) (fun ha _ _ => ?_) (List.pairwise_of_forall fun _ _ => trivial)
  intro l hl
  simp [Log.pushedQueues, (hpl _ ha).queues] at hl

end Comps

end Dace
