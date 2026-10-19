-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Scope

/-!
# Specification of the scope and concurrency functions

Proofs about the total functions of `Dace/SDFG/Scope.lean`:

* `mem_closure`: the breadth-first closure is exactly reflexive-transitive
  reachability (given a bound on the reachable elements), and `closure_nodup`;
* `State.mem_scopeNodes`: under `State.ScopeWF`, `id ∈ s.scopeNodes e ↔
  e ∈ s.scopeChain id`, and `State.scopeNodes_nodup`;
* `State.components`: a partition of the nodes into weak-connectivity classes
  (`mem_components_iff`, `components_cover`, `components_flatten_nodup`,
  `components_flatten_perm`);
* `State.mem_treeEdges`: memlet trees are the `prev`/`next` closure;
* `State.topoOrder_spec`: Kahn's algorithm returns a topological order, and
  `State.isTopologicalOrder_perm`: on duplicate-free lists such an order is a
  permutation.
-/

set_option autoImplicit false

namespace Dace

/-! ## `appendNew` -/

section AppendNew

variable {α : Type} [BEq α] [LawfulBEq α]

theorem appendNew_eq_append (acc ys : List α) : ∃ t, appendNew acc ys = acc ++ t := by
  induction ys generalizing acc with
  | nil => exact ⟨[], by simp [appendNew]⟩
  | cons y ys ih =>
    simp only [appendNew, List.foldl_cons]
    split
    · exact ih acc
    · obtain ⟨t, ht⟩ := ih (acc ++ [y])
      exact ⟨[y] ++ t, by simpa [appendNew] using ht⟩

theorem mem_appendNew {acc ys : List α} {x : α} : x ∈ appendNew acc ys ↔ x ∈ acc ∨ x ∈ ys := by
  induction ys generalizing acc with
  | nil => simp [appendNew]
  | cons y ys ih =>
    simp only [appendNew, List.foldl_cons] at ih ⊢
    split
    · next h =>
      rw [ih]
      have : y ∈ acc := by simpa using h
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (List.mem_cons_of_mem _ h)
      · rintro (h | h)
        · exact Or.inl h
        · rcases List.mem_cons.mp h with rfl | h
          · exact Or.inl this
          · exact Or.inr h
    · rw [ih]; simp [or_assoc]

theorem nodup_appendNew {acc ys : List α} (h : acc.Nodup) : (appendNew acc ys).Nodup := by
  induction ys generalizing acc with
  | nil => simpa [appendNew] using h
  | cons y ys ih =>
    simp only [appendNew, List.foldl_cons] at ih ⊢
    split
    · exact ih h
    · next hy =>
      apply ih
      rw [List.nodup_append]
      refine ⟨h, List.nodup_singleton y, ?_⟩
      intro a ha b hb
      simp only [List.mem_singleton] at hb
      subst hb
      rintro rfl
      exact hy (by simpa using ha)

end AppendNew

/-! ## Breadth-first closure -/

section Closure

variable {α : Type} [BEq α] [LawfulBEq α] (step : α → List α)

/-- One `step` edge. -/
def StepRel (x y : α) : Prop := y ∈ step x

theorem closureFrom_sound :
    ∀ (fuel : Nat) (seen : List α) (i : Nat) (y : α), y ∈ closureFrom step fuel seen i →
      ∃ x ∈ seen, Relation.ReflTransGen (StepRel step) x y := by
  intro fuel
  induction fuel with
  | zero => intro seen i y hy; exact ⟨y, hy, .refl⟩
  | succ fuel ih =>
    intro seen i y hy
    simp only [closureFrom] at hy
    split at hy
    · next x hx =>
      obtain ⟨z, hz, hzy⟩ := ih _ _ y hy
      have hxs : x ∈ seen := List.mem_of_getElem? hx
      rcases mem_appendNew.mp hz with hz | hz
      · exact ⟨z, hz, hzy⟩
      · exact ⟨x, hxs, Relation.ReflTransGen.head hz hzy⟩
    · exact ⟨y, hy, .refl⟩

/-- Invariant-based completeness of `closureFrom`: with enough fuel, the
result is closed under `step`, contains `seen` and has no duplicates. -/
theorem closureFrom_complete (U : List α) (hU : ∀ x, ∀ y ∈ step x, y ∈ U) :
    ∀ (fuel : Nat) (seen : List α) (i : Nat), seen.Nodup → (∀ x ∈ seen, x ∈ U) →
      i ≤ seen.length → U.length + 1 ≤ fuel + i →
      (∀ x ∈ seen.take i, ∀ y ∈ step x, y ∈ seen) →
      (closureFrom step fuel seen i).Nodup ∧ (∀ x ∈ seen, x ∈ closureFrom step fuel seen i) ∧
      (∀ x ∈ closureFrom step fuel seen i, ∀ y ∈ step x, y ∈ closureFrom step fuel seen i) := by
  intro fuel
  induction fuel with
  | zero =>
    intro seen i hnd hsub hi hfuel _
    have := List.Nodup.length_le_of_subset hnd hsub
    omega
  | succ fuel ih =>
    intro seen i hnd hsub hi hfuel hdone
    simp only [closureFrom]
    split
    · next x hx =>
      have hlt : i < seen.length := by
        rcases List.getElem?_eq_some_iff.mp hx with ⟨h, _⟩; exact h
      have hxe : seen[i] = x := by
        rcases List.getElem?_eq_some_iff.mp hx with ⟨_, h⟩; exact h
      obtain ⟨t, ht⟩ := appendNew_eq_append seen (step x)
      have hnd' := nodup_appendNew (ys := step x) hnd
      have hsub' : ∀ z ∈ appendNew seen (step x), z ∈ U := by
        intro z hz
        rcases mem_appendNew.mp hz with hz | hz
        · exact hsub z hz
        · exact hU x z hz
      have hlen : seen.length ≤ (appendNew seen (step x)).length := by rw [ht]; simp
      have htake : (appendNew seen (step x)).take (i + 1) = seen.take i ++ [x] := by
        rw [ht, List.take_append_of_le_length (by omega), List.take_add_one, List.getElem?_eq_getElem hlt,
          hxe]
        rfl
      obtain ⟨h1, h2, h3⟩ := ih (appendNew seen (step x)) (i + 1) hnd' hsub' (by omega) (by omega)
        (by
          rw [htake]
          intro z hz y hy
          rcases List.mem_append.mp hz with hz | hz
          · exact mem_appendNew.mpr (Or.inl (hdone z hz y hy))
          · rw [List.mem_singleton.mp hz] at hy; exact mem_appendNew.mpr (Or.inr hy))
      exact ⟨h1, fun z hz => h2 z (mem_appendNew.mpr (Or.inl hz)), h3⟩
    · next hx =>
      have hge : seen.length ≤ i := List.getElem?_eq_none_iff.mp hx
      have htk : seen.take i = seen := List.take_of_length_le hge
      rw [htk] at hdone
      exact ⟨hnd, fun z hz => hz, hdone⟩

variable {step}

/-- Characterization of `closure`: given a list `U` of length at most
`bound` containing `start` and every result of `step`, the closure contains
exactly the elements reachable from `start`. -/
theorem mem_closure {bound : Nat} {start : α} (U : List α) (hU : ∀ x, ∀ y ∈ step x, y ∈ U)
    (hs : start ∈ U) (hb : U.length ≤ bound) {y : α} :
    y ∈ closure step bound start ↔ Relation.ReflTransGen (StepRel step) start y := by
  constructor
  · intro hy
    obtain ⟨x, hx, hxy⟩ := closureFrom_sound step _ _ _ y hy
    rw [List.mem_singleton.mp hx] at hxy; exact hxy
  · obtain ⟨_, h2, h3⟩ := closureFrom_complete step U hU (bound + 1) [start] 0 (List.nodup_singleton _)
      (by simpa using hs) (by simp) (by omega) (by simp)
    intro hy
    induction hy with
    | refl => exact h2 start (List.mem_singleton_self _)
    | tail _ hbc ih => exact h3 _ ih _ hbc

theorem closure_nodup {bound : Nat} {start : α} (U : List α) (hU : ∀ x, ∀ y ∈ step x, y ∈ U)
    (hs : start ∈ U) (hb : U.length ≤ bound) : (closure step bound start).Nodup :=
  (closureFrom_complete step U hU (bound + 1) [start] 0 (List.nodup_singleton _)
    (by simpa using hs) (by simp) (by omega) (by simp)).1

end Closure

theorem DNode.not_isEntry_and_isExit (n : DNode) : ¬ (n.isEntry = true ∧ n.isExit = true) := by
  rcases n with ⟨_, _, _, _, _, k⟩
  cases k <;> simp [DNode.isEntry, DNode.isExit]

namespace State

variable {s : State}

/-! ## Nodes -/

theorem node?_some {id : NodeId} {n : DNode} (h : s.node? id = some n) : n ∈ s.nodes ∧ n.id = id :=
  ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

theorem node?_of_mem (hnd : (s.nodes.map (·.id)).Nodup) {n : DNode} (h : n ∈ s.nodes) :
    s.node? n.id = some n := by
  unfold node?
  obtain ⟨m, hm⟩ : ∃ m, s.nodes.find? (·.id = n.id) = some m :=
    Option.isSome_iff_exists.mp (List.find?_isSome.mpr ⟨n, h, by simp⟩)
  have hm1 := List.mem_of_find?_eq_some hm
  have hm2 : m.id = n.id := by simpa using List.find?_some hm
  rw [hm, List.inj_on_of_nodup_map hnd hm1 h hm2]

theorem isEntryNode_iff {id : NodeId} : s.isEntryNode id = true ↔ ∃ m, s.node? id = some m ∧ m.isEntry = true := by
  unfold isEntryNode
  cases s.node? id <;> simp

/-- Soundness of the `ScopeWF` decision procedure. -/
theorem scopeWF?_sound (h : s.scopeWF? = true) : s.ScopeWF := by
  simp only [scopeWF?, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  refine ⟨h1, fun n hn p hp => ?_, h3⟩
  have := h2 n hn
  rw [hp] at this
  exact this

/-! ## Scope chains -/

theorem parent?_some {id p : NodeId} (h : s.parent? id = some p) :
    ∃ n, s.node? id = some n ∧ n.scopeEntry = some p := by
  unfold parent? at h
  split at h
  · next n hn =>
    split at h
    · next q hq =>
      split at h
      · simp at h
      · exact ⟨n, hn, by rw [hq]; simpa using h⟩
    · simp at h
  · simp at h

theorem parent?_of_child (hwf : s.ScopeWF) {n : DNode} (hn : n ∈ s.nodes) {p : NodeId}
    (hp : n.scopeEntry = some p) : s.parent? n.id = some p := by
  have hnode := node?_of_mem hwf.nodup hn
  unfold parent?
  rw [hnode]
  simp only [hp]
  split
  · next hc =>
    exfalso
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨m, hm, hme⟩ := isEntryNode_iff.mp (hwf.parentEntry n hn p hp)
    rw [hc.2, hnode] at hm
    cases hm
    exact DNode.not_isEntry_and_isExit n ⟨hme, hc.1⟩
  · rfl

theorem scopeChainFuel_length : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → L.length < f
  | 0, _, _, h => by simp [scopeChainFuel] at h
  | f + 1, id, L, h => by
    simp only [scopeChainFuel] at h
    split at h
    · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      have := scopeChainFuel_length hL'
      simp only [List.length_cons]
      omega
    · cases h; simp

theorem scopeChainFuel_cons {f : Nat} {id p : NodeId} {L : List NodeId}
    (h : s.scopeChainFuel f id = some L) (hp : s.parent? id = some p) :
    ∃ f' L', f = f' + 1 ∧ s.scopeChainFuel f' p = some L' ∧ L = p :: L' := by
  cases f with
  | zero => simp [scopeChainFuel] at h
  | succ f =>
    simp only [scopeChainFuel, hp] at h
    obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
    exact ⟨f, L', rfl, hL', rfl⟩

theorem scopeChainFuel_mono : ∀ {f : Nat} (k : Nat) {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → s.scopeChainFuel (f + k) id = some L
  | 0, _, _, _, h => by simp [scopeChainFuel] at h
  | f + 1, k, id, L, h => by
    rw [Nat.add_right_comm]
    simp only [scopeChainFuel] at h ⊢
    split at h
    · next p hp =>
      obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      rw [scopeChainFuel_mono k hL']
      rfl
    · exact h

/-- Complete scope chains do not depend on the fuel. -/
theorem scopeChainFuel_det {f f' : Nat} {id : NodeId} {L L' : List NodeId}
    (h : s.scopeChainFuel f id = some L) (h' : s.scopeChainFuel f' id = some L') : L = L' := by
  have a := scopeChainFuel_mono f' h
  have b := scopeChainFuel_mono f h'
  rw [Nat.add_comm] at b
  rw [a] at b
  exact Option.some.inj b

/-- A complete chain is closed under `parent?`. -/
theorem scopeChainFuel_parent_mem {q r : NodeId} : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → q ∈ L → s.parent? q = some r → r ∈ L
  | 0, _, _, h, _, _ => by simp [scopeChainFuel] at h
  | f + 1, id, L, h, hq, hr => by
    simp only [scopeChainFuel] at h
    split at h
    · next p hp =>
      obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      rcases List.mem_cons.mp hq with rfl | hq
      · obtain ⟨_, L'', _, _, rfl⟩ := scopeChainFuel_cons hL' hr
        exact List.mem_cons_of_mem _ (List.mem_cons_self ..)
      · exact List.mem_cons_of_mem _ (scopeChainFuel_parent_mem hL' hq hr)
    · cases h; simp at hq

/-- The chain of an element of a complete chain is a shorter sub-chain. -/
theorem scopeChainFuel_suffix {q : NodeId} : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → q ∈ L →
      ∃ f' L', s.scopeChainFuel f' q = some L' ∧ L'.length < L.length ∧ ∀ x ∈ L', x ∈ L
  | 0, _, _, h, _ => by simp [scopeChainFuel] at h
  | f + 1, id, L, h, hq => by
    simp only [scopeChainFuel] at h
    split at h
    · next p hp =>
      obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      rcases List.mem_cons.mp hq with rfl | hq
      · exact ⟨f, L', hL', by simp, fun x hx => List.mem_cons_of_mem _ hx⟩
      · obtain ⟨f'', L'', h1, h2, h3⟩ := scopeChainFuel_suffix hL' hq
        exact ⟨f'', L'', h1, by simp; omega, fun x hx => List.mem_cons_of_mem _ (h3 x hx)⟩
    · cases h; simp at hq

/-- A node is not in its own complete scope chain. -/
theorem not_mem_scopeChainFuel {f : Nat} {id : NodeId} {L : List NodeId}
    (h : s.scopeChainFuel f id = some L) : id ∉ L := by
  intro hm
  obtain ⟨_, L', h1, h2, _⟩ := scopeChainFuel_suffix h hm
  rw [scopeChainFuel_det h1 h] at h2
  exact Nat.lt_irrefl _ h2

theorem scopeChainFuel_nodup : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → L.Nodup
  | 0, _, _, h => by simp [scopeChainFuel] at h
  | f + 1, id, L, h => by
    simp only [scopeChainFuel] at h
    split at h
    · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      exact List.nodup_cons.mpr ⟨not_mem_scopeChainFuel hL', scopeChainFuel_nodup hL'⟩
    · cases h; simp

/-- No element of the full chain `p :: L` of `p` has `p` as its parent. -/
theorem not_parent_in_chain {f : Nat} {p y : NodeId} {L : List NodeId}
    (h : s.scopeChainFuel f p = some L) (hy : y ∈ p :: L) (hp : s.parent? y = some p) : False := by
  rcases List.mem_cons.mp hy with rfl | hy
  · obtain ⟨_, _, _, _, rfl⟩ := scopeChainFuel_cons h hp
    exact not_mem_scopeChainFuel h (List.mem_cons_self ..)
  · exact not_mem_scopeChainFuel h (scopeChainFuel_parent_mem h hy hp)

/-- Within the full chain `id :: L` of a node, at most one element has a
given parent. -/
theorem scopeChainFuel_unique_parent {x y e : NodeId} : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → x ∈ id :: L → y ∈ id :: L →
      s.parent? x = some e → s.parent? y = some e → x = y
  | 0, _, _, h, _, _, _, _ => by simp [scopeChainFuel] at h
  | f + 1, id, L, h, hx, hy, hpx, hpy => by
    simp only [scopeChainFuel] at h
    split at h
    · next p hp =>
      obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      rcases List.mem_cons.mp hx with hx' | hx' <;> rcases List.mem_cons.mp hy with hy' | hy'
      · rw [hx', hy']
      · rw [hx', hp] at hpx; cases hpx
        exact (not_parent_in_chain hL' hy' hpy).elim
      · rw [hy', hp] at hpy; cases hpy
        exact (not_parent_in_chain hL' hx' hpx).elim
      · exact scopeChainFuel_unique_parent hL' hx' hy' hpx hpy
    · cases h
      simp only [List.mem_singleton] at hx hy
      rw [hx, hy]

/-! ## Scope nodes -/

theorem mem_scopeNodesFuel_succ {F : Nat} {e id : NodeId} :
    id ∈ s.scopeNodesFuel (F + 1) e ↔
      ∃ c ∈ s.nodes, c.scopeEntry = some e ∧ (id = c.id ∨ (c.isEntry = true ∧ id ∈ s.scopeNodesFuel F c.id)) := by
  simp only [scopeNodesFuel, children, List.mem_flatMap, List.mem_filter, decide_eq_true_eq, List.mem_cons]
  constructor
  · rintro ⟨c, ⟨hc, he⟩, h⟩
    refine ⟨c, hc, he, ?_⟩
    rcases h with h | h
    · exact Or.inl h
    · split at h
      · next hce => exact Or.inr ⟨hce, h⟩
      · simp at h
  · rintro ⟨c, hc, he, h⟩
    refine ⟨c, ⟨hc, he⟩, ?_⟩
    rcases h with h | ⟨hce, h⟩
    · exact Or.inl h
    · exact Or.inr (by simp [hce, h])

theorem mem_scopeNodesFuel_node : ∀ {F : Nat} {e id : NodeId}, id ∈ s.scopeNodesFuel F e →
    ∃ n ∈ s.nodes, n.id = id
  | 0, _, _, h => by simp [scopeNodesFuel] at h
  | F + 1, _, _, h => by
    obtain ⟨c, hc, _, h⟩ := mem_scopeNodesFuel_succ.mp h
    rcases h with rfl | ⟨_, h⟩
    · exact ⟨c, hc, rfl⟩
    · exact mem_scopeNodesFuel_node h

/-- Soundness of `scopeNodesFuel` with respect to scope chains (any fuel). -/
theorem scopeNodesFuel_chain (hwf : s.ScopeWF) {f : Nat} {L : List NodeId} : ∀ {F : Nat} {e id : NodeId},
    id ∈ s.scopeNodesFuel F e → s.scopeChainFuel f id = some L → e ∈ L
  | 0, _, _, h, _ => by simp [scopeNodesFuel] at h
  | F + 1, e, id, h, hL => by
    obtain ⟨c, hc, he, h⟩ := mem_scopeNodesFuel_succ.mp h
    have hpc := parent?_of_child hwf hc he
    rcases h with rfl | ⟨_, h⟩
    · obtain ⟨_, _, _, _, rfl⟩ := scopeChainFuel_cons hL hpc
      exact List.mem_cons_self ..
    · exact scopeChainFuel_parent_mem hL (scopeNodesFuel_chain hwf h hL) hpc

theorem scopeNodesFuel_extend (hwf : s.ScopeWF) : ∀ {F : Nat} {e p : NodeId},
    p ∈ s.scopeNodesFuel F e → s.isEntryNode p = true → ∀ n ∈ s.nodes, n.scopeEntry = some p →
      n.id ∈ s.scopeNodesFuel (F + 1) e
  | 0, _, _, h => by simp [scopeNodesFuel] at h
  | F + 1, e, p, h => by
    intro hpe n hn hnp
    obtain ⟨c, hc, he, h⟩ := mem_scopeNodesFuel_succ.mp h
    refine mem_scopeNodesFuel_succ.mpr ⟨c, hc, he, Or.inr ⟨?_, ?_⟩⟩
    · rcases h with rfl | ⟨hce, _⟩
      · obtain ⟨m, hm, hme⟩ := isEntryNode_iff.mp hpe
        rw [node?_of_mem hwf.nodup hc] at hm
        cases hm
        exact hme
      · exact hce
    · rcases h with rfl | ⟨_, h⟩
      · exact mem_scopeNodesFuel_succ.mpr ⟨n, hn, hnp, Or.inl rfl⟩
      · exact scopeNodesFuel_extend hwf h hpe n hn hnp

/-- Completeness of `scopeNodesFuel` with respect to complete scope chains. -/
theorem chain_scopeNodesFuel (hwf : s.ScopeWF) : ∀ {f : Nat} {id : NodeId} {L : List NodeId},
    s.scopeChainFuel f id = some L → ∀ e ∈ L, ∀ F, L.length ≤ F → id ∈ s.scopeNodesFuel F e
  | 0, _, _, h => by simp [scopeChainFuel] at h
  | f + 1, id, L, h => by
    intro e he F hF
    simp only [scopeChainFuel] at h
    split at h
    · next p hp =>
      obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.mp h
      obtain ⟨n, hn, hnp⟩ := parent?_some hp
      obtain ⟨hnm, rfl⟩ := node?_some hn
      obtain ⟨F, rfl⟩ : ∃ F', F = F' + 1 := ⟨F - 1, by simp at hF; omega⟩
      rcases List.mem_cons.mp he with rfl | he
      · exact mem_scopeNodesFuel_succ.mpr ⟨n, hnm, hnp, Or.inl rfl⟩
      · have hpF := chain_scopeNodesFuel hwf hL' e he F (by simp at hF; omega)
        exact scopeNodesFuel_extend hwf hpF (hwf.parentEntry n hnm p hnp) n hnm hnp
    · cases h; simp at he

/-- **Specification of `scopeNodes`**: on a state with well-formed scopes, a
node is inside the scope of `e` iff `e` is on its scope chain. -/
theorem mem_scopeNodes (hwf : s.ScopeWF) {e id : NodeId} : id ∈ s.scopeNodes e ↔ e ∈ s.scopeChain id := by
  unfold scopeNodes scopeChain
  constructor
  · intro h
    obtain ⟨n, hn, rfl⟩ := mem_scopeNodesFuel_node h
    obtain ⟨L, hL⟩ := Option.isSome_iff_exists.mp (hwf.acyclic n hn)
    rw [hL]
    exact scopeNodesFuel_chain hwf h hL
  · intro h
    cases hL : s.scopeChainFuel (s.nodes.length + 1) id with
    | none => rw [hL] at h; simp at h
    | some L =>
      rw [hL] at h
      have := scopeChainFuel_length hL
      exact chain_scopeNodesFuel hwf hL e h _ (by simp at this ⊢; omega)

theorem mem_scopeNodes_node {e id : NodeId} (h : id ∈ s.scopeNodes e) : ∃ n ∈ s.nodes, n.id = id :=
  mem_scopeNodesFuel_node h

/-- Membership in `scopeNodes` through the parent: the scope of `e` contains
its children and, transitively, the contents of its child entries. -/
theorem mem_scopeNodes_iff_parent (hwf : s.ScopeWF) {e id : NodeId} :
    id ∈ s.scopeNodes e ↔ ∃ p, s.parent? id = some p ∧ (p = e ∨ p ∈ s.scopeNodes e) := by
  rw [mem_scopeNodes hwf]
  unfold scopeChain
  constructor
  · intro h
    cases hL : s.scopeChainFuel (s.nodes.length + 1) id with
    | none => rw [hL] at h; simp at h
    | some L =>
      rw [hL] at h
      cases hp : s.parent? id with
      | none =>
        cases hf : s.nodes.length + 1 with
        | zero => omega
        | succ f => rw [hf] at hL; simp [scopeChainFuel, hp] at hL; subst hL; simp at h
      | some p =>
        obtain ⟨f', L', _, hL', rfl⟩ := scopeChainFuel_cons hL hp
        refine ⟨p, rfl, ?_⟩
        rcases List.mem_cons.mp h with rfl | h'
        · exact Or.inl rfl
        · right
          obtain ⟨n, hn, hnp⟩ := parent?_some hp
          obtain ⟨hnm, rfl⟩ := node?_some hn
          obtain ⟨m, hm, hme⟩ := isEntryNode_iff.mp (hwf.parentEntry n hnm p hnp)
          obtain ⟨hmm, rfl⟩ := node?_some hm
          rw [mem_scopeNodes hwf]
          unfold scopeChain
          obtain ⟨L₂, hL₂⟩ := Option.isSome_iff_exists.mp (hwf.acyclic m hmm)
          rw [hL₂, scopeChainFuel_det hL₂ hL']
          exact h'
  · rintro ⟨p, hp, h⟩
    obtain ⟨n, hn, hnp⟩ := parent?_some hp
    obtain ⟨hnm, rfl⟩ := node?_some hn
    obtain ⟨L, hL⟩ := Option.isSome_iff_exists.mp (hwf.acyclic n hnm)
    rw [hL]
    obtain ⟨_, L', _, hL', rfl⟩ := scopeChainFuel_cons hL hp
    rcases h with rfl | h
    · exact List.mem_cons_self ..
    · rw [mem_scopeNodes hwf] at h
      unfold scopeChain at h
      cases hL₂ : s.scopeChainFuel (s.nodes.length + 1) p with
      | none => rw [hL₂] at h; simp at h
      | some L₂ =>
        rw [hL₂] at h
        rw [scopeChainFuel_det hL₂ hL'] at h
        exact List.mem_cons_of_mem _ h

theorem scopeNodesFuel_nodup (hwf : s.ScopeWF) : ∀ (F : Nat) (e : NodeId), (s.scopeNodesFuel F e).Nodup
  | 0, _ => by simp [scopeNodesFuel]
  | F + 1, e => by
    have hchain : ∀ (c : DNode) (z : NodeId) (f : Nat) (L : List NodeId), z ∈ s.scopeNodesFuel F c.id →
        s.scopeChainFuel f z = some L → c.id ∈ L :=
      fun _ _ _ _ hz hL => scopeNodesFuel_chain hwf hz hL
    simp only [scopeNodesFuel]
    rw [List.nodup_flatMap]
    constructor
    · intro c hc
      simp only [children, List.mem_filter] at hc
      refine List.nodup_cons.mpr ⟨?_, ?_⟩
      · split
        · intro hm
          obtain ⟨L, hL⟩ := Option.isSome_iff_exists.mp (hwf.acyclic c hc.1)
          exact not_mem_scopeChainFuel hL (hchain _ _ _ _ hm hL)
        · simp
      · split
        · exact scopeNodesFuel_nodup hwf F c.id
        · simp
    · have hpw : (s.children (some e)).Pairwise (fun a b => a.id ≠ b.id) :=
        (List.pairwise_map.mp hwf.nodup).filter _
      refine hpw.imp_of_mem ?_
      intro c₁ c₂ hc₁ hc₂ hne z hz₁ hz₂
      simp only [children, List.mem_filter, decide_eq_true_eq] at hc₁ hc₂
      -- the full chain of `z` contains both `c₁` and `c₂`, both with parent `e`
      have hzn : ∃ n ∈ s.nodes, n.id = z := by
        simp only [List.mem_cons] at hz₁
        rcases hz₁ with rfl | hz₁
        · exact ⟨c₁, hc₁.1, rfl⟩
        · split at hz₁
          · exact mem_scopeNodesFuel_node hz₁
          · simp at hz₁
      obtain ⟨n, hn, rfl⟩ := hzn
      obtain ⟨L, hL⟩ := Option.isSome_iff_exists.mp (hwf.acyclic n hn)
      have hin : ∀ {c : DNode}, c ∈ s.nodes →
          n.id ∈ c.id :: (if c.isEntry = true then s.scopeNodesFuel F c.id else []) → c.id ∈ n.id :: L := by
        intro c hc hz
        rcases List.mem_cons.mp hz with h | hz
        · rw [h]; exact List.mem_cons_self ..
        · split at hz
          · exact List.mem_cons_of_mem _ (hchain _ _ _ _ hz hL)
          · simp at hz
      exact hne (scopeChainFuel_unique_parent hL (hin hc₁.1 hz₁) (hin hc₂.1 hz₂)
        (parent?_of_child hwf hc₁.1 hc₁.2) (parent?_of_child hwf hc₂.1 hc₂.2))

theorem scopeNodes_nodup (hwf : s.ScopeWF) (e : NodeId) : (s.scopeNodes e).Nodup :=
  scopeNodesFuel_nodup hwf _ e

/-! ## Components -/

/-- Undirected adjacency through an edge of the state (dependency edges count). -/
def Adj (s : State) (x y : NodeId) : Prop :=
  ∃ e ∈ s.edges, (e.src = x ∧ e.dst = y) ∨ (e.src = y ∧ e.dst = x)

/-- Weak connectivity: the reflexive-transitive closure of `Adj`. -/
def Connected (s : State) : NodeId → NodeId → Prop := Relation.ReflTransGen s.Adj

theorem Adj.symm {x y : NodeId} (h : s.Adj x y) : s.Adj y x := by
  obtain ⟨e, he, h⟩ := h
  exact ⟨e, he, h.symm⟩

instance : Std.Symm s.Adj := ⟨fun _ _ h => h.symm⟩

theorem Connected.symm {x y : NodeId} (h : s.Connected x y) : s.Connected y x :=
  Relation.ReflTransGen.stdSymm.symm _ _ h

theorem Connected.trans {x y z : NodeId} (h₁ : s.Connected x y) (h₂ : s.Connected y z) :
    s.Connected x z :=
  Relation.ReflTransGen.trans h₁ h₂

theorem Connected.of_adj {x y : NodeId} (h : s.Adj x y) : s.Connected x y :=
  Relation.ReflTransGen.single h

theorem mem_neighbors {x y : NodeId} : y ∈ s.neighbors x ↔ s.Adj x y := by
  simp only [neighbors, List.mem_filterMap, Adj]
  constructor
  · rintro ⟨e, he, h⟩
    refine ⟨e, he, ?_⟩
    split_ifs at h with h1 h2
    · cases h; exact Or.inl ⟨h1, rfl⟩
    · cases h; exact Or.inr ⟨rfl, h2⟩
  · rintro ⟨e, he, h⟩
    refine ⟨e, he, ?_⟩
    rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · simp [h1, h2]
    · by_cases h3 : e.src = x
      · simp only [h3, ite_true, Option.some.injEq]
        rw [h2, ← h3, h1]
      · split_ifs
        simp_all

/-- All edge endpoints plus `x`: a bound for the component of `x`. -/
def endpointsWith (s : State) (x : NodeId) : List NodeId := x :: s.edges.flatMap fun e => [e.src, e.dst]

theorem endpointsWith_length (x : NodeId) : (s.endpointsWith x).length ≤ 2 * s.edges.length + 1 := by
  simp only [endpointsWith, List.length_cons, List.length_flatMap, List.length_nil]
  simp [Nat.mul_comm]

theorem neighbors_sub (x : NodeId) : ∀ (z y : NodeId), y ∈ s.neighbors z → y ∈ s.endpointsWith x := by
  intro z y hy
  obtain ⟨e, he, h⟩ := mem_neighbors.mp hy
  refine List.mem_cons_of_mem _ (List.mem_flatMap.mpr ⟨e, he, ?_⟩)
  rcases h with ⟨_, rfl⟩ | ⟨rfl, _⟩ <;> simp

theorem stepRel_neighbors : StepRel s.neighbors = s.Adj := by
  funext x y
  exact propext mem_neighbors

/-- The component of `x` is its weak-connectivity class. -/
theorem mem_component {x y : NodeId} : y ∈ s.component x ↔ s.Connected x y := by
  unfold component Connected
  rw [mem_closure (s.endpointsWith x) (neighbors_sub x) (List.mem_cons_self ..) (endpointsWith_length x),
    stepRel_neighbors]

theorem component_nodup (x : NodeId) : (s.component x).Nodup :=
  closure_nodup (s.endpointsWith x) (neighbors_sub x) (List.mem_cons_self ..) (endpointsWith_length x)

theorem self_mem_component (x : NodeId) : x ∈ s.component x :=
  mem_component.mpr .refl

theorem componentsFrom_spec : ∀ (ns : List DNode) (seen : List NodeId),
    (∀ x ∈ seen, ∀ y, s.Connected x y → y ∈ seen) →
    (∀ c ∈ s.componentsFrom seen ns, ∃ n ∈ ns, c = s.component n.id) ∧
    (∀ n ∈ ns, n.id ∈ seen ∨ ∃ c ∈ s.componentsFrom seen ns, n.id ∈ c) ∧
    (s.componentsFrom seen ns).flatten.Nodup ∧
    (∀ z ∈ (s.componentsFrom seen ns).flatten, z ∉ seen)
  | [], _, _ => by simp [componentsFrom]
  | n :: rest, seen, hclosed => by
    simp only [componentsFrom]
    split
    · next hn =>
      have hn : n.id ∈ seen := by simpa using hn
      obtain ⟨h1, h2, h3, h4⟩ := componentsFrom_spec rest seen hclosed
      refine ⟨fun c hc => ?_, ?_, h3, h4⟩
      · obtain ⟨m, hm, rfl⟩ := h1 c hc
        exact ⟨m, List.mem_cons_of_mem _ hm, rfl⟩
      intro m hm
      rcases List.mem_cons.mp hm with rfl | hm
      · exact Or.inl hn
      · exact h2 m hm
    · next hn =>
      have hn : n.id ∉ seen := by simpa using hn
      have hclosed' : ∀ x ∈ s.component n.id ++ seen, ∀ y, s.Connected x y → y ∈ s.component n.id ++ seen := by
        intro x hx y hxy
        rcases List.mem_append.mp hx with hx | hx
        · exact List.mem_append.mpr (Or.inl (mem_component.mpr ((mem_component.mp hx).trans hxy)))
        · exact List.mem_append.mpr (Or.inr (hclosed x hx y hxy))
      obtain ⟨h1, h2, h3, h4⟩ := componentsFrom_spec rest _ hclosed'
      refine ⟨?_, ?_, ?_, ?_⟩
      · intro c hc
        rcases List.mem_cons.mp hc with rfl | hc
        · exact ⟨n, List.mem_cons_self .., rfl⟩
        · obtain ⟨m, hm, rfl⟩ := h1 c hc
          exact ⟨m, List.mem_cons_of_mem _ hm, rfl⟩
      · intro m hm
        rcases List.mem_cons.mp hm with rfl | hm
        · exact Or.inr ⟨_, List.mem_cons_self .., self_mem_component _⟩
        · rcases h2 m hm with h | ⟨c, hc, hmc⟩
          · rcases List.mem_append.mp h with h | h
            · exact Or.inr ⟨_, List.mem_cons_self .., h⟩
            · exact Or.inl h
          · exact Or.inr ⟨c, List.mem_cons_of_mem _ hc, hmc⟩
      · rw [List.flatten_cons, List.nodup_append]
        refine ⟨component_nodup _, h3, ?_⟩
        intro a ha b hb hab
        subst hab
        exact h4 a hb (List.mem_append.mpr (Or.inl ha))
      · intro z hz hzs
        rw [List.flatten_cons] at hz
        rcases List.mem_append.mp hz with hz | hz
        · exact hn (hclosed z hzs n.id (mem_component.mp hz).symm)
        · exact h4 z hz (List.mem_append.mpr (Or.inr hzs))

theorem components_spec :
    (∀ c ∈ s.components, ∃ n ∈ s.nodes, c = s.component n.id) ∧
    (∀ n ∈ s.nodes, ∃ c ∈ s.components, n.id ∈ c) ∧
    s.components.flatten.Nodup := by
  obtain ⟨h1, h2, h3, _⟩ := componentsFrom_spec (s := s) s.nodes [] (by simp)
  exact ⟨h1, fun n hn => (h2 n hn).resolve_left (by simp), h3⟩

/-- Every node lies in some component. -/
theorem components_cover {n : DNode} (hn : n ∈ s.nodes) : ∃ c ∈ s.components, n.id ∈ c :=
  components_spec.2.1 n hn

/-- Components are pairwise disjoint and duplicate-free. -/
theorem components_flatten_nodup : s.components.flatten.Nodup :=
  components_spec.2.2

/-- A component is the weak-connectivity class of each of its members. -/
theorem mem_components_iff {c : List NodeId} (hc : c ∈ s.components) {x y : NodeId} (hx : x ∈ c) :
    y ∈ c ↔ s.Connected x y := by
  obtain ⟨_, _, rfl⟩ := components_spec.1 c hc
  rw [mem_component] at hx ⊢
  exact ⟨fun h => hx.symm.trans h, fun h => hx.trans h⟩

/-- Nodes joined by an edge lie in the same component. -/
theorem components_edge {c : List NodeId} (hc : c ∈ s.components) {e : DEdge} (he : e ∈ s.edges) :
    e.src ∈ c ↔ e.dst ∈ c := by
  have hadj : s.Adj e.src e.dst := ⟨e, he, Or.inl ⟨rfl, rfl⟩⟩
  constructor
  · intro h; exact (mem_components_iff hc h).mpr (Connected.of_adj hadj)
  · intro h; exact (mem_components_iff hc h).mpr (Connected.of_adj hadj.symm)

/-- Two components sharing a node are equal. -/
theorem components_eq_of_mem {c₁ c₂ : List NodeId} (h₁ : c₁ ∈ s.components) (h₂ : c₂ ∈ s.components)
    {x : NodeId} (hx₁ : x ∈ c₁) (hx₂ : x ∈ c₂) : c₁ = c₂ := by
  by_contra hne
  have hpw := (List.nodup_flatten.mp (components_flatten_nodup (s := s))).2
  exact List.Pairwise.forall hpw h₁ h₂ hne hx₁ hx₂

/-- When every edge endpoint is a node, the components partition the node ids. -/
theorem components_flatten_perm (hnd : (s.nodes.map (·.id)).Nodup)
    (hedges : ∀ e ∈ s.edges, (∃ n ∈ s.nodes, n.id = e.src) ∧ ∃ n ∈ s.nodes, n.id = e.dst) :
    s.components.flatten.Perm (s.nodes.map (·.id)) := by
  rw [List.perm_ext_iff_of_nodup components_flatten_nodup hnd]
  intro x
  simp only [List.mem_flatten, List.mem_map]
  constructor
  · rintro ⟨c, hc, hx⟩
    obtain ⟨n, hn, rfl⟩ := components_spec.1 c hc
    have hconn := mem_component.mp hx
    induction hconn with
    | refl => exact ⟨n, hn, rfl⟩
    | tail _ hyz _ =>
      obtain ⟨e, he, h⟩ := hyz
      rcases h with ⟨_, rfl⟩ | ⟨rfl, _⟩
      · exact (hedges e he).2
      · exact (hedges e he).1
  · rintro ⟨n, hn, rfl⟩
    exact components_cover hn

/-! ## Memlet trees -/

theorem prevEdges_sub {e f : DEdge} (h : f ∈ s.prevEdges e) : f ∈ s.edges := by
  unfold prevEdges at h
  split at h
  · split at h
    · exact (List.mem_filter.mp (List.mem_filter.mp h).1).1
    · simp at h
  · simp at h

theorem nextEdges_sub {e f : DEdge} (h : f ∈ s.nextEdges e) : f ∈ s.edges := by
  unfold nextEdges at h
  split at h
  · split at h
    · exact (List.mem_filter.mp (List.mem_filter.mp h).1).1
    · simp at h
  · simp at h

/-- One memlet-tree step: to a `prev` or `next` edge. -/
def TreeStep (s : State) (e f : DEdge) : Prop := f ∈ s.prevEdges e ∨ f ∈ s.nextEdges e

theorem treeStep_sub (e : DEdge) : ∀ (x y : DEdge), y ∈ s.prevEdges x ++ s.nextEdges x → y ∈ e :: s.edges := by
  intro x y hy
  rcases List.mem_append.mp hy with hy | hy
  · exact List.mem_cons_of_mem _ (prevEdges_sub hy)
  · exact List.mem_cons_of_mem _ (nextEdges_sub hy)

/-- `treeEdges e` is the closure of `e` under `prev`/`next`. -/
theorem mem_treeEdges {e f : DEdge} : f ∈ s.treeEdges e ↔ Relation.ReflTransGen s.TreeStep e f := by
  unfold treeEdges
  rw [mem_closure (e :: s.edges) (treeStep_sub e) (List.mem_cons_self ..) (by simp)]
  have : StepRel (fun f => s.prevEdges f ++ s.nextEdges f) = s.TreeStep := by
    funext x y
    exact propext List.mem_append
  rw [this]

theorem treeEdges_nodup (e : DEdge) : (s.treeEdges e).Nodup :=
  closure_nodup (e :: s.edges) (treeStep_sub e) (List.mem_cons_self ..) (by simp)

/-! ## Topological orders -/

theorem idxOf?_append_singleton {P : List NodeId} {r a : NodeId} :
    (P ++ [r]).idxOf? a = if a ∈ P then P.idxOf? a else if r = a then some P.length else none := by
  unfold List.idxOf?
  rw [List.findIdx?_append]
  by_cases ha : a ∈ P
  · obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp (List.isSome_idxOf?.mpr ha)
    unfold List.idxOf? at hi
    simp [ha, hi]
  · have hn : P.findIdx? (· == a) = none := List.idxOf?_eq_none_iff.mpr ha
    simp only [ha, hn, ite_false, Option.none_or, List.findIdx?_cons, List.findIdx?_nil, beq_iff_eq]
    by_cases hr : r = a <;> simp [hr]

/-- A permutation of `ids` in which every edge between listed nodes goes
forward is a topological order of `ids`. -/
theorem isTopologicalOrder_of_perm {ids P : List NodeId} (hids : ids.Nodup) (hP : P.Perm ids)
    (hresp : ∀ e ∈ s.edges, ∀ i j, P.idxOf? e.src = some i → P.idxOf? e.dst = some j → i < j) :
    s.isTopologicalOrder ids P = true := by
  simp only [isTopologicalOrder, respectsEdges, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
  refine ⟨⟨⟨hP.length_eq, fun x hx => by simpa using hP.symm.subset hx⟩, hP.nodup_iff.mpr hids⟩,
    fun e he => ?_⟩
  split
  · next i j hi hj => exact decide_eq_true (hresp e he i j hi hj)
  · rfl

/-- On duplicate-free `ids`, a topological order is a permutation of `ids`. -/
theorem isTopologicalOrder_perm {ids o : List NodeId} (hids : ids.Nodup)
    (h : s.isTopologicalOrder ids o = true) : o.Perm ids := by
  simp only [isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨hlen, hsub⟩, hnd⟩, _⟩ := h
  have hsub' : ids ⊆ o := fun x hx => by simpa using hsub x hx
  exact ((List.subperm_of_subset hids hsub').perm_of_length_le (by omega)).symm

theorem topoOrder_go_spec {ids : List NodeId} (hids : ids.Nodup) (hloop : ∀ e ∈ s.edges, e.src ≠ e.dst) :
    ∀ (fuel : Nat) (placed remaining o : List NodeId),
      (placed.reverse ++ remaining).Perm ids →
      (∀ e ∈ s.edges, e.dst ∈ placed → e.src ∉ remaining) →
      (∀ e ∈ s.edges, ∀ i j, placed.reverse.idxOf? e.src = some i → placed.reverse.idxOf? e.dst = some j →
        i < j) →
      topoOrder.go s fuel placed remaining = some o → s.isTopologicalOrder ids o = true := by
  intro fuel
  induction fuel with
  | zero =>
    intro placed remaining o hperm _ hresp h
    simp only [topoOrder.go] at h
    split at h
    · next hr =>
      cases h
      have : remaining = [] := List.isEmpty_iff.mp hr
      subst this
      simp only [List.append_nil] at hperm
      exact isTopologicalOrder_of_perm hids hperm hresp
    · cases h
  | succ fuel ih =>
    intro placed remaining o hperm hdst hresp h
    simp only [topoOrder.go] at h
    split at h
    · cases h
      simp only [List.append_nil] at hperm
      exact isTopologicalOrder_of_perm hids hperm hresp
    · split at h
      · cases h
      · next r rest hready =>
        have hr : r ∈ remaining.filter (fun n => (s.inEdges n).all fun e => !remaining.contains e.src || e.src = n) := by
          rw [hready]; exact List.mem_cons_self ..
        simp only [List.mem_filter, List.all_eq_true] at hr
        obtain ⟨hrm, hrin⟩ := hr
        have hnd : (placed.reverse ++ remaining).Nodup := hperm.nodup_iff.mpr hids
        have hrP : r ∉ placed.reverse := by
          intro h'
          exact (List.nodup_append.mp hnd).2.2 r h' r hrm rfl
        refine ih (r :: placed) (remaining.erase r) o ?_ ?_ ?_ h
        · rw [List.reverse_cons, List.append_assoc, List.singleton_append]
          exact ((List.perm_cons_erase hrm).symm.append_left _).trans hperm
        · intro e he hed hes
          have hes' := List.mem_of_mem_erase hes
          rcases List.mem_cons.mp hed with hed | hed
          · have hin : e ∈ s.inEdges r := List.mem_filter.mpr ⟨he, by simpa using hed⟩
            have := hrin e hin
            simp only [Bool.or_eq_true, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not,
              decide_eq_true_eq] at this
            rcases this with h1 | h1
            · exact h1 hes'
            · exact hloop e he (h1.trans hed.symm)
          · exact hdst e he hed hes'
        · intro e he i j hi hj
          rw [List.reverse_cons, idxOf?_append_singleton] at hi hj
          by_cases hs : e.src ∈ placed.reverse <;> by_cases hd : e.dst ∈ placed.reverse <;>
            simp only [hs, hd, ite_true, ite_false] at hi hj
          · exact hresp e he i j hi hj
          · split at hj
            · cases hj
              obtain ⟨hlt, _⟩ := List.idxOf?_eq_some_iff.mp hi
              exact hlt
            · cases hj
          · split at hi
            · next hsr =>
              exfalso
              exact hdst e he (by simpa using hd) (hsr ▸ hrm)
            · cases hi
          · split at hi
            · next hsr =>
              split at hj
              · next hdr => exact (hloop e he (hsr.symm.trans hdr)).elim
              · cases hj
            · cases hi

/-- **Specification of `topoOrder`**: on duplicate-free node lists of a state
without self-loops, Kahn's algorithm returns a topological order. -/
theorem topoOrder_spec {ids o : List NodeId} (hids : ids.Nodup) (hloop : ∀ e ∈ s.edges, e.src ≠ e.dst)
    (h : s.topoOrder ids = some o) : s.isTopologicalOrder ids o = true :=
  topoOrder_go_spec hids hloop _ [] ids o (by simp) (by simp) (by simp) h

end State

end Dace
