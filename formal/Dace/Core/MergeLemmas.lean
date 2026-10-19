-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Merge

/-!
# Algebra of logs and of their parallel merge

Lemmas about `Log.seq`, `Log.conflicts`, `Log.pairwiseFree`, `Log.unionAll`,
`Log.dropNames` and `merge` used by the congruence and order-independence
proofs:

* `Log.seq` is associative with `Log.empty` as unit (`Log.seq_assoc`,
  `Log.empty_seq`; `Log.seq_empty` is in `SDFG/Semantics/Exec.lean`) and keeps
  plain-write locations duplicate free (`Log.WF.seq`).
* `Log.conflicts a b = false` is characterized by the symmetric proposition
  `Log.Free a b` (`Log.conflicts_eq_false_iff`); conflicts are symmetric,
  antitone in the logs (`Log.Sub`) and distribute over `unionAll`.
* `Log.apply` applies one log to a memory; a conflict-free `merge` is the
  sequential application of its logs in list order (`merge_cons_seq`) and does
  not depend on that order (`merge_perm`) when no two logs touch the same
  queue.
* `Log.Le g' g` (`g'` reads at most what `g` reads and has the same effects up
  to reordering of commuting entries) is preserved by `seq`, `unionAll` and
  `dropNames`, and `merge` respects it (`merge_le`).

Per-location characterizations of the merge phases (`Merge.writeCells_eval`
and friends) reduce most statements to per-key list equalities.
-/

set_option autoImplicit false

namespace Dace

/-! ## Generic list lemmas -/

namespace ListLemmas

/-- Folding pointwise updates of a function, evaluated at one key. -/
theorem foldl_update_eval {κ β α : Type} [DecidableEq κ] (k : α → κ) (step : β → α → β)
    (xs : List α) (f : κ → β) (j : κ) :
    xs.foldl (fun f x => Function.update f (k x) (step (f (k x)) x)) f j =
      (xs.filter (fun x => k x = j)).foldl step (f j) := by
  induction xs generalizing f with
  | nil => rfl
  | cons x xs ih =>
    rw [List.foldl_cons, ih]
    by_cases h : k x = j
    · subst h; simp
    · simp [h, Function.update_of_ne (Ne.symm h)]

/-- Pointwise updates at different keys commute. -/
theorem update_step_comm {κ β α : Type} [DecidableEq κ] (k : α → κ) (step : β → α → β)
    (x y : α) (h : k x ≠ k y) (f : κ → β) :
    (fun f x => Function.update f (k x) (step (f (k x)) x))
        ((fun f x => Function.update f (k x) (step (f (k x)) x)) f x) y =
      (fun f x => Function.update f (k x) (step (f (k x)) x))
        ((fun f x => Function.update f (k x) (step (f (k x)) x)) f y) x := by
  simp only [Function.update_of_ne h, Function.update_of_ne (Ne.symm h)]
  exact Function.update_comm h _ _ f

theorem filter_key_filter_not_contains {α κ : Type} [DecidableEq κ] (k : α → κ) (xs : List α)
    (L : List κ) (j : κ) :
    (xs.filter (fun x => !L.contains (k x))).filter (fun x => k x = j) =
      if L.contains j then [] else xs.filter (fun x => k x = j) := by
  rw [List.filter_filter]
  split
  · next h =>
    rw [List.filter_eq_nil_iff]
    intro x _ hx
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hx
    rw [hx.1, h] at hx
    exact Bool.false_ne_true hx.2.symm
  · next h =>
    apply List.filter_congr
    intro x _
    by_cases hx : k x = j
    · simp only [hx, decide_true, Bool.true_and, Bool.not_eq_eq_eq_not, Bool.not_true]
      simpa using h
    · simp [hx]

/-- Filtering by a predicate that is constant on one key class. -/
theorem filter_key_filter_const {α κ : Type} [DecidableEq κ] (k : α → κ) (xs : List α)
    (p : κ → Bool) (j : κ) :
    (xs.filter (fun x => p (k x))).filter (fun x => k x = j) =
      if p j then xs.filter (fun x => k x = j) else [] := by
  rw [List.filter_filter]
  split
  · next h =>
    apply List.filter_congr
    intro x _
    by_cases hx : k x = j
    · simp [hx, h]
    · simp [hx]
  · next h =>
    rw [List.filter_eq_nil_iff]
    intro x _ hx
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hx
    rw [hx.1] at hx
    exact h hx.2

theorem foldl_last_indep {β : Type} (ws : List (Loc × β)) (l : Loc)
    (h : (ws.map Prod.fst).contains l = true) (d d' : β) :
    (ws.filter (fun x => x.1 = l)).foldl (fun _ x => x.2) d =
      (ws.filter (fun x => x.1 = l)).foldl (fun _ x => x.2) d' := by
  cases hf : ws.filter (fun x => x.1 = l) with
  | nil =>
    exfalso
    rw [List.filter_eq_nil_iff] at hf
    simp only [List.contains_iff_mem, List.mem_map] at h
    obtain ⟨x, hx, rfl⟩ := h
    exact hf x hx (by simp)
  | cons y ys => rfl

theorem contains_fst_filter_or {α β : Type} [DecidableEq α] (ws : List (α × β)) (L : List α) (l : α) :
    (((ws.filter (fun x => !L.contains x.1)).map Prod.fst).contains l || L.contains l) =
      ((ws.map Prod.fst).contains l || L.contains l) := by
  cases h : L.contains l
  · rw [Bool.or_false, Bool.or_false, Bool.eq_iff_iff]
    simp only [List.contains_iff_mem, List.mem_map, List.mem_filter]
    constructor
    · rintro ⟨x, ⟨hx, _⟩, rfl⟩; exact ⟨x, hx, rfl⟩
    · rintro ⟨x, hx, rfl⟩; exact ⟨x, ⟨hx, by simpa using h⟩, rfl⟩
  · simp

theorem contains_fst_filter_and {α γ β : Type} [DecidableEq α] (ws : List (α × β)) (p : α → γ)
    (P : γ → Bool) (l : α) :
    ((ws.filter (fun x => P (p x.1))).map Prod.fst).contains l = ((ws.map Prod.fst).contains l && P (p l)) := by
  rw [Bool.eq_iff_iff]
  simp only [List.contains_iff_mem, List.mem_map, List.mem_filter, Bool.and_eq_true]
  constructor
  · rintro ⟨x, ⟨hx, hp⟩, rfl⟩; exact ⟨⟨x, hx, rfl⟩, hp⟩
  · rintro ⟨⟨x, hx, rfl⟩, hp⟩; exact ⟨x, ⟨hx, hp⟩, rfl⟩

/-- `x` commutes past every element of `ys`. -/
theorem foldl_comm_right {α β : Type} (f : β → α → β) (x : α) (ys : List α)
    (h : ∀ y ∈ ys, ∀ z, f (f z x) y = f (f z y) x) (z : β) :
    ys.foldl f (f z x) = f (ys.foldl f z) x := by
  induction ys generalizing z with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.foldl_cons]
    rw [h y (List.mem_cons_self ..) z]
    exact ih (fun y' hy' => h y' (List.mem_cons_of_mem _ hy')) _

/-- Two blocks whose elements commute pairwise can be swapped. -/
theorem foldl_append_comm {α β : Type} (f : β → α → β) (xs ys : List α)
    (h : ∀ x ∈ xs, ∀ y ∈ ys, ∀ z, f (f z x) y = f (f z y) x) (z : β) :
    (xs ++ ys).foldl f z = (ys ++ xs).foldl f z := by
  induction xs generalizing z with
  | nil => simp
  | cons x xs ih =>
    rw [List.cons_append, List.foldl_cons, ih (fun x' hx' => h x' (List.mem_cons_of_mem _ hx')),
      List.foldl_append, List.foldl_append, List.foldl_cons,
      foldl_comm_right f x ys (h x (List.mem_cons_self ..)) z]

/-- Folding over the concatenation of a permuted list of blocks, where blocks
related by a symmetric `R` commute elementwise and the list is `R`-pairwise. -/
theorem foldl_flatMap_perm {α β γ : Type} (f : β → α → β) (proj : γ → List α) (R : γ → γ → Prop)
    (hR : ∀ {a b}, R a b → R b a)
    (hcomm : ∀ a b, R a b → ∀ x ∈ proj a, ∀ y ∈ proj b, ∀ z, f (f z x) y = f (f z y) x)
    {gs gs' : List γ} (hp : gs.Perm gs') (hpw : gs.Pairwise R) (z : β) :
    (gs.flatMap proj).foldl f z = (gs'.flatMap proj).foldl f z := by
  induction hp generalizing z with
  | nil => rfl
  | cons x _ ih =>
    simp only [List.flatMap_cons, List.foldl_append]
    exact ih (List.pairwise_cons.1 hpw).2 _
  | swap x y l =>
    simp only [List.flatMap_cons, ← List.append_assoc]
    rw [List.foldl_append (l := proj y ++ proj x), List.foldl_append (l := proj x ++ proj y),
      foldl_append_comm f (proj y) (proj x)
        (hcomm y x ((List.pairwise_cons.1 hpw).1 x (List.mem_cons_self ..))) z]
  | trans p₁ _ ih₁ ih₂ =>
    rw [ih₁ hpw, ih₂ ((p₁.pairwise_iff hR).1 hpw)]

theorem mem_of_forall₂ {α β : Type} {S : α → β → Prop} {xs : List α} {ys : List β}
    (h : List.Forall₂ S xs ys) {x : α} (hx : x ∈ xs) : ∃ y ∈ ys, S x y := by
  induction h with
  | nil => simp at hx
  | cons hab _ ih =>
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, List.mem_cons_self .., hab⟩
    · obtain ⟨y, hy, hs⟩ := ih hx
      exact ⟨y, List.mem_cons_of_mem _ hy, hs⟩

/-- Pairwise relations transfer along a pointwise relation that maps related
pairs to related pairs. -/
theorem pairwise_of_forall₂ {α β : Type} {R : β → β → Prop} {R' : α → α → Prop} {S : α → β → Prop}
    (hS : ∀ a' a b' b, S a' a → S b' b → R a b → R' a' b')
    {xs : List α} {ys : List β} (h : List.Forall₂ S xs ys) (hp : ys.Pairwise R) : xs.Pairwise R' := by
  induction h with
  | nil => exact List.Pairwise.nil
  | cons hab _ ih =>
    rw [List.pairwise_cons] at hp ⊢
    refine ⟨fun b' hb' => ?_, ih hp.2⟩
    obtain ⟨b, hb, hs⟩ := mem_of_forall₂ (by assumption) hb'
    exact hS _ _ _ _ hab hs (hp.1 b hb)

end ListLemmas

open ListLemmas

/-! ## Per-key equality of lists -/

/-- Two lists agree key by key: they have the same entries, and entries with
the same key occur in the same order (entries with different keys may be
reordered). -/
def KeyEq {α κ : Type} [DecidableEq κ] (key : α → κ) (xs ys : List α) : Prop :=
  ∀ k, xs.filter (fun x => key x = k) = ys.filter (fun x => key x = k)

namespace KeyEq

variable {α κ : Type} [DecidableEq κ] {key : α → κ}

theorem refl (xs : List α) : KeyEq key xs xs := fun _ => rfl

theorem symm {xs ys : List α} (h : KeyEq key xs ys) : KeyEq key ys xs := fun k => (h k).symm

theorem trans {xs ys zs : List α} (h₁ : KeyEq key xs ys) (h₂ : KeyEq key ys zs) : KeyEq key xs zs :=
  fun k => (h₁ k).trans (h₂ k)

theorem append {xs xs' ys ys' : List α} (h₁ : KeyEq key xs ys) (h₂ : KeyEq key xs' ys') :
    KeyEq key (xs ++ xs') (ys ++ ys') := fun k => by
  rw [List.filter_append, List.filter_append, h₁ k, h₂ k]

theorem filter {xs ys : List α} (h : KeyEq key xs ys) (p : α → Bool) :
    KeyEq key (xs.filter p) (ys.filter p) := fun k => by
  rw [List.filter_filter, List.filter_filter]
  have e : (fun x => decide (key x = k) && p x) = fun x => p x && decide (key x = k) := by
    funext x; exact Bool.and_comm _ _
  have := congrArg (List.filter p) (h k)
  rw [List.filter_filter, List.filter_filter] at this
  rw [← e] at this
  exact this

theorem mem_iff {xs ys : List α} (h : KeyEq key xs ys) (x : α) : x ∈ xs ↔ x ∈ ys := by
  have hx := h (key x)
  constructor
  · intro hm
    have : x ∈ xs.filter (fun y => key y = key x) := List.mem_filter.2 ⟨hm, by simp⟩
    rw [hx] at this
    exact (List.mem_filter.1 this).1
  · intro hm
    have : x ∈ ys.filter (fun y => key y = key x) := List.mem_filter.2 ⟨hm, by simp⟩
    rw [← hx] at this
    exact (List.mem_filter.1 this).1

theorem contains_map {xs ys : List α} (h : KeyEq key xs ys) (k : κ) :
    (xs.map key).contains k = (ys.map key).contains k := by
  rw [Bool.eq_iff_iff]
  simp only [List.contains_iff_mem, List.mem_map]
  exact ⟨fun ⟨x, hx, e⟩ => ⟨x, (h.mem_iff x).1 hx, e⟩, fun ⟨x, hx, e⟩ => ⟨x, (h.mem_iff x).2 hx, e⟩⟩

theorem perm {xs ys : List α} (h : KeyEq key xs ys) : xs.Perm ys := by
  classical
  rw [List.perm_iff_count]
  intro a
  rw [← List.count_filter (p := fun x => decide (key x = key a)) (by simp), h (key a),
    List.count_filter (by simp)]

theorem flatMap {β : Type} {f g : β → List α} {bs : List β} (h : ∀ b ∈ bs, KeyEq key (f b) (g b)) :
    KeyEq key (bs.flatMap f) (bs.flatMap g) := by
  induction bs with
  | nil => exact refl _
  | cons b bs ih =>
    simp only [List.flatMap_cons]
    exact (h b (List.mem_cons_self ..)).append (ih fun b' hb' => h b' (List.mem_cons_of_mem _ hb'))

end KeyEq

/-! ## Sequential composition -/

namespace Log

theorem seq_assoc (a b c : Log) : (a.seq b).seq c = a.seq (b.seq c) := by
  simp only [seq, Log.mk.injEq, List.filter_append, List.filter_filter, List.append_assoc, List.map_append,
    List.contains_append, contains_fst_filter_or, Bool.not_or]
  refine ⟨trivial, ?_, ?_, trivial, trivial⟩ <;>
  · congr 1
    exact List.filter_congr fun _ _ => Bool.and_comm _ _

theorem empty_seq (g : Log) : Log.empty.seq g = g := by
  cases g; simp [seq, Log.empty]

theorem seq_writeLocs_contains (a b : Log) (l : Loc) :
    ((a.seq b).writes.map Prod.fst).contains l =
      ((a.writes.map Prod.fst).contains l || (b.writes.map Prod.fst).contains l) := by
  simp only [seq, List.map_append, List.contains_append, contains_fst_filter_or]

/-- The invariant `seq` maintains: at most one plain write per location. -/
def WF (g : Log) : Prop := (g.writes.map Prod.fst).Nodup

theorem WF.empty : Log.empty.WF := List.nodup_nil

theorem WF.seq {a b : Log} (ha : a.WF) (hb : b.WF) : (a.seq b).WF := by
  unfold WF Log.seq
  simp only [List.map_append]
  refine List.nodup_append.2 ⟨(List.Sublist.map Prod.fst List.filter_sublist).nodup ha, hb, ?_⟩
  intro l hl l' hl' heq
  subst heq
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
  have := (List.mem_filter.1 hx).2
  simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at this
  exact (by simpa using this : x.1 ∉ b.writes.map Prod.fst) hl'

/-! ## Conflicts -/

/-- One half of non-interference of two parallel logs; `Free` is its
symmetric closure. -/
structure HalfFree (a b : Log) : Prop where
  write : ∀ l ∈ a.writeLocs, l ∉ b.writeLocs ∧ l ∉ b.reads ∧ l ∉ b.wcrLocs
  wcrRead : ∀ l ∈ a.wcrLocs, l ∉ b.reads
  wcrOp : ∀ x ∈ a.wcrs, ∀ y ∈ b.wcrs, x.1 = y.1 → x.2.1 = y.2.1
  aliasName : ∀ x ∈ a.aliasSets, ∀ y ∈ b.aliasSets, x.1 ≠ y.1
  queue : ∀ l ∈ a.exclusiveQueues, l ∉ b.pushedQueues

/-- Two logs of parallel executions do not conflict (`conflicts_eq_false_iff`). -/
def Free (a b : Log) : Prop := HalfFree a b ∧ HalfFree b a

theorem conflicts_eq_false_iff (a b : Log) : conflicts a b = false ↔ Free a b := by
  simp only [conflicts, Bool.or_eq_false_iff, List.any_eq_false, Bool.or_eq_true, List.contains_iff_mem,
    List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq, not_or, not_exists, not_and, Prod.forall]
  constructor
  · rintro ⟨⟨⟨⟨⟨⟨⟨hw, hwb⟩, hop⟩, hcr⟩, hcrb⟩, hal⟩, hq⟩, hqb⟩
    refine ⟨⟨fun l hl => ⟨(hw l hl).1.1, (hw l hl).1.2, (hw l hl).2⟩, hcr, ?_, ?_, hq⟩,
      ⟨?_, hcrb, ?_, ?_, hqb⟩⟩
    · rintro ⟨l, op, v⟩ hx ⟨l', op', v'⟩ hy (rfl : l = l')
      exact not_not.1 (hop l op v hx l op' v' hy rfl)
    · rintro ⟨n, b⟩ hx ⟨n', b'⟩ hy
      exact hal n b hx n' b' hy
    · intro l hl
      exact ⟨fun h => (hw l h).1.1 hl, (hwb l hl).1, (hwb l hl).2⟩
    · rintro ⟨l, op, v⟩ hx ⟨l', op', v'⟩ hy (rfl : l = l')
      exact (not_not.1 (hop l op' v' hy l op v hx rfl)).symm
    · rintro ⟨n, b⟩ hx ⟨n', b'⟩ hy hn
      exact hal n' b' hy n b hx hn.symm
  · rintro ⟨⟨hw, hcr, hop, hal, hq⟩, ⟨hw', hcr', -, -, hq'⟩⟩
    refine ⟨⟨⟨⟨⟨⟨⟨fun l hl => ⟨⟨(hw l hl).1, (hw l hl).2.1⟩, (hw l hl).2.2⟩,
      fun l hl => ⟨(hw' l hl).2.1, (hw' l hl).2.2⟩⟩, ?_⟩, hcr⟩, hcr'⟩, ?_⟩, hq⟩, hq'⟩
    · intro l op v hx l' op' v' hy hl
      exact not_not.2 (hop (l, op, v) hx (l', op', v') hy hl)
    · intro n b hx n' b' hy
      exact hal (n, b) hx (n', b') hy

theorem Free.symm {a b : Log} (h : Free a b) : Free b a := ⟨h.2, h.1⟩

theorem conflicts_comm (a b : Log) : conflicts a b = conflicts b a := by
  cases h : conflicts b a
  · exact (conflicts_eq_false_iff a b).2 ((conflicts_eq_false_iff b a).1 h).symm
  · by_contra h'
    simp only [Bool.not_eq_true] at h'
    rw [(conflicts_eq_false_iff b a).2 ((conflicts_eq_false_iff a b).1 h').symm] at h
    exact Bool.false_ne_true h

theorem mem_exclusiveQueues {g : Log} {l : Loc} :
    l ∈ g.exclusiveQueues ↔ ∃ f, (l, QueueEffect.exclusive f) ∈ g.queues := by
  simp only [exclusiveQueues, List.mem_filterMap, Prod.exists]
  constructor
  · rintro ⟨l', e, he, h⟩
    cases e with
    | pushOnly => simp at h
    | exclusive f => simp only [Option.some.injEq] at h; subst h; exact ⟨f, he⟩
  · rintro ⟨f, hf⟩
    exact ⟨l, .exclusive f, hf, rfl⟩

theorem mem_pushedQueues {g : Log} {l : Loc} : l ∈ g.pushedQueues ↔ ∃ e, (l, e) ∈ g.queues := by
  simp [pushedQueues]

/-- Entry-wise inclusion of logs (a sub-log has no more conflicts). -/
structure Sub (a' a : Log) : Prop where
  reads : ∀ l ∈ a'.reads, l ∈ a.reads
  writes : ∀ x ∈ a'.writes, x ∈ a.writes
  wcrs : ∀ x ∈ a'.wcrs, x ∈ a.wcrs
  aliasSets : ∀ x ∈ a'.aliasSets, x ∈ a.aliasSets
  queues : ∀ x ∈ a'.queues, x ∈ a.queues

namespace Sub

theorem refl (a : Log) : Sub a a := ⟨fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => h⟩

theorem trans {a b c : Log} (h₁ : Sub a b) (h₂ : Sub b c) : Sub a c :=
  ⟨fun x hx => h₂.reads x (h₁.reads x hx), fun x hx => h₂.writes x (h₁.writes x hx),
    fun x hx => h₂.wcrs x (h₁.wcrs x hx), fun x hx => h₂.aliasSets x (h₁.aliasSets x hx),
    fun x hx => h₂.queues x (h₁.queues x hx)⟩

theorem writeLocs {a' a : Log} (h : Sub a' a) {l : Loc} (hl : l ∈ a'.writeLocs) : l ∈ a.writeLocs := by
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
  exact List.mem_map.2 ⟨x, h.writes x hx, rfl⟩

theorem wcrLocs {a' a : Log} (h : Sub a' a) {l : Loc} (hl : l ∈ a'.wcrLocs) : l ∈ a.wcrLocs := by
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
  exact List.mem_map.2 ⟨x, h.wcrs x hx, rfl⟩

theorem exclusiveQueues {a' a : Log} (h : Sub a' a) {l : Loc} (hl : l ∈ a'.exclusiveQueues) :
    l ∈ a.exclusiveQueues := by
  obtain ⟨f, hf⟩ := mem_exclusiveQueues.1 hl
  exact mem_exclusiveQueues.2 ⟨f, h.queues _ hf⟩

theorem pushedQueues {a' a : Log} (h : Sub a' a) {l : Loc} (hl : l ∈ a'.pushedQueues) :
    l ∈ a.pushedQueues := by
  obtain ⟨e, he⟩ := mem_pushedQueues.1 hl
  exact mem_pushedQueues.2 ⟨e, h.queues _ he⟩

end Sub

theorem HalfFree.mono {a' a b' b : Log} (ha : Sub a' a) (hb : Sub b' b) (h : HalfFree a b) : HalfFree a' b' where
  write l hl :=
    let ⟨h₁, h₂, h₃⟩ := h.write l (ha.writeLocs hl)
    ⟨fun h' => h₁ (hb.writeLocs h'), fun h' => h₂ (hb.reads l h'), fun h' => h₃ (hb.wcrLocs h')⟩
  wcrRead l hl h' := h.wcrRead l (ha.wcrLocs hl) (hb.reads l h')
  wcrOp x hx y hy := h.wcrOp x (ha.wcrs x hx) y (hb.wcrs y hy)
  aliasName x hx y hy := h.aliasName x (ha.aliasSets x hx) y (hb.aliasSets y hy)
  queue l hl h' := h.queue l (ha.exclusiveQueues hl) (hb.pushedQueues h')

theorem Free.mono {a' a b' b : Log} (ha : Sub a' a) (hb : Sub b' b) (h : Free a b) : Free a' b' :=
  ⟨h.1.mono ha hb, h.2.mono hb ha⟩

/-- Sub-logs (fewer reads, a subset of the effects) have no new conflicts. -/
theorem conflicts_mono {a' a b' b : Log} (ha : Sub a' a) (hb : Sub b' b) (h : conflicts a b = false) :
    conflicts a' b' = false :=
  (conflicts_eq_false_iff _ _).2 (((conflicts_eq_false_iff _ _).1 h).mono ha hb)

/-! ### Unions -/

theorem sub_unionAll {g : Log} {gs : List Log} (h : g ∈ gs) : Sub g (unionAll gs) where
  reads _ hx := List.mem_flatMap.2 ⟨g, h, hx⟩
  writes _ hx := List.mem_flatMap.2 ⟨g, h, hx⟩
  wcrs _ hx := List.mem_flatMap.2 ⟨g, h, hx⟩
  aliasSets _ hx := List.mem_flatMap.2 ⟨g, h, hx⟩
  queues _ hx := List.mem_flatMap.2 ⟨g, h, hx⟩

theorem unionAll_sub {gs : List Log} {a : Log} (h : ∀ g ∈ gs, Sub g a) : Sub (unionAll gs) a where
  reads x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).reads x hx
  writes x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).writes x hx
  wcrs x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).wcrs x hx
  aliasSets x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).aliasSets x hx
  queues x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).queues x hx

theorem seq_sub_unionAll (a b : Log) : Sub (a.seq b) (unionAll [a, b]) where
  reads x hx := by
    simp only [seq, List.mem_append, List.mem_filter] at hx
    simp only [unionAll, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_append]
    tauto
  writes x hx := by
    simp only [seq, List.mem_append, List.mem_filter] at hx
    simp only [unionAll, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_append]
    tauto
  wcrs x hx := by
    simp only [seq, List.mem_append, List.mem_filter] at hx
    simp only [unionAll, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_append]
    tauto
  aliasSets x hx := by simpa [seq, unionAll] using hx
  queues x hx := by simpa [seq, unionAll] using hx

theorem HalfFree.unionAll_right {a : Log} {gs : List Log} (h : ∀ g ∈ gs, HalfFree a g) :
    HalfFree a (unionAll gs) where
  write l hl := by
    refine ⟨fun h' => ?_, fun h' => ?_, fun h' => ?_⟩
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 h'
      obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
      exact ((h g hg).write _ hl).1 (List.mem_map.2 ⟨x, hx, rfl⟩)
    · obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 h'
      exact ((h g hg).write _ hl).2.1 hx
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 h'
      obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
      exact ((h g hg).write _ hl).2.2 (List.mem_map.2 ⟨x, hx, rfl⟩)
  wcrRead l hl h' := by
    obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 h'
    exact (h g hg).wcrRead l hl hx
  wcrOp x hx y hy := by
    obtain ⟨g, hg, hy⟩ := List.mem_flatMap.1 hy
    exact (h g hg).wcrOp x hx y hy
  aliasName x hx y hy := by
    obtain ⟨g, hg, hy⟩ := List.mem_flatMap.1 hy
    exact (h g hg).aliasName x hx y hy
  queue l hl h' := by
    obtain ⟨e, he⟩ := mem_pushedQueues.1 h'
    obtain ⟨g, hg, he⟩ := List.mem_flatMap.1 he
    exact (h g hg).queue l hl (mem_pushedQueues.2 ⟨e, he⟩)

theorem HalfFree.unionAll_left {a : Log} {gs : List Log} (h : ∀ g ∈ gs, HalfFree g a) :
    HalfFree (unionAll gs) a where
  write l hl := by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
    obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
    exact (h g hg).write _ (List.mem_map.2 ⟨x, hx, rfl⟩)
  wcrRead l hl := by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
    obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
    exact (h g hg).wcrRead _ (List.mem_map.2 ⟨x, hx, rfl⟩)
  wcrOp x hx y hy := by
    obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
    exact (h g hg).wcrOp x hx y hy
  aliasName x hx y hy := by
    obtain ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx
    exact (h g hg).aliasName x hx y hy
  queue l hl := by
    obtain ⟨f, hf⟩ := mem_exclusiveQueues.1 hl
    obtain ⟨g, hg, hf⟩ := List.mem_flatMap.1 hf
    exact (h g hg).queue l (mem_exclusiveQueues.2 ⟨f, hf⟩)

theorem free_unionAll_iff {a : Log} {gs : List Log} : Free a (unionAll gs) ↔ ∀ g ∈ gs, Free a g :=
  ⟨fun h _ hg => h.mono (Sub.refl a) (sub_unionAll hg),
    fun h => ⟨HalfFree.unionAll_right fun g hg => (h g hg).1, HalfFree.unionAll_left fun g hg => (h g hg).2⟩⟩

theorem conflicts_unionAll {a : Log} {gs : List Log} :
    conflicts a (unionAll gs) = false ↔ ∀ g ∈ gs, conflicts a g = false := by
  simp only [conflicts_eq_false_iff]
  exact free_unionAll_iff

/-! ### Pairwise freedom -/

theorem pairwiseFree_iff_pairwise (gs : List Log) :
    pairwiseFree gs = true ↔ gs.Pairwise (fun a b => conflicts a b = false) := by
  induction gs with
  | nil => simp [pairwiseFree]
  | cons g gs ih =>
    simp only [pairwiseFree, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, ih,
      List.pairwise_cons]

theorem pairwiseFree_iff_pairwise_free (gs : List Log) : pairwiseFree gs = true ↔ gs.Pairwise Free := by
  rw [pairwiseFree_iff_pairwise]
  simp only [conflicts_eq_false_iff]

theorem pairwiseFree_cons {g : Log} {gs : List Log} :
    pairwiseFree (g :: gs) = true ↔ conflicts g (unionAll gs) = false ∧ pairwiseFree gs = true := by
  rw [pairwiseFree_iff_pairwise, List.pairwise_cons, ← pairwiseFree_iff_pairwise, conflicts_unionAll]

theorem pairwiseFree_perm {gs gs' : List Log} (h : gs.Perm gs') : pairwiseFree gs = pairwiseFree gs' := by
  rw [Bool.eq_iff_iff, pairwiseFree_iff_pairwise, pairwiseFree_iff_pairwise]
  exact h.pairwise_iff fun {a b} (hab : conflicts a b = false) => (conflicts_comm b a).trans hab

/-- A pointwise sub-list of a conflict-free list is conflict free. -/
theorem pairwiseFree_of_sub {gs' gs : List Log} (h : List.Forall₂ Sub gs' gs) (hf : pairwiseFree gs = true) :
    pairwiseFree gs' = true := by
  rw [pairwiseFree_iff_pairwise] at hf ⊢
  exact pairwise_of_forall₂ (R := fun a b => conflicts a b = false) (R' := fun a b => conflicts a b = false)
    (fun _ _ _ _ ha hb hc => conflicts_mono ha hb hc) h hf

end Log

/-! ## The merge phases, location by location -/

namespace Merge

/-- Effect of one queue entry on the queue content. -/
def qstep (q : List Val) : QueueEffect → List Val
  | .pushOnly vs => q ++ vs
  | .exclusive final => final

def writeCells (c : Loc → Val) (ws : List (Loc × Val)) : Loc → Val :=
  ws.foldl (fun c x => Function.update c x.1 x.2) c

def wcrCells (w : WcrSem) (c : Loc → Val) (cs : List (Loc × WcrOp × Val)) : Loc → Val :=
  cs.foldl (fun c x => Function.update c x.1 (w x.2.1 (c x.1) x.2.2)) c

def aliasTable (t : Name → Option AliasBinding) (as : List (Name × AliasBinding)) :
    Name → Option AliasBinding :=
  as.foldl (fun t x => Function.update t x.1 (some x.2)) t

def queueTable (q : Loc → List Val) (qs : List (Loc × QueueEffect)) : Loc → List Val :=
  qs.foldl (fun q x => Function.update q x.1 (qstep (q x.1) x.2)) q

/-- Fold WCR contributions (to one location) onto a value. -/
def wcrFold (w : WcrSem) (v : Val) (cs : List (Loc × WcrOp × Val)) : Val :=
  cs.foldl (fun acc x => w x.2.1 acc x.2.2) v

theorem applyWrites_eq (m : Mem) (ws : List (Loc × Val)) :
    applyWrites m ws = { m with cells := writeCells m.cells ws } := by
  induction ws generalizing m with
  | nil => rfl
  | cons x ws ih => exact ih _

theorem applyWcrs_eq (w : WcrSem) (m : Mem) (cs : List (Loc × WcrOp × Val)) :
    applyWcrs w m cs = { m with cells := wcrCells w m.cells cs } := by
  induction cs generalizing m with
  | nil => rfl
  | cons x cs ih => exact ih _

theorem applyAliases_eq (m : Mem) (as : List (Name × AliasBinding)) :
    applyAliases m as = { m with aliases := aliasTable m.aliases as } := by
  induction as generalizing m with
  | nil => rfl
  | cons x as ih => exact ih _

theorem applyQueues_eq (m : Mem) (qs : List (Loc × QueueEffect)) :
    applyQueues m qs = { m with queues := queueTable m.queues qs } := by
  induction qs generalizing m with
  | nil => rfl
  | cons x qs ih =>
    obtain ⟨l, e⟩ := x
    cases e <;> exact ih _

theorem writeCells_eval (c : Loc → Val) (ws : List (Loc × Val)) (l : Loc) :
    writeCells c ws l = (ws.filter (fun x => x.1 = l)).foldl (fun _ x => x.2) (c l) :=
  foldl_update_eval Prod.fst (fun _ x => x.2) ws c l

theorem wcrCells_eval (w : WcrSem) (c : Loc → Val) (cs : List (Loc × WcrOp × Val)) (l : Loc) :
    wcrCells w c cs l = wcrFold w (c l) (cs.filter (fun x => x.1 = l)) :=
  foldl_update_eval Prod.fst (fun (acc : Val) (x : Loc × WcrOp × Val) => w x.2.1 acc x.2.2) cs c l

theorem aliasTable_eval (t : Name → Option AliasBinding) (as : List (Name × AliasBinding)) (n : Name) :
    aliasTable t as n = (as.filter (fun x => x.1 = n)).foldl (fun _ x => some x.2) (t n) :=
  foldl_update_eval Prod.fst (fun _ x => some x.2) as t n

theorem queueTable_eval (q : Loc → List Val) (qs : List (Loc × QueueEffect)) (l : Loc) :
    queueTable q qs l = (qs.filter (fun x => x.1 = l)).foldl (fun q x => qstep q x.2) (q l) :=
  foldl_update_eval Prod.fst (fun q x => qstep q x.2) qs q l

theorem wcrFold_append (w : WcrSem) (v : Val) (xs ys : List (Loc × WcrOp × Val)) :
    wcrFold w v (xs ++ ys) = wcrFold w (wcrFold w v xs) ys := List.foldl_append

/-- Contributions with one operator can be folded in any order. -/
theorem wcrFold_perm (w : WcrSem) (hl : w.Lawful) (v : Val) {xs ys : List (Loc × WcrOp × Val)}
    (hp : xs.Perm ys) (hop : ∀ x ∈ xs, ∀ y ∈ xs, x.2.1 = y.2.1) : wcrFold w v xs = wcrFold w v ys := by
  refine hp.foldl_eq' (fun x hx y hy z => ?_) v
  rw [hop x hx y hy]
  exact hl.comm_assoc _ _ _ _

end Merge

open Merge

namespace Log

/-- Apply the effects of one log to a memory: plain writes, then WCR
contributions, alias bindings and queue effects (`merge` of a conflict-free
list applies the union of its logs, `merge_eq_apply`). -/
def apply (g : Log) (w : WcrSem) (m : Mem) : Mem :=
  Merge.applyQueues (Merge.applyAliases (Merge.applyWcrs w (Merge.applyWrites m g.writes) g.wcrs) g.aliasSets)
    g.queues

theorem apply_eq (g : Log) (w : WcrSem) (m : Mem) :
    g.apply w m = { cells := wcrCells w (writeCells m.cells g.writes) g.wcrs,
                    queues := queueTable m.queues g.queues,
                    aliases := aliasTable m.aliases g.aliasSets } := by
  simp only [apply, applyWrites_eq, applyWcrs_eq, applyAliases_eq, applyQueues_eq]

theorem apply_empty (w : WcrSem) (m : Mem) : Log.empty.apply w m = m := rfl

/-- Applying a sequential composition applies its parts in order (plain writes
and contributions that `seq` drops are overwritten by the second log). -/
theorem apply_seq (w : WcrSem) (a b : Log) (m : Mem) : (a.seq b).apply w m = b.apply w (a.apply w m) := by
  simp only [apply_eq, seq, aliasTable, queueTable, List.foldl_append, Mem.mk.injEq, and_true]
  funext l
  simp only [wcrCells_eval, writeCells_eval, List.filter_append, wcrFold, List.foldl_append,
    filter_key_filter_not_contains]
  cases h : (b.writes.map Prod.fst).contains l
  · have : b.writes.filter (fun x => x.1 = l) = [] := by
      rw [List.filter_eq_nil_iff]
      intro x hx hxl
      simp only [decide_eq_true_eq] at hxl
      subst hxl
      have hn : x.1 ∉ b.writes.map Prod.fst := by simpa using h
      exact hn (List.mem_map.2 ⟨x, hx, rfl⟩)
    simp [this]
  · simp only [ite_true, List.foldl_nil]
    congr 1
    exact foldl_last_indep _ _ h _ _

/-- Under no conflict, the union of a log and a list of logs is their
sequential composition. -/
theorem unionAll_cons_eq_seq {g : Log} {gs : List Log} (h : conflicts g (unionAll gs) = false) :
    unionAll (g :: gs) = g.seq (unionAll gs) := by
  obtain ⟨h₁, h₂⟩ := (conflicts_eq_false_iff _ _).1 h
  have hu : ∀ {β : Type} (f : Log → List β), (g :: gs).flatMap f = f g ++ gs.flatMap f := fun _ => rfl
  simp only [unionAll, seq, hu, Log.mk.injEq]
  refine ⟨?_, ?_, ?_, trivial, trivial⟩
  · congr 1
    symm
    rw [List.filter_eq_self]
    intro l hl
    have : l ∉ g.writeLocs := fun hc => (h₁.write l hc).2.1 hl
    simpa [writeLocs] using this
  · congr 1
    symm
    rw [List.filter_eq_self]
    intro x hx
    have : x.1 ∉ (unionAll gs).writeLocs := fun hc => (h₂.write x.1 hc).1 (List.mem_map.2 ⟨x, hx, rfl⟩)
    simpa [writeLocs, unionAll] using this
  · congr 1
    symm
    rw [List.filter_eq_self]
    intro x hx
    have : x.1 ∉ (unionAll gs).writeLocs := fun hc => (h₂.write x.1 hc).2.2 (List.mem_map.2 ⟨x, hx, rfl⟩)
    simpa [writeLocs, unionAll] using this

end Log

/-! ## Merge -/

theorem merge_eq_apply (w : WcrSem) (m : Mem) {gs : List Log} (h : Log.pairwiseFree gs = true) :
    merge w m gs = .ok ((Log.unionAll gs).apply w m) := by
  simp [merge, h, Log.apply]

theorem merge_isOk_iff (w : WcrSem) (m : Mem) (gs : List Log) :
    (∃ m', merge w m gs = .ok m') ↔ Log.pairwiseFree gs = true := by
  constructor
  · rintro ⟨m', hm⟩
    by_contra h
    simp only [merge, h, ite_false] at hm
    repeat' split at hm
    all_goals simp at hm
  · intro h
    exact ⟨_, merge_eq_apply w m h⟩

theorem merge_singleton (w : WcrSem) (m : Mem) (g : Log) : merge w m [g] = .ok (g.apply w m) := by
  rw [merge_eq_apply w m (by simp [Log.pairwiseFree])]
  cases g
  simp [Log.unionAll]

/-- A conflict-free merge applies the first log and merges the rest onto the
result: component merge is sequential application (in any order, `merge_perm`). -/
theorem merge_cons_seq (w : WcrSem) (m : Mem) {g : Log} {gs : List Log}
    (h : Log.pairwiseFree (g :: gs) = true) : merge w m (g :: gs) = merge w (g.apply w m) gs := by
  obtain ⟨hc, hf⟩ := Log.pairwiseFree_cons.1 h
  rw [merge_eq_apply w m h, merge_eq_apply w _ hf, Log.unionAll_cons_eq_seq hc, Log.apply_seq]

/-- Merging a conflict-free list is applying its logs one after the other. -/
theorem merge_eq_foldl (w : WcrSem) (m : Mem) {gs : List Log} (h : Log.pairwiseFree gs = true) :
    merge w m gs = .ok (gs.foldl (fun m g => g.apply w m) m) := by
  induction gs generalizing m with
  | nil => rfl
  | cons g gs ih =>
    rw [merge_cons_seq w m h, ih _ (Log.pairwiseFree_cons.1 h).2]
    rfl

namespace Log

/-- No queue is touched by both logs (pushes of parallel logs to one queue are
appended in list order, `Decisions.md#D25`, so their merge depends on the order). -/
def QueuesDisjoint (a b : Log) : Prop := ∀ l ∈ a.pushedQueues, l ∉ b.pushedQueues

theorem QueuesDisjoint.symm {a b : Log} (h : QueuesDisjoint a b) : QueuesDisjoint b a :=
  fun l hb ha => h l ha hb

end Log

/-- For a conflict-free list in which no two logs touch the same queue, the
merge does not depend on the order of the logs (WCR semantics lawful). -/
theorem merge_perm (w : WcrSem) (hl : w.Lawful) (m : Mem) {gs gs' : List Log} (hp : gs.Perm gs')
    (hf : Log.pairwiseFree gs = true) (hq : gs.Pairwise Log.QueuesDisjoint) :
    merge w m gs = merge w m gs' := by
  rw [merge_eq_apply w m hf, merge_eq_apply w m ((Log.pairwiseFree_perm hp) ▸ hf)]
  have hpw := (Log.pairwiseFree_iff_pairwise_free gs).1 hf
  have hsymm : ∀ {a b : Log}, Log.Free a b → Log.Free b a := fun h => h.symm
  simp only [Log.apply_eq, Log.unionAll, Except.ok.injEq, Mem.mk.injEq]
  refine ⟨?_, ?_, ?_⟩
  · -- cells: writes, then contributions
    have hw : writeCells m.cells (gs.flatMap (·.writes)) = writeCells m.cells (gs'.flatMap (·.writes)) := by
      refine foldl_flatMap_perm _ _ Log.Free hsymm (fun a b hab x hx y hy z => ?_) hp hpw _
      have hne : x.1 ≠ y.1 := fun e =>
        (hab.1.write x.1 (List.mem_map.2 ⟨x, hx, rfl⟩)).1 (e ▸ List.mem_map.2 ⟨y, hy, rfl⟩)
      exact update_step_comm Prod.fst (fun _ x => x.2) x y hne z
    rw [hw]
    refine foldl_flatMap_perm _ _ Log.Free hsymm (fun a b hab x hx y hy z => ?_) hp hpw _
    by_cases hxy : x.1 = y.1
    · obtain ⟨lx, opx, vx⟩ := x
      obtain ⟨ly, opy, vy⟩ := y
      simp only at hxy
      subst hxy
      have hop : opx = opy := hab.1.wcrOp _ hx _ hy rfl
      subst hop
      simp only [Function.update_self, Function.update_idem]
      rw [hl.comm_assoc]
    · exact update_step_comm Prod.fst (fun (acc : Val) (x : Loc × WcrOp × Val) => w x.2.1 acc x.2.2) x y hxy z
  · -- queues
    have hpq : gs.Pairwise (fun a b => Log.Free a b ∧ Log.QueuesDisjoint a b) :=
      List.pairwise_and_iff.2 ⟨hpw, hq⟩
    refine foldl_flatMap_perm _ _ _ (fun h => ⟨h.1.symm, h.2.symm⟩) (fun a b hab x hx y hy z => ?_) hp hpq _
    have hne : x.1 ≠ y.1 := fun e =>
      hab.2 x.1 (Log.mem_pushedQueues.2 ⟨x.2, hx⟩) (e ▸ Log.mem_pushedQueues.2 ⟨y.2, hy⟩)
    exact update_step_comm Prod.fst (fun q x => qstep q x.2) x y hne z
  · -- aliases
    refine foldl_flatMap_perm _ _ Log.Free hsymm (fun a b hab x hx y hy z => ?_) hp hpw _
    exact update_step_comm Prod.fst (fun _ x => some x.2) x y (hab.1.aliasName x hx y hy) z

/-! ## Private names -/

namespace Log

private theorem bool_aux (p q : Bool) : (!p && !q) = (!(q && !p) && !p) := by
  cases p <;> cases q <;> rfl

theorem dropNames_seq (a b : Log) (P : List Name) :
    (a.seq b).dropNames P = (a.dropNames P).seq (b.dropNames P) := by
  simp only [seq, dropNames, Log.mk.injEq, List.filter_append, List.filter_filter,
    contains_fst_filter_and (p := Loc.name) (P := fun n => !P.contains n)]
  refine ⟨?_, ?_, ?_, trivial, trivial⟩ <;>
  · congr 1
    apply List.filter_congr
    intro x _
    exact bool_aux _ _

theorem dropNames_unionAll (gs : List Log) (P : List Name) :
    (unionAll gs).dropNames P = unionAll (gs.map (·.dropNames P)) := by
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    simp only [unionAll, dropNames, List.flatMap_cons, List.map_cons, List.filter_append, Log.mk.injEq] at ih ⊢
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih
    exact ⟨by rw [h1], by rw [h2], by rw [h3], by rw [h4], by rw [h5]⟩

theorem dropNames_sub (g : Log) (P : List Name) : Sub (g.dropNames P) g where
  reads _ h := (List.mem_filter.1 h).1
  writes _ h := (List.mem_filter.1 h).1
  wcrs _ h := (List.mem_filter.1 h).1
  aliasSets _ h := (List.mem_filter.1 h).1
  queues _ h := (List.mem_filter.1 h).1

theorem dropNames_nil (g : Log) : g.dropNames [] = g := by
  cases g; simp [dropNames]

theorem dropNames_dropNames (g : Log) (P Q : List Name) :
    (g.dropNames P).dropNames Q = g.dropNames (P ++ Q) := by
  simp only [dropNames, List.filter_filter, List.contains_append, Bool.not_or, Log.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
  · apply List.filter_congr
    intro x _
    exact Bool.and_comm _ _

/-- Dropping more names gives a smaller log. -/
theorem dropNames_mono {P Q : List Name} (h : ∀ n ∈ P, n ∈ Q) (g : Log) :
    Sub (g.dropNames Q) (g.dropNames P) := by
  have key : ∀ n, (!Q.contains n) = true → (!P.contains n) = true := by
    intro n hn
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at hn ⊢
    have hq : n ∉ Q := by simpa using hn
    have hp : n ∉ P := fun hp => hq (h n hp)
    simpa using hp
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
  · intro x hx
    obtain ⟨hx, hq⟩ := List.mem_filter.1 hx
    exact List.mem_filter.2 ⟨hx, key _ hq⟩

theorem WF.dropNames {g : Log} (h : g.WF) (P : List Name) : (g.dropNames P).WF :=
  (List.Sublist.map Prod.fst List.filter_sublist).nodup h

/-! ## Union and the refinement order on logs -/

/-- Entry-wise concatenation of two logs (the union of parallel logs). -/
def union (a b : Log) : Log :=
  { reads := a.reads ++ b.reads, writes := a.writes ++ b.writes, wcrs := a.wcrs ++ b.wcrs,
    aliasSets := a.aliasSets ++ b.aliasSets, queues := a.queues ++ b.queues }

theorem unionAll_cons (g : Log) (gs : List Log) : unionAll (g :: gs) = g.union (unionAll gs) := rfl

/-- Sequential composition of non-conflicting logs is their union. -/
theorem seq_eq_union {a b : Log} (h : conflicts a b = false) : a.seq b = a.union b := by
  have h' : conflicts a (unionAll [b]) = false := conflicts_unionAll.2 (by simpa using h)
  have e := (unionAll_cons_eq_seq h').symm
  have hb : unionAll [b] = b := by cases b; simp [unionAll]
  rw [hb] at e
  rw [e, unionAll_cons, hb]

/-- `g'` refines `g` as a log of the same execution: it reads at most what `g`
reads and has the same effects, up to reordering entries that commute. Entries
with different keys commute; WCR contributions to one location only need to
fold to the same value under every lawful WCR semantics. Logs ordered by `Le`
have no more conflicts (`Le.sub`) and the same merge (`merge_le`). -/
structure Le (g' g : Log) : Prop where
  reads : ∀ l ∈ g'.reads, l ∈ g.reads
  writes : KeyEq Prod.fst g'.writes g.writes
  wcrs : g'.wcrs.Perm g.wcrs
  wcrFold : ∀ w : WcrSem, w.Lawful → ∀ l v,
    Merge.wcrFold w v (g'.wcrs.filter (fun x => x.1 = l)) = Merge.wcrFold w v (g.wcrs.filter (fun x => x.1 = l))
  aliasSets : KeyEq Prod.fst g'.aliasSets g.aliasSets
  queues : KeyEq Prod.fst g'.queues g.queues

namespace Le

theorem refl (g : Log) : Le g g :=
  ⟨fun _ h => h, KeyEq.refl _, List.Perm.refl _, fun _ _ _ _ => rfl, KeyEq.refl _, KeyEq.refl _⟩

theorem trans {a b c : Log} (h₁ : Le a b) (h₂ : Le b c) : Le a c :=
  ⟨fun l hl => h₂.reads l (h₁.reads l hl), h₁.writes.trans h₂.writes, h₁.wcrs.trans h₂.wcrs,
    fun w hl l v => (h₁.wcrFold w hl l v).trans (h₂.wcrFold w hl l v), h₁.aliasSets.trans h₂.aliasSets,
    h₁.queues.trans h₂.queues⟩

theorem sub {g' g : Log} (h : Le g' g) : Sub g' g :=
  ⟨h.reads, fun x hx => (h.writes.mem_iff x).1 hx, fun _ hx => h.wcrs.subset hx,
    fun x hx => (h.aliasSets.mem_iff x).1 hx, fun x hx => (h.queues.mem_iff x).1 hx⟩

theorem wf_iff {g' g : Log} (h : Le g' g) : g'.WF ↔ g.WF :=
  (h.writes.perm.map Prod.fst).nodup_iff

theorem union {a' a b' b : Log} (ha : Le a' a) (hb : Le b' b) : Le (a'.union b') (a.union b) where
  reads l hl := by
    rcases List.mem_append.1 hl with hl | hl
    · exact List.mem_append_left _ (ha.reads l hl)
    · exact List.mem_append_right _ (hb.reads l hl)
  writes := ha.writes.append hb.writes
  wcrs := ha.wcrs.append hb.wcrs
  wcrFold w hl l v := by
    simp only [Log.union, List.filter_append, Merge.wcrFold_append, ha.wcrFold w hl, hb.wcrFold w hl]
  aliasSets := ha.aliasSets.append hb.aliasSets
  queues := ha.queues.append hb.queues

theorem unionAll {gs' gs : List Log} (h : List.Forall₂ Le gs' gs) : Le (Log.unionAll gs') (Log.unionAll gs) := by
  induction h with
  | nil => exact refl _
  | cons hab _ ih =>
    rw [unionAll_cons, unionAll_cons]
    exact hab.union ih

theorem seq {a' a b' b : Log} (ha : Le a' a) (hb : Le b' b) : Le (a'.seq b') (a.seq b) := by
  have ea : ∀ l, (a'.writes.map Prod.fst).contains l = (a.writes.map Prod.fst).contains l :=
    ha.writes.contains_map
  have eb : ∀ l, (b'.writes.map Prod.fst).contains l = (b.writes.map Prod.fst).contains l :=
    hb.writes.contains_map
  refine ⟨?_, ?_, ?_, ?_, ha.aliasSets.append hb.aliasSets, ha.queues.append hb.queues⟩
  · intro l hl
    simp only [Log.seq, List.mem_append, List.mem_filter, ea] at hl ⊢
    rcases hl with hl | ⟨hl, hc⟩
    · exact Or.inl (ha.reads l hl)
    · exact Or.inr ⟨hb.reads l hl, hc⟩
  · simp only [Log.seq, eb]
    exact (ha.writes.filter _).append hb.writes
  · simp only [Log.seq, eb]
    exact (ha.wcrs.filter _).append hb.wcrs
  · intro w hl l v
    simp only [Log.seq, List.filter_append, filter_key_filter_not_contains, eb, Merge.wcrFold_append]
    split
    · exact hb.wcrFold w hl l _
    · rw [ha.wcrFold w hl, hb.wcrFold w hl]

theorem dropNames {g' g : Log} (h : Le g' g) (P : List Name) : Le (g'.dropNames P) (g.dropNames P) where
  reads l hl := by
    obtain ⟨hl, hp⟩ := List.mem_filter.1 hl
    exact List.mem_filter.2 ⟨h.reads l hl, hp⟩
  writes := h.writes.filter _
  wcrs := h.wcrs.filter _
  wcrFold w hl l v := by
    simp only [Log.dropNames]
    rw [filter_key_filter_const Prod.fst g'.wcrs (fun l => !P.contains l.name),
      filter_key_filter_const Prod.fst g.wcrs (fun l => !P.contains l.name)]
    split
    · exact h.wcrFold w hl l v
    · rfl
  aliasSets := h.aliasSets.filter _
  queues := h.queues.filter _

/-- Logs ordered by `Le` have the same effect on memory. -/
theorem apply_eq {g' g : Log} (h : Le g' g) (w : WcrSem) (hl : w.Lawful) (m : Mem) :
    g'.apply w m = g.apply w m := by
  rw [Log.apply_eq, Log.apply_eq]
  congr 1
  · funext l
    rw [wcrCells_eval, wcrCells_eval, writeCells_eval, writeCells_eval, h.writes l, h.wcrFold w hl]
  · funext l
    rw [queueTable_eval, queueTable_eval, h.queues l]
  · funext n
    rw [aliasTable_eval, aliasTable_eval, h.aliasSets n]

end Le

theorem KeyEq.append_comm_of_disjoint {α κ : Type} [DecidableEq κ] {key : α → κ} {xs ys : List α}
    (h : ∀ x ∈ xs, ∀ y ∈ ys, key x ≠ key y) : KeyEq key (xs ++ ys) (ys ++ xs) := fun k => by
  rw [List.filter_append, List.filter_append]
  by_cases hx : xs.filter (fun x => key x = k) = []
  · rw [hx]; simp
  · have hy : ys.filter (fun x => key x = k) = [] := by
      rw [List.filter_eq_nil_iff]
      intro y hy hyk
      obtain ⟨x, hxm⟩ := List.exists_mem_of_ne_nil _ hx
      obtain ⟨hxm, hxk⟩ := List.mem_filter.1 hxm
      simp only [decide_eq_true_eq] at hxk hyk
      exact h x hxm y hy (hxk.trans hyk.symm)
    rw [hy]; simp

/-- Two non-conflicting logs that touch disjoint queues can be composed in
either order (up to `Le`; the reads are the same set). Lift to a common prefix
with `seq_assoc` and `Le.seq (Le.refl _)`. -/
theorem Le.seq_comm {a b : Log} (h : conflicts a b = false) (hq : QueuesDisjoint a b) :
    Le (a.seq b) (b.seq a) := by
  have hf := (conflicts_eq_false_iff a b).1 h
  rw [seq_eq_union h, seq_eq_union ((conflicts_comm b a).trans h)]
  refine ⟨?_, ?_, List.perm_append_comm, ?_, ?_, ?_⟩
  · intro l hl
    simp only [Log.union, List.mem_append] at hl ⊢
    exact hl.symm
  · exact KeyEq.append_comm_of_disjoint fun x hx y hy e =>
      (hf.1.write x.1 (List.mem_map.2 ⟨x, hx, rfl⟩)).1 (e ▸ List.mem_map.2 ⟨y, hy, rfl⟩)
  · intro w hl l v
    simp only [Log.union, List.filter_append]
    by_cases ha : a.wcrs.filter (fun x => x.1 = l) = []
    · rw [ha]; simp
    by_cases hb : b.wcrs.filter (fun x => x.1 = l) = []
    · rw [hb]; simp
    obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hb
    obtain ⟨hy, hyl⟩ := List.mem_filter.1 hy
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ ha
    obtain ⟨hx, hxl⟩ := List.mem_filter.1 hx
    simp only [decide_eq_true_eq] at hxl hyl
    -- every contribution to `l` uses the operator of `y`
    have hop : ∀ z ∈ a.wcrs.filter (fun x => x.1 = l) ++ b.wcrs.filter (fun x => x.1 = l), z.2.1 = y.2.1 := by
      intro z hz
      rcases List.mem_append.1 hz with hz | hz
      · obtain ⟨hz, hzl⟩ := List.mem_filter.1 hz
        simp only [decide_eq_true_eq] at hzl
        exact hf.1.wcrOp z hz y hy (hzl.trans hyl.symm)
      · obtain ⟨hz, hzl⟩ := List.mem_filter.1 hz
        simp only [decide_eq_true_eq] at hzl
        exact (hf.1.wcrOp x hx z hz (hxl.trans hzl.symm)).symm.trans (hf.1.wcrOp x hx y hy (hxl.trans hyl.symm))
    exact Merge.wcrFold_perm w hl v List.perm_append_comm
      fun z hz z' hz' => (hop z hz).trans (hop z' hz').symm
  · exact KeyEq.append_comm_of_disjoint fun x hx y hy => hf.1.aliasName x hx y hy
  · exact KeyEq.append_comm_of_disjoint fun x hx y hy e =>
      hq x.1 (mem_pushedQueues.2 ⟨x.2, hx⟩) (e ▸ mem_pushedQueues.2 ⟨y.2, hy⟩)

end Log

/-- Merge respects `Log.Le` pointwise: if the original list is conflict free,
so is the refined one, with the same result. -/
theorem merge_le (w : WcrSem) (hl : w.Lawful) (m : Mem) {gs' gs : List Log} (h : List.Forall₂ Log.Le gs' gs)
    (hf : Log.pairwiseFree gs = true) : Log.pairwiseFree gs' = true ∧ merge w m gs' = merge w m gs := by
  have hf' := Log.pairwiseFree_of_sub (h.imp fun _ _ => Log.Le.sub) hf
  exact ⟨hf', by rw [merge_eq_apply w m hf', merge_eq_apply w m hf, (Log.Le.unionAll h).apply_eq w hl]⟩

/-- A conflict in the refined list is a conflict in the original one. -/
theorem merge_le_conflict {gs' gs : List Log} (h : List.Forall₂ Log.Le gs' gs)
    (hc : Log.pairwiseFree gs' = false) : Log.pairwiseFree gs = false := by
  cases hf : Log.pairwiseFree gs
  · rfl
  · rw [Log.pairwiseFree_of_sub (h.imp fun _ _ => Log.Le.sub) hf] at hc
    exact hc

/-! ## Coherence of one-entry logs with the local view

Each memory access of a sequential execution updates its local view and logs
one entry (`St.write`, `St.wcr`, `St.push`, `St.popN`, `St.bindAlias`); applying
that entry to the view before the access gives the view after it. With
`Log.apply_seq` this makes the local view the initial memory with the log
applied. -/

theorem Mem.pushMany_eq (m : Mem) (l : Loc) (vs : List Val) :
    m.pushMany l vs =
      { m with queues := Function.update m.queues (m.resolve l) (m.queues (m.resolve l) ++ vs) } := by
  induction vs generalizing m with
  | nil =>
    simp only [Mem.pushMany, List.foldl_nil, List.append_nil, Function.update_eq_self]
  | cons v vs ih =>
    have e : (m.push l v).resolve l = m.resolve l := by
      simp only [Mem.push, Mem.resolve, Mem.resolveFuel_set_queues]
    show (m.push l v).pushMany l vs = _
    rw [ih, e]
    simp only [Mem.push, Function.update_self, Function.update_idem, List.append_assoc, List.singleton_append]

namespace Log

theorem apply_read (w : WcrSem) (m : Mem) (l : Loc) : ({ reads := [l] } : Log).apply w m = m := rfl

theorem apply_write (w : WcrSem) (m : Mem) (l : Loc) (v : Val) :
    ({ writes := [(m.resolve l, v)] } : Log).apply w m = m.write l v := rfl

theorem apply_wcr (w : WcrSem) (m : Mem) (l : Loc) (op : WcrOp) (v : Val) :
    ({ wcrs := [(m.resolve l, op, v)] } : Log).apply w m = m.write l (w op (m.read l) v) := by
  simp only [apply_eq, Mem.write, Mem.read, wcrCells, writeCells, queueTable, aliasTable, List.foldl_cons,
    List.foldl_nil]

theorem apply_push (w : WcrSem) (m : Mem) (l : Loc) (vs : List Val) :
    ({ queues := [(m.resolve l, .pushOnly vs)] } : Log).apply w m = m.pushMany l vs := by
  rw [Mem.pushMany_eq]
  rfl

theorem apply_exclusive (w : WcrSem) (m : Mem) (r : Loc) (rest : List Val) :
    ({ queues := [(r, .exclusive rest)] } : Log).apply w m =
      { m with queues := Function.update m.queues r rest } := rfl

theorem apply_alias (w : WcrSem) (m : Mem) (n : Name) (b : AliasBinding) :
    ({ aliasSets := [(n, b)] } : Log).apply w m = m.setAlias n b := rfl

end Log

end Dace
