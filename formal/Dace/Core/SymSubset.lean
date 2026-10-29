-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Data
import Dace.Core.SymSearch

/-!
# Verified symbolic reasoning about subsets

Sound, incomplete decision procedures over subsets, relative to the symbol
facts (`Constraints.satisfies`):

* `Expr.nonneg?`, `Expr.pos?`: sign of an integer expression;
* `Subset.disjoint?`: two subsets have no common point;
* `Subset.subsetOf?`: every point of one subset is a point of another;
* `iterDisjoint?`: two **distinct** iterations of a map never touch a common
  *linear offset* (`DataDesc.linearOffset`) through two accesses.

Each procedure builds `Query`s (`Core/SymCert`), adds the defining axioms of
opaque atoms (`Query.withAxioms`) and decides them by certificate search
(`Query.decide`).  An arbitrary point of a range dimension `lo:hi:step` is
encoded by an auxiliary variable `k ≥ 0` as `lo + step·k ≤ hi`
(`Dim.mem_elems`); index dimensions (`lo = hi`) need no variable.

`iterDisjoint?` splits on the first map parameter `k` in which two distinct
iteration points differ (`exists_firstDiff`): parameters before `k` are
shared, parameters from `k` on get a primed copy (fresh symbols `'p`,
substituted into the second access), and the gap `i'ₖ ≥ iₖ + stepₖ`
(or the symmetric one) holds by `Dim.elems_sep`.  A case is closed by
*index separation* (same descriptor, strides dominating the shape, both
points in bounds, some dimension separated: `dominates_inj`) or by ordering
the offsets, `off' − off − 1 ≥ 0` or `off − off' − 1 ≥ 0`; queries are tried
without case splits first.  `iterObligations?` returns the
undecided queries instead (kernel fallback, `Query.toLeanFile`), with
soundness assuming their validity.
-/

namespace Dace

set_option autoImplicit false

/-! ## Evaluation depends only on free symbols; substitution -/

section ExprLemmas

variable {I : ExprInterp}

mutual
theorem Expr.eval_congr {ρ ρ' : Valuation} : ∀ (e : Expr), (∀ s ∈ e.freeSyms, ρ s = ρ' s) →
    e.eval I ρ = e.eval I ρ'
  | .lit l, _ => by cases l <;> rfl
  | .sym s, h => by simp only [Expr.eval]; exact h s (by simp [Expr.freeSyms])
  | .un op a, h => by
    simp only [Expr.eval]
    rw [Expr.eval_congr a (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .bin op a b, h => by
    simp only [Expr.eval]
    rw [Expr.eval_congr a (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_congr b (fun s hs => h s (by simp [Expr.freeSyms, hs]))]
  | .ite c t e, h => by
    simp only [Expr.eval]
    rw [Expr.eval_congr c (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_congr t (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_congr e (fun s hs => h s (by simp [Expr.freeSyms, hs]))]
  | .app f args, h => by
    simp only [Expr.eval]
    rw [Expr.evalList_congr args (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .subscript d idx, h => by
    simp only [Expr.eval]
    rw [Expr.evalList_congr idx (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .attr b f, h => by
    simp only [Expr.eval]
    rw [Expr.eval_congr b (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .approx a o, h => by
    simp only [Expr.eval]
    exact Expr.eval_congr a (fun s hs => h s (by simp [Expr.freeSyms, hs]))

theorem Expr.evalList_congr {ρ ρ' : Valuation} : ∀ (es : Exprs), (∀ s ∈ Expr.freeSymsList es, ρ s = ρ' s) →
    Expr.evalList I ρ es = Expr.evalList I ρ' es
  | .nil, _ => rfl
  | .cons e es, h => by
    simp only [Expr.evalList]
    rw [Expr.eval_congr e (fun s hs => h s (by simp [Expr.freeSymsList, hs])),
      Expr.evalList_congr es (fun s hs => h s (by simp [Expr.freeSymsList, hs]))]
end

/-- The expression a symbol is replaced by. -/
abbrev Expr.substOf (σ : Sym → Option Expr) (s : Sym) : Expr := (σ s).getD (.sym s)

mutual
/-- Evaluation of a substituted expression. -/
theorem Expr.eval_subst {σ : Sym → Option Expr} {ρ ρ' : Valuation} : ∀ (e : Expr),
    (∀ s ∈ e.freeSyms, (Expr.substOf σ s).eval I ρ' = ρ s) → (e.subst σ).eval I ρ' = e.eval I ρ
  | .lit l, _ => by cases l <;> rfl
  | .sym s, h => by simp only [Expr.subst, Expr.eval]; exact h s (by simp [Expr.freeSyms])
  | .un op a, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.eval_subst a (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .bin op a b, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.eval_subst a (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_subst b (fun s hs => h s (by simp [Expr.freeSyms, hs]))]
  | .ite c t e, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.eval_subst c (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_subst t (fun s hs => h s (by simp [Expr.freeSyms, hs])),
      Expr.eval_subst e (fun s hs => h s (by simp [Expr.freeSyms, hs]))]
  | .app f args, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.evalList_subst args (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .subscript d idx, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.evalList_subst idx (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .attr b f, h => by
    simp only [Expr.subst, Expr.eval]
    rw [Expr.eval_subst b (fun s hs => h s (by simpa [Expr.freeSyms] using hs))]
  | .approx a o, h => by
    simp only [Expr.subst, Expr.eval]
    exact Expr.eval_subst a (fun s hs => h s (by simp [Expr.freeSyms, hs]))

theorem Expr.evalList_subst {σ : Sym → Option Expr} {ρ ρ' : Valuation} : ∀ (es : Exprs),
    (∀ s ∈ Expr.freeSymsList es, (Expr.substOf σ s).eval I ρ' = ρ s) →
    Expr.evalList I ρ' (Expr.substList σ es) = Expr.evalList I ρ es
  | .nil, _ => rfl
  | .cons e es, h => by
    simp only [Expr.substList, Expr.evalList]
    rw [Expr.eval_subst e (fun s hs => h s (by simp [Expr.freeSymsList, hs])),
      Expr.evalList_subst es (fun s hs => h s (by simp [Expr.freeSymsList, hs]))]
end

mutual
/-- Free symbols of a substituted expression. -/
theorem Expr.freeSyms_subst {σ : Sym → Option Expr} {x : Sym} : ∀ (e : Expr),
    x ∈ (e.subst σ).freeSyms → ∃ s ∈ e.freeSyms, x ∈ (Expr.substOf σ s).freeSyms
  | .lit l, h => by simp [Expr.subst, Expr.freeSyms] at h
  | .sym s, h => ⟨s, by simp [Expr.freeSyms], by simpa [Expr.subst] using h⟩
  | .un op a, h => by
    simp only [Expr.subst, Expr.freeSyms] at h
    obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst a h
    exact ⟨s, by simpa [Expr.freeSyms] using hs, hx⟩
  | .bin op a b, h => by
    simp only [Expr.subst, Expr.freeSyms, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst a h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst b h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
  | .ite c t e, h => by
    simp only [Expr.subst, Expr.freeSyms, List.mem_append] at h
    rcases h with (h | h) | h
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst c h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst t h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst e h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
  | .app f args, h => by
    simp only [Expr.subst, Expr.freeSyms] at h
    obtain ⟨s, hs, hx⟩ := Expr.freeSymsList_subst args h
    exact ⟨s, by simpa [Expr.freeSyms] using hs, hx⟩
  | .subscript d idx, h => by
    simp only [Expr.subst, Expr.freeSyms] at h
    obtain ⟨s, hs, hx⟩ := Expr.freeSymsList_subst idx h
    exact ⟨s, by simpa [Expr.freeSyms] using hs, hx⟩
  | .attr b f, h => by
    simp only [Expr.subst, Expr.freeSyms] at h
    obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst b h
    exact ⟨s, by simpa [Expr.freeSyms] using hs, hx⟩
  | .approx a o, h => by
    simp only [Expr.subst, Expr.freeSyms, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst a h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst o h
      exact ⟨s, by simp [Expr.freeSyms, hs], hx⟩

theorem Expr.freeSymsList_subst {σ : Sym → Option Expr} {x : Sym} : ∀ (es : Exprs),
    x ∈ Expr.freeSymsList (Expr.substList σ es) → ∃ s ∈ Expr.freeSymsList es, x ∈ (Expr.substOf σ s).freeSyms
  | .nil, h => by simp [Expr.substList, Expr.freeSymsList] at h
  | .cons e es, h => by
    simp only [Expr.substList, Expr.freeSymsList, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst e h
      exact ⟨s, by simp [Expr.freeSymsList, hs], hx⟩
    · obtain ⟨s, hs, hx⟩ := Expr.freeSymsList_subst es h
      exact ⟨s, by simp [Expr.freeSymsList, hs], hx⟩
end

theorem Expr.evalInt_congr {ρ ρ' : Valuation} {e : Expr} (h : ∀ s ∈ e.freeSyms, ρ s = ρ' s) :
    e.evalInt I ρ = e.evalInt I ρ' := by
  simp only [Expr.evalInt, Expr.eval_congr e h]

theorem Expr.evalInt_subst {σ : Sym → Option Expr} {ρ ρ' : Valuation} {e : Expr}
    (h : ∀ s ∈ e.freeSyms, (Expr.substOf σ s).eval I ρ' = ρ s) :
    (e.subst σ).evalInt I ρ' = e.evalInt I ρ := by
  simp only [Expr.evalInt, Expr.eval_subst e h]

end ExprLemmas

/-! ## Points of dimensions, ranges and subsets -/

theorem List.mapM_option_some_iff {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {bs : List β}, l.mapM f = some bs ↔ List.Forall₂ (fun a b => f a = some b) l bs
  | [], bs => by cases bs <;> simp
  | a :: l, bs => by
    rw [List.mapM_cons]
    cases bs with
    | nil =>
      simp only [bind, Option.bind_eq_some_iff, pure, Option.some.injEq, reduceCtorEq, and_false, exists_false,
        List.forall₂_nil_right_iff]
    | cons b bs =>
      simp only [bind, Option.bind_eq_some_iff, pure, Option.some.injEq, List.cons.injEq, List.forall₂_cons,
        ← List.mapM_option_some_iff (l := l)]
      constructor
      · rintro ⟨b', hb', bs', hbs', rfl, rfl⟩; exact ⟨hb', hbs'⟩
      · rintro ⟨hb, hbs⟩; exact ⟨b, hb, bs, hbs, rfl, rfl⟩

theorem List.Forall₂.exists_left_of_mem {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ {b : β}, b ∈ l₂ → ∃ a ∈ l₁, R a b
  | _, _, .nil, _, hb => by simp at hb
  | _, _, .cons (a := a) h hs, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, by simp, h⟩
    · obtain ⟨a', ha', h'⟩ := List.Forall₂.exists_left_of_mem hs hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

theorem List.Forall₂.exists_right_of_mem {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ {a : α}, a ∈ l₁ → ∃ b ∈ l₂, R a b
  | _, _, .nil, _, ha => by simp at ha
  | _, _, .cons (b := b) h hs, a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨b, by simp, h⟩
    · obtain ⟨b', hb', h'⟩ := List.Forall₂.exists_right_of_mem hs ha
      exact ⟨b', List.mem_cons_of_mem _ hb', h'⟩

theorem List.mapM_option_congr {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = g a) → l.mapM f = l.mapM g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, List.mapM_cons, h a (by simp),
      List.mapM_option_congr (fun b hb => h b (List.mem_cons_of_mem _ hb))]

theorem List.mapM_option_map {α β γ : Type} {f : β → Option γ} {g : α → β} :
    ∀ {l : List α}, (l.map g).mapM f = l.mapM (f ∘ g)
  | [] => rfl
  | a :: l => by
    rw [List.map_cons, List.mapM_cons, List.mapM_cons, List.mapM_option_map]; rfl

namespace Dim

variable {I : ExprInterp} {ρ : Valuation}

/-- Membership in an enumerated dimension. -/
theorem mem_enum {lo hi step x : Int} (hs : 0 < step) :
    (∃ k : Nat, k < (if hi < lo then 0 else ((hi - lo) / step).toNat + 1) ∧ lo + step * k = x) ↔
      lo ≤ x ∧ x ≤ hi ∧ ∃ k : Int, 0 ≤ k ∧ x = lo + step * k := by
  constructor
  · rintro ⟨k, hk, rfl⟩
    split at hk
    · omega
    · rename_i hlh
      have hq : 0 ≤ (hi - lo) / step := Int.ediv_nonneg (by omega) hs.le
      have hk' : (k : Int) ≤ (hi - lo) / step := by omega
      have := (Int.le_ediv_iff_mul_le hs).1 hk'
      refine ⟨?_, ?_, k, by omega, rfl⟩
      · have : 0 ≤ step * (k : Int) := mul_nonneg hs.le (by omega)
        omega
      · rw [mul_comm]; omega
  · rintro ⟨h1, h2, k, hk, rfl⟩
    refine ⟨k.toNat, ?_, by simp [Int.toNat_of_nonneg hk]⟩
    have : k ≤ (hi - lo) / step := (Int.le_ediv_iff_mul_le hs).2 (by rw [mul_comm]; omega)
    have : 0 ≤ step * k := mul_nonneg hs.le hk
    split <;> omega

/-- **Points of a dimension**: the elements of `lo:hi:step` are the
`lo + step·k ≤ hi`, `k ≥ 0`, for a positive step. -/
theorem mem_elems {d : Dim} {xs : List Int} (h : d.elems I ρ = some xs) :
    ∃ lo hi step, d.lo.evalInt I ρ = some lo ∧ d.hi.evalInt I ρ = some hi ∧ d.step.evalInt I ρ = some step ∧
      0 < step ∧ ∀ x, x ∈ xs ↔ lo ≤ x ∧ x ≤ hi ∧ ∃ k : Int, 0 ≤ k ∧ x = lo + step * k := by
  unfold elems at h
  simp only [bind, Option.bind_eq_some_iff, pure] at h
  obtain ⟨lo, hlo, hi, hhi, step, hst, h⟩ := h
  split at h
  · simp at h
  · rename_i hs
    simp only [Option.some.injEq] at h
    subst h
    refine ⟨lo, hi, step, hlo, hhi, hst, by omega, fun x => ?_⟩
    rw [← mem_enum (by omega)]
    simp only [List.mem_map, List.mem_flatMap, List.mem_range, List.mem_singleton]
    constructor
    · rintro ⟨k, ⟨a, ha, rfl⟩, rfl⟩; exact ⟨a, ha, rfl⟩
    · rintro ⟨a, ha, rfl⟩; exact ⟨a, ⟨a, ha, rfl⟩, rfl⟩

/-- **Separation**: distinct elements of a dimension differ by at least the
step. -/
theorem elems_sep {d : Dim} {xs : List Int} (h : d.elems I ρ = some xs) {step : Int}
    (hst : d.step.evalInt I ρ = some step) {x y : Int} (hx : x ∈ xs) (hy : y ∈ xs) (hlt : x < y) :
    x + step ≤ y := by
  obtain ⟨lo, hi, step', -, -, hst', hs, hmem⟩ := mem_elems h
  rw [hst] at hst'; cases hst'
  obtain ⟨-, -, k, -, rfl⟩ := (hmem x).1 hx
  obtain ⟨-, -, k', -, rfl⟩ := (hmem y).1 hy
  have : k < k' := by
    by_contra hc
    have : step * k' ≤ step * k := mul_le_mul_of_nonneg_left (by omega) hs.le
    omega
  have : step * (k + 1) ≤ step * k' := mul_le_mul_of_nonneg_left (by omega) hs.le
  linarith

/-- The element of an index dimension is its lower bound. -/
theorem elems_index {d : Dim} {xs : List Int} (h : d.elems I ρ = some xs) (hi : d.isIndex = true) {x : Int}
    (hx : x ∈ xs) : d.lo.evalInt I ρ = some x := by
  obtain ⟨lo, hi', step, hlo, hhi, -, -, hmem⟩ := mem_elems h
  have hd : d.lo = d.hi := by simpa [isIndex] using hi
  rw [hd] at hlo; rw [hlo] at hhi; cases hhi
  obtain ⟨h1, h2, -⟩ := (hmem x).1 hx
  rw [← hd] at hlo
  rw [hlo]; congr 1; omega

theorem elems_congr {ρ' : Valuation} {d : Dim} (h : ∀ s ∈ d.freeSyms, ρ s = ρ' s) :
    d.elems I ρ = d.elems I ρ' := by
  unfold elems
  simp only [freeSyms, List.mem_append] at h
  rw [Expr.evalInt_congr (e := d.lo) (fun s hs => h s (by simp [hs])),
    Expr.evalInt_congr (e := d.hi) (fun s hs => h s (by simp [hs])),
    Expr.evalInt_congr (e := d.step) (fun s hs => h s (by simp [hs]))]

theorem elems_subst {σ : Sym → Option Expr} {ρ' : Valuation} {d : Dim}
    (h : ∀ s ∈ d.freeSyms, (Expr.substOf σ s).eval I ρ' = ρ s) : (d.subst σ).elems I ρ' = d.elems I ρ := by
  unfold elems
  simp only [freeSyms, List.mem_append] at h
  simp only [subst]
  rw [Expr.evalInt_subst (e := d.lo) (fun s hs => h s (by simp [hs])),
    Expr.evalInt_subst (e := d.hi) (fun s hs => h s (by simp [hs])),
    Expr.evalInt_subst (e := d.step) (fun s hs => h s (by simp [hs]))]

end Dim

namespace Range

variable {I : ExprInterp} {ρ : Valuation}

/-- **Points of a range**: an index vector is a point iff it picks an element
of every dimension. -/
theorem mem_points {r : Range} {pts : List Idx} (h : r.points I ρ = some pts) {idx : Idx} :
    idx ∈ pts ↔ List.Forall₂ (fun x d => ∃ xs, d.elems I ρ = some xs ∧ x ∈ xs) idx r.dims := by
  simp only [points, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
  obtain ⟨es, hes, rfl⟩ := h
  rw [List.mapM_option_some_iff] at hes
  rw [mem_product]
  generalize r.dims = ds at hes ⊢
  induction hes generalizing idx with
  | nil => simp
  | cons hd _ ih =>
    cases idx with
    | nil => simp
    | cons x idx =>
      simp only [List.forall₂_cons, ih]
      constructor
      · rintro ⟨hx, hr⟩; exact ⟨⟨_, hd, hx⟩, hr⟩
      · rintro ⟨⟨xs, hxs, hx⟩, hr⟩; rw [hd] at hxs; cases hxs; exact ⟨hx, hr⟩

theorem points_congr {ρ' : Valuation} {r : Range} (h : ∀ s ∈ r.freeSyms, ρ s = ρ' s) :
    r.points I ρ = r.points I ρ' := by
  unfold points
  rw [List.mapM_option_congr (l := r.dims) (g := Dim.elems I ρ') (fun d hd =>
    Dim.elems_congr (fun s hs => h s (List.mem_flatMap.2 ⟨d, hd, hs⟩)))]

theorem points_subst {σ : Sym → Option Expr} {ρ' : Valuation} {r : Range}
    (h : ∀ s ∈ r.freeSyms, (Expr.substOf σ s).eval I ρ' = ρ s) : (r.subst σ).points I ρ' = r.points I ρ := by
  unfold points subst
  simp only
  rw [List.mapM_option_map, List.mapM_option_congr (l := r.dims) (f := Dim.elems I ρ' ∘ Dim.subst σ)
    (g := Dim.elems I ρ) (fun d hd =>
    Dim.elems_subst (fun s hs => h s (List.mem_flatMap.2 ⟨d, hd, hs⟩)))]

end Range

namespace Subset

variable {I : ExprInterp} {ρ : Valuation}

/-- The ranges of a subset. -/
def ranges : Subset → List Range
  | range r => [r]
  | union rs => rs

theorem freeSyms_eq (S : Subset) : S.freeSyms = S.ranges.flatMap Range.freeSyms := by
  cases S <;> simp [freeSyms, ranges]

theorem ranges_subst (σ : Sym → Option Expr) (S : Subset) : (S.subst σ).ranges = S.ranges.map (Range.subst σ) := by
  cases S <;> simp [subst, ranges]

/-- A point of a subset is a point of one of its ranges. -/
theorem mem_points {S : Subset} {pts : List Idx} (h : S.points I ρ = some pts) {idx : Idx} (hi : idx ∈ pts) :
    ∃ r ∈ S.ranges, ∃ ps, r.points I ρ = some ps ∧ idx ∈ ps := by
  cases S with
  | range r => exact ⟨r, by simp [ranges], pts, h, hi⟩
  | union rs =>
    simp only [points, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨pss, hpss, rfl⟩ := h
    obtain ⟨ps, hps, hi⟩ := List.mem_flatten.1 hi
    rw [List.mapM_option_some_iff] at hpss
    obtain ⟨r, hr, hrp⟩ := List.Forall₂.exists_left_of_mem hpss hps
    exact ⟨r, hr, ps, hrp, hi⟩

/-- A range of a defined subset is defined, and its points are points of the
subset. -/
theorem points_of_mem_ranges {S : Subset} {pts : List Idx} (h : S.points I ρ = some pts) {r : Range}
    (hr : r ∈ S.ranges) : ∃ ps, r.points I ρ = some ps ∧ ∀ x ∈ ps, x ∈ pts := by
  cases S with
  | range r' =>
    simp only [ranges, List.mem_singleton] at hr; subst hr
    exact ⟨pts, h, fun _ hx => hx⟩
  | union rs =>
    simp only [points, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨pss, hpss, rfl⟩ := h
    rw [List.mapM_option_some_iff] at hpss
    obtain ⟨ps, hps, hrp⟩ := List.Forall₂.exists_right_of_mem hpss hr
    exact ⟨ps, hrp, fun x hx => List.mem_flatten.2 ⟨ps, hps, hx⟩⟩

end Subset

theorem DataDesc.linearOffset_congr {I : ExprInterp} {ρ ρ' : Valuation} {d : DataDesc} {idx : Idx}
    (h : ∀ s ∈ d.effStrides.flatMap Expr.freeSyms ++ d.startOffset.freeSyms, ρ s = ρ' s) :
    d.linearOffset I ρ idx = d.linearOffset I ρ' idx := by
  unfold DataDesc.linearOffset
  rw [List.mapM_option_congr (l := d.effStrides) (g := Expr.evalInt I ρ') (fun e he =>
      Expr.evalInt_congr (fun s hs => h s (List.mem_append_left _ (List.mem_flatMap.2 ⟨e, he, hs⟩)))),
    Expr.evalInt_congr (e := d.startOffset) (fun s hs => h s (List.mem_append_right _ hs))]

/-! ## Environments with auxiliary variables -/

/-- An atom usable in a query under `ρ`: an auxiliary variable or a defined
atom. -/
def GoodAtom (I : ExprInterp) (ρ : Valuation) (a : Atom) : Prop := a.isVar = true ∨ AtomOk I ρ a

/-- The environment of a valuation with values for the auxiliary variables. -/
def envOf (I : ExprInterp) (ρ : Valuation) (vals : Nat → Int) : Env := fun a =>
  match a with
  | .var n => vals n
  | a => atomEnv I ρ a

/-- Hypotheses that hold under `env` and have good atoms. -/
def HoldsGood (I : ExprInterp) (ρ : Valuation) (env : Env) (hs : List Ineq) : Prop :=
  ∀ h ∈ hs, h.holds env ∧ h.poly.AllAtoms (GoodAtom I ρ)

/-- Disjunctions that hold under `env` and have good atoms. -/
def DisjGood (I : ExprInterp) (ρ : Valuation) (env : Env) (ds : List Disj) : Prop :=
  ∀ d ∈ ds, d.holds env ∧ ∀ h ∈ d.1 ++ d.2, h.poly.AllAtoms (GoodAtom I ρ)

section Env

variable {I : ExprInterp} {ρ : Valuation}

theorem HoldsGood.append {env : Env} {hs hs' : List Ineq} (h : HoldsGood I ρ env hs) (h' : HoldsGood I ρ env hs') :
    HoldsGood I ρ env (hs ++ hs') := by
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  · exact h x hx
  · exact h' x hx

theorem HoldsGood.nil {env : Env} : HoldsGood I ρ env [] := by intro x hx; simp at hx

theorem envOf_of_not_var {vals : Nat → Int} {a : Atom} (h : a.isVar = false) : envOf I ρ vals a = atomEnv I ρ a := by
  cases a <;> simp_all [envOf, Atom.isVar]

theorem eval_envOf {vals : Nat → Int} {p : Poly} (hp : p.AllAtoms (AtomOk I ρ)) :
    p.eval (envOf I ρ vals) = p.eval (atomEnv I ρ) :=
  Poly.eval_congr (Poly.allAtoms_mono (fun _ ha => envOf_of_not_var ha.1) hp)

theorem Poly.allAtoms_good {p : Poly} (hp : p.AllAtoms (AtomOk I ρ)) : p.AllAtoms (GoodAtom I ρ) :=
  Poly.allAtoms_mono (fun _ ha => Or.inr ha) hp

theorem Poly.allAtoms_good_var (n : Nat) : (Poly.var (.var n)).AllAtoms (GoodAtom I ρ) :=
  Poly.allAtoms_var (Or.inl rfl)

@[simp] theorem envOf_var {vals : Nat → Int} (n : Nat) : envOf I ρ vals (.var n) = vals n := rfl

/-- The polynomial of an evaluating integer-closed expression. -/
theorem poly_of_expr {e : Expr} {p : Poly} (hp : e.toPoly? = some p) (hs : ∀ s ∈ e.freeSyms, IntBound ρ s)
    {v : Int} (hv : e.evalInt I ρ = some v) (vals : Nat → Int) :
    p.eval (envOf I ρ vals) = v ∧ p.AllAtoms (GoodAtom I ρ) := by
  have hv' := hv
  simp only [Expr.evalInt, bind, Option.bind_eq_some_iff] at hv'
  obtain ⟨w, hw, -⟩ := hv'
  have hok := toPoly?_atomOk hp hs hw
  exact ⟨by rw [eval_envOf hok, ← toPoly?_sound hp hs hv], Poly.allAtoms_good hok⟩

/-- A symbol bound to an integer is a defined atom. -/
theorem sym_atomOk {s : Sym} {n : Int} (h : ρ s = some (.int n)) :
    AtomOk I ρ (.sym s) ∧ atomEnv I ρ (.sym s) = n := by
  have hv : atomEnv I ρ (.sym s) = n := by simp [atomEnv, h, Val.toInt?]
  refine ⟨⟨rfl, by simp [Atom.expr, Expr.eval, h, hv], ?_⟩, hv⟩
  intro s' hs'; simp [Atom.expr, Expr.freeSyms] at hs'; subst hs'; exact ⟨n, h⟩

theorem sym_poly {s : Sym} {n : Int} (h : ρ s = some (.int n)) (vals : Nat → Int) :
    (Poly.var (.sym s)).eval (envOf I ρ vals) = n ∧ (Poly.var (.sym s)).AllAtoms (GoodAtom I ρ) := by
  obtain ⟨hok, hv⟩ := sym_atomOk (I := I) h
  exact ⟨by simp [envOf, hv], Poly.allAtoms_var (Or.inr hok)⟩

/-- Atoms whose free symbols take the same values under two valuations. -/
theorem atomOk_congr {ρ' : Valuation} {a : Atom} (h : ∀ s ∈ a.expr.freeSyms, ρ s = ρ' s) (ha : AtomOk I ρ a) :
    AtomOk I ρ' a ∧ atomEnv I ρ' a = atomEnv I ρ a := by
  have henv : atomEnv I ρ' a = atomEnv I ρ a := by
    cases a with
    | sym s => simp [atomEnv, h s (by simp [Atom.expr, Expr.freeSyms])]
    | opq e => simp [atomEnv, Expr.evalInt_congr (e := e) (fun s hs => (h s hs).symm)]
    | var n => rfl
  obtain ⟨hv, he, hs⟩ := ha
  refine ⟨⟨hv, ?_, ?_⟩, henv⟩
  · rw [henv, ← he]; exact (Expr.eval_congr _ h).symm
  · intro s hs'; obtain ⟨n, hn⟩ := hs s hs'; exact ⟨n, (h s hs') ▸ hn⟩

end Env

/-! ## Queries over facts -/

namespace SymSubset

/-- Every free symbol of every atom of the polynomial avoids `bad`. -/
def polyAvoids (bad : List Sym) (p : Poly) : Bool := p.atoms.all fun a => a.expr.freeSyms.all (· ∉ bad)

/-- The fact hypotheses whose atoms do not mention `bad` symbols (map
parameters and their primed copies, which shadow outer symbols). -/
def factsAvoiding (C : Constraints) (bad : List Sym) : List Ineq × List Disj :=
  (C.toHyps.1.filter fun h => polyAvoids bad h.poly,
   C.toHyps.2.filter fun d => (d.1 ++ d.2).all fun h => polyAvoids bad h.poly)

/-- A query with the (filtered) facts as additional hypotheses. -/
def factQuery (C : Constraints) (bad : List Sym) (hyps : List Ineq) (goal : Poly) : Query :=
  { hyps := (factsAvoiding C bad).1 ++ hyps, disj := (factsAvoiding C bad).2, goal }

/-- Decide a query after adding the axioms of its opaque atoms. -/
def decideQ (q : Query) (depth : Nat := 2) : Bool := q.withAxioms.decide depth

theorem decideQ_sound {q : Query} {depth : Nat} (h : decideQ q depth = true) : q.withAxioms.Valid :=
  Query.decide_sound h

variable {I : ExprInterp} {ρ : Valuation}

/-- Apply a valid query (with axioms) to the environment of a valuation. -/
theorem query_apply (q : Query) (hv : q.withAxioms.Valid) (vals : Nat → Int)
    (hh : HoldsGood I ρ (envOf I ρ vals) q.hyps) (hd : DisjGood I ρ (envOf I ρ vals) q.disj)
    (hg : q.goal.AllAtoms (GoodAtom I ρ)) : 0 ≤ q.goal.eval (envOf I ρ vals) := by
  refine Query.withAxioms_sound (I := I) (ρ := ρ) q 3 hv (envOf I ρ vals) ?_ (fun h hm => (hh h hm).1)
    (fun d hm => (hd d hm).1)
  intro a ha hvar
  have hgood : GoodAtom I ρ a := by
    simp only [Query.atoms, mem_addNew, List.not_mem_nil, false_or, List.mem_append, List.mem_flatMap,
      Ineq.atoms, Disj.atoms] at ha
    rcases ha with (ha | ⟨h, hm, ha⟩) | ⟨d, hm, h, hh', ha⟩
    · exact Poly.allAtoms_iff.1 hg a ha
    · exact Poly.allAtoms_iff.1 (hh h hm).2 a ha
    · exact Poly.allAtoms_iff.1 ((hd d hm).2 h (List.mem_append.2 hh')) a ha
  rcases hgood with h | h
  · simp [h] at hvar
  · exact ⟨h, envOf_of_not_var hvar⟩

theorem polyAvoids_agree {ρ' : Valuation} {bad : List Sym} {p : Poly} (h : polyAvoids bad p = true)
    (hag : ∀ s, s ∉ bad → ρ s = ρ' s) (hok : p.AllAtoms (AtomOk I ρ)) :
    p.AllAtoms (AtomOk I ρ') ∧ p.eval (atomEnv I ρ') = p.eval (atomEnv I ρ) := by
  have hav : ∀ a ∈ p.atoms, ∀ s ∈ a.expr.freeSyms, ρ s = ρ' s := by
    intro a ha s hs
    simp only [polyAvoids, List.all_eq_true, decide_eq_true_eq] at h
    exact hag s (h a ha s hs)
  have hc : ∀ a ∈ p.atoms, AtomOk I ρ' a ∧ atomEnv I ρ' a = atomEnv I ρ a :=
    fun a ha => atomOk_congr (hav a ha) (Poly.allAtoms_iff.1 hok a ha)
  exact ⟨Poly.allAtoms_iff.2 fun a ha => (hc a ha).1, Poly.eval_congr (Poly.allAtoms_iff.2 fun a ha => (hc a ha).2)⟩

theorem ineq_transfer {ρ' : Valuation} {bad : List Sym} {h : Ineq} (hav : polyAvoids bad h.poly = true)
    (hag : ∀ s, s ∉ bad → ρ s = ρ' s) (hok : h.poly.AllAtoms (AtomOk I ρ)) (hh : h.holds (atomEnv I ρ))
    (vals : Nat → Int) : h.holds (envOf I ρ' vals) ∧ h.poly.AllAtoms (GoodAtom I ρ') := by
  obtain ⟨hok', he⟩ := polyAvoids_agree hav hag hok
  refine ⟨?_, Poly.allAtoms_good hok'⟩
  unfold Ineq.holds at hh ⊢
  rw [eval_envOf hok', he]
  exact hh

/-- **Facts as hypotheses under a valuation agreeing outside `bad`.** -/
theorem factsAvoiding_sound {ρ' : Valuation} {C : Constraints} {bad : List Sym} (hC : C.satisfies I ρ)
    (hag : ∀ s, s ∉ bad → ρ s = ρ' s) (vals : Nat → Int) :
    HoldsGood I ρ' (envOf I ρ' vals) (factsAvoiding C bad).1 ∧
      DisjGood I ρ' (envOf I ρ' vals) (factsAvoiding C bad).2 := by
  obtain ⟨hh, hd, hah, had⟩ := Constraints.toHyps_sound hC
  constructor
  · intro h hm
    obtain ⟨hm, hav⟩ := List.mem_filter.1 hm
    exact ineq_transfer hav hag (hah h hm) (hh h hm) vals
  · intro d hm
    obtain ⟨hm, hav⟩ := List.mem_filter.1 hm
    have hav' := List.all_eq_true.1 hav
    have ht : ∀ h ∈ d.1 ++ d.2, h.holds (atomEnv I ρ) → h.holds (envOf I ρ' vals) ∧
        h.poly.AllAtoms (GoodAtom I ρ') := fun h hh' hx =>
      ineq_transfer (hav' h hh') hag (had d hm h hh') hx vals
    refine ⟨?_, fun h hh' => ?_⟩
    · rcases hd d hm with hl | hr
      · exact Or.inl fun h hh' => (ht h (List.mem_append_left _ hh') (hl h hh')).1
      · exact Or.inr fun h hh' => (ht h (List.mem_append_right _ hh') (hr h hh')).1
    · exact Poly.allAtoms_good (polyAvoids_agree (hav' h hh') hag (had d hm h hh')).1

/-- Soundness of a fact query: hypotheses holding under the environment give
the goal. -/
theorem factQuery_sound {ρ' : Valuation} {C : Constraints} {bad : List Sym} {hyps : List Ineq} {goal : Poly}
    (hv : (factQuery C bad hyps goal).withAxioms.Valid) (hC : C.satisfies I ρ) (hag : ∀ s, s ∉ bad → ρ s = ρ' s)
    (vals : Nat → Int) (hh : HoldsGood I ρ' (envOf I ρ' vals) hyps) (hg : goal.AllAtoms (GoodAtom I ρ')) :
    0 ≤ goal.eval (envOf I ρ' vals) := by
  obtain ⟨fh, fd⟩ := factsAvoiding_sound (ρ' := ρ') (bad := bad) hC hag vals
  exact query_apply (factQuery C bad hyps goal) hv vals (fh.append hh) fd hg

end SymSubset

/-! ## Encoding points of ranges -/

namespace SymSubset

open Poly

/-- An arbitrary element of dimension `d`, using auxiliary variable `v` for
the step count: its polynomial and its hypotheses.  Index dimensions need no
variable. -/
def encodeDim (d : Dim) (v : Nat) : Option (Poly × List Ineq) := do
  let lo ← d.lo.toPoly?
  if d.isIndex then pure (lo, [])
  else
    let hi ← d.hi.toPoly?
    let st ← d.step.toPoly?
    let k := Poly.var (.var v)
    let x := Poly.add lo (Poly.mul st k)
    pure (x, [Ineq.ge k, Ineq.le lo x, Ineq.le x hi, Ineq.le (Poly.const 1) st])

/-- Encoding of the dimensions from position `j` on; dimension `m` uses
auxiliary variable `2·m + tag`. -/
def encodeDims (tag : Nat) : Nat → List Dim → Option (List Poly × List Ineq)
  | _, [] => some ([], [])
  | j, d :: ds => do
    let (x, hs) ← encodeDim d (2 * j + tag)
    let (xs, hs') ← encodeDims tag (j + 1) ds
    pure (x :: xs, hs ++ hs')

/-- Encoding of an arbitrary point of a range. -/
def encodeRange (r : Range) (tag : Nat) : Option (List Poly × List Ineq) := encodeDims tag 0 r.dims

variable {I : ExprInterp} {ρ : Valuation}

theorem encodeDim_sound {d : Dim} {v : Nat} {x : Poly} {hs : List Ineq} (he : encodeDim d v = some (x, hs))
    (hsy : ∀ s ∈ d.freeSyms, IntBound ρ s) {xs : List Int} (hel : d.elems I ρ = some xs) {a : Int}
    (ha : a ∈ xs) : ∃ k : Int, ∀ vals : Nat → Int, vals v = k →
      x.eval (envOf I ρ vals) = a ∧ HoldsGood I ρ (envOf I ρ vals) hs ∧ x.AllAtoms (GoodAtom I ρ) := by
  have hlo : ∀ s ∈ d.lo.freeSyms, IntBound ρ s := fun s h => hsy s (by simp [Dim.freeSyms, h])
  have hhi : ∀ s ∈ d.hi.freeSyms, IntBound ρ s := fun s h => hsy s (by simp [Dim.freeSyms, h])
  have hst : ∀ s ∈ d.step.freeSyms, IntBound ρ s := fun s h => hsy s (by simp [Dim.freeSyms, h])
  unfold encodeDim at he
  simp only [bind, Option.bind_eq_some_iff, pure] at he
  obtain ⟨lo, hlop, he⟩ := he
  split at he
  · rename_i hidx
    simp only [Option.some.injEq, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    have hv := Dim.elems_index hel hidx ha
    refine ⟨0, fun vals _ => ?_⟩
    obtain ⟨h1, h2⟩ := poly_of_expr hlop hlo hv vals
    exact ⟨h1, HoldsGood.nil, h2⟩
  · cases hhip : d.hi.toPoly? with
    | none => simp [hhip] at he
    | some hi =>
    cases hstp : d.step.toPoly? with
    | none => simp [hhip, hstp] at he
    | some st =>
    simp only [hhip, hstp, Option.bind_some, Option.some.injEq, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    obtain ⟨vlo, vhi, vst, hvlo, hvhi, hvst, hpos, hmem⟩ := Dim.mem_elems hel
    obtain ⟨h1, h2, k, hk, rfl⟩ := (hmem a).1 ha
    refine ⟨k, fun vals hvk => ?_⟩
    obtain ⟨elo, glo⟩ := poly_of_expr hlop hlo hvlo vals
    obtain ⟨ehi, ghi⟩ := poly_of_expr hhip hhi hvhi vals
    obtain ⟨est, gst⟩ := poly_of_expr hstp hst hvst vals
    have gk := Poly.allAtoms_good_var (I := I) (ρ := ρ) v
    have gx := Poly.allAtoms_add glo (Poly.allAtoms_mul gst gk)
    have ex : (Poly.add lo (Poly.mul st (Poly.var (.var v)))).eval (envOf I ρ vals) = vlo + vst * k := by
      simp [elo, est, hvk]
    refine ⟨ex, ?_, gx⟩
    intro h hm
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
    rcases hm with rfl | rfl | rfl | rfl
    · exact ⟨by simp [hvk, hk], gk⟩
    · exact ⟨by simp only [Ineq.holds_le, ex, elo]; omega, Ineq.le_atoms glo gx⟩
    · exact ⟨by simp only [Ineq.holds_le, ex, ehi]; omega, Ineq.le_atoms gx ghi⟩
    · exact ⟨by simp only [Ineq.holds_le, est, eval_const]; omega, Ineq.le_atoms (Poly.allAtoms_const 1) gst⟩

theorem encodeDims_sound (tag : Nat) : ∀ (j : Nat) (ds : List Dim) (xs : List Poly) (hs : List Ineq),
    encodeDims tag j ds = some (xs, hs) → (∀ s ∈ ds.flatMap Dim.freeSyms, IntBound ρ s) →
    ∀ (idx : Idx), List.Forall₂ (fun x d => ∃ es, d.elems I ρ = some es ∧ x ∈ es) idx ds →
    ∃ f : Nat → Int, ∀ vals : Nat → Int, (∀ m, j ≤ m → vals (2 * m + tag) = f m) →
      List.Forall₂ (fun p a => p.eval (envOf I ρ vals) = a) xs idx ∧ HoldsGood I ρ (envOf I ρ vals) hs ∧
      ∀ p ∈ xs, p.AllAtoms (GoodAtom I ρ)
  | j, [], xs, hs, he, _, idx, hidx => by
    simp only [encodeDims, Option.some.injEq, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    cases hidx
    exact ⟨fun _ => 0, fun _ _ => ⟨.nil, HoldsGood.nil, by simp⟩⟩
  | j, d :: ds, xs, hs, he, hsy, idx, hidx => by
    simp only [encodeDims, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at he
    obtain ⟨⟨x, h1⟩, hd, ⟨xs', h2⟩, hds, he⟩ := he
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    cases hidx with
    | cons ha hrest =>
      rename_i a idx'
      obtain ⟨es, hes, hae⟩ := ha
      obtain ⟨k, hk⟩ := encodeDim_sound hd (fun s h => hsy s (List.mem_flatMap.2 ⟨d, by simp, h⟩)) hes hae
      obtain ⟨f, hf⟩ := encodeDims_sound tag (j + 1) ds xs' h2 hds
        (fun s h => hsy s (by
          obtain ⟨d', hd', h⟩ := List.mem_flatMap.1 h
          exact List.mem_flatMap.2 ⟨d', List.mem_cons_of_mem _ hd', h⟩)) idx' hrest
      refine ⟨fun m => if m = j then k else f m, fun vals hv => ?_⟩
      obtain ⟨e1, g1, a1⟩ := hk vals (by simpa using hv j le_rfl)
      obtain ⟨e2, g2, a2⟩ := hf vals (fun m hm => by
        rw [hv m (by omega)]; simp [show m ≠ j by omega])
      refine ⟨.cons e1 e2, g1.append g2, ?_⟩
      intro p hp
      rcases List.mem_cons.1 hp with rfl | hp
      · exact a1
      · exact a2 p hp

theorem encodeRange_sound {r : Range} {tag : Nat} {xs : List Poly} {hs : List Ineq}
    (he : encodeRange r tag = some (xs, hs)) (hsy : ∀ s ∈ r.freeSyms, IntBound ρ s) {pts : List Idx}
    (hp : r.points I ρ = some pts) {idx : Idx} (hi : idx ∈ pts) :
    ∃ f : Nat → Int, ∀ vals : Nat → Int, (∀ m, vals (2 * m + tag) = f m) →
      List.Forall₂ (fun p a => p.eval (envOf I ρ vals) = a) xs idx ∧ HoldsGood I ρ (envOf I ρ vals) hs ∧
      ∀ p ∈ xs, p.AllAtoms (GoodAtom I ρ) := by
  obtain ⟨f, hf⟩ := encodeDims_sound tag 0 r.dims xs hs he hsy idx ((Range.mem_points hp).1 hi)
  exact ⟨f, fun vals hv => hf vals (fun m _ => hv m)⟩

/-! ## Linear offsets -/

/-- `Σ xᵢ·sᵢ` as a polynomial, in the order of `Range.linearize`. -/
def linPoly (xs ss : List Poly) : Poly := ((xs.zip ss).map fun (x, s) => Poly.mul x s).foldl Poly.add []

/-- The polynomial of the linear offset of a point given by polynomials. -/
def offsetPoly? (d : DataDesc) (xs : List Poly) : Option Poly := do
  let ss ← d.effStrides.mapM Expr.toPoly?
  let st ← d.startOffset.toPoly?
  if ss.length = xs.length then pure (Poly.add st (linPoly xs ss)) else none

/-- Free symbols of the strides and start offset. -/
def offSyms (d : DataDesc) : List Sym := d.effStrides.flatMap Expr.freeSyms ++ d.startOffset.freeSyms

theorem linPoly_eval (env : Env) : ∀ (xs ss : List Poly) (idx strides : List Int) (accP : Poly) (accI : Int),
    List.Forall₂ (fun p a => p.eval env = a) xs idx → List.Forall₂ (fun p a => p.eval env = a) ss strides →
    accP.eval env = accI →
    (((xs.zip ss).map fun (x, s) => Poly.mul x s).foldl Poly.add accP).eval env =
      ((idx.zip strides).map fun (i, s) => i * s).foldl (· + ·) accI
  | [], _, [], _, _, _, _, _, h => by simpa using h
  | _ :: _, [], _ :: _, [], _, _, _, _, h => by simpa using h
  | x :: xs, s :: ss, a :: idx, b :: strides, accP, accI, .cons hx hxs, .cons hs hss, h => by
    simp only [List.zip_cons_cons, List.map_cons, List.foldl_cons]
    exact linPoly_eval env xs ss idx strides _ _ hxs hss (by simp [h, hx, hs])
  | _ :: _, _ :: _, _ :: _, [], _, _, _, h, _ => by cases h
  | _ :: _, [], _ :: _, _ :: _, _, _, _, h, _ => by cases h

theorem linPoly_atoms {P : Atom → Prop} : ∀ (xs ss : List Poly) (accP : Poly), accP.AllAtoms P →
    (∀ p ∈ xs, p.AllAtoms P) → (∀ p ∈ ss, p.AllAtoms P) →
    (((xs.zip ss).map fun (x, s) => Poly.mul x s).foldl Poly.add accP).AllAtoms P
  | [], _, _, h, _, _ => by simpa using h
  | _ :: _, [], _, h, _, _ => by simpa using h
  | x :: xs, s :: ss, accP, h, hx, hs => by
    simp only [List.zip_cons_cons, List.map_cons, List.foldl_cons]
    exact linPoly_atoms xs ss _ (Poly.allAtoms_add h (Poly.allAtoms_mul (hx x (by simp)) (hs s (by simp))))
      (fun p hp => hx p (List.mem_cons_of_mem _ hp)) (fun p hp => hs p (List.mem_cons_of_mem _ hp))

theorem offsetPoly?_sound {d : DataDesc} {xs : List Poly} {P : Poly} (he : offsetPoly? d xs = some P)
    (hsy : ∀ s ∈ offSyms d, IntBound ρ s) (vals : Nat → Int) {idx : Idx}
    (hx : List.Forall₂ (fun p a => p.eval (envOf I ρ vals) = a) xs idx) (hxa : ∀ p ∈ xs, p.AllAtoms (GoodAtom I ρ))
    {o : Int} (ho : d.linearOffset I ρ idx = some o) : P.eval (envOf I ρ vals) = o ∧ P.AllAtoms (GoodAtom I ρ) := by
  unfold offsetPoly? at he
  simp only [bind, Option.bind_eq_some_iff, pure] at he
  obtain ⟨ss, hss, st, hst, he⟩ := he
  split at he
  · simp only [Option.some.injEq] at he; subst he
    unfold DataDesc.linearOffset at ho
    simp only [bind, Option.bind_eq_some_iff, pure] at ho
    obtain ⟨strides, hstr, start, hstart, ho⟩ := ho
    split at ho
    · simp only [Option.some.injEq] at ho; subst ho
      rw [List.mapM_option_some_iff] at hss hstr
      have hsyS : ∀ e ∈ d.effStrides, ∀ s ∈ e.freeSyms, IntBound ρ s := fun e he s hs =>
        hsy s (List.mem_append_left _ (List.mem_flatMap.2 ⟨e, he, hs⟩))
      -- strides as polynomials
      have key : ∀ (es : List Expr) (ps : List Poly) (vs : List Int), (∀ e ∈ es, ∀ s ∈ e.freeSyms, IntBound ρ s) →
          List.Forall₂ (fun e p => e.toPoly? = some p) es ps → List.Forall₂ (fun e v => e.evalInt I ρ = some v) es vs →
          List.Forall₂ (fun p a => p.eval (envOf I ρ vals) = a) ps vs ∧ ∀ p ∈ ps, p.AllAtoms (GoodAtom I ρ) := by
        intro es ps vs hsy' h1 h2
        induction h1 generalizing vs with
        | nil => cases h2; exact ⟨.nil, by simp⟩
        | cons hp _ ih =>
          cases h2 with
          | cons hv h2 =>
            obtain ⟨e1, a1⟩ := poly_of_expr hp (hsy' _ (by simp)) hv vals
            obtain ⟨e2, a2⟩ := ih _ (fun e he => hsy' e (List.mem_cons_of_mem _ he)) h2
            refine ⟨.cons e1 e2, fun p hp' => ?_⟩
            rcases List.mem_cons.1 hp' with rfl | hp'
            · exact a1
            · exact a2 p hp'
      obtain ⟨es, ea⟩ := key _ _ _ hsyS hss hstr
      obtain ⟨e0, a0⟩ := poly_of_expr hst (fun s hs => hsy s (List.mem_append_right _ hs)) hstart vals
      refine ⟨?_, Poly.allAtoms_add a0 (linPoly_atoms _ _ _ (by simp [Poly.AllAtoms]) hxa ea)⟩
      simp only [Poly.eval_add, e0, linPoly, Range.linearize]
      rw [linPoly_eval _ xs ss idx strides [] 0 hx es (by simp)]
    · simp at ho
  · simp at he

/-! ## Helpers -/

/-- All symbols are integer symbols of the facts. -/
def intSymsOk (C : Constraints) (ss : List Sym) : Bool := ss.all (· ∈ C.intSyms)

theorem intSymsOk_sound {C : Constraints} {ss : List Sym} (h : intSymsOk C ss = true) (hC : C.satisfies I ρ) :
    ∀ s ∈ ss, IntBound ρ s := by
  intro s hs
  simp only [intSymsOk, List.all_eq_true, decide_eq_true_eq] at h
  exact hC.1 s (h s hs)

/-- Auxiliary values interleaving two assignments (even and odd variables). -/
def interleave (f g : Nat → Int) : Nat → Int := fun n => if n % 2 = 0 then f (n / 2) else g (n / 2)

theorem interleave_even (f g : Nat → Int) (m : Nat) : interleave f g (2 * m + 0) = f m := by
  simp [interleave]

theorem interleave_odd (f g : Nat → Int) (m : Nat) : interleave f g (2 * m + 1) = g m := by
  simp only [interleave, show (2 * m + 1) % 2 = 1 by omega, show (2 * m + 1) / 2 = m by omega]; simp

theorem zip_pair {α β : Type} {R S : α → β → Prop} : ∀ {xs ys : List α} {p q : List β},
    List.Forall₂ R xs p → List.Forall₂ S ys q → ∀ {x y : α}, (x, y) ∈ xs.zip ys → p = q → ∃ a, R x a ∧ S y a
  | _, _, _, _, .nil, _, _, _, h, _ => by simp at h
  | _, _, _, _, .cons _ _, .nil, _, _, h, _ => by simp at h
  | _, _, _, _, .cons (a := x') hx hxs, .cons hy hys, x, y, h, hpq => by
    simp only [List.zip_cons_cons, List.mem_cons, Prod.mk.injEq] at h
    simp only [List.cons.injEq] at hpq
    obtain ⟨rfl, hpq⟩ := hpq
    rcases h with ⟨rfl, rfl⟩ | h
    · exact ⟨_, hx, hy⟩
    · exact zip_pair hxs hys h hpq

end SymSubset

/-! ## Sign of an expression -/

open SymSubset in
/-- Is the expression non-negative under the facts? -/
def Expr.nonneg? (C : Constraints) (e : Expr) : Bool :=
  intSymsOk C e.freeSyms &&
    match e.toPoly? with
    | some p => decideQ (factQuery C [] [] p)
    | none => false

open SymSubset in
/-- Is the expression positive under the facts? -/
def Expr.pos? (C : Constraints) (e : Expr) : Bool :=
  intSymsOk C e.freeSyms &&
    match e.toPoly? with
    | some p => decideQ (factQuery C [] [] (Poly.sub p (Poly.const 1)))
    | none => false

section Sign

open SymSubset

variable {I : ExprInterp} {ρ : Valuation}

/-- **Soundness of `nonneg?`.** -/
theorem Expr.nonneg?_sound {C : Constraints} {e : Expr} (h : e.nonneg? C = true) (hC : C.satisfies I ρ)
    {v : Int} (hv : e.evalInt I ρ = some v) : 0 ≤ v := by
  simp only [Expr.nonneg?, Bool.and_eq_true] at h
  obtain ⟨hs, h⟩ := h
  split at h
  · rename_i p hp
    obtain ⟨ev, ga⟩ := poly_of_expr hp (intSymsOk_sound hs hC) hv (fun _ => 0)
    have := factQuery_sound (decideQ_sound h) hC (fun _ _ => rfl) (fun _ => 0) HoldsGood.nil ga
    omega
  · simp at h

/-- **Soundness of `pos?`.** -/
theorem Expr.pos?_sound {C : Constraints} {e : Expr} (h : e.pos? C = true) (hC : C.satisfies I ρ)
    {v : Int} (hv : e.evalInt I ρ = some v) : 0 < v := by
  simp only [Expr.pos?, Bool.and_eq_true] at h
  obtain ⟨hs, h⟩ := h
  split at h
  · rename_i p hp
    obtain ⟨ev, ga⟩ := poly_of_expr hp (intSymsOk_sound hs hC) hv (fun _ => 0)
    have := factQuery_sound (decideQ_sound h) hC (fun _ _ => rfl) (fun _ => 0) HoldsGood.nil
      (Poly.allAtoms_sub ga (Poly.allAtoms_const 1))
    simp only [Poly.eval_sub, ev, Poly.eval_const] at this
    omega
  · simp at h

end Sign

/-! ## Disjointness and inclusion of subsets -/

namespace SymSubset

/-- Two ranges have no common point: some dimension separates them. -/
def rangeDisjoint? (C : Constraints) (rA rB : Range) : Bool :=
  if rA.ndim ≠ rB.ndim then true
  else
    match encodeRange rA 0, encodeRange rB 1 with
    | some (xs, hA), some (ys, hB) =>
      (xs.zip ys).any fun (x, y) =>
        decideQ (factQuery C [] (hA ++ hB) (Poly.sub (Poly.sub y x) (Poly.const 1))) ||
          decideQ (factQuery C [] (hA ++ hB) (Poly.sub (Poly.sub x y) (Poly.const 1)))
    | _, _ => false

/-- Element `x` (under hypotheses `hyps`) lies in dimension `d`: `d` is an
index equal to `x`, or a unit-step dimension with `lo ≤ x ≤ hi`. -/
def dimContains? (C : Constraints) (hyps : List Ineq) (x : Poly) (d : Dim) : Bool :=
  match d.lo.toPoly?, d.hi.toPoly?, d.step.toPoly? with
  | some lo, some hi, some st =>
    if d.isIndex then
      decideQ (factQuery C [] hyps (Poly.sub x lo)) && decideQ (factQuery C [] hyps (Poly.sub lo x))
    else
      st.const? == some 1 && decideQ (factQuery C [] hyps (Poly.sub x lo)) &&
        decideQ (factQuery C [] hyps (Poly.sub hi x))
  | _, _, _ => false

/-- Every point of `rA` is a point of `rB`. -/
def rangeSubset? (C : Constraints) (rA rB : Range) : Bool :=
  match encodeRange rA 0 with
  | some (xs, hA) => xs.length == rB.dims.length && (xs.zip rB.dims).all fun (x, d) => dimContains? C hA x d
  | none => false

end SymSubset

open SymSubset in
/-- The subsets have no common point. -/
def Subset.disjoint? (C : Constraints) (A B : Subset) : Bool :=
  intSymsOk C (A.freeSyms ++ B.freeSyms) && A.ranges.all fun rA => B.ranges.all fun rB => rangeDisjoint? C rA rB

open SymSubset in
/-- Every point of `A` is a point of `B` (each range of `A` lies in one range
of `B`). -/
def Subset.subsetOf? (C : Constraints) (A B : Subset) : Bool :=
  intSymsOk C (A.freeSyms ++ B.freeSyms) && A.ranges.all fun rA => B.ranges.any fun rB => rangeSubset? C rA rB

namespace SymSubset

variable {I : ExprInterp} {ρ : Valuation}

theorem length_of_mem_points {r : Range} {pts : List Idx} (h : r.points I ρ = some pts) {p : Idx} (hp : p ∈ pts) :
    p.length = r.ndim :=
  ((Range.mem_points h).1 hp).length_eq

theorem rangeDisjoint?_sound {C : Constraints} {rA rB : Range} (h : rangeDisjoint? C rA rB = true)
    (hC : C.satisfies I ρ) (hsA : ∀ s ∈ rA.freeSyms, IntBound ρ s) (hsB : ∀ s ∈ rB.freeSyms, IntBound ρ s)
    {ps qs : List Idx} (hpA : rA.points I ρ = some ps) (hpB : rB.points I ρ = some qs) {p q : Idx}
    (hp : p ∈ ps) (hq : q ∈ qs) : p ≠ q := by
  intro hpq
  unfold rangeDisjoint? at h
  split at h
  · rename_i hn
    exact hn (by rw [← length_of_mem_points hpA hp, ← length_of_mem_points hpB hq, hpq])
  · split at h
    · rename_i xs hA ys hB heA heB
      obtain ⟨f, hf⟩ := encodeRange_sound heA hsA hpA hp
      obtain ⟨g, hg⟩ := encodeRange_sound heB hsB hpB hq
      let vals := interleave f g
      obtain ⟨ex, gA, ax⟩ := hf vals (interleave_even f g)
      obtain ⟨ey, gB, ay⟩ := hg vals (interleave_odd f g)
      obtain ⟨⟨x, y⟩, hxy, hd⟩ := List.any_eq_true.1 h
      obtain ⟨a, hxa, hya⟩ := zip_pair ex ey hxy hpq
      have gx := ax x (List.of_mem_zip hxy).1
      have gy := ay y (List.of_mem_zip hxy).2
      simp only [Bool.or_eq_true] at hd
      rcases hd with hd | hd
      · have := factQuery_sound (decideQ_sound hd) hC (fun _ _ => rfl) vals (gA.append gB)
          (Poly.allAtoms_sub (Poly.allAtoms_sub gy gx) (Poly.allAtoms_const 1))
        simp only [Poly.eval_sub, hxa, hya, Poly.eval_const] at this
        omega
      · have := factQuery_sound (decideQ_sound hd) hC (fun _ _ => rfl) vals (gA.append gB)
          (Poly.allAtoms_sub (Poly.allAtoms_sub gx gy) (Poly.allAtoms_const 1))
        simp only [Poly.eval_sub, hxa, hya, Poly.eval_const] at this
        omega
    · simp at h

theorem dimContains?_sound {C : Constraints} {hyps : List Ineq} {x : Poly} {d : Dim}
    (h : dimContains? C hyps x d = true) (hC : C.satisfies I ρ) (hs : ∀ s ∈ d.freeSyms, IntBound ρ s)
    (vals : Nat → Int) (hh : HoldsGood I ρ (envOf I ρ vals) hyps) (hx : x.AllAtoms (GoodAtom I ρ))
    {es : List Int} (hel : d.elems I ρ = some es) : x.eval (envOf I ρ vals) ∈ es := by
  have hlo : ∀ s ∈ d.lo.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Dim.freeSyms, h])
  have hhi : ∀ s ∈ d.hi.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Dim.freeSyms, h])
  have hst : ∀ s ∈ d.step.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Dim.freeSyms, h])
  obtain ⟨vlo, vhi, vst, hvlo, hvhi, hvst, hpos, hmem⟩ := Dim.mem_elems hel
  unfold dimContains? at h
  split at h
  · rename_i lo hi st hlop hhip hstp
    obtain ⟨elo, glo⟩ := poly_of_expr hlop hlo hvlo vals
    obtain ⟨ehi, ghi⟩ := poly_of_expr hhip hhi hvhi vals
    obtain ⟨est, -⟩ := poly_of_expr hstp hst hvst vals
    split at h
    · rename_i hidx
      simp only [Bool.and_eq_true] at h
      have h1 := factQuery_sound (decideQ_sound h.1) hC (fun _ _ => rfl) vals hh (Poly.allAtoms_sub hx glo)
      have h2 := factQuery_sound (decideQ_sound h.2) hC (fun _ _ => rfl) vals hh (Poly.allAtoms_sub glo hx)
      simp only [Poly.eval_sub, elo] at h1 h2
      have hd : d.lo = d.hi := by simpa [Dim.isIndex] using hidx
      rw [hd, hvhi] at hvlo; cases hvlo
      exact (hmem _).2 ⟨by omega, by omega, 0, le_rfl, by simp; omega⟩
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hc, h1⟩, h2⟩ := h
      have e1 := Poly.eval_of_const? (envOf I ρ vals) hc
      rw [est] at e1; subst e1
      have h1 := factQuery_sound (decideQ_sound h1) hC (fun _ _ => rfl) vals hh (Poly.allAtoms_sub hx glo)
      have h2 := factQuery_sound (decideQ_sound h2) hC (fun _ _ => rfl) vals hh (Poly.allAtoms_sub ghi hx)
      simp only [Poly.eval_sub, elo, ehi] at h1 h2
      exact (hmem _).2 ⟨by omega, by omega, x.eval (envOf I ρ vals) - vlo, by omega, by ring⟩
  · simp at h

theorem rangeSubset?_sound {C : Constraints} {rA rB : Range} (h : rangeSubset? C rA rB = true)
    (hC : C.satisfies I ρ) (hsA : ∀ s ∈ rA.freeSyms, IntBound ρ s) (hsB : ∀ s ∈ rB.freeSyms, IntBound ρ s)
    {ps qs : List Idx} (hpA : rA.points I ρ = some ps) (hpB : rB.points I ρ = some qs) {p : Idx}
    (hp : p ∈ ps) : p ∈ qs := by
  unfold rangeSubset? at h
  split at h
  · rename_i xs hA heA
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
    obtain ⟨hlen, hall⟩ := h
    obtain ⟨f, hf⟩ := encodeRange_sound heA hsA hpA hp
    obtain ⟨ex, gA, ax⟩ := hf (interleave f f) (interleave_even f f)
    have hdef : ∀ d ∈ rB.dims, ∃ es, d.elems I ρ = some es := by
      simp only [Range.points, bind, Option.bind_eq_some_iff] at hpB
      obtain ⟨ess, hess, -⟩ := hpB
      rw [List.mapM_option_some_iff] at hess
      intro d hd
      obtain ⟨es, -, h⟩ := List.Forall₂.exists_right_of_mem hess hd
      exact ⟨es, h⟩
    rw [Range.mem_points hpB]
    have key : ∀ (xs : List Poly) (p : Idx) (ds : List Dim), List.Forall₂ (fun x a => x.eval (envOf I ρ (interleave f f)) = a) xs p →
        xs.length = ds.length → (∀ xd ∈ xs.zip ds, dimContains? C hA xd.1 xd.2 = true) → (∀ x ∈ xs,
        x.AllAtoms (GoodAtom I ρ)) → (∀ d ∈ ds, ∃ es, d.elems I ρ = some es) → (∀ d ∈ ds, ∀ s ∈ d.freeSyms,
        IntBound ρ s) → List.Forall₂ (fun x d => ∃ es, d.elems I ρ = some es ∧ x ∈ es) p ds := by
      intro xs p ds hx
      induction hx generalizing ds with
      | nil => intro hl; cases ds <;> simp_all
      | @cons x a xs' p' he _ ih =>
        intro hl hc hga hde hds
        cases ds with
        | nil => simp at hl
        | cons d ds =>
          obtain ⟨es, hes⟩ := hde d (by simp)
          refine .cons ⟨es, hes, ?_⟩ (ih ds (by simpa using hl) (fun xd hxd => hc xd (by simp [hxd]))
            (fun y hy => hga y (List.mem_cons_of_mem _ hy)) (fun d' hd' => hde d' (List.mem_cons_of_mem _ hd'))
            (fun d' hd' => hds d' (List.mem_cons_of_mem _ hd')))
          rw [← he]
          exact dimContains?_sound (hc (x, d) (by simp)) hC (hds d (by simp)) (interleave f f) gA (hga x (by simp)) hes
    exact key xs p rB.dims ex hlen (fun xd hxd => hall xd hxd) ax hdef
      (fun d hd s hs => hsB s (List.mem_flatMap.2 ⟨d, hd, hs⟩))
  · simp at h

end SymSubset

section SubsetSound

open SymSubset

variable {I : ExprInterp} {ρ : Valuation}

/-- **Soundness of `disjoint?`**: no point of `A` is a point of `B`. -/
theorem Subset.disjoint?_sound {C : Constraints} {A B : Subset} (h : A.disjoint? C B = true)
    (hC : C.satisfies I ρ) {ps qs : List Idx} (hA : A.points I ρ = some ps) (hB : B.points I ρ = some qs) :
    ∀ p ∈ ps, p ∉ qs := by
  intro p hp hq
  simp only [Subset.disjoint?, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨hs, h⟩ := h
  have hs := intSymsOk_sound hs hC
  obtain ⟨rA, hrA, psA, hpA, hpA'⟩ := Subset.mem_points hA hp
  obtain ⟨rB, hrB, psB, hpB, hpB'⟩ := Subset.mem_points hB hq
  exact rangeDisjoint?_sound (h rA hrA rB hrB) hC
    (fun s h' => hs s (List.mem_append_left _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rA, hrA, h'⟩)))
    (fun s h' => hs s (List.mem_append_right _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rB, hrB, h'⟩)))
    hpA hpB hpA' hpB' rfl

/-- **Soundness of `subsetOf?`**: every point of `A` is a point of `B`
(when `B` is defined). -/
theorem Subset.subsetOf?_sound {C : Constraints} {A B : Subset} (h : A.subsetOf? C B = true)
    (hC : C.satisfies I ρ) {ps qs : List Idx} (hA : A.points I ρ = some ps) (hB : B.points I ρ = some qs) :
    ∀ p ∈ ps, p ∈ qs := by
  intro p hp
  simp only [Subset.subsetOf?, Bool.and_eq_true, List.all_eq_true, List.any_eq_true] at h
  obtain ⟨hs, h⟩ := h
  have hs := intSymsOk_sound hs hC
  obtain ⟨rA, hrA, psA, hpA, hpA'⟩ := Subset.mem_points hA hp
  obtain ⟨rB, hrB, hsub⟩ := h rA hrA
  obtain ⟨psB, hpB, hpB'⟩ := Subset.points_of_mem_ranges hB hrB
  exact hpB' p (rangeSubset?_sound hsub hC
    (fun s h' => hs s (List.mem_append_left _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rA, hrA, h'⟩)))
    (fun s h' => hs s (List.mem_append_right _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rB, hrB, h'⟩)))
    hpA hpB hpA')

end SubsetSound

/-! ## Distinct iterations of a map -/

/-- `ρ'` is `ρ` with the map parameters bound to the iteration point `i`
(as `Valuation.bindIdx` does for distinct parameters). -/
def Binds (ρ : Valuation) (params : List Sym) (i : Idx) (ρ' : Valuation) : Prop :=
  List.Forall₂ (fun p v => ρ' p = some (.int v)) params i ∧ ∀ s, s ∉ params → ρ' s = ρ s

/-- **Distinct points of a list are separated at their first difference.** -/
theorem exists_firstDiff : ∀ {i i' : List Int}, i.length = i'.length → i ≠ i' →
    ∃ k a a' r r', i.take k = i'.take k ∧ i.drop k = a :: r ∧ i'.drop k = a' :: r' ∧ a ≠ a'
  | [], [], _, h => absurd rfl h
  | x :: xs, y :: ys, hl, h => by
    by_cases hxy : x = y
    · subst hxy
      obtain ⟨k, a, a', r, r', h1, h2, h3, h4⟩ :=
        exists_firstDiff (i := xs) (i' := ys) (by simpa using hl) (fun e => h (by rw [e]))
      exact ⟨k + 1, a, a', r, r', by simp [h1], by simpa using h2, by simpa using h3, h4⟩
    · exact ⟨0, x, y, xs, ys, rfl, rfl, rfl, hxy⟩
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

namespace SymSubset

/-- Primed copy of a map parameter (`'` cannot occur in a Python identifier;
freshness is checked anyway). -/
def prime (s : Sym) : Sym := "'" ++ s

theorem prime_inj {a b : Sym} (h : prime a = prime b) : a = b := (String.append_right_inj "'").mp h

/-- Rename the parameters `ps` to their primed copies. -/
def primeSubst (ps : List Sym) : Sym → Option Expr := fun s => if s ∈ ps then some (.sym (prime s)) else none

/-- Polynomials `(param, lo, hi, step)` of the map range. -/
def rangePolys? : List Sym → List Dim → Option (List (Sym × Poly × Poly × Poly))
  | [], [] => some []
  | p :: ps, d :: ds => do
    let lo ← d.lo.toPoly?
    let hi ← d.hi.toPoly?
    let st ← d.step.toPoly?
    let rest ← rangePolys? ps ds
    pure ((p, lo, hi, st) :: rest)
  | _, _ => none

/-- `lo ≤ p ≤ hi` for every (renamed) parameter. -/
def rangeHyps (pr : Sym → Sym) (rp : List (Sym × Poly × Poly × Poly)) : List Ineq :=
  rp.flatMap fun (p, lo, hi, _) => [Ineq.le lo (Poly.var (.sym (pr p))), Ineq.le (Poly.var (.sym (pr p))) hi]

/-- The gap at the first differing parameter: `p + step ≤ p'` (forward) or
`p' + step ≤ p`. -/
def gapHyp (fwd : Bool) : List (Sym × Poly × Poly × Poly) → Option Ineq
  | (p, _, _, st) :: _ =>
    let x := Poly.var (.sym p)
    let x' := Poly.var (.sym (prime p))
    some (if fwd then Ineq.le (Poly.add x st) x' else Ineq.le (Poly.add x' st) x)
  | [] => none

/-! ### Index separation for in-bounds accesses -/

/-- Strides dominate the shape: the last stride is positive and every stride
covers the extent of the next dimension (row-major strides do). -/
def Dominates : List Int → List Int → Prop
  | [s], [_] => 1 ≤ s
  | s :: s' :: ss, _ :: n' :: ns => s' * n' ≤ s ∧ Dominates (s' :: ss) (n' :: ns)
  | _, _ => False

/-- Every coordinate is within `0 ≤ x < n`. -/
def InBounds (x sh : List Int) : Prop := List.Forall₂ (fun a n => 0 ≤ a ∧ a < n) x sh

/-- Goals `s_last − 1 ≥ 0` and `s − s'·n' ≥ 0` of `Dominates`. -/
def domGoals : List Poly → List Poly → Option (List Poly)
  | [s], [_] => some [Poly.sub s (Poly.const 1)]
  | s :: s' :: ss, _ :: n' :: ns => do
    let rest ← domGoals (s' :: ss) (n' :: ns)
    pure (Poly.sub s (Poly.mul s' n') :: rest)
  | _, _ => none

/-- Goals `x ≥ 0` and `n − x − 1 ≥ 0` of `InBounds`. -/
def boundGoals (xs sh : List Poly) : List Poly :=
  (xs.zip sh).flatMap fun (x, n) => [x, Poly.sub (Poly.sub n x) (Poly.const 1)]

/-- The polynomial only has integer facts symbols outside `bad` as atoms. -/
def symPolyOk (C : Constraints) (bad : List Sym) (p : Poly) : Bool :=
  p.atoms.all fun a => match a with
    | .sym s => s ∈ C.intSyms && s ∉ bad
    | _ => false

/-- Index separation (same descriptor): conditions (strides dominate the
shape, both points in bounds) and per-dimension separation alternatives. -/
def sepPart? (C : Constraints) (bad : List Sym) (hyps : List Ineq) (d : DataDesc) (xs ys : List Poly) :
    Option (List Query × List (Query × Query)) := do
  let ss ← d.effStrides.mapM Expr.toPoly?
  let sh ← d.shape.mapM Expr.toPoly?
  let dom ← domGoals ss sh
  if xs.length = sh.length ∧ ys.length = sh.length ∧ sh.all (symPolyOk C bad) = true then
    pure (dom.map (factQuery C bad []) ++ (boundGoals xs sh ++ boundGoals ys sh).map (factQuery C bad hyps),
      (xs.zip ys).map fun (x, y) =>
        (factQuery C bad hyps (Poly.sub (Poly.sub y x) (Poly.const 1)),
         factQuery C bad hyps (Poly.sub (Poly.sub x y) (Poly.const 1))))
  else none

/-- A case of the iteration analysis: first differing parameter `k`,
direction, and the ranges of both accesses. -/
structure IterCase where
  k : Nat
  fwd : Bool
  rA : Range
  rB : Range

def iterCases (n : Nat) (A B : Subset) : List IterCase :=
  (List.range n).flatMap fun k => [true, false].flatMap fun fwd =>
    A.ranges.flatMap fun rA => B.ranges.map fun rB => ⟨k, fwd, rA, rB⟩

/-- The queries of a case: offsets ordered `off' > off` (`up`) or
`off' < off` (`down`), or index separation (`sep`, same descriptor only). -/
structure CaseQ where
  fwd : Bool
  up : Query
  down : Query
  sep : Option (List Query × List (Query × Query))

/-- Either of two queries, the one of the case's direction first. -/
def CaseQ.either (c : CaseQ) (d : Nat) (a b : Query) : Bool :=
  if c.fwd then decideQ a d || decideQ b d else decideQ b d || decideQ a d

/-- Index separation succeeds with at most `d` case splits per query. -/
def CaseQ.sepDecided (c : CaseQ) (d : Nat) : Bool :=
  match c.sep with
  | some (conds, alts) => (alts.any fun (a, b) => c.either d a b) && conds.all (decideQ · d)
  | none => false

/-- One strategy succeeds with at most `d` case splits per query: index
separation, then the offset orderings (the case's direction first). -/
def CaseQ.decidedAt (c : CaseQ) (d : Nat) : Bool := c.sepDecided d || c.either d c.up c.down

/-- Without case splits first (failures are cheap), then with two levels. -/
def CaseQ.decided (c : CaseQ) : Bool := c.decidedAt 0 || c.decidedAt 2

/-- The query emitted for an undecided case (the ordering of the direction). -/
def CaseQ.obligation (c : CaseQ) : Query := (if c.fwd then c.up else c.down).withAxioms

/-- One strategy's queries are valid. -/
def CaseQ.Holds (c : CaseQ) : Prop :=
  c.up.withAxioms.Valid ∨ c.down.withAxioms.Valid ∨
    ∃ conds alts, c.sep = some (conds, alts) ∧ (∀ q ∈ conds, q.withAxioms.Valid) ∧
      ∃ ab ∈ alts, ab.1.withAxioms.Valid ∨ ab.2.withAxioms.Valid

theorem CaseQ.either_sound {c : CaseQ} {d : Nat} {a b : Query} (h : c.either d a b = true) :
    a.withAxioms.Valid ∨ b.withAxioms.Valid := by
  unfold CaseQ.either at h
  split at h <;> simp only [Bool.or_eq_true] at h <;> rcases h with h | h
  · exact Or.inl (decideQ_sound h)
  · exact Or.inr (decideQ_sound h)
  · exact Or.inr (decideQ_sound h)
  · exact Or.inl (decideQ_sound h)

theorem CaseQ.decidedAt_sound {c : CaseQ} {d : Nat} (h : c.decidedAt d = true) : c.Holds := by
  unfold CaseQ.decidedAt at h
  simp only [Bool.or_eq_true] at h
  rcases h with h | h
  · unfold CaseQ.sepDecided at h
    split at h
    · rename_i conds alts hs
      simp only [Bool.and_eq_true, List.all_eq_true, List.any_eq_true] at h
      obtain ⟨⟨ab, hab, h2⟩, h1⟩ := h
      exact Or.inr (Or.inr ⟨conds, alts, hs, fun q hq => decideQ_sound (h1 q hq), ab, hab, CaseQ.either_sound h2⟩)
    · simp at h
  · rcases CaseQ.either_sound h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

theorem CaseQ.decided_sound {c : CaseQ} (h : c.decided = true) : c.Holds := by
  unfold CaseQ.decided at h
  simp only [Bool.or_eq_true] at h
  rcases h with h | h
  · exact CaseQ.decidedAt_sound h
  · exact CaseQ.decidedAt_sound h

theorem CaseQ.obligation_sound {c : CaseQ} (h : c.obligation.Valid) : c.Holds := by
  unfold CaseQ.obligation at h
  split at h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)

/-- The queries of a case. -/
def caseQueries? (C : Constraints) (params : List Sym) (rp : List (Sym × Poly × Poly × Poly))
    (dA dB : DataDesc) (c : IterCase) : Option CaseQ := do
  let bad := params ++ params.map prime
  let gap ← gapHyp c.fwd (rp.drop c.k)
  let (xs, hA) ← encodeRange c.rA 0
  let (ys, hB) ← encodeRange (c.rB.subst (primeSubst (params.drop c.k))) 1
  let oA ← offsetPoly? dA xs
  let oB ← offsetPoly? dB ys
  let hyps := rangeHyps id rp ++ rangeHyps prime (rp.drop c.k) ++ [gap] ++ hA ++ hB
  pure { fwd := c.fwd
         up := factQuery C bad hyps (Poly.sub (Poly.sub oB oA) (Poly.const 1))
         down := factQuery C bad hyps (Poly.sub (Poly.sub oA oB) (Poly.const 1))
         sep := if dA = dB then sepPart? C bad hyps dA xs ys else none }

/-- Syntactic side conditions: one parameter per dimension; range and
descriptor symbols are integer facts symbols distinct from the parameters
and their primed copies; access symbols are integer symbols or parameters. -/
def iterChecks (C : Constraints) (params : List Sym) (rng : Range) (dA : DataDesc) (A : Subset) (dB : DataDesc)
    (B : Subset) : Bool :=
  let pr := params.map prime
  params.length == rng.dims.length && params.all (· ∉ pr) &&
    (rng.freeSyms ++ offSyms dA ++ offSyms dB).all (fun s => s ∈ C.intSyms && s ∉ params ++ pr) &&
    (A.freeSyms ++ B.freeSyms).all (fun s => (s ∈ C.intSyms || s ∈ params) && s ∉ pr)

/-- All case queries, `none` if the accesses are outside the supported
fragment. -/
def iterQueries? (C : Constraints) (params : List Sym) (rng : Range) (dA : DataDesc) (A : Subset) (dB : DataDesc)
    (B : Subset) : Option (List CaseQ) :=
  if iterChecks C params rng dA A dB B then do
    let rp ← rangePolys? params rng.dims
    (iterCases params.length A B).mapM (caseQueries? C params rp dA dB)
  else none

end SymSubset

open SymSubset in
/-- **Assumed-queries mode**: the queries (with atom axioms) that remain
undecided; `none` outside the supported fragment.  Each listed query is a
closed statement over ℤ (`Query.toLeanFile`). -/
def iterObligations? (C : Constraints) (params : List Sym) (rng : Range) (dA : DataDesc) (A : Subset)
    (dB : DataDesc) (B : Subset) : Option (List Query) :=
  (iterQueries? C params rng dA A dB B).map fun qs => qs.filterMap fun c =>
    if c.decided then none else some c.obligation

/-- Two distinct iterations of the map `params ∈ rng` never touch a common
linear offset through access `A` (descriptor `dA`) in one iteration and `B`
(descriptor `dB`) in the other. -/
def iterDisjoint? (C : Constraints) (params : List Sym) (rng : Range) (dA : DataDesc) (A : Subset)
    (dB : DataDesc) (B : Subset) : Bool :=
  match iterObligations? C params rng dA A dB B with
  | some [] => true
  | _ => false

namespace SymSubset

/-! ### Lemmas for the iteration analysis -/

theorem forall₂_imp_mem {α β : Type} {R S : α → β → Prop} : ∀ {l₁ : List α} {l₂ : List β},
    List.Forall₂ R l₁ l₂ → (∀ a ∈ l₁, ∀ b, R a b → S a b) → List.Forall₂ S l₁ l₂
  | _, _, .nil, _ => .nil
  | _, _, .cons h hs, H => .cons (H _ (by simp) _ h) (forall₂_imp_mem hs fun a ha => H a (List.mem_cons_of_mem _ ha))

theorem forall₂_agree {α β : Type} {R S : α → β → Prop} : ∀ {l₁ : List α} {l₂ : List β},
    List.Forall₂ R l₁ l₂ → List.Forall₂ S l₁ l₂ → ∀ a ∈ l₁, ∃ b, R a b ∧ S a b
  | _, _, .nil, _, a, ha => by simp at ha
  | _, _, .cons h hs, .cons h' hs', a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨_, h, h'⟩
    · exact forall₂_agree hs hs' a ha

theorem rangePolys?_drop : ∀ (k : Nat) (ps : List Sym) (ds : List Dim) (rp : List (Sym × Poly × Poly × Poly)),
    rangePolys? ps ds = some rp → rangePolys? (ps.drop k) (ds.drop k) = some (rp.drop k)
  | 0, _, _, _, h => h
  | _ + 1, [], [], rp, h => by simp [rangePolys?] at h; subst h; rfl
  | k + 1, p :: ps, d :: ds, rp, h => by
    simp only [rangePolys?, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨lo, -, hi, -, st, -, rest, hr, rfl⟩ := h
    simpa using rangePolys?_drop k ps ds rest hr
  | _ + 1, [], _ :: _, _, h => by simp [rangePolys?] at h
  | _ + 1, _ :: _, [], _, h => by simp [rangePolys?] at h

theorem Range.freeSyms_subst {σ : Sym → Option Expr} {r : Range} {x : Sym} (h : x ∈ (r.subst σ).freeSyms) :
    ∃ s ∈ r.freeSyms, x ∈ (Expr.substOf σ s).freeSyms := by
  simp only [Range.subst, Range.freeSyms, List.flatMap_map, List.mem_flatMap] at h
  obtain ⟨d, hd, hx⟩ := h
  have : ∃ s ∈ d.freeSyms, x ∈ (Expr.substOf σ s).freeSyms := by
    simp only [Dim.subst, Dim.freeSyms, List.mem_append] at hx
    rcases hx with ((hx | hx) | hx) | hx <;>
      obtain ⟨s, hs, hx⟩ := Expr.freeSyms_subst _ hx <;> exact ⟨s, by simp [Dim.freeSyms, hs], hx⟩
  obtain ⟨s, hs, hx⟩ := this
  exact ⟨s, List.mem_flatMap.2 ⟨d, hd, hs⟩, hx⟩

/-- The valuation of a case: parameters as in the first iteration, primed
copies of the parameters `ps` as in the second. -/
def primedVal (ρi ρi' : Valuation) (ps : List Sym) : Valuation := fun s =>
  match ps.find? (fun t => prime t == s) with
  | some t => ρi' t
  | none => ρi s

theorem primedVal_prime {ρi ρi' : Valuation} {ps : List Sym} {t : Sym} (ht : t ∈ ps) :
    primedVal ρi ρi' ps (prime t) = ρi' t := by
  unfold primedVal
  split
  · rename_i t' h
    have := List.find?_some h
    simp only [beq_iff_eq] at this
    rw [prime_inj this]
  · rename_i h
    exact absurd (by simp) (List.find?_eq_none.1 h t ht)

theorem primedVal_other {ρi ρi' : Valuation} {ps : List Sym} {s : Sym} (hs : ∀ t ∈ ps, prime t ≠ s) :
    primedVal ρi ρi' ps s = ρi s := by
  unfold primedVal
  split
  · rename_i t' h
    have := List.find?_some h
    simp only [beq_iff_eq] at this
    exact absurd this (hs t' (List.mem_of_find?_eq_some h))
  · rfl

variable {I : ExprInterp} {ρ : Valuation}

theorem rangeHyps_sound {ρ₂ : Valuation} (pr : Sym → Sym) (vals : Nat → Int) :
    ∀ (ps : List Sym) (ds : List Dim) (rp : List (Sym × Poly × Poly × Poly)) (vs : List Int),
    rangePolys? ps ds = some rp → List.Forall₂ (fun p v => ρ₂ (pr p) = some (.int v)) ps vs →
    List.Forall₂ (fun v d => ∃ es, d.elems I ρ = some es ∧ v ∈ es) vs ds →
    (∀ d ∈ ds, ∀ s ∈ d.freeSyms, ρ s = ρ₂ s ∧ IntBound ρ s) →
    HoldsGood I ρ₂ (envOf I ρ₂ vals) (rangeHyps pr rp)
  | [], [], rp, _, h, _, _, _ => by simp [rangePolys?] at h; subst h; exact HoldsGood.nil
  | p :: ps, d :: ds, rp, v :: vs, h, .cons hp hps, .cons hv hvs, hsy => by
    simp only [rangePolys?, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨lo, hlo, hi, hhi, st, -, rest, hr, rfl⟩ := h
    have ih := rangeHyps_sound pr vals ps ds rest vs hr hps hvs (fun d' hd' => hsy d' (List.mem_cons_of_mem _ hd'))
    obtain ⟨es, hes, hve⟩ := hv
    obtain ⟨vlo, vhi, vst, hvlo, hvhi, -, -, hmem⟩ := Dim.mem_elems hes
    obtain ⟨h1, h2, -⟩ := (hmem v).1 hve
    have hs := hsy d (by simp)
    have bnd : ∀ e : Expr, (∀ s ∈ e.freeSyms, s ∈ d.freeSyms) → ∀ w, e.evalInt I ρ = some w →
        e.evalInt I ρ₂ = some w ∧ ∀ s ∈ e.freeSyms, IntBound ρ₂ s := by
      intro e he w hw
      refine ⟨by rw [← Expr.evalInt_congr (fun s hs' => (hs s (he s hs')).1)]; exact hw, fun s hs' => ?_⟩
      obtain ⟨n, hn⟩ := (hs s (he s hs')).2
      exact ⟨n, (hs s (he s hs')).1 ▸ hn⟩
    obtain ⟨elo, ilo⟩ := bnd d.lo (fun s h => by simp [Dim.freeSyms, h]) vlo hvlo
    obtain ⟨ehi, ihi⟩ := bnd d.hi (fun s h => by simp [Dim.freeSyms, h]) vhi hvhi
    obtain ⟨plo, glo⟩ := poly_of_expr hlo ilo elo vals
    obtain ⟨phi, ghi⟩ := poly_of_expr hhi ihi ehi vals
    obtain ⟨px, gx⟩ := sym_poly (I := I) hp vals
    refine HoldsGood.append ?_ ih
    intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact ⟨by simp only [Ineq.holds_le, plo, px]; exact h1, Ineq.le_atoms glo gx⟩
    · exact ⟨by simp only [Ineq.holds_le, phi, px]; exact h2, Ineq.le_atoms gx ghi⟩
  | [], _ :: _, _, _, h, _, _, _ => by simp [rangePolys?] at h
  | _ :: _, [], _, _, h, _, _, _ => by simp [rangePolys?] at h
  | _ :: _, _ :: _, _, [], _, h, _, _ => by cases h

/-! ### Index separation -/

theorem foldl_add_eq (l : List Int) (c : Int) : l.foldl (· + ·) c = c + l.foldl (· + ·) 0 := by
  induction l generalizing c with
  | nil => simp
  | cons a l ih => simp only [List.foldl_cons]; rw [ih (c + a), ih (0 + a)]; ring

theorem linearize_cons (a s : Int) (xs ss : List Int) :
    Range.linearize (a :: xs) (s :: ss) = a * s + Range.linearize xs ss := by
  simp only [Range.linearize, List.zip_cons_cons, List.map_cons, List.foldl_cons]
  rw [foldl_add_eq]; ring

@[simp] theorem linearize_nil (ss : List Int) : Range.linearize [] ss = 0 := by simp [Range.linearize]

/-- Linear offsets of in-bounds points under dominating strides lie in
`[0, s·n)`. -/
theorem dominates_bound : ∀ (ss : List Int) (s n : Int) (ns x : List Int), Dominates (s :: ss) (n :: ns) →
    InBounds x (n :: ns) → 1 ≤ s ∧ 0 ≤ Range.linearize x (s :: ss) ∧ Range.linearize x (s :: ss) ≤ s * n - 1
  | [], s, n, ns, x, h, hx => by
    cases ns with
    | cons _ _ => simp [Dominates] at h
    | nil =>
      simp only [Dominates] at h
      cases hx with
      | @cons a _ x' _ ha hx' =>
        cases hx'
        simp only [linearize_cons, linearize_nil, add_zero]
        have := mul_le_mul_of_nonneg_right (show a ≤ n - 1 by omega) (show (0 : Int) ≤ s by omega)
        exact ⟨h, mul_nonneg ha.1 (by omega), by linarith⟩
  | s' :: ss, s, n, ns, x, h, hx => by
    cases ns with
    | nil => simp [Dominates] at h
    | cons n' ns =>
      simp only [Dominates] at h
      obtain ⟨h1, h2⟩ := h
      cases hx with
      | @cons a _ x' _ ha hx' =>
        obtain ⟨i1, i2, i3⟩ := dominates_bound ss s' n' ns x' h2 hx'
        have hn' : 1 ≤ n' := by
          cases hx' with
          | cons hb _ => omega
        have hs : 1 ≤ s := by nlinarith
        rw [linearize_cons]
        have := mul_le_mul_of_nonneg_right (show a ≤ n - 1 by omega) (show (0 : Int) ≤ s by omega)
        have : 0 ≤ a * s := mul_nonneg ha.1 (by omega)
        exact ⟨hs, by linarith, by nlinarith⟩

/-- **Strided linearization is injective on in-bounds points** when the
strides dominate the shape. -/
theorem dominates_inj : ∀ (ss : List Int) (s n : Int) (ns x y : List Int), Dominates (s :: ss) (n :: ns) →
    InBounds x (n :: ns) → InBounds y (n :: ns) →
    Range.linearize x (s :: ss) = Range.linearize y (s :: ss) → x = y
  | ss, s, n, ns, x, y, h, hx, hy, he => by
    obtain ⟨hs, -, -⟩ := dominates_bound ss s n ns x h hx
    cases hx with
    | @cons a _ x' _ ha hx' =>
    cases hy with
    | @cons b _ y' _ hb hy' =>
    rw [linearize_cons, linearize_cons] at he
    cases ss with
    | nil =>
      cases ns with
      | cons _ _ => simp [Dominates] at h
      | nil =>
        cases hx'; cases hy'
        simp only [linearize_nil, add_zero] at he
        rw [mul_right_cancel₀ (show s ≠ 0 by omega) he]
    | cons s' ss =>
      cases ns with
      | nil => simp [Dominates] at h
      | cons n' ns =>
        simp only [Dominates] at h
        obtain ⟨h1, h2⟩ := h
        obtain ⟨i1, l1, u1⟩ := dominates_bound ss s' n' ns x' h2 hx'
        obtain ⟨-, l2, u2⟩ := dominates_bound ss s' n' ns y' h2 hy'
        have hab : a = b := by
          rcases lt_trichotomy a b with hlt | heq | hgt
          · have := mul_le_mul_of_nonneg_right (show a + 1 ≤ b by omega) (show (0 : Int) ≤ s by omega)
            nlinarith
          · exact heq
          · have := mul_le_mul_of_nonneg_right (show b + 1 ≤ a by omega) (show (0 : Int) ≤ s by omega)
            nlinarith
        subst hab
        rw [dominates_inj ss s' n' ns x' y' h2 hx' hy' (by linarith)]

theorem dominates_inj' {st sh x y : List Int} (h : Dominates st sh) (hx : InBounds x sh) (hy : InBounds y sh)
    (he : Range.linearize x st = Range.linearize y st) : x = y := by
  cases st with
  | nil => cases sh <;> simp [Dominates] at h
  | cons s ss =>
    cases sh with
    | nil => cases ss <;> simp [Dominates] at h
    | cons n ns => exact dominates_inj ss s n ns x y h hx hy he

theorem domGoals_sound (env : Env) : ∀ (ss sh gs : List Poly) (stv : List Int),
    domGoals ss sh = some gs → (∀ g ∈ gs, 0 ≤ g.eval env) → List.Forall₂ (fun p v => p.eval env = v) ss stv →
    Dominates stv (sh.map (Poly.eval env))
  | [s], [n], gs, stv, h, hg, hs => by
    simp only [domGoals, Option.some.injEq] at h
    subst h
    cases hs with
    | cons e hs' =>
      cases hs'
      have := hg (Poly.sub s (Poly.const 1)) (by simp)
      simp only [Poly.eval_sub, Poly.eval_const] at this
      simp only [List.map_cons, List.map_nil, Dominates]
      omega
  | s :: s' :: ss, n :: n' :: ns, gs, stv, h, hg, hs => by
    simp only [domGoals, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨rest, hr, rfl⟩ := h
    cases hs with
    | cons e hs' =>
      cases hs' with
      | cons e' hs'' =>
        have ih := domGoals_sound env (s' :: ss) (n' :: ns) rest _ hr
          (fun g hg' => hg g (List.mem_cons_of_mem _ hg')) (.cons e' hs'')
        have := hg _ List.mem_cons_self
        simp only [Poly.eval_sub, Poly.eval_mul] at this
        simp only [List.map_cons, Dominates] at ih ⊢
        exact ⟨by rw [← e, ← e']; linarith, ih⟩
  | [], _, _, _, h, _, _ => by simp [domGoals] at h
  | [_], [], _, _, h, _, _ => by simp [domGoals] at h
  | [_], _ :: _ :: _, _, _, h, _, _ => by simp [domGoals] at h
  | _ :: _ :: _, [], _, _, h, _, _ => by simp [domGoals] at h
  | _ :: _ :: _, [_], _, _, h, _, _ => by simp [domGoals] at h

theorem domGoals_atoms {P : Atom → Prop} : ∀ (ss sh gs : List Poly), domGoals ss sh = some gs →
    (∀ p ∈ ss, p.AllAtoms P) → (∀ p ∈ sh, p.AllAtoms P) → ∀ g ∈ gs, g.AllAtoms P
  | [s], [n], gs, h, hs, _ => by
    simp only [domGoals, Option.some.injEq] at h
    subst h
    intro g hg
    simp only [List.mem_singleton] at hg
    subst hg
    exact Poly.allAtoms_sub (hs s (by simp)) (Poly.allAtoms_const 1)
  | s :: s' :: ss, n :: n' :: ns, gs, h, hs, hn => by
    simp only [domGoals, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨rest, hr, rfl⟩ := h
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact Poly.allAtoms_sub (hs s (by simp)) (Poly.allAtoms_mul (hs s' (by simp)) (hn n' (by simp)))
    · exact domGoals_atoms (s' :: ss) (n' :: ns) rest hr (fun p hp => hs p (List.mem_cons_of_mem _ hp))
        (fun p hp => hn p (List.mem_cons_of_mem _ hp)) g hg
  | [], _, _, h, _, _ => by simp [domGoals] at h
  | [_], [], _, h, _, _ => by simp [domGoals] at h
  | [_], _ :: _ :: _, _, h, _, _ => by simp [domGoals] at h
  | _ :: _ :: _, [], _, h, _, _ => by simp [domGoals] at h
  | _ :: _ :: _, [_], _, h, _, _ => by simp [domGoals] at h

theorem boundGoals_cons (x n : Poly) (xs sh : List Poly) :
    boundGoals (x :: xs) (n :: sh) = x :: Poly.sub (Poly.sub n x) (Poly.const 1) :: boundGoals xs sh := by
  simp [boundGoals]

theorem boundGoals_sound (env : Env) : ∀ (xs sh : List Poly) (p : List Int),
    (∀ g ∈ boundGoals xs sh, 0 ≤ g.eval env) → List.Forall₂ (fun x a => x.eval env = a) xs p →
    xs.length = sh.length → InBounds p (sh.map (Poly.eval env))
  | [], [], [], _, _, _ => .nil
  | x :: xs, n :: sh, a :: p, hg, .cons e hxs, hl => by
    rw [boundGoals_cons] at hg
    have h1 := hg x (by simp)
    have h2 := hg _ (List.mem_cons_of_mem _ List.mem_cons_self)
    simp only [Poly.eval_sub, Poly.eval_const] at h2
    exact .cons ⟨by omega, by omega⟩ (boundGoals_sound env xs sh p
      (fun g hg' => hg g (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hg'))) hxs (by simpa using hl))
  | [], _ :: _, _, _, _, hl => by simp at hl
  | _ :: _, [], _, _, _, hl => by simp at hl
  | [], [], _ :: _, _, h, _ => by cases h
  | _ :: _, _ :: _, [], _, h, _ => by cases h

theorem boundGoals_atoms {P : Atom → Prop} {xs sh : List Poly} (hx : ∀ p ∈ xs, p.AllAtoms P)
    (hn : ∀ p ∈ sh, p.AllAtoms P) : ∀ g ∈ boundGoals xs sh, g.AllAtoms P := by
  intro g hg
  simp only [boundGoals, List.mem_flatMap] at hg
  obtain ⟨⟨x, n⟩, hxn, hg⟩ := hg
  have hx' := hx x (List.of_mem_zip hxn).1
  have hn' := hn n (List.of_mem_zip hxn).2
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl
  · exact hx'
  · exact Poly.allAtoms_sub (Poly.allAtoms_sub hn' hx') (Poly.allAtoms_const 1)

theorem symPolyOk_good {C : Constraints} {bad : List Sym} {ρ₂ : Valuation} {p : Poly}
    (h : symPolyOk C bad p = true) (hi : ∀ s ∈ C.intSyms, s ∉ bad → IntBound ρ₂ s) :
    p.AllAtoms (GoodAtom I ρ₂) := by
  rw [Poly.allAtoms_iff]
  intro a ha
  have := List.all_eq_true.1 h a ha
  cases a with
  | sym s =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at this
    obtain ⟨n, hn⟩ := hi s this.1 this.2
    exact Or.inr (sym_atomOk hn).1
  | opq _ => simp at this
  | var _ => simp at this

theorem polys_of_exprs {ρ₂ : Valuation} (vals : Nat → Int) : ∀ (es : List Expr) (ps : List Poly) (vs : List Int),
    (∀ e ∈ es, ∀ s ∈ e.freeSyms, IntBound ρ₂ s) →
    List.Forall₂ (fun e p => e.toPoly? = some p) es ps → List.Forall₂ (fun e v => e.evalInt I ρ₂ = some v) es vs →
    List.Forall₂ (fun p a => p.eval (envOf I ρ₂ vals) = a) ps vs ∧ ∀ p ∈ ps, p.AllAtoms (GoodAtom I ρ₂)
  | [], [], [], _, _, _ => ⟨.nil, by simp⟩
  | e :: es, p :: ps, v :: vs, hsy, .cons hp h1, .cons hv h2 => by
    obtain ⟨e1, a1⟩ := poly_of_expr hp (hsy e (by simp)) hv vals
    obtain ⟨e2, a2⟩ := polys_of_exprs vals es ps vs (fun e' he => hsy e' (List.mem_cons_of_mem _ he)) h1 h2
    refine ⟨.cons e1 e2, fun p' hp' => ?_⟩
    rcases List.mem_cons.1 hp' with rfl | hp'
    · exact a1
    · exact a2 p' hp'
  | [], _ :: _, _, _, h, _ => by cases h
  | _ :: _, [], _, _, h, _ => by cases h
  | [], [], _ :: _, _, _, h => by cases h
  | _ :: _, _ :: _, [], _, _, h => by cases h

/-- **Index separation**: when strides dominate the shape and both points are
in bounds, points separated in some dimension have distinct offsets. -/
theorem sepPart?_sound {C : Constraints} {bad : List Sym} {hyps : List Ineq} {d : DataDesc} {xs ys : List Poly}
    {conds : List Query} {alts : List (Query × Query)} (hs : sepPart? C bad hyps d xs ys = some (conds, alts))
    (hconds : ∀ q ∈ conds, q.withAxioms.Valid) (halt : ∃ ab ∈ alts, ab.1.withAxioms.Valid ∨ ab.2.withAxioms.Valid)
    (hC : C.satisfies I ρ) {ρ₂ : Valuation} (hag : ∀ s, s ∉ bad → ρ s = ρ₂ s) (vals : Nat → Int)
    (hh : HoldsGood I ρ₂ (envOf I ρ₂ vals) hyps) {p q : Idx}
    (hx : List.Forall₂ (fun x a => x.eval (envOf I ρ₂ vals) = a) xs p)
    (hy : List.Forall₂ (fun x a => x.eval (envOf I ρ₂ vals) = a) ys q)
    (gx : ∀ x ∈ xs, x.AllAtoms (GoodAtom I ρ₂)) (gy : ∀ y ∈ ys, y.AllAtoms (GoodAtom I ρ₂))
    (hsy : ∀ s ∈ offSyms d, IntBound ρ₂ s) {o o' : Int} (ho : d.linearOffset I ρ₂ p = some o)
    (ho' : d.linearOffset I ρ₂ q = some o') : o ≠ o' := by
  unfold sepPart? at hs
  simp only [bind, Option.bind_eq_some_iff, pure] at hs
  obtain ⟨ss, hss, sh, -, dom, hdom, hs⟩ := hs
  split at hs
  · rename_i hcond
    obtain ⟨hlx, hly, hshok⟩ := hcond
    simp only [Option.some.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    unfold DataDesc.linearOffset at ho ho'
    simp only [bind, Option.bind_eq_some_iff, pure] at ho ho'
    obtain ⟨stv, hstv, start, hstart, ho⟩ := ho
    obtain ⟨stv', hstv', start', hstart', ho'⟩ := ho'
    rw [hstv, Option.some.injEq] at hstv'
    rw [hstart, Option.some.injEq] at hstart'
    subst hstv' hstart'
    split at ho
    · split at ho'
      · simp only [Option.some.injEq] at ho ho'
        subst ho ho'
        rw [List.mapM_option_some_iff] at hss hstv
        obtain ⟨es, ea⟩ := polys_of_exprs vals _ _ _
          (fun e he s hs => hsy s (List.mem_append_left _ (List.mem_flatMap.2 ⟨e, he, hs⟩))) hss hstv
        have gsh : ∀ n ∈ sh, n.AllAtoms (GoodAtom I ρ₂) := fun n hn =>
          symPolyOk_good (List.all_eq_true.1 hshok n hn) (fun s hs hb => by
            obtain ⟨k, hk⟩ := hC.1 s hs
            exact ⟨k, by rw [← hag s hb]; exact hk⟩)
        have hdg : ∀ g ∈ dom, 0 ≤ g.eval (envOf I ρ₂ vals) := fun g hg =>
          factQuery_sound (hconds _ (List.mem_append_left _ (List.mem_map.2 ⟨g, hg, rfl⟩))) hC hag vals
            HoldsGood.nil (domGoals_atoms _ _ _ hdom ea gsh g hg)
        have hD := domGoals_sound _ ss sh dom stv hdom hdg es
        have hbg : ∀ g ∈ boundGoals xs sh ++ boundGoals ys sh, 0 ≤ g.eval (envOf I ρ₂ vals) := fun g hg =>
          factQuery_sound (hconds _ (List.mem_append_right _ (List.mem_map.2 ⟨g, hg, rfl⟩))) hC hag vals hh (by
            rcases List.mem_append.1 hg with hg | hg
            · exact boundGoals_atoms gx gsh g hg
            · exact boundGoals_atoms gy gsh g hg)
        have hbx := boundGoals_sound _ xs sh p (fun g hg => hbg g (List.mem_append_left _ hg)) hx hlx
        have hby := boundGoals_sound _ ys sh q (fun g hg => hbg g (List.mem_append_right _ hg)) hy hly
        obtain ⟨ab, hab, hv⟩ := halt
        obtain ⟨⟨x, y⟩, hxy, rfl⟩ := List.mem_map.1 hab
        intro heq
        have hpq := dominates_inj' hD hbx hby (by omega)
        obtain ⟨a, hxa, hya⟩ := zip_pair hx hy hxy hpq
        have gx' := gx x (List.of_mem_zip hxy).1
        have gy' := gy y (List.of_mem_zip hxy).2
        rcases hv with hv | hv
        · have := factQuery_sound hv hC hag vals hh
            (Poly.allAtoms_sub (Poly.allAtoms_sub gy' gx') (Poly.allAtoms_const 1))
          simp only [Poly.eval_sub, hxa, hya, Poly.eval_const] at this
          omega
        · have := factQuery_sound hv hC hag vals hh
            (Poly.allAtoms_sub (Poly.allAtoms_sub gx' gy') (Poly.allAtoms_const 1))
          simp only [Poly.eval_sub, hxa, hya, Poly.eval_const] at this
          omega
      · simp at ho'
    · simp at ho
  · simp at hs

/-- **The core of `iterDisjoint?`**: in one case of the analysis, a valid
alternative separates the two offsets. -/
theorem caseQueries?_sound {C : Constraints} {params : List Sym} {rng : Range} {dA dB : DataDesc}
    {rp : List (Sym × Poly × Poly × Poly)} {c : IterCase} {cq : CaseQ}
    (hcq : caseQueries? C params rp dA dB c = some cq) (hrp : rangePolys? params rng.dims = some rp)
    (hpr : ∀ s ∈ params, s ∉ params.map prime)
    (hrs : ∀ s ∈ rng.freeSyms ++ offSyms dA ++ offSyms dB, s ∈ C.intSyms ∧ s ∉ params ++ params.map prime)
    (hab : ∀ s ∈ c.rA.freeSyms ++ c.rB.freeSyms, (s ∈ C.intSyms ∨ s ∈ params) ∧ s ∉ params.map prime)
    (hC : C.satisfies I ρ) {i i' : Idx}
    (hi : List.Forall₂ (fun x d => ∃ es, d.elems I ρ = some es ∧ x ∈ es) i rng.dims)
    (hi' : List.Forall₂ (fun x d => ∃ es, d.elems I ρ = some es ∧ x ∈ es) i' rng.dims)
    {a a' : Int} {r r' : List Int} (htake : i.take c.k = i'.take c.k) (hdrop : i.drop c.k = a :: r)
    (hdrop' : i'.drop c.k = a' :: r') (hdir : if c.fwd then a < a' else a' < a)
    {ρi ρi' : Valuation} (hb : Binds ρ params i ρi) (hb' : Binds ρ params i' ρi')
    {psA psB : List Idx} (hpA : c.rA.points I ρi = some psA) (hpB : c.rB.points I ρi' = some psB)
    {p q : Idx} (hp : p ∈ psA) (hq : q ∈ psB) {o o' : Int} (ho : dA.linearOffset I ρi p = some o)
    (ho' : dB.linearOffset I ρi' q = some o') (hv : cq.Holds) : o ≠ o' := by
  -- the case valuation
  let P := params.drop c.k
  let ρ₂ := primedVal ρi ρi' P
  have hPsub : ∀ t ∈ P, t ∈ params := fun t ht => List.mem_of_mem_drop ht
  have notPrime : ∀ s, s ∉ params.map prime → ∀ t ∈ P, prime t ≠ s := by
    intro s hs t ht he
    exact hs (List.mem_map.2 ⟨t, hPsub t ht, he⟩)
  have ρ₂_other : ∀ s, s ∉ params.map prime → ρ₂ s = ρi s := fun s hs => primedVal_other (notPrime s hs)
  have ρ₂_prime : ∀ t ∈ P, ρ₂ (prime t) = ρi' t := fun t ht => primedVal_prime ht
  have hbad : ∀ s, s ∉ params ++ params.map prime → ρ s = ρ₂ s := by
    intro s hs
    simp only [List.mem_append, not_or] at hs
    rw [ρ₂_other s hs.2, hb.2 s hs.1]
  have intρi : ∀ {ρ' : Valuation} {j : Idx}, Binds ρ params j ρ' → ∀ s, s ∈ C.intSyms ∨ s ∈ params →
      IntBound ρ' s := by
    intro ρ' j hb s hs
    by_cases hsp : s ∈ params
    · obtain ⟨v, -, hv⟩ := List.Forall₂.exists_right_of_mem hb.1 hsp
      exact ⟨v, hv⟩
    · obtain ⟨n, hn⟩ := hC.1 s (hs.resolve_right hsp)
      exact ⟨n, by rw [hb.2 s hsp]; exact hn⟩
  -- unfold the case
  unfold caseQueries? at hcq
  simp only [bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at hcq
  obtain ⟨gap, hgap, ⟨xs, hA⟩, heA, ⟨ys, hB⟩, heB, oA, hoA, oB, hoB, hcq⟩ := hcq
  -- access A under ρ₂
  have hagA : ∀ s ∈ c.rA.freeSyms, ρi s = ρ₂ s := fun s hs =>
    (ρ₂_other s (hab s (List.mem_append_left _ hs)).2).symm
  have hpA₂ : c.rA.points I ρ₂ = some psA := by rw [← Range.points_congr hagA]; exact hpA
  have hsA : ∀ s ∈ c.rA.freeSyms, IntBound ρ₂ s := fun s hs => by
    obtain ⟨n, hn⟩ := intρi hb s (hab s (List.mem_append_left _ hs)).1
    exact ⟨n, (hagA s hs) ▸ hn⟩
  obtain ⟨f, hf⟩ := encodeRange_sound heA hsA hpA₂ hp
  -- access B (renamed) under ρ₂
  let σ := primeSubst P
  have hσ : ∀ s ∈ c.rB.freeSyms, (Expr.substOf σ s).eval I ρ₂ = ρi' s := by
    intro s hs
    have hs' := hab s (List.mem_append_right _ hs)
    by_cases hsP : s ∈ P
    · simp only [Expr.substOf, σ, primeSubst, hsP, ↓reduceIte, Option.getD_some, Expr.eval]
      exact ρ₂_prime s hsP
    · simp only [Expr.substOf, σ, primeSubst, hsP, ↓reduceIte, Option.getD_none, Expr.eval]
      rw [ρ₂_other s hs'.2]
      by_cases hsp : s ∈ params
      · have hst : s ∈ params.take c.k := by
          rw [← List.take_append_drop c.k params] at hsp
          rcases List.mem_append.1 hsp with h | h
          · exact h
          · exact absurd h hsP
        have h1 := List.forall₂_take c.k hb.1
        have h2 := List.forall₂_take c.k hb'.1
        rw [← htake] at h2
        obtain ⟨v, e1, e2⟩ := forall₂_agree h1 h2 s hst
        rw [e1, e2]
      · rw [hb.2 s hsp, hb'.2 s hsp]
  have hσsym : ∀ s, ∃ y, Expr.substOf σ s = .sym y := by
    intro s
    by_cases hsP : s ∈ P
    · exact ⟨prime s, by simp [Expr.substOf, σ, primeSubst, hsP]⟩
    · exact ⟨s, by simp [Expr.substOf, σ, primeSubst, hsP]⟩
  have hpB₂ : (c.rB.subst σ).points I ρ₂ = some psB := by rw [Range.points_subst hσ]; exact hpB
  have hsB : ∀ x ∈ (c.rB.subst σ).freeSyms, IntBound ρ₂ x := by
    intro x hx
    obtain ⟨s, hs, hx⟩ := Range.freeSyms_subst hx
    obtain ⟨y, hy⟩ := hσsym s
    have he := hσ s hs
    rw [hy] at hx he
    simp only [Expr.freeSyms, List.mem_singleton] at hx
    subst hx
    obtain ⟨n, hn⟩ := intρi hb' s (hab s (List.mem_append_right _ hs)).1
    exact ⟨n, by simp only [Expr.eval] at he; rw [he, hn]⟩
  obtain ⟨g, hg⟩ := encodeRange_sound heB hsB hpB₂ hq
  let vals := interleave f g
  obtain ⟨ex, gA, ax⟩ := hf vals (interleave_even f g)
  obtain ⟨ey, gB, ay⟩ := hg vals (interleave_odd f g)
  -- offsets
  have hoff : ∀ (d : DataDesc), (∀ s ∈ offSyms d, s ∈ C.intSyms ∧ s ∉ params ++ params.map prime) →
      (∀ s ∈ offSyms d, IntBound ρ₂ s) ∧ ∀ {ρ' : Valuation} {j : Idx}, Binds ρ params j ρ' → ∀ idx,
        d.linearOffset I ρ' idx = d.linearOffset I ρ₂ idx := by
    intro d hd
    refine ⟨fun s hs => ?_, fun {ρ' j} hb idx => DataDesc.linearOffset_congr fun s hs => ?_⟩
    · obtain ⟨n, hn⟩ := hC.1 s (hd s hs).1
      exact ⟨n, by rw [← hbad s (hd s hs).2]; exact hn⟩
    · have := (hd s hs).2
      simp only [List.mem_append, not_or] at this
      rw [hb.2 s this.1, hbad s (by simpa using this)]
  obtain ⟨sA, cA⟩ := hoff dA (fun s hs => hrs s (by simp [hs]))
  obtain ⟨sB, cB⟩ := hoff dB (fun s hs => hrs s (by simp [hs]))
  obtain ⟨eA, gOA⟩ := offsetPoly?_sound hoA sA vals ex ax (by rw [← cA hb]; exact ho)
  obtain ⟨eB, gOB⟩ := offsetPoly?_sound hoB sB vals ey ay (by rw [← cB hb']; exact ho')
  -- range hypotheses
  have hrsy : ∀ d ∈ rng.dims, ∀ s ∈ d.freeSyms, ρ s = ρ₂ s ∧ IntBound ρ s := by
    intro d hd s hs
    have := hrs s (List.mem_append_left _ (List.mem_append_left _ (List.mem_flatMap.2 ⟨d, hd, hs⟩)))
    exact ⟨hbad s this.2, hC.1 s this.1⟩
  have R1 : HoldsGood I ρ₂ (envOf I ρ₂ vals) (rangeHyps id rp) :=
    rangeHyps_sound id vals params rng.dims rp i hrp
      (forall₂_imp_mem hb.1 fun s hs v hv => by simp only [id]; rw [ρ₂_other s (hpr s hs)]; exact hv) hi hrsy
  have R2 : HoldsGood I ρ₂ (envOf I ρ₂ vals) (rangeHyps prime (rp.drop c.k)) :=
    rangeHyps_sound prime vals P (rng.dims.drop c.k) (rp.drop c.k) (i'.drop c.k)
      (rangePolys?_drop c.k _ _ _ hrp)
      (forall₂_imp_mem (List.forall₂_drop c.k hb'.1) fun s hs v hv => by rw [ρ₂_prime s hs]; exact hv)
      (List.forall₂_drop c.k hi') (fun d hd => hrsy d (List.mem_of_mem_drop hd))
  -- the gap
  have G : HoldsGood I ρ₂ (envOf I ρ₂ vals) [gap] := by
    have hrpk := rangePolys?_drop c.k _ _ _ hrp
    have hbk := List.forall₂_drop c.k hb.1
    have hbk' := List.forall₂_drop c.k hb'.1
    have hik := List.forall₂_drop c.k hi
    have hik' := List.forall₂_drop c.k hi'
    rw [hdrop] at hbk hik
    rw [hdrop'] at hbk' hik'
    cases hP : params.drop c.k with
    | nil => rw [hP] at hbk; cases hbk
    | cons pk ps =>
      cases hD : rng.dims.drop c.k with
      | nil => rw [hD] at hik; cases hik
      | cons dk ds =>
        rw [hP, hD] at hrpk
        rw [hP] at hbk hbk'
        rw [hD] at hik hik'
        simp only [rangePolys?, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at hrpk
        obtain ⟨lo, -, hi, -, st, hstp, rest, -, hrpk⟩ := hrpk
        rw [← hrpk] at hgap
        simp only [gapHyp, Option.some.injEq] at hgap
        subst hgap
        cases hbk with | cons e1 _ =>
        cases hbk' with | cons e2 _ =>
        cases hik with | cons m1 _ =>
        cases hik' with | cons m2 _ =>
        obtain ⟨es, hes, ha⟩ := m1
        obtain ⟨es', hes', ha'⟩ := m2
        rw [hes] at hes'; cases hes'
        obtain ⟨vlo, vhi, vst, -, -, hvst, -, -⟩ := Dim.mem_elems hes
        have hdk : dk ∈ rng.dims := List.mem_of_mem_drop (by rw [hD]; simp)
        have hpk : pk ∈ P := by simp [P, hP]
        have hst₂ : dk.step.evalInt I ρ₂ = some vst := by
          rw [← Expr.evalInt_congr (fun s hs => (hrsy dk hdk s (by simp [Dim.freeSyms, hs])).1)]; exact hvst
        obtain ⟨est, gst⟩ := poly_of_expr hstp (fun s hs => by
          obtain ⟨n, hn⟩ := (hrsy dk hdk s (by simp [Dim.freeSyms, hs])).2
          exact ⟨n, (hrsy dk hdk s (by simp [Dim.freeSyms, hs])).1 ▸ hn⟩) hst₂ vals
        obtain ⟨ex1, gx1⟩ := sym_poly (I := I) (vals := vals)
          (show ρ₂ pk = some (.int a) by rw [ρ₂_other pk (hpr pk (hPsub pk hpk))]; exact e1)
        obtain ⟨ex2, gx2⟩ := sym_poly (I := I) (vals := vals)
          (show ρ₂ (prime pk) = some (.int a') by rw [ρ₂_prime pk hpk]; exact e2)
        intro x hx
        simp only [List.mem_singleton] at hx
        subst hx
        cases hfwd : c.fwd
        · rw [hfwd] at hdir
          simp only [Bool.false_eq_true, ↓reduceIte] at hdir ⊢
          have := Dim.elems_sep hes hvst ha' ha hdir
          exact ⟨by simp only [Ineq.holds_le, Poly.eval_add, ex1, ex2, est]; exact this,
            Ineq.le_atoms (Poly.allAtoms_add gx2 gst) gx1⟩
        · rw [hfwd] at hdir
          simp only [↓reduceIte] at hdir ⊢
          have := Dim.elems_sep hes hvst ha ha' hdir
          exact ⟨by simp only [Ineq.holds_le, Poly.eval_add, ex1, ex2, est]; exact this,
            Ineq.le_atoms (Poly.allAtoms_add gx1 gst) gx2⟩
  have H := (((R1.append R2).append G).append gA).append gB
  -- apply whichever alternative is valid
  have up : (factQuery C (params ++ params.map prime) (rangeHyps id rp ++ rangeHyps prime (rp.drop c.k) ++ [gap] ++
      hA ++ hB) (Poly.sub (Poly.sub oB oA) (Poly.const 1))).withAxioms.Valid → o ≠ o' := by
    intro hv
    have := factQuery_sound hv hC hbad vals H (Poly.allAtoms_sub (Poly.allAtoms_sub gOB gOA) (Poly.allAtoms_const 1))
    simp only [Poly.eval_sub, eA, eB, Poly.eval_const] at this
    omega
  have down : (factQuery C (params ++ params.map prime) (rangeHyps id rp ++ rangeHyps prime (rp.drop c.k) ++ [gap] ++
      hA ++ hB) (Poly.sub (Poly.sub oA oB) (Poly.const 1))).withAxioms.Valid → o ≠ o' := by
    intro hv
    have := factQuery_sound hv hC hbad vals H (Poly.allAtoms_sub (Poly.allAtoms_sub gOA gOB) (Poly.allAtoms_const 1))
    simp only [Poly.eval_sub, eA, eB, Poly.eval_const] at this
    omega
  subst hcq
  rcases hv with hv | hv | ⟨conds, alts, hsep, hconds, halt⟩
  · exact up hv
  · exact down hv
  · simp only at hsep
    split at hsep
    · rename_i hd
      subst hd
      exact sepPart?_sound hsep hconds halt hC hbad vals H ex ey ax ay sA (by rw [← cA hb]; exact ho)
        (by rw [← cB hb']; exact ho')
    · simp at hsep

end SymSubset

section IterSound

open SymSubset

variable {I : ExprInterp} {ρ : Valuation}

/-- **Soundness of the assumed-queries mode**: if every returned obligation
is valid, two distinct iterations `i ≠ i'` of the map never touch a common
linear offset through `A` (in `i`) and `B` (in `i'`). -/
theorem iterObligations?_sound {C : Constraints} {params : List Sym} {rng : Range} {dA dB : DataDesc}
    {A B : Subset} {qs : List Query} (h : iterObligations? C params rng dA A dB B = some qs)
    (hqs : ∀ q ∈ qs, q.Valid) (hC : C.satisfies I ρ) {pts : List Idx} (hrng : rng.points I ρ = some pts)
    {i i' : Idx} (hi : i ∈ pts) (hi' : i' ∈ pts) (hne : i ≠ i') {ρi ρi' : Valuation}
    (hb : Binds ρ params i ρi) (hb' : Binds ρ params i' ρi') {psA psB : List Idx}
    (hA : A.points I ρi = some psA) (hB : B.points I ρi' = some psB) {p q : Idx} (hp : p ∈ psA) (hq : q ∈ psB)
    {o o' : Int} (ho : dA.linearOffset I ρi p = some o) (ho' : dB.linearOffset I ρi' q = some o') : o ≠ o' := by
  simp only [iterObligations?, Option.map_eq_some_iff] at h
  obtain ⟨cqs, hcqs, rfl⟩ := h
  unfold iterQueries? at hcqs
  split at hcqs
  · rename_i hchk
    simp only [bind, Option.bind_eq_some_iff] at hcqs
    obtain ⟨rp, hrp, hmap⟩ := hcqs
    simp only [iterChecks, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, decide_eq_true_eq,
      Bool.or_eq_true] at hchk
    obtain ⟨⟨⟨hlen, hpr⟩, hrs⟩, hab⟩ := hchk
    have hiF := (Range.mem_points hrng).1 hi
    have hiF' := (Range.mem_points hrng).1 hi'
    obtain ⟨k, a, a', r, r', htake, hdrop, hdrop', hne'⟩ :=
      exists_firstDiff (by rw [hiF.length_eq, hiF'.length_eq]) hne
    have hk : k < params.length := by
      have : (i.drop k).length = r.length + 1 := by rw [hdrop]; simp
      rw [List.length_drop] at this
      rw [hlen, ← hiF.length_eq]; omega
    obtain ⟨rA, hrA, psA', hpA, hp'⟩ := Subset.mem_points hA hp
    obtain ⟨rB, hrB, psB', hpB, hq'⟩ := Subset.mem_points hB hq
    let c : IterCase := ⟨k, decide (a < a'), rA, rB⟩
    have hc : c ∈ iterCases params.length A B := by
      simp only [iterCases, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false,
        List.mem_map]
      refine ⟨k, hk, decide (a < a'), by cases decide (a < a') <;> simp, rA, hrA, rB, hrB, rfl⟩
    rw [List.mapM_option_some_iff] at hmap
    obtain ⟨cq, hmem, hcq⟩ := List.Forall₂.exists_right_of_mem hmap hc
    have hv : cq.Holds := by
      by_cases hd : cq.decided = true
      · exact CaseQ.decided_sound hd
      · refine CaseQ.obligation_sound (hqs _ (List.mem_filterMap.2 ⟨cq, hmem, ?_⟩))
        simp [hd]
    have hdir : if c.fwd then a < a' else a' < a := by
      by_cases hlt : a < a'
      · simp [c, hlt]
      · simp only [c, hlt, decide_false, Bool.false_eq_true, ↓reduceIte]; omega
    refine caseQueries?_sound (rng := rng) hcq hrp hpr
      (fun s hs => ⟨(hrs s hs).1, (hrs s hs).2⟩) (fun s hs => ?_) hC hiF hiF' htake hdrop hdrop' hdir hb hb'
      hpA hpB hp' hq' ho ho' hv
    have hs' : s ∈ A.freeSyms ++ B.freeSyms := by
      rcases List.mem_append.1 hs with hs | hs
      · exact List.mem_append_left _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rA, hrA, hs⟩)
      · exact List.mem_append_right _ (by rw [Subset.freeSyms_eq]; exact List.mem_flatMap.2 ⟨rB, hrB, hs⟩)
    exact hab s hs'
  · simp at hcqs

/-- **Soundness of `iterDisjoint?`**: two distinct iterations `i ≠ i'` of the
map never touch a common linear offset through `A` (in `i`) and `B` (in
`i'`). -/
theorem iterDisjoint?_sound {C : Constraints} {params : List Sym} {rng : Range} {dA dB : DataDesc}
    {A B : Subset} (h : iterDisjoint? C params rng dA A dB B = true) (hC : C.satisfies I ρ) {pts : List Idx}
    (hrng : rng.points I ρ = some pts) {i i' : Idx} (hi : i ∈ pts) (hi' : i' ∈ pts) (hne : i ≠ i')
    {ρi ρi' : Valuation} (hb : Binds ρ params i ρi) (hb' : Binds ρ params i' ρi') {psA psB : List Idx}
    (hA : A.points I ρi = some psA) (hB : B.points I ρi' = some psB) {p q : Idx} (hp : p ∈ psA) (hq : q ∈ psB)
    {o o' : Int} (ho : dA.linearOffset I ρi p = some o) (ho' : dB.linearOffset I ρi' q = some o') : o ≠ o' := by
  unfold iterDisjoint? at h
  split at h
  · rename_i hobl
    exact iterObligations?_sound hobl (by simp) hC hrng hi hi' hne hb hb' hA hB hp hq ho ho'
  · simp at h

end IterSound

end Dace
