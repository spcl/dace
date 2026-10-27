-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.LogShift
import Dace.SDFG.Clean
import Dace.SDFG.Footprinted
import Dace.SDFG.Names

/-!
# Simulation under renaming, partial agreement and less initialized memory

A run of the SDFG semantics in context `ctx₁` from memory `m₁` is
*simulated* by a run of the same program in context `ctx₂` from memory `m₂`
when

* the contexts agree on the names the program mentions (`Ctx.Good`,
  `Ctx.Strong`, `SNames`, `Region.Names`): equal descriptors, and memory names
  renamed by a permutation `π`;
* `m₂` is `m₁` renamed by `π` on the memory names satisfying `P`, except
  that some cells may be uninitialized (`Mem.Sim π P`); the run touches only
  names satisfying `P`.

The simulating run ends in undefined behavior, or like the simulated run
(the same error, or simulating final states).  Uses (`Decisions.md#D68`):

* `π = 1`, `P` everything, `ctx₂ = ctx₁`: allocation makes memory less
  initialized (the pair checker matches tree segments against states);
* `π = 1`, `P` excluding names the program never touches: memories that
  differ there;
* `π` a swap of the per-call transients of a nested SDFG with the tree's
  names, `ctx₁` the tree-named inner context, `ctx₂` the nested call's
  context: the nested call is matched against the flattened tree.
-/

set_option autoImplicit false

namespace Dace

/-! ## Optional values: undefined or equal -/

/-- `a` is undefined, or equal to `b`. -/
def OptLe {α : Type} (a b : Option α) : Prop := a = none ∨ a = b

namespace OptLe

variable {α β : Type}

theorem refl (a : Option α) : OptLe a a := .inr rfl

theorem none' (b : Option α) : OptLe none b := .inl rfl

theorem bind {a₂ a₁ : Option α} {f₂ f₁ : α → Option β} (h : OptLe a₂ a₁) (hf : ∀ x, OptLe (f₂ x) (f₁ x)) :
    OptLe (a₂ >>= f₂) (a₁ >>= f₁) := by
  rcases h with rfl | rfl
  · exact .inl rfl
  · cases a₂ with
    | none => exact .inl rfl
    | some x => exact hf x

theorem map {a₂ a₁ : Option α} (f : α → β) (h : OptLe a₂ a₁) : OptLe (a₂.map f) (a₁.map f) := by
  rcases h with rfl | rfl
  · exact .inl rfl
  · exact .inr rfl

theorem mapM {f₂ f₁ : α → Option β} (hf : ∀ x, OptLe (f₂ x) (f₁ x)) :
    ∀ xs : List α, OptLe (xs.mapM f₂) (xs.mapM f₁)
  | [] => .inr rfl
  | x :: xs => by
    simp only [List.mapM_cons]
    exact bind (hf x) fun y => bind (mapM hf xs) fun ys => .inr rfl

theorem of_eq {a₂ a₁ : Option α} (h : a₂ = a₁) : OptLe a₂ a₁ := .inr h

theorem some_iff {a₂ : Option α} {b : α} : OptLe a₂ (some b) ↔ a₂ = none ∨ a₂ = some b := Iff.rfl

theorem eq_of_some {a₂ a₁ : Option α} {x : α} (h : OptLe a₂ a₁) (h₂ : a₂ = some x) : a₁ = some x := by
  rcases h with h | h
  · rw [h] at h₂; cases h₂
  · rw [← h, h₂]

end OptLe

/-! ## Outcomes: undefined behavior, the same error, or related results -/

/-- The outcome `r₂` simulates `r₁`: it is undefined behavior, or `r₁` is
the same error, or both are results related by `R`. -/
def ResSim {α β : Type} (R : α → β → Prop) (r₂ : Res α) (r₁ : Res β) : Prop :=
  (∃ e, r₂ = .err e ∧ e.isUB = true) ∨ (∃ e, r₁ = .err e ∧ r₂ = .err e) ∨ ∃ a b, r₂ = .ok a ∧ r₁ = .ok b ∧ R a b

namespace ResSim

variable {α β γ δ : Type} {R : α → β → Prop} {S : γ → δ → Prop}

theorem ok {a : α} {b : β} (h : R a b) : ResSim R (.ok a) (.ok b) := .inr (.inr ⟨a, b, rfl, rfl, h⟩)

theorem err (e : ErrReason) : ResSim R (.err e : Res α) (.err e : Res β) := .inr (.inl ⟨e, rfl, rfl⟩)

theorem ub {e : ErrReason} (hu : e.isUB = true) (r₁ : Res β) : ResSim R (.err e : Res α) r₁ := .inl ⟨e, rfl, hu⟩

theorem bind {r₂ : Res α} {r₁ : Res β} {f₂ : α → Res γ} {f₁ : β → Res δ} (h : ResSim R r₂ r₁)
    (hf : ∀ a b, R a b → ResSim S (f₂ a) (f₁ b)) : ResSim S (r₂ >>= f₂) (r₁ >>= f₁) := by
  rcases h with ⟨e, rfl, hu⟩ | ⟨e, rfl, rfl⟩ | ⟨a, b, rfl, rfl, hab⟩
  · exact .inl ⟨e, rfl, hu⟩
  · exact .inr (.inl ⟨e, rfl, rfl⟩)
  · exact hf a b hab

theorem mono {R' : α → β → Prop} {r₂ : Res α} {r₁ : Res β} (h : ResSim R r₂ r₁) (hR : ∀ a b, R a b → R' a b) :
    ResSim R' r₂ r₁ := by
  rcases h with h | h | ⟨a, b, h₂, h₁, hab⟩
  · exact .inl h
  · exact .inr (.inl h)
  · exact .inr (.inr ⟨a, b, h₂, h₁, hR a b hab⟩)

theorem of_eq {R : α → α → Prop} {r : Res α} (hR : ∀ a, R a a) : ResSim R r r := by
  cases r with
  | ok a => exact ok (hR a)
  | err e => exact err e

theorem map {r₂ : Res α} {r₁ : Res β} (f : α → γ) (g : β → δ) (h : ResSim R r₂ r₁)
    (hfg : ∀ a b, R a b → S (f a) (g b)) : ResSim S (r₂.map f) (r₁.map g) := by
  rcases h with ⟨e, rfl, hu⟩ | ⟨e, rfl, rfl⟩ | ⟨a, b, rfl, rfl, hab⟩
  · exact .inl ⟨e, rfl, hu⟩
  · exact .inr (.inl ⟨e, rfl, rfl⟩)
  · exact ok (hfg a b hab)

/-- Lift an optional value that is undefined or equal, failing with an
undefined-behavior error. -/
theorem ofOption {o₂ o₁ : Option α} {e : ErrReason} (hu : e.isUB = true) (h : OptLe o₂ o₁) :
    ResSim Eq (Res.ofOption o₂ e) (Res.ofOption o₁ e) := by
  rcases h with rfl | rfl
  · exact .inl ⟨e, rfl, hu⟩
  · exact of_eq (R := Eq) fun _ => rfl

/-- Simulation along a `foldlM` over the same list. -/
theorem foldlM {τ : Type} {f₂ : α → τ → Res α} {f₁ : β → τ → Res β}
    (hf : ∀ x a b, R a b → ResSim R (f₂ a x) (f₁ b x)) :
    ∀ (xs : List τ) {a : α} {b : β}, R a b → ResSim R (xs.foldlM f₂ a) (xs.foldlM f₁ b)
  | [], _, _, h => ok h
  | x :: xs, _, _, h => by
    simp only [List.foldlM_cons]
    exact bind (hf x _ _ h) fun a' b' h' => foldlM hf xs h'

theorem bindEq {r₂ r₁ : Res α} {f₂ : α → Res γ} {f₁ : α → Res δ} (h : ResSim Eq r₂ r₁)
    (hf : ∀ a, ResSim S (f₂ a) (f₁ a)) : ResSim S (r₂ >>= f₂) (r₁ >>= f₁) :=
  bind h fun a b hab => hab ▸ hf a

/-- Both sides branch on the same condition. -/
theorem ite {c : Bool} {a₂ b₂ : Res α} {a₁ b₁ : Res β} (ht : c = true → ResSim R a₂ a₁)
    (hf : c = false → ResSim R b₂ b₁) : ResSim R (if c = true then a₂ else b₂) (if c = true then a₁ else b₁) := by
  cases c
  · exact hf rfl
  · exact ht rfl

theorem ok_cases {r₂ : Res α} {b : β} (h : ResSim R r₂ (.ok b)) :
    (∃ e, r₂ = .err e ∧ e.isUB = true) ∨ ∃ a, r₂ = .ok a ∧ R a b := by
  rcases h with ⟨e, rfl, hu⟩ | ⟨e, h₁, -⟩ | ⟨a, b', rfl, h₁, hab⟩
  · exact .inl ⟨e, rfl, hu⟩
  · cases h₁
  · cases h₁; exact .inr ⟨a, rfl, hab⟩

theorem of_err {e' e : ErrReason} (h : e'.isUB = true ∨ e' = e) : ResSim R (.err e' : Res α) (.err e : Res β) := by
  rcases h with h | rfl
  · exact .inl ⟨e', rfl, h⟩
  · exact .inr (.inl ⟨e', rfl, rfl⟩)

theorem fmapEq {r₂ r₁ : Res α} (g : α → γ) (h : ResSim Eq r₂ r₁) : ResSim Eq (g <$> r₂) (g <$> r₁) :=
  bindEq (S := Eq) h fun a => ok rfl

theorem filterMapM {τ : Type} {f₂ f₁ : τ → Res (Option α)} (hf : ∀ x, ResSim Eq (f₂ x) (f₁ x)) :
    ∀ xs : List τ, ResSim Eq (xs.filterMapM f₂) (xs.filterMapM f₁)
  | [] => ok rfl
  | x :: xs => by
    simp only [List.filterMapM_cons]
    refine bindEq (hf x) fun b => ?_
    cases b with
    | none => exact filterMapM hf xs
    | some b => exact fmapEq _ (filterMapM hf xs)

theorem mapM_mem {τ : Type} {f₂ f₁ : τ → Res α} :
    ∀ (xs : List τ), (∀ x ∈ xs, ResSim Eq (f₂ x) (f₁ x)) → ResSim Eq (xs.mapM f₂) (xs.mapM f₁)
  | [], _ => ok rfl
  | x :: xs, hf => by
    simp only [List.mapM_cons]
    exact bindEq (hf x List.mem_cons_self) fun a =>
      bindEq (mapM_mem xs fun x' hx' => hf x' (List.mem_cons_of_mem _ hx')) fun as => ok rfl

theorem filterMapM_mem {τ : Type} {f₂ f₁ : τ → Res (Option α)} :
    ∀ (xs : List τ), (∀ x ∈ xs, ResSim Eq (f₂ x) (f₁ x)) → ResSim Eq (xs.filterMapM f₂) (xs.filterMapM f₁)
  | [], _ => ok rfl
  | x :: xs, hf => by
    simp only [List.filterMapM_cons]
    refine bindEq (hf x List.mem_cons_self) fun b => ?_
    cases b with
    | none => exact filterMapM_mem xs fun x' hx' => hf x' (List.mem_cons_of_mem _ hx')
    | some b => exact fmapEq _ (filterMapM_mem xs fun x' hx' => hf x' (List.mem_cons_of_mem _ hx'))

theorem mapM {τ : Type} {f₂ f₁ : τ → Res α} (hf : ∀ x, ResSim Eq (f₂ x) (f₁ x)) :
    ∀ xs : List τ, ResSim Eq (xs.mapM f₂) (xs.mapM f₁)
  | [] => ok rfl
  | x :: xs => by
    simp only [List.mapM_cons]
    exact bindEq (hf x) fun a => bindEq (mapM hf xs) fun as => ok rfl

/-- If the simulating side fails, it is undefined behavior or the simulated
side fails the same way. -/
theorem err_inv' {r₂ : Res α} {r₁ : Res β} {e : ErrReason} (h : ResSim R r₂ r₁) (he : r₂ = .err e) :
    ∃ e', e' = e ∧ (e.isUB = true ∨ r₁ = .err e) := by
  rcases h with ⟨e', rfl, hu⟩ | ⟨e', h₁, rfl⟩ | ⟨a, b, rfl, -, -⟩
  · cases he; exact ⟨_, rfl, .inl hu⟩
  · cases he; exact ⟨_, rfl, .inr h₁⟩
  · cases he

theorem err_inv {r₂ : Res α} {e : ErrReason} (h : ResSim R r₂ (.err e : Res β)) :
    ∃ e', r₂ = .err e' ∧ (e'.isUB = true ∨ e' = e) := by
  rcases h with ⟨e', rfl, hu⟩ | ⟨e', h₁, rfl⟩ | ⟨a, b, -, h₁, -⟩
  · exact ⟨e', rfl, .inl hu⟩
  · cases h₁; exact ⟨_, rfl, .inr rfl⟩
  · cases h₁

end ResSim

/-! ## Monotone evaluation -/

section EvalMono

variable {I₂ I₁ : ExprInterp} (happ : I₂.app = I₁.app) (hattr : I₂.readAttr = I₁.readAttr)
  {ρ₂ ρ₁ : Valuation}
include happ hattr

mutual

/-- **Evaluation is monotone**: if the data reads and the symbols an
expression reads are each undefined or equal, so is its value. -/
theorem Expr.eval_mono : ∀ e : Expr, (∀ d ∈ e.readData, ∀ is, OptLe (I₂.readData d is) (I₁.readData d is)) →
    (∀ x ∈ e.freeSyms, OptLe (ρ₂ x) (ρ₁ x)) → OptLe (e.eval I₂ ρ₂) (e.eval I₁ ρ₁)
  | .lit l, _, _ => by cases l <;> exact .inr rfl
  | .sym s, _, hρ => by simp only [Expr.eval]; exact hρ s (by simp [Expr.freeSyms])
  | .un op a, hrd, hρ => by
    simp only [Expr.eval]
    exact OptLe.bind (Expr.eval_mono a (by simpa [Expr.readData] using hrd) (by simpa [Expr.freeSyms] using hρ))
      fun v => .inr rfl
  | .bin op a b, hrd, hρ => by
    simp only [Expr.eval]
    simp only [Expr.readData, Expr.freeSyms, List.mem_append] at hrd hρ
    exact OptLe.bind (Expr.eval_mono a (fun d hd => hrd d (.inl hd)) (fun x hx => hρ x (.inl hx))) fun v =>
      OptLe.bind (Expr.eval_mono b (fun d hd => hrd d (.inr hd)) (fun x hx => hρ x (.inr hx))) fun w => .inr rfl
  | .ite c t e, hrd, hρ => by
    simp only [Expr.eval]
    simp only [Expr.readData, Expr.freeSyms, List.mem_append] at hrd hρ
    refine OptLe.bind (Expr.eval_mono c (fun d hd => hrd d (.inl (.inl hd))) (fun x hx => hρ x (.inl (.inl hx))))
      fun cv => OptLe.bind (.inr rfl) fun b => ?_
    cases b
    · exact Expr.eval_mono e (fun d hd => hrd d (.inr hd)) (fun x hx => hρ x (.inr hx))
    · exact Expr.eval_mono t (fun d hd => hrd d (.inl (.inr hd))) (fun x hx => hρ x (.inl (.inr hx)))
  | .app f args, hrd, hρ => by
    simp only [Expr.eval]
    refine OptLe.bind (Expr.evalList_mono args (by simpa [Expr.readData] using hrd)
      (by simpa [Expr.freeSyms] using hρ)) fun vs => ?_
    rw [happ]; exact .inr rfl
  | .subscript d idx, hrd, hρ => by
    simp only [Expr.eval]
    simp only [Expr.readData, List.mem_cons] at hrd
    exact OptLe.bind (Expr.evalList_mono idx (fun d' hd' => hrd d' (.inr hd')) (by simpa [Expr.freeSyms] using hρ))
      fun vs => OptLe.bind (.inr rfl) fun is => hrd d (.inl rfl) is
  | .attr b f, hrd, hρ => by
    simp only [Expr.eval]
    refine OptLe.bind (Expr.eval_mono b (by simpa [Expr.readData] using hrd) (by simpa [Expr.freeSyms] using hρ))
      fun v => ?_
    rw [hattr]; exact .inr rfl
  | .approx a o, hrd, hρ => by
    simp only [Expr.eval]
    simp only [Expr.readData, Expr.freeSyms, List.mem_append] at hrd hρ
    exact Expr.eval_mono a (fun d hd => hrd d (.inl hd)) (fun x hx => hρ x (.inl hx))

theorem Expr.evalList_mono : ∀ es : Exprs, (∀ d ∈ Expr.readDataList es, ∀ is, OptLe (I₂.readData d is) (I₁.readData d is)) →
    (∀ x ∈ Expr.freeSymsList es, OptLe (ρ₂ x) (ρ₁ x)) →
    OptLe (Expr.evalList I₂ ρ₂ es) (Expr.evalList I₁ ρ₁ es)
  | .nil, _, _ => .inr rfl
  | .cons e es, hrd, hρ => by
    simp only [Expr.evalList]
    simp only [Expr.readDataList, Expr.freeSymsList, List.mem_append] at hrd hρ
    exact OptLe.bind (Expr.eval_mono e (fun d hd => hrd d (.inl hd)) (fun x hx => hρ x (.inl hx))) fun v =>
      OptLe.bind (Expr.evalList_mono es (fun d hd => hrd d (.inr hd)) (fun x hx => hρ x (.inr hx))) fun vs => .inr rfl

end

/-- Monotone evaluation under one predicate `Q` covering every name the
expression reads. -/
theorem Expr.eval_mono' {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (e : Expr) (he : ∀ x ∈ e.names, Q x) :
    OptLe (e.eval I₂ ρ₂) (e.eval I₁ ρ₁) :=
  Expr.eval_mono happ hattr e (fun d hd is => hrd d is (he d (List.mem_append_left _ hd)))
    (fun x hx => hρ x (he x (List.mem_append_right _ hx)))

theorem Expr.evalInt_mono' {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (e : Expr) (he : ∀ x ∈ e.names, Q x) :
    OptLe (e.evalInt I₂ ρ₂) (e.evalInt I₁ ρ₁) :=
  OptLe.bind (Expr.eval_mono' happ hattr hrd hρ e he) fun _ => .inr rfl

theorem Expr.evalBool_mono' {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (e : Expr) (he : ∀ x ∈ e.names, Q x) :
    OptLe (e.evalBool I₂ ρ₂) (e.evalBool I₁ ρ₁) :=
  OptLe.bind (Expr.eval_mono' happ hattr hrd hρ e he) fun _ => .inr rfl

theorem Dim.elems_mono {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (d : Dim) (hd : ∀ x ∈ d.names, Q x) :
    OptLe (d.elems I₂ ρ₂) (d.elems I₁ ρ₁) := by
  unfold Dim.elems
  simp only [Dim.names, List.mem_append] at hd
  exact OptLe.bind (Expr.evalInt_mono' happ hattr hrd hρ _ fun x hx => hd x (.inl (.inl (.inl hx)))) fun lo =>
    OptLe.bind (Expr.evalInt_mono' happ hattr hrd hρ _ fun x hx => hd x (.inl (.inl (.inr hx)))) fun hi =>
      OptLe.bind (Expr.evalInt_mono' happ hattr hrd hρ _ fun x hx => hd x (.inl (.inr hx))) fun step => .inr rfl

omit happ hattr in
theorem OptLe.mapM_mem {α β : Type} {f₂ f₁ : α → Option β} :
    ∀ (xs : List α), (∀ x ∈ xs, OptLe (f₂ x) (f₁ x)) → OptLe (xs.mapM f₂) (xs.mapM f₁)
  | [], _ => .inr rfl
  | x :: xs, hf => by
    simp only [List.mapM_cons]
    exact OptLe.bind (hf x List.mem_cons_self) fun y =>
      OptLe.bind (OptLe.mapM_mem xs fun x' hx' => hf x' (List.mem_cons_of_mem _ hx')) fun ys => .inr rfl

theorem Range.points_mono {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (r : Range) (hr : ∀ x ∈ r.names, Q x) :
    OptLe (r.points I₂ ρ₂) (r.points I₁ ρ₁) := by
  unfold Range.points
  refine OptLe.bind (OptLe.mapM_mem r.dims fun d hd => Dim.elems_mono happ hattr hrd hρ d fun x hx => ?_) fun _ => .inr rfl
  exact hr x (List.mem_flatMap.2 ⟨d, hd, hx⟩)

theorem Subset.points_mono {Q : Name → Prop} (hrd : ∀ d is, Q d → OptLe (I₂.readData d is) (I₁.readData d is))
    (hρ : ∀ x, Q x → OptLe (ρ₂ x) (ρ₁ x)) (S : Subset) (hS : ∀ x ∈ S.names, Q x) :
    OptLe (S.points I₂ ρ₂) (S.points I₁ ρ₁) := by
  cases S with
  | range r => exact Range.points_mono happ hattr hrd hρ r hS
  | union rs =>
    simp only [Subset.points]
    refine OptLe.bind (OptLe.mapM_mem rs fun r hr => Range.points_mono happ hattr hrd hρ r fun x hx => ?_)
      fun _ => .inr rfl
    exact hS x (List.mem_flatMap.2 ⟨r, hr, hx⟩)

end EvalMono

/-! ## Renaming memory names -/

/-- Rename the memory name of a location. -/
def Loc.ren (π : Equiv.Perm Name) (l : Loc) : Loc := ⟨π l.name, l.path⟩

/-- Rename the base of an alias binding. -/
def AliasBinding.ren (π : Equiv.Perm Name) (b : AliasBinding) : AliasBinding := { b with base := π b.base }

section Ren

variable (π : Equiv.Perm Name)

@[simp] theorem Loc.ren_name (l : Loc) : (l.ren π).name = π l.name := rfl
@[simp] theorem Loc.ren_path (l : Loc) : (l.ren π).path = l.path := rfl

theorem Loc.ren_inj {l l' : Loc} : l.ren π = l'.ren π ↔ l = l' := by
  obtain ⟨n, p⟩ := l
  obtain ⟨n', p'⟩ := l'
  simp [Loc.ren]

theorem Loc.ren_elem (n : Name) (o : Int) : (Loc.elem n o).ren π = Loc.elem (π n) o := rfl

/-- Every location is a renamed one. -/
theorem Loc.ren_symm (l : Loc) : (l.ren π.symm).ren π = l := by
  simp [Loc.ren]

@[simp] theorem Loc.ren_one (l : Loc) : l.ren 1 = l := rfl
@[simp] theorem AliasBinding.ren_one (b : AliasBinding) : b.ren 1 = b := rfl

theorem Mem.applyAlias_ren (b : AliasBinding) (l : Loc) :
    Mem.applyAlias (b.ren π) (l.ren π) = (Mem.applyAlias b l).ren π := by
  unfold Mem.applyAlias
  simp only [Loc.ren_path]
  split <;> rfl

end Ren

/-- The memory `m₂` is `m₁` with its names renamed by `π`, on the memory
names satisfying `P`, except that some cells may be uninitialized; aliases of
names satisfying `P` bind bases satisfying `P`. -/
structure Mem.Sim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (m₂ m₁ : Mem) : Prop where
  aliases : ∀ n, P n → m₂.aliases (π n) = (m₁.aliases n).map (AliasBinding.ren π)
  aliasP : ∀ n b, P n → m₁.aliases n = some b → P b.base
  cells : ∀ l, P l.name → (L l.name ∧ m₂.cells (l.ren π) = .undef) ∨ m₂.cells (l.ren π) = m₁.cells l
  queues : ∀ l, P l.name → m₂.queues (l.ren π) = m₁.queues l

namespace Mem.Sim

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {m₂ m₁ : Mem}

theorem refl (m : Mem) : Mem.Sim 1 (fun _ => True) L m m :=
  ⟨fun n _ => by cases h : m.aliases n <;> simp [h], fun _ _ _ _ => trivial, fun _ _ => .inr rfl, fun _ _ => rfl⟩

theorem resolveFuel (h : Mem.Sim π P L m₂ m₁) : ∀ (k : Nat) (l : Loc), P l.name →
    m₂.resolveFuel k (l.ren π) = (m₁.resolveFuel k l).ren π ∧ P (m₁.resolveFuel k l).name
  | 0, _, hl => ⟨rfl, hl⟩
  | k + 1, l, hl => by
    simp only [Mem.resolveFuel, Loc.ren_name, h.aliases l.name hl]
    cases hb : m₁.aliases l.name with
    | none => exact ⟨rfl, hl⟩
    | some b =>
      simp only [Option.map_some]
      rw [Mem.applyAlias_ren]
      exact resolveFuel h k _ (h.aliasP _ b hl hb)

theorem resolve (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) :
    m₂.resolve (l.ren π) = (m₁.resolve l).ren π := (h.resolveFuel 64 l hl).1

theorem resolveP (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) : P (m₁.resolve l).name :=
  (h.resolveFuel 64 l hl).2

theorem isAliased (h : Mem.Sim π P L m₂ m₁) {n : Name} (hn : P n) : m₂.isAliased (π n) = m₁.isAliased n := by
  simp only [Mem.isAliased, h.aliases n hn]
  cases m₁.aliases n <;> rfl

theorem read (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) :
    m₂.read (l.ren π) = .undef ∨ m₂.read (l.ren π) = m₁.read l := by
  simp only [Mem.read, h.resolve hl]
  rcases h.cells _ (h.resolveP hl) with ⟨-, h₁⟩ | h₁
  exacts [.inl h₁, .inr h₁]

theorem readL (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) :
    (L (m₁.resolve l).name ∧ m₂.read (l.ren π) = .undef) ∨ m₂.read (l.ren π) = m₁.read l := by
  simp only [Mem.read, h.resolve hl]
  exact h.cells _ (h.resolveP hl)

/-- Updating the cell at a renamed location. -/
theorem updCell (h : Mem.Sim π P L m₂ m₁) (l : Loc) (v₂ v₁ : Val)
    (hv : P l.name → (L l.name ∧ v₂ = .undef) ∨ v₂ = v₁) :
    Mem.Sim π P L { m₂ with cells := Function.update m₂.cells (l.ren π) v₂ }
      { m₁ with cells := Function.update m₁.cells l v₁ } := by
  refine ⟨h.aliases, h.aliasP, fun l' hl' => ?_, h.queues⟩
  by_cases hl : l' = l
  · subst hl; simpa using hv hl'
  · have : l'.ren π ≠ l.ren π := fun e => hl ((Loc.ren_inj π).1 e)
    simp only [Function.update_of_ne this, Function.update_of_ne hl]
    exact h.cells l' hl'

theorem updQueue (h : Mem.Sim π P L m₂ m₁) (l : Loc) (q : List Val) :
    Mem.Sim π P L { m₂ with queues := Function.update m₂.queues (l.ren π) q }
      { m₁ with queues := Function.update m₁.queues l q } := by
  refine ⟨h.aliases, h.aliasP, h.cells, fun l' hl' => ?_⟩
  by_cases hl : l' = l
  · subst hl; simp
  · have : l'.ren π ≠ l.ren π := fun e => hl ((Loc.ren_inj π).1 e)
    simp only [Function.update_of_ne this, Function.update_of_ne hl]
    exact h.queues l' hl'

theorem write (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) (v : Val) :
    Mem.Sim π P L (m₂.write (l.ren π) v) (m₁.write l v) := by
  simp only [Mem.write, h.resolve hl]
  exact h.updCell _ v v fun _ => .inr rfl

theorem setAlias (h : Mem.Sim π P L m₂ m₁) (n : Name) (b : AliasBinding) (hb : P n → P b.base) :
    Mem.Sim π P L (m₂.setAlias (π n) (b.ren π)) (m₁.setAlias n b) := by
  refine ⟨fun n' hn' => ?_, fun n' b' hn' hb' => ?_, h.cells, h.queues⟩
  · simp only [Mem.setAlias]
    by_cases hn : n' = n
    · subst hn; simp
    · have : π n' ≠ π n := fun e => hn (π.injective e)
      simp only [Function.update_of_ne this, Function.update_of_ne hn]
      exact h.aliases n' hn'
  · simp only [Mem.setAlias] at hb'
    by_cases hn : n' = n
    · subst hn
      simp only [Function.update_self, Option.some.injEq] at hb'
      subst hb'
      exact hb hn'
    · simp only [Function.update_of_ne hn] at hb'
      exact h.aliasP n' b' hn' hb'

theorem push (h : Mem.Sim π P L m₂ m₁) {l : Loc} (hl : P l.name) (v : Val) :
    Mem.Sim π P L (m₂.push (l.ren π) v) (m₁.push l v) := by
  simp only [Mem.push, h.resolve hl, h.queues _ (h.resolveP hl)]
  exact h.updQueue _ _

theorem pushMany {l : Loc} (hl : P l.name) :
    ∀ (vs : List Val) {m₂ m₁ : Mem}, Mem.Sim π P L m₂ m₁ → Mem.Sim π P L (m₂.pushMany (l.ren π) vs) (m₁.pushMany l vs)
  | [], _, _, h => h
  | v :: vs, _, _, h => by
    simp only [Mem.pushMany, List.foldl_cons]
    exact pushMany hl vs (h.push hl v)

theorem setQueue (h : Mem.Sim π P L m₂ m₁) (l : Loc) (q : List Val) :
    Mem.Sim π P L (m₂.setQueue (l.ren π) q) (m₁.setQueue l q) := h.updQueue l q

theorem alloc (h : Mem.Sim π P L m₂ m₁) (n : Name) : Mem.Sim π P L (m₂.alloc (π n)) (m₁.alloc n) := by
  refine ⟨h.aliases, h.aliasP, fun l hl => ?_, fun l hl => ?_⟩
  · simp only [Mem.alloc, Loc.ren_name, (Equiv.injective π).eq_iff]
    split
    · exact .inr rfl
    · exact h.cells l hl
  · simp only [Mem.alloc, Loc.ren_name, (Equiv.injective π).eq_iff]
    split
    · rfl
    · exact h.queues l hl

theorem allocCells (h : Mem.Sim π P L m₂ m₁) (n : Name) : Mem.Sim π P L (m₂.allocCells (π n)) (m₁.allocCells n) := by
  refine ⟨h.aliases, h.aliasP, fun l hl => ?_, h.queues⟩
  simp only [Mem.allocCells, Loc.ren_name, (Equiv.injective π).eq_iff]
  split
  · exact .inr rfl
  · exact h.cells l hl

/-- Allocating cells only on the simulating side keeps the simulation. -/
theorem allocCells_left (h : Mem.Sim π P L m₂ m₁) (n : Name) (hL : L (π.symm n)) :
    Mem.Sim π P L (m₂.allocCells n) m₁ := by
  refine ⟨h.aliases, h.aliasP, fun l hl => ?_, h.queues⟩
  simp only [Mem.allocCells]
  split
  · next he =>
    refine .inl ⟨?_, rfl⟩
    have : l.name = π.symm n := by rw [← he]; simp
    rw [this]; exact hL
  · exact h.cells l hl

theorem allocAll :
    ∀ (ns : List Name) {m₂ m₁ : Mem}, Mem.Sim π P L m₂ m₁ → Mem.Sim π P L (m₂.allocAll (ns.map π)) (m₁.allocAll ns)
  | [], _, _, h => h
  | n :: ns, _, _, h => by
    simp only [Mem.allocAll, List.map_cons, List.foldl_cons]
    exact allocAll ns (h.alloc n)

theorem writeChunk (h : Mem.Sim π P L m₂ m₁) (n : Name) :
    ∀ vs : List Val, Mem.Sim π P L (m₂.writeChunk (π n) vs) (m₁.writeChunk n vs) := by
  intro vs
  unfold Mem.writeChunk
  suffices ∀ (xs : List (Val × Nat)) (a₂ a₁ : Mem), Mem.Sim π P L a₂ a₁ →
      Mem.Sim π P L (xs.foldl (fun acc (v, k) => { acc with cells := Function.update acc.cells (Loc.elem (π n) k) v }) a₂)
        (xs.foldl (fun acc (v, k) => { acc with cells := Function.update acc.cells (Loc.elem n k) v }) a₁) from
    this _ _ _ h
  intro xs
  induction xs with
  | nil => intro _ _ h; exact h
  | cons x xs ih =>
    intro a₂ a₁ h
    obtain ⟨v, k⟩ := x
    simp only [List.foldl_cons]
    exact ih _ _ (by rw [← Loc.ren_elem]; exact h.updCell _ v v fun _ => .inr rfl)

end Mem.Sim

/-! ## Renaming logs -/

namespace Log

variable (π : Equiv.Perm Name)

/-- Rename every location and alias name of a log. -/
def ren (g : Log) : Log :=
  { reads := g.reads.map (Loc.ren π)
    writes := g.writes.map fun (l, v) => (l.ren π, v)
    wcrs := g.wcrs.map fun (l, op, v) => (l.ren π, op, v)
    aliasSets := g.aliasSets.map fun (n, b) => (π n, b.ren π)
    queues := g.queues.map fun (l, e) => (l.ren π, e) }

@[simp] theorem ren_empty : Log.empty.ren π = Log.empty := rfl

theorem ren_one (g : Log) : g.ren 1 = g := by
  obtain ⟨r, w, c, a, q⟩ := g
  have h : Loc.ren 1 = id := funext fun _ => rfl
  simp [ren, h]

theorem contains_ren (ls : List Loc) (l : Loc) : (ls.map (Loc.ren π)).contains (l.ren π) = ls.contains l := by
  have hf : Function.Injective (Loc.ren π) := fun _ _ h => (Loc.ren_inj π).1 h
  simp [hf.eq_iff]

theorem _root_.List.map_filter_comm {α β : Type} (f : α → β) (p : α → Bool) (q : β → Bool)
    (h : ∀ x, q (f x) = p x) : ∀ xs : List α, (xs.filter p).map f = (xs.map f).filter q
  | [] => rfl
  | x :: xs => by
    simp only [List.filter_cons, List.map_cons, h]
    split <;> simp [List.map_filter_comm f p q h xs]

theorem writeLocs_ren' (ws : List (Loc × Val)) :
    (ws.map fun x => (x.1.ren π, x.2)).map Prod.fst = (ws.map Prod.fst).map (Loc.ren π) := by
  simp [Function.comp_def]

theorem ren_seq (a b : Log) : (a.seq b).ren π = (a.ren π).seq (b.ren π) := by
  simp only [ren, seq, List.map_append, Log.mk.injEq, writeLocs_ren', and_true]
  refine ⟨?_, ?_, ?_⟩
  · congr 1
    exact List.map_filter_comm _ _ _ (fun l => by simp only [contains_ren]) _
  · congr 1
    exact List.map_filter_comm _ _ _ (fun x => by simp only [contains_ren]) _
  · congr 1
    exact List.map_filter_comm _ _ _ (fun x => by simp only [contains_ren]) _

theorem writeLocs_ren (g : Log) : (g.ren π).writeLocs = g.writeLocs.map (Loc.ren π) := by
  simp [writeLocs, ren, Function.comp_def]

theorem wcrLocs_ren (g : Log) : (g.ren π).wcrLocs = g.wcrLocs.map (Loc.ren π) := by
  simp [wcrLocs, ren, Function.comp_def]

theorem reads_ren (g : Log) : (g.ren π).reads = g.reads.map (Loc.ren π) := rfl

theorem pushedQueues_ren (g : Log) : (g.ren π).pushedQueues = g.pushedQueues.map (Loc.ren π) := by
  simp [pushedQueues, ren, Function.comp_def]

theorem exclusiveQueues_ren (g : Log) : (g.ren π).exclusiveQueues = g.exclusiveQueues.map (Loc.ren π) := by
  simp only [exclusiveQueues, ren, List.filterMap_map, List.map_filterMap]
  congr 1; funext x; obtain ⟨l, e⟩ := x; cases e <;> rfl

theorem mem_ren {l : Loc} {xs : List Loc} : l.ren π ∈ xs.map (Loc.ren π) ↔ l ∈ xs := by
  have hf : Function.Injective (Loc.ren π) := fun _ _ h => (Loc.ren_inj π).1 h
  exact List.mem_map_of_injective hf

theorem conflicts_ren (a b : Log) : conflicts (a.ren π) (b.ren π) = conflicts a b := by
  have hf : Function.Injective (Loc.ren π) := fun _ _ h => (Loc.ren_inj π).1 h
  unfold conflicts
  simp only [writeLocs_ren, wcrLocs_ren, reads_ren, pushedQueues_ren, exclusiveQueues_ren, List.any_map,
    Function.comp_def]
  simp [ren, List.any_map, Function.comp_def, hf.eq_iff, π.injective.eq_iff, mem_ren]

theorem pairwiseFree_ren : ∀ gs : List Log, pairwiseFree (gs.map (ren π)) = pairwiseFree gs
  | [] => rfl
  | g :: gs => by
    simp only [List.map_cons, pairwiseFree, List.all_map, Function.comp_def, conflicts_ren, pairwiseFree_ren gs]

theorem ren_unionAll (gs : List Log) : (unionAll gs).ren π = unionAll (gs.map (ren π)) := by
  simp [unionAll, ren, List.map_flatMap, List.flatMap_map]

theorem ren_dropNames (g : Log) (ns : List Name) : (g.dropNames ns).ren π = (g.ren π).dropNames (ns.map π) := by
  have hc : ∀ n, (ns.map π).contains (π n) = ns.contains n := by
    intro n; simp [π.injective.eq_iff]
  simp only [dropNames, ren, Log.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    exact List.map_filter_comm _ _ _ (fun x => by simp only [Loc.ren_name, hc]) _

end Log

/-! ## Merging renamed logs -/

/-- Every alias binding of the log whose name satisfies `P` binds a base
satisfying `P`. -/
def Log.AliasP (P : Name → Prop) (g : Log) : Prop := ∀ x ∈ g.aliasSets, P x.1 → P x.2.base

/-- Every memory name in the log satisfies `P`. -/
structure Log.NamesP (P : Name → Prop) (g : Log) : Prop where
  reads : ∀ l ∈ g.reads, P l.name
  writes : ∀ x ∈ g.writes, P x.1.name
  wcrs : ∀ x ∈ g.wcrs, P x.1.name
  aliasSets : ∀ x ∈ g.aliasSets, P x.1 ∧ P x.2.base
  queues : ∀ x ∈ g.queues, P x.1.name

theorem Log.NamesP.aliasP {P : Name → Prop} {g : Log} (h : Log.NamesP P g) : Log.AliasP P g :=
  fun x hx _ => (h.aliasSets x hx).2

namespace Merge

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {w : WcrSem}

theorem applyWrites_sim : ∀ (ws : List (Loc × Val)) {m₂ m₁ : Mem}, Mem.Sim π P L m₂ m₁ →
    Mem.Sim π P L (applyWrites m₂ (ws.map fun (l, v) => (l.ren π, v))) (applyWrites m₁ ws)
  | [], _, _, h => h
  | (l, v) :: ws, _, _, h => by
    simp only [applyWrites, List.map_cons, List.foldl_cons] at *
    exact applyWrites_sim ws (h.updCell l v v fun _ => .inr rfl)

theorem applyWcrs_sim (hu : ∀ op v, w op .undef v = .undef) :
    ∀ (cs : List (Loc × WcrOp × Val)) {m₂ m₁ : Mem}, Mem.Sim π P L m₂ m₁ →
      Mem.Sim π P L (applyWcrs w m₂ (cs.map fun (l, op, v) => (l.ren π, op, v))) (applyWcrs w m₁ cs)
  | [], _, _, h => h
  | (l, op, v) :: cs, _, _, h => by
    simp only [applyWcrs, List.map_cons, List.foldl_cons] at *
    refine applyWcrs_sim hu cs (h.updCell l _ _ fun hl => ?_)
    rcases h.cells l hl with ⟨hL, h₁⟩ | h₁
    · rw [h₁, hu]; exact .inl ⟨hL, rfl⟩
    · rw [h₁]; exact .inr rfl

theorem applyAliases_sim : ∀ (as : List (Name × AliasBinding)) {m₂ m₁ : Mem}, (∀ x ∈ as, P x.1 → P x.2.base) →
    Mem.Sim π P L m₂ m₁ → Mem.Sim π P L (applyAliases m₂ (as.map fun (n, b) => (π n, b.ren π))) (applyAliases m₁ as)
  | [], _, _, _, h => h
  | (n, b) :: as, _, _, ha, h => by
    simp only [applyAliases, List.map_cons, List.foldl_cons] at *
    exact applyAliases_sim as (fun x hx => ha x (List.mem_cons_of_mem _ hx))
      (h.setAlias n b (ha (n, b) List.mem_cons_self))

theorem applyQueues_sim : ∀ (qs : List (Loc × QueueEffect)) {m₂ m₁ : Mem}, Mem.Sim π P L m₂ m₁ →
    Mem.Sim π P L (applyQueues m₂ (qs.map fun (l, e) => (l.ren π, e))) (applyQueues m₁ qs)
  | [], _, _, h => h
  | (l, e) :: qs, _, _, h => by
    simp only [applyQueues, List.map_cons, List.foldl_cons] at *
    refine applyQueues_sim qs ?_
    cases e with
    | pushOnly vs =>
      simp only
      by_cases hl : P l.name
      · rw [h.queues l hl]
        exact h.updQueue l _
      · refine ⟨h.aliases, h.aliasP, h.cells, fun l' hl' => ?_⟩
        have hne : l' ≠ l := fun e => hl (e ▸ hl')
        have : l'.ren π ≠ l.ren π := fun e => hne ((Loc.ren_inj π).1 e)
        simp only [Function.update_of_ne this, Function.update_of_ne hne]
        exact h.queues l' hl'
    | exclusive final => exact h.updQueue l final

end Merge

/-- **Merging renamed logs onto a simulating memory**: both merges fail, or
both succeed with simulating results. -/
theorem merge_sim {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {w : WcrSem} (hu : ∀ op v, w op .undef v = .undef)
    {m₂ m₁ : Mem} (h : Mem.Sim π P L m₂ m₁) (gs : List Log) (hg : ∀ g ∈ gs, Log.NamesP P g) :
    (∃ e₂ e₁, merge w m₂ (gs.map (Log.ren π)) = .error e₂ ∧ merge w m₁ gs = .error e₁) ∨
      ∃ a₂ a₁, merge w m₂ (gs.map (Log.ren π)) = .ok a₂ ∧ merge w m₁ gs = .ok a₁ ∧ Mem.Sim π P L a₂ a₁ := by
  unfold merge
  rw [Log.pairwiseFree_ren]
  split
  · refine .inr ⟨_, _, rfl, rfl, ?_⟩
    rw [← Log.ren_unionAll]
    refine Merge.applyQueues_sim _ (Merge.applyAliases_sim _ ?_ (Merge.applyWcrs_sim hu _
      (Merge.applyWrites_sim _ h)))
    intro x hx
    simp only [Log.unionAll, List.mem_flatMap] at hx
    obtain ⟨g, hgm, hx⟩ := hx
    exact (hg g hgm).aliasP x hx
  · refine .inl ?_
    have herr : ∀ (gs : List Log), ∃ e, (match Merge.findConflict gs with
        | some (g, h) => Except.error (Merge.witness g h)
        | none => (Except.error (.race default) : Except MergeError Mem)) = .error e := by
      intro gs
      cases Merge.findConflict gs with
      | none => exact ⟨_, rfl⟩
      | some p => exact ⟨_, rfl⟩
    obtain ⟨e₂, h₂⟩ := herr (gs.map (Log.ren π))
    obtain ⟨e₁, h₁⟩ := herr gs
    exact ⟨e₂, e₁, h₂, h₁⟩

/-! ## Contexts agreeing on names -/

/-- Contexts related by `π` on what every run reads: the interpretation and
the control-flow hierarchy (mention counts of private transients). -/
structure CtxSim (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx) : Prop where
  I : ctx₂.I = ctx₁.I
  top : ctx₂.g.top = ctx₁.g.top

/-- The memory names of fresh names (`Ctx.scope`, read by nested
invocations only) are related by `π` and satisfy `P`. -/
def ScopeSim (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx) : Prop :=
  ∀ x, ctx₂.scope x = π (ctx₁.scope x) ∧ P (ctx₁.scope x)

namespace Ctx

/-- Agreement on a name read in an expression: the same descriptor and, for
a container, its memory name renamed by `π` and satisfying `P`. -/
def Good₀ (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx) (n : Name) : Prop :=
  ctx₂.desc? n = ctx₁.desc? n ∧ (ctx₁.desc? n ≠ none → ctx₂.cname n = π (ctx₁.cname n) ∧ P (ctx₁.cname n))

/-- Agreement on a container name (memlet data, access node): the same
descriptor, the memory name renamed by `π` and satisfying `P`, and agreement
on the names its shape reads. -/
structure Strong (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx) (n : Name) : Prop where
  desc : ctx₂.desc? n = ctx₁.desc? n
  cname : ctx₂.cname n = π (ctx₁.cname n)
  memP : P (ctx₁.cname n)
  shape : ∀ d, ctx₁.desc? n = some d → ∀ x ∈ d.fullRange.names, Good₀ π P ctx₂ ctx₁ x

theorem Strong.good₀ {π : Equiv.Perm Name} {P : Name → Prop} {ctx₂ ctx₁ : Ctx} {n : Name}
    (h : Strong π P ctx₂ ctx₁ n) : Good₀ π P ctx₂ ctx₁ n := ⟨h.desc, fun _ => ⟨h.cname, h.memP⟩⟩

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁)
  {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) (ρ : Valuation)
include hc hm

theorem addr_sim {n : Name} (hg : Good₀ π P ctx₂ ctx₁ n) (idx : Idx) :
    ctx₂.addr ρ m₂ n idx = (ctx₁.addr ρ m₁ n idx).map (Loc.ren π) := by
  unfold Ctx.addr
  rw [hg.1]
  cases hd : ctx₁.desc? n with
  | none => rfl
  | some d =>
    obtain ⟨hcn, hP⟩ := hg.2 (by simp [hd])
    simp only
    rw [hcn, hm.isAliased hP]
    rw [show ctx₂.elemOffset ρ d idx = ctx₁.elemOffset ρ d idx by simp only [Ctx.elemOffset, hc.I]]
    split
    · rfl
    · cases ctx₁.elemOffset ρ d idx <;> rfl

theorem addrP {n : Name} (hg : Good₀ π P ctx₂ ctx₁ n) {idx : Idx} {l : Loc}
    (h : ctx₁.addr ρ m₁ n idx = .ok l) : P l.name := by
  unfold Ctx.addr at h
  cases hd : ctx₁.desc? n with
  | none => rw [hd] at h; cases h
  | some d =>
    rw [hd] at h
    simp only at h
    split at h
    · cases h
    · split at h
      · cases h; exact (hg.2 (by simp [hd])).2
      · cases h

theorem checkBounds_sim (n : Name) (hd : ctx₂.desc? n = ctx₁.desc? n) (idx : Idx) (oob : Bool) :
    ctx₂.checkBounds ρ n idx oob = ctx₁.checkBounds ρ n idx oob := by
  unfold Ctx.checkBounds
  rw [hd, hc.I]

theorem loc_sim {n : Name} (hg : Good₀ π P ctx₂ ctx₁ n) (idx : Idx) (oob : Bool) :
    ctx₂.loc ρ m₂ n idx oob = (ctx₁.loc ρ m₁ n idx oob).map (Loc.ren π) := by
  unfold Ctx.loc
  rw [addr_sim hc hm ρ hg, checkBounds_sim hc hm ρ n hg.1]
  cases ctx₁.addr ρ m₁ n idx with
  | err e => rfl
  | ok l => cases ctx₁.checkBounds ρ n idx oob <;> rfl

theorem locP {n : Name} (hg : Good₀ π P ctx₂ ctx₁ n) {idx : Idx} {oob : Bool} {l : Loc}
    (h : ctx₁.loc ρ m₁ n idx oob = .ok l) : P l.name := by
  unfold Ctx.loc at h
  cases ha : ctx₁.addr ρ m₁ n idx with
  | err e => rw [ha] at h; cases h
  | ok l' =>
    rw [ha] at h
    simp only [Res.ok_bind] at h
    cases hb : ctx₁.checkBounds ρ n idx oob with
    | err e => rw [hb] at h; cases h
    | ok u =>
      rw [hb] at h
      simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj] at h
      subst h
      exact addrP hc hm ρ hg ha

theorem readData_mono {n : Name} (hg : Good₀ π P ctx₂ ctx₁ n) (idx : Idx) :
    OptLe ((ctx₂.exprInterp ρ m₂).readData n idx) ((ctx₁.exprInterp ρ m₁).readData n idx) := by
  simp only [Ctx.exprInterp]
  rw [loc_sim hc hm ρ hg]
  cases hl : ctx₁.loc ρ m₁ n idx false with
  | err e => exact .inr rfl
  | ok l =>
    simp only [Res.map]
    rcases hm.read (locP hc hm ρ hg hl) with h | h <;> rw [h]
    · exact .inl rfl
    · exact .inr rfl

theorem val_mono {x : Sym} (hg : Good₀ π P ctx₂ ctx₁ x) : OptLe (ctx₂.val ρ m₂ x) (ctx₁.val ρ m₁ x) := by
  unfold Ctx.val
  cases ρ x with
  | some v => exact .inr rfl
  | none =>
    simp only
    rw [hg.1]
    cases hd : ctx₁.desc? x with
    | none => exact .inr rfl
    | some d =>
      simp only
      split
      · rw [loc_sim hc hm ρ hg]
        cases hl : ctx₁.loc ρ m₁ x [] false with
        | err e => exact .inr rfl
        | ok l =>
          simp only [Res.map]
          rcases hm.read (locP hc hm ρ hg hl) with h | h <;> rw [h]
          · exact .inl rfl
          · exact .inr rfl
      · exact .inr rfl

theorem happ : (ctx₂.exprInterp ρ m₂).app = (ctx₁.exprInterp ρ m₁).app := by
  simp only [Ctx.exprInterp, hc.I]

theorem hattr : (ctx₂.exprInterp ρ m₂).readAttr = (ctx₁.exprInterp ρ m₁).readAttr := by
  simp only [Ctx.exprInterp, hc.I]

theorem eval_mono (e : Expr) (he : ∀ x ∈ e.names, Good₀ π P ctx₂ ctx₁ x) :
    OptLe (ctx₂.eval ρ m₂ e) (ctx₁.eval ρ m₁ e) :=
  Expr.eval_mono' (happ hc hm ρ) (hattr hc hm ρ) (Q := Good₀ π P ctx₂ ctx₁)
    (fun d is hg => readData_mono hc hm ρ hg is) (fun x hg => val_mono hc hm ρ hg) e he

theorem evalInt_mono (e : Expr) (he : ∀ x ∈ e.names, Good₀ π P ctx₂ ctx₁ x) :
    OptLe (ctx₂.evalInt ρ m₂ e) (ctx₁.evalInt ρ m₁ e) :=
  Expr.evalInt_mono' (happ hc hm ρ) (hattr hc hm ρ) (Q := Good₀ π P ctx₂ ctx₁)
    (fun d is hg => readData_mono hc hm ρ hg is) (fun x hg => val_mono hc hm ρ hg) e he

theorem evalR_sim (e : Expr) (he : ∀ x ∈ e.names, Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.evalR ρ m₂ e) (ctx₁.evalR ρ m₁ e) :=
  ResSim.ofOption rfl (eval_mono hc hm ρ e he)

theorem evalIntR_sim (e : Expr) (he : ∀ x ∈ e.names, Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.evalIntR ρ m₂ e) (ctx₁.evalIntR ρ m₁ e) :=
  ResSim.ofOption rfl (evalInt_mono hc hm ρ e he)

theorem evalBoolR_sim (e : Expr) (he : ∀ x ∈ e.names, Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.evalBoolR ρ m₂ e) (ctx₁.evalBoolR ρ m₁ e) :=
  ResSim.ofOption rfl (Expr.evalBool_mono' (happ hc hm ρ) (hattr hc hm ρ) (Q := Good₀ π P ctx₂ ctx₁)
    (fun d is hg => readData_mono hc hm ρ hg is) (fun x hg => val_mono hc hm ρ hg) e he)

theorem pointsOf_sim {n : Name} (hd : Strong π P ctx₂ ctx₁ n) (sub : Option Subset)
    (hs : ∀ x ∈ optSubsetNames sub, Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.pointsOf ρ m₂ n sub) (ctx₁.pointsOf ρ m₁ n sub) := by
  unfold Ctx.pointsOf
  rw [hd.desc]
  cases hdd : ctx₁.desc? n with
  | none => exact ResSim.err _
  | some d =>
    refine ResSim.ofOption rfl (Subset.points_mono (happ hc hm ρ) (hattr hc hm ρ) (Q := Good₀ π P ctx₂ ctx₁)
      (fun d is hg => readData_mono hc hm ρ hg is) (fun x hg => val_mono hc hm ρ hg) _ ?_)
    intro x hx
    cases sub with
    | some S => exact hs x (by simpa [optSubsetNames] using hx)
    | none => exact hd.shape d hdd x (by simpa [Subset.names] using hx)

theorem rangePoints_sim (r : Range) (hr : ∀ x ∈ r.names, Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.rangePoints ρ m₂ r) (ctx₁.rangePoints ρ m₁ r) :=
  ResSim.ofOption rfl (Range.points_mono (happ hc hm ρ) (hattr hc hm ρ) (Q := Good₀ π P ctx₂ ctx₁)
    (fun d is hg => readData_mono hc hm ρ hg is) (fun x hg => val_mono hc hm ρ hg) r hr)

end Ctx

/-! ## Simulating states -/

theorem Log.NamesP.seq {P : Name → Prop} {a b : Log} (ha : Log.NamesP P a) (hb : Log.NamesP P b) :
    Log.NamesP P (a.seq b) := by
  refine ⟨fun x hx => ?_, fun x hx => ?_, fun x hx => ?_, fun x hx => ?_, fun x hx => ?_⟩ <;>
    simp only [Log.seq, List.mem_append, List.mem_filter] at hx
  · rcases hx with hx | hx
    exacts [ha.reads x hx, hb.reads x hx.1]
  · rcases hx with hx | hx
    exacts [ha.writes x hx.1, hb.writes x hx]
  · rcases hx with hx | hx
    exacts [ha.wcrs x hx.1, hb.wcrs x hx]
  · rcases hx with hx | hx
    exacts [ha.aliasSets x hx, hb.aliasSets x hx]
  · rcases hx with hx | hx
    exacts [ha.queues x hx, hb.queues x hx]

theorem Log.NamesP.empty (P : Name → Prop) : Log.NamesP P Log.empty :=
  ⟨fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim⟩

theorem Log.NamesP.read {P : Name → Prop} {l : Loc} (h : P l.name) : Log.NamesP P { reads := [l] } :=
  ⟨fun x hx => (List.mem_singleton.1 hx) ▸ h, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim⟩

theorem Log.NamesP.write {P : Name → Prop} {l : Loc} (v : Val) (h : P l.name) : Log.NamesP P { writes := [(l, v)] } :=
  ⟨fun _ hx => (List.not_mem_nil hx).elim, fun x hx => (List.mem_singleton.1 hx) ▸ h,
    fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim⟩

theorem Log.NamesP.wcr {P : Name → Prop} {l : Loc} (op : WcrOp) (v : Val) (h : P l.name) :
    Log.NamesP P { wcrs := [(l, op, v)] } :=
  ⟨fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun x hx => (List.mem_singleton.1 hx) ▸ h, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim⟩

theorem Log.NamesP.queue {P : Name → Prop} {l : Loc} (e : QueueEffect) (h : P l.name) :
    Log.NamesP P { queues := [(l, e)] } :=
  ⟨fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun x hx => (List.mem_singleton.1 hx) ▸ h⟩

theorem Log.NamesP.alias {P : Name → Prop} {n : Name} {b : AliasBinding} (hn : P n) (hb : P b.base) :
    Log.NamesP P { aliasSets := [(n, b)] } :=
  ⟨fun _ hx => (List.not_mem_nil hx).elim, fun _ hx => (List.not_mem_nil hx).elim,
    fun _ hx => (List.not_mem_nil hx).elim, fun x hx => (List.mem_singleton.1 hx) ▸ ⟨hn, hb⟩,
    fun _ hx => (List.not_mem_nil hx).elim⟩

theorem Log.NamesP.dropNames {P : Name → Prop} {g : Log} (h : Log.NamesP P g) (ns : List Name) :
    Log.NamesP P (g.dropNames ns) :=
  ⟨fun x hx => h.reads x (List.mem_filter.1 hx).1, fun x hx => h.writes x (List.mem_filter.1 hx).1,
    fun x hx => h.wcrs x (List.mem_filter.1 hx).1, fun x hx => h.aliasSets x (List.mem_filter.1 hx).1,
    fun x hx => h.queues x (List.mem_filter.1 hx).1⟩

theorem Log.NamesP.stripQueue {P : Name → Prop} {g : Log} (h : Log.NamesP P g) (l : Loc) :
    Log.NamesP P (stripQueue g l) :=
  ⟨h.reads, h.writes, h.wcrs, h.aliasSets, fun x hx => h.queues x (List.mem_filter.1 hx).1⟩

theorem Log.NamesP.unionAll {P : Name → Prop} {gs : List Log} (h : ∀ g ∈ gs, Log.NamesP P g) :
    Log.NamesP P (Log.unionAll gs) := by
  refine ⟨fun x hx => ?_, fun x hx => ?_, fun x hx => ?_, fun x hx => ?_, fun x hx => ?_⟩ <;>
    simp only [Log.unionAll, List.mem_flatMap] at hx <;> obtain ⟨g, hg, hx⟩ := hx
  exacts [(h g hg).reads x hx, (h g hg).writes x hx, (h g hg).wcrs x hx, (h g hg).aliasSets x hx,
    (h g hg).queues x hx]

/-- A dataflow state simulating another. -/
structure St.Sim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (a₂ a₁ : St) : Prop where
  mem : Mem.Sim π P L a₂.mem a₁.mem
  log : a₂.log = a₁.log.ren π
  vals : a₂.vals = a₁.vals
  logP : Log.NamesP P a₁.log

/-- A control-flow state simulating another. -/
structure CSt.Sim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (c₂ c₁ : CSt) : Prop where
  ρ : c₂.ρ = c₁.ρ
  mem : Mem.Sim π P L c₂.mem c₁.mem
  log : c₂.log = c₁.log.ren π
  logP : Log.NamesP P c₁.log

/-- A result with a state: the same value, simulating states. -/
def St.SimR {α : Type} (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : α × St) : Prop :=
  x₂.1 = x₁.1 ∧ St.Sim π P L x₂.2 x₁.2

namespace St.Sim

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {st₂ st₁ : St}

theorem logged (h : St.Sim π P L st₂ st₁) (g : Log) (hg : Log.NamesP P g) :
    St.Sim π P L (st₂.logged (g.ren π)) (st₁.logged g) :=
  ⟨h.mem, by simp only [St.logged, h.log, Log.ren_seq], h.vals, h.logP.seq hg⟩

theorem withMem (h : St.Sim π P L st₂ st₁) {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) :
    St.Sim π P L { st₂ with mem := m₂ } { st₁ with mem := m₁ } := ⟨hm, h.log, h.vals, h.logP⟩

theorem forNode (h : St.Sim π P L st₂ st₁) : St.Sim π P L st₂.forNode st₁.forNode :=
  ⟨h.mem, rfl, h.vals, Log.NamesP.empty P⟩

theorem afterNode {a₂ a₁ : St} (h : St.Sim π P L st₂ st₁) (ha : St.Sim π P L a₂ a₁) :
    St.Sim π P L (st₂.afterNode a₂) (st₁.afterNode a₁) :=
  ⟨ha.mem, by simp only [St.afterNode, h.log, ha.log, Log.ren_seq], ha.vals, h.logP.seq ha.logP⟩

theorem setVal (h : St.Sim π P L st₂ st₁) (key : NodeId × Conn) (v : Val) :
    St.Sim π P L (st₂.setVal key v) (st₁.setVal key v) := ⟨h.mem, h.log, by simp [St.setVal, h.vals], h.logP⟩

theorem val? (h : St.Sim π P L st₂ st₁) (key : NodeId × Conn) : st₂.val? key = st₁.val? key := by
  simp [St.val?, h.vals]

theorem bindAlias (h : St.Sim π P L st₂ st₁) (n : Name) (b : AliasBinding) (hn : P n) (hb : P b.base) :
    St.Sim π P L (st₂.bindAlias (π n) (b.ren π)) (st₁.bindAlias n b) :=
  ⟨h.mem.setAlias n b (fun _ => hb), by simp only [St.bindAlias, h.log, Log.ren_seq]; rfl, h.vals,
    h.logP.seq (Log.NamesP.alias hn hb)⟩

end St.Sim

namespace St

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation)
  {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁)
include hc h

theorem read_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (idx : Idx) (oob : Bool) :
    ResSim (St.SimR π P L) (st₂.read ctx₂ ρ n idx oob) (st₁.read ctx₁ ρ n idx oob) := by
  unfold St.read
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind]
    rcases h.mem.read hP with hu | heq
    · rw [hu]; exact ResSim.ub rfl _
    · rw [heq]
      split
      · exact ResSim.err _
      · refine ResSim.ok ⟨rfl, ?_⟩
        have := h.logged { reads := [st₁.mem.resolve l] } (Log.NamesP.read (h.mem.resolveP hP))
        simpa [Log.ren, h.mem.resolve hP] using this

theorem readMany_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (pts : List Idx) (oob : Bool) :
    ResSim (St.SimR π P L) (st₂.readMany ctx₂ ρ n pts oob) (st₁.readMany ctx₁ ρ n pts oob) := by
  unfold St.readMany
  refine ResSim.foldlM (R := St.SimR π P L) (fun p a b hab => ?_) pts ⟨rfl, h⟩
  obtain ⟨acc₂, a₂⟩ := a
  obtain ⟨acc₁, a₁⟩ := b
  obtain ⟨hacc, ha⟩ := hab
  simp only at hacc ha ⊢
  subst hacc
  exact ResSim.bind (read_sim hc ρ ha hg p oob) fun x y ⟨hv, hs⟩ => by
    obtain ⟨v₂, s₂⟩ := x
    obtain ⟨v₁, s₁⟩ := y
    simp only at hv hs
    subst hv
    exact ResSim.ok ⟨rfl, hs⟩

theorem write_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (idx : Idx) (oob : Bool) (v : Val) :
    ResSim (St.Sim π P L) (st₂.write ctx₂ ρ n idx oob v) (st₁.write ctx₁ ρ n idx oob v) := by
  unfold St.write
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind, Res.pure_eq]
    refine ResSim.ok ⟨h.mem.write hP v, ?_, h.vals, h.logP.seq (Log.NamesP.write v (h.mem.resolveP hP))⟩
    simp only [h.log, Log.ren_seq, h.mem.resolve hP]
    rfl

theorem wcr_sim (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef) {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n)
    (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    ResSim (St.Sim π P L) (st₂.wcr ctx₂ ρ n idx oob op v) (st₁.wcr ctx₁ ρ n idx oob op v) := by
  unfold St.wcr
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind, Res.pure_eq, hc.I]
    refine ResSim.ok ⟨?_, ?_, h.vals, h.logP.seq (Log.NamesP.wcr op v (h.mem.resolveP hP))⟩
    · simp only [Mem.write, h.mem.resolve hP]
      refine h.mem.updCell _ _ _ fun _ => ?_
      rcases h.mem.readL hP with ⟨hL, h₁⟩ | h₁ <;> rw [h₁]
      · rw [hu]; exact .inl ⟨hL, rfl⟩
      · exact .inr rfl
    · simp only [h.log, Log.ren_seq, h.mem.resolve hP]
      rfl

theorem store_sim (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef) {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n)
    (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    ResSim (St.Sim π P L) (st₂.store ctx₂ ρ n idx oob w v) (st₁.store ctx₁ ρ n idx oob w v) := by
  cases w with
  | none => exact write_sim hc ρ h hg idx oob v
  | some op => exact wcr_sim hc ρ h hu hg idx oob op v

theorem queue_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (idx : Idx) :
    ResSim Eq (st₂.queue ctx₂ ρ n idx) (st₁.queue ctx₁ ρ n idx) := by
  unfold St.queue
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind, Res.pure_eq, h.mem.resolve hP, h.mem.queues _ (h.mem.resolveP hP)]
    exact ResSim.ok rfl

theorem push_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (idx : Idx) (vs : List Val) :
    ResSim (St.Sim π P L) (st₂.push ctx₂ ρ n idx vs) (st₁.push ctx₁ ρ n idx vs) := by
  unfold St.push
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind, Res.pure_eq]
    split
    · exact ResSim.ok h
    · refine ResSim.ok ⟨Mem.Sim.pushMany hP vs h.mem, ?_, h.vals, h.logP.seq (Log.NamesP.queue _ (h.mem.resolveP hP))⟩
      simp only [h.log, Log.ren_seq, h.mem.resolve hP]
      rfl

theorem popN_sim {n : Name} (hg : Ctx.Good₀ π P ctx₂ ctx₁ n) (idx : Idx) (k : Nat) :
    ResSim (St.SimR π P L) (st₂.popN ctx₂ ρ n idx k) (st₁.popN ctx₁ ρ n idx k) := by
  unfold St.popN
  rw [Ctx.loc_sim hc h.mem ρ hg]
  cases hl : ctx₁.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    have hP := Ctx.locP hc h.mem ρ hg hl
    simp only [Res.map, Res.ok_bind, Res.pure_eq, h.mem.resolve hP, h.mem.queues _ (h.mem.resolveP hP)]
    split
    · exact ResSim.ok ⟨rfl, h⟩
    · split
      · exact ResSim.ub rfl _
      · refine ResSim.ok ⟨rfl, ⟨h.mem.updQueue _ _, ?_, h.vals, h.logP.seq (Log.NamesP.queue _ (h.mem.resolveP hP))⟩⟩
        simp only [h.log, Log.ren_seq]
        rfl

end St

/-! ## Contexts agreeing on the names of a state -/

/-- The contexts agree on every name of state `s`. -/
structure SNames (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx) (s : State) : Prop where
  data : ∀ x ∈ s.dataNames, Ctx.Strong π P ctx₂ ctx₁ x
  expr : ∀ x ∈ s.exprNames, Ctx.Good₀ π P ctx₂ ctx₁ x
  /-- Only states with nested SDFGs read the fresh-name scope. -/
  scope : (∃ n ∈ s.nodes, ∃ g mapping ni, n.kind = .nested g mapping ni) → ScopeSim π P ctx₂ ctx₁

namespace SNames

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} {s : State} (hS : SNames π P ctx₂ ctx₁ s)
include hS

theorem src {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset} (h : s.srcEndpoint e = .data d sub) :
    Ctx.Strong π P ctx₂ ctx₁ d ∧ ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x := by
  obtain ⟨hd, hsub⟩ := State.srcEndpoint_names h
  refine ⟨hS.data d hd, fun x hx => hS.expr x ?_⟩
  rcases hsub with rfl | rfl
  · simp [optSubsetNames] at hx
  · exact State.leafSrcSubset_names he x hx

theorem dst {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset} (h : s.dstEndpoint e = .data d sub) :
    Ctx.Strong π P ctx₂ ctx₁ d ∧ ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x := by
  obtain ⟨hd, rfl⟩ := State.dstEndpoint_names h
  exact ⟨hS.data d hd, fun x hx => hS.expr x (State.leafDstSubset_names he x hx)⟩

theorem node_expr {n : DNode} (hn : n ∈ s.nodes) {x : Name} (hx : x ∈ n.exprNames) : Ctx.Good₀ π P ctx₂ ctx₁ x :=
  hS.expr x (State.mem_exprNames_node hn hx)

end SNames

/-! ## Leaf functions -/

section Leaf

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation)
  (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
include hc

theorem streamPoint_sim {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (streamPoint ctx₂ ρ m₂ d sub) (streamPoint ctx₁ ρ m₁ d sub) := by
  unfold streamPoint
  exact ResSim.bindEq (Ctx.pointsOf_sim hc hm ρ hd sub hs) fun pts => ResSim.of_eq fun _ => rfl

theorem lowerCorner_sim {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) {n : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ n)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.lowerCorner ρ m₂ n sub) (ctx₁.lowerCorner ρ m₁ n sub) := by
  unfold Ctx.lowerCorner
  refine ResSim.bindEq (Ctx.pointsOf_sim hc hm ρ hd sub hs) fun _ => ?_
  cases sub with
  | none => simp only [hd.desc]; exact ResSim.of_eq fun _ => rfl
  | some S =>
    simp only
    split
    · exact ResSim.err _
    · rename_i r hr
      refine ResSim.mapM_mem _ (fun (dim : Dim) hdim => ResSim.ofOption rfl
        (Ctx.evalInt_mono hc hm ρ dim.lo fun x hx => hs x ?_))
      simp only [optSubsetNames, Option.map_some, Option.getD_some]
      cases S with
      | range r' =>
        simp only [Option.some.injEq] at hr
        subst hr
        exact List.mem_flatMap.2 ⟨dim, hdim, by simp [Dim.names, hx]⟩
      | union rs =>
        simp only at hr
        exact List.mem_flatMap.2 ⟨r, List.mem_of_mem_head? hr, List.mem_flatMap.2 ⟨dim, hdim, by simp [Dim.names, hx]⟩⟩

theorem readScalarData_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) (oob : Bool) :
    ResSim (St.SimR π P L) (readScalarData ctx₂ ρ d sub oob st₂) (readScalarData ctx₁ ρ d sub oob st₁) := by
  unfold readScalarData
  refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hd sub hs) fun pts => ?_
  split
  · rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
    refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ?_)
    · exact ResSim.bind (St.popN_sim hc ρ h hd.good₀ _ 1) fun x y hxy => by
        obtain ⟨hv, hs⟩ := hxy
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
    · exact ResSim.bind (St.read_sim hc ρ h hd.good₀ _ oob) fun x y hxy => by
        obtain ⟨hv, hs⟩ := hxy
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
  · exact ResSim.err _

theorem readEdgeScalar_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {e : DEdge} (he : e ∈ s.edges)
    {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (St.SimR π P L) (readEdgeScalar ctx₂ ρ s e st₂) (readEdgeScalar ctx₁ ρ s e st₁) := by
  unfold readEdgeScalar
  split
  · rw [h.val?]; exact ResSim.ok ⟨rfl, h⟩
  · rename_i d sub hsrc
    obtain ⟨hd, hs⟩ := hS.src he hsrc
    exact readScalarData_sim hc ρ h hd sub hs _
  · exact ResSim.err _

include hu in
theorem copyData_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {sd : Name} (hsd : Ctx.Strong π P ctx₂ ctx₁ sd)
    (ssub : Option Subset) (hss : ∀ x ∈ optSubsetNames ssub, Ctx.Good₀ π P ctx₂ ctx₁ x)
    {dd : Name} (hdd : Ctx.Strong π P ctx₂ ctx₁ dd)
    (dsub : Option Subset) (hds : ∀ x ∈ optSubsetNames dsub, Ctx.Good₀ π P ctx₂ ctx₁ x) (w : Option WcrOp)
    (oob : Bool) :
    ResSim (St.Sim π P L) (copyData ctx₂ ρ st₂ sd ssub dd dsub w oob) (copyData ctx₁ ρ st₁ sd ssub dd dsub w oob) := by
  unfold copyData
  rw [show ctx₂.isStream sd = ctx₁.isStream sd by simp only [Ctx.isStream, hsd.desc],
    show ctx₂.isStream dd = ctx₁.isStream dd by simp only [Ctx.isStream, hdd.desc]]
  split
  · exact ResSim.err _
  · refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hdd dsub hds) fun dpts => ?_
    refine ResSim.bindEq (streamPoint_sim hc ρ h.mem hsd ssub hss) fun sp => ?_
    refine ResSim.bind (St.popN_sim hc ρ h hsd.good₀ sp dpts.length) fun x y ⟨hv, hs⟩ => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    simp only at hv hs ⊢
    subst hv
    exact ResSim.foldlM (fun qv b₂ b₁ hb => St.store_sim hc ρ hb hu hdd.good₀ qv.1 oob w qv.2) _ hs
  · refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hsd ssub hss) fun spts => ?_
    refine ResSim.bindEq (streamPoint_sim hc ρ h.mem hdd dsub hds) fun dp => ?_
    refine ResSim.bind (St.readMany_sim hc ρ h hsd.good₀ spts oob) fun x y ⟨hv, hs⟩ => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    simp only at hv hs ⊢
    subst hv
    exact St.push_sim hc ρ hs hdd.good₀ dp vs₂
  · refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hsd ssub hss) fun spts => ?_
    refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hdd dsub hds) fun dpts => ?_
    refine ResSim.foldlM (fun pq b₂ b₁ hb => ?_) _ h
    exact ResSim.bind (St.read_sim hc ρ hb hsd.good₀ pq.1 oob) fun x y ⟨hv, hs⟩ => by
      obtain ⟨v₂, a₂⟩ := x
      obtain ⟨v₁, a₁⟩ := y
      simp only at hv hs ⊢
      subst hv
      exact St.store_sim hc ρ hs hu hdd.good₀ pq.2 oob w v₂

include hu in
theorem execCopy_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {e : DEdge} (he : e ∈ s.edges) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (execCopy ctx₂ ρ s e st₂) (execCopy ctx₁ ρ s e st₁) := by
  unfold execCopy
  split
  · exact ResSim.ok h
  · split
    · rename_i sd ssub dd dsub hsrc hdst
      obtain ⟨hsd, hss⟩ := hS.src he hsrc
      obtain ⟨hdd, hds⟩ := hS.dst he hdst
      exact copyData_sim hc ρ hu h hsd ssub hss hdd dsub hds _ _
    · exact ResSim.err _

omit hc in
theorem forall₂_loc_mem' {ctx : Ctx} {ρ : Valuation} {m : Mem} {d : Name} {oob : Bool} {pts : List Idx}
    {ls : List Loc} (hf : List.Forall₂ (fun p l => ctx.loc ρ m d p oob = .ok l) pts ls) {p : Idx} (hp : p ∈ pts) :
    ∃ l ∈ ls, ctx.loc ρ m d p oob = .ok l := by
  induction hf with
  | nil => cases hp
  | cons hl _ ih =>
    rcases List.mem_cons.1 hp with rfl | hp
    · exact ⟨_, List.mem_cons_self .., hl⟩
    · obtain ⟨l, hl', h⟩ := ih hp
      exact ⟨l, List.mem_cons_of_mem _ hl', h⟩

omit hc in
/-- After a normal `readMany`, every point's cell is initialized. -/
theorem St.readMany_defined {ctx : Ctx} {ρ : Valuation} {st st' : St} {n : Name} {pts : List Idx} {oob : Bool}
    {vs : List Val} (h : st.readMany ctx ρ n pts oob = .ok (vs, st')) :
    st'.mem = st.mem ∧ ∀ p ∈ pts, ∀ l, ctx.loc ρ st.mem n p oob = .ok l → (st.mem.read l).isUndef = false := by
  unfold St.readMany at h
  obtain ⟨ls, hf, -, rfl, hd⟩ := St.readMany_ok ctx ρ pts [] st vs st' h
  refine ⟨rfl, fun p hp l hl => ?_⟩
  obtain ⟨l', hl', hloc⟩ := forall₂_loc_mem' hf hp
  rw [hl] at hloc
  cases hloc
  exact hd l hl'

theorem bindInData_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) (oob : Bool) :
    ResSim (St.SimR π P L) (bindInData ctx₂ ρ d sub oob st₂) (bindInData ctx₁ ρ d sub oob st₁) := by
  unfold bindInData
  rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
  refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ?_)
  · refine ResSim.bindEq (streamPoint_sim hc ρ h.mem hd sub hs) fun p => ?_
    exact ResSim.bindEq (St.queue_sim hc ρ h hd.good₀ p) fun q => ResSim.ok ⟨rfl, h⟩
  · refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hd sub hs) fun pts => ?_
    split
    · exact ResSim.bind (St.read_sim hc ρ h hd.good₀ _ oob) fun x y hxy => by
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        obtain ⟨hv, hs⟩ := hxy
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
    · refine ResSim.bindEq (lowerCorner_sim hc ρ h.mem hd sub hs) fun lo => ?_
      cases hr₂ : st₂.readMany ctx₂ ρ d pts oob with
      | err e =>
        have := St.readMany_sim hc ρ h hd.good₀ pts oob
        rw [hr₂] at this
        obtain ⟨_, -, hu'⟩ := ResSim.err_inv' this rfl
        rcases hu' with hu' | h₁
        · exact ResSim.ub hu' _
        · rw [h₁]; exact ResSim.err _
      | ok x₂ =>
        obtain ⟨vs₂, a₂⟩ := x₂
        have hs' := St.readMany_sim hc ρ h hd.good₀ pts oob
        rw [hr₂] at hs'
        rcases hs' with ⟨e, he, -⟩ | ⟨e, -, he⟩ | ⟨a, b, ha, hb, hab⟩
        · cases he
        · cases he
        · cases ha
          obtain ⟨vs₁, a₁⟩ := b
          obtain ⟨-, hst⟩ := hab
          simp only at hst
          rw [hb]
          simp only [Res.ok_bind, Res.pure_eq]
          refine ResSim.ok ⟨?_, hst⟩
          obtain ⟨hm₂, hd₂⟩ := St.readMany_defined hr₂
          obtain ⟨hm₁, -⟩ := St.readMany_defined hb
          simp only [hm₂, hm₁]
          congr 1
          funext rel
          split
          · split
            · next hc' =>
              rw [Ctx.loc_sim hc h.mem ρ hd.good₀]
              cases hl : ctx₁.loc ρ st₁.mem d (List.zipWith (· + ·) lo rel) oob with
              | err e => rfl
              | ok l =>
                simp only [Res.map]
                have hdef := hd₂ _ (by simpa using hc') (l.ren π) (by rw [Ctx.loc_sim hc h.mem ρ hd.good₀, hl]; rfl)
                rcases h.mem.read (Ctx.locP hc h.mem ρ hd.good₀ hl) with h₁ | h₁
                · rw [h₁] at hdef; cases hdef
                · rw [h₁]
            · rfl
          · rfl

theorem bindIn_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {e : DEdge} (he : e ∈ s.edges) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.SimR π P L) (bindIn ctx₂ ρ s e st₂) (bindIn ctx₁ ρ s e st₁) := by
  unfold bindIn
  split
  · rw [h.val?]; exact ResSim.ok ⟨rfl, h⟩
  · rename_i d sub hsrc
    obtain ⟨hd, hs⟩ := hS.src he hsrc
    exact ResSim.bind (bindInData_sim hc ρ h hd sub hs _) fun x y hxy => by
      obtain ⟨c₂, a₂⟩ := x
      obtain ⟨c₁, a₁⟩ := y
      obtain ⟨hv, hs⟩ := hxy
      simp only at hv hs ⊢
      subst hv
      exact ResSim.ok ⟨rfl, hs⟩
  · exact ResSim.err _

theorem bindInputs_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.SimR π P L) (bindInputs ctx₂ ρ s n st₂) (bindInputs ctx₁ ρ s n st₁) := by
  unfold bindInputs
  have hfold : ∀ (es : List DEdge), (∀ e ∈ es, e ∈ s.edges) → ∀ {a : List (Conn × ConnIn) × St}
      {b : List (Conn × ConnIn) × St}, St.SimR π P L a b →
      ResSim (St.SimR π P L)
        (es.foldlM (fun (acc, st) e =>
          match e.data with
          | .dep => pure (acc, st)
          | .memlet _ => do
            let (ci, st) ← bindIn ctx₂ ρ s e st
            match ci with
            | some c => pure (acc ++ [(e.dstConn.getD "", c)], st)
            | none => pure (acc, st)) a)
        (es.foldlM (fun (acc, st) e =>
          match e.data with
          | .dep => pure (acc, st)
          | .memlet _ => do
            let (ci, st) ← bindIn ctx₁ ρ s e st
            match ci with
            | some c => pure (acc ++ [(e.dstConn.getD "", c)], st)
            | none => pure (acc, st)) b) := by
    intro es hes
    induction es with
    | nil => intro a b hab; exact ResSim.ok hab
    | cons e es ih =>
      intro a b hab
      obtain ⟨acc₂, a₂⟩ := a
      obtain ⟨acc₁, a₁⟩ := b
      obtain ⟨hacc, ha⟩ := hab
      simp only at hacc ha
      subst hacc
      simp only [List.foldlM_cons]
      refine ResSim.bind (R := St.SimR π P L) ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
      split
      · exact ResSim.ok ⟨rfl, ha⟩
      · refine ResSim.bind (bindIn_sim hc ρ hS (hes e List.mem_cons_self) ha) fun x y hxy => ?_
        obtain ⟨c₂, a₂'⟩ := x
        obtain ⟨c₁, a₁'⟩ := y
        obtain ⟨hv, hs⟩ := hxy
        simp only at hv hs ⊢
        subst hv
        split
        · exact ResSim.ok ⟨rfl, hs⟩
        · exact ResSim.ok ⟨rfl, hs⟩
  refine ResSim.bind (hfold _ (fun e he => State.mem_inEdges he) ⟨rfl, h⟩) fun x y hxy => ?_
  obtain ⟨bs₂, a₂⟩ := x
  obtain ⟨bs₁, a₁⟩ := y
  obtain ⟨hv, hs⟩ := hxy
  simp only at hv hs ⊢
  subst hv
  exact ResSim.ok ⟨rfl, hs⟩

include hu

theorem storeScalar_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x)
    (w : Option WcrOp) (oob : Bool) (v : Val) :
    ResSim (St.Sim π P L) (storeScalar ctx₂ ρ st₂ d sub w oob v) (storeScalar ctx₁ ρ st₁ d sub w oob v) := by
  unfold storeScalar
  rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
  refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ?_)
  · exact ResSim.bindEq (streamPoint_sim hc ρ h.mem hd sub hs) fun p => St.push_sim hc ρ h hd.good₀ p [v]
  · refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hd sub hs) fun pts => ?_
    split
    · exact St.store_sim hc ρ h hu hd.good₀ _ oob w v
    · exact ResSim.err _
    · exact ResSim.err _

theorem applyScalar_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (c : Conn) (v : Val) :
    ResSim (St.Sim π P L) (applyScalar ctx₂ ρ s n st₂ c v) (applyScalar ctx₁ ρ s n st₁ c v) := by
  unfold applyScalar
  suffices ∀ es : List DEdge, (∀ e ∈ es, e ∈ s.edges) → ∀ {a b : St}, St.Sim π P L a b →
      ResSim (St.Sim π P L) (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet m =>
          match s.dstEndpoint e with
          | .code key => pure (st.setVal key v)
          | .data d sub => storeScalar ctx₂ ρ st d sub m.wcr m.allowOob v
          | _ => .err (.malformed s!"output connector '{c}' without a valid destination")) a)
        (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet m =>
          match s.dstEndpoint e with
          | .code key => pure (st.setVal key v)
          | .data d sub => storeScalar ctx₁ ρ st d sub m.wcr m.allowOob v
          | _ => .err (.malformed s!"output connector '{c}' without a valid destination")) b) from
    this _ (fun e he => State.edge_of_outEdgesOn he) h
  intro es hes
  induction es with
  | nil => intro a b hab; exact ResSim.ok hab
  | cons e es ih =>
    intro a b hab
    simp only [List.foldlM_cons]
    refine ResSim.bind ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
    split
    · exact ResSim.ok hab
    · split
      · exact ResSim.ok (hab.setVal _ v)
      · rename_i d sub hdst
        obtain ⟨hd, hs⟩ := hS.dst (hes e List.mem_cons_self) hdst
        exact storeScalar_sim hc ρ hu hab hd sub hs _ _ v
      · exact ResSim.err _

theorem storeArrayWrite_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x)
    (w : Option WcrOp) (oob : Bool) (rel : Idx) (v : Val) :
    ResSim (St.Sim π P L) (storeArrayWrite ctx₂ ρ st₂ d sub w oob rel v)
      (storeArrayWrite ctx₁ ρ st₁ d sub w oob rel v) := by
  unfold storeArrayWrite
  rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
  refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ResSim.err _) (fun _ => ?_)
  refine ResSim.bindEq (lowerCorner_sim hc ρ h.mem hd sub hs) fun lo => ?_
  by_cases hl : rel.length = lo.length
  · rw [if_pos hl, if_pos hl]
    refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hd sub hs) fun pts => ?_
    by_cases hc' : pts.contains (List.zipWith (· + ·) lo rel) = true
    · rw [if_pos hc', if_pos hc']
      exact St.store_sim hc ρ h hu hd.good₀ _ oob w v
    · rw [if_neg hc', if_neg hc']
      exact ResSim.err _
  · rw [if_neg hl, if_neg hl]
    exact ResSim.err _

theorem applyArrayWrite_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (c : Conn) (rel : Idx) (v : Val) :
    ResSim (St.Sim π P L) (applyArrayWrite ctx₂ ρ s n st₂ c rel v) (applyArrayWrite ctx₁ ρ s n st₁ c rel v) := by
  unfold applyArrayWrite
  suffices ∀ es : List DEdge, (∀ e ∈ es, e ∈ s.edges) → ∀ {a b : St}, St.Sim π P L a b →
      ResSim (St.Sim π P L) (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet m =>
          match s.dstEndpoint e with
          | .data d sub => storeArrayWrite ctx₂ ρ st d sub m.wcr m.allowOob rel v
          | _ => .err (.malformed s!"array write through connector '{c}' to a non-container")) a)
        (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet m =>
          match s.dstEndpoint e with
          | .data d sub => storeArrayWrite ctx₁ ρ st d sub m.wcr m.allowOob rel v
          | _ => .err (.malformed s!"array write through connector '{c}' to a non-container")) b) from
    this _ (fun e he => State.edge_of_outEdgesOn he) h
  intro es hes
  induction es with
  | nil => intro a b hab; exact ResSim.ok hab
  | cons e es ih =>
    intro a b hab
    simp only [List.foldlM_cons]
    refine ResSim.bind ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
    split
    · exact ResSim.ok hab
    · split
      · rename_i d sub hdst
        obtain ⟨hd, hs⟩ := hS.dst (hes e List.mem_cons_self) hdst
        exact storeArrayWrite_sim hc ρ hu hab hd sub hs _ _ rel v
      · exact ResSim.err _

omit hu in
theorem pushTo_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) (vs : List Val) :
    ResSim (St.Sim π P L) (pushTo ctx₂ ρ st₂ d sub vs) (pushTo ctx₁ ρ st₁ d sub vs) := by
  unfold pushTo
  rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
  refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ResSim.err _)
  exact ResSim.bindEq (streamPoint_sim hc ρ h.mem hd sub hs) fun p => St.push_sim hc ρ h hd.good₀ p vs

omit hu in
theorem applyPush_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (c : Conn) (vs : List Val) :
    ResSim (St.Sim π P L) (applyPush ctx₂ ρ s n st₂ c vs) (applyPush ctx₁ ρ s n st₁ c vs) := by
  unfold applyPush
  suffices ∀ es : List DEdge, (∀ e ∈ es, e ∈ s.edges) → ∀ {a b : St}, St.Sim π P L a b →
      ResSim (St.Sim π P L) (es.foldlM (fun st e =>
        match s.dstEndpoint e with
        | .data d sub => pushTo ctx₂ ρ st d sub vs
        | _ => .err (.malformed s!"push through connector '{c}' to a non-container")) a)
        (es.foldlM (fun st e =>
        match s.dstEndpoint e with
        | .data d sub => pushTo ctx₁ ρ st d sub vs
        | _ => .err (.malformed s!"push through connector '{c}' to a non-container")) b) from
    this _ (fun e he => State.edge_of_outEdgesOn he) h
  intro es hes
  induction es with
  | nil => intro a b hab; exact ResSim.ok hab
  | cons e es ih =>
    intro a b hab
    simp only [List.foldlM_cons]
    refine ResSim.bind ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
    split
    · rename_i d sub hdst
      obtain ⟨hd, hs⟩ := hS.dst (hes e List.mem_cons_self) hdst
      exact pushTo_sim hc ρ hab hd sub hs vs
    · exact ResSim.err _

omit hu in
theorem popFrom_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (sub : Option Subset) (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) (k : Nat) :
    ResSim (St.Sim π P L) (popFrom ctx₂ ρ st₂ d sub k) (popFrom ctx₁ ρ st₁ d sub k) := by
  unfold popFrom
  rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
  refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ResSim.err _)
  refine ResSim.bindEq (streamPoint_sim hc ρ h.mem hd sub hs) fun p => ?_
  exact ResSim.bind (St.popN_sim hc ρ h hd.good₀ p k) fun x y ⟨_, hs⟩ => ResSim.ok hs

omit hu in
theorem applyPop_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (c : Conn) (k : Nat) :
    ResSim (St.Sim π P L) (applyPop ctx₂ ρ s n st₂ c k) (applyPop ctx₁ ρ s n st₁ c k) := by
  unfold applyPop
  split
  · exact ResSim.err _
  · rename_i e hie
    split
    · rename_i d sub hsrc
      obtain ⟨hd, hs⟩ := hS.src (State.edge_of_inEdgeOn hie) hsrc
      exact popFrom_sim hc ρ h hd sub hs k
    · exact ResSim.err _

theorem applyOut_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (out : TaskletOut) :
    ResSim (St.Sim π P L) (applyOut ctx₂ ρ s n st₂ out) (applyOut ctx₁ ρ s n st₁ out) := by
  unfold applyOut
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π P L) (fun x a b hab => by
    obtain ⟨c, k⟩ := x; exact applyPop_sim hc ρ hS n hab c k) _ h) fun a b hab => ?_
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π P L) (fun x a b hab => by
    obtain ⟨c, v⟩ := x; exact applyScalar_sim hc ρ hu hS n hab c v) _ hab) fun a b hab => ?_
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π P L) (fun x a b hab => by
    obtain ⟨c, rel, v⟩ := x; exact applyArrayWrite_sim hc ρ hu hS n hab c rel v) _ hab) fun a b hab => ?_
  exact ResSim.foldlM (R := St.Sim π P L) (fun x a b hab => by
    obtain ⟨c, vs⟩ := x; exact applyPush_sim hc ρ hS n hab c vs) _ hab

theorem execCode_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) (run : (Conn → Option ConnIn) → Option TaskletOut) (ue : ErrReason) :
    ResSim (St.Sim π P L) (execCode ctx₂ ρ s n st₂ run ue) (execCode ctx₁ ρ s n st₁ run ue) := by
  unfold execCode
  split
  · exact ResSim.err _
  · refine ResSim.bind (bindInputs_sim hc ρ hS n h) fun x y hxy => ?_
    obtain ⟨f₂, a₂⟩ := x
    obtain ⟨f₁, a₁⟩ := y
    obtain ⟨hv, hs⟩ := hxy
    simp only at hv hs ⊢
    subst hv
    split
    · exact ResSim.err _
    · exact applyOut_sim hc ρ hu hS n hs _

end Leaf

/-- `Ctx.isViewDef` with the view test as a parameter. -/
def State.isViewDefWith (isView : Name → Bool) (s : State) (e : DEdge) : Bool :=
  let viewNode? (id : NodeId) : Option DNode :=
    (s.node? id).filter fun n => match n.kind with
      | .access d => isView d
      | _ => false
  let rs := s.pathStart e
  let rd := s.dstRootEdge e
  (match viewNode? rs.src with
   | some v => s.viewEdge? v = some (rs, false)
   | none => false) ||
  (match viewNode? rd.dst with
   | some v => s.viewEdge? v = some (rd, true)
   | none => false)

theorem State.isViewDefWith_congr {s : State} {f₂ f₁ : Name → Bool} (hf : ∀ d ∈ s.dataNames, f₂ d = f₁ d)
    {e : DEdge} : s.isViewDefWith f₂ e = s.isViewDefWith f₁ e := by
  have key : ∀ id, (s.node? id).filter (fun n => match n.kind with | .access d => f₂ d | _ => false) =
      (s.node? id).filter (fun n => match n.kind with | .access d => f₁ d | _ => false) := by
    intro id
    cases h : s.node? id with
    | none => rfl
    | some n =>
      cases hk : n.kind with
      | access d =>
        simp only [Option.filter_some, hk, hf d (State.accessData_names h hk)]
      | _ => simp only [Option.filter_some, hk]
  unfold State.isViewDefWith
  simp only [key]

namespace SNames

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} {s : State} (hS : SNames π P ctx₂ ctx₁ s)
include hS

theorem nodeFilter {p₂ p₁ : DNode → Bool} (hp : ∀ n ∈ s.nodes, p₂ n = p₁ n) (id : NodeId) :
    (s.node? id).filter p₂ = (s.node? id).filter p₁ := by
  cases h : s.node? id with
  | none => rfl
  | some n => simp only [Option.filter_some, hp n (State.node?_mem h)]

omit hS in
theorem isViewDef_eq (ctx : Ctx) (e : DEdge) :
    ctx.isViewDef s e = State.isViewDefWith ctx.isView s e := rfl

theorem viewOrder (ids : List NodeId) : s.viewOrder ctx₂ ids = s.viewOrder ctx₁ ids := by
  unfold State.viewOrder
  have : ids.filter (s.isViewNode ctx₂) = ids.filter (s.isViewNode ctx₁) := List.filter_congr fun id _ => by
    unfold State.isViewNode
    cases hn : s.node? id with
    | none => rfl
    | some n =>
      simp only
      cases hk : n.kind with
      | access d =>
        simp only [Ctx.isView,
          (hS.data d (State.mem_dataNames_node (State.node?_mem hn) (by simp [DNode.dataNames, hk]))).desc]
      | _ => rfl
  rw [this]

theorem isViewDef {e : DEdge} : ctx₂.isViewDef s e = ctx₁.isViewDef s e := by
  rw [isViewDef_eq, isViewDef_eq]
  refine State.isViewDefWith_congr fun d hd => ?_
  simp only [Ctx.isView, (hS.data d hd).desc]

theorem refSetTarget? {e : DEdge} : Dace.refSetTarget? ctx₂ s e = Dace.refSetTarget? ctx₁ s e := by
  unfold Dace.refSetTarget?
  simp only
  split
  · cases h : s.node? (s.dstRootEdge e).dst with
    | none => rfl
    | some n =>
      simp only [Option.bind_some]
      unfold DNode.accessData?
      split
      · rename_i d hk
        have := (State.accessData_names h hk)
        have hr : ctx₂.isReference d = ctx₁.isReference d := by
          simp only [Ctx.isReference, (hS.data d this).desc]
        simp only [Option.filter_some, hr]
      · rfl
  · rfl

end SNames

theorem refSetTarget?_names {ctx : Ctx} {s : State} {e : DEdge} {d : Name} (h : refSetTarget? ctx s e = some d) :
    d ∈ s.dataNames := by
  unfold refSetTarget? at h
  simp only at h
  split at h
  · cases hn : s.node? (s.dstRootEdge e).dst with
    | none => rw [hn] at h; cases h
    | some n =>
      rw [hn] at h
      simp only [Option.bind_some] at h
      unfold DNode.accessData? at h
      split at h
      · rename_i d' hk
        simp only [Option.filter_some] at h
        split at h
        · cases h; exact (State.accessData_names hn hk)
        · cases h
      · cases h
  · cases h

/-! ## Leaf functions (views, references, simple nodes, scopes) -/

section Leaf2

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation)
  (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
include hc

theorem reduceData_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) (props : ReduceProps)
    (hid : ∀ ide, props.identity = some ide → ∀ x ∈ ide.names, Ctx.Good₀ π P ctx₂ ctx₁ x)
    {sd : Name} (hsd : Ctx.Strong π P ctx₂ ctx₁ sd)
    (ssub : Option Subset) (hss : ∀ x ∈ optSubsetNames ssub, Ctx.Good₀ π P ctx₂ ctx₁ x)
    {dd : Name} (hdd : Ctx.Strong π P ctx₂ ctx₁ dd)
    (dsub : Option Subset) (hds : ∀ x ∈ optSubsetNames dsub, Ctx.Good₀ π P ctx₂ ctx₁ x)
    (outWcr : Option WcrOp) (soob doob : Bool) (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef) :
    ResSim (St.Sim π P L) (reduceData ctx₂ ρ st₂ props sd ssub dd dsub outWcr soob doob)
      (reduceData ctx₁ ρ st₁ props sd ssub dd dsub outWcr soob doob) := by
  unfold reduceData
  rw [show ctx₂.isStream sd = ctx₁.isStream sd by simp only [Ctx.isStream, hsd.desc],
    show ctx₂.isStream dd = ctx₁.isStream dd by simp only [Ctx.isStream, hdd.desc]]
  refine ResSim.ite (c := ctx₁.isStream sd || ctx₁.isStream dd) (fun _ => ResSim.err _) (fun _ => ?_)
  refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hsd ssub hss) fun spts => ?_
  refine ResSim.bindEq (Ctx.pointsOf_sim hc h.mem ρ hdd dsub hds) fun dpts => ?_
  simp only
  split
  · exact ResSim.err _
  · refine ResSim.bind (St.readMany_sim hc ρ h hsd.good₀ spts soob) fun x y hxy => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    obtain ⟨hv, hs⟩ := hxy
    simp only at hv hs ⊢
    subst hv
    rw [hc.I]
    refine ResSim.foldlM (fun gq b₂ b₁ hb => ?_) _ hs
    obtain ⟨g, q⟩ := gq
    simp only
    split
    · rename_i ide hide
      exact ResSim.bindEq (Ctx.evalR_sim hc hb.mem ρ _ (hid ide hide)) fun i =>
        St.store_sim hc ρ hb hu hdd.good₀ q doob outWcr _
    · split
      · exact ResSim.ok hb
      · exact St.wcr_sim hc ρ hb hu hdd.good₀ q doob _ _

theorem execReduce_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) (props : ReduceProps)
    (hid : ∀ ide, props.identity = some ide → ∀ x ∈ ide.names, Ctx.Good₀ π P ctx₂ ctx₁ x)
    {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) (hu : ∀ op v, ctx₁.I.wcr op .undef v = .undef) :
    ResSim (St.Sim π P L) (execReduce ctx₂ ρ s n props st₂) (execReduce ctx₁ ρ s n props st₁) := by
  unfold execReduce
  dsimp only
  split
  · rename_i ie oe hins houts
    have hie : ie ∈ s.edges := State.mem_inEdges (List.mem_filter.1 (hins ▸ List.mem_singleton_self ie)).1
    have hoe : oe ∈ s.edges := State.mem_outEdges (List.mem_filter.1 (houts ▸ List.mem_singleton_self oe)).1
    split
    · rename_i sd ssub dd dsub om hsrc hdst _
      obtain ⟨hsd, hss⟩ := hS.src hie hsrc
      obtain ⟨hdd, hds⟩ := hS.dst hoe hdst
      exact reduceData_sim hc ρ h props hid hsd ssub hss hdd dsub hds _ _ _ hu
    · exact ResSim.err _
  · exact ResSim.err _

theorem aliasTo_sim {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {d base : Name} (hd : Ctx.Strong π P ctx₂ ctx₁ d)
    (hb : Ctx.Strong π P ctx₂ ctx₁ base) (sub : Option Subset)
    (hs : ∀ x ∈ optSubsetNames sub, Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim (St.Sim π P L) (aliasTo ctx₂ ρ st₂ d base sub) (aliasTo ctx₁ ρ st₁ d base sub) := by
  unfold aliasTo
  rw [show ctx₂.reinterprets d base = ctx₁.reinterprets d base by
    simp only [Ctx.reinterprets, hd.desc, hb.desc]]
  refine ResSim.ite (c := ctx₁.reinterprets d base) (fun _ => ?_) (fun _ => ?_)
  · simp only [hd.desc, hb.desc]; exact ResSim.err _
  refine ResSim.bindEq (lowerCorner_sim hc ρ h.mem hb sub hs) fun lo => ?_
  rw [Ctx.addr_sim hc h.mem ρ hb.good₀]
  cases ha : ctx₁.addr ρ st₁.mem base lo with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq, hd.cname]
    exact ResSim.ok (h.bindAlias (ctx₁.cname d) { base := l.name, offset := l.offset } hd.memP
      (Ctx.addrP hc h.mem ρ hb.good₀ ha))

theorem bindView_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {v : DNode} (hv : v ∈ s.nodes) {d : Name}
    (hk : v.kind = .access d) {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (bindView ctx₂ ρ s v d st₂) (bindView ctx₁ ρ s v d st₁) := by
  unfold bindView
  split
  · exact ResSim.err _
  · rename_i e isIn hve
    have he := State.viewEdge?_mem hve
    have hd : Ctx.Strong π P ctx₂ ctx₁ d := hS.data d (State.mem_dataNames_node hv (by simp [DNode.dataNames, hk]))
    simp only
    split
    · exact ResSim.err _
    · rename_i base hbase
      have hb : Ctx.Strong π P ctx₂ ctx₁ base := by
        split at hbase
        · obtain ⟨n, hn, hd'⟩ := Option.bind_eq_some_iff.1 hbase
          unfold DNode.accessData? at hd'
          split at hd'
          · rename_i b hkb; cases hd'; exact hS.data _ (State.accessData_names hn hkb)
          · cases hd'
        · obtain ⟨n, hn, hd'⟩ := Option.bind_eq_some_iff.1 hbase
          unfold DNode.accessData? at hd'
          split at hd'
          · rename_i b hkb; cases hd'; exact hS.data _ (State.accessData_names hn hkb)
          · cases hd'
      refine aliasTo_sim hc ρ h hd hb _ fun x hx => hS.expr x ?_
      split at hx
      · exact State.leafSrcSubset_names he x hx
      · exact State.leafDstSubset_names he x hx

theorem bindViews_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (ids : List NodeId) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (bindViews ctx₂ ρ s ids st₂) (bindViews ctx₁ ρ s ids st₁) := by
  unfold bindViews
  refine ResSim.foldlM (fun id a b hab => ?_) _ h
  split
  · rename_i n hn
    split
    · rename_i d hk
      have hd := hS.data d (State.accessData_names hn hk)
      rw [show ctx₂.isView d = ctx₁.isView d by simp only [Ctx.isView, hd.desc]]
      exact ResSim.ite (c := ctx₁.isView d) (fun _ => bindView_sim hc ρ hS (State.node?_mem hn) hk hab)
        (fun _ => ResSim.ok hab)
    · exact ResSim.ok hab
  · exact ResSim.ok hab

theorem refSet_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {e : DEdge} (he : e ∈ s.edges) {d : Name}
    (hd : Ctx.Strong π P ctx₂ ctx₁ d) {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (refSet ctx₂ ρ s e d st₂) (refSet ctx₁ ρ s e d st₁) := by
  unfold refSet
  split
  · rename_i base sub hsrc
    obtain ⟨hb, hs⟩ := hS.src he hsrc
    exact aliasTo_sim hc ρ h hd hb sub hs
  · exact ResSim.err _
  · exact ResSim.err _

include hu in
theorem execCopiesInto_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (execCopiesInto ctx₂ ρ s n st₂) (execCopiesInto ctx₁ ρ s n st₁) := by
  unfold execCopiesInto
  suffices ∀ es : List DEdge, (∀ e ∈ es, e ∈ s.edges) → ∀ {a b : St}, St.Sim π P L a b →
      ResSim (St.Sim π P L) (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet _ =>
          if ctx₂.isViewDef s e || s.isRouting e then pure st
          else
            match refSetTarget? ctx₂ s e with
            | some d => refSet ctx₂ ρ s e d st
            | none =>
              match s.srcEndpoint e with
              | .data _ _ => execCopy ctx₂ ρ s e st
              | _ => pure st) a)
        (es.foldlM (fun st e =>
        match e.data with
        | .dep => pure st
        | .memlet _ =>
          if ctx₁.isViewDef s e || s.isRouting e then pure st
          else
            match refSetTarget? ctx₁ s e with
            | some d => refSet ctx₁ ρ s e d st
            | none =>
              match s.srcEndpoint e with
              | .data _ _ => execCopy ctx₁ ρ s e st
              | _ => pure st) b) from
    this _ (fun e he => State.mem_inEdges he) h
  intro es hes
  induction es with
  | nil => intro a b hab; exact ResSim.ok hab
  | cons e es ih =>
    intro a b hab
    simp only [List.foldlM_cons]
    refine ResSim.bind ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
    have he := hes e List.mem_cons_self
    split
    · exact ResSim.ok hab
    · rw [hS.isViewDef, hS.refSetTarget?]
      refine ResSim.ite (c := ctx₁.isViewDef s e || s.isRouting e) (fun _ => ResSim.ok hab) (fun _ => ?_)
      split
      · rename_i d hd
        exact refSet_sim hc ρ hS he (hS.data d (refSetTarget?_names hd)) hab
      · split
        · exact execCopy_sim hc ρ hu hS he hab
        · exact ResSim.ok hab

include hu in
theorem execSimple_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.Sim π P L) (execSimple ctx₂ ρ s n st₂) (execSimple ctx₁ ρ s n st₁) := by
  unfold execSimple
  split
  · exact execCopiesInto_sim hc ρ hu hS n h
  · exact execCopiesInto_sim hc ρ hu hS n h
  · exact execCopiesInto_sim hc ρ hu hS n h
  · rw [hc.I]; exact execCode_sim hc ρ hu hS n h _ _
  · rename_i l hk
    split
    · rename_i props hr
      refine execReduce_sim hc ρ hS n props (fun ide hide x hx => hS.node_expr hn ?_) h hu
      simp [DNode.exprNames, hk, hr, hide, hx]
    · rw [hc.I]; exact execCode_sim hc ρ hu hS n h _ _
  · exact ResSim.err _
  · exact ResSim.err _
  · exact ResSim.err _

theorem bindDynamicInputs_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (St.SimR π P L) (bindDynamicInputs ctx₂ ρ s n st₂) (bindDynamicInputs ctx₁ ρ s n st₁) := by
  unfold bindDynamicInputs
  suffices ∀ es : List DEdge, (∀ e ∈ es, e ∈ s.edges) → ∀ {a b : Valuation × St}, St.SimR π P L a b →
      ResSim (St.SimR π P L) (es.foldlM (fun (ρ, st) e =>
        if s.isDynamicInput e then
          match e.data, e.dstConn with
          | .memlet _, some c => do
            let (v?, st) ← readEdgeScalar ctx₂ ρ s e st
            match v? with
            | some v => pure (Valuation.set ρ c v, st)
            | none => pure (ρ, st)
          | _, _ => pure (ρ, st)
        else pure (ρ, st)) a)
        (es.foldlM (fun (ρ, st) e =>
        if s.isDynamicInput e then
          match e.data, e.dstConn with
          | .memlet _, some c => do
            let (v?, st) ← readEdgeScalar ctx₁ ρ s e st
            match v? with
            | some v => pure (Valuation.set ρ c v, st)
            | none => pure (ρ, st)
          | _, _ => pure (ρ, st)
        else pure (ρ, st)) b) from
    this _ (fun e he => State.mem_inEdges he) ⟨rfl, h⟩
  intro es hes
  induction es with
  | nil => intro a b hab; exact ResSim.ok hab
  | cons e es ih =>
    intro a b hab
    obtain ⟨ρ₂, a₂⟩ := a
    obtain ⟨ρ₁, a₁⟩ := b
    obtain ⟨hρ, ha⟩ := hab
    simp only at hρ ha
    subst hρ
    simp only [List.foldlM_cons]
    refine ResSim.bind ?_ fun x y hxy => ih (fun e' he' => hes e' (List.mem_cons_of_mem _ he')) hxy
    split
    · split
      · refine ResSim.bind (readEdgeScalar_sim hc ρ₂ hS (hes e List.mem_cons_self) ha) fun x y hxy => ?_
        obtain ⟨v₂, b₂⟩ := x
        obtain ⟨v₁, b₁⟩ := y
        obtain ⟨hv, hs⟩ := hxy
        simp only at hv hs ⊢
        subst hv
        split
        · exact ResSim.ok ⟨rfl, hs⟩
        · exact ResSim.ok ⟨rfl, hs⟩
      · exact ResSim.ok ⟨rfl, ha⟩
    · exact ResSim.ok ⟨rfl, ha⟩

/-- The result of a scope set-up: the same valuation and points, simulating
states. -/
def SetupSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : Valuation × St × List Idx) : Prop :=
  x₂.1 = x₁.1 ∧ St.Sim π P L x₂.2.1 x₁.2.1 ∧ x₂.2.2 = x₁.2.2

theorem mapSetup_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {mp : MapNode}
    (hk : n.kind = .mapEntry mp) {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (SetupSim π P L) (mapSetup ctx₂ ρ s n mp st₂) (mapSetup ctx₁ ρ s n mp st₁) := by
  unfold mapSetup
  refine ResSim.bind (bindDynamicInputs_sim hc ρ hS n h) fun x y hxy => ?_
  obtain ⟨ρ₂, a₂⟩ := x
  obtain ⟨ρ₁, a₁⟩ := y
  obtain ⟨hρ, ha⟩ := hxy
  simp only at hρ ha ⊢
  subst hρ
  exact ResSim.bindEq (Ctx.rangePoints_sim hc ha.mem ρ₂ mp.range fun x hx =>
    hS.node_expr hn (by simp [DNode.exprNames, hk, hx])) fun pts => ResSim.ok ⟨rfl, ha, rfl⟩

omit hc in
theorem scopeFinish_sim {ctx₂ ctx₁ : Ctx} (hI : ctx₂.I = ctx₁.I) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {pre₂ pre₁ : St} (h : St.Sim π P L pre₂ pre₁) (logs : List Log) (hl : ∀ g ∈ logs, Log.NamesP P g) :
    ResSim (St.Sim π P L) (scopeFinish ctx₂ pre₂ (logs.map (Log.ren π))) (scopeFinish ctx₁ pre₁ logs) := by
  unfold scopeFinish
  rw [hI]
  rcases merge_sim hw h.mem logs hl with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨ha, ?_, h.vals, h.logP.seq (Log.NamesP.unionAll hl)⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]

omit hc in
theorem stateFinish_sim {ctx₂ ctx₁ : Ctx} (hI : ctx₂.I = ctx₁.I) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {c₂ c₁ : CSt} (h : CSt.Sim π P L c₂ c₁) (logs : List Log) (hl : ∀ g ∈ logs, Log.NamesP P g) :
    ResSim (fun x₂ x₁ => x₂.1 = x₁.1 ∧ CSt.Sim π P L x₂.2 x₁.2) (stateFinish ctx₂ c₂ (logs.map (Log.ren π)))
      (stateFinish ctx₁ c₁ logs) := by
  unfold stateFinish
  rw [hI]
  rcases merge_sim hw h.mem logs hl with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨rfl, h.ρ, ha, ?_, h.logP.seq (Log.NamesP.unionAll hl)⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]

end Leaf2

/-! ## Allocation, pointwise -/

theorem Mem.allocAll_cells : ∀ (ns : List Name) (m : Mem) (l : Loc),
    (m.allocAll ns).cells l = if l.name ∈ ns then .undef else m.cells l
  | [], m, l => by simp [Mem.allocAll]
  | n :: ns, m, l => by
    have e : m.allocAll (n :: ns) = (m.alloc n).allocAll ns := rfl
    rw [e, Mem.allocAll_cells ns]
    by_cases h₁ : l.name ∈ ns <;> by_cases h₂ : l.name = n <;> simp [Mem.alloc, h₁, h₂]

theorem Mem.allocAll_queues : ∀ (ns : List Name) (m : Mem) (l : Loc),
    (m.allocAll ns).queues l = if l.name ∈ ns then [] else m.queues l
  | [], m, l => by simp [Mem.allocAll]
  | n :: ns, m, l => by
    have e : m.allocAll (n :: ns) = (m.alloc n).allocAll ns := rfl
    rw [e, Mem.allocAll_queues ns]
    by_cases h₁ : l.name ∈ ns <;> by_cases h₂ : l.name = n <;> simp [Mem.alloc, h₁, h₂]

theorem Ctx.allocate_cells (ctx : Ctx) : ∀ (ts : List Name) (m : Mem) (l : Loc),
    (ctx.allocate ts m).cells l = if (∃ t ∈ ts, ctx.cname t = l.name) then .undef else m.cells l
  | [], m, l => by simp [Ctx.allocate]
  | t :: ts, m, l => by
    have e : ctx.allocate (t :: ts) m =
        ctx.allocate ts (if ctx.isStream t then m.alloc (ctx.cname t) else m.allocCells (ctx.cname t)) := rfl
    rw [e, Ctx.allocate_cells ctx ts]
    by_cases h₁ : ∃ t ∈ ts, ctx.cname t = l.name
    · have : ∃ t' ∈ t :: ts, ctx.cname t' = l.name := by
        obtain ⟨t', h, h'⟩ := h₁; exact ⟨t', List.mem_cons_of_mem _ h, h'⟩
      simp only [h₁, this, ite_true]
    · by_cases h₂ : ctx.cname t = l.name
      · have : ∃ t' ∈ t :: ts, ctx.cname t' = l.name := ⟨t, List.mem_cons_self, h₂⟩
        simp only [h₁, this, ite_true, ite_false]
        split <;> simp [Mem.alloc, Mem.allocCells, h₂]
      · have : ¬ ∃ t' ∈ t :: ts, ctx.cname t' = l.name := by
          rintro ⟨t', h, h'⟩
          rcases List.mem_cons.1 h with rfl | h
          · exact h₂ h'
          · exact h₁ ⟨t', h, h'⟩
        simp only [h₁, this, ite_false]
        split <;> simp [Mem.alloc, Mem.allocCells, Ne.symm h₂]

theorem Ctx.allocate_queues (ctx : Ctx) : ∀ (ts : List Name) (m : Mem) (l : Loc),
    (ctx.allocate ts m).queues l =
      if (∃ t ∈ ts, ctx.isStream t = true ∧ ctx.cname t = l.name) then [] else m.queues l
  | [], m, l => by simp [Ctx.allocate]
  | t :: ts, m, l => by
    have e : ctx.allocate (t :: ts) m =
        ctx.allocate ts (if ctx.isStream t then m.alloc (ctx.cname t) else m.allocCells (ctx.cname t)) := rfl
    rw [e, Ctx.allocate_queues ctx ts]
    by_cases h₁ : ∃ t ∈ ts, ctx.isStream t = true ∧ ctx.cname t = l.name
    · have : ∃ t' ∈ t :: ts, ctx.isStream t' = true ∧ ctx.cname t' = l.name := by
        obtain ⟨t', h, h'⟩ := h₁; exact ⟨t', List.mem_cons_of_mem _ h, h'⟩
      simp only [h₁, this, ite_true]
    · by_cases h₂ : ctx.isStream t = true ∧ ctx.cname t = l.name
      · have : ∃ t' ∈ t :: ts, ctx.isStream t' = true ∧ ctx.cname t' = l.name := ⟨t, List.mem_cons_self, h₂⟩
        simp only [h₁, this, ite_true, ite_false, h₂.1]
        simp [Mem.alloc, h₂.2]
      · have : ¬ ∃ t' ∈ t :: ts, ctx.isStream t' = true ∧ ctx.cname t' = l.name := by
          rintro ⟨t', h, h'⟩
          rcases List.mem_cons.1 h with rfl | h
          · exact h₂ h'
          · exact h₁ ⟨t', h, h'⟩
        simp only [h₁, this, ite_false]
        split
        · rename_i hs
          have : ¬ ctx.cname t = l.name := fun h => h₂ ⟨hs, h⟩
          simp [Mem.alloc, Ne.symm this]
        · simp [Mem.allocCells]

/-! ## Private and allocated transients -/

theorem Ctx.mem_containerNames {ctx : Ctx} {t : Name} : t ∈ ctx.containerNames ↔ (ctx.desc? t).isSome := by
  unfold Ctx.containerNames Ctx.desc? SDFG.array?
  rw [List.mem_eraseDups]
  simp only [List.mem_map, Option.isSome_map]
  constructor
  · rintro ⟨⟨t', d⟩, hm, rfl⟩
    exact List.find?_isSome.2 ⟨_, hm, by simp⟩
  · intro h
    obtain ⟨x, hx, hp⟩ := List.find?_isSome.1 h
    simp only [decide_eq_true_eq] at hp
    exact ⟨x, hx, hp⟩

section Transients

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁)
  {s : State} (hS : SNames π P ctx₂ ctx₁ s)
include hS

theorem mentioned_good {t : Name} (h : 0 < (s.nodes.filter (·.mentions t)).length) :
    Ctx.Good₀ π P ctx₂ ctx₁ t := by
  obtain ⟨n, hn⟩ := List.length_pos_iff_exists_mem.1 h
  obtain ⟨hn, hm⟩ := List.mem_filter.1 hn
  rcases State.mentions_names hn hm with h | h
  · exact (hS.data t h).good₀
  · exact hS.expr t h

include hc

theorem privateTransients_mem (entry : NodeId) (x : Name) :
    x ∈ ctx₂.privateTransients s entry ↔ x ∈ (ctx₁.privateTransients s entry).map π := by
  unfold Ctx.privateTransients
  simp only [List.mem_map, List.mem_filterMap]
  have hcount : ∀ t, ctx₂.g.countMentions t = ctx₁.g.countMentions t := by
    intro t; simp only [SDFG.countMentions, hc.top]
  have hpos : ∀ t, 0 < (s.nodes.filter fun n => n.mentions t && (s.scopeNodes entry).contains n.id).length →
      0 < (s.nodes.filter (·.mentions t)).length := by
    intro t hh
    obtain ⟨n, hn⟩ := List.length_pos_iff_exists_mem.1 hh
    obtain ⟨hn, hm⟩ := List.mem_filter.1 hn
    simp only [Bool.and_eq_true] at hm
    exact List.length_pos_iff_exists_mem.2 ⟨n, List.mem_filter.2 ⟨hn, hm.1⟩⟩
  constructor
  · rintro ⟨t, ht, hx⟩
    cases hd : ctx₂.desc? t with
    | none => rw [hd] at hx; cases hx
    | some d =>
      rw [hd] at hx
      simp only at hx
      split at hx
      · rename_i hsc
        split at hx
        · rename_i hh
          simp only [Option.some.injEq] at hx
          subst hx
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
          have hg := mentioned_good hS (hpos t hh.1)
          have hd₁ : ctx₁.desc? t = some d := by rw [← hg.1, hd]
          obtain ⟨hcn, -⟩ := hg.2 (by simp [hd₁])
          refine ⟨ctx₁.cname t, ⟨t, Ctx.mem_containerNames.2 (by simp [hd₁]), ?_⟩, hcn.symm⟩
          simp only [hd₁, hsc, ite_true]
          rw [if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hh.1, hh.2.trans (hcount t)⟩)]
        · cases hx
      · cases hx
  · rintro ⟨y, ⟨t, ht, hy⟩, rfl⟩
    cases hd : ctx₁.desc? t with
    | none => rw [hd] at hy; cases hy
    | some d =>
      rw [hd] at hy
      simp only at hy
      split at hy
      · rename_i hsc
        split at hy
        · rename_i hh
          simp only [Option.some.injEq] at hy
          subst hy
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
          have hg := mentioned_good hS (hpos t hh.1)
          have hd₂ : ctx₂.desc? t = some d := by rw [hg.1, hd]
          obtain ⟨hcn, -⟩ := hg.2 (by simp [hd])
          refine ⟨t, Ctx.mem_containerNames.2 (by simp [hd₂]), ?_⟩
          simp only [hd₂, hsc, ite_true]
          rw [if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hh.1, hh.2.trans (hcount t).symm⟩), hcn]
        · cases hy
      · cases hy

omit hc in
theorem stateTransients_dir {c₂ c₁ : Ctx} (hcnt : ∀ t, c₂.g.countMentions t = c₁.g.countMentions t)
    (hdesc : ∀ t, 0 < (s.nodes.filter (·.mentions t)).length → c₂.desc? t = c₁.desc? t) {x : Name}
    (hx : x ∈ c₂.stateTransients s) : x ∈ c₁.stateTransients s := by
  unfold Ctx.stateTransients at hx ⊢
  obtain ⟨t, ht, hx⟩ := List.mem_filterMap.1 hx
  refine List.mem_filterMap.2 ⟨t, ?_, ?_⟩
  · cases hd : c₂.desc? t with
    | none => rw [hd] at hx; cases hx
    | some d =>
      rw [hd] at hx
      simp only at hx
      split at hx
      · split at hx
        · rename_i hh
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
          exact Ctx.mem_containerNames.2 (by rw [← hdesc t hh.1, hd]; rfl)
        · cases hx
      · cases hx
  · cases hd : c₂.desc? t with
    | none => rw [hd] at hx; cases hx
    | some d =>
      rw [hd] at hx
      simp only at hx
      split at hx
      · rename_i hsc
        split at hx
        · rename_i hh
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
          have hd₁ : c₁.desc? t = some d := by rw [← hdesc t hh.1, hd]
          simp only [hd₁, hsc, ite_true]
          rw [if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hh.1, hh.2.trans (hcnt t)⟩)]
          exact hx
        · cases hx
      · cases hx

theorem stateTransients_mem (x : Name) : x ∈ ctx₂.stateTransients s ↔ x ∈ ctx₁.stateTransients s := by
  have hcount : ∀ t, ctx₂.g.countMentions t = ctx₁.g.countMentions t := by
    intro t; simp only [SDFG.countMentions, hc.top]
  exact ⟨stateTransients_dir hS hcount (fun t h => (mentioned_good hS h).1),
    stateTransients_dir hS (fun t => (hcount t).symm) (fun t h => (mentioned_good hS h).1.symm)⟩

omit hc in
theorem stateTransients_good {t : Name} (ht : t ∈ ctx₁.stateTransients s) :
    Ctx.Good₀ π P ctx₂ ctx₁ t ∧ (ctx₁.desc? t).isSome := by
  unfold Ctx.stateTransients at ht
  obtain ⟨t', ht', hx⟩ := List.mem_filterMap.1 ht
  cases hd : ctx₁.desc? t' with
  | none => rw [hd] at hx; cases hx
  | some d =>
    rw [hd] at hx
    simp only at hx
    split at hx
    · split at hx
      · rename_i hh
        simp only [Option.some.injEq] at hx
        subst hx
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
        exact ⟨mentioned_good hS hh.1, by simp [hd]⟩
      · cases hx
    · cases hx

end Transients

/-! ### Allocation of simulating memories -/

theorem Mem.Sim.allocAll_congr {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {m₂ m₁ : Mem} (h : Mem.Sim π P L m₂ m₁)
    {ns₂ ns₁ : List Name} (hn : ∀ x, x ∈ ns₂ ↔ x ∈ ns₁.map π) :
    Mem.Sim π P L (m₂.allocAll ns₂) (m₁.allocAll ns₁) := by
  have hc : ∀ l : Loc, (l.ren π).name ∈ ns₂ ↔ l.name ∈ ns₁ := by
    intro l
    rw [Loc.ren_name, hn]
    simp only [List.mem_map, π.injective.eq_iff, exists_eq_right]
  refine ⟨fun n hn' => ?_, fun n b hn' hb => ?_, fun l hl => ?_, fun l hl => ?_⟩
  · rw [Mem.allocAll_aliases, Mem.allocAll_aliases]; exact h.aliases n hn'
  · rw [Mem.allocAll_aliases] at hb; exact h.aliasP n b hn' hb
  · rw [Mem.allocAll_cells, Mem.allocAll_cells]
    by_cases h₁ : l.name ∈ ns₁
    · rw [if_pos ((hc l).2 h₁), if_pos h₁]; exact .inr rfl
    · rw [if_neg (fun h' => h₁ ((hc l).1 h')), if_neg h₁]; exact h.cells l hl
  · rw [Mem.allocAll_queues, Mem.allocAll_queues]
    by_cases h₁ : l.name ∈ ns₁
    · rw [if_pos ((hc l).2 h₁), if_pos h₁]
    · rw [if_neg (fun h' => h₁ ((hc l).1 h')), if_neg h₁]; exact h.queues l hl

theorem Log.dropNames_congr {g : Log} {ns₂ ns₁ : List Name} (hn : ∀ x, x ∈ ns₂ ↔ x ∈ ns₁) :
    g.dropNames ns₂ = g.dropNames ns₁ := by
  have : ∀ x, ns₂.contains x = ns₁.contains x := fun x => by
    cases h : ns₁.contains x
    · have h' : x ∉ ns₁ := by simpa using h
      simpa [hn] using h'
    · have h' : x ∈ ns₁ := by simpa using h
      simpa [hn] using h'
  simp only [Log.dropNames, this]

theorem iterStart_sim {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {pre₂ pre₁ : St} (h : St.Sim π P L pre₂ pre₁)
    {priv₂ priv₁ : List Name} (hp : ∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) :
    St.Sim π P L (iterStart priv₂ pre₂) (iterStart priv₁ pre₁) :=
  ⟨h.mem.allocAll_congr hp, rfl, h.vals, Log.NamesP.empty P⟩

theorem allocate_sim {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} {m₂ m₁ : Mem}
    (hm : Mem.Sim π P L m₂ m₁) {ts₂ ts₁ : List Name} (hts : ∀ x, x ∈ ts₂ ↔ x ∈ ts₁)
    (hg : ∀ t ∈ ts₁, ctx₂.isStream t = ctx₁.isStream t ∧ ctx₂.cname t = π (ctx₁.cname t)) :
    Mem.Sim π P L (ctx₂.allocate ts₂ m₂) (ctx₁.allocate ts₁ m₁) := by
  have hcell : ∀ l : Loc, (∃ t ∈ ts₂, ctx₂.cname t = (l.ren π).name) ↔ (∃ t ∈ ts₁, ctx₁.cname t = l.name) := by
    intro l
    simp only [Loc.ren_name]
    constructor
    · rintro ⟨t, ht, he⟩
      have ht₁ := (hts t).1 ht
      exact ⟨t, ht₁, π.injective ((hg t ht₁).2.symm.trans he)⟩
    · rintro ⟨t, ht, he⟩
      exact ⟨t, (hts t).2 ht, by rw [(hg t ht).2, he]⟩
  have hqueue : ∀ l : Loc, (∃ t ∈ ts₂, ctx₂.isStream t = true ∧ ctx₂.cname t = (l.ren π).name) ↔
      (∃ t ∈ ts₁, ctx₁.isStream t = true ∧ ctx₁.cname t = l.name) := by
    intro l
    simp only [Loc.ren_name]
    constructor
    · rintro ⟨t, ht, hs, he⟩
      have ht₁ := (hts t).1 ht
      exact ⟨t, ht₁, (hg t ht₁).1 ▸ hs, π.injective ((hg t ht₁).2.symm.trans he)⟩
    · rintro ⟨t, ht, hs, he⟩
      exact ⟨t, (hts t).2 ht, (hg t ht).1.symm ▸ hs, by rw [(hg t ht).2, he]⟩
  refine ⟨fun n hn => ?_, fun n b hn hb => ?_, fun l hl => ?_, fun l hl => ?_⟩
  · rw [Ctx.allocate_aliases, Ctx.allocate_aliases]; exact hm.aliases n hn
  · rw [Ctx.allocate_aliases] at hb; exact hm.aliasP n b hn hb
  · rw [Ctx.allocate_cells, Ctx.allocate_cells]
    by_cases h₁ : ∃ t ∈ ts₁, ctx₁.cname t = l.name
    · rw [if_pos ((hcell l).2 h₁), if_pos h₁]; exact .inr rfl
    · rw [if_neg (fun h' => h₁ ((hcell l).1 h')), if_neg h₁]; exact hm.cells l hl
  · rw [Ctx.allocate_queues, Ctx.allocate_queues]
    by_cases h₁ : ∃ t ∈ ts₁, ctx₁.isStream t = true ∧ ctx₁.cname t = l.name
    · rw [if_pos ((hqueue l).2 h₁), if_pos h₁]
    · rw [if_neg (fun h' => h₁ ((hqueue l).1 h')), if_neg h₁]; exact hm.queues l hl

/-! ### Traces -/

/-- Rename the deltas of a trace. -/
def renTrace (π : Equiv.Perm Name) (tr : List (NodeId × Log)) : List (NodeId × Log) :=
  tr.map fun (id, g) => (id, g.ren π)

theorem State.hbFree_ren (s : State) (π : Equiv.Perm Name) :
    ∀ tr : List (NodeId × Log), s.hbFree (renTrace π tr) = s.hbFree tr
  | [] => rfl
  | x :: xs => by
    simp only [renTrace, List.map_cons, State.hbFree, List.all_map, Function.comp_def, State.hbConflict,
      Log.conflicts_ren]
    rw [← State.hbFree_ren s π xs]
    rfl

/-! ### Consume scopes and nested SDFGs -/

theorem Log.ren_stripQueue (π : Equiv.Perm Name) (g : Log) (l : Loc) :
    (g.stripQueue l).ren π = (g.ren π).stripQueue (l.ren π) := by
  simp only [Log.stripQueue, Log.ren]
  congr 1
  refine List.map_filter_comm _ _ _ (fun x => ?_) _
  simp only [ne_eq, decide_not, Loc.ren_inj]

theorem Log.pushedTo_ren (π : Equiv.Perm Name) (g : Log) (l : Loc) : (g.ren π).pushedTo (l.ren π) = g.pushedTo l := by
  simp only [Log.pushedTo, Log.ren, List.flatMap_map, Loc.ren_inj]

/-- Rename the memory names of connector bindings. -/
def renConns (π : Equiv.Perm Name) (conns : List (Name × Name)) : List (Name × Name) :=
  conns.map fun (c, l) => (c, π l)

section Leaf3

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation)
include hc

theorem consumeStream_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {m₂ m₁ : Mem}
    (hm : Mem.Sim π P L m₂ m₁) :
    ResSim (fun l₂ l₁ => l₂ = Loc.ren π l₁ ∧ P l₁.name) (consumeStream ctx₂ ρ s n m₂) (consumeStream ctx₁ ρ s n m₁) := by
  unfold consumeStream
  split
  · exact ResSim.err _
  · rename_i e he
    have hmem : e ∈ s.edges := State.mem_inEdges (List.mem_of_find?_eq_some he)
    split
    · rename_i d sub hsrc
      obtain ⟨hd, hs⟩ := hS.src hmem hsrc
      rw [show ctx₂.isStream d = ctx₁.isStream d by simp only [Ctx.isStream, hd.desc]]
      refine ResSim.ite (c := ctx₁.isStream d) (fun _ => ?_) (fun _ => ResSim.err _)
      refine ResSim.bindEq (streamPoint_sim hc ρ hm hd sub hs) fun p => ?_
      rw [Ctx.loc_sim hc hm ρ hd.good₀]
      cases hl : ctx₁.loc ρ m₁ d p false with
      | err e => exact ResSim.err e
      | ok l =>
        have hP := Ctx.locP hc hm ρ hd.good₀ hl
        simp only [Res.map, Res.ok_bind, Res.pure_eq, hm.resolve hP]
        exact ResSim.ok ⟨rfl, hm.resolveP hP⟩
    · exact ResSim.err _

/-- The result of a consume set-up. -/
def ConsumeSetupSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : Valuation × St × Loc × Int) : Prop :=
  x₂.1 = x₁.1 ∧ St.Sim π P L x₂.2.1 x₁.2.1 ∧ x₂.2.2.1 = x₁.2.2.1.ren π ∧ P x₁.2.2.1.name ∧ x₂.2.2.2 = x₁.2.2.2

theorem consumeSetup_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {c : ConsumeNode}
    (hk : n.kind = .consumeEntry c) {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (ConsumeSetupSim π P L) (consumeSetup ctx₂ ρ s n c st₂) (consumeSetup ctx₁ ρ s n c st₁) := by
  unfold consumeSetup
  refine ResSim.bind (bindDynamicInputs_sim hc ρ hS n h) fun x y hxy => ?_
  obtain ⟨ρ₂, a₂⟩ := x
  obtain ⟨ρ₁, a₁⟩ := y
  obtain ⟨hρ, ha⟩ := hxy
  simp only at hρ ha ⊢
  subst hρ
  refine ResSim.bind (consumeStream_sim hc ρ₂ hS n ha.mem) fun l₂ l₁ ⟨hl, hP⟩ => ?_
  exact ResSim.bindEq (Ctx.evalIntR_sim hc ha.mem ρ₂ _ fun x hx =>
    hS.node_expr hn (by simp [DNode.exprNames, hk, hx])) fun np => ResSim.ok ⟨rfl, ha, hl, hP, rfl⟩

theorem consumeCond_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {c : ConsumeNode}
    (hk : n.kind = .consumeEntry c) {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) (l : Loc) (q : List Val) :
    ResSim Eq (consumeCond ctx₂ ρ c m₂ (l.ren π) q) (consumeCond ctx₁ ρ c m₁ l q) := by
  unfold consumeCond
  split
  · exact ResSim.ok rfl
  · rename_i e he
    exact ResSim.map _ _ (Ctx.evalBoolR_sim hc (hm.setQueue l q) ρ _ fun x hx =>
      hS.node_expr hn (by simp [DNode.exprNames, hk, he, hx])) fun a b h => by rw [h]

omit hc in
theorem taskStart_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (entry : NodeId) (c : ConsumeNode) (l : Loc)
    {priv₂ priv₁ : List Name} (hp : ∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) {pre₂ pre₁ : St}
    (h : St.Sim π P L pre₂ pre₁) (q : List Val) :
    St.Sim π P L (taskStart ctx₂ s entry c (l.ren π) priv₂ pre₂ q) (taskStart ctx₁ s entry c l priv₁ pre₁ q) := by
  unfold taskStart chunkMemName
  refine ⟨?_, rfl, h.vals, Log.NamesP.empty P⟩
  have hq := (h.mem.allocAll_congr hp).setQueue l (q.drop c.chunksize)
  cases hd : s.consumedStream? entry with
  | none => exact hq
  | some d =>
    simp only [Option.map_some]
    rw [(hS.data _ (State.consumedStream?_names hd).2).cname]
    exact hq.writeChunk _ _

omit hc in
theorem taskLog_ren {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {priv₂ priv₁ : List Name}
    (hp : ∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) (l : Loc) :
    taskLog st₂ priv₂ (l.ren π) = (taskLog st₁ priv₁ l).ren π := by
  unfold taskLog
  rw [h.log, Log.dropNames_congr hp, ← Log.ren_dropNames, Log.ren_stripQueue]

omit hc in
theorem taskPushed_ren {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) {priv₂ priv₁ : List Name}
    (hp : ∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) (l : Loc) :
    taskPushed st₂ priv₂ (l.ren π) = taskPushed st₁ priv₁ l := by
  unfold taskPushed
  rw [h.log, Log.dropNames_congr hp, ← Log.ren_dropNames, Log.pushedTo_ren]

omit hc in
theorem consumeFinish_sim {ctx₂ ctx₁ : Ctx} (hI : ctx₂.I = ctx₁.I) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {pre₂ pre₁ : St} (h : St.Sim π P L pre₂ pre₁) (l : Loc) (hlP : P l.name) (logs : List Log)
    (hl : ∀ g ∈ logs, Log.NamesP P g) (rest pushed : List Val) :
    ResSim (St.Sim π P L) (consumeFinish ctx₂ pre₂ (l.ren π) (logs.map (Log.ren π)) rest pushed)
      (consumeFinish ctx₁ pre₁ l logs rest pushed) := by
  unfold consumeFinish
  rw [hI]
  rcases merge_sim hw h.mem logs hl with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨ha.setQueue l _, ?_, h.vals,
      (h.logP.seq (Log.NamesP.unionAll hl)).seq (Log.NamesP.queue _ hlP)⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]
    rfl

/-! ### Nested SDFGs -/

theorem nestedBindings_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) :
    nestedBindings ctx₂ s n = (nestedBindings ctx₁ s n).map (renConns π) := by
  unfold nestedBindings
  have hbase : ∀ c base sub, nestedEndpoint s n c = .data base sub → Ctx.Strong π P ctx₂ ctx₁ base := by
    intro c base sub h
    unfold nestedEndpoint at h
    split at h
    · rename_i e he
      exact (hS.src (State.mem_inEdges (List.mem_of_find?_eq_some he)) h).1
    · split at h
      · rename_i e rest he
        exact (hS.dst (State.edge_of_outEdgesOn (he ▸ List.mem_cons_self)) h).1
      · cases h
  suffices ∀ (cs : List Conn) (acc : List (Name × Name)),
      cs.foldlM (fun acc c =>
        match nestedEndpoint s n c with
        | .data base _ => pure (acc ++ [(c, ctx₂.cname base)])
        | .code _ => pure acc
        | .invalid => Res.err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) (renConns π acc) =
      (cs.foldlM (fun acc c =>
        match nestedEndpoint s n c with
        | .data base _ => pure (acc ++ [(c, ctx₁.cname base)])
        | .code _ => pure acc
        | .invalid => Res.err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) acc).map
        (renConns π) from this _ []
  intro cs
  induction cs with
  | nil => intro acc; rfl
  | cons c cs ih =>
    intro acc
    simp only [List.foldlM_cons]
    split
    · next base sub hep =>
      have e : renConns π acc ++ [(c, ctx₂.cname base)] = renConns π (acc ++ [(c, ctx₁.cname base)]) := by
        simp [renConns, (hbase c base sub hep).cname]
      exact (congrArg (fun x => List.foldlM _ x cs) e).trans (ih _)
    · exact ih acc
    · rfl

theorem lookup_renConns (conns : List (Name × Name)) (n : Name) :
    (renConns π conns).lookup n = (conns.lookup n).map π := by
  induction conns with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨c, l⟩ := x
    simp only [renConns, List.map_cons, List.lookup] at ih ⊢
    split <;> simp_all

omit hc in
theorem lookup_P {conns : List (Name × Name)} (hP : ∀ x ∈ conns, P x.2) {n l : Name}
    (h : conns.lookup n = some l) : P l := by
  induction conns with
  | nil => cases h
  | cons x xs ih =>
    obtain ⟨c, l'⟩ := x
    simp only [List.lookup] at h
    split at h
    · cases h; exact hP _ List.mem_cons_self
    · exact ih (fun y hy => hP y (List.mem_cons_of_mem _ hy)) h

omit hc in
theorem enter_cname_ren (hsc : ScopeSim π P ctx₂ ctx₁) (inner : SDFG) (node : NodeId) :
    ∀ (conns : List (Name × Name)) (x : Name),
      (ctx₂.enter inner node (renConns π conns)).cname x = π ((ctx₁.enter inner node conns).cname x)
  | [], x => by simp only [Ctx.enter, renConns, List.map_nil, List.lookup, Option.getD_none, (hsc _).1]
  | (c, l) :: cs, x => by
    have ih := enter_cname_ren hsc inner node cs x
    simp only [Ctx.enter, renConns, List.map_cons, List.lookup] at ih ⊢
    split <;> simp_all

/-- Entering a nested SDFG from contexts related by `π`, with connector
bindings related by `π` and satisfying `P`: the inner contexts agree on every
name. -/
theorem enter_sim (hsc : ScopeSim π P ctx₂ ctx₁) (inner : SDFG) (node : NodeId) (conns : List (Name × Name))
    (hP : ∀ x ∈ conns, P x.2) :
    CtxSim π P (ctx₂.enter inner node (renConns π conns)) (ctx₁.enter inner node conns) ∧
      (∀ x, Ctx.Strong π P (ctx₂.enter inner node (renConns π conns)) (ctx₁.enter inner node conns) x) ∧
      ScopeSim π P (ctx₂.enter inner node (renConns π conns)) (ctx₁.enter inner node conns) := by
  have hcn : ∀ x, (ctx₂.enter inner node (renConns π conns)).cname x = π ((ctx₁.enter inner node conns).cname x) :=
    fun x => enter_cname_ren hsc inner node conns x
  have hPc : ∀ x, P ((ctx₁.enter inner node conns).cname x) := by
    intro x
    simp only [Ctx.enter]
    cases h : conns.lookup x with
    | none => exact (hsc _).2
    | some l => exact lookup_P hP h
  refine ⟨⟨hc.I, rfl⟩, fun x => ?_, fun x => ⟨by simp only [Ctx.enter, (hsc _).1], (hsc _).2⟩⟩
  exact ⟨rfl, hcn x, hPc x, fun d _ y _ => ⟨rfl, fun _ => ⟨hcn y, hPc y⟩⟩⟩

theorem nestedValuation_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes)
    {inner : SDFG} {mapping : List (Sym × Expr)} {ni : Bool} (hk : n.kind = .nested inner mapping ni)
    {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) :
    ResSim Eq (nestedValuation ctx₂ ρ m₂ mapping) (nestedValuation ctx₁ ρ m₁ mapping) := by
  unfold nestedValuation
  refine ResSim.bindEq (ResSim.mapM_mem mapping fun xe hxe => ?_) fun vs => ResSim.ok rfl
  refine ResSim.bindEq (Ctx.evalR_sim hc hm ρ xe.2 fun x hx => hS.node_expr hn ?_) fun v => ResSim.ok rfl
  simp only [DNode.exprNames, hk]
  exact List.mem_flatMap.2 ⟨xe, hxe, List.mem_cons_of_mem _ hx⟩

/-- Contexts agreeing on every name agree on the names of any state. -/
theorem SNames.ofAll (hall : ∀ x, Ctx.Strong π P ctx₂ ctx₁ x) (hsc : ScopeSim π P ctx₂ ctx₁) (s : State) :
    SNames π P ctx₂ ctx₁ s :=
  ⟨fun x _ => hall x, fun x _ => (hall x).good₀, fun _ => hsc⟩

theorem bindCodeInputs_sim (hall : ∀ x, Ctx.Strong π P ctx₂ ctx₁ x) (ρ' : Valuation) (s : State) (n : DNode)
    {st₂ st₁ : St} (h : St.Sim π P L st₂ st₁) :
    ResSim (fun (x₂ x₁ : St × List Name) => St.Sim π P L x₂.1 x₁.1 ∧ x₂.2 = x₁.2.map π ∧ ∀ y ∈ x₁.2, P y)
      (bindCodeInputs ctx₂ ρ' s n st₂) (bindCodeInputs ctx₁ ρ' s n st₁) := by
  unfold bindCodeInputs
  refine ResSim.foldlM (R := fun (x₂ x₁ : St × List Name) => St.Sim π P L x₂.1 x₁.1 ∧ x₂.2 = x₁.2.map π ∧
    ∀ y ∈ x₁.2, P y) (fun c a b hab => ?_) _ ⟨h, rfl, fun _ h => by cases h⟩
  obtain ⟨a₂, cs₂⟩ := a
  obtain ⟨a₁, cs₁⟩ := b
  obtain ⟨ha, hc', hP⟩ := hab
  simp only at ha hc' hP ⊢
  subst hc'
  have hPc : ∀ y ∈ cs₁ ++ [ctx₁.cname c], P y := by
    intro y hy
    rcases List.mem_append.1 hy with hy | hy
    · exact hP y hy
    · simp only [List.mem_singleton] at hy; subst hy; exact (hall c).memP
  split
  · rw [ha.val?]
    split
    · rw [Ctx.loc_sim hc ha.mem ρ' (hall c).good₀]
      cases hl : ctx₁.loc ρ' a₁.mem c [] false with
      | err e => exact ResSim.err e
      | ok l =>
        simp only [Res.map, Res.ok_bind, Res.pure_eq]
        exact ResSim.ok ⟨ha.withMem (ha.mem.write (Ctx.locP hc ha.mem ρ' (hall c).good₀ hl) _),
          by simp [(hall c).cname], hPc⟩
    · exact ResSim.ok ⟨ha, by simp [(hall c).cname], hPc⟩
  · exact ResSim.ok ⟨ha, rfl, hP⟩

end Leaf3

section Leaf4

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation)
include hc

omit hc in
theorem nestedBindings_P {s : State} (hS : SNames π P ctx₂ ctx₁ s) (n : DNode) {conns : List (Name × Name)}
    (h : nestedBindings ctx₁ s n = .ok conns) : ∀ x ∈ conns, P x.2 := by
  unfold nestedBindings at h
  have hbase : ∀ c base sub, nestedEndpoint s n c = .data base sub → P (ctx₁.cname base) := by
    intro c base sub h
    unfold nestedEndpoint at h
    split at h
    · rename_i e he
      exact (hS.src (State.mem_inEdges (List.mem_of_find?_eq_some he)) h).1.memP
    · split at h
      · rename_i e rest he
        exact (hS.dst (State.edge_of_outEdgesOn (he ▸ List.mem_cons_self)) h).1.memP
      · cases h
  suffices ∀ (cs : List Conn) (acc : List (Name × Name)), (∀ x ∈ acc, P x.2) → ∀ conns,
      cs.foldlM (fun acc c =>
        match nestedEndpoint s n c with
        | .data base _ => pure (acc ++ [(c, ctx₁.cname base)])
        | .code _ => pure acc
        | .invalid => Res.err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) acc = .ok conns →
      ∀ x ∈ conns, P x.2 from this _ [] (fun _ h => by cases h) _ h
  intro cs
  induction cs with
  | nil => intro acc hacc conns h; simp only [List.foldlM_nil, Res.pure_eq, Res.ok_inj] at h; subst h; exact hacc
  | cons c cs ih =>
    intro acc hacc conns h
    simp only [List.foldlM_cons] at h
    split at h
    · next base sub hep =>
      refine ih _ (fun x hx => ?_) conns h
      rcases List.mem_append.1 hx with hx | hx
      · exact hacc x hx
      · simp only [List.mem_singleton] at hx; subst hx; exact hbase c base sub hep
    · exact ih acc hacc conns h
    · cases h

/-- The result of a nested set-up: inner contexts agreeing on every name, the
same inner valuation, simulating states, renamed private names satisfying
`P`. -/
def NestedSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : Ctx × Valuation × St × List Name) : Prop :=
  (CtxSim π P x₂.1 x₁.1 ∧ (∀ y, Ctx.Strong π P x₂.1 x₁.1 y) ∧ ScopeSim π P x₂.1 x₁.1) ∧ x₂.2.1 = x₁.2.1 ∧
    St.Sim π P L x₂.2.2.1 x₁.2.2.1 ∧ x₂.2.2.2 = x₁.2.2.2.map π ∧ ∀ y ∈ x₁.2.2.2, P y

theorem nestedSetup_sim {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {inner : SDFG}
    {mapping : List (Sym × Expr)} {ni : Bool} (hk : n.kind = .nested inner mapping ni) {st₂ st₁ : St}
    (h : St.Sim π P L st₂ st₁) :
    ResSim (NestedSim π P L) (nestedSetup ctx₂ ρ s n inner mapping st₂) (nestedSetup ctx₁ ρ s n inner mapping st₁) := by
  unfold nestedSetup
  rw [nestedBindings_sim hc hS]
  cases hb : nestedBindings ctx₁ s n with
  | err e => exact ResSim.err e
  | ok conns =>
    simp only [Res.map, Res.ok_bind]
    obtain ⟨hcI, hall, hscI⟩ := enter_sim hc (hS.scope ⟨n, hn, inner, mapping, ni, hk⟩) inner n.id conns
      (nestedBindings_P hS n hb)
    refine ResSim.bindEq (nestedValuation_sim hc ρ hS hn hk h.mem) fun ρ' => ?_
    have hs : St.Sim π P L
        { st₂ with mem := st₂.mem.allocAll (inner.callTransients.map (ctx₂.enter inner n.id (renConns π conns)).cname) }
        { st₁ with mem := st₁.mem.allocAll (inner.callTransients.map (ctx₁.enter inner n.id conns).cname) } :=
      h.withMem (h.mem.allocAll_congr fun x => by
        rw [List.map_map, List.map_congr_left fun t _ => (hall t).cname]; rfl)
    refine ResSim.bind (bindCodeInputs_sim hcI hall ρ' s n hs) fun x y hxy => ?_
    obtain ⟨a₂, c₂⟩ := x
    obtain ⟨a₁, c₁⟩ := y
    obtain ⟨ha, hcs, hP⟩ := hxy
    simp only at ha hcs hP ⊢
    refine ResSim.ok ⟨⟨hcI, hall, hscI⟩, rfl, ha, ?_, ?_⟩
    · simp only [hcs, List.map_append, List.map_map]
      congr 1
      exact List.map_congr_left fun t _ => (hall t).cname
    · intro y hy
      rcases List.mem_append.1 hy with hy | hy
      · exact hP y hy
      · obtain ⟨t, -, rfl⟩ := List.mem_map.1 hy
        exact (hall t).memP

omit hc in
theorem nestedFinish_sim {c₂ c₁ : Ctx} (hcI : CtxSim π P c₂ c₁) (hall : ∀ y, Ctx.Strong π P c₂ c₁ y)
    (ρ' : Valuation) (s : State) (n : DNode) (dropped : List Name) {pre₂ pre₁ : St} (h : St.Sim π P L pre₂ pre₁)
    {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) (g : Log) (hg : Log.NamesP P g) :
    ResSim (St.Sim π P L) (nestedFinish c₂ ρ' s n (dropped.map π) pre₂ m₂ (g.ren π))
      (nestedFinish c₁ ρ' s n dropped pre₁ m₁ g) := by
  unfold nestedFinish
  refine ResSim.foldlM (fun e a b hab => ?_) _ ⟨hm, ?_, h.vals, h.logP.seq (hg.dropNames _)⟩
  · split
    · split
      · rw [Ctx.loc_sim hcI hm ρ' (hall _).good₀]
        cases hl : c₁.loc ρ' m₁ _ [] false with
        | err e => exact ResSim.err e
        | ok l =>
          simp only [Res.map, Res.ok_bind]
          rcases hm.read (Ctx.locP hcI hm ρ' (hall _).good₀ hl) with h₁ | h₁ <;> rw [h₁]
          · exact ResSim.ub rfl _
          · split
            · exact ResSim.ok (hab.setVal _ _)
            · exact ResSim.err _
      · exact ResSim.ok hab
    · exact ResSim.ok hab
  · simp only [h.log, Log.ren_seq, Log.ren_dropNames]

/-! ### Control flow -/

/-- The control-flow expressions of a region and the states of its blocks
satisfy the agreement (`SNames`): the hypothesis of the control-flow
simulation. -/
def IEdge.names (e : IEdge) : List Name := e.cond.names ++ e.assigns.flatMap fun (_, x) => x.names

theorem trueEdges_sim {rg : Region} (hr : ∀ e ∈ rg.edges, ∀ x ∈ e.names, Ctx.Good₀ π P ctx₂ ctx₁ x)
    {c₂ c₁ : CSt} (h : CSt.Sim π P L c₂ c₁) (bid : BlockId) :
    ResSim Eq (trueEdges ctx₂ rg c₂ bid) (trueEdges ctx₁ rg c₁ bid) := by
  unfold trueEdges
  refine ResSim.filterMapM_mem _ fun e he => ?_
  rw [h.ρ]
  exact ResSim.bindEq (Ctx.evalBoolR_sim hc h.mem c₁.ρ e.cond fun x hx =>
    hr e (List.mem_filter.1 he).1 x (List.mem_append_left _ hx)) fun b => ResSim.ok rfl

theorem applyAssigns_sim {c₂ c₁ : CSt} (h : CSt.Sim π P L c₂ c₁) (as : List (Sym × Expr))
    (ha : ∀ x ∈ as.flatMap (fun (_, e) => e.names), Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim (CSt.Sim π P L) (applyAssigns ctx₂ c₂ as) (applyAssigns ctx₁ c₁ as) := by
  unfold applyAssigns
  rw [h.ρ]
  refine ResSim.bindEq (ResSim.mapM_mem as fun xe hxe => ?_) fun vs => ?_
  · exact ResSim.bindEq (Ctx.evalR_sim hc h.mem c₁.ρ xe.2 fun x hx => ha x (List.mem_flatMap.2 ⟨xe, hxe, hx⟩))
      fun v => ResSim.ok rfl
  · exact ResSim.ok ⟨rfl, h.mem, h.log, h.logP⟩

theorem applyAssign?_sim {c₂ c₁ : CSt} (h : CSt.Sim π P L c₂ c₁) (a : Option (Sym × Expr))
    (ha : ∀ x ∈ (a.map fun (_, e) => e.names).getD [], Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim (CSt.Sim π P L) (applyAssign? ctx₂ c₂ a) (applyAssign? ctx₁ c₁ a) := by
  cases a with
  | none => exact ResSim.ok h
  | some a => exact applyAssigns_sim hc h [a] (by simpa using ha)

theorem evalBoolR_cst_sim {c₂ c₁ : CSt} (h : CSt.Sim π P L c₂ c₁) (e : Expr)
    (he : ∀ x ∈ e.names, Ctx.Good₀ π P ctx₂ ctx₁ x) :
    ResSim Eq (ctx₂.evalBoolR c₂.ρ c₂.mem e) (ctx₁.evalBoolR c₁.ρ c₁.mem e) := by
  rw [h.ρ]; exact Ctx.evalBoolR_sim hc h.mem c₁.ρ e he

end Leaf4

/-! ## Contexts agreeing on the names of a control-flow hierarchy -/

section RegionNames

variable (π : Equiv.Perm Name) (P : Name → Prop) (ctx₂ ctx₁ : Ctx)

/-- Agreement on all names of an optional expression. -/
def OptNames (e : Option Expr) : Prop := ∀ x ∈ (e.map Expr.names).getD [], Ctx.Good₀ π P ctx₂ ctx₁ x

/-- Agreement on the names of an optional assignment. -/
def AssignNames (a : Option (Sym × Expr)) : Prop := ∀ x ∈ (a.map fun (_, e) => e.names).getD [], Ctx.Good₀ π P ctx₂ ctx₁ x

mutual

/-- The contexts agree on the names of every state and control-flow
expression of a region. -/
def Region.Names : Region → Prop
  | ⟨_, _, blocks, _, edges⟩ =>
    Blocks.Names blocks ∧ ∀ e ∈ edges, ∀ x ∈ e.names, Ctx.Good₀ π P ctx₂ ctx₁ x

def Blocks.Names : List Block → Prop
  | [] => True
  | b :: bs => Block.Names b ∧ Blocks.Names bs

def Block.Names : Block → Prop
  | .state s => SNames π P ctx₂ ctx₁ s
  | .region r => Region.Names r
  | .loop ⟨_, _, init, cond, update, _, _, body⟩ =>
    AssignNames π P ctx₂ ctx₁ init ∧ (∀ x ∈ cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ update ∧ Region.Names body
  | .cond _ brs => Branches.Names brs
  | .breakBlock | .continueBlock | .returnBlock => True

def Branches.Names : List (Option Expr × Region) → Prop
  | [] => True
  | (c, r) :: rest => OptNames π P ctx₂ ctx₁ c ∧ Region.Names r ∧ Branches.Names rest

end

end RegionNames

section RegionNamesLemmas

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx}

theorem Blocks.Names.get : ∀ {bs : List Block} {i : Nat} {b : Block}, Blocks.Names π P ctx₂ ctx₁ bs → bs[i]? = some b →
    Block.Names π P ctx₂ ctx₁ b
  | [], _, _, _, h => by simp at h
  | b :: bs, 0, b', hb, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact hb.1
  | b :: bs, i + 1, b', hb, h => by
    simp only [List.getElem?_cons_succ] at h
    exact Blocks.Names.get hb.2 h

theorem Region.Names.block {rg : Region} (h : Region.Names π P ctx₂ ctx₁ rg) {bid : BlockId} {b : Block}
    (hb : rg.block? bid = some b) : Block.Names π P ctx₂ ctx₁ b := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg
  exact Blocks.Names.get h.1 hb

theorem Region.Names.edges {rg : Region} (h : Region.Names π P ctx₂ ctx₁ rg) :
    ∀ e ∈ rg.edges, ∀ x ∈ e.names, Ctx.Good₀ π P ctx₂ ctx₁ x := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg
  exact h.2

theorem Region.Names.assigns {rg : Region} (h : Region.Names π P ctx₂ ctx₁ rg) {e : IEdge} (he : e ∈ rg.edges) :
    ∀ x ∈ e.assigns.flatMap (fun (_, x) => x.names), Ctx.Good₀ π P ctx₂ ctx₁ x :=
  fun x hx => h.edges e he x (List.mem_append_right _ hx)

theorem SNames.ofAll' (hall : ∀ x, Ctx.Strong π P ctx₂ ctx₁ x) (hsc : ScopeSim π P ctx₂ ctx₁) (s : State) :
    SNames π P ctx₂ ctx₁ s :=
  ⟨fun x _ => hall x, fun x _ => (hall x).good₀, fun _ => hsc⟩

section OfAll

variable (hall : ∀ x, Ctx.Strong π P ctx₂ ctx₁ x) (hsc : ScopeSim π P ctx₂ ctx₁)
include hall hsc

mutual

/-- Contexts agreeing on every name agree on any hierarchy. -/
theorem Region.Names.ofAll : ∀ rg : Region, Region.Names π P ctx₂ ctx₁ rg
  | ⟨_, _, blocks, _, _⟩ => ⟨Blocks.Names.ofAll blocks, fun _ _ x _ => (hall x).good₀⟩

theorem Blocks.Names.ofAll : ∀ bs : List Block, Blocks.Names π P ctx₂ ctx₁ bs
  | [] => trivial
  | b :: bs => ⟨Block.Names.ofAll b, Blocks.Names.ofAll bs⟩

theorem Block.Names.ofAll : ∀ b : Block, Block.Names π P ctx₂ ctx₁ b
  | .state s => SNames.ofAll' hall hsc s
  | .region r => Region.Names.ofAll r
  | .loop ⟨_, _, _, _, _, _, _, body⟩ =>
    ⟨fun x _ => (hall x).good₀, fun x _ => (hall x).good₀, fun x _ => (hall x).good₀, Region.Names.ofAll body⟩
  | .cond _ brs => Branches.Names.ofAll brs
  | .breakBlock | .continueBlock | .returnBlock => trivial

theorem Branches.Names.ofAll : ∀ brs : List (Option Expr × Region), Branches.Names π P ctx₂ ctx₁ brs
  | [] => trivial
  | (_, r) :: rest => ⟨fun x _ => (hall x).good₀, Region.Names.ofAll r, Branches.Names.ofAll rest⟩

end

end OfAll

end RegionNamesLemmas

section RelHelpers

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} {ctx₂ ctx₁ : Ctx}

theorem consumeCond_sim' (hc : CtxSim π P ctx₂ ctx₁) (ρ : Valuation) {c : ConsumeNode}
    (hcond : OptNames π P ctx₂ ctx₁ c.condition) {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁) (l : Loc) (q : List Val) :
    ResSim Eq (consumeCond ctx₂ ρ c m₂ (l.ren π) q) (consumeCond ctx₁ ρ c m₁ l q) := by
  unfold consumeCond
  split
  · exact ResSim.ok rfl
  · rename_i e he
    exact ResSim.map _ _ (Ctx.evalBoolR_sim hc (hm.setQueue l q) ρ _ fun x hx => hcond x (by simp [he, hx]))
      fun a b h => by rw [h]

theorem SNames.consume {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {c : ConsumeNode}
    (hk : n.kind = .consumeEntry c) : OptNames π P ctx₂ ctx₁ c.condition := by
  intro x hx
  refine hS.node_expr hn ?_
  cases hc : c.condition with
  | none => simp [hc] at hx
  | some e => simp only [hc, Option.map_some, Option.getD_some] at hx; simp [DNode.exprNames, hk, hc, hx]

theorem Block.Names.loop {l : LoopRegion} (h : Block.Names π P ctx₂ ctx₁ (.loop l)) :
    AssignNames π P ctx₂ ctx₁ l.init ∧ (∀ x ∈ l.cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ l.update ∧ Region.Names π P ctx₂ ctx₁ l.body := by
  obtain ⟨_, _, _, _, _, _, _, _⟩ := l
  exact h

end RelHelpers

/-! ## The execution relations -/

/-- A sequential result: simulating states, renamed trace. -/
def SeqSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : St × List (NodeId × Log)) : Prop :=
  St.Sim π P L x₂.1 x₁.1 ∧ x₂.2 = renTrace π x₁.2

/-- A control-flow result: the same signal, simulating states. -/
def CtlSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : Ctl × CSt) : Prop := x₂.1 = x₁.1 ∧ CSt.Sim π P L x₂.2 x₁.2

/-- Logs of parallel executions: renamed, aliases of names satisfying `P`
binding bases satisfying `P`. -/
def LogsSim (π : Equiv.Perm Name) (P : Name → Prop) (l₂ l₁ : List Log) : Prop :=
  l₂ = l₁.map (Log.ren π) ∧ ∀ g ∈ l₁, Log.NamesP P g

/-- Consume tasks: renamed logs, the same queues. -/
def TasksSim (π : Equiv.Perm Name) (P : Name → Prop) (x₂ x₁ : List Log × List Val × List Val) : Prop :=
  LogsSim π P x₂.1 x₁.1 ∧ x₂.2 = x₁.2

/-- An SDFG invocation: the same valuation, simulating memory, renamed log. -/
def InvSim (π : Equiv.Perm Name) (P : Name → Prop) (L : Name → Prop) (x₂ x₁ : Valuation × Mem × Log) : Prop :=
  x₂.1 = x₁.1 ∧ Mem.Sim π P L x₂.2.1 x₁.2.1 ∧ x₂.2.2 = x₁.2.2.ren π ∧ Log.NamesP P x₁.2.2

theorem nestedSetup_I {ctx ctx' : Ctx} {ρ ρ' : Valuation} {s : State} {n : DNode} {inner : SDFG}
    {mapping : List (Sym × Expr)} {st st' : St} {dropped : List Name}
    (h : nestedSetup ctx ρ s n inner mapping st = .ok (ctx', ρ', st', dropped)) : ctx'.I = ctx.I := by
  unfold nestedSetup at h
  cases hb : nestedBindings ctx s n with
  | err e => rw [hb] at h; cases h
  | ok conns =>
    rw [hb] at h
    simp only [Res.ok_bind] at h
    cases hv : nestedValuation ctx ρ st.mem mapping with
    | err e => rw [hv] at h; cases h
    | ok ρ₀ =>
      rw [hv] at h
      simp only [Res.ok_bind] at h
      cases hc : bindCodeInputs (ctx.enter inner n.id conns) ρ₀ s n
          { st with mem := st.mem.allocAll (inner.callTransients.map (ctx.enter inner n.id conns).cname) } with
      | err e => rw [hc] at h; cases h
      | ok x =>
        rw [hc] at h
        simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
        rw [← h.1]
        rfl

theorem trueEdges_cons_mem {ctx : Ctx} {rg : Region} {c : CSt} {bid : BlockId} {ie : IEdge} {es : List IEdge}
    (h : trueEdges ctx rg c bid = .ok (ie :: es)) : ie ∈ rg.edges := by
  unfold trueEdges at h
  have key : ∀ (xs : List IEdge) (out : List IEdge), xs.filterMapM (fun e => do
      let b ← ctx.evalBoolR c.ρ c.mem e.cond
      pure (if b then some e else none)) = .ok out → ∀ y ∈ out, y ∈ xs := by
    intro xs
    induction xs with
    | nil => intro out h y hy; simp only [List.filterMapM_nil, Res.pure_eq, Res.ok_inj] at h; subst h; cases hy
    | cons x xs ih =>
      intro out h y hy
      simp only [List.filterMapM_cons] at h
      cases hb : ctx.evalBoolR c.ρ c.mem x.cond with
      | err e => rw [hb] at h; cases h
      | ok b =>
        rw [hb] at h
        simp only [Res.ok_bind, Res.pure_eq] at h
        cases b with
        | false =>
          simp only [Bool.false_eq_true, ite_false] at h
          exact List.mem_cons_of_mem _ (ih out h y hy)
        | true =>
          simp only [ite_true] at h
          cases hr : xs.filterMapM (fun e => do
              let b ← ctx.evalBoolR c.ρ c.mem e.cond
              Res.ok (if b then some e else none)) with
          | err e => rw [hr] at h; cases h
          | ok out' =>
            rw [hr] at h
            simp only [Functor.map, Res.ok_bind, Res.pure_eq, Res.ok_inj] at h
            subst h
            rcases List.mem_cons.1 hy with rfl | hy
            · exact List.mem_cons_self
            · exact List.mem_cons_of_mem _ (ih out' hr y hy)
  exact (List.mem_filter.1 (key _ _ h ie List.mem_cons_self)).1

section Rel

variable {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop}

mutual

theorem NodeExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {n : DNode} (hn : n ∈ s.nodes) {st₁ : St}
    {r₁ : Res St} :
    NodeExec ctx₁ ρ s n st₁ r₁ → ∀ st₂, St.Sim π P L st₂ st₁ →
      ∃ r₂, NodeExec ctx₂ ρ s n st₂ r₂ ∧ ResSim (St.Sim π P L) r₂ r₁
  | .simple hs, st₂, h => ⟨_, .simple hs, execSimple_sim hc ρ hw hS hn h⟩
  | .mapSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ mapSetup_sim hc ρ hS hn hk h)
    exact ⟨_, .mapSetupErr hk h₂, ResSim.of_err hu⟩
  | .mapIterErr hk hs hi, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ mapSetup_sim hc ρ hS hn hk h) with ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, pts₂⟩, h₂, hρ, ha, hp⟩
    · exact ⟨_, .mapSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hp
      rw [hρ, hp] at h₂
      obtain ⟨r₂, hi₂, hr⟩ := IterExec.sim hc hw hS hi _ _ (privateTransients_mem hc hS n.id) ha
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .mapIterErr hk h₂ hi₂, ResSim.of_err hu⟩
  | .mapScope hk hs hi, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ mapSetup_sim hc ρ hS hn hk h) with ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, pts₂⟩, h₂, hρ, ha, hp⟩
    · exact ⟨_, .mapSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hp
      rw [hρ, hp] at h₂
      obtain ⟨r₂, hi₂, hr⟩ := IterExec.sim hc hw hS hi _ _ (privateTransients_mem hc hS n.id) ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl, hlP⟩
      · exact ⟨_, .mapIterErr hk h₂ hi₂, ResSim.ub hu _⟩
      · refine ⟨_, .mapScope hk h₂ hi₂, ?_⟩
        rw [hl]
        exact scopeFinish_sim hc.I hw ha _ hlP
  | .consumeSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ consumeSetup_sim hc ρ hS hn hk h)
    exact ⟨_, .consumeSetupErr hk h₂, ResSim.of_err hu⟩
  | .consumeTasksErr hk hs ht, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ consumeSetup_sim hc ρ hS hn hk h) with
      ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, l₂, np₂⟩, h₂, hρ, ha, hl, hlP, hnp⟩
    · exact ⟨_, .consumeSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hl hlP hnp
      rw [hρ, hl, hnp] at h₂
      obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.sim hc hw hS (hS.consume hn hk) ht _ _
        (privateTransients_mem hc hS n.id) ha hlP
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      refine ⟨_, .consumeTasksErr hk h₂ (by rw [ha.mem.queues _ hlP]; exact ht₂), ResSim.of_err hu⟩
  | .consumeScope hk hs ht, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ consumeSetup_sim hc ρ hS hn hk h) with
      ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, l₂, np₂⟩, h₂, hρ, ha, hl, hlP, hnp⟩
    · exact ⟨_, .consumeSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hl hlP hnp
      rw [hρ, hl, hnp] at h₂
      obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.sim hc hw hS (hS.consume hn hk) ht _ _
        (privateTransients_mem hc hS n.id) ha hlP
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨logs₂, rest₂, pushed₂⟩, rfl, ⟨hl', hlP'⟩, hq⟩
      · exact ⟨_, .consumeTasksErr hk h₂ (by rw [ha.mem.queues _ hlP]; exact ht₂), ResSim.ub hu _⟩
      · simp only [Prod.mk.injEq] at hl' hq
        obtain ⟨rfl, rfl⟩ := hq
        subst hl'
        exact ⟨_, .consumeScope hk h₂ (by rw [ha.mem.queues _ hlP]; exact ht₂),
          consumeFinish_sim hc.I hw ha _ hlP _ hlP' _ _⟩
  | .nestedSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ nestedSetup_sim hc ρ hS hn hk h)
    exact ⟨_, .nestedSetupErr hk h₂, ResSim.of_err hu⟩
  | .nestedInnerErr hk hs hr, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ nestedSetup_sim hc ρ hS hn hk h) with
      ⟨e, h₂, hu⟩ | ⟨⟨c₂, ρ₂, a₂, d₂⟩, h₂, ⟨hcI, hall, hscI⟩, hρ, ha, hd, hdP⟩
    · exact ⟨_, .nestedSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hcI hall hscI hρ ha hd hdP
      rw [hρ, hd] at h₂
      obtain ⟨r₂, hr₂, hres⟩ := SDFGExec.sim hcI (by rw [nestedSetup_I hs]; exact hw)
        (Region.Names.ofAll hall hscI _) hr _ ha.mem
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hres
      exact ⟨_, .nestedInnerErr hk h₂ hr₂, ResSim.of_err hu⟩
  | .nested hk hs hr, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ nestedSetup_sim hc ρ hS hn hk h) with
      ⟨e, h₂, hu⟩ | ⟨⟨c₂, ρ₂, a₂, d₂⟩, h₂, ⟨hcI, hall, hscI⟩, hρ, ha, hd, hdP⟩
    · exact ⟨_, .nestedSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hcI hall hscI hρ ha hd hdP
      rw [hρ, hd] at h₂
      obtain ⟨r₂, hr₂, hres⟩ := SDFGExec.sim hcI (by rw [nestedSetup_I hs]; exact hw)
        (Region.Names.ofAll hall hscI _) hr _ ha.mem
      rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ρ₃, m₃, g₃⟩, rfl, hρ₃, hm₃, hg₃, hgP⟩
      · exact ⟨_, .nestedInnerErr hk h₂ hr₂, ResSim.ub hu _⟩
      · simp only at hρ₃ hm₃ hg₃ hgP
        subst hρ₃ hg₃
        exact ⟨_, .nested hk h₂ hr₂, nestedFinish_sim hcI hall _ s n _ ha hm₃ _ hgP⟩

theorem SeqExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {ids : List NodeId} {st₁ : St}
    {r₁ : Res (St × List (NodeId × Log))} :
    SeqExec ctx₁ ρ s ids st₁ r₁ → ∀ st₂, St.Sim π P L st₂ st₁ →
      ∃ r₂, SeqExec ctx₂ ρ s ids st₂ r₂ ∧ ResSim (SeqSim π P L) r₂ r₁
  | .nil, st₂, h => ⟨_, .nil, ResSim.ok ⟨h, rfl⟩⟩
  | .missing hm, st₂, h => ⟨_, .missing hm, ResSim.err _⟩
  | .headErr hn hx, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hc hw hS (State.node?_mem hn) hx _ h.forNode
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hn hx₂, ResSim.of_err hu⟩
  | .tailErr hn hx hrest, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hc hw hS (State.node?_mem hn) hx _ h.forNode
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hn hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.sim hc hw hS hrest _ (h.afterNode ha)
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hn hx₂ hrest₂, ResSim.of_err hu⟩
  | .cons hn hx hrest, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hc hw hS (State.node?_mem hn) hx _ h.forNode
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hn hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.sim hc hw hS hrest _ (h.afterNode ha)
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .tailErr hn hx₂ hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hn hx₂ hrest₂, ResSim.ok ⟨hb, ?_⟩⟩
        simp only at htr ⊢
        rw [htr, ha.log]
        rfl

theorem OrderedExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {all direct : List NodeId} {st₁ : St} {r₁ : Res St} :
    OrderedExec ctx₁ ρ s all direct st₁ r₁ → ∀ st₂, St.Sim π P L st₂ st₁ →
      ∃ r₂, OrderedExec ctx₂ ρ s all direct st₂ r₂ ∧ ResSim (St.Sim π P L) r₂ r₁
  | .prologueErr hp, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hp ▸ bindViews_sim hc ρ hS (s.viewOrder ctx₁ direct) h)
    exact ⟨_, .prologueErr (by rw [hS.viewOrder]; exact h₂), ResSim.of_err hu⟩
  | .seqErr ho hp hs, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim hc ρ hS (s.viewOrder ctx₁ direct) h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr (by rw [hS.viewOrder]; exact h₂), ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hc hw hS hs _ ha
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .seqErr ho (by rw [hS.viewOrder]; exact h₂) hs₂, ResSim.of_err hu⟩
  | .hbRace ho hp hs hf, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim hc ρ hS (s.viewOrder ctx₁ direct) h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr (by rw [hS.viewOrder]; exact h₂), ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hc hw hS hs _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .seqErr ho (by rw [hS.viewOrder]; exact h₂) hs₂, ResSim.ub hu _⟩
      · simp only at htr
        subst htr
        exact ⟨_, .hbRace ho (by rw [hS.viewOrder]; exact h₂) hs₂ (by rw [State.hbFree_ren]; exact hf), ResSim.ub rfl _⟩
  | .run ho hp hs hf, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim hc ρ hS (s.viewOrder ctx₁ direct) h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr (by rw [hS.viewOrder]; exact h₂), ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hc hw hS hs _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .seqErr ho (by rw [hS.viewOrder]; exact h₂) hs₂, ResSim.ub hu _⟩
      · simp only at htr
        subst htr
        exact ⟨_, .run ho (by rw [hS.viewOrder]; exact h₂) hs₂ (by rw [State.hbFree_ren]; exact hf), ResSim.ok hb⟩

theorem IterExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {entry : NodeId} {params : List Sym}
    {priv₁ : List Name} {pre₁ : St} {pts : List Idx} {r₁ : Res (List Log)} :
    IterExec ctx₁ ρ s entry params priv₁ pre₁ pts r₁ → ∀ priv₂ pre₂, (∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) →
      St.Sim π P L pre₂ pre₁ →
      ∃ r₂, IterExec ctx₂ ρ s entry params priv₂ pre₂ pts r₂ ∧ ResSim (LogsSim π P) r₂ r₁
  | .nil, priv₂, pre₂, hp, h => ⟨_, .nil, ResSim.ok ⟨rfl, fun _ h => by cases h⟩⟩
  | .headErr hb, priv₂, pre₂, hp, h => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hc hw hS hb _ (iterStart_sim h hp)
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hb₂, ResSim.of_err hu⟩
  | .tailErr hrest, priv₂, pre₂, hp, h => by
    obtain ⟨r₃, hrest₂, hr₃⟩ := IterExec.sim hc hw hS hrest _ _ hp h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
    exact ⟨_, .tailErr hrest₂, ResSim.of_err hu⟩
  | .cons hb hrest, priv₂, pre₂, hp, h => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hc hw hS hb _ (iterStart_sim h hp)
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := IterExec.sim hc hw hS hrest _ _ hp h
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl, hlP⟩
      · exact ⟨_, .tailErr hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hb₂ hrest₂, ResSim.ok ⟨?_, ?_⟩⟩
        · rw [hl, ha.log, Log.dropNames_congr hp, ← Log.ren_dropNames]
          rfl
        · intro g hg
          rcases List.mem_cons.1 hg with rfl | hg
          · exact ha.logP.dropNames _
          · exact hlP g hg

theorem ConsumeTasks.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {c : ConsumeNode}
    (hcond : OptNames π P ctx₂ ctx₁ c.condition) {entry : NodeId} {l : Loc} {np : Int}
    {priv₁ : List Name} {pre₁ : St} {q : List Val} {r₁ : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx₁ ρ s entry c l np priv₁ pre₁ q r₁ → ∀ priv₂ pre₂, (∀ x, x ∈ priv₂ ↔ x ∈ priv₁.map π) →
      St.Sim π P L pre₂ pre₁ → P l.name →
      ∃ r₂, ConsumeTasks ctx₂ ρ s entry c (l.ren π) np priv₂ pre₂ q r₂ ∧ ResSim (TasksSim π P) r₂ r₁
  | .condErr hcd, priv₂, pre₂, hp, h, hl => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hcd ▸ consumeCond_sim' hc ρ hcond h.mem l q)
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .stop hcd, priv₂, pre₂, hp, h, hl => by
    rcases ResSim.ok_cases (hcd ▸ consumeCond_sim' hc ρ hcond h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · exact ⟨_, .stop h₂, ResSim.ok ⟨⟨rfl, fun _ h => by cases h⟩, rfl⟩⟩
  | .bodyErr hcd hq hpe hb, priv₂, pre₂, hp, h, hl => by
    rcases ResSim.ok_cases (hcd ▸ consumeCond_sim' hc ρ hcond h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hc hw hS hb _ (taskStart_sim hS entry c l hp h q)
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .bodyErr h₂ hq hpe hb₂, ResSim.of_err hu⟩
  | .restErr hcd hq hpe hb hrest, priv₂, pre₂, hp, h, hl => by
    rcases ResSim.ok_cases (hcd ▸ consumeCond_sim' hc ρ hcond h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hc hw hS hb _ (taskStart_sim hS entry c l hp h q)
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
      · exact ⟨_, .bodyErr h₂ hq hpe hb₂, ResSim.ub hu _⟩
      · obtain ⟨r₃, hrest₂, hr₃⟩ := ConsumeTasks.sim hc hw hS hcond hrest _ _ hp h hl
        obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
        exact ⟨_, .restErr h₂ hq hpe hb₂ hrest₂, ResSim.of_err hu⟩
  | .step hcd hq hpe hb hrest, priv₂, pre₂, hp, h, hl => by
    rcases ResSim.ok_cases (hcd ▸ consumeCond_sim' hc ρ hcond h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hc hw hS hb _ (taskStart_sim hS entry c l hp h q)
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
      · exact ⟨_, .bodyErr h₂ hq hpe hb₂, ResSim.ub hu _⟩
      · obtain ⟨r₃, hrest₂, hr₃⟩ := ConsumeTasks.sim hc hw hS hcond hrest _ _ hp h hl
        rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨⟨logs₂, rest₂, pushed₂⟩, rfl, ⟨hl', hlP⟩, hq'⟩
        · exact ⟨_, .restErr h₂ hq hpe hb₂ hrest₂, ResSim.ub hu _⟩
        · simp only [Prod.mk.injEq] at hl' hq'
          obtain ⟨rfl, rfl⟩ := hq'
          subst hl'
          refine ⟨_, .step h₂ hq hpe hb₂ hrest₂, ResSim.ok ⟨⟨?_, ?_⟩, ?_⟩⟩
          · simp only [List.map_cons, taskLog_ren ha hp]
          · intro g hg
            rcases List.mem_cons.1 hg with rfl | hg
            · exact (ha.logP.dropNames _).stripQueue _
            · exact hlP g hg
          · simp only [taskPushed_ren ha hp]

theorem CompsExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {ρ : Valuation} {s : State} (hS : SNames π P ctx₂ ctx₁ s) {m₁ : Mem} {comps : List (List NodeId)}
    {r₁ : Res (List Log)} :
    CompsExec ctx₁ ρ s m₁ comps r₁ → ∀ m₂, Mem.Sim π P L m₂ m₁ →
      ∃ r₂, CompsExec ctx₂ ρ s m₂ comps r₂ ∧ ResSim (LogsSim π P) r₂ r₁
  | .nil, m₂, h => ⟨_, .nil, ResSim.ok ⟨rfl, fun _ h => by cases h⟩⟩
  | .headErr hcp, m₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.sim hc hw hS hcp { mem := m₂ } ⟨h, rfl, rfl, Log.NamesP.empty P⟩
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hc₂, ResSim.of_err hu⟩
  | .tailErr hrest, m₂, h => by
    obtain ⟨r₃, hrest₂, hr₃⟩ := CompsExec.sim hc hw hS hrest _ h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
    exact ⟨_, .tailErr hrest₂, ResSim.of_err hu⟩
  | .cons hcp hrest, m₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.sim hc hw hS hcp { mem := m₂ } ⟨h, rfl, rfl, Log.NamesP.empty P⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hc₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := CompsExec.sim hc hw hS hrest _ h
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl, hlP⟩
      · exact ⟨_, .tailErr hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hc₂ hrest₂, ResSim.ok ⟨?_, ?_⟩⟩
        · rw [hl, ha.log]; rfl
        · intro g hg
          rcases List.mem_cons.1 hg with rfl | hg
          · exact ha.logP
          · exact hlP g hg

theorem StateExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {s : State} (hS : SNames π P ctx₂ ctx₁ s) {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    StateExec ctx₁ c₁ s r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, StateExec ctx₂ c₂ s r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .compsErr hcp, c₂, h => by
    have ha := allocate_sim h.mem (stateTransients_mem hc hS) fun t ht =>
      let hg := stateTransients_good hS ht
      ⟨by simp only [Ctx.isStream, hg.1.1], (hg.1.2 (Option.isSome_iff_ne_none.mp hg.2)).1⟩
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.sim hc hw hS hcp _ ha
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .compsErr (by rw [h.ρ]; exact hc₂), ResSim.of_err hu⟩
  | .run hcp, c₂, h => by
    have ha := allocate_sim h.mem (stateTransients_mem hc hS) fun t ht =>
      let hg := stateTransients_good hS ht
      ⟨by simp only [Ctx.isStream, hg.1.1], (hg.1.2 (Option.isSome_iff_ne_none.mp hg.2)).1⟩
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.sim hc hw hS hcp _ ha
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl, hlP⟩
    · exact ⟨_, .compsErr (by rw [h.ρ]; exact hc₂), ResSim.ub hu _⟩
    · subst hl
      exact ⟨_, .run (by rw [h.ρ]; exact hc₂), (stateFinish_sim hc.I hw h _ hlP).mono fun _ _ hx => hx⟩

theorem BlockExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {b : Block} (hb : Block.Names π P ctx₂ ctx₁ b) {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    BlockExec ctx₁ c₁ b r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, BlockExec ctx₂ c₂ b r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .state hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := StateExec.sim hc hw hb hx _ h
    exact ⟨_, .state hx₂, hr⟩
  | .region hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hb hx _ h
    exact ⟨_, .region hx₂, hr⟩
  | .loop hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := LoopExec.sim hc hw hb.loop hx _ h
    exact ⟨_, .loop hx₂, hr⟩
  | .cond hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := CondExec.sim hc hw hb hx _ h
    exact ⟨_, .cond hx₂, hr⟩
  | .brk, c₂, h => ⟨_, .brk, ResSim.ok ⟨rfl, h⟩⟩
  | .cont, c₂, h => ⟨_, .cont, ResSim.ok ⟨rfl, h⟩⟩
  | .ret, c₂, h => ⟨_, .ret, ResSim.ok ⟨rfl, h⟩⟩

theorem RegionAt.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {rg : Region} (hr : Region.Names π P ctx₂ ctx₁ rg) {c₁ : CSt} {bid : BlockId} {r₁ : Res (Ctl × CSt)} :
    RegionAt ctx₁ rg c₁ bid r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, RegionAt ctx₂ rg c₂ bid r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .missing hb, c₂, h => ⟨_, .missing hb, ResSim.err _⟩
  | .blockErr hb hx, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hres
    exact ⟨_, .blockErr hb hx₂, ResSim.of_err hu⟩
  | .exit hb hx hcl, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .exit hb hx₂ hcl, ResSim.ok ⟨rfl, hd⟩⟩
  | .edgesErr hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ trueEdges_sim hc hr.edges hd bid)
      exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.of_err hu⟩
  | .stop hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim hc hr.edges hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · exact ⟨_, .stop hb hx₂ h₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .ambiguous hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim hc hr.edges hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · exact ⟨_, .ambiguous hb hx₂ h₂, ResSim.err _⟩
  | .assignErr hb hx he ha, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim hc hr.edges hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · have hie : _ ∈ rg.edges := trueEdges_cons_mem he
        obtain ⟨e', ha₂, hu⟩ := ResSim.err_inv (ha ▸ applyAssigns_sim hc hd _ (hr.assigns hie))
        exact ⟨_, .assignErr hb hx₂ h₂ ha₂, ResSim.of_err hu⟩
  | .step hb hx he ha hn, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := BlockExec.sim hc hw (hr.block hb) hx _ h
    rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim hc hr.edges hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · have hie : _ ∈ rg.edges := trueEdges_cons_mem he
        rcases ResSim.ok_cases (ha ▸ applyAssigns_sim hc hd _ (hr.assigns hie)) with ⟨e, ha₂, hu⟩ | ⟨d₃, ha₂, hd₃⟩
        · exact ⟨_, .assignErr hb hx₂ h₂ ha₂, ResSim.ub hu _⟩
        · obtain ⟨r₃, hn₂, hr₃⟩ := RegionAt.sim hc hw hr hn _ hd₃
          exact ⟨_, .step hb hx₂ h₂ ha₂ hn₂, hr₃⟩

theorem RegionExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {rg : Region} (hr : Region.Names π P ctx₂ ctx₁ rg) {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    RegionExec ctx₁ c₁ rg r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, RegionExec ctx₂ c₂ rg r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .start hx, c₂, h => by
    obtain ⟨r₂, hx₂, hres⟩ := RegionAt.sim hc hw hr hx _ h
    exact ⟨_, .start hx₂, hres⟩

theorem LoopExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {l : LoopRegion}
    (hl : AssignNames π P ctx₂ ctx₁ l.init ∧ (∀ x ∈ l.cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ l.update ∧ Region.Names π P ctx₂ ctx₁ l.body)
    {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    LoopExec ctx₁ c₁ l r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, LoopExec ctx₂ c₂ l r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .initErr hi, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hi ▸ applyAssign?_sim hc h l.init hl.1)
    exact ⟨_, .initErr h₂, ResSim.of_err hu⟩
  | .guardFirst hi hv hg, c₂, h => by
    rcases ResSim.ok_cases (hi ▸ applyAssign?_sim hc h l.init hl.1) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .initErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hg₂, hr⟩ := LoopGuard.sim hc hw hl hg _ hd
      exact ⟨_, .guardFirst h₂ hv hg₂, hr⟩
  | .bodyFirst hi hv hb, c₂, h => by
    rcases ResSim.ok_cases (hi ▸ applyAssign?_sim hc h l.init hl.1) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .initErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hc hw hl hb _ hd
      exact ⟨_, .bodyFirst h₂ hv hb₂, hr⟩

theorem LoopGuard.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {l : LoopRegion}
    (hl : AssignNames π P ctx₂ ctx₁ l.init ∧ (∀ x ∈ l.cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ l.update ∧ Region.Names π P ctx₂ ctx₁ l.body)
    {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    LoopGuard ctx₁ c₁ l r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, LoopGuard ctx₂ c₂ l r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .condErr hcd, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1)
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .exit hcd, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · exact ⟨_, .exit h₂, ResSim.ok ⟨rfl, h⟩⟩
  | .enter hcd hb, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hc hw hl hb _ h
      exact ⟨_, .enter h₂ hb₂, hr⟩

theorem LoopIter.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {l : LoopRegion}
    (hl : AssignNames π P ctx₂ ctx₁ l.init ∧ (∀ x ∈ l.cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ l.update ∧ Region.Names π P ctx₂ ctx₁ l.body)
    {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    LoopIter ctx₁ c₁ l r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, LoopIter ctx₂ c₂ l r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .bodyErr hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hc hw hl.2.2.2 hb _ h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .bodyErr hb₂, ResSim.of_err hu⟩
  | .ret hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hc hw hl.2.2.2 hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .ret hb₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .brk hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hc hw hl.2.2.2 hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .brk hb₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .tail hb hcl ht, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hc hw hl.2.2.2 hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      obtain ⟨r₃, ht₂, hr₃⟩ := LoopTail.sim hc hw hl ht _ hd
      exact ⟨_, .tail hb₂ hcl ht₂, hr₃⟩

theorem LoopTail.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {l : LoopRegion}
    (hl : AssignNames π P ctx₂ ctx₁ l.init ∧ (∀ x ∈ l.cond.names, Ctx.Good₀ π P ctx₂ ctx₁ x) ∧
      AssignNames π P ctx₂ ctx₁ l.update ∧ Region.Names π P ctx₂ ctx₁ l.body)
    {c₁ : CSt} {r₁ : Res (Ctl × CSt)} :
    LoopTail ctx₁ c₁ l r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, LoopTail ctx₂ c₂ l r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .updateErr hv hu', c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hu' ▸ applyAssign?_sim hc h l.update hl.2.2.1)
    exact ⟨_, .updateErr hv h₂, ResSim.of_err hu⟩
  | .updateGuard hv hu' hg, c₂, h => by
    rcases ResSim.ok_cases (hu' ▸ applyAssign?_sim hc h l.update hl.2.2.1) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .updateErr hv h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hg₂, hr⟩ := LoopGuard.sim hc hw hl hg _ hd
      exact ⟨_, .updateGuard hv h₂ hg₂, hr⟩
  | .condFirstErr hv hcd, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1)
    exact ⟨_, .condFirstErr hv h₂, ResSim.of_err hu⟩
  | .condFirstExit hv hcd, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · exact ⟨_, .condFirstExit hv h₂, ResSim.ok ⟨rfl, h⟩⟩
  | .condFirstUpdateErr hv hcd hu', c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · obtain ⟨e', h₃, hu⟩ := ResSim.err_inv (hu' ▸ applyAssign?_sim hc h l.update hl.2.2.1)
      exact ⟨_, .condFirstUpdateErr hv h₂ h₃, ResSim.of_err hu⟩
  | .condFirstNext hv hcd hu' hb, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h l.cond hl.2.1) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · rcases ResSim.ok_cases (hu' ▸ applyAssign?_sim hc h l.update hl.2.2.1) with ⟨e, h₃, hu⟩ | ⟨d, h₃, hd⟩
      · exact ⟨_, .condFirstUpdateErr hv h₂ h₃, ResSim.ub hu _⟩
      · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hc hw hl hb _ hd
        exact ⟨_, .condFirstNext hv h₂ h₃ hb₂, hr⟩

theorem CondExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {brs : List (Option Expr × Region)} (hb : Branches.Names π P ctx₂ ctx₁ brs) {c₁ : CSt}
    {r₁ : Res (Ctl × CSt)} :
    CondExec ctx₁ c₁ brs r₁ → ∀ c₂, CSt.Sim π P L c₂ c₁ →
      ∃ r₂, CondExec ctx₂ c₂ brs r₂ ∧ ResSim (CtlSim π P L) r₂ r₁
  | .nil, c₂, h => ⟨_, .nil, ResSim.ok ⟨rfl, h⟩⟩
  | .els hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hb.2.1 hx _ h
    exact ⟨_, .els hx₂, hr⟩
  | .condErr hcd, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hcd ▸ evalBoolR_cst_sim hc h _ (fun x hx => hb.1 x (by simp [hx])))
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .taken hcd hx, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h _ (fun x hx => hb.1 x (by simp [hx]))) with
      ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hb.2.1 hx _ h
      exact ⟨_, .taken h₂ hx₂, hr⟩
  | .skipped hcd hx, c₂, h => by
    rcases ResSim.ok_cases (hcd ▸ evalBoolR_cst_sim hc h _ (fun x hx => hb.1 x (by simp [hx]))) with
      ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hx₂, hr⟩ := CondExec.sim hc hw hb.2.2 hx _ h
      exact ⟨_, .skipped h₂ hx₂, hr⟩

theorem SDFGExec.sim {ctx₂ ctx₁ : Ctx} (hc : CtxSim π P ctx₂ ctx₁) (hw : ∀ op v, ctx₁.I.wcr op .undef v = .undef)
    {g : SDFG} (hg : Region.Names π P ctx₂ ctx₁ g.top) {ρ : Valuation} {m₁ : Mem}
    {r₁ : Res (Valuation × Mem × Log)} :
    SDFGExec ctx₁ ρ m₁ g r₁ → ∀ m₂, Mem.Sim π P L m₂ m₁ →
      ∃ r₂, SDFGExec ctx₂ ρ m₂ g r₂ ∧ ResSim (InvSim π P L) r₂ r₁
  | .err hx, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hg hx { ρ := g.withConstants ρ, mem := m₂ }
      ⟨rfl, h, rfl, Log.NamesP.empty P⟩
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .err hx₂, ResSim.of_err hu⟩
  | .escape hx hcl, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hg hx { ρ := g.withConstants ρ, mem := m₂ }
      ⟨rfl, h, rfl, Log.NamesP.empty P⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .err hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .escape hx₂ hcl, ResSim.err _⟩
  | .run hx hcl, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hc hw hg hx { ρ := g.withConstants ρ, mem := m₂ }
      ⟨rfl, h, rfl, Log.NamesP.empty P⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .err hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .run hx₂ hcl, ResSim.ok ⟨hd.ρ, hd.mem, hd.log, hd.logP⟩⟩

end

end Rel

/-! ## The identity case -/

/-- A context simulates itself, with no renaming and no restriction. -/
theorem CtxSim.refl (ctx : Ctx) : CtxSim 1 (fun _ => True) ctx ctx := ⟨rfl, rfl⟩

theorem Ctx.Strong.refl (ctx : Ctx) (n : Name) : Ctx.Strong 1 (fun _ => True) ctx ctx n :=
  ⟨rfl, rfl, trivial, fun _ _ _ _ => ⟨rfl, fun _ => ⟨rfl, trivial⟩⟩⟩

theorem SNames.refl (ctx : Ctx) (s : State) : SNames 1 (fun _ => True) ctx ctx s :=
  ⟨fun x _ => Ctx.Strong.refl ctx x, fun x _ => (Ctx.Strong.refl ctx x).good₀, fun _ _ => ⟨rfl, trivial⟩⟩

theorem Log.NamesP.true (g : Log) : Log.NamesP (fun _ => True) g :=
  ⟨fun _ _ => trivial, fun _ _ => trivial, fun _ _ => trivial, fun _ _ => ⟨trivial, trivial⟩, fun _ _ => trivial⟩

/-- Allocating transients that are no streams on the simulating side alone
keeps the simulation (their queues are untouched). -/
theorem allocate_left {π : Equiv.Perm Name} {P : Name → Prop} {L : Name → Prop} (ctx : Ctx) {m₂ m₁ : Mem} (hm : Mem.Sim π P L m₂ m₁)
    (ts : List Name) (hs : ∀ t ∈ ts, ctx.isStream t = false) (hL : ∀ t ∈ ts, L (π.symm (ctx.cname t))) :
    Mem.Sim π P L (ctx.allocate ts m₂) m₁ := by
  unfold Ctx.allocate
  induction ts generalizing m₂ with
  | nil => exact hm
  | cons t ts ih =>
    simp only [List.foldl_cons, hs t List.mem_cons_self, Bool.false_eq_true, ite_false]
    exact ih (hm.allocCells_left _ (hL t List.mem_cons_self)) (fun t' ht' => hs t' (List.mem_cons_of_mem _ ht'))
      fun t' ht' => hL t' (List.mem_cons_of_mem _ ht')

end Dace
