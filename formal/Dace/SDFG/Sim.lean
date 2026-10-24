-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.LogShift
import Dace.SDFG.Clean
import Dace.SDFG.Footprinted

/-!
# Simulation under renaming and less initialized memory (SDFG)

A run of the SDFG semantics is *simulated* by a run of the same program in a
context whose memory names are renamed by a permutation `π`
(`Ctx.ren`), from a memory that is the renamed memory except that some cells
may be uninitialized (`Mem.Sim`): the simulating run either ends in undefined
behavior (it read an uninitialized cell, or any other UB error), or ends like
the simulated run (the same error, or a related final state: the renamed
memory with possibly more uninitialized cells, the renamed log, the same
values).

Uses (`Decisions.md#D68`):

* `π = 1`: allocation makes memory less initialized, so a run from the
  unallocated memory is simulated by the run from the allocated one; the pair
  checker matches a tree segment against an SDFG state this way, since trees
  allocate fewer times than SDFGs;
* `π` a swap of the per-call transients of a nested SDFG with the tree's
  names for them: the nested call is matched against the flattened tree.

This file holds the foundations: the order `OptLe` on optional values,
monotone evaluation (`Expr.eval_mono`, `Subset.points_mono`), and the outcome
relation `ResSim` with its monadic combinators.
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

/-! ## Monotone evaluation -/

section EvalMono

variable {I₂ I₁ : ExprInterp} (happ : I₂.app = I₁.app) (hattr : I₂.readAttr = I₁.readAttr)
  (hrd : ∀ d is, OptLe (I₂.readData d is) (I₁.readData d is)) {ρ₂ ρ₁ : Valuation} (hρ : ∀ x, OptLe (ρ₂ x) (ρ₁ x))
include happ hattr hrd hρ

mutual

/-- **Evaluation is monotone**: an interpretation whose data reads and a
valuation whose symbols are each undefined or equal evaluate every
expression to undefined or the same value. -/
theorem Expr.eval_mono : ∀ e : Expr, OptLe (e.eval I₂ ρ₂) (e.eval I₁ ρ₁)
  | .lit l => by cases l <;> exact .inr rfl
  | .sym s => by simp only [Expr.eval]; exact hρ s
  | .un op a => by
    simp only [Expr.eval]
    exact OptLe.bind (Expr.eval_mono a) fun v => .inr rfl
  | .bin op a b => by
    simp only [Expr.eval]
    exact OptLe.bind (Expr.eval_mono a) fun v => OptLe.bind (Expr.eval_mono b) fun w => .inr rfl
  | .ite c t e => by
    simp only [Expr.eval]
    refine OptLe.bind (Expr.eval_mono c) fun cv => OptLe.bind (.inr rfl) fun b => ?_
    cases b
    · exact Expr.eval_mono e
    · exact Expr.eval_mono t
  | .app f args => by
    simp only [Expr.eval]
    refine OptLe.bind (Expr.evalList_mono args) fun vs => ?_
    rw [happ]; exact .inr rfl
  | .subscript d idx => by
    simp only [Expr.eval]
    exact OptLe.bind (Expr.evalList_mono idx) fun vs => OptLe.bind (.inr rfl) fun is => hrd d is
  | .attr b f => by
    simp only [Expr.eval]
    refine OptLe.bind (Expr.eval_mono b) fun v => ?_
    rw [hattr]; exact .inr rfl
  | .approx a _ => by simp only [Expr.eval]; exact Expr.eval_mono a

theorem Expr.evalList_mono : ∀ es : Exprs, OptLe (Expr.evalList I₂ ρ₂ es) (Expr.evalList I₁ ρ₁ es)
  | .nil => .inr rfl
  | .cons e es => by
    simp only [Expr.evalList]
    exact OptLe.bind (Expr.eval_mono e) fun v => OptLe.bind (Expr.evalList_mono es) fun vs => .inr rfl

end

theorem Expr.evalInt_mono (e : Expr) : OptLe (e.evalInt I₂ ρ₂) (e.evalInt I₁ ρ₁) :=
  OptLe.bind (Expr.eval_mono happ hattr hrd hρ e) fun _ => .inr rfl

theorem Expr.evalBool_mono (e : Expr) : OptLe (e.evalBool I₂ ρ₂) (e.evalBool I₁ ρ₁) :=
  OptLe.bind (Expr.eval_mono happ hattr hrd hρ e) fun _ => .inr rfl

end EvalMono

section PointsMono

variable {I₂ I₁ : ExprInterp} (happ : I₂.app = I₁.app) (hattr : I₂.readAttr = I₁.readAttr)
  (hrd : ∀ d is, OptLe (I₂.readData d is) (I₁.readData d is)) {ρ₂ ρ₁ : Valuation} (hρ : ∀ x, OptLe (ρ₂ x) (ρ₁ x))
include happ hattr hrd hρ

theorem Dim.elems_mono (d : Dim) : OptLe (d.elems I₂ ρ₂) (d.elems I₁ ρ₁) := by
  unfold Dim.elems
  exact OptLe.bind (Expr.evalInt_mono happ hattr hrd hρ _) fun lo =>
    OptLe.bind (Expr.evalInt_mono happ hattr hrd hρ _) fun hi =>
      OptLe.bind (Expr.evalInt_mono happ hattr hrd hρ _) fun step => .inr rfl

theorem Range.points_mono (r : Range) : OptLe (r.points I₂ ρ₂) (r.points I₁ ρ₁) := by
  unfold Range.points
  exact OptLe.bind (OptLe.mapM (Dim.elems_mono happ hattr hrd hρ) r.dims) fun _ => .inr rfl

theorem Subset.points_mono (S : Subset) : OptLe (S.points I₂ ρ₂) (S.points I₁ ρ₁) := by
  cases S with
  | range r => exact Range.points_mono happ hattr hrd hρ r
  | union rs =>
    simp only [Subset.points]
    exact OptLe.bind (OptLe.mapM (Range.points_mono happ hattr hrd hρ) rs) fun _ => .inr rfl

end PointsMono

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

/-- The memory `m₂` is `m₁` with its names renamed by `π`, except that some
cells may be uninitialized. -/
structure Mem.Sim (π : Equiv.Perm Name) (m₂ m₁ : Mem) : Prop where
  aliases : ∀ n, m₂.aliases (π n) = (m₁.aliases n).map (AliasBinding.ren π)
  cells : ∀ l, m₂.cells (l.ren π) = .undef ∨ m₂.cells (l.ren π) = m₁.cells l
  queues : ∀ l, m₂.queues (l.ren π) = m₁.queues l

namespace Mem.Sim

variable {π : Equiv.Perm Name} {m₂ m₁ : Mem}

theorem refl (m : Mem) : Mem.Sim 1 m m :=
  ⟨fun n => by cases h : m.aliases n <;> simp [h], fun _ => .inr rfl, fun _ => rfl⟩

theorem resolveFuel (h : Mem.Sim π m₂ m₁) : ∀ (k : Nat) (l : Loc),
    m₂.resolveFuel k (l.ren π) = (m₁.resolveFuel k l).ren π
  | 0, _ => rfl
  | k + 1, l => by
    simp only [Mem.resolveFuel, Loc.ren_name, h.aliases]
    cases hl : m₁.aliases l.name with
    | none => rfl
    | some b =>
      simp only [Option.map_some]
      rw [Mem.applyAlias_ren, resolveFuel h k]

theorem resolve (h : Mem.Sim π m₂ m₁) (l : Loc) : m₂.resolve (l.ren π) = (m₁.resolve l).ren π :=
  h.resolveFuel 64 l

theorem isAliased (h : Mem.Sim π m₂ m₁) (n : Name) : m₂.isAliased (π n) = m₁.isAliased n := by
  simp only [Mem.isAliased, h.aliases]
  cases m₁.aliases n <;> rfl

theorem read (h : Mem.Sim π m₂ m₁) (l : Loc) : m₂.read (l.ren π) = .undef ∨ m₂.read (l.ren π) = m₁.read l := by
  simp only [Mem.read, h.resolve]
  exact h.cells _

theorem cells_of_ne (h : Mem.Sim π m₂ m₁) (l : Loc) (hl : m₂.cells (l.ren π) ≠ .undef) :
    m₂.cells (l.ren π) = m₁.cells l := (h.cells l).resolve_left hl

/-- Updating the cell at a renamed location. -/
theorem updCell (h : Mem.Sim π m₂ m₁) (l : Loc) (v₂ v₁ : Val) (hv : v₂ = .undef ∨ v₂ = v₁) :
    Mem.Sim π { m₂ with cells := Function.update m₂.cells (l.ren π) v₂ }
      { m₁ with cells := Function.update m₁.cells l v₁ } := by
  refine ⟨h.aliases, fun l' => ?_, h.queues⟩
  by_cases hl : l' = l
  · subst hl; simpa using hv
  · have : l'.ren π ≠ l.ren π := fun e => hl ((Loc.ren_inj π).1 e)
    simp only [Function.update_of_ne this, Function.update_of_ne hl]
    exact h.cells l'

theorem updQueue (h : Mem.Sim π m₂ m₁) (l : Loc) (q : List Val) :
    Mem.Sim π { m₂ with queues := Function.update m₂.queues (l.ren π) q }
      { m₁ with queues := Function.update m₁.queues l q } := by
  refine ⟨h.aliases, h.cells, fun l' => ?_⟩
  by_cases hl : l' = l
  · subst hl; simp
  · have : l'.ren π ≠ l.ren π := fun e => hl ((Loc.ren_inj π).1 e)
    simp only [Function.update_of_ne this, Function.update_of_ne hl]
    exact h.queues l'

theorem write (h : Mem.Sim π m₂ m₁) (l : Loc) (v : Val) : Mem.Sim π (m₂.write (l.ren π) v) (m₁.write l v) := by
  simp only [Mem.write, h.resolve]
  exact h.updCell _ v v (.inr rfl)

theorem setAlias (h : Mem.Sim π m₂ m₁) (n : Name) (b : AliasBinding) :
    Mem.Sim π (m₂.setAlias (π n) (b.ren π)) (m₁.setAlias n b) := by
  refine ⟨fun n' => ?_, h.cells, h.queues⟩
  simp only [Mem.setAlias]
  by_cases hn : n' = n
  · subst hn; simp
  · have : π n' ≠ π n := fun e => hn (π.injective e)
    simp only [Function.update_of_ne this, Function.update_of_ne hn]
    exact h.aliases n'

theorem push (h : Mem.Sim π m₂ m₁) (l : Loc) (v : Val) : Mem.Sim π (m₂.push (l.ren π) v) (m₁.push l v) := by
  simp only [Mem.push, h.resolve, h.queues]
  exact h.updQueue _ _

theorem pushMany (l : Loc) :
    ∀ (vs : List Val) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ → Mem.Sim π (m₂.pushMany (l.ren π) vs) (m₁.pushMany l vs)
  | [], _, _, h => h
  | v :: vs, _, _, h => by
    simp only [Mem.pushMany, List.foldl_cons]
    exact pushMany l vs (h.push l v)

theorem setQueue (h : Mem.Sim π m₂ m₁) (l : Loc) (q : List Val) :
    Mem.Sim π (m₂.setQueue (l.ren π) q) (m₁.setQueue l q) := h.updQueue l q

theorem alloc (h : Mem.Sim π m₂ m₁) (n : Name) : Mem.Sim π (m₂.alloc (π n)) (m₁.alloc n) := by
  refine ⟨h.aliases, fun l => ?_, fun l => ?_⟩
  · simp only [Mem.alloc, Loc.ren_name, (Equiv.injective π).eq_iff]
    split
    · exact .inl rfl
    · exact h.cells l
  · simp only [Mem.alloc, Loc.ren_name, (Equiv.injective π).eq_iff]
    split
    · rfl
    · exact h.queues l

theorem allocCells (h : Mem.Sim π m₂ m₁) (n : Name) : Mem.Sim π (m₂.allocCells (π n)) (m₁.allocCells n) := by
  refine ⟨h.aliases, fun l => ?_, h.queues⟩
  simp only [Mem.allocCells, Loc.ren_name, (Equiv.injective π).eq_iff]
  split
  · exact .inl rfl
  · exact h.cells l

/-- Allocating cells only on the simulating side keeps the simulation. -/
theorem allocCells_left (h : Mem.Sim π m₂ m₁) (n : Name) : Mem.Sim π (m₂.allocCells n) m₁ := by
  refine ⟨h.aliases, fun l => ?_, h.queues⟩
  simp only [Mem.allocCells]
  split
  · exact .inl rfl
  · exact h.cells l

theorem allocAll :
    ∀ (ns : List Name) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ → Mem.Sim π (m₂.allocAll (ns.map π)) (m₁.allocAll ns)
  | [], _, _, h => h
  | n :: ns, _, _, h => by
    simp only [Mem.allocAll, List.map_cons, List.foldl_cons]
    exact allocAll ns (h.alloc n)

theorem writeChunk (h : Mem.Sim π m₂ m₁) (n : Name) :
    ∀ vs : List Val, Mem.Sim π (m₂.writeChunk (π n) vs) (m₁.writeChunk n vs) := by
  intro vs
  unfold Mem.writeChunk
  suffices ∀ (xs : List (Val × Nat)) (a₂ a₁ : Mem), Mem.Sim π a₂ a₁ →
      Mem.Sim π (xs.foldl (fun acc (v, k) => { acc with cells := Function.update acc.cells (Loc.elem (π n) k) v }) a₂)
        (xs.foldl (fun acc (v, k) => { acc with cells := Function.update acc.cells (Loc.elem n k) v }) a₁) from
    this _ _ _ h
  intro xs
  induction xs with
  | nil => intro _ _ h; exact h
  | cons x xs ih =>
    intro a₂ a₁ h
    obtain ⟨v, k⟩ := x
    simp only [List.foldl_cons]
    exact ih _ _ (by rw [← Loc.ren_elem]; exact h.updCell _ v v (.inr rfl))

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

namespace Merge

variable {π : Equiv.Perm Name} {w : WcrSem}

theorem applyWrites_sim : ∀ (ws : List (Loc × Val)) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ →
    Mem.Sim π (applyWrites m₂ (ws.map fun (l, v) => (l.ren π, v))) (applyWrites m₁ ws)
  | [], _, _, h => h
  | (l, v) :: ws, _, _, h => by
    simp only [applyWrites, List.map_cons, List.foldl_cons] at *
    exact applyWrites_sim ws (h.updCell l v v (.inr rfl))

theorem applyWcrs_sim (hu : ∀ op v, w op .undef v = .undef) :
    ∀ (cs : List (Loc × WcrOp × Val)) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ →
      Mem.Sim π (applyWcrs w m₂ (cs.map fun (l, op, v) => (l.ren π, op, v))) (applyWcrs w m₁ cs)
  | [], _, _, h => h
  | (l, op, v) :: cs, _, _, h => by
    simp only [applyWcrs, List.map_cons, List.foldl_cons] at *
    refine applyWcrs_sim hu cs (h.updCell l _ _ ?_)
    rcases h.cells l with h₁ | h₁
    · rw [h₁, hu]; exact .inl rfl
    · rw [h₁]; exact .inr rfl

theorem applyAliases_sim : ∀ (as : List (Name × AliasBinding)) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ →
    Mem.Sim π (applyAliases m₂ (as.map fun (n, b) => (π n, b.ren π))) (applyAliases m₁ as)
  | [], _, _, h => h
  | (n, b) :: as, _, _, h => by
    simp only [applyAliases, List.map_cons, List.foldl_cons] at *
    exact applyAliases_sim as (h.setAlias n b)

theorem applyQueues_sim : ∀ (qs : List (Loc × QueueEffect)) {m₂ m₁ : Mem}, Mem.Sim π m₂ m₁ →
    Mem.Sim π (applyQueues m₂ (qs.map fun (l, e) => (l.ren π, e))) (applyQueues m₁ qs)
  | [], _, _, h => h
  | (l, e) :: qs, _, _, h => by
    simp only [applyQueues, List.map_cons, List.foldl_cons] at *
    refine applyQueues_sim qs ?_
    cases e with
    | pushOnly vs =>
      simp only
      rw [h.queues l]
      exact h.updQueue l _
    | exclusive final => exact h.updQueue l final

end Merge

/-- **Merging renamed logs onto a simulating memory**: both merges fail, or
both succeed with simulating results. -/
theorem merge_sim {π : Equiv.Perm Name} {w : WcrSem} (hu : ∀ op v, w op .undef v = .undef) {m₂ m₁ : Mem}
    (h : Mem.Sim π m₂ m₁) (gs : List Log) :
    (∃ e₂ e₁, merge w m₂ (gs.map (Log.ren π)) = .error e₂ ∧ merge w m₁ gs = .error e₁) ∨
      ∃ a₂ a₁, merge w m₂ (gs.map (Log.ren π)) = .ok a₂ ∧ merge w m₁ gs = .ok a₁ ∧ Mem.Sim π a₂ a₁ := by
  unfold merge
  rw [Log.pairwiseFree_ren]
  split
  · refine .inr ⟨_, _, rfl, rfl, ?_⟩
    rw [← Log.ren_unionAll]
    exact Merge.applyQueues_sim _ (Merge.applyAliases_sim _ (Merge.applyWcrs_sim hu _
      (Merge.applyWrites_sim _ h)))
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

/-! ## Renamed contexts and simulating states -/

/-- The context with memory names renamed by `π`. -/
def Ctx.ren (π : Equiv.Perm Name) (ctx : Ctx) : Ctx := { ctx with cname := π ∘ ctx.cname, scope := π ∘ ctx.scope }

/-- A dataflow state simulating another: simulating memory, renamed log,
the same code-edge values. -/
structure St.Sim (π : Equiv.Perm Name) (a₂ a₁ : St) : Prop where
  mem : Mem.Sim π a₂.mem a₁.mem
  log : a₂.log = a₁.log.ren π
  vals : a₂.vals = a₁.vals

/-- A control-flow state simulating another. -/
structure CSt.Sim (π : Equiv.Perm Name) (c₂ c₁ : CSt) : Prop where
  ρ : c₂.ρ = c₁.ρ
  mem : Mem.Sim π c₂.mem c₁.mem
  log : c₂.log = c₁.log.ren π

/-- A result with a state: the same value, simulating states. -/
def St.SimR {α : Type} (π : Equiv.Perm Name) (x₂ x₁ : α × St) : Prop := x₂.1 = x₁.1 ∧ St.Sim π x₂.2 x₁.2

namespace Ctx

variable (π : Equiv.Perm Name) (ctx : Ctx)

@[simp] theorem ren_I : (ctx.ren π).I = ctx.I := rfl
@[simp] theorem ren_g : (ctx.ren π).g = ctx.g := rfl
@[simp] theorem ren_cname (n : Name) : (ctx.ren π).cname n = π (ctx.cname n) := rfl
@[simp] theorem ren_desc? (n : Name) : (ctx.ren π).desc? n = ctx.desc? n := rfl
@[simp] theorem ren_isStream (n : Name) : (ctx.ren π).isStream n = ctx.isStream n := rfl
@[simp] theorem ren_isView (n : Name) : (ctx.ren π).isView n = ctx.isView n := rfl
@[simp] theorem ren_isReference (n : Name) : (ctx.ren π).isReference n = ctx.isReference n := rfl

theorem ren_one : ctx.ren 1 = ctx := rfl

variable {π} {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (ρ : Valuation)
include hm

theorem addr_ren (n : Name) (idx : Idx) :
    (ctx.ren π).addr ρ m₂ n idx = (ctx.addr ρ m₁ n idx).map (Loc.ren π) := by
  unfold Ctx.addr
  simp only [ren_desc?, ren_cname, hm.isAliased]
  cases ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    rw [show (ctx.ren π).elemOffset ρ d idx = ctx.elemOffset ρ d idx from rfl]
    split
    · rfl
    · cases ctx.elemOffset ρ d idx <;> rfl

theorem loc_ren (n : Name) (idx : Idx) (oob : Bool) :
    (ctx.ren π).loc ρ m₂ n idx oob = (ctx.loc ρ m₁ n idx oob).map (Loc.ren π) := by
  unfold Ctx.loc
  rw [ctx.addr_ren hm]
  cases ctx.addr ρ m₁ n idx with
  | err e => rfl
  | ok l =>
    rw [show (ctx.ren π).checkBounds ρ n idx oob = ctx.checkBounds ρ n idx oob from rfl]
    cases ctx.checkBounds ρ n idx oob <;> rfl

theorem readData_mono (n : Name) (idx : Idx) :
    OptLe (((ctx.ren π).exprInterp ρ m₂).readData n idx) ((ctx.exprInterp ρ m₁).readData n idx) := by
  simp only [Ctx.exprInterp, ctx.loc_ren hm]
  cases ctx.loc ρ m₁ n idx false with
  | err e => exact .inr rfl
  | ok l =>
    simp only [Res.map]
    rcases hm.read l with h | h <;> rw [h]
    · exact .inl rfl
    · exact .inr rfl

theorem val_mono (x : Sym) : OptLe ((ctx.ren π).val ρ m₂ x) (ctx.val ρ m₁ x) := by
  unfold Ctx.val
  cases ρ x with
  | some v => exact .inr rfl
  | none =>
    simp only [ren_desc?]
    cases ctx.desc? x with
    | none => exact .inr rfl
    | some d =>
      simp only
      split
      · rw [ctx.loc_ren hm]
        cases ctx.loc ρ m₁ x [] false with
        | err e => exact .inr rfl
        | ok l =>
          simp only [Res.map]
          rcases hm.read l with h | h <;> rw [h]
          · exact .inl rfl
          · exact .inr rfl
      · exact .inr rfl

theorem eval_mono (e : Expr) : OptLe ((ctx.ren π).eval ρ m₂ e) (ctx.eval ρ m₁ e) :=
  Expr.eval_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) e

theorem evalR_sim (e : Expr) : ResSim Eq ((ctx.ren π).evalR ρ m₂ e) (ctx.evalR ρ m₁ e) :=
  ResSim.ofOption rfl (ctx.eval_mono hm ρ e)

theorem evalIntR_sim (e : Expr) : ResSim Eq ((ctx.ren π).evalIntR ρ m₂ e) (ctx.evalIntR ρ m₁ e) :=
  ResSim.ofOption rfl (Expr.evalInt_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) e)

theorem evalBoolR_sim (e : Expr) : ResSim Eq ((ctx.ren π).evalBoolR ρ m₂ e) (ctx.evalBoolR ρ m₁ e) :=
  ResSim.ofOption rfl (Expr.evalBool_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) e)

theorem evalInt_mono (e : Expr) : OptLe ((ctx.ren π).evalInt ρ m₂ e) (ctx.evalInt ρ m₁ e) :=
  Expr.evalInt_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) e

theorem pointsOf_sim (n : Name) (sub : Option Subset) :
    ResSim Eq ((ctx.ren π).pointsOf ρ m₂ n sub) (ctx.pointsOf ρ m₁ n sub) := by
  unfold Ctx.pointsOf
  simp only [ren_desc?]
  cases ctx.desc? n with
  | none => exact ResSim.err _
  | some d =>
    exact ResSim.ofOption rfl (Subset.points_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) _)

theorem rangePoints_sim (r : Range) : ResSim Eq ((ctx.ren π).rangePoints ρ m₂ r) (ctx.rangePoints ρ m₁ r) :=
  ResSim.ofOption rfl (Range.points_mono (I₂ := (ctx.ren π).exprInterp ρ m₂) (I₁ := ctx.exprInterp ρ m₁) rfl rfl (ctx.readData_mono hm ρ) (ctx.val_mono hm ρ) r)

end Ctx

/-! ## Primitive operations on simulating states -/

namespace St.Sim

variable {π : Equiv.Perm Name} {st₂ st₁ : St}

theorem refl (st : St) : St.Sim 1 st st := ⟨Mem.Sim.refl _, (Log.ren_one _).symm, rfl⟩

theorem logged (h : St.Sim π st₂ st₁) (g : Log) : St.Sim π (st₂.logged (g.ren π)) (st₁.logged g) :=
  ⟨h.mem, by simp only [St.logged, h.log, Log.ren_seq], h.vals⟩

theorem withMem (h : St.Sim π st₂ st₁) {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) :
    St.Sim π { st₂ with mem := m₂ } { st₁ with mem := m₁ } := ⟨hm, h.log, h.vals⟩

theorem forNode (h : St.Sim π st₂ st₁) : St.Sim π st₂.forNode st₁.forNode := ⟨h.mem, rfl, h.vals⟩

theorem afterNode {a₂ a₁ : St} (h : St.Sim π st₂ st₁) (ha : St.Sim π a₂ a₁) :
    St.Sim π (st₂.afterNode a₂) (st₁.afterNode a₁) :=
  ⟨ha.mem, by simp only [St.afterNode, h.log, ha.log, Log.ren_seq], ha.vals⟩

theorem setVal (h : St.Sim π st₂ st₁) (key : NodeId × Conn) (v : Val) :
    St.Sim π (st₂.setVal key v) (st₁.setVal key v) := ⟨h.mem, h.log, by simp [St.setVal, h.vals]⟩

theorem val? (h : St.Sim π st₂ st₁) (key : NodeId × Conn) : st₂.val? key = st₁.val? key := by
  simp [St.val?, h.vals]

theorem bindAlias (h : St.Sim π st₂ st₁) (n : Name) (b : AliasBinding) :
    St.Sim π (st₂.bindAlias (π n) (b.ren π)) (st₁.bindAlias n b) :=
  ⟨h.mem.setAlias n b, by simp only [St.bindAlias, h.log, Log.ren_seq]; rfl, h.vals⟩

end St.Sim

namespace St

variable {π : Equiv.Perm Name} (ctx : Ctx) (ρ : Valuation) {st₂ st₁ : St} (h : St.Sim π st₂ st₁)
include h

theorem read_sim (n : Name) (idx : Idx) (oob : Bool) :
    ResSim (St.SimR π) (st₂.read (ctx.ren π) ρ n idx oob) (st₁.read ctx ρ n idx oob) := by
  unfold St.read
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind]
    rcases h.mem.read l with hu | heq
    · rw [hu]; exact ResSim.ub rfl _
    · rw [heq]
      split
      · exact ResSim.err _
      · refine ResSim.ok ⟨rfl, ?_⟩
        have := h.logged { reads := [st₁.mem.resolve l] }
        simpa [Log.ren, h.mem.resolve] using this

theorem readMany_sim (n : Name) (pts : List Idx) (oob : Bool) :
    ResSim (St.SimR π) (st₂.readMany (ctx.ren π) ρ n pts oob) (st₁.readMany ctx ρ n pts oob) := by
  unfold St.readMany
  refine ResSim.foldlM (R := St.SimR π) (fun p a b hab => ?_) pts ⟨rfl, h⟩
  obtain ⟨acc₂, a₂⟩ := a
  obtain ⟨acc₁, a₁⟩ := b
  obtain ⟨hacc, ha⟩ := hab
  simp only at hacc ha ⊢
  subst hacc
  exact ResSim.bind (read_sim ctx ρ ha n p oob) fun x y ⟨hv, hs⟩ => by
    obtain ⟨v₂, s₂⟩ := x
    obtain ⟨v₁, s₁⟩ := y
    simp only at hv hs
    subst hv
    exact ResSim.ok ⟨rfl, hs⟩

theorem write_sim (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    ResSim (St.Sim π) (st₂.write (ctx.ren π) ρ n idx oob v) (st₁.write ctx ρ n idx oob v) := by
  unfold St.write
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq]
    refine ResSim.ok ⟨h.mem.write l v, ?_, h.vals⟩
    simp only [h.log, Log.ren_seq, h.mem.resolve]
    rfl

theorem wcr_sim (hu : ∀ op v, ctx.I.wcr op .undef v = .undef) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp)
    (v : Val) : ResSim (St.Sim π) (st₂.wcr (ctx.ren π) ρ n idx oob op v) (st₁.wcr ctx ρ n idx oob op v) := by
  unfold St.wcr
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx oob with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq, Ctx.ren_I]
    refine ResSim.ok ⟨?_, ?_, h.vals⟩
    · simp only [Mem.write, h.mem.resolve]
      refine h.mem.updCell _ _ _ ?_
      rcases h.mem.read l with h₁ | h₁ <;> rw [h₁]
      · rw [hu]; exact .inl rfl
      · exact .inr rfl
    · simp only [h.log, Log.ren_seq, h.mem.resolve]
      rfl

theorem store_sim (hu : ∀ op v, ctx.I.wcr op .undef v = .undef) (n : Name) (idx : Idx) (oob : Bool)
    (w : Option WcrOp) (v : Val) :
    ResSim (St.Sim π) (st₂.store (ctx.ren π) ρ n idx oob w v) (st₁.store ctx ρ n idx oob w v) := by
  cases w with
  | none => exact write_sim ctx ρ h n idx oob v
  | some op => exact wcr_sim ctx ρ h hu n idx oob op v

theorem queue_sim (n : Name) (idx : Idx) :
    ResSim Eq (st₂.queue (ctx.ren π) ρ n idx) (st₁.queue ctx ρ n idx) := by
  unfold St.queue
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq, h.mem.resolve, h.mem.queues]
    exact ResSim.ok rfl

theorem push_sim (n : Name) (idx : Idx) (vs : List Val) :
    ResSim (St.Sim π) (st₂.push (ctx.ren π) ρ n idx vs) (st₁.push ctx ρ n idx vs) := by
  unfold St.push
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq]
    split
    · exact ResSim.ok h
    · refine ResSim.ok ⟨h.mem.pushMany l vs, ?_, h.vals⟩
      simp only [h.log, Log.ren_seq, h.mem.resolve]
      rfl

theorem popN_sim (n : Name) (idx : Idx) (k : Nat) :
    ResSim (St.SimR π) (st₂.popN (ctx.ren π) ρ n idx k) (st₁.popN ctx ρ n idx k) := by
  unfold St.popN
  rw [ctx.loc_ren h.mem]
  cases ctx.loc ρ st₁.mem n idx false with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq, h.mem.resolve, h.mem.queues]
    split
    · exact ResSim.ok ⟨rfl, h⟩
    · split
      · exact ResSim.ub rfl _
      · refine ResSim.ok ⟨rfl, ⟨h.mem.updQueue _ _, ?_, h.vals⟩⟩
        simp only [h.log, Log.ren_seq]
        rfl

end St

/-! ## Leaf functions -/

section Leaf

variable {π : Equiv.Perm Name} (ctx : Ctx) (ρ : Valuation) (hu : ∀ op v, ctx.I.wcr op .undef v = .undef)

theorem streamPoint_sim {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (d : Name) (sub : Option Subset) :
    ResSim Eq (streamPoint (ctx.ren π) ρ m₂ d sub) (streamPoint ctx ρ m₁ d sub) := by
  unfold streamPoint
  exact ResSim.bindEq (ctx.pointsOf_sim hm ρ d sub) fun pts => ResSim.of_eq fun _ => rfl

theorem lowerCorner_sim {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (n : Name) (sub : Option Subset) :
    ResSim Eq ((ctx.ren π).lowerCorner ρ m₂ n sub) (ctx.lowerCorner ρ m₁ n sub) := by
  unfold Ctx.lowerCorner
  refine ResSim.bindEq (ctx.pointsOf_sim hm ρ n sub) fun _ => ?_
  cases sub with
  | none => simp only [Ctx.ren_desc?]; exact ResSim.of_eq fun _ => rfl
  | some S =>
    simp only
    split
    · exact ResSim.err _
    · exact ResSim.mapM (fun (dim : Dim) => ResSim.ofOption rfl (ctx.evalInt_mono hm ρ dim.lo)) _

theorem readScalarData_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset) (oob : Bool) :
    ResSim (St.SimR π) (readScalarData (ctx.ren π) ρ d sub oob st₂) (readScalarData ctx ρ d sub oob st₁) := by
  unfold readScalarData
  refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ d sub) fun pts => ?_
  split
  · refine ResSim.ite (c := ctx.isStream d) (fun _ => ?_) (fun _ => ?_)
    · exact ResSim.bind (St.popN_sim ctx ρ h d _ 1) fun x y hxy => by
        obtain ⟨hv, hs⟩ := hxy
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
    · exact ResSim.bind (St.read_sim ctx ρ h d _ oob) fun x y hxy => by
        obtain ⟨hv, hs⟩ := hxy
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
  · exact ResSim.err _

theorem readEdgeScalar_sim (s : State) (e : DEdge) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.SimR π) (readEdgeScalar (ctx.ren π) ρ s e st₂) (readEdgeScalar ctx ρ s e st₁) := by
  unfold readEdgeScalar
  split
  · rw [h.val?]; exact ResSim.ok ⟨rfl, h⟩
  · exact readScalarData_sim ctx ρ h _ _ _
  · exact ResSim.err _

include hu in
theorem copyData_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (sd : Name) (ssub : Option Subset) (dd : Name)
    (dsub : Option Subset) (w : Option WcrOp) (oob : Bool) :
    ResSim (St.Sim π) (copyData (ctx.ren π) ρ st₂ sd ssub dd dsub w oob) (copyData ctx ρ st₁ sd ssub dd dsub w oob) := by
  unfold copyData
  simp only [Ctx.ren_isStream]
  split
  · exact ResSim.err _
  · refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ dd dsub) fun dpts => ?_
    refine ResSim.bindEq (streamPoint_sim ctx ρ h.mem sd ssub) fun sp => ?_
    refine ResSim.bind (St.popN_sim ctx ρ h sd sp dpts.length) fun x y ⟨hv, hs⟩ => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    simp only at hv hs ⊢
    subst hv
    exact ResSim.foldlM (fun qv b₂ b₁ hb => St.store_sim ctx ρ hb hu dd qv.1 oob w qv.2) _ hs
  · refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ sd ssub) fun spts => ?_
    refine ResSim.bindEq (streamPoint_sim ctx ρ h.mem dd dsub) fun dp => ?_
    refine ResSim.bind (St.readMany_sim ctx ρ h sd spts oob) fun x y ⟨hv, hs⟩ => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    simp only at hv hs ⊢
    subst hv
    exact St.push_sim ctx ρ hs dd dp vs₂
  · refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ sd ssub) fun spts => ?_
    refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ dd dsub) fun dpts => ?_
    refine ResSim.foldlM (fun pq b₂ b₁ hb => ?_) _ h
    exact ResSim.bind (St.read_sim ctx ρ hb sd pq.1 oob) fun x y ⟨hv, hs⟩ => by
      obtain ⟨v₂, a₂⟩ := x
      obtain ⟨v₁, a₁⟩ := y
      simp only at hv hs ⊢
      subst hv
      exact St.store_sim ctx ρ hs hu dd pq.2 oob w v₂

include hu in
theorem execCopy_sim (s : State) (e : DEdge) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (execCopy (ctx.ren π) ρ s e st₂) (execCopy ctx ρ s e st₁) := by
  unfold execCopy
  split
  · exact ResSim.ok h
  · split
    · exact copyData_sim ctx ρ hu h _ _ _ _ _ _
    · exact ResSim.err _

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

theorem bindInData_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset) (oob : Bool) :
    ResSim (St.SimR π) (bindInData (ctx.ren π) ρ d sub oob st₂) (bindInData ctx ρ d sub oob st₁) := by
  unfold bindInData
  refine ResSim.ite (c := ctx.isStream d) (fun _ => ?_) (fun _ => ?_)
  · refine ResSim.bindEq (streamPoint_sim ctx ρ h.mem d sub) fun p => ?_
    exact ResSim.bindEq (St.queue_sim ctx ρ h d p) fun q => ResSim.ok ⟨rfl, h⟩
  · refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ d sub) fun pts => ?_
    split
    · exact ResSim.bind (St.read_sim ctx ρ h d _ oob) fun x y hxy => by
        obtain ⟨v₂, a₂⟩ := x
        obtain ⟨v₁, a₁⟩ := y
        obtain ⟨hv, hs⟩ := hxy
        simp only at hv hs ⊢
        subst hv
        exact ResSim.ok ⟨rfl, hs⟩
    · refine ResSim.bindEq (lowerCorner_sim ctx ρ h.mem d sub) fun lo => ?_
      cases hr₂ : st₂.readMany (ctx.ren π) ρ d pts oob with
      | err e =>
        have := St.readMany_sim ctx ρ h d pts oob
        rw [hr₂] at this
        obtain ⟨_, -, hu'⟩ := ResSim.err_inv' this rfl
        rcases hu' with hu' | h₁
        · exact ResSim.ub hu' _
        · rw [h₁]; exact ResSim.err _
      | ok x₂ =>
        obtain ⟨vs₂, a₂⟩ := x₂
        have hs := St.readMany_sim ctx ρ h d pts oob
        rw [hr₂] at hs
        rcases hs with ⟨e, he, -⟩ | ⟨e, -, he⟩ | ⟨a, b, ha, hb, hab⟩
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
            · next hc =>
              rw [ctx.loc_ren h.mem]
              cases hl : ctx.loc ρ st₁.mem d (List.zipWith (· + ·) lo rel) oob with
              | err e => rfl
              | ok l =>
                simp only [Res.map]
                have hdef := hd₂ _ (by simpa using hc) (l.ren π) (by rw [ctx.loc_ren h.mem, hl]; rfl)
                rcases h.mem.read l with h₁ | h₁
                · rw [h₁] at hdef; cases hdef
                · rw [h₁]
            · rfl
          · rfl

theorem bindIn_sim (s : State) (e : DEdge) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.SimR π) (bindIn (ctx.ren π) ρ s e st₂) (bindIn ctx ρ s e st₁) := by
  unfold bindIn
  split
  · rw [h.val?]; exact ResSim.ok ⟨rfl, h⟩
  · exact ResSim.bind (bindInData_sim ctx ρ h _ _ _) fun x y hxy => by
      obtain ⟨c₂, a₂⟩ := x
      obtain ⟨c₁, a₁⟩ := y
      obtain ⟨hv, hs⟩ := hxy
      simp only at hv hs ⊢
      subst hv
      exact ResSim.ok ⟨rfl, hs⟩
  · exact ResSim.err _

theorem bindInputs_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.SimR π) (bindInputs (ctx.ren π) ρ s n st₂) (bindInputs ctx ρ s n st₁) := by
  unfold bindInputs
  refine ResSim.bind (S := St.SimR π) (R := St.SimR π)
    (ResSim.foldlM (R := St.SimR π) (fun e a b hab => ?_) _ ⟨rfl, h⟩) fun x y hxy => ?_
  · obtain ⟨acc₂, a₂⟩ := a
    obtain ⟨acc₁, a₁⟩ := b
    obtain ⟨hacc, ha⟩ := hab
    simp only at hacc ha ⊢
    subst hacc
    split
    · exact ResSim.ok ⟨rfl, ha⟩
    · refine ResSim.bind (bindIn_sim ctx ρ s e ha) fun x y hxy => ?_
      obtain ⟨c₂, a₂'⟩ := x
      obtain ⟨c₁, a₁'⟩ := y
      obtain ⟨hv, hs⟩ := hxy
      simp only at hv hs ⊢
      subst hv
      split
      · exact ResSim.ok ⟨rfl, hs⟩
      · exact ResSim.ok ⟨rfl, hs⟩
  · obtain ⟨bs₂, a₂⟩ := x
    obtain ⟨bs₁, a₁⟩ := y
    obtain ⟨hv, hs⟩ := hxy
    simp only at hv hs ⊢
    subst hv
    exact ResSim.ok ⟨rfl, hs⟩

include hu

theorem storeScalar_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset)
    (w : Option WcrOp) (oob : Bool) (v : Val) :
    ResSim (St.Sim π) (storeScalar (ctx.ren π) ρ st₂ d sub w oob v) (storeScalar ctx ρ st₁ d sub w oob v) := by
  unfold storeScalar
  refine ResSim.ite (c := ctx.isStream d) (fun _ => ?_) (fun _ => ?_)
  · exact ResSim.bindEq (streamPoint_sim ctx ρ h.mem d sub) fun p => St.push_sim ctx ρ h d p [v]
  · refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ d sub) fun pts => ?_
    split
    · exact St.store_sim ctx ρ h hu d _ oob w v
    · exact ResSim.err _
    · exact ResSim.err _

theorem applyScalar_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (c : Conn) (v : Val) :
    ResSim (St.Sim π) (applyScalar (ctx.ren π) ρ s n st₂ c v) (applyScalar ctx ρ s n st₁ c v) := by
  unfold applyScalar
  refine ResSim.foldlM (fun e a b hab => ?_) _ h
  split
  · exact ResSim.ok hab
  · split
    · exact ResSim.ok (hab.setVal _ v)
    · exact storeScalar_sim ctx ρ hu hab _ _ _ _ v
    · exact ResSim.err _

theorem storeArrayWrite_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset)
    (w : Option WcrOp) (oob : Bool) (rel : Idx) (v : Val) :
    ResSim (St.Sim π) (storeArrayWrite (ctx.ren π) ρ st₂ d sub w oob rel v)
      (storeArrayWrite ctx ρ st₁ d sub w oob rel v) := by
  unfold storeArrayWrite
  refine ResSim.ite (c := ctx.isStream d) (fun _ => ResSim.err _) (fun _ => ?_)
  refine ResSim.bindEq (lowerCorner_sim ctx ρ h.mem d sub) fun lo => ?_
  by_cases hl : rel.length = lo.length
  · rw [if_pos hl, if_pos hl]
    refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ d sub) fun pts => ?_
    by_cases hc : pts.contains (List.zipWith (· + ·) lo rel) = true
    · rw [if_pos hc, if_pos hc]
      exact St.store_sim ctx ρ h hu d _ oob w v
    · rw [if_neg hc, if_neg hc]
      exact ResSim.err _
  · rw [if_neg hl, if_neg hl]
    exact ResSim.err _

theorem applyArrayWrite_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (c : Conn) (rel : Idx)
    (v : Val) : ResSim (St.Sim π) (applyArrayWrite (ctx.ren π) ρ s n st₂ c rel v) (applyArrayWrite ctx ρ s n st₁ c rel v) := by
  unfold applyArrayWrite
  refine ResSim.foldlM (fun e a b hab => ?_) _ h
  split
  · exact ResSim.ok hab
  · split
    · exact storeArrayWrite_sim ctx ρ hu hab _ _ _ _ rel v
    · exact ResSim.err _

omit hu in
theorem pushTo_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset) (vs : List Val) :
    ResSim (St.Sim π) (pushTo (ctx.ren π) ρ st₂ d sub vs) (pushTo ctx ρ st₁ d sub vs) := by
  unfold pushTo
  refine ResSim.ite (c := ctx.isStream d) (fun _ => ?_) (fun _ => ResSim.err _)
  exact ResSim.bindEq (streamPoint_sim ctx ρ h.mem d sub) fun p => St.push_sim ctx ρ h d p vs

omit hu in
theorem applyPush_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (c : Conn) (vs : List Val) :
    ResSim (St.Sim π) (applyPush (ctx.ren π) ρ s n st₂ c vs) (applyPush ctx ρ s n st₁ c vs) := by
  unfold applyPush
  refine ResSim.foldlM (fun e a b hab => ?_) _ h
  split
  · exact pushTo_sim ctx ρ hab _ _ vs
  · exact ResSim.err _

omit hu in
theorem popFrom_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d : Name) (sub : Option Subset) (k : Nat) :
    ResSim (St.Sim π) (popFrom (ctx.ren π) ρ st₂ d sub k) (popFrom ctx ρ st₁ d sub k) := by
  unfold popFrom
  refine ResSim.ite (c := ctx.isStream d) (fun _ => ?_) (fun _ => ResSim.err _)
  refine ResSim.bindEq (streamPoint_sim ctx ρ h.mem d sub) fun p => ?_
  exact ResSim.bind (St.popN_sim ctx ρ h d p k) fun x y ⟨_, hs⟩ => ResSim.ok hs

omit hu in
theorem applyPop_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (c : Conn) (k : Nat) :
    ResSim (St.Sim π) (applyPop (ctx.ren π) ρ s n st₂ c k) (applyPop ctx ρ s n st₁ c k) := by
  unfold applyPop
  split
  · exact ResSim.err _
  · split
    · exact popFrom_sim ctx ρ h _ _ k
    · exact ResSim.err _

theorem applyOut_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (out : TaskletOut) :
    ResSim (St.Sim π) (applyOut (ctx.ren π) ρ s n st₂ out) (applyOut ctx ρ s n st₁ out) := by
  unfold applyOut
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π) (fun x a b hab => by
    obtain ⟨c, k⟩ := x; exact applyPop_sim ctx ρ s n hab c k) _ h) fun a b hab => ?_
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π) (fun x a b hab => by
    obtain ⟨c, v⟩ := x; exact applyScalar_sim ctx ρ hu s n hab c v) _ hab) fun a b hab => ?_
  refine ResSim.bind (ResSim.foldlM (R := St.Sim π) (fun x a b hab => by
    obtain ⟨c, rel, v⟩ := x; exact applyArrayWrite_sim ctx ρ hu s n hab c rel v) _ hab) fun a b hab => ?_
  exact ResSim.foldlM (R := St.Sim π) (fun x a b hab => by
    obtain ⟨c, vs⟩ := x; exact applyPush_sim ctx ρ s n hab c vs) _ hab

theorem execCode_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (ue : ErrReason) :
    ResSim (St.Sim π) (execCode (ctx.ren π) ρ s n st₂ run ue) (execCode ctx ρ s n st₁ run ue) := by
  unfold execCode
  split
  · exact ResSim.err _
  · refine ResSim.bind (bindInputs_sim ctx ρ s n h) fun x y hxy => ?_
    obtain ⟨f₂, a₂⟩ := x
    obtain ⟨f₁, a₁⟩ := y
    obtain ⟨hv, hs⟩ := hxy
    simp only at hv hs ⊢
    subst hv
    split
    · exact ResSim.err _
    · exact applyOut_sim ctx ρ hu s n hs _

theorem reduceData_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (props : ReduceProps) (sd : Name)
    (ssub : Option Subset) (dd : Name) (dsub : Option Subset) (outWcr : Option WcrOp) (soob doob : Bool) :
    ResSim (St.Sim π) (reduceData (ctx.ren π) ρ st₂ props sd ssub dd dsub outWcr soob doob)
      (reduceData ctx ρ st₁ props sd ssub dd dsub outWcr soob doob) := by
  unfold reduceData
  refine ResSim.ite (c := ctx.isStream sd || ctx.isStream dd) (fun _ => ResSim.err _) (fun _ => ?_)
  refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ sd ssub) fun spts => ?_
  refine ResSim.bindEq (ctx.pointsOf_sim h.mem ρ dd dsub) fun dpts => ?_
  simp only
  split
  · exact ResSim.err _
  · refine ResSim.bind (St.readMany_sim ctx ρ h sd spts soob) fun x y hxy => ?_
    obtain ⟨vs₂, a₂⟩ := x
    obtain ⟨vs₁, a₁⟩ := y
    obtain ⟨hv, hs⟩ := hxy
    simp only at hv hs ⊢
    subst hv
    refine ResSim.foldlM (fun gq b₂ b₁ hb => ?_) _ hs
    obtain ⟨g, q⟩ := gq
    simp only
    split
    · exact ResSim.bindEq (ctx.evalR_sim hb.mem ρ _) fun i => St.store_sim ctx ρ hb hu dd q doob outWcr _
    · split
      · exact ResSim.ok hb
      · exact St.wcr_sim ctx ρ hb hu dd q doob _ _

theorem execReduce_sim (s : State) (n : DNode) (props : ReduceProps) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (execReduce (ctx.ren π) ρ s n props st₂) (execReduce ctx ρ s n props st₁) := by
  unfold execReduce
  dsimp only
  split
  · split
    · exact reduceData_sim ctx ρ hu h props _ _ _ _ _ _ _
    · exact ResSim.err _
  · exact ResSim.err _

omit hu in
theorem aliasTo_sim {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (d base : Name) (sub : Option Subset) :
    ResSim (St.Sim π) (aliasTo (ctx.ren π) ρ st₂ d base sub) (aliasTo ctx ρ st₁ d base sub) := by
  unfold aliasTo
  refine ResSim.ite (c := ctx.reinterprets d base) (fun _ => ResSim.err _) (fun _ => ?_)
  refine ResSim.bindEq (lowerCorner_sim ctx ρ h.mem base sub) fun lo => ?_
  rw [ctx.addr_ren h.mem]
  cases ctx.addr ρ st₁.mem base lo with
  | err e => exact ResSim.err e
  | ok l =>
    simp only [Res.map, Res.ok_bind, Res.pure_eq, Ctx.ren_cname]
    exact ResSim.ok (h.bindAlias (ctx.cname d) { base := l.name, offset := l.offset })

omit hu in
theorem bindView_sim (s : State) (v : DNode) (d : Name) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (bindView (ctx.ren π) ρ s v d st₂) (bindView ctx ρ s v d st₁) := by
  unfold bindView
  split
  · exact ResSim.err _
  · simp only
    split
    · exact ResSim.err _
    · exact aliasTo_sim ctx ρ h _ _ _

omit hu in
theorem bindViews_sim (s : State) (ids : List NodeId) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (bindViews (ctx.ren π) ρ s ids st₂) (bindViews ctx ρ s ids st₁) := by
  unfold bindViews
  refine ResSim.foldlM (fun id a b hab => ?_) _ h
  split
  · split
    · exact ResSim.ite (c := ctx.isView _) (fun _ => bindView_sim ctx ρ s _ _ hab) (fun _ => ResSim.ok hab)
    · exact ResSim.ok hab
  · exact ResSim.ok hab

omit hu in
theorem refSet_sim (s : State) (e : DEdge) (d : Name) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (refSet (ctx.ren π) ρ s e d st₂) (refSet ctx ρ s e d st₁) := by
  unfold refSet
  split
  · exact aliasTo_sim ctx ρ h _ _ _
  · exact ResSim.err _
  · exact ResSim.err _

theorem execCopiesInto_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (execCopiesInto (ctx.ren π) ρ s n st₂) (execCopiesInto ctx ρ s n st₁) := by
  unfold execCopiesInto
  refine ResSim.foldlM (fun e a b hab => ?_) _ h
  split
  · exact ResSim.ok hab
  · refine ResSim.ite (c := ctx.isViewDef s e || s.isRouting e) (fun _ => ResSim.ok hab) (fun _ => ?_)
    rw [show refSetTarget? (ctx.ren π) s e = refSetTarget? ctx s e from rfl]
    split
    · exact refSet_sim ctx ρ s e _ hab
    · split
      · exact execCopy_sim ctx ρ hu s e hab
      · exact ResSim.ok hab

theorem execSimple_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.Sim π) (execSimple (ctx.ren π) ρ s n st₂) (execSimple ctx ρ s n st₁) := by
  unfold execSimple
  split
  · exact execCopiesInto_sim ctx ρ hu s n h
  · exact execCopiesInto_sim ctx ρ hu s n h
  · exact execCopiesInto_sim ctx ρ hu s n h
  · exact execCode_sim ctx ρ hu s n h _ _
  · split
    · exact execReduce_sim ctx ρ hu s n _ h
    · exact execCode_sim ctx ρ hu s n h _ _
  · exact ResSim.err _
  · exact ResSim.err _
  · exact ResSim.err _

omit hu in
theorem bindDynamicInputs_sim (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (St.SimR π) (bindDynamicInputs (ctx.ren π) ρ s n st₂) (bindDynamicInputs ctx ρ s n st₁) := by
  unfold bindDynamicInputs
  refine ResSim.foldlM (R := St.SimR π) (fun e a b hab => ?_) _ ⟨rfl, h⟩
  obtain ⟨ρ₂, a₂⟩ := a
  obtain ⟨ρ₁, a₁⟩ := b
  obtain ⟨hρ, ha⟩ := hab
  simp only at hρ ha ⊢
  subst hρ
  split
  · split
    · refine ResSim.bind (readEdgeScalar_sim ctx ρ₂ s e ha) fun x y hxy => ?_
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
def SetupSim (π : Equiv.Perm Name) (x₂ x₁ : Valuation × St × List Idx) : Prop :=
  x₂.1 = x₁.1 ∧ St.Sim π x₂.2.1 x₁.2.1 ∧ x₂.2.2 = x₁.2.2

omit hu in
theorem mapSetup_sim (s : State) (n : DNode) (mp : MapNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (SetupSim π) (mapSetup (ctx.ren π) ρ s n mp st₂) (mapSetup ctx ρ s n mp st₁) := by
  unfold mapSetup
  refine ResSim.bind (bindDynamicInputs_sim ctx ρ s n h) fun x y hxy => ?_
  obtain ⟨ρ₂, a₂⟩ := x
  obtain ⟨ρ₁, a₁⟩ := y
  obtain ⟨hρ, ha⟩ := hxy
  simp only at hρ ha ⊢
  subst hρ
  exact ResSim.bindEq (ctx.rangePoints_sim ha.mem ρ₂ mp.range) fun pts => ResSim.ok ⟨rfl, ha, rfl⟩

omit hu in
theorem scopeFinish_sim (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {pre₂ pre₁ : St} (h : St.Sim π pre₂ pre₁)
    (logs : List Log) :
    ResSim (St.Sim π) (scopeFinish (ctx.ren π) pre₂ (logs.map (Log.ren π))) (scopeFinish ctx pre₁ logs) := by
  unfold scopeFinish
  simp only [Ctx.ren_I]
  rcases merge_sim hw h.mem logs with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨ha, ?_, h.vals⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]

omit hu in
theorem stateFinish_sim (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₂ c₁ : CSt} (h : CSt.Sim π c₂ c₁)
    (logs : List Log) :
    ResSim (fun x₂ x₁ => x₂.1 = x₁.1 ∧ CSt.Sim π x₂.2 x₁.2) (stateFinish (ctx.ren π) c₂ (logs.map (Log.ren π)))
      (stateFinish ctx c₁ logs) := by
  unfold stateFinish
  simp only [Ctx.ren_I]
  rcases merge_sim hw h.mem logs with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨rfl, h.ρ, ha, ?_⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]

/-! ### Iterations and private transients -/

omit hu in
theorem privateTransients_ren (s : State) (entry : NodeId) :
    (ctx.ren π).privateTransients s entry = (ctx.privateTransients s entry).map π := by
  unfold Ctx.privateTransients
  rw [show (ctx.ren π).containerNames = ctx.containerNames from rfl, List.map_filterMap]
  refine List.filterMap_congr fun t _ => ?_
  rw [show (ctx.ren π).desc? t = ctx.desc? t from rfl, show (ctx.ren π).g = ctx.g from rfl]
  cases ctx.desc? t with
  | none => rfl
  | some d =>
    simp only
    split
    · split <;> simp_all [Ctx.ren]
    · rfl

omit hu in
theorem iterStart_sim {pre₂ pre₁ : St} (h : St.Sim π pre₂ pre₁) (priv : List Name) :
    St.Sim π (iterStart (priv.map π) pre₂) (iterStart priv pre₁) :=
  ⟨h.mem.allocAll priv, rfl, h.vals⟩

omit hu in
theorem allocate_sim {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (ts : List Name) :
    Mem.Sim π ((ctx.ren π).allocate ts m₂) (ctx.allocate ts m₁) := by
  unfold Ctx.allocate
  induction ts generalizing m₂ m₁ with
  | nil => exact hm
  | cons t ts ih =>
    simp only [List.foldl_cons, Ctx.ren_isStream, Ctx.ren_cname]
    refine ih ?_
    by_cases hs : ctx.isStream t = true
    · simp only [hs, ite_true]
      exact hm.alloc _
    · simp only [hs, Bool.false_eq_true, ite_false]
      exact hm.allocCells _

omit hu in
/-- Allocating more on the simulating side alone keeps the simulation when
the allocated containers are no streams (their queues are untouched). -/
theorem allocate_left {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (ts : List Name)
    (hs : ∀ t ∈ ts, ctx.isStream t = false) : Mem.Sim π (ctx.allocate ts m₂) m₁ := by
  unfold Ctx.allocate
  induction ts generalizing m₂ with
  | nil => exact hm
  | cons t ts ih =>
    simp only [List.foldl_cons, hs t List.mem_cons_self, Bool.false_eq_true, if_false]
    exact ih (hm.allocCells_left _) fun t' ht' => hs t' (List.mem_cons_of_mem _ ht')

/-! ### Traces -/

/-- Rename the deltas of a trace. -/
def renTrace (π : Equiv.Perm Name) (tr : List (NodeId × Log)) : List (NodeId × Log) :=
  tr.map fun (id, g) => (id, g.ren π)

omit hu in
theorem State.hbFree_ren (s : State) (π : Equiv.Perm Name) :
    ∀ tr : List (NodeId × Log), s.hbFree (renTrace π tr) = s.hbFree tr
  | [] => rfl
  | x :: xs => by
    simp only [renTrace, List.map_cons, State.hbFree, List.all_map, Function.comp_def, State.hbConflict,
      Log.conflicts_ren]
    rw [← State.hbFree_ren s π xs]
    rfl

/-! ### Consume scopes -/

omit hu in
theorem consumeStream_sim (s : State) (n : DNode) {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) :
    ResSim (fun l₂ l₁ => l₂ = Loc.ren π l₁) (consumeStream (ctx.ren π) ρ s n m₂) (consumeStream ctx ρ s n m₁) := by
  unfold consumeStream
  split
  · exact ResSim.err _
  · split
    · refine ResSim.ite (c := ctx.isStream _) (fun _ => ?_) (fun _ => ResSim.err _)
      refine ResSim.bindEq (streamPoint_sim ctx ρ hm _ _) fun p => ?_
      rw [ctx.loc_ren hm]
      cases ctx.loc ρ m₁ _ p false with
      | err e => exact ResSim.err e
      | ok l =>
        simp only [Res.map, Res.ok_bind, Res.pure_eq, hm.resolve]
        exact ResSim.ok rfl
    · exact ResSim.err _

/-- The result of a consume set-up. -/
def ConsumeSetupSim (π : Equiv.Perm Name) (x₂ x₁ : Valuation × St × Loc × Int) : Prop :=
  x₂.1 = x₁.1 ∧ St.Sim π x₂.2.1 x₁.2.1 ∧ x₂.2.2.1 = x₁.2.2.1.ren π ∧ x₂.2.2.2 = x₁.2.2.2

omit hu in
theorem consumeSetup_sim (s : State) (n : DNode) (c : ConsumeNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (ConsumeSetupSim π) (consumeSetup (ctx.ren π) ρ s n c st₂) (consumeSetup ctx ρ s n c st₁) := by
  unfold consumeSetup
  refine ResSim.bind (bindDynamicInputs_sim ctx ρ s n h) fun x y hxy => ?_
  obtain ⟨ρ₂, a₂⟩ := x
  obtain ⟨ρ₁, a₁⟩ := y
  obtain ⟨hρ, ha⟩ := hxy
  simp only at hρ ha ⊢
  subst hρ
  refine ResSim.bind (consumeStream_sim ctx ρ₂ s n ha.mem) fun l₂ l₁ hl => ?_
  exact ResSim.bindEq (ctx.evalIntR_sim ha.mem ρ₂ _) fun np => ResSim.ok ⟨rfl, ha, hl, rfl⟩

omit hu in
theorem consumeCond_sim (c : ConsumeNode) {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (l : Loc) (q : List Val) :
    ResSim Eq (consumeCond (ctx.ren π) ρ c m₂ (l.ren π) q) (consumeCond ctx ρ c m₁ l q) := by
  unfold consumeCond
  split
  · exact ResSim.ok rfl
  · exact ctx.evalBoolR_sim (hm.setQueue l q) ρ _

omit hu in
theorem taskStart_sim (s : State) (entry : NodeId) (c : ConsumeNode) (l : Loc) (priv : List Name)
    {pre₂ pre₁ : St} (h : St.Sim π pre₂ pre₁) (q : List Val) :
    St.Sim π (taskStart (ctx.ren π) s entry c (l.ren π) (priv.map π) pre₂ q) (taskStart ctx s entry c l priv pre₁ q) := by
  unfold taskStart chunkMemName
  refine ⟨?_, rfl, h.vals⟩
  simp only [Ctx.ren_cname]
  have hq := (h.mem.allocAll priv).setQueue l (q.drop c.chunksize)
  cases s.consumedStream? entry with
  | none => exact hq
  | some d => exact hq.writeChunk _ _

omit hu in
theorem Log.ren_stripQueue (g : Log) (l : Loc) : (g.stripQueue l).ren π = (g.ren π).stripQueue (l.ren π) := by
  simp only [Log.stripQueue, Log.ren]
  congr 1
  refine List.map_filter_comm _ _ _ (fun x => ?_) _
  simp only [ne_eq, decide_not, Loc.ren_inj]

omit hu in
theorem Log.pushedTo_ren (g : Log) (l : Loc) : (g.ren π).pushedTo (l.ren π) = g.pushedTo l := by
  simp only [Log.pushedTo, Log.ren, List.flatMap_map, Loc.ren_inj]

omit hu in
theorem taskLog_ren {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (priv : List Name) (l : Loc) :
    taskLog st₂ (priv.map π) (l.ren π) = (taskLog st₁ priv l).ren π := by
  unfold taskLog
  rw [h.log, ← Log.ren_dropNames, Log.ren_stripQueue]

omit hu in
theorem taskPushed_ren {st₂ st₁ : St} (h : St.Sim π st₂ st₁) (priv : List Name) (l : Loc) :
    taskPushed st₂ (priv.map π) (l.ren π) = taskPushed st₁ priv l := by
  unfold taskPushed
  rw [h.log, ← Log.ren_dropNames, Log.pushedTo_ren]

omit hu in
theorem consumeFinish_sim (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {pre₂ pre₁ : St} (h : St.Sim π pre₂ pre₁)
    (l : Loc) (logs : List Log) (rest pushed : List Val) :
    ResSim (St.Sim π) (consumeFinish (ctx.ren π) pre₂ (l.ren π) (logs.map (Log.ren π)) rest pushed)
      (consumeFinish ctx pre₁ l logs rest pushed) := by
  unfold consumeFinish
  simp only [Ctx.ren_I]
  rcases merge_sim hw h.mem logs with ⟨e₂, e₁, h₂, h₁⟩ | ⟨a₂, a₁, h₂, h₁, ha⟩
  · rw [h₂, h₁]; exact ResSim.ub rfl _
  · rw [h₂, h₁]
    refine ResSim.ok ⟨ha.setQueue l _, ?_, h.vals⟩
    simp only [h.log, Log.ren_seq, Log.ren_unionAll]
    rfl

/-! ### Nested SDFGs -/

/-- Rename the memory names of connector bindings. -/
def renConns (π : Equiv.Perm Name) (conns : List (Name × Name)) : List (Name × Name) :=
  conns.map fun (c, l) => (c, π l)

theorem lookup_renConns (conns : List (Name × Name)) (n : Name) :
    (renConns π conns).lookup n = (conns.lookup n).map π := by
  induction conns with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨c, l⟩ := x
    simp only [renConns, List.map_cons, List.lookup_cons] at ih ⊢
    split <;> simp_all

omit hu in
theorem enter_ren (inner : SDFG) (node : NodeId) (conns : List (Name × Name)) :
    (ctx.ren π).enter inner node (renConns π conns) = (ctx.enter inner node conns).ren π := by
  simp only [Ctx.enter, Ctx.ren, Ctx.mk.injEq, true_and]
  refine ⟨funext fun n => ?_, rfl⟩
  simp only [Function.comp_apply]
  induction conns with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨c, l⟩ := x
    simp only [renConns, List.map_cons, List.lookup] at ih ⊢
    split <;> simp_all

omit hu in
theorem nestedBindings_sim (s : State) (n : DNode) :
    nestedBindings (ctx.ren π) s n = (nestedBindings ctx s n).map (renConns π) := by
  unfold nestedBindings
  suffices ∀ (cs : List Conn) (acc : List (Name × Name)),
      cs.foldlM (fun acc c =>
        match nestedEndpoint s n c with
        | .data base _ => pure (acc ++ [(c, (ctx.ren π).cname base)])
        | .code _ => pure acc
        | .invalid => Res.err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) (renConns π acc) =
      (cs.foldlM (fun acc c =>
        match nestedEndpoint s n c with
        | .data base _ => pure (acc ++ [(c, ctx.cname base)])
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
    · next base sub _ =>
      have e : renConns π acc ++ [(c, (ctx.ren π).cname base)] = renConns π (acc ++ [(c, ctx.cname base)]) := by
        simp [renConns]
      exact (congrArg (fun x => List.foldlM _ x cs) e).trans (ih _)
    · exact ih acc
    · rfl

omit hu in
theorem nestedValuation_sim {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (mapping : List (Sym × Expr)) :
    ResSim Eq (nestedValuation (ctx.ren π) ρ m₂ mapping) (nestedValuation ctx ρ m₁ mapping) := by
  unfold nestedValuation
  refine ResSim.bindEq (ResSim.mapM (fun (xe : Sym × Expr) => ?_) mapping) fun vs => ResSim.ok rfl
  exact ResSim.bindEq (ctx.evalR_sim hm ρ xe.2) fun v => ResSim.ok rfl

omit hu in
theorem bindCodeInputs_sim (ρ' : Valuation) (s : State) (n : DNode) {st₂ st₁ : St} (h : St.Sim π st₂ st₁) :
    ResSim (fun (x₂ x₁ : St × List Name) => St.Sim π x₂.1 x₁.1 ∧ x₂.2 = x₁.2.map π)
      (bindCodeInputs (ctx.ren π) ρ' s n st₂) (bindCodeInputs ctx ρ' s n st₁) := by
  unfold bindCodeInputs
  refine ResSim.foldlM (R := fun (x₂ x₁ : St × List Name) => St.Sim π x₂.1 x₁.1 ∧ x₂.2 = x₁.2.map π)
    (fun c a b hab => ?_) _
    ⟨h, rfl⟩
  obtain ⟨a₂, cs₂⟩ := a
  obtain ⟨a₁, cs₁⟩ := b
  obtain ⟨ha, hc⟩ := hab
  simp only at ha hc ⊢
  subst hc
  split
  · rw [ha.val?]
    split
    · rw [ctx.loc_ren ha.mem]
      cases ctx.loc ρ' a₁.mem c [] false with
      | err e => exact ResSim.err e
      | ok l =>
        simp only [Res.map, Res.ok_bind, Res.pure_eq]
        exact ResSim.ok ⟨ha.withMem (ha.mem.write l _), by simp⟩
    · exact ResSim.ok ⟨ha, by simp⟩
  · exact ResSim.ok ⟨ha, rfl⟩

/-- The result of a nested set-up: renamed inner context, the same inner
valuation, simulating states, renamed private names. -/
def NestedSim (π : Equiv.Perm Name) (x₂ x₁ : Ctx × Valuation × St × List Name) : Prop :=
  x₂.1 = x₁.1.ren π ∧ x₂.2.1 = x₁.2.1 ∧ St.Sim π x₂.2.2.1 x₁.2.2.1 ∧ x₂.2.2.2 = x₁.2.2.2.map π

omit hu in
theorem nestedSetup_sim (s : State) (n : DNode) (inner : SDFG) (mapping : List (Sym × Expr)) {st₂ st₁ : St}
    (h : St.Sim π st₂ st₁) :
    ResSim (NestedSim π) (nestedSetup (ctx.ren π) ρ s n inner mapping st₂) (nestedSetup ctx ρ s n inner mapping st₁) := by
  unfold nestedSetup
  rw [nestedBindings_sim]
  cases nestedBindings ctx s n with
  | err e => exact ResSim.err e
  | ok conns =>
    simp only [Res.map, Res.ok_bind]
    rw [enter_ren]
    refine ResSim.bindEq (nestedValuation_sim ctx ρ h.mem mapping) fun ρ' => ?_
    have hs : St.Sim π { st₂ with mem := ((ctx.enter inner n.id conns).ren π).allocate inner.callTransients st₂.mem }
        { st₁ with mem := (ctx.enter inner n.id conns).allocate inner.callTransients st₁.mem } :=
      h.withMem (allocate_sim _ h.mem _)
    refine ResSim.bind (bindCodeInputs_sim _ ρ' s n hs) fun x y hxy => ?_
    obtain ⟨a₂, c₂⟩ := x
    obtain ⟨a₁, c₁⟩ := y
    obtain ⟨ha, hc⟩ := hxy
    simp only at ha hc ⊢
    refine ResSim.ok ⟨rfl, rfl, ha, ?_⟩
    simp only [hc, List.map_append, List.map_map]
    rfl

omit hu in
theorem nestedFinish_sim (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (dropped : List Name)
    {pre₂ pre₁ : St} (h : St.Sim π pre₂ pre₁) {m₂ m₁ : Mem} (hm : Mem.Sim π m₂ m₁) (g : Log) :
    ResSim (St.Sim π) (nestedFinish (ctx'.ren π) ρ' s n (dropped.map π) pre₂ m₂ (g.ren π))
      (nestedFinish ctx' ρ' s n dropped pre₁ m₁ g) := by
  unfold nestedFinish
  refine ResSim.foldlM (fun e a b hab => ?_) _ ⟨hm, ?_, h.vals⟩
  · split
    · split
      · rw [ctx'.loc_ren hm]
        cases ctx'.loc ρ' m₁ _ [] false with
        | err e => exact ResSim.err e
        | ok l =>
          simp only [Res.map, Res.ok_bind]
          rcases hm.read l with h₁ | h₁ <;> rw [h₁]
          · exact ResSim.ub rfl _
          · split
            · exact ResSim.ok (hab.setVal _ _)
            · exact ResSim.err _
      · exact ResSim.ok hab
    · exact ResSim.ok hab
  · simp only [h.log, Log.ren_seq, Log.ren_dropNames]

/-! ### Control flow -/

omit hu in
theorem trueEdges_sim (rg : Region) {c₂ c₁ : CSt} (h : CSt.Sim π c₂ c₁) (bid : BlockId) :
    ResSim Eq (trueEdges (ctx.ren π) rg c₂ bid) (trueEdges ctx rg c₁ bid) := by
  unfold trueEdges
  refine ResSim.filterMapM (fun e => ?_) _
  rw [h.ρ]
  exact ResSim.bindEq (ctx.evalBoolR_sim h.mem c₁.ρ e.cond) fun b => ResSim.ok rfl

omit hu in
theorem applyAssigns_sim {c₂ c₁ : CSt} (h : CSt.Sim π c₂ c₁) (as : List (Sym × Expr)) :
    ResSim (CSt.Sim π) (applyAssigns (ctx.ren π) c₂ as) (applyAssigns ctx c₁ as) := by
  unfold applyAssigns
  rw [h.ρ]
  refine ResSim.bindEq (ResSim.mapM (fun (xe : Sym × Expr) => ?_) as) fun vs => ?_
  · exact ResSim.bindEq (ctx.evalR_sim h.mem c₁.ρ xe.2) fun v => ResSim.ok rfl
  · exact ResSim.ok ⟨rfl, h.mem, h.log⟩

omit hu in
theorem applyAssign?_sim {c₂ c₁ : CSt} (h : CSt.Sim π c₂ c₁) (a : Option (Sym × Expr)) :
    ResSim (CSt.Sim π) (applyAssign? (ctx.ren π) c₂ a) (applyAssign? ctx c₁ a) := by
  cases a with
  | none => exact ResSim.ok h
  | some a => exact applyAssigns_sim ctx h [a]

omit hu in
theorem evalBoolR_cst_sim {c₂ c₁ : CSt} (h : CSt.Sim π c₂ c₁) (e : Expr) :
    ResSim Eq ((ctx.ren π).evalBoolR c₂.ρ c₂.mem e) (ctx.evalBoolR c₁.ρ c₁.mem e) := by
  rw [h.ρ]; exact ctx.evalBoolR_sim h.mem c₁.ρ e

end Leaf

/-! ## The execution relations -/

/-- A sequential result: simulating states, renamed trace. -/
def SeqSim (π : Equiv.Perm Name) (x₂ x₁ : St × List (NodeId × Log)) : Prop :=
  St.Sim π x₂.1 x₁.1 ∧ x₂.2 = renTrace π x₁.2

/-- A control-flow result: the same signal, simulating states. -/
def CtlSim (π : Equiv.Perm Name) (x₂ x₁ : Ctl × CSt) : Prop := x₂.1 = x₁.1 ∧ CSt.Sim π x₂.2 x₁.2

/-- Logs of parallel executions: renamed. -/
def LogsSim (π : Equiv.Perm Name) (l₂ l₁ : List Log) : Prop := l₂ = l₁.map (Log.ren π)

/-- Consume tasks: renamed logs, the same queues. -/
def TasksSim (π : Equiv.Perm Name) (x₂ x₁ : List Log × List Val × List Val) : Prop :=
  x₂.1 = x₁.1.map (Log.ren π) ∧ x₂.2 = x₁.2

/-- An SDFG invocation: the same valuation, simulating memory, renamed log. -/
def InvSim (π : Equiv.Perm Name) (x₂ x₁ : Valuation × Mem × Log) : Prop :=
  x₂.1 = x₁.1 ∧ Mem.Sim π x₂.2.1 x₁.2.1 ∧ x₂.2.2 = x₁.2.2.ren π

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
          { st with mem := (ctx.enter inner n.id conns).allocate inner.callTransients st.mem } with
      | err e => rw [hc] at h; cases h
      | ok x =>
        rw [hc] at h
        simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
        rw [← h.1]
        rfl

section Rel

variable {π : Equiv.Perm Name}

mutual

theorem NodeExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {n : DNode} {st₁ : St} {r₁ : Res St} :
    NodeExec ctx ρ s n st₁ r₁ → ∀ st₂, St.Sim π st₂ st₁ →
      ∃ r₂, NodeExec (ctx.ren π) ρ s n st₂ r₂ ∧ ResSim (St.Sim π) r₂ r₁
  | .simple hs, st₂, h => ⟨_, .simple hs, execSimple_sim ctx ρ hw s n h⟩
  | .mapSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ mapSetup_sim ctx ρ s n _ h)
    exact ⟨_, .mapSetupErr hk h₂, ResSim.of_err hu⟩
  | .mapIterErr hk hs hi, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ mapSetup_sim ctx ρ s n _ h) with ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, pts₂⟩, h₂, hρ, ha, hp⟩
    · exact ⟨_, .mapSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hp
      rw [hρ, hp] at h₂
      obtain ⟨r₂, hi₂, hr⟩ := IterExec.sim hw hi _ ha
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .mapIterErr hk h₂ (by rw [privateTransients_ren]; exact hi₂), ResSim.of_err hu⟩
  | .mapScope hk hs hi, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ mapSetup_sim ctx ρ s n _ h) with ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, pts₂⟩, h₂, hρ, ha, hp⟩
    · exact ⟨_, .mapSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hp
      rw [hρ, hp] at h₂
      obtain ⟨r₂, hi₂, hr⟩ := IterExec.sim hw hi _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl⟩
      · exact ⟨_, .mapIterErr hk h₂ (by rw [privateTransients_ren]; exact hi₂), ResSim.ub hu _⟩
      · refine ⟨_, .mapScope hk h₂ (by rw [privateTransients_ren]; exact hi₂), ?_⟩
        rw [hl]
        exact scopeFinish_sim ctx hw ha _
  | .consumeSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ consumeSetup_sim ctx ρ s n _ h)
    exact ⟨_, .consumeSetupErr hk h₂, ResSim.of_err hu⟩
  | .consumeTasksErr hk hs ht, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ consumeSetup_sim ctx ρ s n _ h) with
      ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, l₂, np₂⟩, h₂, hρ, ha, hl, hnp⟩
    · exact ⟨_, .consumeSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hl hnp
      rw [hρ, hl, hnp] at h₂
      obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.sim hw ht _ ha
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      refine ⟨_, .consumeTasksErr hk h₂ (by rw [privateTransients_ren, ha.mem.queues]; exact ht₂), ResSim.of_err hu⟩
  | .consumeScope hk hs ht, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ consumeSetup_sim ctx ρ s n _ h) with
      ⟨e, h₂, hu⟩ | ⟨⟨ρ₂, a₂, l₂, np₂⟩, h₂, hρ, ha, hl, hnp⟩
    · exact ⟨_, .consumeSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hρ ha hl hnp
      rw [hρ, hl, hnp] at h₂
      obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.sim hw ht _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨logs₂, rest₂, pushed₂⟩, rfl, hl, hq⟩
      · exact ⟨_, .consumeTasksErr hk h₂ (by rw [privateTransients_ren, ha.mem.queues]; exact ht₂),
          ResSim.ub hu _⟩
      · simp only [Prod.mk.injEq] at hl hq
        obtain ⟨rfl, rfl⟩ := hq
        subst hl
        exact ⟨_, .consumeScope hk h₂ (by rw [privateTransients_ren, ha.mem.queues]; exact ht₂),
          consumeFinish_sim ctx hw ha _ _ _ _⟩
  | .nestedSetupErr hk he, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ nestedSetup_sim ctx ρ s n _ _ h)
    exact ⟨_, .nestedSetupErr hk h₂, ResSim.of_err hu⟩
  | .nestedInnerErr hk hs hr, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ nestedSetup_sim ctx ρ s n _ _ h) with
      ⟨e, h₂, hu⟩ | ⟨⟨c₂, ρ₂, a₂, d₂⟩, h₂, hc, hρ, ha, hd⟩
    · exact ⟨_, .nestedSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hc hρ ha hd
      rw [hc, hρ, hd] at h₂
      obtain ⟨r₂, hr₂, hres⟩ := SDFGExec.sim (by rw [nestedSetup_I hs]; exact hw) hr _ ha.mem
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hres
      exact ⟨_, .nestedInnerErr hk h₂ hr₂, ResSim.of_err hu⟩
  | .nested hk hs hr, st₂, h => by
    rcases ResSim.ok_cases (hs ▸ nestedSetup_sim ctx ρ s n _ _ h) with
      ⟨e, h₂, hu⟩ | ⟨⟨c₂, ρ₂, a₂, d₂⟩, h₂, hc, hρ, ha, hd⟩
    · exact ⟨_, .nestedSetupErr hk h₂, ResSim.ub hu _⟩
    · simp only at hc hρ ha hd
      rw [hc, hρ, hd] at h₂
      obtain ⟨r₂, hr₂, hres⟩ := SDFGExec.sim (by rw [nestedSetup_I hs]; exact hw) hr _ ha.mem
      rcases ResSim.ok_cases hres with ⟨e, rfl, hu⟩ | ⟨⟨ρ₃, m₃, g₃⟩, rfl, hρ₃, hm₃, hg₃⟩
      · exact ⟨_, .nestedInnerErr hk h₂ hr₂, ResSim.ub hu _⟩
      · simp only at hρ₃ hm₃ hg₃
        subst hρ₃ hg₃
        exact ⟨_, .nested hk h₂ hr₂, nestedFinish_sim _ _ s n _ ha hm₃ _⟩

theorem SeqExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {ids : List NodeId} {st₁ : St} {r₁ : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s ids st₁ r₁ → ∀ st₂, St.Sim π st₂ st₁ →
      ∃ r₂, SeqExec (ctx.ren π) ρ s ids st₂ r₂ ∧ ResSim (SeqSim π) r₂ r₁
  | .nil, st₂, h => ⟨_, .nil, ResSim.ok ⟨h, rfl⟩⟩
  | .missing hm, st₂, h => ⟨_, .missing hm, ResSim.err _⟩
  | .headErr hn hx, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hw hx _ h.forNode
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hn hx₂, ResSim.of_err hu⟩
  | .tailErr hn hx hrest, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hw hx _ h.forNode
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hn hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.sim hw hrest _ (h.afterNode ha)
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hn hx₂ hrest₂, ResSim.of_err hu⟩
  | .cons hn hx hrest, st₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.sim hw hx _ h.forNode
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hn hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.sim hw hrest _ (h.afterNode ha)
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .tailErr hn hx₂ hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hn hx₂ hrest₂, ResSim.ok ⟨hb, ?_⟩⟩
        simp only at htr ⊢
        rw [htr, ha.log]
        rfl

theorem OrderedExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {all direct : List NodeId} {st₁ : St} {r₁ : Res St} :
    OrderedExec ctx ρ s all direct st₁ r₁ → ∀ st₂, St.Sim π st₂ st₁ →
      ∃ r₂, OrderedExec (ctx.ren π) ρ s all direct st₂ r₂ ∧ ResSim (St.Sim π) r₂ r₁
  | .prologueErr hp, st₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hp ▸ bindViews_sim ctx ρ s direct h)
    exact ⟨_, .prologueErr h₂, ResSim.of_err hu⟩
  | .seqErr ho hp hs, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim ctx ρ s direct h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hw hs _ ha
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .seqErr ho h₂ hs₂, ResSim.of_err hu⟩
  | .hbRace ho hp hs hf, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim ctx ρ s direct h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hw hs _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .seqErr ho h₂ hs₂, ResSim.ub hu _⟩
      · simp only at htr
        subst htr
        exact ⟨_, .hbRace ho h₂ hs₂ (by rw [State.hbFree_ren]; exact hf), ResSim.ub rfl _⟩
  | .run ho hp hs hf, st₂, h => by
    rcases ResSim.ok_cases (hp ▸ bindViews_sim ctx ρ s direct h) with ⟨e, h₂, hu⟩ | ⟨a₂, h₂, ha⟩
    · exact ⟨_, .prologueErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hs₂, hr⟩ := SeqExec.sim hw hs _ ha
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨b₂, tr₂⟩, rfl, hb, htr⟩
      · exact ⟨_, .seqErr ho h₂ hs₂, ResSim.ub hu _⟩
      · simp only at htr
        subst htr
        exact ⟨_, .run ho h₂ hs₂ (by rw [State.hbFree_ren]; exact hf), ResSim.ok hb⟩

theorem IterExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {entry : NodeId} {params : List Sym} {priv : List Name} {pre₁ : St} {pts : List Idx} {r₁ : Res (List Log)} :
    IterExec ctx ρ s entry params priv pre₁ pts r₁ → ∀ pre₂, St.Sim π pre₂ pre₁ →
      ∃ r₂, IterExec (ctx.ren π) ρ s entry params (priv.map π) pre₂ pts r₂ ∧ ResSim (LogsSim π) r₂ r₁
  | .nil, pre₂, h => ⟨_, .nil, ResSim.ok rfl⟩
  | .headErr hb, pre₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (iterStart_sim h priv)
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hb₂, ResSim.of_err hu⟩
  | .tailErr hb hrest, pre₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (iterStart_sim h priv)
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := IterExec.sim hw hrest _ h
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hb₂ hrest₂, ResSim.of_err hu⟩
  | .cons hb hrest, pre₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (iterStart_sim h priv)
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := IterExec.sim hw hrest _ h
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl⟩
      · exact ⟨_, .tailErr hb₂ hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hb₂ hrest₂, ResSim.ok ?_⟩
        unfold LogsSim at hl ⊢
        rw [hl, ha.log, ← Log.ren_dropNames]
        rfl

theorem ConsumeTasks.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {entry : NodeId} {c : ConsumeNode} {l : Loc} {np : Int} {priv : List Name} {pre₁ : St} {q : List Val}
    {r₁ : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx ρ s entry c l np priv pre₁ q r₁ → ∀ pre₂, St.Sim π pre₂ pre₁ →
      ∃ r₂, ConsumeTasks (ctx.ren π) ρ s entry c (l.ren π) np (priv.map π) pre₂ q r₂ ∧ ResSim (TasksSim π) r₂ r₁
  | .condErr hc, pre₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hc ▸ consumeCond_sim ctx ρ c h.mem l q)
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .stop hc, pre₂, h => by
    rcases ResSim.ok_cases (hc ▸ consumeCond_sim ctx ρ c h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · exact ⟨_, .stop h₂, ResSim.ok ⟨rfl, rfl⟩⟩
  | .bodyErr hc hq hp hb, pre₂, h => by
    rcases ResSim.ok_cases (hc ▸ consumeCond_sim ctx ρ c h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (taskStart_sim ctx s entry c l priv h q)
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
      exact ⟨_, .bodyErr h₂ hq hp hb₂, ResSim.of_err hu⟩
  | .restErr hc hq hp hb hrest, pre₂, h => by
    rcases ResSim.ok_cases (hc ▸ consumeCond_sim ctx ρ c h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (taskStart_sim ctx s entry c l priv h q)
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
      · exact ⟨_, .bodyErr h₂ hq hp hb₂, ResSim.ub hu _⟩
      · obtain ⟨r₃, hrest₂, hr₃⟩ := ConsumeTasks.sim hw hrest _ h
        obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
        exact ⟨_, .restErr h₂ hq hp hb₂ hrest₂, ResSim.of_err hu⟩
  | .step hc hq hp hb hrest, pre₂, h => by
    rcases ResSim.ok_cases (hc ▸ consumeCond_sim ctx ρ c h.mem l q) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.sim hw hb _ (taskStart_sim ctx s entry c l priv h q)
      rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
      · exact ⟨_, .bodyErr h₂ hq hp hb₂, ResSim.ub hu _⟩
      · obtain ⟨r₃, hrest₂, hr₃⟩ := ConsumeTasks.sim hw hrest _ h
        rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨⟨logs₂, rest₂, pushed₂⟩, rfl, hl, hq'⟩
        · exact ⟨_, .restErr h₂ hq hp hb₂ hrest₂, ResSim.ub hu _⟩
        · simp only [Prod.mk.injEq] at hl hq'
          obtain ⟨rfl, rfl⟩ := hq'
          subst hl
          refine ⟨_, .step h₂ hq hp hb₂ hrest₂, ResSim.ok ⟨?_, ?_⟩⟩
          · simp only [List.map_cons, taskLog_ren ha]
          · simp only [taskPushed_ren ha]

theorem CompsExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {s : State}
    {m₁ : Mem} {comps : List (List NodeId)} {r₁ : Res (List Log)} :
    CompsExec ctx ρ s m₁ comps r₁ → ∀ m₂, Mem.Sim π m₂ m₁ →
      ∃ r₂, CompsExec (ctx.ren π) ρ s m₂ comps r₂ ∧ ResSim (LogsSim π) r₂ r₁
  | .nil, m₂, h => ⟨_, .nil, ResSim.ok rfl⟩
  | .headErr hc, m₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.sim hw hc { mem := m₂ } ⟨h, rfl, rfl⟩
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr hc₂, ResSim.of_err hu⟩
  | .tailErr hc hrest, m₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.sim hw hc { mem := m₂ } ⟨h, rfl, rfl⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hc₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := CompsExec.sim hw hrest _ h
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hc₂ hrest₂, ResSim.of_err hu⟩
  | .cons hc hrest, m₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.sim hw hc { mem := m₂ } ⟨h, rfl, rfl⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, ha⟩
    · exact ⟨_, .headErr hc₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := CompsExec.sim hw hrest _ h
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl⟩
      · exact ⟨_, .tailErr hc₂ hrest₂, ResSim.ub hu _⟩
      · refine ⟨_, .cons hc₂ hrest₂, ResSim.ok ?_⟩
        unfold LogsSim at hl ⊢
        rw [hl, ha.log]
        rfl

theorem StateExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {s : State}
    {r₁ : Res (Ctl × CSt)} :
    StateExec ctx c₁ s r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, StateExec (ctx.ren π) c₂ s r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .compsErr hc, c₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.sim hw hc _ (allocate_sim ctx h.mem _)
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .compsErr (by rw [h.ρ]; exact hc₂), ResSim.of_err hu⟩
  | .run hc, c₂, h => by
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.sim hw hc _ (allocate_sim ctx h.mem _)
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, hl⟩
    · exact ⟨_, .compsErr (by rw [h.ρ]; exact hc₂), ResSim.ub hu _⟩
    · unfold LogsSim at hl
      subst hl
      exact ⟨_, .run (by rw [h.ρ]; exact hc₂), (stateFinish_sim ctx hw h _).mono fun _ _ hx => hx⟩

theorem BlockExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {b : Block}
    {r₁ : Res (Ctl × CSt)} :
    BlockExec ctx c₁ b r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, BlockExec (ctx.ren π) c₂ b r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .state hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := StateExec.sim hw hx _ h
    exact ⟨_, .state hx₂, hr⟩
  | .region hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx _ h
    exact ⟨_, .region hx₂, hr⟩
  | .loop hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := LoopExec.sim hw hx _ h
    exact ⟨_, .loop hx₂, hr⟩
  | .cond hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := CondExec.sim hw hx _ h
    exact ⟨_, .cond hx₂, hr⟩
  | .brk, c₂, h => ⟨_, .brk, ResSim.ok ⟨rfl, h⟩⟩
  | .cont, c₂, h => ⟨_, .cont, ResSim.ok ⟨rfl, h⟩⟩
  | .ret, c₂, h => ⟨_, .ret, ResSim.ok ⟨rfl, h⟩⟩

theorem RegionAt.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {rg : Region} {c₁ : CSt}
    {bid : BlockId} {r₁ : Res (Ctl × CSt)} :
    RegionAt ctx rg c₁ bid r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, RegionAt (ctx.ren π) rg c₂ bid r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .missing hb, c₂, h => ⟨_, .missing hb, ResSim.err _⟩
  | .blockErr hb hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .blockErr hb hx₂, ResSim.of_err hu⟩
  | .exit hb hx hc, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .exit hb hx₂ hc, ResSim.ok ⟨rfl, hd⟩⟩
  | .edgesErr hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (he ▸ trueEdges_sim ctx rg hd bid)
      exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.of_err hu⟩
  | .stop hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim ctx rg hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · exact ⟨_, .stop hb hx₂ h₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .ambiguous hb hx he, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim ctx rg hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · exact ⟨_, .ambiguous hb hx₂ h₂, ResSim.err _⟩
  | .assignErr hb hx he ha, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim ctx rg hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · obtain ⟨e', ha₂, hu⟩ := ResSim.err_inv (ha ▸ applyAssigns_sim ctx hd _)
        exact ⟨_, .assignErr hb hx₂ h₂ ha₂, ResSim.of_err hu⟩
  | .step hb hx he ha hn, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := BlockExec.sim hw hx _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .blockErr hb hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      rcases ResSim.ok_cases (he ▸ trueEdges_sim ctx rg hd bid) with ⟨e, h₂, hu⟩ | ⟨es, h₂, rfl⟩
      · exact ⟨_, .edgesErr hb hx₂ h₂, ResSim.ub hu _⟩
      · rcases ResSim.ok_cases (ha ▸ applyAssigns_sim ctx hd _) with ⟨e, ha₂, hu⟩ | ⟨d₃, ha₂, hd₃⟩
        · exact ⟨_, .assignErr hb hx₂ h₂ ha₂, ResSim.ub hu _⟩
        · obtain ⟨r₃, hn₂, hr₃⟩ := RegionAt.sim hw hn _ hd₃
          exact ⟨_, .step hb hx₂ h₂ ha₂ hn₂, hr₃⟩

theorem RegionExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {rg : Region}
    {r₁ : Res (Ctl × CSt)} :
    RegionExec ctx c₁ rg r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, RegionExec (ctx.ren π) c₂ rg r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .start hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionAt.sim hw hx _ h
    exact ⟨_, .start hx₂, hr⟩

theorem LoopExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {l : LoopRegion}
    {r₁ : Res (Ctl × CSt)} :
    LoopExec ctx c₁ l r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, LoopExec (ctx.ren π) c₂ l r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .initErr hi, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hi ▸ applyAssign?_sim ctx h l.init)
    exact ⟨_, .initErr h₂, ResSim.of_err hu⟩
  | .guardFirst hi hv hg, c₂, h => by
    rcases ResSim.ok_cases (hi ▸ applyAssign?_sim ctx h l.init) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .initErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hg₂, hr⟩ := LoopGuard.sim hw hg _ hd
      exact ⟨_, .guardFirst h₂ hv hg₂, hr⟩
  | .bodyFirst hi hv hb, c₂, h => by
    rcases ResSim.ok_cases (hi ▸ applyAssign?_sim ctx h l.init) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .initErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hw hb _ hd
      exact ⟨_, .bodyFirst h₂ hv hb₂, hr⟩

theorem LoopGuard.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {l : LoopRegion}
    {r₁ : Res (Ctl × CSt)} :
    LoopGuard ctx c₁ l r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, LoopGuard (ctx.ren π) c₂ l r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .condErr hc, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hc ▸ evalBoolR_cst_sim ctx h l.cond)
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .exit hc, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h l.cond) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · exact ⟨_, .exit h₂, ResSim.ok ⟨rfl, h⟩⟩
  | .enter hc hb, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h l.cond) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hw hb _ h
      exact ⟨_, .enter h₂ hb₂, hr⟩

theorem LoopIter.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {l : LoopRegion}
    {r₁ : Res (Ctl × CSt)} :
    LoopIter ctx c₁ l r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, LoopIter (ctx.ren π) c₂ l r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .bodyErr hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hw hb _ h
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .bodyErr hb₂, ResSim.of_err hu⟩
  | .ret hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hw hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .ret hb₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .brk hb, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hw hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .brk hb₂, ResSim.ok ⟨rfl, hd⟩⟩
  | .tail hb hc ht, c₂, h => by
    obtain ⟨r₂, hb₂, hr⟩ := RegionExec.sim hw hb _ h
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .bodyErr hb₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      obtain ⟨r₃, ht₂, hr₃⟩ := LoopTail.sim hw ht _ hd
      exact ⟨_, .tail hb₂ hc ht₂, hr₃⟩

theorem LoopTail.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt} {l : LoopRegion}
    {r₁ : Res (Ctl × CSt)} :
    LoopTail ctx c₁ l r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, LoopTail (ctx.ren π) c₂ l r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .updateErr hv hu', c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hu' ▸ applyAssign?_sim ctx h l.update)
    exact ⟨_, .updateErr hv h₂, ResSim.of_err hu⟩
  | .updateGuard hv hu' hg, c₂, h => by
    rcases ResSim.ok_cases (hu' ▸ applyAssign?_sim ctx h l.update) with ⟨e, h₂, hu⟩ | ⟨d, h₂, hd⟩
    · exact ⟨_, .updateErr hv h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hg₂, hr⟩ := LoopGuard.sim hw hg _ hd
      exact ⟨_, .updateGuard hv h₂ hg₂, hr⟩
  | .condFirstErr hv hc, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hc ▸ evalBoolR_cst_sim ctx h l.cond)
    exact ⟨_, .condFirstErr hv h₂, ResSim.of_err hu⟩
  | .condFirstExit hv hc, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h l.cond) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · exact ⟨_, .condFirstExit hv h₂, ResSim.ok ⟨rfl, h⟩⟩
  | .condFirstUpdateErr hv hc hu', c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h l.cond) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · obtain ⟨e', h₃, hu⟩ := ResSim.err_inv (hu' ▸ applyAssign?_sim ctx h l.update)
      exact ⟨_, .condFirstUpdateErr hv h₂ h₃, ResSim.of_err hu⟩
  | .condFirstNext hv hc hu' hb, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h l.cond) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condFirstErr hv h₂, ResSim.ub hu _⟩
    · rcases ResSim.ok_cases (hu' ▸ applyAssign?_sim ctx h l.update) with ⟨e, h₃, hu⟩ | ⟨d, h₃, hd⟩
      · exact ⟨_, .condFirstUpdateErr hv h₂ h₃, ResSim.ub hu _⟩
      · obtain ⟨r₂, hb₂, hr⟩ := LoopIter.sim hw hb _ hd
        exact ⟨_, .condFirstNext hv h₂ h₃ hb₂, hr⟩

theorem CondExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {c₁ : CSt}
    {brs : List (Option Expr × Region)} {r₁ : Res (Ctl × CSt)} :
    CondExec ctx c₁ brs r₁ → ∀ c₂, CSt.Sim π c₂ c₁ →
      ∃ r₂, CondExec (ctx.ren π) c₂ brs r₂ ∧ ResSim (CtlSim π) r₂ r₁
  | .nil, c₂, h => ⟨_, .nil, ResSim.ok ⟨rfl, h⟩⟩
  | .els hx, c₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx _ h
    exact ⟨_, .els hx₂, hr⟩
  | .condErr hc, c₂, h => by
    obtain ⟨e', h₂, hu⟩ := ResSim.err_inv (hc ▸ evalBoolR_cst_sim ctx h _)
    exact ⟨_, .condErr h₂, ResSim.of_err hu⟩
  | .taken hc hx, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h _) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx _ h
      exact ⟨_, .taken h₂ hx₂, hr⟩
  | .skipped hc hx, c₂, h => by
    rcases ResSim.ok_cases (hc ▸ evalBoolR_cst_sim ctx h _) with ⟨e, h₂, hu⟩ | ⟨b, h₂, rfl⟩
    · exact ⟨_, .condErr h₂, ResSim.ub hu _⟩
    · obtain ⟨r₂, hx₂, hr⟩ := CondExec.sim hw hx _ h
      exact ⟨_, .skipped h₂ hx₂, hr⟩

theorem SDFGExec.sim {ctx : Ctx} (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {ρ : Valuation} {m₁ : Mem}
    {g : SDFG} {r₁ : Res (Valuation × Mem × Log)} :
    SDFGExec ctx ρ m₁ g r₁ → ∀ m₂, Mem.Sim π m₂ m₁ →
      ∃ r₂, SDFGExec (ctx.ren π) ρ m₂ g r₂ ∧ ResSim (InvSim π) r₂ r₁
  | .err hx, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx { ρ := g.withConstants ρ, mem := m₂ } ⟨rfl, h, rfl⟩
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .err hx₂, ResSim.of_err hu⟩
  | .escape hx hc, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx { ρ := g.withConstants ρ, mem := m₂ } ⟨rfl, h, rfl⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .err hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .escape hx₂ hc, ResSim.err _⟩
  | .run hx hc, m₂, h => by
    obtain ⟨r₂, hx₂, hr⟩ := RegionExec.sim hw hx { ρ := g.withConstants ρ, mem := m₂ } ⟨rfl, h, rfl⟩
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨⟨ctl₂, d₂⟩, rfl, hctl, hd⟩
    · exact ⟨_, .err hx₂, ResSim.ub hu _⟩
    · simp only at hctl hd
      subst hctl
      exact ⟨_, .run hx₂ hc, ResSim.ok ⟨hd.ρ, hd.mem, hd.log⟩⟩

end

end Rel

end Dace
