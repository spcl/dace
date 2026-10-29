-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Facts
import Dace.Core.Parse

/-!
# Polynomial normal form of integer expressions

The verified symbolic reasoning of the formalization works on polynomials
with integer coefficients over **atoms**: symbols, opaque integer subterms
(floor/ceiling division, modulo, `Min`, `Max`, `Abs`) and auxiliary variables
that clients introduce (iteration indices, witnesses).

* `Expr.toPoly?` succeeds only on *integer-closed* expressions: integer
  literals, symbols, `+ - *`, unary minus, `floor`/`ceiling` (the identity on
  integers), powers with a literal non-negative exponent, and the opaque
  operators above.  Division `/`, floats and booleans are rejected, which rules
  out traps such as `2*x` with `x = 1/2`.
* `toPoly?_sound`: when all free symbols are bound to integers and the
  expression evaluates, its value is the polynomial evaluated at `atomEnv`.
* The arguments of an opaque atom are stored in normal form (`Poly.toExpr`),
  so `Min(i + T, N)` and `Min(T + i, N)` are the same atom.
* `Atom.axioms`: the defining (in)equalities of an opaque atom in terms of the
  polynomials of its arguments (`c·q ≤ x ≤ c·q + c − 1` for `q = x // c`,
  `c > 0`, …), sound by `atomEnv_axioms`.  Disjunctive facts (`Min` equals
  one of its arguments, case split on the sign of a symbolic divisor) are
  `Disj`s, consumed by case splits of certificates (`Core/SymCert`).

Normal form: a monomial is a list of atoms sorted by `Atom.key`; a polynomial
is a list of (monomial, non-zero coefficient) pairs with like terms combined,
sorted by degree, then key.  Canonicity is only needed for completeness: all
soundness lemmas hold for any list.
-/

namespace Dace

set_option autoImplicit false

/-! ## Atoms, monomials, polynomials -/

/-- Atoms: symbols, opaque integer subterms, auxiliary variables. -/
inductive Atom where
  | sym (s : Sym)
  | opq (e : Expr)
  | var (n : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- Assignment of integers to atoms. -/
abbrev Env := Atom → Int

namespace Atom

/-- Sort key (also used for display). -/
def key : Atom → String
  | sym s => s
  | opq e => "{" ++ e.toPy ++ "}"
  | var n => s!"#{n}"

/-- The expression an atom stands for (auxiliary variables have none and are
mapped to `0`). -/
def expr : Atom → Expr
  | sym s => .sym s
  | opq e => e
  | var _ => .lit (.int 0)

/-- Auxiliary variables. -/
def isVar : Atom → Bool
  | var _ => true
  | _ => false

/-- Key order (a total preorder). -/
def keyLe (a b : Atom) : Bool := !decide (b.key < a.key)

end Atom

/-- A monomial: a product of atoms (sorted by key in normal form). -/
abbrev Mono := List Atom

namespace Mono

def eval (env : Env) (m : Mono) : Int := (m.map env).prod

def norm (m : Mono) : Mono := m.mergeSort Atom.keyLe

def key (m : Mono) : String := "*".intercalate (m.map Atom.key)

/-- Normal-form order of monomials: by degree, then key. -/
def le (m n : Mono) : Bool :=
  m.length < n.length || (m.length == n.length && !decide (n.key < m.key))

@[simp] theorem eval_nil (env : Env) : eval env [] = 1 := rfl

@[simp] theorem eval_cons (env : Env) (a : Atom) (m : Mono) :
    eval env (a :: m) = env a * eval env m := by simp [eval]

theorem eval_append (env : Env) (m n : Mono) : eval env (m ++ n) = eval env m * eval env n := by
  simp [eval, List.prod_append]

theorem eval_norm (env : Env) (m : Mono) : eval env (norm m) = eval env m := by
  unfold eval norm
  exact ((List.mergeSort_perm m Atom.keyLe).map env).prod_eq

end Mono

/-- A polynomial: a list of (monomial, coefficient) terms. -/
abbrev Poly := List (Mono × Int)

namespace Poly

def eval (env : Env) (p : Poly) : Int := (p.map fun t => t.2 * Mono.eval env t.1).sum

@[simp] theorem eval_nil (env : Env) : eval env [] = 0 := rfl

@[simp] theorem eval_cons (env : Env) (t : Mono × Int) (p : Poly) :
    eval env (t :: p) = t.2 * Mono.eval env t.1 + eval env p := by simp [eval]

theorem eval_append (env : Env) (p q : Poly) : eval env (p ++ q) = eval env p + eval env q := by
  simp [eval, List.sum_append]

/-- Add a term, combining it with an equal monomial if present. -/
def addTerm (t : Mono × Int) : Poly → Poly
  | [] => [t]
  | u :: p => if u.1 = t.1 then (u.1, u.2 + t.2) :: p else u :: addTerm t p

theorem eval_addTerm (env : Env) (t : Mono × Int) (p : Poly) :
    eval env (addTerm t p) = t.2 * Mono.eval env t.1 + eval env p := by
  induction p with
  | nil => simp [addTerm]
  | cons u p ih =>
    unfold addTerm
    split
    · rename_i h; simp [h]; ring
    · simp [ih]; ring

/-- Combine like terms. -/
def combine (p : Poly) : Poly := p.foldr addTerm []

theorem eval_combine (env : Env) (p : Poly) : eval env (combine p) = eval env p := by
  induction p with
  | nil => rfl
  | cons t p ih => simp [combine, eval_addTerm] at ih ⊢; rw [ih]

theorem eval_filter_nonzero (env : Env) (p : Poly) :
    eval env (p.filter fun t => t.2 != 0) = eval env p := by
  induction p with
  | nil => rfl
  | cons t p ih =>
    by_cases h : t.2 = 0
    · simp [h, ih]
    · simp [h, ih]

theorem eval_perm (env : Env) {p q : Poly} (h : p.Perm q) : eval env p = eval env q :=
  (h.map _).sum_eq

/-- Normal form: sorted monomials, like terms combined, zeros dropped, terms
sorted. -/
def norm (p : Poly) : Poly :=
  ((combine (p.map fun t => (Mono.norm t.1, t.2))).filter fun t => t.2 != 0).mergeSort
    fun t u => Mono.le t.1 u.1

theorem eval_map_norm (env : Env) (p : Poly) :
    eval env (p.map fun t => (Mono.norm t.1, t.2)) = eval env p := by
  induction p with
  | nil => rfl
  | cons t p ih => simp [Mono.eval_norm, ih]

@[simp] theorem eval_norm (env : Env) (p : Poly) : eval env (norm p) = eval env p := by
  unfold norm
  rw [eval_perm env (List.mergeSort_perm _ _), eval_filter_nonzero, eval_combine, eval_map_norm]

/-! ### Operations -/

def const (c : Int) : Poly := if c = 0 then [] else [([], c)]

def var (a : Atom) : Poly := [([a], 1)]

def neg (p : Poly) : Poly := p.map fun t => (t.1, -t.2)

def add (p q : Poly) : Poly := norm (p ++ q)

def sub (p q : Poly) : Poly := add p (neg q)

def smul (c : Int) (p : Poly) : Poly := norm (p.map fun t => (t.1, c * t.2))

def mulRaw (p q : Poly) : Poly := p.flatMap fun t => q.map fun u => (t.1 ++ u.1, t.2 * u.2)

def mul (p q : Poly) : Poly := norm (mulRaw p q)

def pow (p : Poly) : Nat → Poly
  | 0 => const 1
  | n + 1 => mul (pow p n) p

/-- The value of a constant polynomial. -/
def const? : Poly → Option Int
  | [] => some 0
  | [([], c)] => some c
  | _ => none

/-- Is the polynomial a negative constant? -/
def isNegConst (p : Poly) : Bool :=
  match const? p with
  | some c => decide (c < 0)
  | none => false

/-- All atoms of a polynomial. -/
def atoms (p : Poly) : List Atom := p.flatMap (·.1)

@[simp] theorem eval_const (env : Env) (c : Int) : eval env (const c) = c := by
  unfold const; split <;> simp [*]

@[simp] theorem eval_var (env : Env) (a : Atom) : eval env (var a) = env a := by
  simp [var]

@[simp] theorem eval_neg (env : Env) (p : Poly) : eval env (neg p) = - eval env p := by
  induction p with
  | nil => rfl
  | cons t p ih => simp [neg] at ih ⊢; rw [ih]; ring

@[simp] theorem eval_add (env : Env) (p q : Poly) : eval env (add p q) = eval env p + eval env q := by
  simp [add, eval_append]

@[simp] theorem eval_sub (env : Env) (p q : Poly) : eval env (sub p q) = eval env p - eval env q := by
  simp [sub]; ring

@[simp] theorem eval_smul (env : Env) (c : Int) (p : Poly) : eval env (smul c p) = c * eval env p := by
  unfold smul; rw [eval_norm]
  induction p with
  | nil => simp
  | cons t p ih => simp at ih ⊢; rw [ih]; ring

theorem eval_mulRaw (env : Env) (p q : Poly) : eval env (mulRaw p q) = eval env p * eval env q := by
  induction p with
  | nil => simp [mulRaw]
  | cons t p ih =>
    have hq : ∀ q : Poly, eval env (q.map fun u => (t.1 ++ u.1, t.2 * u.2)) =
        t.2 * Mono.eval env t.1 * eval env q := by
      intro q
      induction q with
      | nil => simp
      | cons u q ihq => simp [Mono.eval_append] at ihq ⊢; rw [ihq]; ring
    simp only [mulRaw, List.flatMap_cons, eval_append] at ih ⊢
    rw [hq, ih]; simp; ring

@[simp] theorem eval_mul (env : Env) (p q : Poly) : eval env (mul p q) = eval env p * eval env q := by
  simp [mul, eval_mulRaw]

@[simp] theorem eval_pow (env : Env) (p : Poly) (n : Nat) : eval env (pow p n) = eval env p ^ n := by
  induction n with
  | zero => simp [pow]
  | succ n ih => simp [pow, ih, pow_succ]

theorem eval_of_const? (env : Env) {p : Poly} {c : Int} (h : const? p = some c) : eval env p = c := by
  unfold const? at h
  split at h <;> simp_all

theorem isNegConst_eval (env : Env) {p : Poly} (h : isNegConst p = true) : eval env p < 0 := by
  unfold isNegConst at h
  split at h
  · rename_i c hc; rw [eval_of_const? env hc]; simpa using h
  · simp at h

/-! ### Atoms of polynomials -/

/-- Every atom of the polynomial satisfies `P`. -/
def AllAtoms (P : Atom → Prop) (p : Poly) : Prop := ∀ t ∈ p, ∀ a ∈ t.1, P a

variable {P : Atom → Prop}

theorem allAtoms_append {p q : Poly} (hp : AllAtoms P p) (hq : AllAtoms P q) : AllAtoms P (p ++ q) := by
  intro t ht a ha
  rcases List.mem_append.1 ht with h | h
  · exact hp t h a ha
  · exact hq t h a ha

theorem allAtoms_addTerm {t : Mono × Int} {p : Poly} (ht : ∀ a ∈ t.1, P a) (hp : AllAtoms P p) :
    AllAtoms P (addTerm t p) := by
  induction p with
  | nil => intro u hu; simp [addTerm] at hu; subst hu; exact ht
  | cons u p ih =>
    unfold addTerm
    split
    · intro v hv a ha
      simp at hv
      rcases hv with rfl | hv
      · exact hp u (by simp) a ha
      · exact hp v (by simp [hv]) a ha
    · intro v hv a ha
      simp at hv
      rcases hv with rfl | hv
      · exact hp v (by simp) a ha
      · exact ih (fun w hw => hp w (by simp [hw])) v hv a ha

theorem allAtoms_norm {p : Poly} (hp : AllAtoms P p) : AllAtoms P (norm p) := by
  have h1 : AllAtoms P (p.map fun t => (Mono.norm t.1, t.2)) := by
    intro t ht a ha
    simp only [List.mem_map] at ht
    obtain ⟨u, hu, rfl⟩ := ht
    exact hp u hu a ((List.mergeSort_perm u.1 Atom.keyLe).mem_iff.1 ha)
  have h2 : ∀ q : Poly, AllAtoms P q → AllAtoms P (combine q) := by
    intro q hq
    induction q with
    | nil => intro t ht; simp [combine] at ht
    | cons t q ih =>
      exact allAtoms_addTerm (hq t (by simp)) (ih fun u hu => hq u (by simp [hu]))
  intro t ht a ha
  have ht' := (List.mergeSort_perm _ _).mem_iff.1 ht
  exact h2 _ h1 t (List.mem_filter.1 ht').1 a ha

theorem allAtoms_const (c : Int) : AllAtoms P (const c) := by
  intro t ht a ha; unfold const at ht; split at ht <;> simp_all

theorem allAtoms_var {a : Atom} (h : P a) : AllAtoms P (var a) := by
  intro t ht b hb; simp [var] at ht; subst ht; simp at hb; subst hb; exact h

theorem allAtoms_neg {p : Poly} (hp : AllAtoms P p) : AllAtoms P (neg p) := by
  intro t ht a ha
  simp only [neg, List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact hp u hu a ha

theorem allAtoms_add {p q : Poly} (hp : AllAtoms P p) (hq : AllAtoms P q) : AllAtoms P (add p q) :=
  allAtoms_norm (allAtoms_append hp hq)

theorem allAtoms_sub {p q : Poly} (hp : AllAtoms P p) (hq : AllAtoms P q) : AllAtoms P (sub p q) :=
  allAtoms_add hp (allAtoms_neg hq)

theorem allAtoms_smul (c : Int) {p : Poly} (hp : AllAtoms P p) : AllAtoms P (smul c p) := by
  apply allAtoms_norm
  intro t ht a ha
  simp only [List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact hp u hu a ha

theorem allAtoms_mul {p q : Poly} (hp : AllAtoms P p) (hq : AllAtoms P q) : AllAtoms P (mul p q) := by
  apply allAtoms_norm
  intro t ht a ha
  simp only [mulRaw, List.mem_flatMap, List.mem_map] at ht
  obtain ⟨u, hu, v, hv, rfl⟩ := ht
  rcases List.mem_append.1 ha with h | h
  · exact hp u hu a h
  · exact hq v hv a h

theorem allAtoms_pow {p : Poly} (hp : AllAtoms P p) (n : Nat) : AllAtoms P (pow p n) := by
  induction n with
  | zero => exact allAtoms_const 1
  | succ n ih => exact allAtoms_mul ih hp

theorem allAtoms_mono {Q : Atom → Prop} (h : ∀ a, P a → Q a) {p : Poly} (hp : AllAtoms P p) :
    AllAtoms Q p := fun t ht a ha => h a (hp t ht a ha)

theorem allAtoms_iff {p : Poly} : AllAtoms P p ↔ ∀ a ∈ atoms p, P a := by
  simp only [AllAtoms, atoms, List.mem_flatMap]
  constructor
  · rintro h a ⟨t, ht, ha⟩; exact h t ht a ha
  · intro h t ht a ha; exact h a ⟨t, ht, ha⟩

theorem mono_eval_congr {env env' : Env} : ∀ {m : Mono}, (∀ a ∈ m, env a = env' a) →
    Mono.eval env m = Mono.eval env' m
  | [], _ => rfl
  | a :: m, h => by
    simp only [Mono.eval_cons]
    rw [h a (by simp), mono_eval_congr (fun b hb => h b (List.mem_cons_of_mem _ hb))]

theorem eval_congr {env env' : Env} : ∀ {p : Poly}, AllAtoms (fun a => env a = env' a) p →
    eval env p = eval env' p
  | [], _ => rfl
  | t :: p, h => by
    simp only [eval_cons]
    rw [mono_eval_congr (h t (by simp)), eval_congr (fun u hu => h u (List.mem_cons_of_mem _ hu))]

end Poly

/-! ## Back to expressions -/

/-- A monomial as an expression (product of the atoms' expressions). -/
def Mono.toExpr : Mono → Expr
  | [] => .lit (.int 1)
  | [a] => a.expr
  | a :: b :: m => .bin .mul a.expr (Mono.toExpr (b :: m))

/-- A term as an expression. -/
def Poly.termExpr (t : Mono × Int) : Expr :=
  if t.1 = [] then .lit (.int t.2)
  else if t.2 = 1 then Mono.toExpr t.1
  else .bin .mul (.lit (.int t.2)) (Mono.toExpr t.1)

/-- A polynomial as an expression (used for the canonical arguments of opaque
atoms and for display). -/
def Poly.toExpr : Poly → Expr
  | [] => .lit (.int 0)
  | [t] => Poly.termExpr t
  | t :: u :: p => .bin .add (Poly.termExpr t) (Poly.toExpr (u :: p))

/-- Symbol `s` is bound to an integer. -/
def IntBound (ρ : Valuation) (s : Sym) : Prop := ∃ n : Int, ρ s = some (.int n)

/-- The atom environment of a valuation: symbols and opaque subterms take
their integer values (`0` when undefined), auxiliary variables `0`. -/
def atomEnv (I : ExprInterp) (ρ : Valuation) : Env
  | .sym s => ((ρ s).bind Val.toInt?).getD 0
  | .opq e => (e.evalInt I ρ).getD 0
  | .var _ => 0

/-- An atom is *defined* under a valuation: it is not an auxiliary variable,
its expression evaluates to the integer `atomEnv` assigns it, and all its free
symbols are integers. -/
def AtomOk (I : ExprInterp) (ρ : Valuation) (a : Atom) : Prop :=
  a.isVar = false ∧ a.expr.eval I ρ = some (.int (atomEnv I ρ a)) ∧
    ∀ s ∈ a.expr.freeSyms, IntBound ρ s

section ToExpr

variable {I : ExprInterp} {ρ : Valuation}

theorem Mono.toExpr_eval : ∀ (m : Mono), (∀ a ∈ m, AtomOk I ρ a) →
    (Mono.toExpr m).eval I ρ = some (.int (Mono.eval (atomEnv I ρ) m))
  | [], _ => by simp [Mono.toExpr, Expr.eval]
  | [a], h => by simpa [Mono.toExpr] using (h a (by simp)).2.1
  | a :: b :: m, h => by
    have ih := Mono.toExpr_eval (b :: m) (fun c hc => h c (List.mem_cons_of_mem _ hc))
    rw [Mono.toExpr, Expr.eval, (h a (by simp)).2.1, ih]
    simp [Eval.arith, Eval.intArith]

theorem Mono.freeSyms_toExpr : ∀ (m : Mono) (s : Sym), s ∈ (Mono.toExpr m).freeSyms →
    ∃ a ∈ m, s ∈ a.expr.freeSyms
  | [], s, h => by simp [Mono.toExpr, Expr.freeSyms] at h
  | [a], s, h => ⟨a, by simp, by simpa [Mono.toExpr] using h⟩
  | a :: b :: m, s, h => by
    simp only [Mono.toExpr, Expr.freeSyms, List.mem_append] at h
    rcases h with h | h
    · exact ⟨a, by simp, h⟩
    · obtain ⟨c, hc, hs⟩ := Mono.freeSyms_toExpr (b :: m) s h
      exact ⟨c, List.mem_cons_of_mem _ hc, hs⟩

theorem Poly.termExpr_eval (t : Mono × Int) (h : ∀ a ∈ t.1, AtomOk I ρ a) :
    (Poly.termExpr t).eval I ρ = some (.int (t.2 * Mono.eval (atomEnv I ρ) t.1)) := by
  unfold Poly.termExpr
  split
  · rename_i h0; simp [h0, Expr.eval]
  · split
    · rename_i h1; rw [Mono.toExpr_eval _ h]; simp [h1]
    · rw [Expr.eval, Mono.toExpr_eval _ h]
      simp [Expr.eval, Eval.arith, Eval.intArith]

theorem Poly.freeSyms_termExpr (t : Mono × Int) (s : Sym) (h : s ∈ (Poly.termExpr t).freeSyms) :
    ∃ a ∈ t.1, s ∈ a.expr.freeSyms := by
  unfold Poly.termExpr at h
  split at h
  · simp [Expr.freeSyms] at h
  · split at h
    · exact Mono.freeSyms_toExpr _ _ h
    · simp only [Expr.freeSyms, List.nil_append] at h
      exact Mono.freeSyms_toExpr _ _ h

theorem Poly.toExpr_eval : ∀ (p : Poly), p.AllAtoms (AtomOk I ρ) →
    (Poly.toExpr p).eval I ρ = some (.int (p.eval (atomEnv I ρ)))
  | [], _ => by simp [Poly.toExpr, Expr.eval]
  | [t], h => by
    rw [Poly.toExpr, Poly.termExpr_eval t (h t (by simp))]; simp
  | t :: u :: p, h => by
    have ih := Poly.toExpr_eval (u :: p) (fun v hv => h v (List.mem_cons_of_mem _ hv))
    rw [Poly.toExpr, Expr.eval, Poly.termExpr_eval t (h t (by simp)), ih]
    simp [Eval.arith, Eval.intArith]

theorem Poly.freeSyms_toExpr : ∀ (p : Poly) (s : Sym), s ∈ (Poly.toExpr p).freeSyms →
    ∃ t ∈ p, ∃ a ∈ t.1, s ∈ a.expr.freeSyms
  | [], s, h => by simp [Poly.toExpr, Expr.freeSyms] at h
  | [t], s, h => ⟨t, by simp, Poly.freeSyms_termExpr t s (by simpa [Poly.toExpr] using h)⟩
  | t :: u :: p, s, h => by
    simp only [Poly.toExpr, Expr.freeSyms, List.mem_append] at h
    rcases h with h | h
    · exact ⟨t, by simp, Poly.freeSyms_termExpr t s h⟩
    · obtain ⟨v, hv, hs⟩ := Poly.freeSyms_toExpr (u :: p) s h
      exact ⟨v, List.mem_cons_of_mem _ hv, hs⟩

/-- The free symbols of a normal-form expression are integers when its atoms
are defined. -/
theorem Poly.toExpr_intBound {p : Poly} (h : p.AllAtoms (AtomOk I ρ)) :
    ∀ s ∈ (Poly.toExpr p).freeSyms, IntBound ρ s := by
  intro s hs
  obtain ⟨t, ht, a, ha, hs⟩ := Poly.freeSyms_toExpr p s hs
  exact (h t ht a ha).2.2 s hs

/-- An opaque atom whose expression evaluates to an integer is defined. -/
theorem atomOk_opq {E : Expr} {v : Int} (h : E.eval I ρ = some (.int v))
    (hs : ∀ s ∈ E.freeSyms, IntBound ρ s) : AtomOk I ρ (.opq E) ∧ atomEnv I ρ (.opq E) = v := by
  have hv : atomEnv I ρ (.opq E) = v := by
    simp [atomEnv, Expr.evalInt, h, Val.toInt?]
  exact ⟨⟨rfl, by simp [Atom.expr, h, hv], hs⟩, hv⟩

end ToExpr

/-! ## Integer-closed expressions to polynomials -/

namespace Poly

/-- Polynomial of a unary operator applied to an integer-closed operand. -/
def ofUn? (op : UnOp) (pa : Poly) : Option Poly :=
  match op with
  | .neg => some (neg pa)
  | .floor | .ceil => some pa
  | .abs => some (var (.opq (.un .abs pa.toExpr)))
  | _ => none

/-- Polynomial of a binary operator applied to integer-closed operands.
Powers need a literal exponent in `0..64`. -/
def ofBin? (op : BinOp) (pa pb : Poly) : Option Poly :=
  match op with
  | .add => some (add pa pb)
  | .sub => some (sub pa pb)
  | .mul => some (mul pa pb)
  | .pow =>
    match const? pb with
    | some n => if 0 ≤ n ∧ n ≤ 64 then some (pow pa n.toNat) else none
    | none => none
  | .intFloor | .intCeil | .mod | .min | .max => some (var (.opq (.bin op pa.toExpr pb.toExpr)))
  | _ => none

end Poly

/-- The polynomial of an integer-closed expression; `none` otherwise. -/
def Expr.toPoly? : Expr → Option Poly
  | .lit (.int v) => some (Poly.const v)
  | .sym s => some (Poly.var (.sym s))
  | .un op a => a.toPoly? >>= Poly.ofUn? op
  | .bin op a b => do Poly.ofBin? op (← a.toPoly?) (← b.toPoly?)
  | .approx a _ => a.toPoly?
  | _ => none

section Sound

variable {I : ExprInterp} {ρ : Valuation}

/-- Both halves of the soundness of `toPoly?`: an integer-closed expression
whose free symbols are integers evaluates (if at all) to the integer the
polynomial denotes under `atomEnv`, and all atoms of the polynomial are
defined. -/
theorem toPoly?_spec : ∀ (e : Expr) (p : Poly), e.toPoly? = some p →
    (∀ s ∈ e.freeSyms, IntBound ρ s) → ∀ w, e.eval I ρ = some w →
    w = .int (p.eval (atomEnv I ρ)) ∧ p.AllAtoms (AtomOk I ρ)
  | .lit (.int v), p, hp, _, w, hw => by
    simp only [Expr.toPoly?, Option.some.injEq] at hp
    simp only [Expr.eval, Option.some.injEq] at hw
    subst hp hw
    exact ⟨by simp, Poly.allAtoms_const v⟩
  | .sym s, p, hp, hs, w, hw => by
    simp only [Expr.toPoly?, Option.some.injEq] at hp
    simp only [Expr.eval] at hw
    subst hp
    obtain ⟨n, hn⟩ := hs s (by simp [Expr.freeSyms])
    rw [hn, Option.some.injEq] at hw
    subst hw
    refine ⟨by simp [atomEnv, hn, Val.toInt?], Poly.allAtoms_var ⟨rfl, ?_, ?_⟩⟩
    · simp [Atom.expr, Expr.eval, atomEnv, hn, Val.toInt?]
    · intro s' hs'
      simp [Atom.expr, Expr.freeSyms] at hs'
      subst hs'
      exact ⟨n, hn⟩
  | .un op a, p, hp, hs, w, hw => by
    simp only [Expr.toPoly?, bind, Option.bind_eq_some_iff] at hp
    obtain ⟨pa, hpa, hop⟩ := hp
    simp only [Expr.eval, bind, Option.bind_eq_some_iff] at hw
    obtain ⟨wa, hwa, hu⟩ := hw
    have hsa : ∀ s ∈ a.freeSyms, IntBound ρ s := fun s h => hs s (by simpa [Expr.freeSyms] using h)
    obtain ⟨rfl, hok⟩ := toPoly?_spec a pa hpa hsa wa hwa
    have hE := Poly.toExpr_eval pa hok
    have hEs := Poly.toExpr_intBound hok
    cases op <;> simp only [Poly.ofUn?, reduceCtorEq, Option.some.injEq] at hop <;> subst hop <;>
      simp only [Eval.unary, Option.some.injEq] at hu <;> subst hu
    · exact ⟨by simp, Poly.allAtoms_neg hok⟩
    · obtain ⟨hok', hv⟩ := atomOk_opq (I := I) (ρ := ρ) (E := .un .abs pa.toExpr) (v := |pa.eval (atomEnv I ρ)|)
        (by simp [Expr.eval, hE, Eval.unary]) (by simpa [Expr.freeSyms] using hEs)
      exact ⟨by simp [hv], Poly.allAtoms_var hok'⟩
    · exact ⟨rfl, hok⟩
    · exact ⟨rfl, hok⟩
  | .bin op a b, p, hp, hs, w, hw => by
    simp only [Expr.toPoly?, bind, Option.bind_eq_some_iff] at hp
    obtain ⟨pa, hpa, pb, hpb, hop⟩ := hp
    simp only [Expr.eval, bind, Option.bind_eq_some_iff] at hw
    obtain ⟨wa, hwa, wb, hwb, hu⟩ := hw
    have hsa : ∀ s ∈ a.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Expr.freeSyms, h])
    have hsb : ∀ s ∈ b.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Expr.freeSyms, h])
    obtain ⟨rfl, hoka⟩ := toPoly?_spec a pa hpa hsa wa hwa
    obtain ⟨rfl, hokb⟩ := toPoly?_spec b pb hpb hsb wb hwb
    have hEa := Poly.toExpr_eval pa hoka
    have hEb := Poly.toExpr_eval pb hokb
    have hEs : ∀ s ∈ (Expr.bin op pa.toExpr pb.toExpr).freeSyms, IntBound ρ s := by
      intro s h
      simp only [Expr.freeSyms, List.mem_append] at h
      rcases h with h | h
      · exact Poly.toExpr_intBound hoka s h
      · exact Poly.toExpr_intBound hokb s h
    -- the opaque cases share one argument
    have hopq : ∀ v : Int, Eval.intArith op (pa.eval (atomEnv I ρ)) (pb.eval (atomEnv I ρ)) =
        some (.int v) → p = Poly.var (.opq (.bin op pa.toExpr pb.toExpr)) →
        Val.int v = .int (p.eval (atomEnv I ρ)) ∧ p.AllAtoms (AtomOk I ρ) := by
      intro v hv hp
      obtain ⟨hok', hval⟩ := atomOk_opq (I := I) (ρ := ρ) (E := .bin op pa.toExpr pb.toExpr)
        (by simp only [Expr.eval, hEa, hEb, bind, Option.bind_some, Eval.arith]; exact hv) hEs
      subst hp
      exact ⟨by simp [hval], Poly.allAtoms_var hok'⟩
    simp only [Eval.arith] at hu
    cases op <;> simp only [Poly.ofBin?, reduceCtorEq, Option.some.injEq] at hop
    · subst hop; simp only [Eval.intArith, Option.some.injEq] at hu; subst hu
      exact ⟨by simp, Poly.allAtoms_add hoka hokb⟩
    · subst hop; simp only [Eval.intArith, Option.some.injEq] at hu; subst hu
      exact ⟨by simp, Poly.allAtoms_sub hoka hokb⟩
    · subst hop; simp only [Eval.intArith, Option.some.injEq] at hu; subst hu
      exact ⟨by simp, Poly.allAtoms_mul hoka hokb⟩
    · -- intFloor
      simp only [Eval.intArith, Eval.intFloor] at hu
      split at hu
      · simp at hu
      · simp only [Option.map_some, Option.some.injEq] at hu
        subst hu
        exact hopq _ (by simp [Eval.intArith, Eval.intFloor, *]) hop.symm
    · -- intCeil
      simp only [Eval.intArith, Eval.intCeil] at hu
      split at hu
      · simp at hu
      · simp only [Option.map_some, Option.some.injEq] at hu
        subst hu
        exact hopq _ (by simp [Eval.intArith, Eval.intCeil, *]) hop.symm
    · -- mod
      simp only [Eval.intArith, Eval.pyMod] at hu
      split at hu
      · simp at hu
      · simp only [Option.map_some, Option.some.injEq] at hu
        subst hu
        exact hopq _ (by simp [Eval.intArith, Eval.pyMod, *]) hop.symm
    · -- pow
      split at hop
      · rename_i n hn
        split at hop
        · rename_i hrange
          simp only [Option.some.injEq] at hop
          subst hop
          have hb : pb.eval (atomEnv I ρ) = n := Poly.eval_of_const? _ hn
          rw [hb] at hu
          simp only [Eval.intArith, show ¬ n < 0 from by omega, ite_false, Option.some.injEq] at hu
          subst hu
          exact ⟨by simp, Poly.allAtoms_pow hoka _⟩
        · simp at hop
      · simp at hop
    · subst hop; simp only [Eval.intArith, Option.some.injEq] at hu; subst hu
      exact hopq _ (by simp [Eval.intArith]) rfl
    · subst hop; simp only [Eval.intArith, Option.some.injEq] at hu; subst hu
      exact hopq _ (by simp [Eval.intArith]) rfl
  | .approx a o, p, hp, hs, w, hw => by
    simp only [Expr.toPoly?] at hp
    simp only [Expr.eval] at hw
    exact toPoly?_spec a p hp (fun s h => hs s (by simp [Expr.freeSyms, h])) w hw
  | .lit (.float _ _), p, hp, _, _, _ => by simp [Expr.toPoly?] at hp
  | .lit (.bool _), p, hp, _, _, _ => by simp [Expr.toPoly?] at hp
  | .ite _ _ _, p, hp, _, _, _ => by simp [Expr.toPoly?] at hp
  | .app _ _, p, hp, _, _, _ => by simp [Expr.toPoly?] at hp
  | .subscript _ _, p, hp, _, _, _ => by simp [Expr.toPoly?] at hp
  | .attr _ _, p, hp, _, _, _ => by simp [Expr.toPoly?] at hp

/-- **Soundness of the polynomial normal form.**  If `e` is integer-closed
(`toPoly? e = some p`), all its free symbols are bound to integers and it
evaluates to the integer `v`, then `v` is the value of `p` at `atomEnv`. -/
theorem toPoly?_sound {e : Expr} {p : Poly} (hp : e.toPoly? = some p)
    (hs : ∀ s ∈ e.freeSyms, ∃ n : Int, ρ s = some (.int n)) {v : Int} (hv : e.evalInt I ρ = some v) :
    v = p.eval (atomEnv I ρ) := by
  simp only [Expr.evalInt, bind, Option.bind_eq_some_iff] at hv
  obtain ⟨w, hw, hv⟩ := hv
  obtain ⟨rfl, -⟩ := toPoly?_spec e p hp hs w hw
  simpa [Val.toInt?] using hv.symm

/-- The atoms of the polynomial of an evaluating integer-closed expression
are defined. -/
theorem toPoly?_atomOk {e : Expr} {p : Poly} (hp : e.toPoly? = some p)
    (hs : ∀ s ∈ e.freeSyms, IntBound ρ s) {w : Val} (hw : e.eval I ρ = some w) :
    p.AllAtoms (AtomOk I ρ) :=
  (toPoly?_spec e p hp hs w hw).2

end Sound

/-! ## Inequalities and disjunctions -/

inductive IneqKind where
  | ge  -- `p ≥ 0`
  | eq  -- `p = 0`
  deriving DecidableEq, Repr, Inhabited

/-- `poly ≥ 0` or `poly = 0`.  Strict inequalities are tightened over ℤ
(`a < b` is `b − a − 1 ≥ 0`). -/
structure Ineq where
  kind : IneqKind
  poly : Poly
  deriving DecidableEq, Repr, Inhabited

namespace Ineq

def holds (env : Env) (h : Ineq) : Prop :=
  match h.kind with
  | .ge => 0 ≤ h.poly.eval env
  | .eq => h.poly.eval env = 0

def ge (p : Poly) : Ineq := ⟨.ge, p⟩

/-- `a ≤ b` -/
def le (a b : Poly) : Ineq := ge (Poly.sub b a)

/-- `a < b`, tightened to `a + 1 ≤ b`. -/
def lt (a b : Poly) : Ineq := ge (Poly.sub (Poly.sub b a) (Poly.const 1))

/-- `a = b` -/
def eqOf (a b : Poly) : Ineq := ⟨.eq, Poly.sub a b⟩

@[simp] theorem holds_le (env : Env) (a b : Poly) : (le a b).holds env ↔ a.eval env ≤ b.eval env := by
  simp [le, ge, holds]

@[simp] theorem holds_lt (env : Env) (a b : Poly) : (lt a b).holds env ↔ a.eval env < b.eval env := by
  simp [lt, ge, holds]; omega

@[simp] theorem holds_eqOf (env : Env) (a b : Poly) : (eqOf a b).holds env ↔ a.eval env = b.eval env := by
  simp [eqOf, holds]; omega

@[simp] theorem holds_ge (env : Env) (p : Poly) : (ge p).holds env ↔ 0 ≤ p.eval env := by
  simp [ge, holds]

section Atoms

variable {P : Atom → Prop} {a b : Poly}

theorem le_atoms (ha : a.AllAtoms P) (hb : b.AllAtoms P) : (le a b).poly.AllAtoms P :=
  Poly.allAtoms_sub hb ha

theorem lt_atoms (ha : a.AllAtoms P) (hb : b.AllAtoms P) : (lt a b).poly.AllAtoms P :=
  Poly.allAtoms_sub (Poly.allAtoms_sub hb ha) (Poly.allAtoms_const 1)

theorem eqOf_atoms (ha : a.AllAtoms P) (hb : b.AllAtoms P) : (eqOf a b).poly.AllAtoms P :=
  Poly.allAtoms_sub ha hb

end Atoms

end Ineq

/-- A disjunction of two conjunctions. -/
abbrev Disj := List Ineq × List Ineq

def Disj.holds (env : Env) (d : Disj) : Prop :=
  (∀ h ∈ d.1, h.holds env) ∨ (∀ h ∈ d.2, h.holds env)

/-- Every polynomial of a list of inequalities and of a list of disjunctions
has only atoms satisfying `P`. -/
def AxiomsAtoms (P : Atom → Prop) (ax : List Ineq × List Disj) : Prop :=
  (∀ h ∈ ax.1, h.poly.AllAtoms P) ∧ (∀ d ∈ ax.2, ∀ h ∈ d.1 ++ d.2, h.poly.AllAtoms P)

/-! ## Defining axioms of opaque atoms -/

namespace Atom

open Poly Ineq

/-- Range of a remainder `r` of a division by `y`: `0 ≤ r ≤ y − 1` if
`y > 0`, `y + 1 ≤ r ≤ 0` if `y < 0` (a case split when `y` is not
constant). -/
def remAxioms (r y : Poly) : List Ineq × List Disj :=
  match const? y with
  | some c =>
    if 0 < c then ([le (const 0) r, le r (sub y (const 1))], [])
    else if c < 0 then ([le (add y (const 1)) r, le r (const 0)], [])
    else ([], [])
  | none =>
    ([], [([lt (const 0) y, le (const 0) r, le r (sub y (const 1))],
           [lt y (const 0), le (add y (const 1)) r, le r (const 0)])])

/-- Axioms of an opaque binary atom `q = op(A, B)` with `x`, `y` the
polynomials of `A`, `B`. -/
def binAxioms (op : BinOp) (A B : Expr) (x y q : Poly) : List Ineq × List Disj :=
  match op with
  | .min => ([le q x, le q y], [([le x y, eqOf q x], [lt y x, eqOf q y])])
  | .max => ([le x q, le y q], [([le y x, eqOf q x], [lt x y, eqOf q y])])
  | .intFloor => remAxioms (sub x (mul y q)) y
  | .intCeil => remAxioms (sub (mul y q) x) y
  | .mod =>
    let r := remAxioms q y
    (eqOf q (sub x (mul y (Poly.var (Atom.opq (.bin .intFloor A B))))) :: r.1, r.2)
  | _ => ([], [])

/-- The defining (in)equalities of an atom, in terms of the polynomials of
its arguments.  Symbols and auxiliary variables have none. -/
def axioms (a : Atom) : List Ineq × List Disj :=
  match a with
  | .opq (.un .abs A) =>
    match A.toPoly? with
    | some x =>
      let q := Poly.var a
      ([le x q, le (neg x) q], [([le (const 0) x, eqOf q x], [lt x (const 0), eqOf q (neg x)])])
    | none => ([], [])
  | .opq (.bin op A B) =>
    match A.toPoly?, B.toPoly? with
    | some x, some y => binAxioms op A B x y (Poly.var a)
    | _, _ => ([], [])
  | _ => ([], [])

end Atom

section Axioms

variable {I : ExprInterp} {ρ : Valuation}

theorem fdiv_rem (x y : Int) :
    (0 < y → 0 ≤ x - y * x.fdiv y ∧ x - y * x.fdiv y ≤ y - 1) ∧
    (y < 0 → y + 1 ≤ x - y * x.fdiv y ∧ x - y * x.fdiv y ≤ 0) := by
  rw [← Int.fmod_def]
  constructor
  · intro h
    have := Int.fmod_nonneg_of_pos x h
    have := Int.fmod_lt_of_pos x h
    omega
  · intro h
    have e : x.fmod y = -((-x).fmod (-y)) := by
      have := Int.neg_fmod_neg (-x) (-y); simp at this; simp
    have := Int.fmod_nonneg_of_pos (-x) (show 0 < -y by omega)
    have := Int.fmod_lt_of_pos (-x) (show 0 < -y by omega)
    omega

theorem remAxioms_holds (env : Env) (r y : Poly) (hy : y.eval env ≠ 0)
    (h : (0 < y.eval env → 0 ≤ r.eval env ∧ r.eval env ≤ y.eval env - 1) ∧
      (y.eval env < 0 → y.eval env + 1 ≤ r.eval env ∧ r.eval env ≤ 0)) :
    (∀ i ∈ (Atom.remAxioms r y).1, i.holds env) ∧ (∀ d ∈ (Atom.remAxioms r y).2, d.holds env) := by
  unfold Atom.remAxioms
  split
  · rename_i c hc
    have hc' := Poly.eval_of_const? env hc
    split
    · rename_i hpos
      have := h.1 (by omega)
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
        Ineq.holds_le, Poly.eval_const, Poly.eval_sub]
      exact ⟨⟨by omega, by omega⟩, by simp⟩
    · split
      · have := h.2 (by omega)
        simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
          Ineq.holds_le, Poly.eval_const, Poly.eval_add]
        exact ⟨⟨by omega, by omega⟩, by simp⟩
      · simp
  · refine ⟨by simp, ?_⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq]
    unfold Disj.holds
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
      Ineq.holds_le, Ineq.holds_lt, Poly.eval_const, Poly.eval_sub, Poly.eval_add]
    rcases lt_or_gt_of_ne hy with hlt | hgt
    · have := h.2 hlt; right; omega
    · have := h.1 hgt; left; omega

theorem remAxioms_atoms {P : Atom → Prop} {r y : Poly} (hr : r.AllAtoms P) (hy : y.AllAtoms P) :
    AxiomsAtoms P (Atom.remAxioms r y) := by
  have h0 := Poly.allAtoms_const (P := P) 0
  have h1 := Poly.allAtoms_const (P := P) 1
  unfold Atom.remAxioms
  split
  · split
    · constructor
      · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
          Ineq.le, Ineq.ge]
        exact ⟨Poly.allAtoms_sub hr h0, Poly.allAtoms_sub (Poly.allAtoms_sub hy h1) hr⟩
      · simp
    · split
      · constructor
        · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
            Ineq.le, Ineq.ge]
          exact ⟨Poly.allAtoms_sub hr (Poly.allAtoms_add hy h1), Poly.allAtoms_sub h0 hr⟩
        · simp
      · constructor <;> simp
  · constructor
    · simp
    · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq, List.cons_append,
        List.nil_append, forall_eq_or_imp, Ineq.le, Ineq.lt, Ineq.ge]
      refine ⟨Poly.allAtoms_sub (Poly.allAtoms_sub hy h0) h1, Poly.allAtoms_sub hr h0,
        Poly.allAtoms_sub (Poly.allAtoms_sub hy h1) hr, Poly.allAtoms_sub (Poly.allAtoms_sub h0 hy) h1,
        Poly.allAtoms_sub hr (Poly.allAtoms_add hy h1), Poly.allAtoms_sub h0 hr⟩

/-- Unpack a defined binary opaque atom. -/
theorem atomOk_bin {op : BinOp} {A B : Expr} {x y : Poly}
    (ha : AtomOk I ρ (.opq (.bin op A B))) (hx : A.toPoly? = some x) (hy : B.toPoly? = some y) :
    Eval.intArith op (x.eval (atomEnv I ρ)) (y.eval (atomEnv I ρ)) =
      some (.int (atomEnv I ρ (.opq (.bin op A B)))) ∧
    x.AllAtoms (AtomOk I ρ) ∧ y.AllAtoms (AtomOk I ρ) ∧
    (∀ s ∈ A.freeSyms, IntBound ρ s) ∧ (∀ s ∈ B.freeSyms, IntBound ρ s) ∧
    A.eval I ρ = some (.int (x.eval (atomEnv I ρ))) ∧ B.eval I ρ = some (.int (y.eval (atomEnv I ρ))) := by
  obtain ⟨-, he, hs⟩ := ha
  simp only [Atom.expr, Expr.eval, bind, Option.bind_eq_some_iff] at he
  obtain ⟨wa, hwa, wb, hwb, hu⟩ := he
  have hsa : ∀ s ∈ A.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Atom.expr, Expr.freeSyms, h])
  have hsb : ∀ s ∈ B.freeSyms, IntBound ρ s := fun s h => hs s (by simp [Atom.expr, Expr.freeSyms, h])
  obtain ⟨rfl, hoka⟩ := toPoly?_spec A x hx hsa wa hwa
  obtain ⟨rfl, hokb⟩ := toPoly?_spec B y hy hsb wb hwb
  exact ⟨hu, hoka, hokb, hsa, hsb, hwa, hwb⟩

/-- **Soundness of the atom axioms**: the defining (in)equalities of a
defined atom hold under `atomEnv`. -/
theorem atomEnv_axioms {a : Atom} (ha : AtomOk I ρ a) :
    (∀ h ∈ a.axioms.1, h.holds (atomEnv I ρ)) ∧ (∀ d ∈ a.axioms.2, d.holds (atomEnv I ρ)) := by
  unfold Atom.axioms
  split
  · -- abs
    rename_i A
    split
    · rename_i x hx
      obtain ⟨-, he, hs⟩ := ha
      simp only [Atom.expr, Expr.eval, bind, Option.bind_eq_some_iff] at he
      obtain ⟨wa, hwa, hu⟩ := he
      obtain ⟨rfl, -⟩ := toPoly?_spec A x hx (fun s h => hs s (by simpa [Atom.expr, Expr.freeSyms] using h))
        wa hwa
      simp only [Eval.unary, Option.some.injEq, Val.int.injEq, Int.natCast_natAbs] at hu
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
        Ineq.holds_le, Poly.eval_neg, Poly.eval_var, Disj.holds, Ineq.holds_eqOf, Ineq.holds_lt,
        Poly.eval_const]
      rw [← hu]
      refine ⟨⟨le_abs_self _, neg_le_abs _⟩, ?_⟩
      rcases le_or_gt 0 (x.eval (atomEnv I ρ)) with h | h
      · left; exact ⟨h, abs_of_nonneg h⟩
      · right; exact ⟨h, abs_of_neg h⟩
    · simp
  · -- binary
    rename_i op A B
    split
    · rename_i x y hx hy
      obtain ⟨hu, -, -, -, -, hA, hB⟩ := atomOk_bin ha hx hy
      generalize hX : x.eval (atomEnv I ρ) = X at hu hA
      generalize hY : y.eval (atomEnv I ρ) = Y at hu hB
      generalize hV : atomEnv I ρ (.opq (.bin op A B)) = V at hu
      unfold Atom.binAxioms
      cases op
      case intFloor =>
        simp only [Eval.intArith, Eval.intFloor] at hu
        split at hu
        · simp at hu
        · rename_i hy0
          simp only [Option.map_some, Option.some.injEq, Val.int.injEq] at hu
          apply remAxioms_holds _ _ _ (by rw [hY]; exact hy0)
          simp only [Poly.eval_sub, Poly.eval_mul, Poly.eval_var, hX, hY, hV, ← hu]
          exact fdiv_rem X Y
      case intCeil =>
        simp only [Eval.intArith, Eval.intCeil] at hu
        split at hu
        · simp at hu
        · rename_i hy0
          simp only [Option.map_some, Option.some.injEq, Val.int.injEq] at hu
          apply remAxioms_holds _ _ _ (by rw [hY]; exact hy0)
          simp only [Poly.eval_sub, Poly.eval_mul, Poly.eval_var, hX, hY, hV, ← hu]
          have := fdiv_rem (-X) Y
          constructor
          · intro h; have := this.1 h; constructor <;> linarith
          · intro h; have := this.2 h; constructor <;> linarith
      case mod =>
        simp only [Eval.intArith, Eval.pyMod] at hu
        split at hu
        · simp at hu
        · rename_i hy0
          simp only [Option.map_some, Option.some.injEq, Val.int.injEq] at hu
          -- the floor-division atom of the same arguments
          have hQ : atomEnv I ρ (.opq (.bin .intFloor A B)) = X.fdiv Y := by
            simp [atomEnv, Expr.evalInt, Expr.eval, hA, hB, Eval.arith, Eval.intArith, Eval.intFloor, hy0,
              Val.toInt?]
          have hr := remAxioms_holds (atomEnv I ρ) (Poly.var (.opq (.bin .mod A B))) y
            (by rw [hY]; exact hy0) (by
              simp only [Poly.eval_var, hY, hV, ← hu] at *
              have := fdiv_rem X Y
              rw [← Int.fmod_def] at this
              exact this)
          refine ⟨?_, hr.2⟩
          simp only [List.mem_cons, forall_eq_or_imp]
          refine ⟨?_, hr.1⟩
          simp only [Ineq.holds_eqOf, Poly.eval_var, Poly.eval_sub, Poly.eval_mul, hX, hY, hV, hQ, ← hu]
          exact Int.fmod_def X Y
      case min =>
        simp only [Eval.intArith, Option.some.injEq, Val.int.injEq] at hu
        subst hu
        simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
          Ineq.holds_le, Poly.eval_var, Disj.holds, Ineq.holds_eqOf, Ineq.holds_lt, hX, hY, hV]
        refine ⟨⟨min_le_left _ _, min_le_right _ _⟩, ?_⟩
        rcases le_or_gt X Y with h | h
        · left; exact ⟨h, min_eq_left h⟩
        · right; exact ⟨h, min_eq_right h.le⟩
      case max =>
        simp only [Eval.intArith, Option.some.injEq, Val.int.injEq] at hu
        subst hu
        simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
          Ineq.holds_le, Poly.eval_var, Disj.holds, Ineq.holds_eqOf, Ineq.holds_lt, hX, hY, hV]
        refine ⟨⟨le_max_left _ _, le_max_right _ _⟩, ?_⟩
        rcases le_or_gt Y X with h | h
        · left; exact ⟨h, max_eq_left h⟩
        · right; exact ⟨h, max_eq_right h.le⟩
      all_goals simp
    · simp
  · simp

theorem binAxioms_atoms {P : Atom → Prop} (op : BinOp) (A B : Expr) {x y q : Poly}
    (hx : x.AllAtoms P) (hy : y.AllAtoms P) (hq : q.AllAtoms P)
    (hQ : op = .mod → P (.opq (.bin .intFloor A B))) :
    AxiomsAtoms P (Atom.binAxioms op A B x y q) := by
  unfold Atom.binAxioms
  cases op
  case intFloor => exact remAxioms_atoms (Poly.allAtoms_sub hx (Poly.allAtoms_mul hy hq)) hy
  case intCeil => exact remAxioms_atoms (Poly.allAtoms_sub (Poly.allAtoms_mul hy hq) hx) hy
  case mod =>
    have hr := remAxioms_atoms hq hy
    refine ⟨?_, hr.2⟩
    simp only [List.mem_cons, forall_eq_or_imp]
    exact ⟨Poly.allAtoms_sub hq (Poly.allAtoms_sub hx (Poly.allAtoms_mul hy (Poly.allAtoms_var (hQ rfl)))),
      hr.1⟩
  case min =>
    constructor
    · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, Ineq.le, Ineq.ge]
      exact ⟨Poly.allAtoms_sub hx hq, Poly.allAtoms_sub hy hq⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq, List.cons_append, List.nil_append,
        forall_eq_or_imp, Ineq.le, Ineq.lt, Ineq.ge, Ineq.eqOf]
      exact ⟨Poly.allAtoms_sub hy hx, Poly.allAtoms_sub hq hx,
        Poly.allAtoms_sub (Poly.allAtoms_sub hx hy) (Poly.allAtoms_const 1), Poly.allAtoms_sub hq hy⟩
  case max =>
    constructor
    · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, Ineq.le, Ineq.ge]
      exact ⟨Poly.allAtoms_sub hq hx, Poly.allAtoms_sub hq hy⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq, List.cons_append, List.nil_append,
        forall_eq_or_imp, Ineq.le, Ineq.lt, Ineq.ge, Ineq.eqOf]
      exact ⟨Poly.allAtoms_sub hx hy, Poly.allAtoms_sub hq hx,
        Poly.allAtoms_sub (Poly.allAtoms_sub hy hx) (Poly.allAtoms_const 1), Poly.allAtoms_sub hq hy⟩
  all_goals exact ⟨by simp, by simp⟩

/-- The atoms of the axioms of a defined atom are defined (so axioms can be
closed transitively, `Core/SymCert`). -/
theorem axioms_atomOk {a : Atom} (ha : AtomOk I ρ a) : AxiomsAtoms (AtomOk I ρ) a.axioms := by
  unfold Atom.axioms
  split
  · rename_i A
    split
    · rename_i x hx
      have hq := Poly.allAtoms_var ha
      obtain ⟨-, he, hs⟩ := ha
      simp only [Atom.expr, Expr.eval, bind, Option.bind_eq_some_iff] at he
      obtain ⟨wa, hwa, -⟩ := he
      have hok := toPoly?_atomOk hx (fun s h => hs s (by simpa [Atom.expr, Expr.freeSyms] using h)) hwa
      constructor
      · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, Ineq.le,
          Ineq.ge]
        exact ⟨Poly.allAtoms_sub hq hok, Poly.allAtoms_sub hq (Poly.allAtoms_neg hok)⟩
      · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq, List.cons_append,
          List.nil_append, forall_eq_or_imp, Ineq.le, Ineq.lt, Ineq.ge, Ineq.eqOf]
        exact ⟨Poly.allAtoms_sub hok (Poly.allAtoms_const 0), Poly.allAtoms_sub hq hok,
          Poly.allAtoms_sub (Poly.allAtoms_sub (Poly.allAtoms_const 0) hok) (Poly.allAtoms_const 1),
          Poly.allAtoms_sub hq (Poly.allAtoms_neg hok)⟩
    · exact ⟨by simp, by simp⟩
  · rename_i op A B
    split
    · rename_i x y hx hy
      have hq := Poly.allAtoms_var ha
      obtain ⟨hu, hoka, hokb, hsa, hsb, hA, hB⟩ := atomOk_bin ha hx hy
      apply binAxioms_atoms op A B hoka hokb hq
      intro hop
      subst hop
      simp only [Eval.intArith, Eval.pyMod] at hu
      split at hu
      · simp at hu
      · rename_i hy0
        refine (atomOk_opq (I := I) (ρ := ρ) (E := .bin .intFloor A B)
          (v := (x.eval (atomEnv I ρ)).fdiv (y.eval (atomEnv I ρ))) ?_ ?_).1
        · simp [Expr.eval, hA, hB, Eval.arith, Eval.intArith, Eval.intFloor, hy0]
        · intro s h
          simp only [Expr.freeSyms, List.mem_append] at h
          rcases h with h | h
          · exact hsa s h
          · exact hsb s h
    · exact ⟨by simp, by simp⟩
  · exact ⟨by simp, by simp⟩

end Axioms

end Dace
