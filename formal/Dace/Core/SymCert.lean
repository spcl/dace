-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.SymArith

/-!
# Certificates for polynomial entailment over ℤ

A `Query` asks whether hypotheses (inequalities `p ≥ 0`, equalities `p = 0`
and disjunctions of conjunctions) entail a goal `g ≥ 0` for **every**
assignment of integers to atoms (`Query.Valid`).  Atoms are free: facts about
opaque atoms enter as hypotheses (`Query.withAxioms`, `Atom.axioms`).

Answers are certified.  A `Cert` is checked by `Cert.check`, whose soundness
`Cert.check_sound` is the only trusted part; `Query.search` is an untrusted,
incomplete heuristic that produces certificates.

* `Cert.psatz` is a Positivstellensatz-style refutation of the hypotheses
  together with the negated goal `−g − 1 ≥ 0` (integer tightening): a sum of
  non-negative multiples of `≥`-hypotheses, of pairwise products of them,
  of squares, and of arbitrary polynomial multiples of equalities, that
  normalizes to a negative constant.
* `Cert.split k l r` splits on the `k`-th disjunction.

`Query.search` runs Fourier–Motzkin elimination over ℚ treating every
monomial as a variable, first on the hypotheses alone, then adding all
pairwise products of `≥`-hypotheses (which makes symbolic strides
`i·N + j` linear), and case-splits on disjunctions up to a bounded depth.
-/

namespace Dace

set_option autoImplicit false

/-! ## Queries -/

/-- Do hypotheses entail `goal ≥ 0`? -/
structure Query where
  hyps : List Ineq := []
  disj : List Disj := []
  goal : Poly
  deriving DecidableEq, Repr, Inhabited

namespace Query

/-- Validity: for every integer assignment of the atoms, the hypotheses imply
the goal. -/
def Valid (q : Query) : Prop :=
  ∀ env : Env, (∀ h ∈ q.hyps, h.holds env) → (∀ d ∈ q.disj, d.holds env) → 0 ≤ q.goal.eval env

/-- The hypotheses extended by the negated, tightened goal at index `0`. -/
def ext (q : Query) : List Ineq := Ineq.ge (Poly.sub (Poly.const (-1)) q.goal) :: q.hyps

/-- The branch query of disjunction `k`: its chosen side is added to the
hypotheses and the disjunction is removed. -/
def branch (q : Query) (k : Nat) (hs : List Ineq) : Query :=
  { q with hyps := q.hyps ++ hs, disj := q.disj.eraseIdx k }

end Query

/-! ## Certificates -/

/-- A summand of a Positivstellensatz certificate; indices refer to
`Query.ext`. -/
inductive PTerm where
  /-- `c · hᵢ` for a `≥`-hypothesis. -/
  | lin (i c : Nat)
  /-- `c · hᵢ · hⱼ` for `≥`-hypotheses. -/
  | prod (i j c : Nat)
  /-- `p · hᵢ` for an equality, any polynomial `p`. -/
  | eq (i : Nat) (p : Poly)
  /-- `c · p²`. -/
  | sq (p : Poly) (c : Nat)
  deriving Repr, Inhabited

inductive Cert where
  /-- The summands add up to a negative constant. -/
  | psatz (terms : List PTerm)
  /-- Case split on disjunction `k`. -/
  | split (k : Nat) (left right : Cert)
  deriving Repr, Inhabited

namespace PTerm

/-- The polynomial of a summand, if well-formed. -/
def poly? (ext : List Ineq) : PTerm → Option Poly
  | lin i c => do
    let h ← ext[i]?
    if h.kind = .ge then some (Poly.smul c h.poly) else none
  | prod i j c => do
    let h ← ext[i]?
    let h' ← ext[j]?
    if h.kind = .ge ∧ h'.kind = .ge then some (Poly.smul c (Poly.mul h.poly h'.poly)) else none
  | eq i p => do
    let h ← ext[i]?
    if h.kind = .eq then some (Poly.mul p h.poly) else none
  | sq p c => some (Poly.smul c (Poly.mul p p))

theorem poly?_nonneg {ext : List Ineq} {env : Env} (hext : ∀ h ∈ ext, h.holds env) :
    ∀ (t : PTerm) (p : Poly), t.poly? ext = some p → 0 ≤ p.eval env
  | lin i c, p, ht => by
    simp only [poly?, bind, Option.bind_eq_some_iff] at ht
    obtain ⟨h, hh, ht⟩ := ht
    split at ht
    · rename_i hk
      simp only [Option.some.injEq] at ht; subst ht
      have := hext h (List.mem_of_getElem? hh)
      simp only [Ineq.holds, hk] at this
      simp only [Poly.eval_smul]; positivity
    · simp at ht
  | prod i j c, p, ht => by
    simp only [poly?, bind, Option.bind_eq_some_iff] at ht
    obtain ⟨h, hh, h', hh', ht⟩ := ht
    split at ht
    · rename_i hk
      simp only [Option.some.injEq] at ht; subst ht
      have h1 := hext h (List.mem_of_getElem? hh)
      have h2 := hext h' (List.mem_of_getElem? hh')
      simp only [Ineq.holds, hk.1, hk.2] at h1 h2
      simp only [Poly.eval_smul, Poly.eval_mul]; positivity
    · simp at ht
  | eq i q, p, ht => by
    simp only [poly?, bind, Option.bind_eq_some_iff] at ht
    obtain ⟨h, hh, ht⟩ := ht
    split at ht
    · rename_i hk
      simp only [Option.some.injEq] at ht; subst ht
      have := hext h (List.mem_of_getElem? hh)
      simp only [Ineq.holds, hk] at this
      simp [this]
    · simp at ht
  | sq q c, p, ht => by
    simp only [poly?, Option.some.injEq] at ht; subst ht
    simp only [Poly.eval_smul, Poly.eval_mul]
    exact mul_nonneg (by positivity) (mul_self_nonneg _)

end PTerm

/-- Sum of the summands' polynomials. -/
def PTerm.sum? (ext : List Ineq) : List PTerm → Option Poly
  | [] => some []
  | t :: ts => do Poly.add (← t.poly? ext) (← PTerm.sum? ext ts)

theorem PTerm.sum?_nonneg {ext : List Ineq} {env : Env} (hext : ∀ h ∈ ext, h.holds env) :
    ∀ (ts : List PTerm) (p : Poly), PTerm.sum? ext ts = some p → 0 ≤ p.eval env
  | [], p, h => by simp [PTerm.sum?] at h; subst h; simp
  | t :: ts, p, h => by
    simp only [PTerm.sum?, bind, Option.bind_eq_some_iff] at h
    obtain ⟨a, ha, b, hb, h⟩ := h
    simp only [Option.some.injEq] at h; subst h
    simp only [Poly.eval_add]
    have := PTerm.poly?_nonneg hext t a ha
    have := PTerm.sum?_nonneg hext ts b hb
    omega

/-- Check a certificate. -/
def Cert.check (q : Query) : Cert → Bool
  | .psatz ts =>
    match PTerm.sum? q.ext ts with
    | some p => p.isNegConst
    | none => false
  | .split k l r =>
    match q.disj[k]? with
    | some d => l.check (q.branch k d.1) && r.check (q.branch k d.2)
    | none => false

/-- **Soundness of certificate checking.** -/
theorem Cert.check_sound : ∀ (c : Cert) (q : Query), c.check q = true → q.Valid
  | .psatz ts, q, h => by
    intro env hh _
    simp only [Cert.check] at h
    split at h
    · rename_i p hp
      by_contra hg
      have hext : ∀ i ∈ q.ext, i.holds env := by
        intro i hi
        simp only [Query.ext, List.mem_cons] at hi
        rcases hi with rfl | hi
        · simp only [Ineq.holds_ge, Poly.eval_sub, Poly.eval_const]; omega
        · exact hh i hi
      have h1 := PTerm.sum?_nonneg hext ts p hp
      have h2 := Poly.isNegConst_eval env h
      omega
    · simp at h
  | .split k l r, q, h => by
    intro env hh hd
    simp only [Cert.check] at h
    split at h
    · rename_i d hdk
      simp only [Bool.and_eq_true] at h
      have hmem : d ∈ q.disj := List.mem_of_getElem? hdk
      have hsub : ∀ side, (∀ i ∈ side, Ineq.holds env i) →
          (∀ i ∈ (q.branch k side).hyps, i.holds env) ∧ (∀ d' ∈ (q.branch k side).disj, d'.holds env) := by
        intro side hs
        refine ⟨?_, ?_⟩
        · intro i hi
          simp only [Query.branch, List.mem_append] at hi
          rcases hi with hi | hi
          · exact hh i hi
          · exact hs i hi
        · intro d' hd'
          exact hd d' (List.mem_of_mem_eraseIdx hd')
      rcases hd d hmem with hs | hs
      · obtain ⟨a, b⟩ := hsub d.1 hs
        exact Cert.check_sound l _ h.1 env a b
      · obtain ⟨a, b⟩ := hsub d.2 hs
        exact Cert.check_sound r _ h.2 env a b
    · simp at h

/-! ## Adding the axioms of opaque atoms -/

def Ineq.atoms (h : Ineq) : List Atom := h.poly.atoms

def Disj.atoms (d : Disj) : List Atom := (d.1 ++ d.2).flatMap Ineq.atoms

/-- All atoms of a list of axioms. -/
def axiomsAtoms (ax : List Ineq × List Disj) : List Atom :=
  ax.1.flatMap Ineq.atoms ++ ax.2.flatMap Disj.atoms

/-- Append the elements of `l` that are not in `acc` yet. -/
def addNew (acc : List Atom) : List Atom → List Atom
  | [] => acc
  | a :: l => addNew (if a ∈ acc then acc else acc ++ [a]) l

theorem mem_addNew {x : Atom} : ∀ {acc l : List Atom}, x ∈ addNew acc l ↔ x ∈ acc ∨ x ∈ l
  | acc, [] => by simp [addNew]
  | acc, a :: l => by
    rw [addNew, mem_addNew]
    split
    · rename_i h
      constructor
      · rintro (h' | h')
        · exact Or.inl h'
        · exact Or.inr (List.mem_cons_of_mem _ h')
      · rintro (h' | h')
        · exact Or.inl h'
        · rcases List.mem_cons.1 h' with rfl | h'
          · exact Or.inl h
          · exact Or.inr h'
    · simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
      tauto

/-- Atoms reachable through axioms, `n` rounds. -/
def atomClosure : Nat → List Atom → List Atom
  | 0, as => as
  | n + 1, as => atomClosure n (addNew as (as.flatMap fun a => axiomsAtoms a.axioms))

/-- The axioms of a list of atoms. -/
def axiomsOf (as : List Atom) : List Ineq × List Disj :=
  (as.flatMap (·.axioms.1), as.flatMap (·.axioms.2))

/-- The atoms of a query. -/
def Query.atoms (q : Query) : List Atom :=
  addNew [] (q.goal.atoms ++ q.hyps.flatMap Ineq.atoms ++ q.disj.flatMap Disj.atoms)

/-- Add the axioms of the query's atoms (closed `depth` rounds). -/
def Query.withAxioms (q : Query) (depth : Nat := 3) : Query :=
  let ax := axiomsOf (atomClosure depth q.atoms)
  { q with hyps := q.hyps ++ ax.1, disj := q.disj ++ ax.2 }

section Axioms

variable {I : ExprInterp} {ρ : Valuation}

theorem Ineq.holds_congr {env env' : Env} {h : Ineq} (hc : ∀ a ∈ h.atoms, env a = env' a) :
    h.holds env ↔ h.holds env' := by
  have := Poly.eval_congr (Poly.allAtoms_iff.2 hc)
  unfold Ineq.holds; split <;> rw [this]

theorem Disj.holds_congr {env env' : Env} {d : Disj} (hc : ∀ a ∈ d.atoms, env a = env' a) :
    d.holds env ↔ d.holds env' := by
  have hi : ∀ h ∈ d.1 ++ d.2, (h.holds env ↔ h.holds env') := fun h hh =>
    Ineq.holds_congr fun a ha => hc a (List.mem_flatMap.2 ⟨h, hh, ha⟩)
  unfold Disj.holds
  constructor
  · rintro (h | h)
    · exact Or.inl fun i hi' => (hi i (List.mem_append_left _ hi')).1 (h i hi')
    · exact Or.inr fun i hi' => (hi i (List.mem_append_right _ hi')).1 (h i hi')
  · rintro (h | h)
    · exact Or.inl fun i hi' => (hi i (List.mem_append_left _ hi')).2 (h i hi')
    · exact Or.inr fun i hi' => (hi i (List.mem_append_right _ hi')).2 (h i hi')

theorem axiomsAtoms_ok {a : Atom} (ha : AtomOk I ρ a) : ∀ b ∈ axiomsAtoms a.axioms, AtomOk I ρ b := by
  have hax := axioms_atomOk ha
  intro b hb
  simp only [axiomsAtoms, List.mem_append, List.mem_flatMap, Disj.atoms, Ineq.atoms] at hb
  rcases hb with ⟨h, hh, hb⟩ | ⟨d, hd, h, hh, hb⟩
  · exact Poly.allAtoms_iff.1 (hax.1 h hh) b hb
  · exact Poly.allAtoms_iff.1 (hax.2 d hd h (List.mem_append.2 hh)) b hb

theorem axioms_var {a : Atom} (h : a.isVar = true) : a.axioms = ([], []) := by
  cases a <;> simp_all [Atom.isVar, Atom.axioms]

theorem atomClosure_ok : ∀ (n : Nat) (as : List Atom), (∀ a ∈ as, a.isVar = false → AtomOk I ρ a) →
    ∀ a ∈ atomClosure n as, a.isVar = false → AtomOk I ρ a
  | 0, as, h => h
  | n + 1, as, h => by
    apply atomClosure_ok n
    intro a ha hv
    rcases mem_addNew.1 ha with ha | ha
    · exact h a ha hv
    · obtain ⟨b, hb, ha⟩ := List.mem_flatMap.1 ha
      cases hbv : b.isVar
      · exact axiomsAtoms_ok (h b hb hbv) a ha
      · simp [axioms_var hbv, axiomsAtoms] at ha

/-- **Soundness of adding atom axioms**: a valid query with axioms entails the
original query under any environment that agrees with `atomEnv` on the
query's non-auxiliary atoms, provided these are defined. -/
theorem Query.withAxioms_sound (q : Query) (n : Nat) (hv : (q.withAxioms n).Valid) (env : Env)
    (hok : ∀ a ∈ q.atoms, a.isVar = false → AtomOk I ρ a ∧ env a = atomEnv I ρ a)
    (hh : ∀ h ∈ q.hyps, h.holds env) (hd : ∀ d ∈ q.disj, d.holds env) : 0 ≤ q.goal.eval env := by
  let env' : Env := fun a => if a.isVar then env a else atomEnv I ρ a
  have hag : ∀ a ∈ q.atoms, env a = env' a := by
    intro a ha
    simp only [env']
    split
    · rfl
    · rename_i hvar; exact (hok a ha (by simpa using hvar)).2
  have hqa : ∀ x, x ∈ q.goal.atoms ∨ x ∈ q.hyps.flatMap Ineq.atoms ∨ x ∈ q.disj.flatMap Disj.atoms →
      x ∈ q.atoms := by
    intro x hx
    simp only [Query.atoms, mem_addNew, List.mem_append, List.not_mem_nil, false_or]
    tauto
  have hcl := atomClosure_ok (I := I) (ρ := ρ) n q.atoms (fun a ha hvar => (hok a ha hvar).1)
  -- atoms of axioms of the closure are defined, so `env'` is `atomEnv` there
  have hax : ∀ a ∈ atomClosure n q.atoms,
      (∀ h ∈ a.axioms.1, h.holds env') ∧ (∀ d ∈ a.axioms.2, d.holds env') := by
    intro a ha
    cases hav : a.isVar
    · have hoka := hcl a ha hav
      have hs := atomEnv_axioms hoka
      have hatoms := axiomsAtoms_ok hoka
      have heq : ∀ b ∈ axiomsAtoms a.axioms, atomEnv I ρ b = env' b := by
        intro b hb
        have := (hatoms b hb).1
        simp [env', this]
      constructor
      · intro h hh
        exact (Ineq.holds_congr fun b hb =>
          heq b (List.mem_append_left _ (List.mem_flatMap.2 ⟨h, hh, hb⟩))).1 (hs.1 h hh)
      · intro d hd
        exact (Disj.holds_congr fun b hb =>
          heq b (List.mem_append_right _ (List.mem_flatMap.2 ⟨d, hd, hb⟩))).1 (hs.2 d hd)
    · simp [axioms_var hav]
  have hgoal := hv env' (by
      intro h hm
      simp only [Query.withAxioms, axiomsOf, List.mem_append, List.mem_flatMap] at hm
      rcases hm with hm | ⟨a, ha, hm⟩
      · exact (Ineq.holds_congr fun b hb =>
          hag b (hqa b (Or.inr (Or.inl (List.mem_flatMap.2 ⟨h, hm, hb⟩))))).1 (hh h hm)
      · exact (hax a ha).1 h hm)
    (by
      intro d hd'
      simp only [Query.withAxioms, axiomsOf, List.mem_append, List.mem_flatMap] at hd'
      rcases hd' with hd' | ⟨a, ha, hd'⟩
      · exact (Disj.holds_congr fun b hb =>
          hag b (hqa b (Or.inr (Or.inr (List.mem_flatMap.2 ⟨d, hd', hb⟩))))).1 (hd d hd')
      · exact (hax a ha).2 d hd')
  rw [Poly.eval_congr (Poly.allAtoms_iff.2 fun b hb => hag b (hqa b (Or.inl hb)))]
  exact hgoal

end Axioms

/-! ## Facts as hypotheses -/

/-- Hypotheses of a relation between integer-closed sides.  `ne` becomes a
disjunction; `dvd` (an existential) is not used. -/
def Relation.toHyps (r : Relation) : List Ineq × List Disj :=
  match r.lhs.toPoly?, r.rhs.toPoly? with
  | some a, some b =>
    match r.kind with
    | .lt => ([Ineq.lt a b], [])
    | .le => ([Ineq.le a b], [])
    | .eq => ([Ineq.eqOf a b], [])
    | .ne => ([], [([Ineq.lt a b], [Ineq.lt b a])])
    | .dvd => ([], [])
  | _, _ => ([], [])

/-- Hypotheses of a sign predicate on a polynomial. -/
def Pred.toHyps (p : Pred) (x : Poly) : List Ineq × List Disj :=
  let z := Poly.const 0
  match p with
  | .pos => ([Ineq.lt z x], [])
  | .nonneg => ([Ineq.le z x], [])
  | .neg => ([Ineq.lt x z], [])
  | .nonpos => ([Ineq.le x z], [])
  | .nonzero => ([], [([Ineq.lt z x], [Ineq.lt x z])])

/-- The facts usable as hypotheses: sign predicates of integer symbols and
relations whose free symbols are all integer symbols. -/
def Constraints.toHyps (C : Constraints) : List Ineq × List Disj :=
  let signs := (C.sign.filter fun sp => sp.1 ∈ C.intSyms).map fun sp => sp.2.toHyps (Poly.var (.sym sp.1))
  let rels := (C.rels.filter fun r => (r.lhs.freeSyms ++ r.rhs.freeSyms).all (· ∈ C.intSyms)).map
    Relation.toHyps
  ((signs ++ rels).flatMap (·.1), (signs ++ rels).flatMap (·.2))

section FactsSound

variable {I : ExprInterp} {ρ : Valuation}

/-- Hypothesis lists that hold under an environment, with defined atoms. -/
def HypsOk (I : ExprInterp) (ρ : Valuation) (ax : List Ineq × List Disj) : Prop :=
  (∀ h ∈ ax.1, h.holds (atomEnv I ρ)) ∧ (∀ d ∈ ax.2, d.holds (atomEnv I ρ)) ∧
    AxiomsAtoms (AtomOk I ρ) ax

theorem Pred.toHyps_ok {p : Pred} {s : Sym} {n : Int} (hn : ρ s = some (.int n))
    (hp : p.holds (n : ℚ)) : HypsOk I ρ (p.toHyps (Poly.var (.sym s))) := by
  have hv : atomEnv I ρ (.sym s) = n := by simp [atomEnv, hn, Val.toInt?]
  have hok : AtomOk I ρ (.sym s) := ⟨rfl, by simp [Atom.expr, Expr.eval, hn, hv], by
    intro s' h; simp [Atom.expr, Expr.freeSyms] at h; subst h; exact ⟨n, hn⟩⟩
  have hx := Poly.allAtoms_var hok
  have h0 := Poly.allAtoms_const (P := AtomOk I ρ) 0
  cases p <;> simp only [Pred.holds] at hp <;> refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [Pred.toHyps, hv, Disj.holds, Ineq.lt_atoms, Ineq.le_atoms, hx, h0] <;>
    try (first | exact_mod_cast hp | omega)
  have : n ≠ 0 := by exact_mod_cast hp
  omega

theorem Relation.toHyps_ok {r : Relation} (hr : r.holds I ρ)
    (hs : ∀ s ∈ r.lhs.freeSyms ++ r.rhs.freeSyms, IntBound ρ s) : HypsOk I ρ r.toHyps := by
  unfold Relation.toHyps
  split
  · rename_i a b ha hb
    obtain ⟨qa, qb, hqa, hqb, hk⟩ := hr
    simp only [Expr.evalRat, bind, Option.bind_eq_some_iff] at hqa hqb
    obtain ⟨wa, hwa, hqa⟩ := hqa
    obtain ⟨wb, hwb, hqb⟩ := hqb
    obtain ⟨rfl, hoka⟩ := toPoly?_spec r.lhs a ha (fun s h => hs s (List.mem_append_left _ h)) wa hwa
    obtain ⟨rfl, hokb⟩ := toPoly?_spec r.rhs b hb (fun s h => hs s (List.mem_append_right _ h)) wb hwb
    simp only [Val.toRat?, Option.some.injEq] at hqa hqb
    subst hqa hqb
    cases hkind : r.kind <;> rw [hkind] at hk <;> simp only [RelKind.holds] at hk <;>
      refine ⟨?_, ?_, ?_, ?_⟩ <;>
      simp [Disj.holds, Ineq.lt_atoms, Ineq.le_atoms, Ineq.eqOf_atoms, hoka, hokb] <;>
      exact_mod_cast hk
  · exact ⟨by simp, by simp, by simp, by simp⟩

/-- **Soundness of the facts as hypotheses**: a valuation satisfying the
facts satisfies their hypotheses under `atomEnv`, and their atoms are
defined. -/
theorem Constraints.toHyps_sound {C : Constraints} (h : C.satisfies I ρ) : HypsOk I ρ C.toHyps := by
  obtain ⟨hint, hsign, hrel⟩ := h
  have hall : ∀ ax ∈ ((C.sign.filter fun sp => sp.1 ∈ C.intSyms).map fun sp =>
        sp.2.toHyps (Poly.var (.sym sp.1))) ++
      ((C.rels.filter fun r => (r.lhs.freeSyms ++ r.rhs.freeSyms).all (· ∈ C.intSyms)).map
        Relation.toHyps), HypsOk I ρ ax := by
    intro ax hax
    rcases List.mem_append.1 hax with hax | hax
    · obtain ⟨sp, hsp, rfl⟩ := List.mem_map.1 hax
      obtain ⟨hmem, hi⟩ := List.mem_filter.1 hsp
      obtain ⟨n, hn⟩ := hint sp.1 (by simpa using hi)
      obtain ⟨q, hq, hp⟩ := hsign sp hmem
      rw [hn] at hq
      simp only [Option.bind_some, Val.toRat?, Option.some.injEq] at hq
      subst hq
      exact Pred.toHyps_ok hn hp
    · obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hax
      obtain ⟨hmem, hi⟩ := List.mem_filter.1 hr
      refine Relation.toHyps_ok (hrel r hmem) ?_
      intro s hs
      exact hint s (by simpa using (List.all_eq_true.1 hi) s hs)
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i hi
    obtain ⟨ax, hax, hi⟩ := List.mem_flatMap.1 hi
    exact (hall ax hax).1 i hi
  · intro d hd
    obtain ⟨ax, hax, hd⟩ := List.mem_flatMap.1 hd
    exact (hall ax hax).2.1 d hd
  · intro i hi
    obtain ⟨ax, hax, hi⟩ := List.mem_flatMap.1 hi
    exact (hall ax hax).2.2.1 i hi
  · intro d hd
    obtain ⟨ax, hax, hd⟩ := List.mem_flatMap.1 hd
    exact (hall ax hax).2.2.2 d hd

end FactsSound

end Dace
