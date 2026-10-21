-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Footprint
import Dace.Equiv.Order
import Dace.Equiv.Equiv
import Dace.SDFG.Clean
import Dace.Core.SymSubset

/-!
# Static race freedom (`raceFree?`)

A sound, incomplete check that an SDFG never ends in a race
(`RaceFree`, R1/R2 design §3.6).  Races arise only where parallel logs are
merged (map iterations, state components) and in the happens-before check
of a sequential execution (`Decisions.md#D61`); every other function of the
semantics is race-free (`Dace.SDFG.Clean`).  The check bounds the logs by
static footprints (`Dace.SDFG.Footprint`) and shows them pairwise free of
conflicts:

* two accesses never conflict when their kinds are compatible (both reads,
  or WCR with the same operator) or they name different containers (no
  container is aliased in the fragment);
* two **distinct iterations** of a map never touch a common element through
  two accesses of the same container (`iterDisjoint?`, `Dace.Core.SymSubset`);
* two nodes of a scope body or component that no path orders, and two
  components of a state, access different containers or compatibly.

The fragment: the static footprint exists (no consume scopes, nested SDFGs,
views, references, streams or dynamic map inputs), map parameters are
distinct and fresh for the facts, and accesses that need `iterDisjoint?` are
outside nested maps and pure.  Symbols in map ranges, subsets and shapes must
be integer facts (`Constraints.intSyms`) that the program never assigns.
-/

set_option autoImplicit false

namespace Dace

/-! ## Separated accesses give conflict-free logs -/

/-- Access kinds whose parallel accesses to one location never conflict:
two reads, or two WCR contributions with the same operator. -/
def AccKind.compat : AccKind → AccKind → Bool
  | .read, .read => true
  | .wcr o, .wcr o' => decide (o = o')
  | _, _ => false

theorem AccKind.compat_comm (a b : AccKind) : a.compat b = b.compat a := by
  cases a <;> cases b <;> simp [AccKind.compat, eq_comm]

/-- Accesses `a` (under `ρa`) and `b` (under `ρb`) are separated: compatible
kinds, or no common location. -/
def Access.Sep (ctx : Ctx) (m₀ : Mem) (ρa ρb : Valuation) (a b : Access) : Prop :=
  a.kind.compat b.kind = true ∨ ∀ l, a.Hit ctx m₀ ρa l → b.Hit ctx m₀ ρb l → False

theorem Access.Sep.symm {ctx : Ctx} {m₀ : Mem} {ρa ρb : Valuation} {a b : Access}
    (h : Access.Sep ctx m₀ ρa ρb a b) : Access.Sep ctx m₀ ρb ρa b a := by
  rcases h with h | h
  · exact Or.inl (by rw [AccKind.compat_comm]; exact h)
  · exact Or.inr fun l hb ha => h l ha hb

section Free

variable {ctx : Ctx} {m₀ : Mem} {ρa ρb : Valuation} {A B : List Access}

theorem halfFree_of_sep {g h : Log} (hg : g.Within (LocPred.ofAccesses ctx m₀ ρa A))
    (hh : h.Within (LocPred.ofAccesses ctx m₀ ρb B)) (hs : ∀ a ∈ A, ∀ b ∈ B, Access.Sep ctx m₀ ρa ρb a b) :
    Log.HalfFree g h := by
  have sep : ∀ {a b : Access} {l : Loc}, a ∈ A → b ∈ B → a.Hit ctx m₀ ρa l → b.Hit ctx m₀ ρb l →
      a.kind.compat b.kind = true := by
    intro a b l ha hb hla hlb
    rcases hs a ha b hb with h | h
    · exact h
    · exact (h l hla hlb).elim
  refine ⟨fun l hl => ?_, fun l hl => ?_, fun x hx y hy hxy => ?_, ?_, ?_⟩
  · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
    obtain ⟨a, ha, hk, hla⟩ := hg.writes x hx
    refine ⟨fun hl' => ?_, fun hl' => ?_, fun hl' => ?_⟩
    · obtain ⟨y, hy, hyl⟩ := List.mem_map.1 hl'
      obtain ⟨b, hb, hk', hlb⟩ := hh.writes y hy
      rw [hyl] at hlb
      have := sep ha hb hla hlb
      rw [hk, hk'] at this; simp [AccKind.compat] at this
    · obtain ⟨b, hb, hk', hlb⟩ := hh.reads _ hl'
      have := sep ha hb hla hlb
      rw [hk, hk'] at this; simp [AccKind.compat] at this
    · obtain ⟨y, hy, hyl⟩ := List.mem_map.1 hl'
      obtain ⟨b, hb, hk', hlb⟩ := hh.wcrs y hy
      rw [hyl] at hlb
      have := sep ha hb hla hlb
      rw [hk, hk'] at this; simp [AccKind.compat] at this
  · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
    intro hl'
    obtain ⟨a, ha, hk, hla⟩ := hg.wcrs x hx
    obtain ⟨b, hb, hk', hlb⟩ := hh.reads _ hl'
    have := sep ha hb hla hlb
    rw [hk, hk'] at this; simp [AccKind.compat] at this
  · obtain ⟨a, ha, hk, hla⟩ := hg.wcrs x hx
    obtain ⟨b, hb, hk', hlb⟩ := hh.wcrs y hy
    rw [← hxy] at hlb
    have := sep ha hb hla hlb
    rw [hk, hk'] at this; simpa [AccKind.compat] using this
  · intro x hx; rw [hg.aliasSets] at hx; cases hx
  · intro l hl
    simp only [Log.exclusiveQueues, hg.queues] at hl
    simp at hl

/-- **Separated footprints give conflict-free logs.** -/
theorem free_of_sep {g h : Log} (hg : g.Within (LocPred.ofAccesses ctx m₀ ρa A))
    (hh : h.Within (LocPred.ofAccesses ctx m₀ ρb B)) (hs : ∀ a ∈ A, ∀ b ∈ B, Access.Sep ctx m₀ ρa ρb a b) :
    Log.Free g h :=
  ⟨halfFree_of_sep hg hh hs, halfFree_of_sep hh hg fun b hb a ha => (hs a ha b hb).symm⟩

end Free

/-! ## Static separation -/

/-- No container name is bound to an alias. -/
def NoAlias (m : Mem) : Prop := ∀ n, m.aliases n = none

section Static

variable {ctx : Ctx} {m₀ : Mem}

theorem Access.hit_eq {ρ : Valuation} {a : Access} {l : Loc} (hna : NoAlias m₀) (h : a.Hit ctx m₀ ρ l) :
    ∃ off, l = Loc.elem (ctx.cname a.data) off := by
  obtain ⟨_, _, off, _, rfl⟩ := h
  exact ⟨off, Mem.resolve_of_none (hna _)⟩

/-- Static separation of two accesses under any valuations: compatible kinds
or different containers. -/
def accSep (a b : Access) : Bool := a.kind.compat b.kind || a.data != b.data

theorem accSep_sound {ρa ρb : Valuation} {a b : Access} (hna : NoAlias m₀) (h : accSep a b = true) :
    Access.Sep ctx m₀ ρa ρb a b := by
  simp only [accSep, Bool.or_eq_true, bne_iff_ne, ne_eq] at h
  rcases h with h | h
  · exact Or.inl h
  · refine Or.inr fun l ha hb => h ?_
    obtain ⟨o, rfl⟩ := Access.hit_eq hna ha
    obtain ⟨o', ho⟩ := Access.hit_eq hna hb
    simp only [Loc.elem, Loc.mk.injEq, Ctx.cname] at ho
    exact (String.append_right_inj _).1 ho.1

theorem bindIdx_cons (ρ : Valuation) (p : Sym) (ps : List Sym) (v : Int) (vs : Idx) :
    ρ.bindIdx (p :: ps) (v :: vs) = fun y => if p = y then some (.int v) else ρ.bindIdx ps vs y := by
  funext y
  simp only [Valuation.bindIdx, Valuation.setAll, List.zip_cons_cons, List.map_cons, List.find?_cons]
  by_cases h : p = y <;> simp [h]

theorem bindIdx_not_mem (ρ : Valuation) : ∀ (ps : List Sym) (i : Idx) {y : Sym}, y ∉ ps → ρ.bindIdx ps i y = ρ y
  | [], _, _, _ => by simp [Valuation.bindIdx, Valuation.setAll]
  | _ :: _, [], _, _ => by simp [Valuation.bindIdx, Valuation.setAll]
  | p :: ps, v :: vs, y, hy => by
    rw [bindIdx_cons]
    have : p ≠ y := fun h => hy (h ▸ List.mem_cons_self ..)
    simp only [this, ↓reduceIte]
    exact bindIdx_not_mem ρ ps vs fun h => hy (List.mem_cons_of_mem _ h)

/-- Map parameters bound to an iteration point (`Binds`). -/
theorem bindIdx_binds (ρ : Valuation) :
    ∀ {ps : List Sym} {i : Idx}, ps.Nodup → ps.length = i.length → Binds ρ ps i (ρ.bindIdx ps i)
  | [], [], _, _ => ⟨.nil, fun s _ => by simp [Valuation.bindIdx, Valuation.setAll]⟩
  | p :: ps, v :: vs, hnd, hl => by
    have hnd' := List.nodup_cons.1 hnd
    obtain ⟨hf, ho⟩ := bindIdx_binds ρ hnd'.2 (by simpa using hl)
    refine ⟨.cons (by rw [bindIdx_cons]; simp) ?_, fun s hs => ?_⟩
    · refine SymSubset.forall₂_imp_mem hf fun q hq w hqw => ?_
      rw [bindIdx_cons]
      have : p ≠ q := fun h => hnd'.1 (h ▸ hq)
      simpa [this] using hqw
    · rw [bindIdx_cons]
      have : p ≠ s := fun h => hs (h ▸ List.mem_cons_self ..)
      simp only [this, ↓reduceIte]
      exact ho s fun h => hs (List.mem_cons_of_mem _ h)

/-- `iterDisjoint?` with undecided obligations accepted by the oracle `orc`
(R1/R2 design §3.5: obligations are discharged outside, e.g. by the Lean
kernel on the file `Query.toLeanFile` emits).  With `orc = fun _ => false`
it is `iterDisjoint?`. -/
def iterDisjointWith? (orc : Query → Bool) (C : Constraints) (params : List Sym) (rng : Range) (dA : DataDesc)
    (A : Subset) (dB : DataDesc) (B : Subset) : Bool :=
  match iterObligations? C params rng dA A dB B with
  | some qs => qs.all orc
  | none => false

theorem iterDisjointWith?_sound {I : ExprInterp} {ρ : Valuation} {orc : Query → Bool}
    (horc : ∀ q, orc q = true → q.Valid) {C : Constraints} {params : List Sym} {rng : Range}
    {dA dB : DataDesc} {A B : Subset} (h : iterDisjointWith? orc C params rng dA A dB B = true)
    (hC : C.satisfies I ρ) {pts : List Idx}
    (hrng : rng.points I ρ = some pts) {i i' : Idx} (hi : i ∈ pts) (hi' : i' ∈ pts) (hne : i ≠ i')
    {ρi ρi' : Valuation} (hb : Binds ρ params i ρi) (hb' : Binds ρ params i' ρi') {psA psB : List Idx}
    (hA : A.points I ρi = some psA) (hB : B.points I ρi' = some psB) {p q : Idx} (hp : p ∈ psA) (hq : q ∈ psB)
    {o o' : Int} (ho : dA.linearOffset I ρi p = some o) (ho' : dB.linearOffset I ρi' q = some o') : o ≠ o' := by
  unfold iterDisjointWith? at h
  split at h
  · rename_i qs hobl
    exact iterObligations?_sound hobl (fun x hx => horc x (List.all_eq_true.1 h x hx)) hC hrng hi hi' hne hb hb'
      hA hB hp hq ho ho'
  · cases h

/-- Static separation of two accesses in two **distinct iterations** of the
map `params ∈ rng`: `accSep`, or both outside nested maps, pure, on the same
non-scalar container, with `iterDisjointWith?` subsets. -/
def iterSep (ctx : Ctx) (orc : Query → Bool) (C : Constraints) (params : List Sym) (rng : Range) (a b : Access) :
    Bool :=
  accSep a b ||
  (a.binders.isEmpty && b.binders.isEmpty && a.data == b.data && ctx.pointsPure a.data a.sub &&
    ctx.pointsPure b.data b.sub &&
    match ctx.desc? a.data with
    | some d => !d.isScalar &&
        iterDisjointWith? orc C params rng d (a.sub.getD (.range d.fullRange)) d (b.sub.getD (.range d.fullRange))
    | none => true)

theorem iterSep_sound {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid) {C : Constraints} {params : List Sym} {rng : Range} {ρ : Valuation} {pts : List Idx}
    {i j : Idx} {a b : Access} (hna : NoAlias m₀) (hC : C.satisfies ctx.I.expr ρ)
    (hrng : rng.points ctx.I.expr ρ = some pts) (hi : i ∈ pts) (hj : j ∈ pts) (hne : i ≠ j)
    (hnd : params.Nodup) (hlen : params.length = rng.dims.length) (h : iterSep ctx orc C params rng a b = true) :
    Access.Sep ctx m₀ (ρ.bindIdx params i) (ρ.bindIdx params j) a b := by
  unfold iterSep at h
  rcases Bool.or_eq_true_iff.1 h with h | h
  · exact accSep_sound hna h
  · simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨hba, hbb⟩, hab⟩, hpa⟩, hpb⟩, hd⟩ := h
    refine Or.inr fun l hla hlb => ?_
    have hsa : a.pureIn ctx = true := by simp [Access.pureIn, hba, hpa]
    have hsb : b.pureIn ctx = true := by simp [Access.pureIn, hbb, hpb]
    obtain ⟨ρ₁, hx₁, d, o, hd₁, ⟨psA, hA, p, hp, ho⟩, hl₁⟩ := Access.hit_static hla hsa
    obtain ⟨ρ₂, hx₂, d', o', hd₂, ⟨psB, hB, q, hq, ho'⟩, hl₂⟩ := Access.hit_static hlb hsb
    rw [hba] at hx₁; rw [hbb] at hx₂
    simp only [Binders.ExtS] at hx₁ hx₂
    subst hx₁ hx₂
    rw [← hab, hd₁] at hd₂
    cases hd₂
    rw [hd₁] at hd
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hd
    obtain ⟨hns, hdis⟩ := hd
    rw [Ctx.elemOffset_of_not_scalar hns] at ho ho'
    have hli := ((Range.mem_points hrng).1 hi).length_eq
    have hlj := ((Range.mem_points hrng).1 hj).length_eq
    have hne' := iterDisjointWith?_sound horc hdis hC hrng hi hj hne (bindIdx_binds ρ hnd (by omega))
      (bindIdx_binds ρ hnd (by omega)) hA hB hp hq ho ho'
    rw [hl₁, Mem.resolve_of_none (hna _), Mem.resolve_of_none (hna _), ← hab] at hl₂
    simp only [Loc.elem, Loc.mk.injEq, List.cons.injEq, Accessor.lin.injEq, and_true, true_and] at hl₂
    exact hne' hl₂

end Static

/-! ## Facts under binding and restriction -/

namespace Constraints

/-- The symbols the facts mention. -/
def syms (C : Constraints) : List Sym :=
  C.intSyms ++ C.sign.map (·.1) ++ C.rels.flatMap fun r => r.lhs.freeSyms ++ r.rhs.freeSyms

/-- The facts plus integer facts for `ps` (map parameters). -/
def addInts (C : Constraints) (ps : List Sym) : Constraints := { C with intSyms := C.intSyms ++ ps }

/-- The facts about the symbols `keep` accepts. -/
def restrict (C : Constraints) (keep : Sym → Bool) : Constraints where
  intSyms := C.intSyms.filter keep
  sign := C.sign.filter fun sp => keep sp.1
  rels := C.rels.filter fun r => (r.lhs.freeSyms ++ r.rhs.freeSyms).all keep

variable {I : ExprInterp} {C : Constraints}

/-- Facts depend only on the values of the symbols they mention. -/
theorem satisfies_congr {ρ ρ' : Valuation} (h : C.satisfies I ρ) (hag : ∀ s ∈ C.syms, ρ' s = ρ s) :
    C.satisfies I ρ' := by
  obtain ⟨hi, hs, hr⟩ := h
  refine ⟨fun s hm => ?_, fun sp hm => ?_, fun r hm => ?_⟩
  · rw [hag s (by simp [syms, hm])]; exact hi s hm
  · rw [hag sp.1 (by simp only [syms, List.mem_append, List.mem_map]; exact Or.inl (Or.inr ⟨sp, hm, rfl⟩))]
    exact hs sp hm
  · obtain ⟨a, b, ha, hb, hk⟩ := hr r hm
    have hsub : ∀ s, s ∈ r.lhs.freeSyms ++ r.rhs.freeSyms → ρ' s = ρ s := fun s hs' =>
      hag s (by simp only [syms, List.mem_append, List.mem_flatMap]; exact Or.inr ⟨r, hm, by simpa using hs'⟩)
    refine ⟨a, b, ?_, ?_, hk⟩
    · simp only [Expr.evalRat] at ha ⊢
      rw [Expr.eval_congr r.lhs fun s hs' => hsub s (List.mem_append_left _ hs')]; exact ha
    · simp only [Expr.evalRat] at hb ⊢
      rw [Expr.eval_congr r.rhs fun s hs' => hsub s (List.mem_append_right _ hs')]; exact hb

theorem satisfies_restrict {ρ : Valuation} (keep : Sym → Bool) (h : C.satisfies I ρ) :
    (C.restrict keep).satisfies I ρ := by
  obtain ⟨hi, hs, hr⟩ := h
  exact ⟨fun s hm => hi s (List.mem_filter.1 hm).1, fun sp hm => hs sp (List.mem_filter.1 hm).1,
    fun r hm => hr r (List.mem_filter.1 hm).1⟩

theorem keep_of_mem_restrict {keep : Sym → Bool} {s : Sym} (h : s ∈ (C.restrict keep).syms) : keep s = true := by
  simp only [syms, restrict, List.mem_append, List.mem_filter, List.mem_map, List.mem_flatMap,
    List.all_eq_true] at h
  rcases h with (⟨-, h⟩ | ⟨sp, ⟨-, h⟩, rfl⟩) | ⟨r, ⟨-, h⟩, hs⟩
  · exact h
  · exact h
  · exact h s (by simpa using hs)

end Constraints

theorem bindIdx_mem (ρ : Valuation) :
    ∀ {ps : List Sym} {i : Idx} {p : Sym}, p ∈ ps → ps.length = i.length → ∃ n : Int, ρ.bindIdx ps i p = some (.int n)
  | [], _, _, hp, _ => absurd hp List.not_mem_nil
  | _ :: _, [], _, _, hl => by simp at hl
  | q :: ps, v :: vs, p, hp, hl => by
    rw [bindIdx_cons]
    by_cases h : q = p
    · exact ⟨v, by simp [h]⟩
    · simp only [h, ↓reduceIte]
      exact bindIdx_mem ρ ((List.mem_cons.1 hp).resolve_left (Ne.symm h)) (by simpa using hl)

/-- Binding fresh map parameters keeps the facts and makes the parameters
integer facts. -/
theorem Constraints.satisfies_addInts {I : ExprInterp} {C : Constraints} {ρ : Valuation} {ps : List Sym} {i : Idx}
    (h : C.satisfies I ρ) (hf : ∀ p ∈ ps, p ∉ C.syms) (hl : ps.length = i.length) :
    (C.addInts ps).satisfies I (ρ.bindIdx ps i) := by
  have hag : ∀ s ∈ C.syms, ρ.bindIdx ps i s = ρ s := fun s hs =>
    bindIdx_not_mem ρ ps i fun hp => hf s hp hs
  obtain ⟨hi, hs, hr⟩ := satisfies_congr h hag
  refine ⟨fun s hm => ?_, hs, hr⟩
  rcases List.mem_append.1 hm with hm | hm
  · exact hi s hm
  · exact bindIdx_mem ρ hm hl

/-! ## Iteration points are distinct -/

theorem Dim.elems_nodup {I : ExprInterp} {ρ : Valuation} {d : Dim} {xs : List Int} (h : d.elems I ρ = some xs) :
    xs.Nodup := by
  unfold Dim.elems at h
  simp only [bind, Option.bind_eq_some_iff] at h
  obtain ⟨lo, -, hi, -, step, -, h⟩ := h
  split at h
  · cases h
  · rename_i hs
    simp only [Option.pure_def, Option.some.injEq] at h
    subst h
    refine List.Nodup.map (fun a b hab => ?_) ?_
    · have hs' : step ≠ 0 := by omega
      have : step * a = step * b := by simpa using hab
      exact Int.eq_of_mul_eq_mul_left hs' this
    · refine List.nodup_flatMap.2 ⟨fun _ _ => List.nodup_singleton _, ?_⟩
      refine (List.nodup_range).imp fun {a b} hab => ?_
      simp only [List.disjoint_left]
      intro x hx hy
      simp only [pure, List.mem_singleton] at hx hy
      exact hab (by omega)

theorem Range.product_nodup : ∀ {es : List (List Int)}, (∀ e ∈ es, e.Nodup) → (Range.product es).Nodup
  | [], _ => by simp [Range.product]
  | e :: es, h => by
    simp only [Range.product]
    have ih := Range.product_nodup fun e' he' => h e' (List.mem_cons_of_mem _ he')
    refine List.nodup_flatMap.2 ⟨fun i _ => ih.map fun a b hab => by simpa using hab, ?_⟩
    refine (h e (List.mem_cons_self ..)).imp fun {i i'} hne => ?_
    simp only [List.disjoint_left, List.mem_map]
    rintro _ ⟨x, -, rfl⟩ ⟨y, -, hy⟩
    simp only [List.cons.injEq] at hy
    exact hne hy.1.symm

/-- The points of a map range are distinct. -/
theorem Range.points_nodup {I : ExprInterp} {ρ : Valuation} {r : Range} {pts : List Idx}
    (h : r.points I ρ = some pts) : pts.Nodup := by
  simp only [Range.points, bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
  obtain ⟨es, hes, rfl⟩ := h
  rw [List.mapM_option_some_iff] at hes
  refine Range.product_nodup fun e he => ?_
  obtain ⟨d, -, hd⟩ := List.Forall₂.exists_left_of_mem hes he
  exact Dim.elems_nodup hd

/-! ## Private transients -/

/-- The accesses of `B` to containers that are not private (`priv` holds
memory names, `Ctx.privateTransients`). -/
def Access.shared (ctx : Ctx) (priv : List Name) (B : List Access) : List Access :=
  B.filter fun a => !priv.contains (ctx.cname a.data)

/-- Dropping the private containers from a log leaves it within the shared
accesses. -/
theorem Log.Within.dropShared {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {B : List Access} {g : Log} (hna : NoAlias m₀)
    (hw : g.Within (LocPred.ofAccesses ctx m₀ ρ B)) (priv : List Name) :
    (g.dropNames priv).Within (LocPred.ofAccesses ctx m₀ ρ (Access.shared ctx priv B)) := by
  have keep : ∀ {a : Access} {l : Loc}, a ∈ B → a.Hit ctx m₀ ρ l → priv.contains l.name = false →
      a ∈ Access.shared ctx priv B := by
    intro a l ha hh hn
    obtain ⟨o, rfl⟩ := Access.hit_eq hna hh
    exact List.mem_filter.2 ⟨ha, by simpa [Loc.elem] using hn⟩
  have hd := hw.dropNames priv
  refine ⟨fun l hl => ?_, fun x hx => ?_, fun x hx => ?_, hd.aliasSets, hd.queues⟩
  · obtain ⟨a, ha, hk, hh⟩ := hw.reads l (List.mem_filter.1 hl).1
    exact ⟨a, keep ha hh (by simpa using (List.mem_filter.1 hl).2), hk, hh⟩
  · obtain ⟨a, ha, hk, hh⟩ := hw.writes x (List.mem_filter.1 hx).1
    exact ⟨a, keep ha hh (by simpa using (List.mem_filter.1 hx).2), hk, hh⟩
  · obtain ⟨a, ha, hk, hh⟩ := hw.wcrs x (List.mem_filter.1 hx).1
    exact ⟨a, keep ha hh (by simpa using (List.mem_filter.1 hx).2), hk, hh⟩

/-! ## Iteration logs paired with their points -/

/-- **`IterDen.footprint`, paired**: the `k`-th iteration log is within the
footprint under the valuation of the `k`-th iteration point. -/
theorem IterExec.fp_pairs {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym}
    {priv : List Name} {pre : St} {F : Nat} {A : List Access} (hinc : FpIncl ctx s F (s.directChildren (some entry)) A)
    (hal : pre.mem.aliases = m₀.aliases) (hna : NoAlias m₀) :
    ∀ {pts : List Idx} {r : Res (List Log)} {logs : List Log}, IterExec ctx ρ s entry params priv pre pts r →
      r = .ok logs → List.Forall₂ (fun g i => g.Within
        (LocPred.ofAccesses ctx m₀ (ρ.bindIdx params i) (Access.shared ctx priv A))) logs pts
  | _, _, _, .nil, hr => by cases hr; exact .nil
  | _, _, _, .headErr _, hr => by cases hr
  | _, _, _, .tailErr _ _, hr => by cases hr
  | _, _, _, .cons hb hrest, hr => by
    cases hr
    exact .cons (((OrderedExec.fp_step hb rfl hinc (by simpa [iterStart] using hal)).2 Log.Within.empty).dropShared hna _)
      (IterExec.fp_pairs hinc hal hna hrest rfl)

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → List.Forall₂ (fun a b => b ∈ l₂ ∧ R a b) l₁ l₂
  | _, _, .nil => .nil
  | _, _, .cons h hs =>
    .cons ⟨List.mem_cons_self .., h⟩ ((forall₂_mem_right hs).imp fun _ _ ⟨hb, hr⟩ => ⟨List.mem_cons_of_mem _ hb, hr⟩)

/-! ## The static check of a state -/

section Check

variable (ctx : Ctx) (s : State) (F : Nat) (orc : Query → Bool)

/-- The static footprints of nodes `p` and `q` are separated (`accSep`). -/
def nodesSep (p q : NodeId) : Bool :=
  match s.node? p, s.node? q with
  | some np, some nq =>
    match nodeFp ctx s F np, nodeFp ctx s F nq with
    | some A, some B => A.all fun a => B.all fun b => accSep a b
    | _, _ => false
  | _, _ => true

/-- No happens-before race among the nodes `ids` of one sequential
execution: every two of them are ordered by a path or separated. -/
def hbOk (ids : List NodeId) : Bool :=
  ids.all fun p => ids.all fun q => p == q || s.reaches p q || s.reaches q p || nodesSep ctx s F p q

/-- Distinct iterations of the map entered at `n` do not conflict (accesses
to its private transients excepted: they are dropped from the iteration
logs, `Decisions.md#D39`). -/
def mapOk (C : Constraints) (n : DNode) (mp : MapNode) : Bool :=
  decide mp.params.Nodup && mp.params.length == mp.range.dims.length && ctx.rangePure mp.range &&
  match scopeFp ctx s F (s.directChildren (some n.id)) with
  | some B =>
    let B' := Access.shared ctx (ctx.privateTransients s n.id) B
    B'.all fun a => B'.all fun b => iterSep ctx orc C mp.params mp.range a b
  | none => false

/-- Race freedom of a sequential execution of the nodes `ids` under facts
`C`, and recursively of the maps among them (`G` bounds the nesting depth). -/
def bodyOk : Nat → Constraints → List NodeId → Bool
  | 0, _, _ => false
  | G + 1, C, ids => (scopeFp ctx s F ids).isSome && hbOk ctx s F ids &&
      ids.all fun id => match s.node? id with
        | some n => match n.kind with
          | .mapEntry mp => mapOk ctx s F orc C n mp && mp.params.all (fun p => !C.syms.contains p) &&
              bodyOk G (C.addInts mp.params) (s.directChildren (some n.id))
          | _ => true
        | none => true

/-- Two components of a state are separated. -/
def compsSep (c₁ c₂ : List NodeId) : Bool :=
  match scopeFp ctx s F (s.topLevelOf c₁), scopeFp ctx s F (s.topLevelOf c₂) with
  | some A, some B => A.all fun a => B.all fun b => accSep a b
  | _, _ => false

end Check

/-- Every two elements at distinct positions are related. -/
def pairwiseB {α : Type} (R : α → α → Bool) : List α → Bool
  | [] => true
  | x :: xs => xs.all (R x) && pairwiseB R xs

theorem pairwiseB_iff {α : Type} (R : α → α → Bool) (l : List α) :
    pairwiseB R l = true ↔ l.Pairwise fun a b => R a b = true := by
  induction l with
  | nil => simp [pairwiseB]
  | cons x xs ih => simp [pairwiseB, ih, List.pairwise_cons]

/-- The static race check of a state under facts `C`. -/
def stateOk (ctx : Ctx) (orc : Query → Bool) (C : Constraints) (s : State) : Bool :=
  let F := s.nodes.length + 1
  s.components.all (fun comp => bodyOk ctx s F orc F C (s.topLevelOf comp)) && pairwiseB (compsSep ctx s F) s.components

theorem bodyOk_succ {orc : Query → Bool} {ctx : Ctx} {s : State} {F G : Nat} {C : Constraints} {ids : List NodeId}
    (h : bodyOk ctx s F orc G C ids = true) : ∃ G', G = G' + 1 := by
  cases G with
  | zero => simp [bodyOk] at h
  | succ G' => exact ⟨G', rfl⟩

theorem bodyOk_spec {orc : Query → Bool} {ctx : Ctx} {s : State} {F G : Nat} {C : Constraints} {ids : List NodeId}
    (h : bodyOk ctx s F orc (G + 1) C ids = true) :
    (∃ A, scopeFp ctx s F ids = some A) ∧ hbOk ctx s F ids = true ∧
      ∀ id ∈ ids, ∀ n, s.node? id = some n → ∀ mp, n.kind = .mapEntry mp →
        mapOk ctx s F orc C n mp = true ∧ (∀ p ∈ mp.params, p ∉ C.syms) ∧
          bodyOk ctx s F orc G (C.addInts mp.params) (s.directChildren (some n.id)) = true := by
  simp only [bodyOk, Bool.and_eq_true, List.all_eq_true, Option.isSome_iff_exists] at h
  obtain ⟨⟨hfp, hhb⟩, hall⟩ := h
  refine ⟨hfp, hhb, fun id hid n hn mp hk => ?_⟩
  have := hall id hid
  simp only [hn, hk, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at this
  obtain ⟨⟨hm, hf⟩, hb⟩ := this
  exact ⟨hm, fun p hp hc => by simpa [hc] using hf p hp, hb⟩

/-! ## Merges of separated logs succeed -/

theorem scopeFinish_clean {ctx : Ctx} {pre : St} {logs : List Log} (h : Log.pairwiseFree logs = true) :
    (scopeFinish ctx pre logs).Clean := by
  unfold scopeFinish
  rw [merge_eq_apply _ _ h]
  exact Res.Clean.ok _

theorem stateFinish_clean {ctx : Ctx} {cst : CSt} {logs : List Log} (h : Log.pairwiseFree logs = true) :
    (stateFinish ctx cst logs).Clean := by
  unfold stateFinish
  rw [merge_eq_apply _ _ h]
  exact Res.Clean.ok _

/-- Iteration logs within separated footprints are pairwise free. -/
theorem iterLogs_free {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid) {ctx : Ctx} {m₀ : Mem} {C : Constraints} {params : List Sym} {rng : Range} {ρ : Valuation}
    {pts : List Idx} {logs : List Log} {B : List Access} (hna : NoAlias m₀) (hC : C.satisfies ctx.I.expr ρ)
    (hrng : rng.points ctx.I.expr ρ = some pts) (hnd : params.Nodup) (hlen : params.length = rng.dims.length)
    (hsep : ∀ a ∈ B, ∀ b ∈ B, iterSep ctx orc C params rng a b = true)
    (hw : List.Forall₂ (fun g i => g.Within (LocPred.ofAccesses ctx m₀ (ρ.bindIdx params i) B)) logs pts) :
    Log.pairwiseFree logs = true := by
  rw [Log.pairwiseFree_iff_pairwise_free]
  refine ListLemmas.pairwise_of_forall₂ (R := fun i j : Idx => i ∈ pts ∧ j ∈ pts ∧ i ≠ j)
    (fun g i h j ⟨hi, hg⟩ ⟨hj, hh⟩ ⟨_, _, hne⟩ =>
      free_of_sep hg hh fun a ha b hb => iterSep_sound horc hna hC hrng hi hj hne hnd hlen (hsep a ha b hb))
    (forall₂_mem_right hw) ?_
  exact ((Range.points_nodup hrng).imp_of_mem fun hi hj hne => ⟨hi, hj, hne⟩)

/-! ## Dataflow executions in the fragment never race -/

section Dataflow

variable {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid)
include horc

variable {ctx : Ctx} {s : State} {F : Nat}

/-- The happens-before check of a sequential execution passes. -/
theorem hbFree_of_ok {ρ : Valuation} {all direct order : List NodeId} {st st₁ st₂ : St}
    {tr : List (NodeId × Log)} {A : List Access} (hna : NoAlias st.mem)
    (ho : s.isTopologicalOrder all order = true) (hp : bindViews ctx ρ s direct st = .ok st₁)
    (hs : SeqExec ctx ρ s (order.filter direct.contains) st₁ (.ok (st₂, tr)))
    (hfp : scopeFp ctx s F direct = some A) (hhb : hbOk ctx s F direct = true) : s.hbFree tr = true := by
  have hinc := scopeFp_incl hfp
  have hal₁ : st₁.mem.aliases = st.mem.aliases :=
    (bindViews_step (m₀ := st.mem) (P := LocPred.ofAccesses ctx st.mem ρ A) hp hinc rfl).1
  have htr := SeqExec.fp_trace (m₀ := st.mem) hs (hinc.of_sub fun x hx => by simpa using (List.mem_filter.1 hx).2)
    hal₁
  have hids := SeqExec.trace_ids hs
  have hnd : (tr.map Prod.fst).Nodup := by
    rw [hids]
    simp only [State.isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq] at ho
    exact ho.1.2.filter _
  rw [State.hbFree_iff]
  refine (List.pairwise_map.1 hnd).imp_of_mem fun {x y} hx hy hne => ?_
  have hxd : x.1 ∈ direct := by
    have : x.1 ∈ tr.map Prod.fst := List.mem_map_of_mem hx
    rw [hids] at this; simpa using (List.mem_filter.1 this).2
  have hyd : y.1 ∈ direct := by
    have : y.1 ∈ tr.map Prod.fst := List.mem_map_of_mem hy
    rw [hids] at this; simpa using (List.mem_filter.1 this).2
  simp only [hbOk, List.all_eq_true, Bool.or_eq_true, beq_iff_eq] at hhb
  rcases hhb x.1 hxd y.1 hyd with ((h | h) | h) | h
  · exact absurd h hne
  · simp [State.hbConflict, h]
  · simp [State.hbConflict, h]
  · obtain ⟨nx, Ax, hnx, hAx, hwx⟩ := htr x hx
    obtain ⟨ny, Ay, hny, hAy, hwy⟩ := htr y hy
    simp only [nodesSep, hnx, hny, hAx, hAy, List.all_eq_true] at h
    have hf := free_of_sep hwx hwy fun a ha b hb => accSep_sound hna (h a ha b hb)
    simp [State.hbConflict, (Log.conflicts_eq_false_iff _ _).2 hf]

mutual

/-- A node of the fragment that passes the check never races. -/
theorem NodeExec.clean {ρ : Valuation} {n : DNode} {st : St} {r : Res St} {G : Nat} {C : Constraints}
    {A : List Access} :
    NodeExec ctx ρ s n st r → C.satisfies ctx.I.expr ρ → NoAlias st.mem → nodeFp ctx s F n = some A →
      (∀ mp, n.kind = .mapEntry mp → mapOk ctx s F orc C n mp = true ∧ (∀ p ∈ mp.params, p ∉ C.syms) ∧
          bodyOk ctx s F orc G (C.addInts mp.params) (s.directChildren (some n.id)) = true) →
      r.Clean
  | .simple _, _, _, _, _ => execSimple_clean _ _ _ _ _
  | .mapSetupErr _ hs, _, _, _, _ => Res.Clean.err_of hs (mapSetup_clean _ _ _ _ _ _)
  | .mapIterErr hk hs hi, hC, hna, hfp, hmap => by
    obtain ⟨f, B', -, hdyn, -, -⟩ := nodeFp_mapEntry hk hfp
    obtain ⟨hρ, hst, hpts⟩ := mapSetup_eq hdyn hs
    subst hρ hst
    obtain ⟨hm, hfresh, hbody⟩ := hmap _ hk
    simp only [mapOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hm
    obtain ⟨⟨⟨hnd, hlen⟩, hrp⟩, -⟩ := hm
    simp only [Ctx.rangePoints, Ctx.range_points_pure ctx _ _ hrp, Res.ofOption_eq_ok] at hpts
    refine Res.Clean.err_of rfl (IterExec.clean hi (fun i hi => Constraints.satisfies_addInts hC hfresh (by
      rw [hlen, ((Range.mem_points hpts).1 hi).length_eq])) hna hbody)
  | .mapScope hk hs hi, hC, hna, hfp, hmap => by
    obtain ⟨f, B', -, hdyn, -, -⟩ := nodeFp_mapEntry hk hfp
    obtain ⟨hρ, hst, hpts⟩ := mapSetup_eq hdyn hs
    subst hρ hst
    obtain ⟨hm, -, -⟩ := hmap _ hk
    simp only [mapOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hm
    obtain ⟨⟨⟨hnd, hlen⟩, hrp⟩, hB⟩ := hm
    split at hB
    · rename_i B hB'
      simp only [List.all_eq_true] at hB
      simp only [Ctx.rangePoints, Ctx.range_points_pure ctx _ _ hrp, Res.ofOption_eq_ok] at hpts
      exact scopeFinish_clean (iterLogs_free horc hna hC hpts hnd hlen hB
        (IterExec.fp_pairs (scopeFp_incl hB') rfl hna hi rfl))
    · cases hB
  | .consumeSetupErr hk _, _, _, hfp, _ => by rw [nodeFp_consumeEntry hk] at hfp; cases hfp
  | .consumeTasksErr hk _ _, _, _, hfp, _ => by rw [nodeFp_consumeEntry hk] at hfp; cases hfp
  | .consumeScope hk _ _, _, _, hfp, _ => by rw [nodeFp_consumeEntry hk] at hfp; cases hfp
  | .nestedSetupErr hk _, _, _, hfp, _ => by rw [nodeFp_nested hk] at hfp; cases hfp
  | .nestedInnerErr hk _ _, _, _, hfp, _ => by rw [nodeFp_nested hk] at hfp; cases hfp
  | .nested hk _ _, _, _, hfp, _ => by rw [nodeFp_nested hk] at hfp; cases hfp

/-- A sequential execution of checked nodes never races. -/
theorem SeqExec.clean {ρ : Valuation} {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} {G : Nat}
    {C : Constraints} {A : List Access} :
    SeqExec ctx ρ s ids st r → C.satisfies ctx.I.expr ρ → NoAlias st.mem → FpIncl ctx s F ids A →
      (∀ id ∈ ids, ∀ n, s.node? id = some n → ∀ mp, n.kind = .mapEntry mp →
        mapOk ctx s F orc C n mp = true ∧ (∀ p ∈ mp.params, p ∉ C.syms) ∧
          bodyOk ctx s F orc G (C.addInts mp.params) (s.directChildren (some n.id)) = true) →
      r.Clean
  | .nil, _, _, _, _ => Res.Clean.ok _
  | .missing _, _, _, _, _ => Res.Clean.err (by clean_err)
  | .headErr hn hnode, hC, hna, hinc, hmaps => by
    obtain ⟨An, hAn, -⟩ := hinc _ (List.mem_cons_self ..) _ hn
    exact Res.Clean.err_of rfl (NodeExec.clean hnode hC hna hAn (hmaps _ (List.mem_cons_self ..) _ hn))
  | .tailErr (n := n) (st' := st') hn hnode hrest, hC, hna, hinc, hmaps => by
    obtain ⟨An, hAn, -⟩ := hinc _ (List.mem_cons_self ..) _ hn
    have hal := (NodeExec.fp_step (m₀ := st.mem) hnode rfl hAn rfl).1
    exact SeqExec.clean hrest hC (fun x => by
        show st'.mem.aliases x = none
        rw [hal]; exact hna x)
      (hinc.of_sub fun x hx => List.mem_cons_of_mem _ hx)
      (fun id hid => hmaps id (List.mem_cons_of_mem _ hid))
  | .cons _ _ _, _, _, _, _ => Res.Clean.ok _

/-- A checked scope body or component never races. -/
theorem OrderedExec.clean {ρ : Valuation} {all direct : List NodeId} {st : St} {r : Res St} {G : Nat}
    {C : Constraints} :
    OrderedExec ctx ρ s all direct st r → C.satisfies ctx.I.expr ρ → NoAlias st.mem →
      bodyOk ctx s F orc (G + 1) C direct = true → r.Clean
  | .prologueErr hp, _, _, _ => Res.Clean.err_of hp (bindViews_clean _ _ _ _ _)
  | .seqErr _ hp hs, hC, hna, hok => by
    obtain ⟨⟨A, hfp⟩, -, hmaps⟩ := bodyOk_spec hok
    have hinc := scopeFp_incl hfp
    have hal₁ := (bindViews_step (m₀ := st.mem) (P := LocPred.ofAccesses ctx st.mem ρ A) hp hinc rfl).1
    exact Res.Clean.err_of rfl (SeqExec.clean hs hC (fun x => by rw [hal₁]; exact hna x)
      (hinc.of_sub fun x hx => by simpa using (List.mem_filter.1 hx).2)
      (fun id hid => hmaps id (by simpa using (List.mem_filter.1 hid).2)))
  | .hbRace ho hp hs hf, _, hna, hok => by
    obtain ⟨⟨A, hfp⟩, hhb, -⟩ := bodyOk_spec hok
    rw [hbFree_of_ok horc hna ho hp hs hfp hhb] at hf
    cases hf
  | .run _ _ _ _, _, _, _ => Res.Clean.ok _

/-- The iterations of a checked map never race. -/
theorem IterExec.clean {ρ : Valuation} {entry : NodeId} {params : List Sym} {priv : List Name} {pre : St}
    {pts : List Idx} {r : Res (List Log)} {G : Nat} {C : Constraints} :
    IterExec ctx ρ s entry params priv pre pts r →
      (∀ i ∈ pts, (C.addInts params).satisfies ctx.I.expr (ρ.bindIdx params i)) → NoAlias pre.mem →
      bodyOk ctx s F orc G (C.addInts params) (s.directChildren (some entry)) = true → r.Clean
  | .nil, _, _, _ => Res.Clean.ok _
  | .headErr hb, hC, hna, hok => by
    obtain ⟨G', rfl⟩ := bodyOk_succ hok
    exact Res.Clean.err_of rfl (OrderedExec.clean hb (hC _ (List.mem_cons_self ..)) (by simpa [NoAlias, iterStart] using hna) hok)
  | .tailErr _ hr, hC, hna, hok => IterExec.clean hr (fun i hi => hC i (List.mem_cons_of_mem _ hi)) hna hok
  | .cons _ _, _, _, _ => Res.Clean.ok _

end

end Dataflow

/-! ## States -/

section StateLevel

variable {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid)
include horc

variable {ctx : Ctx}

theorem CompsExec.clean {ρ : Valuation} {s : State} {m : Mem} {F G : Nat} {C : Constraints} :
    ∀ {comps : List (List NodeId)} {r : Res (List Log)}, CompsExec ctx ρ s m comps r →
      C.satisfies ctx.I.expr ρ → NoAlias m → (∀ comp ∈ comps, bodyOk ctx s F orc G C (s.topLevelOf comp) = true) →
      r.Clean
  | _, _, .nil, _, _, _ => Res.Clean.ok _
  | _, _, .headErr hc, hC, hna, hok => by
    have h := hok _ (List.mem_cons_self ..)
    obtain ⟨G', rfl⟩ := bodyOk_succ h
    exact Res.Clean.err_of rfl (OrderedExec.clean horc hc hC hna h)
  | _, _, .tailErr _ hr, hC, hna, hok =>
    CompsExec.clean hr hC hna fun comp hc => hok comp (List.mem_cons_of_mem _ hc)
  | _, _, .cons _ _, _, _, _ => Res.Clean.ok _

theorem stateOk_spec {C : Constraints} {s : State} (h : stateOk ctx orc C s = true) :
    (∀ comp ∈ s.components, bodyOk ctx s (s.nodes.length + 1) orc (s.nodes.length + 1) C (s.topLevelOf comp) = true) ∧
      s.components.Pairwise fun c₁ c₂ => compsSep ctx s (s.nodes.length + 1) c₁ c₂ = true := by
  simp only [stateOk, Bool.and_eq_true, List.all_eq_true] at h
  exact ⟨h.1, (pairwiseB_iff _ _).1 h.2⟩

/-- The logs of the components of a checked state are pairwise free and bind
no alias. -/
theorem CompsExec.logs_free {ρ : Valuation} {s : State} {m : Mem} {C : Constraints} {logs : List Log}
    (h : CompsExec ctx ρ s m s.components (.ok logs)) (hok : stateOk ctx orc C s = true) (hna : NoAlias m) :
    Log.pairwiseFree logs = true ∧ ∀ g ∈ logs, g.aliasSets = [] := by
  obtain ⟨hbody, hsep⟩ := stateOk_spec horc hok
  have hw := CompsExec.fp_logs (F := s.nodes.length + 1) h rfl
  have hw' := forall₂_mem_right hw.flip
  have hsome : ∀ comp ∈ s.components, ∃ A, scopeFp ctx s (s.nodes.length + 1) (s.topLevelOf comp) = some A :=
    fun comp hc => (bodyOk_spec (G := s.nodes.length) (hbody comp hc)).1
  refine ⟨?_, fun g hg => ?_⟩
  · rw [Log.pairwiseFree_iff_pairwise_free]
    refine ListLemmas.pairwise_of_forall₂ (R := fun c₁ c₂ => compsSep ctx s (s.nodes.length + 1) c₁ c₂ = true)
      (fun g c₁ g' c₂ ⟨hc₁, hg⟩ ⟨hc₂, hg'⟩ hcs => ?_) hw' hsep
    obtain ⟨A, hA⟩ := hsome c₁ hc₁
    obtain ⟨B, hB⟩ := hsome c₂ hc₂
    simp only [compsSep, hA, hB, List.all_eq_true] at hcs
    exact free_of_sep (hg A hA) (hg' B hB) fun a ha b hb => accSep_sound hna (hcs a ha b hb)
  · obtain ⟨comp, hc, hgc⟩ := ListLemmas.mem_of_forall₂ hw.flip hg
    obtain ⟨A, hA⟩ := hsome comp hc
    exact (hgc A hA).aliasSets

/-- A checked state never races, and a normal outcome keeps the valuation
and binds no alias. -/
theorem StateExec.good {cst : CSt} {s : State} {r : Res (Ctl × CSt)} {C : Constraints}
    (h : StateExec ctx cst s r) (hok : stateOk ctx orc C s = true) (hC : C.satisfies ctx.I.expr cst.ρ)
    (hna : NoAlias cst.mem) :
    r.Clean ∧ ∀ ctl cst', r = .ok (ctl, cst') → cst'.ρ = cst.ρ ∧ NoAlias cst'.mem := by
  cases h with
  | compsErr hc =>
    refine ⟨Res.Clean.err_of rfl (CompsExec.clean horc hc hC hna (stateOk_spec horc hok).1), fun _ _ h => by cases h⟩
  | @run _ _ _ logs hc =>
    obtain ⟨hfree, hal⟩ := CompsExec.logs_free horc hc hok hna
    refine ⟨stateFinish_clean hfree, fun ctl cst' hr => ?_⟩
    unfold stateFinish at hr
    split at hr
    · rename_i m' hm
      cases hr
      refine ⟨rfl, fun n => ?_⟩
      have hu : (Log.unionAll logs).aliasSets = [] := by
        simp only [Log.unionAll, List.flatMap_eq_nil_iff]; exact hal
      show m'.aliases n = none
      rw [merge_aliases hm hu]; exact hna n
    · cases hr

end StateLevel

/-! ## Control flow -/

mutual

/-- The static race check of a region: every state passes `sOk` (`stateOk`),
and no interstate or loop assignment assigns a symbol of the facts. -/
def Region.raceOk (sOk : State → Bool) (C : Constraints) : Region → Bool
  | ⟨_, _, blocks, _, edges⟩ =>
    Blocks.raceOk sOk C blocks && edges.all fun e => e.assigns.all fun a => !C.syms.contains a.1

def Blocks.raceOk (sOk : State → Bool) (C : Constraints) : List Block → Bool
  | [] => true
  | b :: bs => Block.raceOk sOk C b && Blocks.raceOk sOk C bs

def Block.raceOk (sOk : State → Bool) (C : Constraints) : Block → Bool
  | .state s => sOk s
  | .region r => Region.raceOk sOk C r
  | .loop ⟨_, _, init, _, update, _, _, body⟩ =>
    init.all (fun a => !C.syms.contains a.1) && update.all (fun a => !C.syms.contains a.1) &&
      Region.raceOk sOk C body
  | .cond _ brs => Branches.raceOk sOk C brs
  | .breakBlock | .continueBlock | .returnBlock => true

def Branches.raceOk (sOk : State → Bool) (C : Constraints) : List (Option Expr × Region) → Bool
  | [] => true
  | (_, r) :: rest => Region.raceOk sOk C r && Branches.raceOk sOk C rest

end

mutual

/-- Symbols a region assigns: interstate assignments, loop variables and
loop initializations/updates. -/
def Region.assigned : Region → List Sym
  | ⟨_, _, blocks, _, edges⟩ => Blocks.assigned blocks ++ edges.flatMap fun e => e.assigns.map (·.1)

def Blocks.assigned : List Block → List Sym
  | [] => []
  | b :: bs => Block.assigned b ++ Blocks.assigned bs

def Block.assigned : Block → List Sym
  | .region r => Region.assigned r
  | .loop ⟨_, loopVar, init, _, update, _, _, body⟩ =>
    loopVar.toList ++ init.toList.map (·.1) ++ update.toList.map (·.1) ++ Region.assigned body
  | .cond _ brs => Branches.assigned brs
  | .state _ | .breakBlock | .continueBlock | .returnBlock => []

def Branches.assigned : List (Option Expr × Region) → List Sym
  | [] => []
  | (_, r) :: rest => Region.assigned r ++ Branches.assigned rest

end

section ControlLemmas

variable {orc : Query → Bool} {ctx : Ctx} {C : Constraints}

theorem Region.raceOk_eq (rg : Region) :
    Region.raceOk (stateOk ctx orc C) C rg =
      (Blocks.raceOk (stateOk ctx orc C) C rg.blocks && rg.edges.all fun e => e.assigns.all fun a => !C.syms.contains a.1) := by
  cases rg; rfl

theorem Block.raceOk_loop (l : LoopRegion) :
    Block.raceOk (stateOk ctx orc C) C (.loop l) =
      (l.init.all (fun a => !C.syms.contains a.1) && l.update.all (fun a => !C.syms.contains a.1) &&
        Region.raceOk (stateOk ctx orc C) C l.body) := by
  cases l; rfl

theorem Blocks.raceOk_mem : ∀ {bs : List Block} {b : Block}, Blocks.raceOk (stateOk ctx orc C) C bs = true → b ∈ bs →
    Block.raceOk (stateOk ctx orc C) C b = true
  | [], _, _, hb => absurd hb List.not_mem_nil
  | b' :: bs, b, h, hb => by
    simp only [Blocks.raceOk, Bool.and_eq_true] at h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h.1
    · exact Blocks.raceOk_mem h.2 hb

/-- The facts hold and no container is aliased. -/
def CInv (I : ExprInterp) (C : Constraints) (cst : CSt) : Prop := C.satisfies I cst.ρ ∧ NoAlias cst.mem

/-- A control-flow outcome is not a race, and a normal one keeps `CInv`. -/
def CGood (I : ExprInterp) (C : Constraints) (r : Res (Ctl × CSt)) : Prop :=
  r.Clean ∧ ∀ ctl cst', r = .ok (ctl, cst') → CInv I C cst'

theorem CGood.err_of {I : ExprInterp} {α : Type} {x : Res α} {e : ErrReason} (h : x = .err e) (hx : x.Clean) :
    CGood I C (.err e) :=
  ⟨Res.Clean.err_of h hx, fun _ _ h => by cases h⟩

theorem CGood.ok {I : ExprInterp} {ctl : Ctl} {cst : CSt} (h : CInv I C cst) : CGood I C (.ok (ctl, cst)) :=
  ⟨Res.Clean.ok _, fun _ _ hr => by cases hr; exact h⟩

theorem mapM_keys {α β γ : Type} {f : α → Res (β × γ)} {key : α → β} (hf : ∀ a b, f a = .ok b → b.1 = key a) :
    ∀ {as : List α} {bs : List (β × γ)}, as.mapM f = .ok bs → bs.map Prod.fst = as.map key
  | [], bs, h => by
    simp only [List.mapM_nil] at h
    cases h; rfl
  | a :: as, bs, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨bs', hbs, h⟩ := Res.bind_eq_ok.1 h
    rw [Res.pure_eq_ok] at h
    subst h
    simp only [List.map_cons, hf a b hb, mapM_keys hf hbs]

theorem setAll_not_mem {ρ : Valuation} {vs : List (Sym × Val)} {y : Sym} (h : y ∉ vs.map Prod.fst) :
    Valuation.setAll ρ vs y = ρ y := by
  simp only [Valuation.setAll]
  have : vs.find? (fun x => decide (x.1 = y)) = none := by
    rw [List.find?_eq_none]
    intro x hx
    simp only [decide_eq_true_eq]
    exact fun hxy => h (hxy ▸ List.mem_map_of_mem hx)
  rw [this]

/-- Assignments of symbols outside the facts keep `CInv`. -/
theorem applyAssigns_inv {cst cst' : CSt} {as : List (Sym × Expr)} (h : applyAssigns ctx cst as = .ok cst')
    (hfresh : ∀ a ∈ as, a.1 ∉ C.syms) (hi : CInv ctx.I.expr C cst) : CInv ctx.I.expr C cst' := by
  unfold applyAssigns at h
  obtain ⟨vs, hvs, h⟩ := Res.bind_eq_ok.1 h
  rw [Res.pure_eq_ok] at h
  subst h
  have hk := mapM_keys (key := Prod.fst) (fun a b hab => by
    obtain ⟨v, -, hv⟩ := Res.bind_eq_ok.1 hab
    rw [Res.pure_eq_ok] at hv
    subst hv; rfl) hvs
  refine ⟨Constraints.satisfies_congr hi.1 fun y hy => setAll_not_mem ?_, hi.2⟩
  rw [hk]
  intro hm
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hm
  exact hfresh a ha hy

theorem applyAssign?_inv {cst cst' : CSt} {a : Option (Sym × Expr)} (h : applyAssign? ctx cst a = .ok cst')
    (hfresh : a.all (fun a => !C.syms.contains a.1) = true) (hi : CInv ctx.I.expr C cst) : CInv ctx.I.expr C cst' := by
  cases a with
  | none => cases h; exact hi
  | some a =>
    refine applyAssigns_inv h (fun b hb => ?_) hi
    simp only [List.mem_singleton] at hb
    subst hb
    simpa using hfresh

theorem filterMapM_mem {α β : Type} {f : α → Res (Option β)} :
    ∀ {l : List α} {ys : List β}, l.filterMapM f = .ok ys → ∀ y ∈ ys, ∃ x ∈ l, f x = .ok (some y)
  | [], ys, h, y, hy => by
    simp only [List.filterMapM_nil, Res.pure_eq_ok] at h
    subst h; exact absurd hy List.not_mem_nil
  | x :: xs, ys, h, y, hy => by
    rw [List.filterMapM_cons] at h
    obtain ⟨o, ho, h⟩ := Res.bind_eq_ok.1 h
    cases o with
    | none =>
      obtain ⟨x', hx', hf⟩ := filterMapM_mem h y hy
      exact ⟨x', List.mem_cons_of_mem _ hx', hf⟩
    | some b =>
      obtain ⟨ys', hys, h⟩ := Res.bind_eq_ok.1 h
      rw [Res.pure_eq_ok] at h
      subst h
      rcases List.mem_cons.1 hy with rfl | hy
      · exact ⟨x, List.mem_cons_self .., ho⟩
      · obtain ⟨x', hx', hf⟩ := filterMapM_mem hys y hy
        exact ⟨x', List.mem_cons_of_mem _ hx', hf⟩

theorem trueEdges_mem {rg : Region} {cst : CSt} {bid : BlockId} {es : List IEdge} {ie : IEdge}
    (h : trueEdges ctx rg cst bid = .ok es) (hie : ie ∈ es) : ie ∈ rg.edges := by
  unfold trueEdges at h
  obtain ⟨x, hx, hf⟩ := filterMapM_mem h ie hie
  obtain ⟨b, -, hf⟩ := Res.bind_eq_ok.1 hf
  rw [Res.pure_eq_ok] at hf
  split at hf
  · cases hf
    exact (List.mem_filter.1 hx).1
  · cases hf

theorem blockOk {rg : Region} {bid : BlockId} {b : Block} (hok : Region.raceOk (stateOk ctx orc C) C rg = true)
    (hb : rg.block? bid = some b) : Block.raceOk (stateOk ctx orc C) C b = true := by
  rw [Region.raceOk_eq, Bool.and_eq_true] at hok
  exact Blocks.raceOk_mem hok.1 (List.mem_of_getElem? hb)

theorem loopInit {l : LoopRegion} (hok : Block.raceOk (stateOk ctx orc C) C (.loop l) = true) :
    l.init.all (fun a => !C.syms.contains a.1) = true := by
  rw [Block.raceOk_loop] at hok; simp only [Bool.and_eq_true] at hok; exact hok.1.1

theorem loopUpdate {l : LoopRegion} (hok : Block.raceOk (stateOk ctx orc C) C (.loop l) = true) :
    l.update.all (fun a => !C.syms.contains a.1) = true := by
  rw [Block.raceOk_loop] at hok; simp only [Bool.and_eq_true] at hok; exact hok.1.2

theorem loopBody {l : LoopRegion} (hok : Block.raceOk (stateOk ctx orc C) C (.loop l) = true) : Region.raceOk (stateOk ctx orc C) C l.body = true := by
  rw [Block.raceOk_loop] at hok; simp only [Bool.and_eq_true] at hok; exact hok.2

end ControlLemmas

section ControlFlow

variable {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid)
include horc

variable {ctx : Ctx} {C : Constraints}

theorem StateExec.cgood {cst : CSt} {s : State} {r : Res (Ctl × CSt)} (h : StateExec ctx cst s r)
    (hok : stateOk ctx orc C s = true) (hi : CInv ctx.I.expr C cst) : CGood ctx.I.expr C r := by
  obtain ⟨hc, hinv⟩ := StateExec.good horc h hok hi.1 hi.2
  exact ⟨hc, fun ctl cst' hr => let ⟨hρ, hna⟩ := hinv ctl cst' hr; ⟨hρ ▸ hi.1, hna⟩⟩

mutual

theorem BlockExec.cgood {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} :
    BlockExec ctx cst b r → Block.raceOk (stateOk ctx orc C) C b = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .state h, hok, hi => StateExec.cgood horc h hok hi
  | .region h, hok, hi => RegionExec.cgood h hok hi
  | .loop h, hok, hi => LoopExec.cgood h hok hi
  | .cond h, hok, hi => CondExec.cgood h hok hi
  | .brk, _, hi => CGood.ok hi
  | .cont, _, hi => CGood.ok hi
  | .ret, _, hi => CGood.ok hi

theorem RegionAt.cgood {rg : Region} {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)} :
    RegionAt ctx rg cst bid r → Region.raceOk (stateOk ctx orc C) C rg = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .missing _, _, _ => ⟨Res.Clean.err (by clean_err), fun _ _ h => by cases h⟩
  | .blockErr hb h, hok, hi => BlockExec.cgood h (blockOk hok hb) hi
  | .exit hb h _, hok, hi => BlockExec.cgood h (blockOk hok hb) hi
  | .edgesErr hb h he, _, _ => CGood.err_of he (trueEdges_clean _ _ _ _)
  | .stop hb h _, hok, hi => CGood.ok ((BlockExec.cgood h (blockOk hok hb) hi).2 _ _ rfl)
  | .ambiguous _ _ _, _, _ => ⟨Res.Clean.err (by clean_err), fun _ _ h => by cases h⟩
  | .assignErr _ _ _ ha, _, _ => CGood.err_of ha (applyAssigns_clean _ _ _)
  | .step (ie := ie) hb h he ha hn, hok, hi => by
    have hi₁ := (BlockExec.cgood h (blockOk hok hb) hi).2 _ _ rfl
    have hie := trueEdges_mem he (List.mem_singleton_self _)
    rw [Region.raceOk_eq] at hok
    have hfresh : ∀ a ∈ ie.assigns, a.1 ∉ C.syms := fun a ha hc => by
      simp only [Bool.and_eq_true, List.all_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hok
      simpa [hc] using hok.2 _ hie a ha
    exact RegionAt.cgood hn (by rw [Region.raceOk_eq]; exact hok) (applyAssigns_inv ha hfresh hi₁)

theorem RegionExec.cgood {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)} :
    RegionExec ctx cst rg r → Region.raceOk (stateOk ctx orc C) C rg = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .start h, hok, hi => RegionAt.cgood h hok hi

theorem LoopExec.cgood {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopExec ctx cst l r → Block.raceOk (stateOk ctx orc C) C (.loop l) = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .initErr hi', _, _ => CGood.err_of hi' (applyAssign?_clean _ _ _)
  | .guardFirst hi' _ hg, hok, hi => LoopGuard.cgood hg hok (applyAssign?_inv hi' (loopInit hok) hi)
  | .bodyFirst hi' _ hb, hok, hi => LoopIter.cgood hb hok (applyAssign?_inv hi' (loopInit hok) hi)

theorem LoopGuard.cgood {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopGuard ctx cst l r → Block.raceOk (stateOk ctx orc C) C (.loop l) = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .condErr hc, _, _ => CGood.err_of hc (Ctx.evalBoolR_clean _ _ _ _)
  | .exit _, _, hi => CGood.ok hi
  | .enter _ hb, hok, hi => LoopIter.cgood hb hok hi

theorem LoopIter.cgood {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopIter ctx cst l r → Block.raceOk (stateOk ctx orc C) C (.loop l) = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .bodyErr hb, hok, hi => RegionExec.cgood hb (loopBody hok) hi
  | .ret hb, hok, hi => CGood.ok ((RegionExec.cgood hb (loopBody hok) hi).2 _ _ rfl)
  | .brk hb, hok, hi => CGood.ok ((RegionExec.cgood hb (loopBody hok) hi).2 _ _ rfl)
  | .tail hb _ ht, hok, hi => LoopTail.cgood ht hok ((RegionExec.cgood hb (loopBody hok) hi).2 _ _ rfl)

theorem LoopTail.cgood {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopTail ctx cst l r → Block.raceOk (stateOk ctx orc C) C (.loop l) = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .updateErr _ hu, _, _ => CGood.err_of hu (applyAssign?_clean _ _ _)
  | .updateGuard _ hu hg, hok, hi => LoopGuard.cgood hg hok (applyAssign?_inv hu (loopUpdate hok) hi)
  | .condFirstErr _ hc, _, _ => CGood.err_of hc (Ctx.evalBoolR_clean _ _ _ _)
  | .condFirstExit _ _, _, hi => CGood.ok hi
  | .condFirstUpdateErr _ _ hu, _, _ => CGood.err_of hu (applyAssign?_clean _ _ _)
  | .condFirstNext _ _ hu hb, hok, hi => LoopIter.cgood hb hok (applyAssign?_inv hu (loopUpdate hok) hi)

theorem CondExec.cgood {cst : CSt} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)} :
    CondExec ctx cst brs r → Branches.raceOk (stateOk ctx orc C) C brs = true → CInv ctx.I.expr C cst → CGood ctx.I.expr C r
  | .nil, _, hi => CGood.ok hi
  | .els h, hok, hi => RegionExec.cgood h (by simp only [Branches.raceOk, Bool.and_eq_true] at hok; exact hok.1) hi
  | .condErr hc, _, _ => CGood.err_of hc (Ctx.evalBoolR_clean _ _ _ _)
  | .taken _ h, hok, hi =>
    RegionExec.cgood h (by simp only [Branches.raceOk, Bool.and_eq_true] at hok; exact hok.1) hi
  | .skipped _ h, hok, hi =>
    CondExec.cgood h (by simp only [Branches.raceOk, Bool.and_eq_true] at hok; exact hok.2) hi

end

end ControlFlow

/-! ## The check does not depend on the interpretation -/

section Indep

variable (ctx : Ctx) (I : Interp) (s : State) (F : Nat) (orc : Query → Bool)

theorem nodeFp_withI : ∀ (G : Nat) (n : DNode), nodeFp { ctx with I } s G n = nodeFp ctx s G n := by
  intro G
  induction G with
  | zero => intro n; rfl
  | succ f ih =>
    intro n
    simp only [nodeFp, ih]
    rfl

theorem scopeFp_withI (ids : List NodeId) : scopeFp { ctx with I } s F ids = scopeFp ctx s F ids := by
  simp only [scopeFp, nodeFp_withI]

theorem hbOk_withI (ids : List NodeId) : hbOk { ctx with I } s F ids = hbOk ctx s F ids := by
  simp only [hbOk, nodesSep, nodeFp_withI]

theorem mapOk_withI (C : Constraints) (n : DNode) (mp : MapNode) :
    mapOk { ctx with I } s F orc C n mp = mapOk ctx s F orc C n mp := by
  simp only [mapOk, scopeFp_withI]
  rfl

theorem bodyOk_withI : ∀ (G : Nat) (C : Constraints) (ids : List NodeId),
    bodyOk { ctx with I } s F orc G C ids = bodyOk ctx s F orc G C ids := by
  intro G
  induction G with
  | zero => intro C ids; rfl
  | succ G ih =>
    intro C ids
    simp only [bodyOk, scopeFp_withI, hbOk_withI, mapOk_withI, ih]

theorem compsSep_withI : compsSep { ctx with I } s F = compsSep ctx s F := by
  funext c₁ c₂
  simp only [compsSep, scopeFp_withI]

theorem stateOk_withI (C : Constraints) : stateOk { ctx with I } orc C s = stateOk ctx orc C s := by
  simp only [stateOk, bodyOk_withI, compsSep_withI]

end Indep

/-! ## Race freedom of an SDFG -/

/-- An SDFG is race-free when no execution from a configuration satisfying
its precondition (`Prog.Pre`: lawful interpretation, facts, no aliases)
ends in a race. -/
def RaceFree (g : SDFG) : Prop :=
  ∀ I ρ₀ m₀, g.prog.Pre I ρ₀ m₀ → ∀ w, ¬ SDFG.Exec I g ρ₀ m₀ (.err (.race w))

/-- The facts the check may use: those about symbols that are neither
assigned anywhere (interstate edges, loops) nor constants. -/
def raceFacts (g : SDFG) : Constraints :=
  g.facts.restrict fun x => !g.top.assigned.contains x && !g.constants.any (·.1 == x)

/-- The static race check of an SDFG with undecided disjointness obligations
accepted by the oracle `orc` (`raceFreeWith_sound` assumes them valid). -/
def raceFreeWith (orc : Query → Bool) (g : SDFG) : Bool :=
  Region.raceOk (stateOk { I := Interp.uninterpreted, g } orc (raceFacts g)) (raceFacts g) g.top

/-- **The static race check** of an SDFG (sound, incomplete): `true` means
`RaceFree` (`raceFree?_sound`). -/
def raceFree? (g : SDFG) : Bool := raceFreeWith (fun _ => false) g

theorem raceFacts_initial {g : SDFG} {I : ExprInterp} {ρ₀ : Valuation} (h : g.facts.satisfies I ρ₀) :
    (raceFacts g).satisfies I (g.withConstants ρ₀) := by
  refine Constraints.satisfies_congr (Constraints.satisfies_restrict _ h) fun y hy => ?_
  have hk := Constraints.keep_of_mem_restrict hy
  simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, List.any_eq_false, beq_iff_eq] at hk
  simp only [SDFG.withConstants]
  have : g.constants.find? (fun c => decide (c.1 = y)) = none := by
    rw [List.find?_eq_none]
    intro c hc
    simpa using hk.2 c hc
  rw [this]

/-- **Soundness of the assumed-obligations mode**: if every query the
oracle accepts is valid, `raceFreeWith` implies race freedom. -/
theorem raceFreeWith_sound {orc : Query → Bool} (horc : ∀ q, orc q = true → q.Valid) {g : SDFG}
    (h : raceFreeWith orc g = true) : RaceFree g := by
  intro I ρ₀ m₀ hpre w hx
  obtain ⟨-, hf, hna⟩ := hpre
  obtain ⟨res, hres, hr⟩ := hx
  have hctx : ({ I := Interp.uninterpreted, g } : Ctx) = { ({ I, g } : Ctx) with I := Interp.uninterpreted } := rfl
  have hok : Region.raceOk (stateOk { I, g } orc (raceFacts g)) (raceFacts g) g.top = true := by
    have := funext fun s => stateOk_withI ({ I, g } : Ctx) Interp.uninterpreted s orc (raceFacts g)
    unfold raceFreeWith at h
    rw [hctx, this] at h
    exact h
  cases hres with
  | err hrg =>
    have hc := (RegionExec.cgood horc hrg hok ⟨raceFacts_initial hf, hna⟩).1
    simp only [Res.map_err, Res.err.injEq] at hr
    exact hc w (by rw [hr])
  | escape _ _ => simp at hr
  | run _ _ => simp at hr

/-- **Soundness of `raceFree?`.** -/
theorem raceFree?_sound {g : SDFG} (h : raceFree? g = true) : RaceFree g :=
  raceFreeWith_sound (fun _ h => by cases h) h

/-! ## Collecting obligations (tooling, not trusted) -/

/-- The undecided disjointness obligations of one map: those of every pair of
shared accesses that only `iterDisjointWith?` could separate. -/
def mapObligations (ctx : Ctx) (s : State) (F : Nat) (C : Constraints) (n : DNode) (mp : MapNode) : List Query :=
  match scopeFp ctx s F (s.directChildren (some n.id)) with
  | some B =>
    let B' := Access.shared ctx (ctx.privateTransients s n.id) B
    B'.flatMap fun a => B'.flatMap fun b =>
      if accSep a b then []
      else match ctx.desc? a.data with
        | some d =>
          ((iterObligations? C mp.params mp.range d (a.sub.getD (.range d.fullRange)) d
            (b.sub.getD (.range d.fullRange))).getD [])
        | none => []
  | none => []

/-- Obligations of the maps among `ids`, recursively (`bodyOk`'s traversal). -/
def bodyObligations (ctx : Ctx) (s : State) (F : Nat) : Nat → Constraints → List NodeId → List Query
  | 0, _, _ => []
  | G + 1, C, ids => ids.flatMap fun id => match s.node? id with
    | some n => match n.kind with
      | .mapEntry mp => mapObligations ctx s F C n mp ++
          bodyObligations ctx s F G (C.addInts mp.params) (s.directChildren (some n.id))
      | _ => []
    | none => []

mutual

/-- Every obligation `raceFreeWith` may ask the oracle about (a superset).
`raceFreeWith (fun q => (raceObligations g).contains q) g` is the
assumed-obligations check of the kernel fallback (`dace-equiv racefree
--emit-obligations`). -/
def Region.raceObligations (ctx : Ctx) (C : Constraints) : Region → List Query
  | ⟨_, _, blocks, _, _⟩ => Blocks.raceObligations ctx C blocks

def Blocks.raceObligations (ctx : Ctx) (C : Constraints) : List Block → List Query
  | [] => []
  | b :: bs => Block.raceObligations ctx C b ++ Blocks.raceObligations ctx C bs

def Block.raceObligations (ctx : Ctx) (C : Constraints) : Block → List Query
  | .state s =>
    let F := s.nodes.length + 1
    s.components.flatMap fun comp => bodyObligations ctx s F F C (s.topLevelOf comp)
  | .region r => Region.raceObligations ctx C r
  | .loop ⟨_, _, _, _, _, _, _, body⟩ => Region.raceObligations ctx C body
  | .cond _ brs => Branches.raceObligations ctx C brs
  | .breakBlock | .continueBlock | .returnBlock => []

def Branches.raceObligations (ctx : Ctx) (C : Constraints) : List (Option Expr × Region) → List Query
  | [] => []
  | (_, r) :: rest => Region.raceObligations ctx C r ++ Branches.raceObligations ctx C rest

end

/-- The obligations of an SDFG, without duplicates. -/
def raceObligations (g : SDFG) : List Query :=
  (Region.raceObligations { I := Interp.uninterpreted, g } (raceFacts g) g.top).eraseDups

end Dace
