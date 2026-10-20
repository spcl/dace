-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Footprint

/-!
# Static footprints in static form (`IterDen.footprint`)

`Dace.SDFG.Footprint` proves that successful dataflow executions in the
fragment log only locations hit by their static footprint (`nodeFp`), where
subsets and map ranges are enumerated as the semantics does: under
`Ctx.exprInterp`/`Ctx.val` of the memory at the time, which may read data
(`Decisions.md#D33`; R1/R2 design F5).  This file removes that memory
dependence for **pure** subsets and ranges (`Ctx.pointsPure`,
`Ctx.rangePure` of `Dace.SDFG.Footprinted`: no `subscript`, no free symbol
naming a scalar container), so that `raceFree?` (R2.7) can combine
footprints with `iterDisjoint?` (`Dace.Core.SymSubset`), which reasons about
`Subset.points` and `DataDesc.linearOffset` under `ctx.I.expr` and a plain
valuation:

* `Access.hit_static`: a hit of a pure access (`Access.pureIn`) is a point of
  its subset under `ctx.I.expr` and the iteration valuation extended by
  values of the inner map parameters within their ranges (`Binders.ExtS`);
* `Ctx.elemOffset_of_not_scalar`: element offsets of non-scalars are
  `DataDesc.linearOffset`; `Mem.resolve_of_none`: without an alias binding
  of the container, resolution is the identity;
* `IterExec.footprint`: the statement `IterDen.footprint` of the design, for
  the logs of the iterations of a map scope.

The fragment (`nodeFp = some _` plus `Access.pureIn` of every access)
excludes, beyond `Dace.SDFG.Footprint`: data in subset bounds and map
ranges.  Unlogged reads (F5) remain unlogged: they are not part of any
footprint.
-/

set_option autoImplicit false

namespace Dace

/-- `Binders.Ext` with ranges enumerated under `I` and the plain valuation. -/
def Binders.ExtS (I : ExprInterp) : Valuation → List (List Sym × Range) → Valuation → Prop
  | ρ, [], ρ' => ρ' = ρ
  | ρ, (ps, r) :: bs, ρ' => ∃ pts, r.points I ρ = some pts ∧ ∃ i ∈ pts, Binders.ExtS I (ρ.bindIdx ps i) bs ρ'

/-- All expressions of the access (binder ranges, enumerated subset) are pure. -/
def Access.pureIn (ctx : Ctx) (a : Access) : Bool :=
  a.binders.all (fun b => ctx.rangePure b.2) && ctx.pointsPure a.data a.sub

theorem Binders.ext_static {ctx : Ctx} :
    ∀ {ρ ρ' : Valuation} {bs : List (List Sym × Range)}, bs.all (fun b => ctx.rangePure b.2) = true →
      Binders.Ext ctx ρ bs ρ' → Binders.ExtS ctx.I.expr ρ bs ρ'
  | _, _, [], _, h => h
  | ρ, _, (ps, r) :: bs, hp, ⟨m, pts, hr, i, hi, h⟩ => by
    simp only [List.all_cons, Bool.and_eq_true] at hp
    simp only [Ctx.rangePoints, Ctx.range_points_pure ctx ρ m hp.1, Res.ofOption_eq_ok] at hr
    exact ⟨pts, hr, i, hi, Binders.ext_static hp.2 h⟩

/-- Enumeration of a pure subset of a known container, in static form. -/
theorem Ctx.pointsOf_static {ctx : Ctx} {ρ : Valuation} {m : Mem} {n : Name} {sub : Option Subset}
    {d : DataDesc} {pts : List Idx} (hd : ctx.desc? n = some d) (hp : ctx.pointsPure n sub = true)
    (h : ctx.pointsOf ρ m n sub = .ok pts) : (sub.getD (.range d.fullRange)).points ctx.I.expr ρ = some pts := by
  have h' : ctx.subsetPure (sub.getD (.range d.fullRange)) = true := by
    cases sub with
    | some s => exact hp
    | none => simpa [Ctx.pointsPure, hd, Ctx.subsetPure] using hp
  simp only [Ctx.pointsOf, hd, Ctx.subset_points_pure ctx ρ m h', Res.ofOption_eq_ok] at h
  exact h

/-- **A hit of a pure access in static form**: the location is the element,
resolved in `m₀`, at the element offset of a point of the access's subset
(enumerated under `ctx.I.expr`) under the valuation extended by the inner map
parameters. -/
theorem Access.hit_static {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {a : Access} {l : Loc}
    (h : a.Hit ctx m₀ ρ l) (hp : a.pureIn ctx = true) :
    ∃ ρ', Binders.ExtS ctx.I.expr ρ a.binders ρ' ∧ ∃ d off, ctx.desc? a.data = some d ∧
      (∃ pts, (a.sub.getD (.range d.fullRange)).points ctx.I.expr ρ' = some pts ∧
        ∃ p ∈ pts, ctx.elemOffset ρ' d p = some off) ∧
      l = m₀.resolve (Loc.elem (ctx.cname a.data) off) := by
  obtain ⟨ρ', hext, off, ⟨d, hd, hoff⟩, hl⟩ := h
  simp only [Access.pureIn, Bool.and_eq_true] at hp
  obtain ⟨m, pts, hpts, p, hpm, ho⟩ := hoff
  exact ⟨ρ', Binders.ext_static hp.1 hext, d, off, hd, ⟨pts, Ctx.pointsOf_static hd hp.2 hpts, p, hpm, ho⟩, hl⟩

theorem Ctx.elemOffset_of_not_scalar {ctx : Ctx} {ρ : Valuation} {d : DataDesc} {idx : Idx}
    (h : d.isScalar = false) : ctx.elemOffset ρ d idx = d.linearOffset ctx.I.expr ρ idx := by
  simp [Ctx.elemOffset, h]

theorem Mem.resolve_of_none {m : Mem} {l : Loc} (h : m.aliases l.name = none) : m.resolve l = l := by
  simp [Mem.resolve, Mem.resolveFuel, h]

/-- **`IterDen.footprint`**: every location in the reads, plain writes or
WCR contributions (with operator `op`) of the log of an iteration of a map
in the fragment is hit, with the matching kind, by an access of the static
footprint of the scope body (`scopeFp` of the entry's direct children), under
the valuation of some iteration point `i` of the map range; the log binds no
alias and touches no queue. -/
theorem IterExec.footprint {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {s : State} {entry : NodeId}
    {params : List Sym} {priv : List Name} {pre : St} {pts : List Idx} {logs : List Log} {F : Nat}
    {A : List Access} (h : IterExec ctx ρ s entry params priv pre pts (.ok logs))
    (hfp : scopeFp ctx s F (s.directChildren (some entry)) = some A) (hal : pre.mem.aliases = m₀.aliases) :
    ∀ g ∈ logs, ∃ i ∈ pts,
      (∀ l ∈ g.reads, ∃ a ∈ A, a.kind = .read ∧ a.Hit ctx m₀ (ρ.bindIdx params i) l) ∧
      (∀ x ∈ g.writes, ∃ a ∈ A, a.kind = .write ∧ a.Hit ctx m₀ (ρ.bindIdx params i) x.1) ∧
      (∀ x ∈ g.wcrs, ∃ a ∈ A, a.kind = .wcr x.2.1 ∧ a.Hit ctx m₀ (ρ.bindIdx params i) x.1) ∧
      g.aliasSets = [] ∧ g.queues = [] := by
  intro g hg
  obtain ⟨i, hi, hw⟩ := IterExec.fp_logs h rfl (scopeFp_incl hfp) hal g hg
  exact ⟨i, hi, hw.reads, hw.writes, hw.wcrs, hw.aliasSets, hw.queues⟩

end Dace
