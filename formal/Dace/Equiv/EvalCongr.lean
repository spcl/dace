-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Footprinted
import Dace.SDFG.LogShift
import Dace.Core.SymSubset

/-!
# Evaluation depends on free and descriptor symbols only

`Ctx.eval` evaluates an expression under a valuation and reads data through
`Ctx.exprInterp` and `Ctx.val`, whose locations depend on the valuation
through the descriptors' shapes, strides and start offsets.  Two valuations
agreeing on the free symbols of the expression and on every descriptor
symbol (`SDFG.descSyms`) therefore evaluate alike (`Ctx.eval_congr`); the
same holds for subsets (`Ctx.pointsOf_congr`) and scalar reads
(`readScalarData_congr`).  Used to sequentialize interstate assignments and
dynamic map inputs in the pair checker.
-/

set_option autoImplicit false

namespace Dace

theorem Valuation.setAll_of_not_mem {ρ : Valuation} {vs : List (Sym × Val)} {y : Sym} (h : y ∉ vs.map Prod.fst) :
    Valuation.setAll ρ vs y = ρ y := by
  simp only [Valuation.setAll]
  have : vs.find? (fun x => decide (x.1 = y)) = none := by
    rw [List.find?_eq_none]
    intro x hx
    simp only [decide_eq_true_eq]
    exact fun hxy => h (hxy ▸ List.mem_map_of_mem hx)
  rw [this]

/-! ## Evaluation depends on free and descriptor symbols only -/

/-- The symbols a descriptor's addressing reads: shape, effective strides
and start offset. -/
def DataDesc.syms (d : DataDesc) : List Sym :=
  (d.shape ++ d.effStrides ++ [d.startOffset]).flatMap Expr.freeSyms

/-- The symbols the descriptors of an SDFG read. -/
def SDFG.descSyms (g : SDFG) : List Sym := g.arrays.flatMap fun p => p.2.syms

theorem List.mapM_evalInt_congr {I : ExprInterp} {ρ ρ' : Valuation} :
    ∀ (es : List Expr), (∀ e ∈ es, ∀ x ∈ e.freeSyms, ρ x = ρ' x) →
      es.mapM (Expr.evalInt I ρ) = es.mapM (Expr.evalInt I ρ')
  | [], _ => rfl
  | e :: es, h => by
    simp only [List.mapM_cons]
    have h₁ : Expr.evalInt I ρ e = Expr.evalInt I ρ' e := by
      unfold Expr.evalInt
      rw [Expr.eval_congr e (h e List.mem_cons_self)]
    rw [h₁, List.mapM_evalInt_congr es (fun e' he' => h e' (List.mem_cons_of_mem _ he'))]

namespace Ctx

section EvalCongr

variable (ctx : Ctx) {ρ ρ' : Valuation} (hd : ∀ x ∈ ctx.g.descSyms, ρ x = ρ' x)
include hd

theorem desc_syms {n : Name} {d : DataDesc} (hn : ctx.desc? n = some d) : ∀ x ∈ d.syms, ρ x = ρ' x := by
  intro x hx
  simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hn
  obtain ⟨p, hp, rfl⟩ := hn
  exact hd x (List.mem_flatMap.2 ⟨p, List.mem_of_find?_eq_some hp, hx⟩)

theorem addr_congr_desc (m : Mem) (n : Name) (idx : Idx) : ctx.addr ρ m n idx = ctx.addr ρ' m n idx := by
  unfold Ctx.addr
  cases hn : ctx.desc? n with
  | none => rfl
  | some d =>
    have hs := ctx.desc_syms hd hn
    have hstart : d.startOffset.evalInt ctx.I.expr ρ = d.startOffset.evalInt ctx.I.expr ρ' := by
      unfold Expr.evalInt
      rw [Expr.eval_congr d.startOffset fun x hx =>
        hs x (List.mem_flatMap.2 ⟨d.startOffset, by simp, hx⟩)]
    have hstr : d.effStrides.mapM (Expr.evalInt ctx.I.expr ρ) = d.effStrides.mapM (Expr.evalInt ctx.I.expr ρ') :=
      List.mapM_evalInt_congr _ fun e he x hx => hs x (List.mem_flatMap.2 ⟨e, by simp [he], hx⟩)
    have hoff : ctx.elemOffset ρ d idx = ctx.elemOffset ρ' d idx := by
      unfold Ctx.elemOffset DataDesc.linearOffset
      rw [hstart, hstr]
    simp only [hoff]

theorem checkBounds_congr (n : Name) (idx : Idx) (oob : Bool) :
    ctx.checkBounds ρ n idx oob = ctx.checkBounds ρ' n idx oob := by
  unfold Ctx.checkBounds
  cases hn : ctx.desc? n with
  | none => rfl
  | some d =>
    have hs := ctx.desc_syms hd hn
    have hsh : d.shape.mapM (fun s => Res.ofOption (s.evalInt ctx.I.expr ρ) (.undefinedExpr s)) =
        d.shape.mapM (fun s => Res.ofOption (s.evalInt ctx.I.expr ρ') (.undefinedExpr s)) := by
      refine mapM_congr_mem (m := Res) _ fun s hsm => ?_
      unfold Expr.evalInt
      rw [Expr.eval_congr s fun x hx => hs x (List.mem_flatMap.2 ⟨s, by simp [hsm], hx⟩)]
    simp only [hsh]

theorem loc_congr_desc (m : Mem) (n : Name) (idx : Idx) (oob : Bool) :
    ctx.loc ρ m n idx oob = ctx.loc ρ' m n idx oob := by
  unfold Ctx.loc
  rw [ctx.addr_congr_desc hd, ctx.checkBounds_congr hd]

theorem exprInterp_congr (m : Mem) : ctx.exprInterp ρ m = ctx.exprInterp ρ' m := by
  simp only [Ctx.exprInterp, ctx.loc_congr_desc hd]

/-- **Evaluation congruence**: an expression evaluates alike under two
valuations agreeing on its free symbols and on the descriptor symbols. -/
theorem eval_congr (m : Mem) (e : Expr) (he : ∀ x ∈ e.freeSyms, ρ x = ρ' x) :
    ctx.eval ρ m e = ctx.eval ρ' m e := by
  unfold Ctx.eval
  rw [ctx.exprInterp_congr hd]
  apply Expr.eval_congr
  intro x hx
  unfold Ctx.val
  rw [he x hx]
  cases ρ' x with
  | some v => rfl
  | none =>
    cases ctx.desc? x with
    | none => rfl
    | some d => simp only [ctx.loc_congr_desc hd]

theorem evalR_congr (m : Mem) (e : Expr) (he : ∀ x ∈ e.freeSyms, ρ x = ρ' x) :
    ctx.evalR ρ m e = ctx.evalR ρ' m e := by
  unfold Ctx.evalR; rw [ctx.eval_congr hd m e he]

end EvalCongr

end Ctx


theorem ofShape_freeSyms {shape : List Expr} {x : Sym} (hx : x ∈ (Range.ofShape shape).freeSyms) :
    ∃ e ∈ shape, x ∈ e.freeSyms := by
  simp only [Range.freeSyms, Range.ofShape, List.flatMap_map, List.mem_flatMap] at hx
  obtain ⟨e, he, hxe⟩ := hx
  refine ⟨e, he, ?_⟩
  have h0 : (0 : Expr).freeSyms = [] := rfl
  have h1 : (1 : Expr).freeSyms = [] := rfl
  have hs : (e - 1).freeSyms = e.freeSyms ++ [] := rfl
  simpa [Dim.freeSyms, h0, h1, hs] using hxe

theorem Subset.points_congr {I : ExprInterp} {ρ ρ' : Valuation} {S : Subset} (h : ∀ x ∈ S.freeSyms, ρ x = ρ' x) :
    S.points I ρ = S.points I ρ' := by
  cases S with
  | range r => exact Range.points_congr h
  | union rs =>
    simp only [Subset.points]
    rw [List.mapM_option_congr (l := rs) (g := Range.points I ρ') (fun r hr =>
      Range.points_congr (fun x hx => h x (List.mem_flatMap.2 ⟨r, hr, hx⟩)))]

namespace Ctx

section PointsCongr

variable (ctx : Ctx) {ρ ρ' : Valuation} (hd : ∀ x ∈ ctx.g.descSyms, ρ x = ρ' x)
include hd

/-- Enumeration of a subset of a container: valuations agreeing on the
descriptor symbols and the subset's free symbols. -/
theorem pointsOf_congr (m : Mem) (n : Name) (sub : Option Subset)
    (hs : ∀ x ∈ (sub.map Subset.freeSyms).getD [], ρ x = ρ' x) : ctx.pointsOf ρ m n sub = ctx.pointsOf ρ' m n sub := by
  unfold Ctx.pointsOf
  cases hn : ctx.desc? n with
  | none => rfl
  | some d =>
    have hsyms := ctx.desc_syms hd hn
    have hsub : ∀ x ∈ (sub.getD (.range d.fullRange)).freeSyms, ctx.val ρ m x = ctx.val ρ' m x := by
      intro x hx
      have hρ : ρ x = ρ' x := by
        cases sub with
        | some S => exact hs x hx
        | none =>
          simp only [Option.getD_none, Subset.freeSyms, DataDesc.fullRange] at hx
          obtain ⟨e, he, hxe⟩ := ofShape_freeSyms hx
          exact hsyms x (List.mem_flatMap.2 ⟨e, List.mem_append_left _ (List.mem_append_left _ he), hxe⟩)
      unfold Ctx.val
      rw [hρ]
      cases ρ' x with
      | some v => rfl
      | none =>
        cases ctx.desc? x with
        | none => rfl
        | some d => simp only [ctx.loc_congr_desc hd]
    simp only []
    rw [ctx.exprInterp_congr hd, Subset.points_congr hsub]

theorem read_congr (st : St) (n : Name) (idx : Idx) (oob : Bool) :
    st.read ctx ρ n idx oob = st.read ctx ρ' n idx oob := by
  unfold St.read; rw [ctx.loc_congr_desc hd]

theorem popN_congr (st : St) (n : Name) (idx : Idx) (k : Nat) :
    st.popN ctx ρ n idx k = st.popN ctx ρ' n idx k := by
  unfold St.popN; rw [ctx.loc_congr_desc hd]

end PointsCongr

end Ctx

/-- **Scalar reads** under valuations agreeing on the descriptor symbols and
the subset's free symbols. -/
theorem readScalarData_congr {ctx : Ctx} {ρ ρ' : Valuation} (hd : ∀ x ∈ ctx.g.descSyms, ρ x = ρ' x)
    (d : Name) (sub : Option Subset) (oob : Bool) (st : St)
    (hs : ∀ x ∈ (sub.map Subset.freeSyms).getD [], ρ x = ρ' x) :
    readScalarData ctx ρ d sub oob st = readScalarData ctx ρ' d sub oob st := by
  unfold readScalarData
  rw [ctx.pointsOf_congr hd st.mem d sub hs]
  simp only [ctx.read_congr hd, ctx.popN_congr hd]

end Dace
