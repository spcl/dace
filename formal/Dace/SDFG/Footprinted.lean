-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.LogShift

/-!
# Footprinted actions

A discipline for the leaf functions of the SDFG semantics
(`Dace.SDFG.Semantics.Leaf`) that run on a sequential execution state
(`St`): an action is **footprinted** (`FootA`, `Foot`) if every normal run
of it from a state `st`

* changes the state exactly by a *plain* delta `δ` (reads, plain writes and
  WCR contributions only, `Log.Plain`): the new local view is `δ` applied to
  the old one, the new log is the old one followed by `δ`, the code-edge
  values are unchanged (`St.step`);
* depends on the memory only through the alias table and the cells that `δ`
  reads: from any state that agrees with `st` there (`Mem.Agree`) and has
  the same code-edge values, it runs normally with the same result and the
  same delta.

Footprinted actions are closed under `bind` (`FootA.bind`, …) and `foldlM`
(`FootA.foldlM`, `Foot.foldlM`); the primitive accesses `St.read`,
`St.readMany`, `St.write`, `St.wcr`, `St.store` are footprinted
(`St.read_foot`, …).  Two facts make the composition work: a location that
the first delta writes has the same value after it from any start
(`Log.apply_cells_congr`), and the old value a WCR contribution combines
with needs no read, because applying the logged contribution recomputes it
(`Log.apply_wcr`).

The semantics evaluates symbolic expressions against the memory
(`Ctx.eval`, `Ctx.pointsOf`: bare scalar container names and subscripts,
`Decisions.md#D33`) without logging the read.  Expressions free of both
(`Ctx.exprPure`) evaluate independently of the memory (`Ctx.eval_pure`);
subsets and ranges built from them enumerate independently of it
(`Ctx.pointsOf_pure`, `Ctx.rangePoints_pure`).

These lemmas serve the frame and order-independence results of
`Dace.SDFG.Frame` and `Dace.Equiv.Order`, and footprint analyses that need
the same leaf-level reasoning.
-/

set_option autoImplicit false

namespace Dace

/-! ## Expressions that do not read memory -/

namespace Expr

private theorem append_nil_iff {α : Type} {a b : List α} : a ++ b = [] ↔ a = [] ∧ b = [] :=
  List.append_eq_nil_iff

mutual

/-- Evaluation depends on the interpretation only through `app` and
`readAttr` when the expression reads no data, and on the valuation only at
its free symbols. -/
theorem eval_congr_interp {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) :
    (e : Expr) → e.readData = [] → (∀ x ∈ e.freeSyms, ρ x = ρ' x) → e.eval I ρ = e.eval I' ρ'
  | .lit (.int _), _, _ => rfl
  | .lit (.float _ _), _, _ => rfl
  | .lit (.bool _), _, _ => rfl
  | .sym x, _, hs => hs x (by simp [freeSyms])
  | .un op a, hr, hs => by
    simp only [eval]
    rw [eval_congr_interp happ hattr a hr hs]
  | .bin op a b, hr, hs => by
    simp only [readData, append_nil_iff] at hr
    simp only [eval]
    rw [eval_congr_interp happ hattr a hr.1 (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_interp happ hattr b hr.2 (fun x hx => hs x (by simp [freeSyms, hx]))]
  | .ite c t f, hr, hs => by
    simp only [readData, append_nil_iff] at hr
    simp only [eval]
    rw [eval_congr_interp happ hattr c hr.1.1 (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_interp happ hattr t hr.1.2 (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_interp happ hattr f hr.2 (fun x hx => hs x (by simp [freeSyms, hx]))]
  | .app fn args, hr, hs => by
    simp only [eval]
    rw [evalList_congr_interp happ hattr args hr hs, happ]
  | .subscript _ _, hr, _ => by simp [readData] at hr
  | .attr b _, hr, hs => by
    simp only [eval]
    rw [eval_congr_interp happ hattr b hr hs, hattr]
  | .approx a o, hr, hs => by
    simp only [readData, append_nil_iff] at hr
    simp only [eval]
    exact eval_congr_interp happ hattr a hr.1 (fun x hx => hs x (by simp [freeSyms, hx]))

theorem evalList_congr_interp {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) :
    (es : Exprs) → readDataList es = [] → (∀ x ∈ freeSymsList es, ρ x = ρ' x) →
      evalList I ρ es = evalList I' ρ' es
  | .nil, _, _ => rfl
  | .cons e es, hr, hs => by
    simp only [readDataList, append_nil_iff] at hr
    simp only [evalList]
    rw [eval_congr_interp happ hattr e hr.1 (fun x hx => hs x (by simp [freeSymsList, hx])),
      evalList_congr_interp happ hattr es hr.2 (fun x hx => hs x (by simp [freeSymsList, hx]))]

end

end Expr

theorem mapM_loop_congr_mem {m : Type → Type} [Monad m] {α β : Type} {f g : α → m β} :
    ∀ (l : List α) (acc : List β), (∀ x ∈ l, f x = g x) → List.mapM.loop f l acc = List.mapM.loop g l acc
  | [], _, _ => rfl
  | x :: xs, acc, h => by
    simp only [List.mapM.loop, h x List.mem_cons_self]
    congr 1
    funext y
    exact mapM_loop_congr_mem xs _ (fun z hz => h z (List.mem_cons_of_mem _ hz))

/-- `mapM` only depends on the function at the list's members (any monad). -/
theorem mapM_congr_mem {m : Type → Type} [Monad m] {α β : Type} {f g : α → m β} (l : List α)
    (h : ∀ x ∈ l, f x = g x) : l.mapM f = l.mapM g :=
  mapM_loop_congr_mem l [] h

namespace Ctx

/-- Does `x` name a scalar container (read from memory when used bare in an
expression, `Ctx.val`)? -/
def isScalarName (ctx : Ctx) (x : Sym) : Bool := (ctx.desc? x).any (·.isScalar)

/-- An expression whose evaluation reads no memory (`DataflowPure`): no
subscripts and no free symbol naming a scalar container. -/
def exprPure (ctx : Ctx) (e : Expr) : Bool :=
  e.readData.isEmpty && e.freeSyms.all fun x => !ctx.isScalarName x

def dimPure (ctx : Ctx) (d : Dim) : Bool := ctx.exprPure d.lo && ctx.exprPure d.hi && ctx.exprPure d.step

def rangePure (ctx : Ctx) (r : Range) : Bool := r.dims.all ctx.dimPure

def subsetPure (ctx : Ctx) : Subset → Bool
  | .range r => ctx.rangePure r
  | .union rs => rs.all ctx.rangePure

theorem val_of_not_scalar (ctx : Ctx) (ρ : Valuation) (m : Mem) {x : Sym} (hx : ctx.isScalarName x = false) :
    ctx.val ρ m x = ρ x := by
  unfold val
  cases hρ : ρ x with
  | some v => rfl
  | none =>
    simp only
    cases hd : ctx.desc? x with
    | none => rfl
    | some d =>
      simp only [isScalarName, hd, Option.any_some] at hx
      simp [hx]

theorem eval_pure_aux (ctx : Ctx) (ρ : Valuation) (m : Mem) {e : Expr} (h : ctx.exprPure e = true) :
    e.eval (ctx.exprInterp ρ m) (ctx.val ρ m) = e.eval ctx.I.expr ρ := by
  simp only [exprPure, Bool.and_eq_true, List.isEmpty_iff, List.all_eq_true, Bool.not_eq_eq_eq_not,
    Bool.not_true] at h
  exact Expr.eval_congr_interp (I := ctx.exprInterp ρ m) (I' := ctx.I.expr) rfl rfl e h.1
    fun x hx => val_of_not_scalar ctx ρ m (h.2 x hx)

/-- **Evaluation of a pure expression does not read memory.** -/
theorem eval_pure (ctx : Ctx) (ρ : Valuation) (m : Mem) {e : Expr} (h : ctx.exprPure e = true) :
    ctx.eval ρ m e = e.eval ctx.I.expr ρ :=
  eval_pure_aux ctx ρ m h

theorem evalInt_pure (ctx : Ctx) (ρ : Valuation) (m : Mem) {e : Expr} (h : ctx.exprPure e = true) :
    ctx.evalInt ρ m e = e.evalInt ctx.I.expr ρ := by
  simp only [evalInt, Expr.evalInt, eval_pure_aux ctx ρ m h]

theorem evalR_pure (ctx : Ctx) (ρ : Valuation) (m m' : Mem) {e : Expr} (h : ctx.exprPure e = true) :
    ctx.evalR ρ m e = ctx.evalR ρ m' e := by
  simp only [evalR, eval_pure ctx ρ m h, eval_pure ctx ρ m' h]

theorem dim_elems_pure (ctx : Ctx) (ρ : Valuation) (m : Mem) {d : Dim} (h : ctx.dimPure d = true) :
    d.elems (ctx.exprInterp ρ m) (ctx.val ρ m) = d.elems ctx.I.expr ρ := by
  simp only [dimPure, Bool.and_eq_true] at h
  simp only [Dim.elems, Expr.evalInt, eval_pure_aux ctx ρ m h.1.1, eval_pure_aux ctx ρ m h.1.2,
    eval_pure_aux ctx ρ m h.2]

theorem range_points_pure (ctx : Ctx) (ρ : Valuation) (m : Mem) {r : Range} (h : ctx.rangePure r = true) :
    r.points (ctx.exprInterp ρ m) (ctx.val ρ m) = r.points ctx.I.expr ρ := by
  simp only [rangePure, List.all_eq_true] at h
  simp only [Range.points]
  congr 1
  exact mapM_congr_mem _ fun d hd => dim_elems_pure ctx ρ m (h d hd)

theorem subset_points_pure (ctx : Ctx) (ρ : Valuation) (m : Mem) {sub : Subset} (h : ctx.subsetPure sub = true) :
    sub.points (ctx.exprInterp ρ m) (ctx.val ρ m) = sub.points ctx.I.expr ρ := by
  cases sub with
  | range r => exact range_points_pure ctx ρ m h
  | union rs =>
    simp only [subsetPure, List.all_eq_true] at h
    simp only [Subset.points]
    congr 1
    exact mapM_congr_mem _ fun r hr => range_points_pure ctx ρ m (h r hr)

/-- The subset of container `n` that `pointsOf` enumerates is pure (the full
range of its descriptor when the subset is missing). -/
def pointsPure (ctx : Ctx) (n : Name) : Option Subset → Bool
  | some s => ctx.subsetPure s
  | none => (ctx.desc? n).all fun d => ctx.rangePure d.fullRange

theorem pointsOf_pure (ctx : Ctx) (ρ : Valuation) (m m' : Mem) {n : Name} {sub : Option Subset}
    (h : ctx.pointsPure n sub = true) : ctx.pointsOf ρ m n sub = ctx.pointsOf ρ m' n sub := by
  unfold pointsOf
  cases hd : ctx.desc? n with
  | none => rfl
  | some d =>
    have h' : ctx.subsetPure (sub.getD (.range d.fullRange)) = true := by
      cases sub with
      | some s => exact h
      | none => simpa [pointsPure, hd, subsetPure] using h
    simp only [subset_points_pure ctx ρ m h', subset_points_pure ctx ρ m' h']

/-- The lower corner of a pure subset does not depend on the memory. -/
theorem lowerCorner_pure (ctx : Ctx) (ρ : Valuation) (m m' : Mem) {n : Name} {sub : Option Subset}
    (h : ctx.pointsPure n sub = true) : ctx.lowerCorner ρ m n sub = ctx.lowerCorner ρ m' n sub := by
  unfold lowerCorner
  rw [pointsOf_pure ctx ρ m m' h]
  cases sub with
  | none => rfl
  | some s =>
    have hr : ∀ r, (match s with | .range r => some r | .union rs => rs.head?) = some r → ctx.rangePure r = true := by
      intro r hr
      cases s with
      | range r' => cases hr; exact h
      | union rs =>
        simp only [pointsPure, subsetPure, List.all_eq_true] at h
        exact h r (List.mem_of_mem_head? hr)
    have key : ∀ r : Range, ctx.rangePure r = true →
        (r.dims.mapM fun dim => Res.ofOption (ctx.evalInt ρ m dim.lo) (.undefinedSubset s)) =
          (r.dims.mapM fun dim => Res.ofOption (ctx.evalInt ρ m' dim.lo) (.undefinedSubset s)) := by
      intro r hp
      simp only [rangePure, List.all_eq_true] at hp
      refine mapM_congr_mem (m := Res) _ fun dim hdim => ?_
      have hd := hp dim hdim
      simp only [dimPure, Bool.and_eq_true] at hd
      rw [evalInt_pure ctx ρ m hd.1.1, evalInt_pure ctx ρ m' hd.1.1]
    simp only
    split
    · rfl
    · next r hrs => rw [key r (hr r hrs)]

theorem rangePoints_pure (ctx : Ctx) (ρ : Valuation) (m m' : Mem) {r : Range} (h : ctx.rangePure r = true) :
    ctx.rangePoints ρ m r = ctx.rangePoints ρ m' r := by
  simp only [rangePoints, range_points_pure ctx ρ m h, range_points_pure ctx ρ m' h]

end Ctx

/-! ## Locations depend on the alias table only -/

theorem Mem.resolveFuel_congr {m m' : Mem} (h : m.aliases = m'.aliases) :
    ∀ (n : Nat) (l : Loc), m.resolveFuel n l = m'.resolveFuel n l
  | 0, _ => rfl
  | n + 1, l => by
    simp only [Mem.resolveFuel, h]
    cases m'.aliases l.name
    · rfl
    · exact Mem.resolveFuel_congr h n _

theorem Mem.resolve_congr {m m' : Mem} (h : m.aliases = m'.aliases) (l : Loc) : m.resolve l = m'.resolve l :=
  Mem.resolveFuel_congr h 64 l

theorem Ctx.addr_congr (ctx : Ctx) (ρ : Valuation) {m m' : Mem} (h : m.aliases = m'.aliases) (n : Name)
    (idx : Idx) : ctx.addr ρ m n idx = ctx.addr ρ m' n idx := by
  unfold Ctx.addr Mem.isAliased
  rw [h]

theorem Ctx.loc_congr (ctx : Ctx) (ρ : Valuation) {m m' : Mem} (h : m.aliases = m'.aliases) (n : Name)
    (idx : Idx) (oob : Bool) : ctx.loc ρ m n idx oob = ctx.loc ρ m' n idx oob := by
  simp only [Ctx.loc, ctx.addr_congr ρ h]

/-! ## Applying a log, cell by cell -/

/-- A log of plain memory accesses: reads, plain writes, WCR contributions;
no alias bindings and no queue effects. -/
structure Log.Plain (g : Log) : Prop where
  aliasSets : g.aliasSets = []
  queues : g.queues = []

namespace Log

open Merge

theorem Plain.empty : Log.empty.Plain := ⟨rfl, rfl⟩

theorem Plain.seq {a b : Log} (ha : a.Plain) (hb : b.Plain) : (a.seq b).Plain :=
  ⟨by simp [Log.seq, ha.aliasSets, hb.aliasSets], by simp [Log.seq, ha.queues, hb.queues]⟩

theorem Plain.unionAll {gs : List Log} (h : ∀ g ∈ gs, g.Plain) : (Log.unionAll gs).Plain :=
  ⟨by simp only [Log.unionAll, List.flatMap_eq_nil_iff]; exact fun g hg => (h g hg).aliasSets,
   by simp only [Log.unionAll, List.flatMap_eq_nil_iff]; exact fun g hg => (h g hg).queues⟩

theorem Plain.of_le {g' g : Log} (hle : Le g' g) (h : g.Plain) : g'.Plain :=
  ⟨List.eq_nil_iff_forall_not_mem.2 fun x hx => by
      have := (hle.aliasSets.mem_iff x).1 hx; rw [h.aliasSets] at this; simp at this,
   List.eq_nil_iff_forall_not_mem.2 fun x hx => by
      have := (hle.queues.mem_iff x).1 hx; rw [h.queues] at this; simp at this⟩

theorem apply_aliases (g : Log) (w : WcrSem) (m : Mem) :
    (g.apply w m).aliases = aliasTable m.aliases g.aliasSets := by
  rw [apply_eq]

theorem Plain.apply_aliases {g : Log} (h : g.Plain) (w : WcrSem) (m : Mem) : (g.apply w m).aliases = m.aliases := by
  rw [Log.apply_aliases, h.aliasSets]
  rfl

theorem Plain.apply_queues {g : Log} (h : g.Plain) (w : WcrSem) (m : Mem) : (g.apply w m).queues = m.queues := by
  rw [apply_eq, h.queues]
  rfl

theorem apply_cells (g : Log) (w : WcrSem) (m : Mem) (l : Loc) :
    (g.apply w m).cells l = wcrFold w (writeCells m.cells g.writes l) (g.wcrs.filter (fun x => x.1 = l)) := by
  rw [apply_eq]
  exact wcrCells_eval ..

theorem writeCells_of_not_mem {c : Loc → Val} {ws : List (Loc × Val)} {l : Loc} (h : l ∉ ws.map Prod.fst) :
    writeCells c ws l = c l := by
  rw [writeCells_eval]
  have : ws.filter (fun x => x.1 = l) = [] := by
    rw [List.filter_eq_nil_iff]
    intro x hx hxl
    simp only [decide_eq_true_eq] at hxl
    exact h (hxl ▸ List.mem_map.2 ⟨x, hx, rfl⟩)
  rw [this]
  rfl

theorem writeCells_of_mem (c c' : Loc → Val) {ws : List (Loc × Val)} {l : Loc} (h : l ∈ ws.map Prod.fst) :
    writeCells c ws l = writeCells c' ws l := by
  rw [writeCells_eval, writeCells_eval]
  exact ListLemmas.foldl_last_indep ws l (by simpa using h) _ _

/-- The value of a cell after applying a log depends on its old value only
if the log does not plainly write it. -/
theorem apply_cells_congr (g : Log) (w : WcrSem) {m m' : Mem} {l : Loc}
    (h : m.cells l = m'.cells l ∨ l ∈ g.writeLocs) : (g.apply w m).cells l = (g.apply w m').cells l := by
  rw [apply_cells, apply_cells]
  congr 1
  rcases h with h | h
  · by_cases hw : l ∈ g.writeLocs
    · exact writeCells_of_mem _ _ hw
    · rw [writeCells_of_not_mem hw, writeCells_of_not_mem hw, h]
  · exact writeCells_of_mem _ _ h

/-- A cell that the log neither writes nor updates keeps its value. -/
theorem apply_cells_untouched (g : Log) (w : WcrSem) (m : Mem) {l : Loc} (hw : l ∉ g.writeLocs)
    (hc : l ∉ g.wcrLocs) : (g.apply w m).cells l = m.cells l := by
  rw [apply_cells, writeCells_of_not_mem hw]
  have : g.wcrs.filter (fun x => x.1 = l) = [] := by
    rw [List.filter_eq_nil_iff]
    intro x hx hxl
    simp only [decide_eq_true_eq] at hxl
    exact hc (hxl ▸ List.mem_map.2 ⟨x, hx, rfl⟩)
  rw [this]
  rfl

theorem mem_seq_reads {a b : Log} {l : Loc} :
    l ∈ (a.seq b).reads ↔ l ∈ a.reads ∨ (l ∈ b.reads ∧ l ∉ a.writeLocs) := by
  simp [Log.seq, writeLocs]

end Log

/-! ## Agreement of memories on a read set -/

/-- `m₂` agrees with `m` on the alias table and on the cells `R`. -/
def Mem.Agree (R : List Loc) (m₂ m : Mem) : Prop := m₂.aliases = m.aliases ∧ ∀ l ∈ R, m₂.cells l = m.cells l

namespace Mem.Agree

theorem refl (R : List Loc) (m : Mem) : Mem.Agree R m m := ⟨rfl, fun _ _ => rfl⟩

theorem symm {R : List Loc} {m₂ m : Mem} (h : Mem.Agree R m₂ m) : Mem.Agree R m m₂ :=
  ⟨h.1.symm, fun l hl => (h.2 l hl).symm⟩

theorem trans {R : List Loc} {m₁ m₂ m₃ : Mem} (h₁ : Mem.Agree R m₁ m₂) (h₂ : Mem.Agree R m₂ m₃) :
    Mem.Agree R m₁ m₃ :=
  ⟨h₁.1.trans h₂.1, fun l hl => (h₁.2 l hl).trans (h₂.2 l hl)⟩

theorem mono {R R' : List Loc} {m₂ m : Mem} (h : Mem.Agree R m₂ m) (hR : ∀ l ∈ R', l ∈ R) : Mem.Agree R' m₂ m :=
  ⟨h.1, fun l hl => h.2 l (hR l hl)⟩

theorem seq_left {a b : Log} {m₂ m : Mem} (h : Mem.Agree (a.seq b).reads m₂ m) : Mem.Agree a.reads m₂ m :=
  h.mono fun _ hl => Log.mem_seq_reads.2 (.inl hl)

/-- Agreement on the reads of `a.seq b` gives agreement on the reads of `b`
after `a`. -/
theorem seq_right {a b : Log} (w : WcrSem) {m₂ m : Mem} (h : Mem.Agree (a.seq b).reads m₂ m) :
    Mem.Agree b.reads (a.apply w m₂) (a.apply w m) := by
  refine ⟨by rw [Log.apply_aliases, Log.apply_aliases, h.1], fun l hl => ?_⟩
  refine Log.apply_cells_congr a w ?_
  by_cases hw : l ∈ a.writeLocs
  · exact .inr hw
  · exact .inl (h.2 l (Log.mem_seq_reads.2 (.inr ⟨hl, hw⟩)))

/-- Applying the same log keeps agreement on any set. -/
theorem apply {R : List Loc} {m₂ m : Mem} (h : Mem.Agree R m₂ m) (g : Log) (w : WcrSem) :
    Mem.Agree R (g.apply w m₂) (g.apply w m) :=
  ⟨by rw [Log.apply_aliases, Log.apply_aliases, h.1], fun l hl => Log.apply_cells_congr g w (.inl (h.2 l hl))⟩

end Mem.Agree

/-! ## Footprinted actions -/

/-- The state after a plain delta `δ`: the local view with `δ` applied, the
log followed by `δ`, the same code-edge values. -/
def St.step (w : WcrSem) (δ : Log) (st : St) : St :=
  { mem := δ.apply w st.mem, log := st.log.seq δ, vals := st.vals }

namespace St

@[simp] theorem step_mem (w : WcrSem) (δ : Log) (st : St) : (st.step w δ).mem = δ.apply w st.mem := rfl
@[simp] theorem step_log (w : WcrSem) (δ : Log) (st : St) : (st.step w δ).log = st.log.seq δ := rfl
@[simp] theorem step_vals (w : WcrSem) (δ : Log) (st : St) : (st.step w δ).vals = st.vals := rfl

theorem step_empty (w : WcrSem) (st : St) : st.step w Log.empty = st := by
  cases st
  simp [step, Log.apply_empty, Log.seq_empty]

theorem step_step (w : WcrSem) (a b : Log) (st : St) : (st.step w a).step w b = st.step w (a.seq b) := by
  simp [step, Log.apply_seq, Log.seq_assoc]

end St

/-- A **footprinted action** with a result: every normal run from `st`
changes the state by a plain delta `δ` (`St.step`), and from every state
that agrees with `st` on the alias table, the cells `δ` reads and the
code-edge values, the action runs normally with the same result and the
same delta. -/
def FootA (w : WcrSem) {α : Type} (f : St → Res (α × St)) : Prop :=
  ∀ st a st', f st = .ok (a, st') → ∃ δ : Log, δ.Plain ∧ st' = st.step w δ ∧
    ∀ st₂ : St, Mem.Agree δ.reads st₂.mem st.mem → st₂.vals = st.vals → f st₂ = .ok (a, st₂.step w δ)

/-- A footprinted action without a result (`FootA` for `St → Res St`). -/
def Foot (w : WcrSem) (f : St → Res St) : Prop :=
  ∀ st st', f st = .ok st' → ∃ δ : Log, δ.Plain ∧ st' = st.step w δ ∧
    ∀ st₂ : St, Mem.Agree δ.reads st₂.mem st.mem → st₂.vals = st.vals → f st₂ = .ok (st₂.step w δ)

namespace FootA

variable {w : WcrSem} {α β γ : Type}

theorem congr {f g : St → Res (α × St)} (h : FootA w f) (he : ∀ st, g st = f st) : FootA w g := by
  have : g = f := funext he
  rw [this]; exact h

theorem pure (a : α) : FootA w (fun st => .ok (a, st)) := by
  intro st a' st' h
  simp only [Res.ok_inj, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨Log.empty, Log.Plain.empty, (St.step_empty w st).symm, fun st₂ _ _ => by rw [St.step_empty]⟩

theorem err (e : ErrReason) : FootA w (fun _ => (.err e : Res (α × St))) := by
  intro st a st' h; cases h

theorem bind {f : St → Res (α × St)} {g : α × St → Res (β × St)} (hf : FootA w f)
    (hg : ∀ a, FootA w (fun st => g (a, st))) : FootA w (fun st => f st >>= g) := by
  intro st b st'' h
  dsimp only at h ⊢
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok p =>
    obtain ⟨a, st'⟩ := p
    rw [hfs] at h
    change g (a, st') = .ok (b, st'') at h
    obtain ⟨δ₁, hp₁, rfl, hf₂⟩ := hf st a st' hfs
    obtain ⟨δ₂, hp₂, rfl, hg₂⟩ := hg a _ b _ h
    refine ⟨δ₁.seq δ₂, hp₁.seq hp₂, St.step_step w _ _ _, fun st₂ hag hv => ?_⟩
    rw [hf₂ st₂ hag.seq_left hv]
    change g (a, st₂.step w δ₁) = _
    rw [← St.step_step]
    exact hg₂ (st₂.step w δ₁) (hag.seq_right w) hv

/-- A head that only depends on the alias table. -/
theorem bind_det {h : St → Res γ} {g : γ → St → Res (β × St)}
    (hh : ∀ st st₂ : St, st₂.mem.aliases = st.mem.aliases → h st₂ = h st) (hg : ∀ c, FootA w (g c)) :
    FootA w (fun st => h st >>= fun c => g c st) := by
  intro st b st' hr
  dsimp only at hr ⊢
  cases hhs : h st with
  | err e => rw [hhs] at hr; cases hr
  | ok c =>
    rw [hhs] at hr
    obtain ⟨δ, hp, rfl, hg₂⟩ := hg c st b _ hr
    refine ⟨δ, hp, rfl, fun st₂ hag hv => ?_⟩
    rw [hh st st₂ hag.1, hhs]
    exact hg₂ st₂ hag hv

theorem foldlM {f : α × St → β → Res (α × St)} :
    ∀ (xs : List β), (∀ b ∈ xs, ∀ a, FootA w (fun st => f (a, st) b)) →
      ∀ (a₀ : α), FootA w (fun st => xs.foldlM f (a₀, st))
  | [], _, a₀ => by
    simpa using (pure a₀ : FootA w (fun st => .ok (a₀, st)))
  | x :: xs, hf, a₀ => by
    simp only [List.foldlM_cons]
    exact bind (hf x List.mem_cons_self a₀)
      (fun a => foldlM xs (fun b hb => hf b (List.mem_cons_of_mem _ hb)) a)

end FootA

namespace Foot

variable {w : WcrSem} {α β γ : Type}

theorem congr {f g : St → Res St} (h : Foot w f) (he : ∀ st, g st = f st) : Foot w g := by
  have : g = f := funext he
  rw [this]; exact h

theorem pure : Foot w (fun st => .ok st) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  exact ⟨Log.empty, Log.Plain.empty, (St.step_empty w st).symm, fun st₂ _ _ => by rw [St.step_empty]⟩

theorem err (e : ErrReason) : Foot w (fun _ => .err e) := by
  intro st st' h; cases h

/-- A footprinted action with a result, followed by one without. -/
theorem bindA {f : St → Res (α × St)} {g : α × St → Res St} (hf : FootA w f)
    (hg : ∀ a, Foot w (fun st => g (a, st))) : Foot w (fun st => f st >>= g) := by
  intro st st'' h
  dsimp only at h ⊢
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok p =>
    obtain ⟨a, st'⟩ := p
    rw [hfs] at h
    change g (a, st') = .ok st'' at h
    obtain ⟨δ₁, hp₁, rfl, hf₂⟩ := hf st a st' hfs
    obtain ⟨δ₂, hp₂, rfl, hg₂⟩ := hg a _ _ h
    refine ⟨δ₁.seq δ₂, hp₁.seq hp₂, St.step_step w _ _ _, fun st₂ hag hv => ?_⟩
    rw [hf₂ st₂ hag.seq_left hv]
    change g (a, st₂.step w δ₁) = _
    rw [← St.step_step]
    exact hg₂ (st₂.step w δ₁) (hag.seq_right w) hv

theorem bind {f g : St → Res St} (hf : Foot w f) (hg : Foot w g) : Foot w (fun st => f st >>= g) := by
  intro st st'' h
  dsimp only at h ⊢
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok st' =>
    rw [hfs] at h
    change g st' = .ok st'' at h
    obtain ⟨δ₁, hp₁, rfl, hf₂⟩ := hf st st' hfs
    obtain ⟨δ₂, hp₂, rfl, hg₂⟩ := hg _ _ h
    refine ⟨δ₁.seq δ₂, hp₁.seq hp₂, St.step_step w _ _ _, fun st₂ hag hv => ?_⟩
    rw [hf₂ st₂ hag.seq_left hv]
    change g (st₂.step w δ₁) = _
    rw [hg₂ (st₂.step w δ₁) (hag.seq_right w) hv, St.step_step]

/-- A head that only depends on the alias table. -/
theorem bind_det {h : St → Res γ} {g : γ → St → Res St}
    (hh : ∀ st st₂ : St, st₂.mem.aliases = st.mem.aliases → h st₂ = h st) (hg : ∀ c, Foot w (g c)) :
    Foot w (fun st => h st >>= fun c => g c st) := by
  intro st st' hr
  dsimp only at hr ⊢
  cases hhs : h st with
  | err e => rw [hhs] at hr; cases hr
  | ok c =>
    rw [hhs] at hr
    obtain ⟨δ, hp, rfl, hg₂⟩ := hg c st _ hr
    refine ⟨δ, hp, rfl, fun st₂ hag hv => ?_⟩
    rw [hh st st₂ hag.1, hhs]
    exact hg₂ st₂ hag hv

theorem foldlM {f : St → β → Res St} :
    ∀ (xs : List β), (∀ b ∈ xs, Foot w (fun st => f st b)) → Foot w (fun st => xs.foldlM f st)
  | [], _ => by simpa using (pure : Foot w (fun st => .ok st))
  | x :: xs, hf => by
    simp only [List.foldlM_cons]
    exact bind (hf x List.mem_cons_self) (foldlM xs fun b hb => hf b (List.mem_cons_of_mem _ hb))

/-- An action that only reads the code-edge values. -/
theorem _root_.Dace.FootA.ofVals {α : Type} (f : List ((NodeId × Conn) × Val) → α) :
    FootA w (fun st => .ok (f st.vals, st)) := by
  intro st a st' h
  simp only [Res.ok_inj, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨Log.empty, Log.Plain.empty, (St.step_empty w st).symm, fun st₂ _ hv => by
    rw [St.step_empty]; dsimp only; rw [hv]⟩

/-- Forget the result of a footprinted action. -/
theorem ofA {f : St → Res (α × St)} (hf : FootA w f) : Foot w (fun st => f st >>= fun p => .ok p.2) :=
  bindA hf fun _ => pure

end Foot

/-! ## Primitive accesses -/

namespace St

variable (ctx : Ctx) (ρ : Valuation)

theorem readAt_foot (w : WcrSem) (l : Loc) :
    FootA w (fun st => .ok (st.mem.read l, st.logged { reads := [st.mem.resolve l] })) := by
  intro st a st' h
  simp only [Res.ok_inj, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  refine ⟨{ reads := [st.mem.resolve l] }, ⟨rfl, rfl⟩, rfl, fun st₂ hag hv => ?_⟩
  have hr : st₂.mem.resolve l = st.mem.resolve l := Mem.resolve_congr hag.1 l
  simp only [Mem.read, hr, hag.2 _ List.mem_cons_self, Res.ok_inj, Prod.mk.injEq, true_and]
  rfl

theorem read_foot (w : WcrSem) (n : Name) (idx : Idx) (oob : Bool) :
    FootA w (fun st => st.read ctx ρ n idx oob) :=
  FootA.bind_det (h := fun st => ctx.loc ρ st.mem n idx oob) (fun _ _ ha => ctx.loc_congr ρ ha _ _ _)
    (fun l => readAt_foot w l)

theorem writeAt_foot (w : WcrSem) (l : Loc) (v : Val) :
    Foot w (fun st => .ok
      { st with mem := st.mem.write l v, log := st.log.seq { writes := [(st.mem.resolve l, v)] } }) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  refine ⟨{ writes := [(st.mem.resolve l, v)] }, ⟨rfl, rfl⟩, ?_, fun st₂ hag hv => ?_⟩
  · simp only [step, Log.apply_write]
  · have hr : st₂.mem.resolve l = st.mem.resolve l := Mem.resolve_congr hag.1 l
    simp only [step, Res.ok_inj]
    rw [← hr, Log.apply_write]

theorem write_foot (w : WcrSem) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    Foot w (fun st => st.write ctx ρ n idx oob v) :=
  Foot.bind_det (h := fun st => ctx.loc ρ st.mem n idx oob) (fun _ _ ha => ctx.loc_congr ρ ha _ _ _)
    (fun l => writeAt_foot w l v)

theorem wcrAt_foot (l : Loc) (op : WcrOp) (v : Val) :
    Foot ctx.I.wcr (fun st => .ok
      { st with mem := st.mem.write l (ctx.I.wcr op (st.mem.read l) v),
                log := st.log.seq { wcrs := [(st.mem.resolve l, op, v)] } }) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  refine ⟨{ wcrs := [(st.mem.resolve l, op, v)] }, ⟨rfl, rfl⟩, ?_, fun st₂ hag hv => ?_⟩
  · simp only [step, Log.apply_wcr]
  · have hr : st₂.mem.resolve l = st.mem.resolve l := Mem.resolve_congr hag.1 l
    simp only [step, Res.ok_inj]
    rw [← hr, Log.apply_wcr]

theorem wcr_foot (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    Foot ctx.I.wcr (fun st => st.wcr ctx ρ n idx oob op v) :=
  Foot.bind_det (h := fun st => ctx.loc ρ st.mem n idx oob) (fun _ _ ha => ctx.loc_congr ρ ha _ _ _)
    (fun l => wcrAt_foot ctx l op v)

theorem store_foot (n : Name) (idx : Idx) (oob : Bool) (wcrOp : Option WcrOp) (v : Val) :
    Foot ctx.I.wcr (fun st => st.store ctx ρ n idx oob wcrOp v) := by
  cases wcrOp with
  | none => exact write_foot ctx ρ _ n idx oob v
  | some op => exact wcr_foot ctx ρ n idx oob op v

theorem readMany_foot (w : WcrSem) (n : Name) (pts : List Idx) (oob : Bool) :
    FootA w (fun st => st.readMany ctx ρ n pts oob) := by
  unfold readMany
  refine FootA.foldlM pts (fun p _ acc => ?_) []
  exact FootA.bind (read_foot ctx ρ w n p oob) (fun v => FootA.pure _)

theorem logged_reads_logged (st : St) (a b : List Loc) :
    (st.logged { reads := a }).logged { reads := b } = st.logged { reads := a ++ b } := by
  simp only [logged, Log.seq_assoc]
  congr 2
  simp [Log.seq]

theorem logged_reads_nil (st : St) : st.logged { reads := [] } = st := by
  cases st
  simp only [logged]
  congr
  exact Log.seq_empty _

/-- A normal `readMany` reads the elements at the locations of its points,
logging exactly those reads. -/
theorem readMany_ok {n : Name} {oob : Bool} :
    ∀ (pts : List Idx) (acc : List Val) (st : St) (vs : List Val) (st' : St),
      pts.foldlM (fun (acc, st) p => do
        let (v, st) ← st.read ctx ρ n p oob
        pure (acc ++ [v], st)) (acc, st) = .ok (vs, st') →
      ∃ ls : List Loc, List.Forall₂ (fun p l => ctx.loc ρ st.mem n p oob = .ok l) pts ls ∧
        vs = acc ++ ls.map st.mem.read ∧ st' = st.logged { reads := ls.map st.mem.resolve }
  | [], acc, st, vs, st', h => by
    simp only [List.foldlM_nil, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨[], .nil, by simp, by simp [logged_reads_nil]⟩
  | p :: ps, acc, st, vs, st', h => by
    simp only [List.foldlM_cons] at h
    cases hl : ctx.loc ρ st.mem n p oob with
    | err e =>
      simp only [read, hl] at h
      cases h
    | ok l =>
      simp only [read, hl] at h
      obtain ⟨ls, hf, hv, hs⟩ := readMany_ok ps _ _ vs st' h
      refine ⟨l :: ls, .cons hl hf, by rw [hv]; simp [logged], ?_⟩
      rw [hs, logged_reads_logged]
      rfl

/-- Conversely, `readMany` succeeds when all its points have locations. -/
theorem readMany_of_forall₂ {n : Name} {oob : Bool} :
    ∀ (pts : List Idx) (ls : List Loc) (acc : List Val) (st : St),
      List.Forall₂ (fun p l => ctx.loc ρ st.mem n p oob = .ok l) pts ls →
      pts.foldlM (fun (acc, st) p => do
        let (v, st) ← st.read ctx ρ n p oob
        pure (acc ++ [v], st)) (acc, st) =
        .ok (acc ++ ls.map st.mem.read, st.logged { reads := ls.map st.mem.resolve })
  | [], [], acc, st, .nil => by simp [logged_reads_nil]
  | p :: ps, l :: ls, acc, st, .cons hl hf => by
    have e := readMany_of_forall₂ ps ls (acc ++ [st.mem.read l]) (st.logged { reads := [st.mem.resolve l] }) hf
    rw [logged_reads_logged] at e
    simp only [List.foldlM_cons, read, hl]
    refine Eq.trans ?_ (e.trans ?_)
    · rfl
    · simp [logged]

end St

end Dace
