-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrLeaf
import Dace.SDFG.Footprint
import Dace.SDFG.Sim
import Dace.STree.Rename
import Dace.STree.LogShift
import Dace.STree.Retarget
import Dace.SDFG.Coherence

/-!
# SDFG and schedule tree: matching of a dataflow segment (M5a)

`SegCorr ctx tc s o cs`: the tree sibling list `cs` is the image of the
SDFG nodes `o` of state `s`, in that order: an access or exit node becomes
its copies (one `CopyNode` per copy edge into it, in edge order, possibly
none), a tasklet or library node the tree node with the same keys
(`Dace.Equiv.CorrLeaf`), a map entry a `MapScope` whose body is the image
of the scope's direct children in a topological order of the scope, a
consume entry without dynamic inputs its `IN_stream` copy followed by a
`ConsumeScope` whose body is matched in the consuming context
(`TCtx.consuming`), nested SDFGs their inlined regions.

**Matching** (`SegCorr.matched`; a simulation relation in the sense of
refinement proofs, see the terminology note in `Dace.Equiv.Frag`): every run of the tree list (followed by any
`rest`) is matched by the sequential SDFG run of `o` from a state with the
same memory and log: the SDFG run is undefined behavior, or fails when the
tree fails (with an error of its own: error messages name containers, which a
nested SDFG may rename), or ends normally in a state with the same memory and log while
the tree continues with `rest`.  The proof is by induction on `SegCorr`;
leaves are equal leaf functions (`execCode_eq`, …) up to the log prefix
(`*_shift`), a map's iterations and a consume scope's tasks are matched one
by one (`iter_matched`, `tasks_matched`), and an iteration or task whose
SDFG trace has a happens-before race is undefined behavior
(`OrderedExec.hbRace`).  Leaf steps keep the (empty) code-edge values
(`*_steps`, `Dace.SDFG.Coherence`).
-/

set_option autoImplicit false

namespace Dace

namespace Corr

/-! ## Code-edge values are untouched by container operations -/

/-- `f` keeps the code-edge values of a state. -/
def KeepsVals {α : Type} (f : St → Res (α × St)) : Prop := ∀ st a st', f st = .ok (a, st') → st'.vals = st.vals

theorem keepsVals_read (ctx : Ctx) (ρ : Valuation) (n : Name) (idx : Idx) (oob : Bool) :
    KeepsVals fun st => st.read ctx ρ n idx oob := by
  intro st a st' h
  simp only [St.read] at h
  cases hl : ctx.loc ρ st.mem n idx oob with
  | err e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Res.ok_bind] at h
    split at h
    · cases h
    simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
    rw [← h.2]; rfl

theorem St.store_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {n : Name} {idx : Idx} {oob : Bool}
    {w : Option WcrOp} {v : Val} (h : st.store ctx ρ n idx oob w v = .ok st') : st'.vals = st.vals := by
  cases w with
  | none =>
    simp only [St.store, St.write] at h
    cases hl : ctx.loc ρ st.mem n idx oob with
    | err e => rw [hl] at h; cases h
    | ok l => rw [hl] at h; simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj] at h; rw [← h]
  | some op =>
    simp only [St.store, St.wcr] at h
    cases hl : ctx.loc ρ st.mem n idx oob with
    | err e => rw [hl] at h; cases h
    | ok l => rw [hl] at h; simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj] at h; rw [← h]

/-- Preservation of a relation along a fold in `Res`. -/
theorem List.foldlM_pres {α β : Type} (P : β → β → Prop) (hrefl : ∀ b, P b b)
    (htrans : ∀ a b c, P a b → P b c → P a c) (f : β → α → Res β) :
    ∀ (xs : List α) (b b' : β), (∀ x ∈ xs, ∀ c c', f c x = .ok c' → P c c') → xs.foldlM f b = .ok b' → P b b'
  | [], b, b', _, h => by
    simp only [List.foldlM_nil, Res.pure_eq, Res.ok_inj] at h
    subst h; exact hrefl b
  | x :: xs, b, b', hf, h => by
    rw [List.foldlM_cons] at h
    cases hx : f b x with
    | err e => rw [hx] at h; cases h
    | ok c =>
      rw [hx, Res.ok_bind] at h
      exact htrans _ _ _ (hf x List.mem_cons_self b c hx)
        (List.foldlM_pres P hrefl htrans f xs c b' (fun y hy => hf y (List.mem_cons_of_mem _ hy)) h)

theorem St.readMany_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {n : Name} {pts : List Idx} {oob : Bool}
    {vs : List Val} (h : st.readMany ctx ρ n pts oob = .ok (vs, st')) : st'.vals = st.vals := by
  unfold St.readMany at h
  have := List.foldlM_pres (fun (p q : List Val × St) => q.2.vals = p.2.vals) (fun _ => rfl)
    (fun _ _ _ h₁ h₂ => h₂.trans h₁) _ pts _ _ (fun p _ ⟨acc, c⟩ ⟨acc', c'⟩ hc => by
      dsimp only at hc
      cases hr : c.read ctx ρ n p oob with
      | err e => rw [hr] at hc; cases hc
      | ok q =>
        obtain ⟨v, c₁⟩ := q
        rw [hr] at hc
        simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hc
        rw [← hc.2]
        exact keepsVals_read ctx ρ n p oob c v c₁ hr) h
  exact this

theorem St.push_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {n : Name} {idx : Idx} {vs : List Val}
    (h : st.push ctx ρ n idx vs = .ok st') : st'.vals = st.vals := by
  simp only [St.push] at h
  cases hl : ctx.loc ρ st.mem n idx with
  | err e => rw [hl] at h; cases h
  | ok l =>
    rw [hl, Res.ok_bind] at h
    split at h <;> simp only [Res.pure_eq, Res.ok_inj] at h <;> rw [← h]

theorem St.popN_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {n : Name} {idx : Idx} {k : Nat} {vs : List Val}
    (h : st.popN ctx ρ n idx k = .ok (vs, st')) : st'.vals = st.vals := by
  simp only [St.popN] at h
  cases hl : ctx.loc ρ st.mem n idx with
  | err e => rw [hl] at h; cases h
  | ok l =>
    rw [hl, Res.ok_bind] at h
    split at h
    · simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h; rw [← h.2]
    · split at h
      · cases h
      · simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h; rw [← h.2]

theorem readScalarData_vals {ctx : Ctx} {ρ : Valuation} {d : Name} {sub : Option Subset} {oob : Bool}
    {st st' : St} {v? : Option Val} (h : readScalarData ctx ρ d sub oob st = .ok (v?, st')) : st'.vals = st.vals := by
  unfold readScalarData at h
  cases hp : ctx.pointsOf ρ st.mem d sub with
  | err e => rw [hp] at h; cases h
  | ok pts =>
    rw [hp] at h
    simp only [Res.ok_bind] at h
    split at h
    · split at h
      · cases hpop : st.popN ctx ρ d _ 1 with
        | err e => rw [hpop] at h; cases h
        | ok q =>
          rw [hpop] at h
          simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
          rw [← h.2]; exact St.popN_vals hpop
      · cases hr : st.read ctx ρ d _ oob with
        | err e => rw [hr] at h; cases h
        | ok q =>
          rw [hr] at h
          simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
          rw [← h.2]; exact keepsVals_read ctx ρ d _ oob st _ _ hr
    · cases h

theorem copyData_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {sd : Name} {ssub : Option Subset} {dd : Name}
    {dsub : Option Subset} {w : Option WcrOp} {oob : Bool} (h : copyData ctx ρ st sd ssub dd dsub w oob = .ok st') :
    st'.vals = st.vals := by
  have hstore : ∀ (ps : List (Idx × Val)) (c c' : St),
      ps.foldlM (fun st (q, v) => st.store ctx ρ dd q oob w v) c = .ok c' → c'.vals = c.vals := by
    intro ps c c' hf
    exact List.foldlM_pres (fun (p q : St) => q.vals = p.vals) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₂.trans h₁) _
      ps c c' (fun ⟨q, v⟩ _ a b hab => St.store_vals hab) hf
  unfold copyData at h
  split at h
  · cases h
  · obtain ⟨dpts, -, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨sp, -, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨⟨vs, c⟩, hp, h⟩ := Res.bind_eq_ok.1 h
    exact (hstore _ _ _ h).trans (St.popN_vals hp)
  · obtain ⟨spts, -, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨dp, -, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨⟨vs, c⟩, hp, h⟩ := Res.bind_eq_ok.1 h
    exact (St.push_vals h).trans (St.readMany_vals hp)
  · obtain ⟨spts, -, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨dpts, -, h⟩ := Res.bind_eq_ok.1 h
    exact List.foldlM_pres (fun (p q : St) => q.vals = p.vals) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₂.trans h₁) _
      _ _ _ (fun ⟨p, q⟩ _ a b hab => by
        obtain ⟨⟨v, a₁⟩, hr, hs⟩ := Res.bind_eq_ok.1 hab
        exact (St.store_vals hs).trans (keepsVals_read ctx ρ sd p oob a v a₁ hr)) h

/-! ## The correspondence -/

/-- An SDFG state of a sequential run and a tree state with the same memory
and log, the SDFG state without code-edge values. -/
def StRel (st : St) (cst : CSt) : Prop := st.mem = cst.mem ∧ st.log = cst.log ∧ st.vals = []

/-- Nodes whose execution is the copies into them. -/
def DNode.isCopyTarget (n : DNode) : Bool :=
  match n.kind with
  | .access _ | .mapExit | .consumeExit => true
  | _ => false

/-! ## Subsets up to the full range -/

/-- A tree memlet's subset of container `base` for the SDFG's subset: the
same, or omitted where the SDFG names the full range (`pointsOf` reads an
omitted subset as the full range). -/
def SubCorr (ctx : Ctx) (base : Name) (tsub ssub : Option Subset) : Prop :=
  tsub = ssub ∨ tsub = none ∧ ∃ dd, ctx.desc? base = some dd ∧ ssub = some (.range dd.fullRange)

theorem mapM_zero_lo (ctx : Ctx) (ρ : Valuation) (m : Mem) (er : ErrReason) :
    ∀ sh : List Expr, (Range.ofShape sh).dims.mapM (fun dim => Res.ofOption (ctx.evalInt ρ m dim.lo) er) =
      pure (sh.map fun _ => 0)
  | [] => rfl
  | x :: xs => by
    have ih := mapM_zero_lo ctx ρ m er xs
    simp only [Range.ofShape, List.map_cons, List.mapM_cons] at ih ⊢
    have h0 : ctx.evalInt ρ m (0 : Expr) = some 0 := rfl
    rw [h0, ih]
    rfl

/-- The lower corner of the full range is the origin. -/
theorem lowerCorner_full {ctx : Ctx} {ρ : Valuation} {m : Mem} {base : Name} {dd : DataDesc}
    (hd : ctx.desc? base = some dd) :
    ctx.lowerCorner ρ m base (some (.range dd.fullRange)) = ctx.lowerCorner ρ m base none := by
  unfold Ctx.lowerCorner
  have hp : ctx.pointsOf ρ m base (some (.range dd.fullRange)) = ctx.pointsOf ρ m base none := by
    simp [Ctx.pointsOf, hd]
  rw [hp]
  cases ctx.pointsOf ρ m base none with
  | err _ => rfl
  | ok _ =>
    simp only [Res.ok_bind, hd, DataDesc.fullRange]
    exact mapM_zero_lo _ _ _ _ _

theorem aliasTo_subCorr {ctx : Ctx} {ρ : Valuation} {st : St} {d base : Name} {tsub ssub : Option Subset}
    (h : SubCorr ctx base tsub ssub) : aliasTo ctx ρ st d base tsub = aliasTo ctx ρ st d base ssub := by
  rcases h with rfl | ⟨rfl, dd, hd, rfl⟩
  · rfl
  · unfold aliasTo
    rw [lowerCorner_full hd]

theorem pointsOf_subCorr {ctx : Ctx} {ρ : Valuation} {m : Mem} {n : Name} {tsub ssub : Option Subset}
    (h : SubCorr ctx n tsub ssub) : ctx.pointsOf ρ m n tsub = ctx.pointsOf ρ m n ssub := by
  rcases h with rfl | ⟨rfl, dd, hd, rfl⟩
  · rfl
  · simp [Ctx.pointsOf, hd]

theorem copyData_subCorr {ctx : Ctx} {ρ : Valuation} {st : St} {sd dd : Name} {ts ss td sdd : Option Subset}
    (hs : SubCorr ctx sd ts ss) (hd : SubCorr ctx dd td sdd) (wcr : Option WcrOp) (oob : Bool) :
    copyData ctx ρ st sd ts dd td wcr oob = copyData ctx ρ st sd ss dd sdd wcr oob := by
  simp only [copyData, streamPoint, pointsOf_subCorr hs, pointsOf_subCorr hd]

/-- A tree copy key for an SDFG copy key: the same up to `SubCorr`. -/
def KeyCorr (ctx : Ctx) (kT kS : CopyKey) : Prop :=
  kT.dst = kS.dst ∧ kT.src = kS.src ∧ kT.wcr = kS.wcr ∧ kT.oob = kS.oob ∧
    SubCorr ctx kS.src kT.ssub kS.ssub ∧ SubCorr ctx kS.dst kT.dsub kS.dsub

/-- The in-edges of a node that execute something (`execCopiesInto`): no
dependency edges, view definitions or routing edges. -/
def _root_.Dace.State.effIn (ctx : Ctx) (s : State) (e : DEdge) : Bool :=
  match e.data with
  | .dep => false
  | .memlet _ => !(ctx.isViewDef s e || s.isRouting e)

/-- A node whose one effective in-edge sets a reference from a container
(`refSet`): the reference, the container and its subset. -/
def _root_.Dace.State.refSetIn? (ctx : Ctx) (s : State) (n : DNode) : Option (Name × Name × Option Subset) :=
  match (s.inEdges n.id).filter (s.effIn ctx) with
  | [e] =>
    match refSetTarget? ctx s e, s.srcEndpoint e with
    | some d, .data base sub => some (d, base, sub)
    | _, _ => none
  | _ => none

theorem execCopiesInto_refSet {ctx : Ctx} {s : State} {n : DNode} {d base : Name} {sub : Option Subset}
    (h : s.refSetIn? ctx n = some (d, base, sub)) (ρ : Valuation) (st : St) :
    execCopiesInto ctx ρ s n st = aliasTo ctx ρ st d base sub := by
  unfold execCopiesInto
  rw [foldlM_filter _ (s.effIn ctx) (fun b x hx => by
    unfold State.effIn at hx
    split at hx
    · rename_i hd; simp [hd]
    · rename_i m hm
      simp only [Bool.not_eq_false'] at hx
      simp [hm, hx])]
  unfold State.refSetIn? at h
  split at h
  · rename_i e he
    rw [he]
    split at h
    · rename_i d' base' sub' hr hs
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      have hmem : e ∈ (s.inEdges n.id).filter (s.effIn ctx) := by rw [he]; exact List.mem_singleton_self e
      have hef := (List.mem_filter.1 hmem).2
      unfold State.effIn at hef
      split at hef
      · cases hef
      · rename_i m hm
        simp only [Bool.not_eq_true'] at hef
        simp only [List.foldlM_cons, List.foldlM_nil, hm, hef, Bool.false_eq_true, ↓reduceIte, hr, refSet, hs]
        cases aliasTo ctx ρ st d' base' sub' <;> rfl
    · cases h
  · cases h

/-- A node whose one effective in-edge sets a reference from a code node
(unsupported in both IRs, `Decisions.md#D55`): the reference. -/
def _root_.Dace.State.refSetCode? (ctx : Ctx) (s : State) (n : DNode) : Option Name :=
  match (s.inEdges n.id).filter (s.effIn ctx) with
  | [e] =>
    match refSetTarget? ctx s e, s.srcEndpoint e with
    | some d, .code _ => some d
    | _, _ => none
  | _ => none

theorem execCopiesInto_refSetCode {ctx : Ctx} {s : State} {n : DNode} {d : Name}
    (h : s.refSetCode? ctx n = some d) (ρ : Valuation) (st : St) : ∃ e, execCopiesInto ctx ρ s n st = .err e := by
  unfold execCopiesInto
  rw [foldlM_filter _ (s.effIn ctx) (fun b x hx => by
    unfold State.effIn at hx
    split at hx
    · rename_i hd; simp [hd]
    · rename_i m hm
      simp only [Bool.not_eq_false'] at hx
      simp [hm, hx])]
  unfold State.refSetCode? at h
  split at h
  · rename_i e he
    rw [he]
    split at h
    · rename_i d' k hr hs
      have hmem : e ∈ (s.inEdges n.id).filter (s.effIn ctx) := by rw [he]; exact List.mem_singleton_self e
      have hef := (List.mem_filter.1 hmem).2
      unfold State.effIn at hef
      split at hef
      · cases hef
      · rename_i m hm
        simp only [Bool.not_eq_true'] at hef
        simp only [List.foldlM_cons, List.foldlM_nil, hm, hef, Bool.false_eq_true, ↓reduceIte, hr, refSet, hs]
        exact ⟨_, rfl⟩
    · cases h
  · cases h

/-! ## Nested SDFGs inside map scopes -/

/-- The SDFG context in which the tree's flattened copy of the top region of
a nested SDFG `inner` runs: the tree's descriptors and memory names. -/
def _root_.Dace.TCtx.innerCtx (tc : TCtx) (inner : SDFG) : Ctx := { tc.ctx with g := { tc.ctx.g with top := inner.top } }

theorem _root_.Dace.TCtx.innerCtx_eraseL (tc : TCtx) (inner : SDFG) : (tc.innerCtx inner).eraseL = tc.ctx.eraseL := rfl

theorem _root_.Dace.TCtx.innerCtx_I (tc : TCtx) (inner : SDFG) : (tc.innerCtx inner).I = tc.ctx.I := rfl

/-- The tree context renamed by `σ` (outer name ↦ inner name of a nested
SDFG, `Ctx.ren`), outside consume scopes. -/
def _root_.Dace.TCtx.renT (tc : TCtx) (σ : Equiv.Perm Name) : TCtx := { tc with ctx := tc.ctx.ren σ, consumed := none }

/-- The tree outcome `r` of a scope body is matched by the SDFG outcomes `X`:
`X` may be undefined, or has an error when `r` has one, or the body ends normally in
the same state, the region ending normally or by a return (which ends the
call, `SDFGExec.run`). -/
def NestedOut (X : Res (Ctl × CSt) → Prop) (r : Res (TCtl × CSt)) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ (∃ e e', r = .err e ∧ X (.err e')) ∨
    ∃ ctl c, (ctl = .normal ∨ ctl = .ret) ∧ r = .ok (.normal, c) ∧ X (.ok (ctl, c))

/-- A symbol mapping that passes symbols on unchanged, into an SDFG without
constants. -/
structure NestedMapOk (ctx : Ctx) (inner : SDFG) (mapping : List (Sym × Expr)) : Prop where
  ident : ∀ p ∈ mapping, p.2 = .sym p.1 ∧ ctx.isScalarName p.1 = false
  consts : ∀ ρ, inner.withConstants ρ = ρ

theorem Mem.allocAll_append (m : Mem) (a b : List Name) : (m.allocAll a).allocAll b = m.allocAll (a ++ b) := by
  simp [Mem.allocAll, List.foldl_append]

theorem Log.dropNames_dropNames (g : Log) (a b : List Name) : (g.dropNames a).dropNames b = g.dropNames (a ++ b) := by
  simp only [Log.dropNames, List.filter_filter, List.contains_append, Bool.not_or]
  congr 1 <;> (apply List.filter_congr; intro x _; rw [Bool.and_comm])

theorem nestedValuation_mapM {ctx : Ctx} {ρ : Valuation} {m : Mem} :
    ∀ (mapping : List (Sym × Expr)), (∀ p ∈ mapping, p.2 = .sym p.1 ∧ ctx.isScalarName p.1 = false) →
      (∃ vs, (mapping.mapM fun (x, e) => do
          let v ← ctx.evalR ρ m e
          pure (x, v)) = .ok vs ∧ ∀ p ∈ vs, ρ p.1 = some p.2) ∨
      ∃ e, (mapping.mapM fun (x, e) => do
          let v ← ctx.evalR ρ m e
          pure (x, v)) = .err e ∧ e.isUB = true
  | [], _ => .inl ⟨[], rfl, fun _ h => by cases h⟩
  | (x, e) :: rest, h => by
    obtain ⟨he, hs⟩ := h (x, e) List.mem_cons_self
    simp only at he hs
    subst he
    have hev : ctx.evalR ρ m (.sym x) = Res.ofOption (ρ x) (.undefinedExpr (.sym x)) := by
      simp only [Ctx.evalR, Ctx.eval, Expr.eval, Ctx.val_of_not_scalar ctx ρ m hs]
    rw [List.mapM_cons]
    cases hx : ρ x with
    | none =>
      refine .inr ⟨.undefinedExpr (.sym x), ?_, rfl⟩
      simp only [hev, hx, Res.ofOption]
      rfl
    | some v =>
      rcases nestedValuation_mapM rest (fun p hp => h p (List.mem_cons_of_mem _ hp)) with ⟨vs, hvs, hall⟩ | ⟨e', he', hu⟩
      · refine .inl ⟨(x, v) :: vs, ?_, ?_⟩
        · simp only [hev, hx, Res.ofOption]
          show Res.bind (Res.ok (x, v)) _ = _
          simp only [Res.bind]
          rw [hvs]; rfl
        · intro p hp
          rcases List.mem_cons.1 hp with rfl | hp
          · exact hx
          · exact hall p hp
      · refine .inr ⟨e', ?_, hu⟩
        simp only [hev, hx, Res.ofOption]
        show Res.bind (Res.ok (x, v)) _ = _
        simp only [Res.bind]
        rw [he']; rfl

/-- An identity symbol mapping leaves the valuation unchanged, unless a
symbol is unbound (undefined behavior). -/
theorem nestedValuation_id {ctx : Ctx} {inner : SDFG} {mapping : List (Sym × Expr)} (hm : NestedMapOk ctx inner mapping)
    (ρ : Valuation) (m : Mem) :
    nestedValuation ctx ρ m mapping = .ok ρ ∨ ∃ e, nestedValuation ctx ρ m mapping = .err e ∧ e.isUB = true := by
  unfold nestedValuation
  rcases nestedValuation_mapM (ρ := ρ) (m := m) mapping hm.ident with ⟨vs, hvs, hall⟩ | ⟨e, he, hu⟩
  · refine .inl ?_
    rw [hvs]
    show Res.ok (Valuation.setAll ρ vs) = Res.ok ρ
    congr 1
    funext y
    unfold Valuation.setAll
    cases hf : vs.find? (·.1 = y) with
    | none => rfl
    | some p =>
      obtain ⟨x, v⟩ := p
      have hmem := List.mem_of_find?_eq_some hf
      have hxy : x = y := by simpa using List.find?_some hf
      subst hxy
      exact (hall _ hmem).symm
  · exact .inr ⟨e, by rw [he]; rfl, hu⟩

/-- Without connectors on code edges, no scalar is copied in. -/
theorem bindCodeInputs_data {ctx' : Ctx} {ρ' : Valuation} {s : State} {n : DNode} {st : St}
    (h : ∀ c ∈ nestedConns n, (nestedEndpoint s n c).isData = true) :
    bindCodeInputs ctx' ρ' s n st = .ok (st, []) := by
  unfold bindCodeInputs
  generalize nestedConns n = cs at h ⊢
  generalize (st, ([] : List Name)) = acc
  induction cs generalizing acc with
  | nil => rfl
  | cons c cs ih =>
    rw [List.foldlM_cons]
    obtain ⟨a, b⟩ := acc
    have hd := h c List.mem_cons_self
    cases hE : nestedEndpoint s n c with
    | data d sub =>
      simp only
      exact ih (fun c' hc' => h c' (List.mem_cons_of_mem _ hc')) (a, b)
    | code k => rw [hE] at hd; cases hd
    | invalid => rw [hE] at hd; cases hd

/-- Without code consumers of the outputs, a nested call ends with the inner
memory and the inner log without the dropped names. -/
theorem nestedFinish_data {ctx' : Ctx} {ρ' : Valuation} {s : State} {n : DNode} {dropped : List Name} {pre : St}
    {m₂ : Mem} {g₂ : Log} (h : s.DataOuts n) :
    nestedFinish ctx' ρ' s n dropped pre m₂ g₂ =
      .ok { mem := m₂, log := pre.log.seq (g₂.dropNames dropped), vals := pre.vals } :=
  nestedFinish_data' h

theorem nestedSetup_eq {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {inner : SDFG}
    {mapping : List (Sym × Expr)} {st : St} {conns : List (Name × Name)} (hb : nestedBindings ctx s n = .ok conns)
    (hcode : ∀ c ∈ nestedConns n, (nestedEndpoint s n c).isData = true) :
    nestedSetup ctx ρ s n inner mapping st = (nestedValuation ctx ρ st.mem mapping).bind fun ρ' =>
      .ok (ctx.enter inner n.id conns, ρ',
        { st with mem := st.mem.allocAll (inner.callTransients.map (ctx.enter inner n.id conns).cname) },
        inner.callTransients.map (ctx.enter inner n.id conns).cname) := by
  unfold nestedSetup
  rw [hb]
  simp only [Res.ok_bind]
  cases hv : nestedValuation ctx ρ st.mem mapping with
  | err e => rfl
  | ok ρ' =>
    simp only [Res.ok_bind]
    rw [bindCodeInputs_data hcode]
    simp only [Res.ok_bind, List.nil_append]
    rfl

/-- The tree context with another interpretation. -/
def _root_.Dace.TCtx.withI (tc : TCtx) (I : Interp) : TCtx := { tc with ctx := { tc.ctx with I } }

/-- The tree list `body` matches the top region of `inner` run in the tree's
context renamed by `σ` (the tree inlines the nested SDFG under the outer
names, `TCtx.renT`), for every interpretation (the checker proves this under
one and transports it). -/
def InnerCorr (tc : TCtx) (σ : Equiv.Perm Name) (inner : SDFG) (body : List SNode) : Prop :=
  ∀ I : Interp, (∀ op v, I.wcr op .undef v = .undef) → ∀ (cst : CSt) (r : Res (TCtl × CSt)),
    STree.ListExec (tc.withI I) body false {} body cst r →
    NestedOut (RegionAt (((tc.withI I).renT σ).innerCtx inner) inner.top cst inner.top.start) r

/-- The context of a nested call and the tree's context agree on the names
of the inner SDFG's top region, for every interpretation. -/
def InnerNames (ctx : Ctx) (tc : TCtx) (inner : SDFG) (node : NodeId) (conns : List (Name × Name)) : Prop :=
  ∀ I : Interp, Region.Names 1 (fun _ => True) ({ ctx with I }.enter inner node conns) ((tc.withI I).innerCtx inner)
    inner.top

/-- Access or exit nodes without copies into them (the exit of a map whose
body is a nested SDFG): they run as no-ops. -/
def IdleNodes (ctx : Ctx) (s : State) (xs : List NodeId) : Prop :=
  ∀ x ∈ xs, ∃ xn, s.node? x = some xn ∧ DNode.isCopyTarget xn = true ∧ s.CopyShape ctx xn ∧ s.copiesInto ctx xn = []

/-- The `DynScopeCopyNode`s of the dynamic inputs `keys`, in order. -/
def DynNodes (tc : TCtx) (keys : List InKey) (ds : List SNode) : Prop :=
  List.Forall₂ (fun k n => ∃ m : Memlet, n = .dynScopeCopy k.conn m ∧ tc.inKey k.conn m = k) keys ds

theorem DNode.isSimple_of_copyTarget' {n : DNode} (hk : DNode.isCopyTarget n = true) : n.isSimple = true := by
  unfold DNode.isCopyTarget at hk
  unfold DNode.isSimple
  split at hk <;> simp_all

/-- Tree nodes that end normally with an empty sibling context and the same
valuation (dataflow nodes, no symbol assignments, no dynamic inputs). -/
def _root_.Dace.SNode.isFlatT : SNode → Bool
  | .mapScope .. | .taskletNode .. | .libraryCall .. | .copyNode .. | .viewNode .. | .refSetNode .. => true
  | _ => false

theorem leafOut_lift_ok {cst : CSt} {R : Res St} {sc' : SibCtx} {ctl : TCtl} {c : CSt}
    (h : leafOut (cst.lift R) = .ok (sc', ctl, c)) : sc' = {} ∧ ctl = .normal ∧ c.ρ = cst.ρ := by
  cases R with
  | err e => cases h
  | ok a =>
    simp only [leafOut, CSt.lift, Res.map_ok, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨rfl, rfl, rfl⟩

theorem STree.NodeExec.flatT_ok {tc : TCtx} {n : SNode} {cst : CSt} {sc' : SibCtx} {ctl : TCtl} {c : CSt}
    (hn : n.isFlatT = true) (hx : STree.NodeExec tc {} n cst (.ok (sc', ctl, c))) :
    sc' = {} ∧ ctl = .normal ∧ c.ρ = cst.ρ := by
  generalize hr : (Res.ok (sc', ctl, c) : Res NodeOut) = r at hx
  cases hx with
  | pendingErr => cases hr
  | leaf _ hs =>
    cases n <;> simp only [SNode.isFlatT, reduceCtorEq] at hn <;>
      simp only [stepLeaf, Option.some.injEq, reduceCtorEq] at hs <;> subst hs <;> exact leafOut_lift_ok hr.symm
  | mapScope _ _ =>
    cases h : scopeFinishT tc cst _ with
    | err e => rw [h] at hr; cases hr
    | ok a =>
      rw [h] at hr
      simp only [dataflowOut, Res.map_ok, Res.ok_inj, Prod.mk.injEq] at hr
      obtain ⟨rfl, rfl, rfl⟩ := hr
      refine ⟨rfl, rfl, ?_⟩
      simp only [scopeFinishT] at h
      cases h' : scopeFinish tc.ctx cst.st _ with
      | err e => rw [h'] at h; cases h
      | ok b => rw [h'] at h; simp only [Res.map_ok, Res.ok_inj] at h; subst h; rfl
  | mapSetupErr => cases hr
  | mapIterErr => cases hr
  | loop _ hl _ => cases n <;> simp [SNode.isFlatT, SNode.loop?] at hn hl
  | _ => simp [SNode.isFlatT] at hn

/-- A run of a list of flat tree nodes followed by `rest`: the run of the flat
prefix alone (in any enclosing list), then of `rest`. -/
theorem STree.ListExec.split_flat {tc : TCtx} {full : List SNode} {gb : Bool} {rest : List SNode} :
    ∀ {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}, body.all SNode.isFlatT = true →
      STree.ListExec tc full gb {} (body ++ rest) cst r →
      (∃ e, r = .err e ∧ ∀ full', STree.ListExec tc full' false {} body cst (.err e)) ∨
      ∃ cst', (∀ full', STree.ListExec tc full' false {} body cst (.ok (.normal, cst'))) ∧ cst'.ρ = cst.ρ ∧
        STree.ListExec tc full gb {} rest cst' r
  | [], cst, _, _, hx => .inr ⟨cst, fun _ => .nil rfl, rfl, hx⟩
  | n :: body, cst, r, hb, hx => by
    simp only [List.all_cons, Bool.and_eq_true] at hb
    rw [List.cons_append] at hx
    cases hx with
    | headErr hn => exact .inl ⟨_, rfl, fun _ => .headErr hn⟩
    | step hn hr =>
      obtain ⟨rfl, -, hρ₁⟩ := STree.NodeExec.flatT_ok hb.1 hn
      rcases STree.ListExec.split_flat hb.2 hr with ⟨e, rfl, hb'⟩ | ⟨cst', hb', hρ, hr'⟩
      · exact .inl ⟨e, rfl, fun full' => .step hn (hb' full')⟩
      · exact .inr ⟨cst', fun full' => .step hn (hb' full'), hρ.trans hρ₁, hr'⟩
    | jump hn hc =>
      obtain ⟨-, rfl, -⟩ := STree.NodeExec.flatT_ok hb.1 hn
      cases hc
    | gotoResume hn _ _ =>
      obtain ⟨-, h, -⟩ := STree.NodeExec.flatT_ok hb.1 hn
      cases h
    | gotoEscape hn _ =>
      obtain ⟨-, h, -⟩ := STree.NodeExec.flatT_ok hb.1 hn
      cases h

/-- The inlined body of a nested SDFG outside map scopes: flat tree nodes, or
nodes without symbol assignments and loops whose gotos all reach the label
the inner returns were rebound to, which ends the body. -/
def NestedShape (body : List SNode) : Prop :=
  body.all SNode.isFlatT = true ∨
    ∃ L pre, body = pre ++ [.stateLabel L] ∧ SNodes.hasLabel L pre = false ∧ SNodes.jumpsTo (some L) pre = true ∧
      SNodes.keepsSyms pre = true

/-- A run of an inlined body followed by `rest`: the body alone fails or
jumps out the same way, or ends normally with the same valuation and `rest`
runs from there. -/
theorem STree.ListExec.split_nested {tc : TCtx} {full : List SNode} {gb : Bool} {rest body : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (hb : NestedShape body) (hx : STree.ListExec tc full gb {} (body ++ rest) cst r) :
    (∃ e, r = .err e ∧ ∀ full', STree.ListExec tc full' false {} body cst (.err e)) ∨
    (∃ ctl c, ctl.isJump = true ∧ r = .ok (ctl, c) ∧ ∀ full', STree.ListExec tc full' false {} body cst (.ok (ctl, c))) ∨
    ∃ c, (∀ full', STree.ListExec tc full' false {} body cst (.ok (.normal, c))) ∧ c.ρ = cst.ρ ∧
      STree.ListExec tc full gb {} rest c r := by
  rcases hb with hf | ⟨L, pre, rfl, hl, hj, hk⟩
  · rcases STree.ListExec.split_flat hf hx with ⟨e, rfl, h⟩ | ⟨c, h, hρ, hr⟩
    · exact .inl ⟨e, rfl, h⟩
    · exact .inr (.inr ⟨c, h, hρ, hr⟩)
  · rw [List.append_assoc, List.singleton_append] at hx
    rcases STree.ListExec.split_label L hl hj hx with ⟨e, rfl, h⟩ | ⟨ctl, c, hc, rfl, h⟩ | ⟨c, h, hr⟩
    · exact .inl ⟨e, rfl, h⟩
    · exact .inr (.inl ⟨ctl, c, hc, rfl, h⟩)
    · have hk' : SNodes.keepsSyms (pre ++ [.stateLabel L]) = true := by
        rw [SNodes.keepsSyms_append, hk]; rfl
      exact .inr (.inr ⟨c, h, STree.ListExec.keeps_ρ (h _) hk' hk' _ _ rfl, hr⟩)

/-- The tree list `cs` is the image of the SDFG nodes `o` of state `s`. -/
inductive SegCorr (ctx : Ctx) : TCtx → State → List NodeId → List SNode → Prop where
  | nil {tc : TCtx} {s : State} : SegCorr ctx tc s [] []
  | copies {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {cps : List (Name × Memlet)}
      (hn : s.node? id = some n) (hk : DNode.isCopyTarget n = true) (hsh : s.CopyShape ctx n)
      (hks : List.Forall₂ (KeyCorr ctx) (cps.map fun p => tc.copyKey p.1 p.2) (s.copiesInto ctx n))
      (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (cps.map (fun p => SNode.copyNode p.1 p.2) ++ cs)
  /-- A node whose one effective in-edge sets a reference from a container
  (`Decisions.md#D32`): the tree's `RefSetNode` from data. -/
  | refSet {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {d base : Name}
      {sub : Option Subset} {m : Memlet} {sd x : DataDesc}
      (hn : s.node? id = some n) (hk : DNode.isCopyTarget n = true) (hr : s.refSetIn? ctx n = some (d, base, sub))
      (hm : m.data = base) (hsub : SubCorr ctx base m.dataSubset sub) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.refSetNode d m (.data sd) x :: cs)
  /-- A node whose one effective in-edge sets a reference from a code node:
  both IRs fail there (`Decisions.md#D55`). -/
  | refSetCode {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {d : Name}
      {m : Memlet} {g cls : String} {x : DataDesc}
      (hn : s.node? id = some n) (hk : DNode.isCopyTarget n = true) (hr : s.refSetCode? ctx n = some d)
      (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.refSetNode d m (.codeNode g cls) x :: cs)
  | tasklet {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {t : Tasklet} {ins outs : MemletMap}
      (hn : s.node? id = some n) (hk : n.kind = .tasklet t) (hsh : s.CodeShape n)
      (hki : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2)
      (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2)
      (hd : s.hasDynamicWcrOut n = false) (hdT : outs.hasDynamicWcr = false) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.taskletNode t ins outs :: cs)
  | library {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {l l' : LibNode}
      {props : List (String × String)} {ins outs : MemletMap}
      (hn : s.node? id = some n) (hk : n.kind = .library l) (hl : l'.classpath = l.classpath ∧ l'.reduce = l.reduce)
      (hsh : s.CodeShape n)
      (hki : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2)
      (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2)
      (hd : s.hasDynamicWcrOut n = false) (hdT : outs.hasDynamicWcr = false) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.libraryCall l' props ins outs :: cs)
  | map {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {mp : MapNode} {oM full : List NodeId}
      {body : List SNode} {keys : List InKey} {ds : List SNode}
      (hn : s.node? id = some n) (hk : n.kind = .mapEntry mp)
      (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (hds : DynNodes tc keys ds)
      (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
      (hdir : full.filter (s.directChildren (some n.id)).contains = oM)
      (hpriv : tc.privateTransients body = ctx.privateTransients s n.id)
      (hb : SegCorr ctx tc s oM body) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (ds ++ .mapScope mp body :: cs)
  /-- A consume scope without dynamic inputs (`Decisions.md#D38`): the
  `IN_stream` copy designates the source of the entry's `IN_stream` edge, the
  stream the entry consumes; the body runs in the consuming context, where
  the stream's memlets read the chunk. -/
  | consume {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode}
      {c : ConsumeNode} {oM full : List NodeId} {body : List SNode} {e : DEdge} {mIn : Memlet}
      (hn : s.node? id = some n) (hk : n.kind = .consumeEntry c) (hkeys : s.dynKeys? n = some [])
      (hin : (s.inEdges n.id).find? (·.dstConn = some "IN_stream") = some e)
      (hsrc : s.srcEndpoint e = .data mIn.data mIn.dataSubset) (hd : s.consumedStream? n.id = some mIn.data)
      (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
      (hdir : full.filter (s.directChildren (some n.id)).contains = oM)
      (hpriv : tc.privateTransients body = ctx.privateTransients s n.id)
      (hb : SegCorr ctx (tc.consuming mIn.data) s oM body) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.dynScopeCopy "IN_stream" mIn :: .consumeScope c body :: cs)
  /-- A map scope whose body is one nested SDFG: the tree inlines the inner
  top region under the outer names (`InnerCorr`, renamed by `σ`); the
  per-call transients of the inner SDFG are private to the tree's iteration
  (`Decisions.md#D68`). -/
  | nestedMap {tc : TCtx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {mp : MapNode} {full : List NodeId}
      {body : List SNode} {keys : List InKey} {ds : List SNode} {nid : NodeId} {nd : DNode} {inner : SDFG}
      {mapping : List (Sym × Expr)} {ni : Bool} {conns : List (Name × Name)} {σ : Equiv.Perm Name}
      (hn : s.node? id = some n) (hk : n.kind = .mapEntry mp)
      (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (hds : DynNodes tc keys ds)
      {xs : List NodeId}
      (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
      (hdir : full.filter (s.directChildren (some n.id)).contains = nid :: xs) (hxs : IdleNodes ctx s xs)
      (hnd : s.node? nid = some nd) (hkn : nd.kind = .nested inner mapping ni)
      (hb : nestedBindings ctx s nd = .ok conns) (hmap : NestedMapOk ctx inner mapping)
      (hcode : ∀ c ∈ nestedConns nd, (nestedEndpoint s nd c).isData = true)
      (hout : s.DataOuts nd)
      (hpriv : ∀ x, x ∈ tc.privateTransients body ↔
        x ∈ ctx.privateTransients s n.id ++ inner.callTransients.map (ctx.enter inner nd.id conns).cname)
      (hnames : InnerNames ctx (tc.renT σ) inner nd.id conns) (hinner : InnerCorr tc σ inner body)
      (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (ds ++ .mapScope mp body :: cs)
  /-- A nested SDFG node outside map scopes: the tree inlines the inner top
  region in the sibling list (`NestedShape`). -/
  | nested {tc : TCtx} {s : State} {id : NodeId} {nd : DNode} {o : List NodeId} {cs body : List SNode} {inner : SDFG}
      {mapping : List (Sym × Expr)} {ni : Bool} {conns : List (Name × Name)} {σ : Equiv.Perm Name}
      (hnd : s.node? id = some nd) (hkn : nd.kind = .nested inner mapping ni)
      (hb : nestedBindings ctx s nd = .ok conns) (hmap : NestedMapOk ctx inner mapping)
      (hcode : ∀ c ∈ nestedConns nd, (nestedEndpoint s nd c).isData = true)
      (hout : s.DataOuts nd)
      (hctr : inner.callTransients = []) (hshape : NestedShape body)
      (hnames : InnerNames ctx (tc.renT σ) inner nd.id conns) (hinner : InnerCorr tc σ inner body)
      (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (body ++ cs)

/-- The nodes of a segment run by themselves: simple nodes, map and consume
scopes, nested SDFGs without per-call transients. -/
theorem SegCorr.flat {ctx : Ctx} {tc : TCtx} {s : State} {o : List NodeId} {cs : List SNode}
    (hc : SegCorr ctx tc s o cs) : ∀ id ∈ o, ∀ n, s.node? id = some n → n.CohIn s := by
  induction hc with
  | nil => intro id hid; cases hid
  | copies hn hk _ _ _ ih | refSet hn hk _ _ _ _ ih | refSetCode hn hk _ _ ih | tasklet hn hk _ _ _ _ _ _ ih | library hn hk _ _ _ _ _ _ _ ih
  | map hn hk _ _ _ _ _ _ _ _ _ ih | nestedMap hn hk _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih =>
    intro id hid n' hn'
    rcases List.mem_cons.1 hid with rfl | hid
    · rw [hn] at hn'
      cases hn'
      refine .inl (.inl ?_)
      first
        | exact .inr ⟨_, hk⟩
        | exact .inl (DNode.isSimple_of_copyTarget' hk)
        | (left; simp [DNode.isSimple, hk])
    · exact ih id hid n' hn'
  | consume hn hk _ _ _ _ _ _ _ _ _ _ ih =>
    intro id hid n' hn'
    rcases List.mem_cons.1 hid with rfl | hid
    · rw [hn] at hn'
      cases hn'
      exact .inr ⟨_, hk⟩
    · exact ih id hid n' hn'
  | nested hn hkn _ _ hcode hout hctr _ _ _ _ ih =>
    intro id hid n' hn'
    rcases List.mem_cons.1 hid with rfl | hid
    · rw [hn] at hn'
      cases hn'
      exact .inl (.inr ⟨_, _, _, hkn, hctr, hcode, hout⟩)
    · exact ih id hid n' hn'

/-- The outcomes of the sequential SDFG run of `o` from `st` that match a
tree run ending in `r` after the image of `o`, continuing with `rest`. -/
def SeqOut (ctx : Ctx) (s : State) (o : List NodeId) (ρ : Valuation) (st : St) (tc : TCtx) (full : List SNode)
    (gb : Bool) (rest : List SNode) (r : Res (TCtl × CSt)) : Prop :=
  (∃ e, e.isUB = true ∧ SeqExec ctx ρ s o st (.err e)) ∨
  (∃ e e', r = .err e ∧ SeqExec ctx ρ s o st (.err e')) ∨
  ∃ cst' st' tr, STree.ListExec tc full gb {} rest cst' r ∧ SeqExec ctx ρ s o st (.ok (st', tr)) ∧
    cst'.ρ = ρ ∧ StRel st' cst'

/-! ## Inversion of tree runs -/

/-- A tree list starting with a leaf whose step is `leafOut R`. -/
theorem leaf_inv {tc : TCtx} {full : List SNode} {gb : Bool} {n : SNode} {rest : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} {R : Res CSt} (hstep : stepLeaf tc {} n cst = some (leafOut R))
    (hx : STree.ListExec tc full gb {} (n :: rest) cst r) :
    (∃ e, R = .err e ∧ r = .err e) ∨ ∃ cst', R = .ok cst' ∧ STree.ListExec tc full gb {} rest cst' r := by
  have hleaf : ∀ {sc : SibCtx} {res : Res NodeOut}, STree.NodeExec tc sc n cst res → sc = {} → res = leafOut R := by
    intro sc res h hsc
    subst hsc
    cases h with
    | pendingErr _ hp => simp [Pending.isEmpty] at hp
    | leaf _ h' => rw [hstep] at h'; exact (Option.some.inj h').symm
    | loop _ hl _ => cases n <;> simp_all [stepLeaf, SNode.loop?]
    | _ => simp [stepLeaf] at hstep
  cases hx with
  | headErr h =>
    have := hleaf h rfl
    cases R with
    | ok c => simp [leafOut, Res.map] at this
    | err e => simp only [leafOut, Res.map, Res.err.injEq] at this; exact .inl ⟨e, rfl, by rw [this]⟩
  | step h hr =>
    have := hleaf h rfl
    cases R with
    | err e => simp [leafOut, Res.map] at this
    | ok c =>
      simp only [leafOut, Res.map, Res.ok.injEq, Prod.mk.injEq] at this
      obtain ⟨rfl, -, rfl⟩ := this
      exact .inr ⟨_, rfl, hr⟩
  | jump h hc =>
    have := hleaf h rfl
    cases R with
    | err e => simp [leafOut, Res.map] at this
    | ok c =>
      simp only [leafOut, Res.map, Res.ok.injEq, Prod.mk.injEq] at this
      rw [this.2.1] at hc; cases hc
  | gotoResume h _ _ =>
    have := hleaf h rfl
    cases R <;> simp [leafOut, Res.map] at this
  | gotoEscape h _ =>
    have := hleaf h rfl
    cases R <;> simp [leafOut, Res.map] at this

/-- A run of a map scope node. -/
theorem mapScope_node_inv {tc : TCtx} {sc : SibCtx} {mp : MapNode} {body : List SNode} {cst : CSt}
    {res : Res NodeOut}
    (h : STree.NodeExec tc sc (.mapScope mp body) cst res) :
    (∃ e, mapSetupT tc sc mp cst = .err e ∧ res = .err e) ∨
    (∃ ρ' pts e, mapSetupT tc sc mp cst = .ok (ρ', pts) ∧
      STree.IterExec tc mp.params body (tc.privateTransients body) { cst with ρ := ρ' } pts (.err e) ∧
      res = .err e) ∨
    ∃ ρ' pts logs, mapSetupT tc sc mp cst = .ok (ρ', pts) ∧
      STree.IterExec tc mp.params body (tc.privateTransients body) { cst with ρ := ρ' } pts (.ok logs) ∧
      res = dataflowOut (scopeFinishT tc cst logs) := by
  generalize hn : SNode.mapScope mp body = n at h
  cases h with
  | pendingErr hc _ => subst hn; simp [SNode.consumesPending, SNode.isDataflowScope] at hc
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | mapSetupErr hs => cases hn; exact .inl ⟨_, hs, rfl⟩
  | mapIterErr hs hi => cases hn; exact .inr (.inl ⟨_, _, _, hs, hi, rfl⟩)
  | mapScope hs hi => cases hn; exact .inr (.inr ⟨_, _, _, hs, hi, rfl⟩)
  | _ => cases hn

/-- A tree list starting with a map scope. -/
theorem map_inv {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {mp : MapNode} {body rest : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (hx : STree.ListExec tc full gb sc (.mapScope mp body :: rest) cst r) :
    (∃ e, mapSetupT tc sc mp cst = .err e ∧ r = .err e) ∨
    (∃ ρ' pts e, mapSetupT tc sc mp cst = .ok (ρ', pts) ∧
      STree.IterExec tc mp.params body (tc.privateTransients body) { cst with ρ := ρ' } pts (.err e) ∧ r = .err e) ∨
    ∃ ρ' pts logs, mapSetupT tc sc mp cst = .ok (ρ', pts) ∧
      STree.IterExec tc mp.params body (tc.privateTransients body) { cst with ρ := ρ' } pts (.ok logs) ∧
      ((∃ e, scopeFinishT tc cst logs = .err e ∧ r = .err e) ∨
        ∃ cst', scopeFinishT tc cst logs = .ok cst' ∧ STree.ListExec tc full gb {} rest cst' r) := by
  have fin : ∀ {logs : List Log} {res : Res NodeOut}, res = dataflowOut (scopeFinishT tc cst logs) →
      (∀ e, res = .err e → ∃ e', scopeFinishT tc cst logs = .err e' ∧ e' = e) ∧
      (∀ sc ctl c, res = .ok (sc, ctl, c) → sc = {} ∧ ctl = .normal ∧ scopeFinishT tc cst logs = .ok c) := by
    intro logs res hres
    subst hres
    refine ⟨fun e he => ?_, fun sc ctl c hc => ?_⟩
    · cases hf : scopeFinishT tc cst logs <;> rw [hf] at he <;> simp_all [dataflowOut, Res.map]
    · cases hf : scopeFinishT tc cst logs <;> rw [hf] at hc <;> simp_all [dataflowOut, Res.map]
  cases hx with
  | headErr h =>
    rcases mapScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', pts, e, hs, hi, he⟩ | ⟨ρ', pts, logs, hs, hi, hres⟩
    · cases he; exact .inl ⟨_, hs, rfl⟩
    · cases he; exact .inr (.inl ⟨_, _, _, hs, hi, rfl⟩)
    · obtain ⟨e', hf, rfl⟩ := (fin hres).1 _ rfl
      exact .inr (.inr ⟨_, _, _, hs, hi, .inl ⟨_, hf, rfl⟩⟩)
  | step h hr =>
    rcases mapScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', pts, e, hs, hi, he⟩ | ⟨ρ', pts, logs, hs, hi, hres⟩
    · cases he
    · cases he
    · obtain ⟨rfl, -, hf⟩ := (fin hres).2 _ _ _ rfl
      exact .inr (.inr ⟨_, _, _, hs, hi, .inr ⟨_, hf, hr⟩⟩)
  | jump h hc =>
    rcases mapScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', pts, e, hs, hi, he⟩ | ⟨ρ', pts, logs, hs, hi, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, rfl, -⟩ := (fin hres).2 _ _ _ rfl
      cases hc
  | gotoResume h _ _ =>
    rcases mapScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', pts, e, hs, hi, he⟩ | ⟨ρ', pts, logs, hs, hi, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, h₂, -⟩ := (fin hres).2 _ _ _ rfl
      cases h₂
  | gotoEscape h _ =>
    rcases mapScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', pts, e, hs, hi, he⟩ | ⟨ρ', pts, logs, hs, hi, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, h₂, -⟩ := (fin hres).2 _ _ _ rfl
      cases h₂

/-- Pending dynamic inputs (no stream), closed `if` chain. -/
abbrev pendingSc (syms : List (Sym × Val)) : SibCtx := { pending := { syms } }

/-- **A run of `DynScopeCopyNode`s**: reads the inputs under the scope's
valuation into the pending symbols. -/
theorem dscopies_run {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL)
    {full : List SNode} {gb : Bool} {rest : List SNode} {r : Res (TCtl × CSt)} :
    ∀ {keys : List InKey} {ds : List SNode}, DynNodes tc keys ds → (∀ k ∈ keys, k.conn ≠ "IN_stream") →
      ∀ {acc : List (Sym × Val)} {cst : CSt}, STree.ListExec tc full gb (pendingSc acc) (ds ++ rest) cst r →
      (∃ e, r = .err e ∧ dynReadAcc ctx cst.ρ acc keys cst.st = .err e) ∨
      ∃ syms y, dynReadAcc ctx cst.ρ acc keys cst.st = .ok (syms, y) ∧ y.vals = [] ∧
        STree.ListExec tc full gb (pendingSc syms) rest (cst.withSt y) r
  | [], [], .nil, _, acc, cst, hx => .inr ⟨acc, cst.st, rfl, rfl, by rw [CSt.withSt_st]; exact hx⟩
  | k :: ks, n :: ns, .cons ⟨m, hn, hk⟩ hns, hnot, acc, cst, hx => by
    subst hn
    have hk' : k.conn ≠ "IN_stream" := hnot k List.mem_cons_self
    -- the node's step
    have hstep : ∀ res, STree.NodeExec tc (pendingSc acc) (.dynScopeCopy k.conn m) cst res →
        res = (readScalarData ctx cst.ρ k.data k.sub k.oob cst.st >>= fun (p : Option Val × St) =>
          match p.1 with
          | some v => Res.ok ((pendingSc (acc ++ [(k.conn, v)]) : SibCtx), TCtl.normal, cst.withSt p.2)
          | none => Res.ok (pendingSc acc, TCtl.normal, cst.withSt p.2)) := by
      intro res h
      cases h with
      | pendingErr hc _ => simp [SNode.consumesPending, SNode.isDynScopeCopy] at hc
      | leaf _ h' =>
        simp only [stepLeaf, Option.some.injEq] at h'
        rw [← h']
        rw [← hk]
        simp only [execDynScopeCopy, hk', ↓reduceIte, TCtx.inKey]
        rcases hid : tc.inData m with ⟨d, sub⟩
        simp only
        rw [Ctx.leaf_eq he readScalarData (fun _ => rfl)]
        cases readScalarData ctx cst.ρ d sub m.allowOob cst.st with
        | err e => rfl
        | ok p =>
          obtain ⟨v?, y⟩ := p
          cases v? <;> rfl
      | loop _ hl _ => simp [SNode.loop?] at hl
    simp only [List.cons_append] at hx
    cases hx with
    | headErr h =>
      have := hstep _ h
      cases hrd : readScalarData ctx cst.ρ k.data k.sub k.oob cst.st with
      | err e =>
        rw [hrd] at this; cases this
        exact .inl ⟨_, rfl, by simp only [dynReadAcc, hrd]; rfl⟩
      | ok p =>
        obtain ⟨v?, y⟩ := p
        rw [hrd] at this
        cases v? <;> cases this
    | step h hr =>
      have := hstep _ h
      cases hrd : readScalarData ctx cst.ρ k.data k.sub k.oob cst.st with
      | err e => rw [hrd] at this; cases this
      | ok p =>
        obtain ⟨v?, y⟩ := p
        rw [hrd] at this
        have hyv : y.vals = [] := by rw [readScalarData_vals hrd]; rfl
        have hρ : (cst.withSt y).ρ = cst.ρ := rfl
        have hst : (cst.withSt y).st = y := by
          obtain ⟨m', l', v'⟩ := y
          simp only at hyv
          simp [CSt.withSt, CSt.st, hyv]
        cases v? with
        | some v =>
          simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this
          obtain ⟨rfl, -, rfl⟩ := this
          rcases dscopies_run he hns (fun k' hk'' => hnot k' (List.mem_cons_of_mem _ hk'')) hr with
            ⟨e, rfl, hd⟩ | ⟨syms, y', hd, hyv', hx'⟩
          · rw [hρ, hst] at hd
            exact .inl ⟨e, rfl, by simp only [dynReadAcc, hrd, Res.ok_bind]; exact hd⟩
          · rw [hρ, hst] at hd
            refine .inr ⟨syms, y', by simp only [dynReadAcc, hrd, Res.ok_bind]; exact hd, hyv', ?_⟩
            have : (cst.withSt y).withSt y' = cst.withSt y' := rfl
            rwa [this] at hx'
        | none =>
          simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this
          obtain ⟨rfl, -, rfl⟩ := this
          rcases dscopies_run he hns (fun k' hk'' => hnot k' (List.mem_cons_of_mem _ hk'')) hr with
            ⟨e, rfl, hd⟩ | ⟨syms, y', hd, hyv', hx'⟩
          · rw [hρ, hst] at hd
            have hd' : dynReadAcc ctx cst.ρ acc (k :: ks) cst.st = .err e := by
              simp only [dynReadAcc, hrd, Res.ok_bind, Option.map_none, Option.toList_none, List.append_nil]
              exact hd
            exact .inl ⟨e, rfl, hd'⟩
          · rw [hρ, hst] at hd
            have hd' : dynReadAcc ctx cst.ρ acc (k :: ks) cst.st = .ok (syms, y') := by
              simp only [dynReadAcc, hrd, Res.ok_bind, Option.map_none, Option.toList_none, List.append_nil]
              exact hd
            refine .inr ⟨syms, y', hd', hyv', ?_⟩
            have : (cst.withSt y).withSt y' = cst.withSt y' := rfl
            rwa [this] at hx'
    | jump h hc =>
      have := hstep _ h
      cases hrd : readScalarData ctx cst.ρ k.data k.sub k.oob cst.st with
      | err e => rw [hrd] at this; cases this
      | ok p =>
        obtain ⟨v?, y⟩ := p
        rw [hrd] at this
        cases v? with
        | some v =>
          simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this
          rw [this.2.1] at hc; cases hc
        | none =>
          simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this
          rw [this.2.1] at hc; cases hc
    | gotoResume h _ _ =>
      have := hstep _ h
      cases hrd : readScalarData ctx cst.ρ k.data k.sub k.oob cst.st with
      | err e => rw [hrd] at this; cases this
      | ok p =>
        obtain ⟨v?, y⟩ := p
        rw [hrd] at this
        cases v? <;> simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this <;> cases this.2.1
    | gotoEscape h _ =>
      have := hstep _ h
      cases hrd : readScalarData ctx cst.ρ k.data k.sub k.oob cst.st with
      | err e => rw [hrd] at this; cases this
      | ok p =>
        obtain ⟨v?, y⟩ := p
        rw [hrd] at this
        cases v? <;> simp only [Res.ok_bind, Res.ok.injEq, Prod.mk.injEq] at this <;> cases this.2.1

/-- **Map set-up with dynamic inputs** in the SDFG. -/
theorem mapSetup_dyn {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {mp : MapNode} {keys : List InKey}
    (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (st : St) :
    mapSetup ctx ρ s n mp st = (dynReadAcc ctx ρ [] keys st >>= fun p =>
      (ctx.rangePoints (Valuation.setAll ρ p.1) p.2.mem mp.range).map fun pts => (Valuation.setAll ρ p.1, p.2, pts)) := by
  unfold mapSetup
  rw [bindDynamicInputs_eq hkeys hok]
  cases dynReadAcc ctx ρ [] keys st with
  | err e => rfl
  | ok p =>
    simp only [Res.map, Res.ok_bind]
    cases ctx.rangePoints (Valuation.setAll ρ p.1) p.2.mem mp.range <;> rfl

/-- **Map set-up with pending dynamic inputs** in the tree. -/
theorem mapSetupT_dyn {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL) {mp : MapNode}
    (syms : List (Sym × Val)) (cst : CSt) :
    mapSetupT tc (pendingSc syms) mp cst =
      (ctx.rangePoints (Valuation.setAll cst.ρ syms) cst.mem mp.range).map fun pts => (Valuation.setAll cst.ρ syms, pts) := by
  simp only [mapSetupT, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  rw [Ctx.leaf_eq he Ctx.rangePoints (fun _ => rfl)]
  cases ctx.rangePoints (Valuation.setAll cst.ρ syms) cst.mem mp.range <;> rfl

/-! ## Consume scopes -/

/-- The stream element an `IN_stream` copy designates (`execDynScopeCopy`). -/
def inStreamPt (ctx : Ctx) (ρ : Valuation) (mem : Mem) (m : Memlet) : Res Idx :=
  if ctx.isStream m.data then streamPoint ctx ρ mem m.data m.dataSubset
  else .err (.malformed s!"IN_stream of non-stream '{m.data}'")

/-- The consumed queue and the number of processing elements of a consume
scope on the element `p` of stream `d`. -/
def consumeTail (ctx : Ctx) (ρ : Valuation) (mem : Mem) (d : Name) (c : ConsumeNode) (p : Idx) : Res (Loc × Int) := do
  let l ← ctx.loc ρ mem d p
  let np ← ctx.evalIntR ρ mem c.numPes
  pure (mem.resolve l, np)

/-- A pending `IN_stream`, no symbols. -/
abbrev streamSc (d : Name) (p : Idx) : SibCtx := { pending := { stream := some (d, p) } }

/-- A tree list starting with an `IN_stream` copy. -/
theorem instream_inv {tc : TCtx} {full : List SNode} {gb : Bool} {m : Memlet} {rest : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (hx : STree.ListExec tc full gb {} (.dynScopeCopy "IN_stream" m :: rest) cst r) :
    (∃ e, inStreamPt tc.ctx cst.ρ cst.mem m = .err e ∧ r = .err e) ∨
    ∃ p, inStreamPt tc.ctx cst.ρ cst.mem m = .ok p ∧ STree.ListExec tc full gb (streamSc m.data p) rest cst r := by
  have hstep : ∀ res, STree.NodeExec tc {} (.dynScopeCopy "IN_stream" m) cst res →
      res = (inStreamPt tc.ctx cst.ρ cst.mem m).map fun p => (streamSc m.data p, TCtl.normal, cst) := by
    intro res h
    cases h with
    | pendingErr hc _ => simp [SNode.consumesPending, SNode.isDynScopeCopy] at hc
    | leaf _ h' =>
      simp only [stepLeaf, Option.some.injEq] at h'
      rw [← h']
      simp only [execDynScopeCopy, ↓reduceIte, inStreamPt]
      split
      · have hm : cst.st.mem = cst.mem := rfl
        rw [hm]
        cases streamPoint tc.ctx cst.ρ cst.mem m.data m.dataSubset with
        | err _ => rfl
        | ok p => show Res.ok (_, _, cst.withSt cst.st) = _; rw [CSt.withSt_st]; rfl
      · rfl
    | loop _ hl _ => simp [SNode.loop?] at hl
  cases hx with
  | headErr h =>
    have := hstep _ h
    cases hp : inStreamPt tc.ctx cst.ρ cst.mem m with
    | err e => rw [hp] at this; cases this; exact .inl ⟨_, rfl, rfl⟩
    | ok p => rw [hp] at this; cases this
  | step h hr =>
    have := hstep _ h
    cases hp : inStreamPt tc.ctx cst.ρ cst.mem m with
    | err e => rw [hp] at this; cases this
    | ok p =>
      rw [hp] at this
      simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at this
      obtain ⟨rfl, -, rfl⟩ := this
      exact .inr ⟨p, rfl, hr⟩
  | jump h hc =>
    have := hstep _ h
    cases hp : inStreamPt tc.ctx cst.ρ cst.mem m with
    | err e => rw [hp] at this; cases this
    | ok p =>
      rw [hp] at this
      simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at this
      rw [this.2.1] at hc; cases hc
  | gotoResume h _ _ =>
    have := hstep _ h
    cases hp : inStreamPt tc.ctx cst.ρ cst.mem m with
    | err e => rw [hp] at this; cases this
    | ok p => rw [hp] at this; simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at this; cases this.2.1
  | gotoEscape h _ =>
    have := hstep _ h
    cases hp : inStreamPt tc.ctx cst.ρ cst.mem m with
    | err e => rw [hp] at this; cases this
    | ok p => rw [hp] at this; simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at this; cases this.2.1

/-- A run of a consume scope node. -/
theorem consumeScope_node_inv {tc : TCtx} {sc : SibCtx} {c : ConsumeNode} {body : List SNode} {cst : CSt}
    {res : Res NodeOut} (h : STree.NodeExec tc sc (.consumeScope c body) cst res) :
    (∃ e, consumeSetupT tc sc c cst = .err e ∧ res = .err e) ∨
    (∃ ρ' d l np e, consumeSetupT tc sc c cst = .ok (ρ', d, l, np) ∧
      STree.ConsumeTasks tc ρ' c d body l np (tc.privateTransients body) cst (cst.mem.queues l) (.err e) ∧
      res = .err e) ∨
    ∃ ρ' d l np logs rest pushed, consumeSetupT tc sc c cst = .ok (ρ', d, l, np) ∧
      STree.ConsumeTasks tc ρ' c d body l np (tc.privateTransients body) cst (cst.mem.queues l)
        (.ok (logs, rest, pushed)) ∧
      res = dataflowOut (consumeFinishT tc cst l logs rest pushed) := by
  generalize hn : SNode.consumeScope c body = n at h
  cases h with
  | pendingErr hc _ => subst hn; simp [SNode.consumesPending, SNode.isDataflowScope] at hc
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | consumeSetupErr hs => cases hn; exact .inl ⟨_, hs, rfl⟩
  | consumeTasksErr hs ht => cases hn; exact .inr (.inl ⟨_, _, _, _, _, hs, ht, rfl⟩)
  | consumeScope hs ht => cases hn; exact .inr (.inr ⟨_, _, _, _, _, _, _, hs, ht, rfl⟩)
  | _ => cases hn

/-- A tree list starting with a consume scope. -/
theorem consume_inv {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {c : ConsumeNode}
    {body rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
    (hx : STree.ListExec tc full gb sc (.consumeScope c body :: rest) cst r) :
    (∃ e, consumeSetupT tc sc c cst = .err e ∧ r = .err e) ∨
    (∃ ρ' d l np e, consumeSetupT tc sc c cst = .ok (ρ', d, l, np) ∧
      STree.ConsumeTasks tc ρ' c d body l np (tc.privateTransients body) cst (cst.mem.queues l) (.err e) ∧
      r = .err e) ∨
    ∃ ρ' d l np logs rs pushed, consumeSetupT tc sc c cst = .ok (ρ', d, l, np) ∧
      STree.ConsumeTasks tc ρ' c d body l np (tc.privateTransients body) cst (cst.mem.queues l)
        (.ok (logs, rs, pushed)) ∧
      ((∃ e, consumeFinishT tc cst l logs rs pushed = .err e ∧ r = .err e) ∨
        ∃ cst', consumeFinishT tc cst l logs rs pushed = .ok cst' ∧ STree.ListExec tc full gb {} rest cst' r) := by
  have fin : ∀ {R : Res CSt} {res : Res NodeOut}, res = dataflowOut R →
      (∀ e, res = .err e → R = .err e) ∧
      (∀ sc ctl c, res = .ok (sc, ctl, c) → sc = {} ∧ ctl = .normal ∧ R = .ok c) := by
    intro R res hres
    subst hres
    refine ⟨fun e he => ?_, fun sc ctl c hc => ?_⟩
    · cases R <;> simp_all [dataflowOut, Res.map]
    · cases R <;> simp_all [dataflowOut, Res.map]
  cases hx with
  | headErr h =>
    rcases consumeScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', d, l, np, e, hs, ht, he⟩ |
      ⟨ρ', d, l, np, logs, rs, pushed, hs, ht, hres⟩
    · cases he; exact .inl ⟨_, hs, rfl⟩
    · cases he; exact .inr (.inl ⟨_, _, _, _, _, hs, ht, rfl⟩)
    · exact .inr (.inr ⟨_, _, _, _, _, _, _, hs, ht, .inl ⟨_, (fin hres).1 _ rfl, rfl⟩⟩)
  | step h hr =>
    rcases consumeScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', d, l, np, e, hs, ht, he⟩ |
      ⟨ρ', d, l, np, logs, rs, pushed, hs, ht, hres⟩
    · cases he
    · cases he
    · obtain ⟨rfl, -, hf⟩ := (fin hres).2 _ _ _ rfl
      exact .inr (.inr ⟨_, _, _, _, _, _, _, hs, ht, .inr ⟨_, hf, hr⟩⟩)
  | jump h hc =>
    rcases consumeScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', d, l, np, e, hs, ht, he⟩ |
      ⟨ρ', d, l, np, logs, rs, pushed, hs, ht, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, rfl, -⟩ := (fin hres).2 _ _ _ rfl
      cases hc
  | gotoResume h _ _ =>
    rcases consumeScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', d, l, np, e, hs, ht, he⟩ |
      ⟨ρ', d, l, np, logs, rs, pushed, hs, ht, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, h₂, -⟩ := (fin hres).2 _ _ _ rfl
      cases h₂
  | gotoEscape h _ =>
    rcases consumeScope_node_inv h with ⟨e, hs, he⟩ | ⟨ρ', d, l, np, e, hs, ht, he⟩ |
      ⟨ρ', d, l, np, logs, rs, pushed, hs, ht, hres⟩
    · cases he
    · cases he
    · obtain ⟨-, h₂, -⟩ := (fin hres).2 _ _ _ rfl
      cases h₂

/-- **Consume set-up without dynamic inputs** in the SDFG: the stream of
the `IN_stream` edge (failing when the tree's `IN_stream` copy fails). -/
theorem consumeSetup_in {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {c : ConsumeNode} {e : DEdge}
    {mIn : Memlet} (hkeys : s.dynKeys? n = some [])
    (hin : (s.inEdges n.id).find? (·.dstConn = some "IN_stream") = some e)
    (hsrc : s.srcEndpoint e = .data mIn.data mIn.dataSubset) (st : St) :
    (∀ e, inStreamPt ctx ρ st.mem mIn = .err e → ∃ e', consumeSetup ctx ρ s n c st = .err e') ∧
    ∀ p, inStreamPt ctx ρ st.mem mIn = .ok p →
      consumeSetup ctx ρ s n c st = (consumeTail ctx ρ st.mem mIn.data c p).map fun q => (ρ, st, q.1, q.2) := by
  unfold consumeSetup consumeStream
  rw [bindDynamicInputs_eq hkeys ⟨List.nodup_nil, fun k hk => by cases hk⟩]
  simp only [dynReadAcc, Res.map, Res.ok_bind, hin, hsrc, inStreamPt]
  rw [show Valuation.setAll ρ [] = ρ from rfl]
  split
  · refine ⟨fun e' hp => ⟨e', by rw [hp]; rfl⟩, fun p hp => ?_⟩
    rw [hp]
    simp only [consumeTail, Res.ok_bind]
    cases ctx.loc ρ st.mem mIn.data p with
    | err _ => rfl
    | ok l => cases ctx.evalIntR ρ st.mem c.numPes <;> rfl
  · exact ⟨fun _ _ => ⟨_, rfl⟩, fun p hp => by cases hp⟩

/-- **Consume set-up with a pending `IN_stream`** in the tree. -/
theorem consumeSetupT_stream (tc : TCtx) (c : ConsumeNode) (d : Name) (p : Idx) (cst : CSt) :
    consumeSetupT tc (streamSc d p) c cst = (consumeTail tc.ctx cst.ρ cst.mem d c p).map fun q => (cst.ρ, d, q.1, q.2) := by
  simp only [consumeSetupT, consumeTail]
  have hρ : Valuation.setAll cst.ρ [] = cst.ρ := rfl
  rw [hρ]
  cases tc.ctx.loc cst.ρ cst.mem d p with
  | err _ => rfl
  | ok l =>
    simp only [Res.ok_bind]
    cases tc.ctx.evalIntR cst.ρ cst.mem c.numPes <;> rfl

/-! ## Matching -/

/-- Prepend a node to the matched SDFG run. -/
theorem SeqOut.cons {ctx : Ctx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {ρ : Valuation} {st : St}
    {tc : TCtx} {full : List SNode} {gb : Bool} {rest : List SNode} {r : Res (TCtl × CSt)}
    (hn : s.node? id = some n)
    (h : (∃ e, e.isUB = true ∧ NodeExec ctx ρ s n st.forNode (.err e)) ∨
      (∃ e e', r = .err e ∧ NodeExec ctx ρ s n st.forNode (.err e')) ∨
      ∃ y, NodeExec ctx ρ s n st.forNode (.ok y) ∧ SeqOut ctx s o ρ (st.afterNode y) tc full gb rest r) :
    SeqOut ctx s (id :: o) ρ st tc full gb rest r := by
  rcases h with ⟨e, hu, hx⟩ | ⟨e, e', hr, hx⟩ | ⟨y, hx, ht⟩
  · exact .inl ⟨e, hu, .headErr hn hx⟩
  · exact .inr (.inl ⟨e, e', hr, .headErr hn hx⟩)
  · rcases ht with ⟨e, hu, ht⟩ | ⟨e, e', hr, ht⟩ | ⟨cst', st', tr, hl, ht, hρ, hrel⟩
    · exact .inl ⟨e, hu, .tailErr hn hx ht⟩
    · exact .inr (.inl ⟨e, e', hr, .tailErr hn hx ht⟩)
    · exact .inr (.inr ⟨cst', st', _, hl, .cons hn hx ht, hρ, hrel⟩)

theorem StRel.forNode_shift {st : St} {cst : CSt} (h : StRel st cst) : cst.st = st.forNode.shift cst.log := by
  obtain ⟨hm, -, hv⟩ := h
  simp [CSt.st, St.forNode, St.shift, Log.seq_empty, hm, hv]

theorem StRel.afterNode {st y : St} {cst cst' : CSt} (h : StRel st cst) (hy : y.vals = [])
    (hc : cst'.mem = y.mem ∧ cst'.log = cst.log.seq y.log) : StRel (st.afterNode y) cst' := by
  obtain ⟨-, hl, -⟩ := h
  exact ⟨hc.1.symm, by simp [St.afterNode, hl, hc.2], hy⟩

/-- The step of one copy. -/
def copyStep (ctx : Ctx) (ρ : Valuation) (x : St) (k : CopyKey) : Res St :=
  copyData ctx ρ x k.src k.ssub k.dst k.dsub k.wcr k.oob

theorem copyStep_keyCorr {ctx : Ctx} {ρ : Valuation} {x : St} {kT kS : CopyKey} (h : KeyCorr ctx kT kS) :
    copyStep ctx ρ x kT = copyStep ctx ρ x kS := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  unfold copyStep
  rw [h1, h2, h3, h4]
  exact copyData_subCorr h5 h6 _ _

theorem foldlM_keyCorr {ctx : Ctx} {ρ : Valuation} : ∀ {ksT ksS : List CopyKey},
    List.Forall₂ (KeyCorr ctx) ksT ksS → ∀ X, ksT.foldlM (copyStep ctx ρ) X = ksS.foldlM (copyStep ctx ρ) X
  | [], [], .nil, _ => rfl
  | _ :: _, kS :: _, .cons h hs, X => by
    simp only [List.foldlM_cons, copyStep_keyCorr h]
    cases copyStep ctx ρ X kS with
    | err _ => rfl
    | ok y => exact foldlM_keyCorr hs y

theorem CSt.st_withSt (cst : CSt) {y : St} (hy : y.vals = []) : (cst.withSt y).st = y := by
  obtain ⟨m, l, v⟩ := y
  simp only at hy
  simp [CSt.withSt, CSt.st, hy]

/-! ## Views -/

/-- What a view access node binds (`bindView`): the view, the container it
views and the subset of it. -/
structure ViewKey where
  d : Name
  base : Name
  sub : Option Subset
  deriving DecidableEq

/-- The key of the view access node `v` of container `d`. -/
def _root_.Dace.State.viewKey? (s : State) (v : DNode) (d : Name) : Option ViewKey :=
  match s.viewEdge? v with
  | none => none
  | some (e, isIn) =>
    let base? := if isIn then (s.node? (s.pathStart e).src).bind DNode.accessData?
                 else (s.node? (s.pathEnd e).dst).bind DNode.accessData?
    let sub := if isIn then State.leafSrcSubset e else State.leafDstSubset e
    base?.map fun base => ⟨d, base, sub⟩

/-- The keys of the view access nodes among `ids`, in order (`bindViews`). -/
def _root_.Dace.State.viewKeys (ctx : Ctx) (s : State) : List NodeId → Option (List ViewKey)
  | [] => some []
  | id :: ids =>
    match s.node? id with
    | some n =>
      match n.kind with
      | .access d =>
        if ctx.isView d then (s.viewKey? n d).bind fun k => (State.viewKeys ctx s ids).map (k :: ·)
        else State.viewKeys ctx s ids
      | _ => State.viewKeys ctx s ids
    | none => State.viewKeys ctx s ids

theorem _root_.Dace.State.viewKeys_withI (ctx : Ctx) (I₂ : Interp) (s : State) :
    ∀ ids : List NodeId, State.viewKeys { ctx with I := I₂ } s ids = State.viewKeys ctx s ids
  | [] => rfl
  | id :: ids => by
    have ih := State.viewKeys_withI ctx I₂ s ids
    simp only [State.viewKeys, ih]
    rfl

/-- Bind one view. -/
def viewStep (ctx : Ctx) (ρ : Valuation) (st : St) (k : ViewKey) : Res St := aliasTo ctx ρ st k.d k.base k.sub

theorem viewStep_vals {ctx : Ctx} {ρ : Valuation} {st st' : St} {k : ViewKey} (h : viewStep ctx ρ st k = .ok st') :
    st'.vals = st.vals := by
  obtain ⟨-, -, -, hv⟩ := aliasTo_steps (kv := true) ctx ρ k.d k.base k.sub st st' h
  exact hv rfl

theorem bindView_key {ctx : Ctx} {ρ : Valuation} {s : State} {v : DNode} {d : Name} {k : ViewKey}
    (hk : s.viewKey? v d = some k) (st : St) : bindView ctx ρ s v d st = viewStep ctx ρ st k := by
  unfold State.viewKey? at hk
  unfold bindView viewStep
  cases hv : s.viewEdge? v with
  | none => rw [hv] at hk; cases hk
  | some p =>
    obtain ⟨e, isIn⟩ := p
    rw [hv] at hk
    simp only at hk ⊢
    cases isIn <;> simp only [Bool.false_eq_true, ↓reduceIte] at hk ⊢
    · cases hb : (s.node? (s.pathEnd e).dst).bind DNode.accessData? with
      | none => rw [hb] at hk; cases hk
      | some base => rw [hb] at hk; simp only [Option.map_some, Option.some.injEq] at hk; subst hk; rfl
    · cases hb : (s.node? (s.pathStart e).src).bind DNode.accessData? with
      | none => rw [hb] at hk; cases hk
      | some base => rw [hb] at hk; simp only [Option.map_some, Option.some.injEq] at hk; subst hk; rfl

/-- The tree's view nodes of a scope, one per key and in order: `ViewNode`s
binding the same view to the same subset (`SubCorr`) of the same container. -/
def ViewsCorr (ctx : Ctx) (ks : List ViewKey) (vs : List SNode) : Prop :=
  List.Forall₂ (fun k n => ∃ m sd vd, n = .viewNode k.d k.base m sd vd ∧ m.data = k.base ∧
    SubCorr ctx k.base m.dataSubset k.sub) ks vs

theorem bindViews_keys {ctx : Ctx} {ρ : Valuation} {s : State} :
    ∀ {ids : List NodeId} {ks : List ViewKey}, s.viewKeys ctx ids = some ks →
      ∀ st, bindViews ctx ρ s ids st = ks.foldlM (viewStep ctx ρ) st
  | [], ks, h, st => by simp only [State.viewKeys, Option.some.injEq] at h; subst h; rfl
  | id :: ids, ks, h, st => by
    unfold bindViews
    simp only [List.foldlM_cons]
    unfold State.viewKeys at h
    cases hn : s.node? id with
    | none =>
      rw [hn] at h
      simp only
      exact bindViews_keys h st
    | some n =>
      rw [hn] at h
      simp only at h ⊢
      cases hk : n.kind with
      | access d =>
        rw [hk] at h
        simp only at h ⊢
        cases hv : ctx.isView d with
        | false => rw [hv] at h; simp only [Bool.false_eq_true, ↓reduceIte]; exact bindViews_keys h st
        | true =>
          rw [hv] at h
          simp only [↓reduceIte] at h ⊢
          cases hvk : s.viewKey? n d with
          | none => rw [hvk] at h; cases h
          | some k =>
            rw [hvk] at h
            simp only [Option.bind_some, Option.map_eq_some_iff] at h
            obtain ⟨ks', hks', rfl⟩ := h
            rw [bindView_key hvk, List.foldlM_cons]
            cases viewStep ctx ρ st k with
            | err e => rfl
            | ok st₁ => exact bindViews_keys hks' st₁
      | _ =>
        rw [hk] at h
        simp only at h ⊢
        exact bindViews_keys h st

/-- No view access node is a direct child of a scope (views are bound only
at the top level of a state, `ScopeViewsFree`). -/
def _root_.Dace.State.scopeViewsFree? (ctx : Ctx) (s : State) : Bool :=
  s.nodes.all fun c => match c.scopeEntry with
    | some _ => match s.node? c.id with
      | some n => match n.kind with
        | .access d => !ctx.isView d
        | _ => true
      | none => true
    | none => true

theorem bindViews_skip {ctx : Ctx} {ρ : Valuation} {s : State} :
    ∀ {ids : List NodeId}, (∀ id ∈ ids, ∀ n, s.node? id = some n → ∀ d, n.kind = .access d → ctx.isView d = false) →
      ∀ st, bindViews ctx ρ s ids st = .ok st
  | [], _, st => rfl
  | id :: ids, h, st => by
    unfold bindViews
    simp only [List.foldlM_cons]
    have ih := bindViews_skip (ρ := ρ) (fun i hi => h i (List.mem_cons_of_mem _ hi)) st
    unfold bindViews at ih
    cases hn : s.node? id with
    | none => simpa using ih
    | some n =>
      simp only
      cases hk : n.kind with
      | access d =>
        simp only [h id List.mem_cons_self n hn d hk, Bool.false_eq_true, ↓reduceIte]
        simpa using ih
      | _ => simpa using ih

theorem _root_.Dace.State.scopeViewsFree?_sound {ctx : Ctx} {s : State} (h : s.scopeViewsFree? ctx = true) :
    ∀ (entry : NodeId) (ρ : Valuation) (st : St),
      bindViews ctx ρ s (s.viewOrder ctx (s.directChildren (some entry))) st = .ok st := by
  intro entry ρ st
  refine bindViews_skip (fun id hid' n hn d hk => ?_) st
  have hid := State.viewOrder_sub hid'
  simp only [State.directChildren, State.children, List.mem_map, List.mem_filter] at hid
  obtain ⟨c, ⟨hc, hce⟩, rfl⟩ := hid
  have := List.all_eq_true.1 h c hc
  simp only [decide_eq_true_eq] at hce
  simp only [hce, hn, hk, Bool.not_eq_true'] at this
  exact this

section Matched

variable {ctx : Ctx} {tc : TCtx} {s : State}
  (h : ∀ (entry : NodeId) (ρ : Valuation) (st : St),
    bindViews ctx ρ s (s.viewOrder ctx (s.directChildren (some entry))) st = .ok st)
  (he : tc.ctx.eraseL = ctx.eraseL)

include he in
theorem I_eq : tc.ctx.I = ctx.I := by
  have := congrArg Ctx.I he
  simpa [Ctx.eraseL] using this

include he in
/-- The tree runs copy nodes one after the other: the fold of `copyData`
over their keys. -/
theorem copies_run {full : List SNode} {gb : Bool} :
    ∀ (cps : List (Name × Memlet)) {more : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
      STree.ListExec tc full gb {} (cps.map (fun p => SNode.copyNode p.1 p.2) ++ more) cst r →
      (∃ e, r = .err e ∧ (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) cst.st = .err e) ∨
      ∃ cst', STree.ListExec tc full gb {} more cst' r ∧
        (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) cst.st = .ok cst'.st ∧ cst'.ρ = cst.ρ
  | [], _, cst, _, hx => .inr ⟨cst, hx, rfl, rfl⟩
  | p :: cps, more, cst, r, hx => by
    simp only [List.map_cons, List.cons_append] at hx
    have hstep : stepLeaf tc {} (.copyNode p.1 p.2) cst =
        some (leafOut (cst.lift (execCopyT tc cst.ρ p.1 p.2 cst.st))) := rfl
    rw [execCopyT_eq he] at hstep
    rcases leaf_inv hstep hx with ⟨e, hR, rfl⟩ | ⟨cst₁, hR, hx₁⟩
    · refine .inl ⟨e, rfl, ?_⟩
      simp only [CSt.lift] at hR
      simp only [List.map_cons, List.foldlM_cons, copyStep]
      cases hc : copyData ctx cst.ρ cst.st (tc.copyKey p.1 p.2).src (tc.copyKey p.1 p.2).ssub (tc.copyKey p.1 p.2).dst
          (tc.copyKey p.1 p.2).dsub (tc.copyKey p.1 p.2).wcr (tc.copyKey p.1 p.2).oob with
      | ok y => rw [hc] at hR; cases hR
      | err e' => rw [hc] at hR; cases hR; rfl
    · simp only [CSt.lift] at hR
      cases hc : copyData ctx cst.ρ cst.st (tc.copyKey p.1 p.2).src (tc.copyKey p.1 p.2).ssub (tc.copyKey p.1 p.2).dst
          (tc.copyKey p.1 p.2).dsub (tc.copyKey p.1 p.2).wcr (tc.copyKey p.1 p.2).oob with
      | err e' => rw [hc] at hR; cases hR
      | ok y =>
        rw [hc] at hR
        simp only [Res.map, Res.ok.injEq] at hR
        subst hR
        have hy : y.vals = [] := (copyData_vals hc).trans rfl
        have hρ : (cst.withSt y).ρ = cst.ρ := rfl
        rcases copies_run cps hx₁ with ⟨e, hr, hf⟩ | ⟨cst', hx', hf, hρ'⟩
        · refine .inl ⟨e, hr, ?_⟩
          simp only [List.map_cons, List.foldlM_cons, copyStep]
          rw [hc, Res.ok_bind, ← CSt.st_withSt cst hy, ← hρ]
          exact hf
        · refine .inr ⟨cst', hx', ?_, hρ'.trans hρ⟩
          simp only [List.map_cons, List.foldlM_cons, copyStep]
          rw [hc, Res.ok_bind, ← CSt.st_withSt cst hy, ← hρ]
          exact hf

omit he in
/-- The tree runs view nodes one after the other: the fold of `aliasTo` over
their keys. -/
theorem views_run (he' : tc.ctx.eraseL = ctx.eraseL) (hcons : tc.consumed = none) {full : List SNode} {gb : Bool} :
    ∀ {ks : List ViewKey} {vs : List SNode} {more : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
      ViewsCorr ctx ks vs → STree.ListExec tc full gb {} (vs ++ more) cst r →
      (∃ e, r = .err e ∧ ks.foldlM (viewStep ctx cst.ρ) cst.st = .err e) ∨
      ∃ cst', STree.ListExec tc full gb {} more cst' r ∧ ks.foldlM (viewStep ctx cst.ρ) cst.st = .ok cst'.st ∧
        cst'.ρ = cst.ρ
  | [], [], _, cst, _, .nil, hx => .inr ⟨cst, hx, rfl, rfl⟩
  | k :: ks, v :: vs, more, cst, r, .cons ⟨m, sd, vd, hv, hmd, hms⟩ hvs, hx => by
    subst hv
    simp only [List.cons_append] at hx
    have hstep : stepLeaf tc {} (.viewNode k.d k.base m sd vd) cst =
        some (leafOut (cst.lift (viewStep ctx cst.ρ cst.st k))) := by
      simp only [stepLeaf, execViewT, hmd, ↓reduceIte, viewStep, TCtx.inData_none hcons]
      rw [Ctx.leaf_eq he' (fun c => aliasTo c cst.ρ cst.st k.d k.base m.dataSubset) (fun _ => rfl),
        aliasTo_subCorr hms]
    rcases leaf_inv hstep hx with ⟨e, hR, rfl⟩ | ⟨cst₁, hR, hx₁⟩
    · refine .inl ⟨e, rfl, ?_⟩
      simp only [CSt.lift] at hR
      simp only [List.foldlM_cons]
      cases hc : viewStep ctx cst.ρ cst.st k with
      | ok y => rw [hc] at hR; cases hR
      | err e' => rw [hc] at hR; cases hR; rfl
    · simp only [CSt.lift] at hR
      cases hc : viewStep ctx cst.ρ cst.st k with
      | err e' => rw [hc] at hR; cases hR
      | ok y =>
        rw [hc] at hR
        simp only [Res.map, Res.ok.injEq] at hR
        subst hR
        have hy : y.vals = [] := (viewStep_vals hc).trans rfl
        have hρ : (cst.withSt y).ρ = cst.ρ := rfl
        rcases views_run he' hcons hvs hx₁ with ⟨e, hr, hf⟩ | ⟨cst', hx', hf, hρ'⟩
        · refine .inl ⟨e, hr, ?_⟩
          simp only [List.foldlM_cons]
          rw [hc, Res.ok_bind, ← CSt.st_withSt cst hy, ← hρ]
          exact hf
        · refine .inr ⟨cst', hx', ?_, hρ'.trans hρ⟩
          simp only [List.foldlM_cons]
          rw [hc, Res.ok_bind, ← CSt.st_withSt cst hy, ← hρ]
          exact hf

/-- One SDFG leaf node that is one tree leaf. -/
theorem leaf_node {full : List SNode} {gb : Bool} {id : NodeId} {n : DNode} {o : List NodeId} {nT : SNode}
    {more rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St}
    (hn : s.node? id = some n) (hsimp : n.isSimple = true) (hout : n.isCode = true → s.OutsData n)
    (hstep : stepLeaf tc {} nT cst = some (leafOut (cst.lift (execSimple ctx cst.ρ s n cst.st))))
    (hx : STree.ListExec tc full gb {} (nT :: more) cst r) (hrel : StRel st cst)
    (ih : ∀ {cst' : CSt} {st' : St}, STree.ListExec tc full gb {} more cst' r → StRel st' cst' →
      cst'.ρ = cst.ρ → SeqOut ctx s o cst.ρ st' tc full gb rest r) :
    SeqOut ctx s (id :: o) cst.ρ st tc full gb rest r := by
  have hn' := List.mem_of_find?_eq_some hn
  rw [hrel.forNode_shift, execSimple_shift] at hstep
  apply SeqOut.cons hn
  cases hy : execSimple ctx cst.ρ s n st.forNode with
  | err e =>
    rw [hy] at hstep
    rcases leaf_inv hstep hx with ⟨e', hR, rfl⟩ | ⟨cst₁, hR, -⟩
    · simp only [Res.map, CSt.lift, Res.err.injEq] at hR
      subst hR
      exact .inr (.inl ⟨_, _, rfl, hy ▸ NodeExec.simple hsimp⟩)
    · cases hR
  | ok y =>
    rw [hy] at hstep
    have hyv : y.vals = [] := by
      obtain ⟨-, -, -, hv⟩ := execSimple_steps (kv := true) ctx cst.ρ s n (fun _ hc => hout hc) st.forNode y hy
      rw [hv rfl]; simpa [St.forNode] using hrel.2.2
    rcases leaf_inv hstep hx with ⟨e', hR, -⟩ | ⟨cst₁, hR, hx₁⟩
    · cases hR
    · simp only [Res.map, CSt.lift, Res.ok.injEq] at hR
      subst hR
      exact .inr (.inr ⟨y, hy ▸ NodeExec.simple hsimp, ih hx₁ (hrel.afterNode hyv ⟨rfl, rfl⟩) rfl⟩)

omit he in
/-- **The iterations of a map**, given one iteration: the tree's iterations,
each matched by an SDFG iteration whose log without the SDFG's private
transients is the tree's without the tree's. -/
theorem iter_matched_of {n : DNode} {mp : MapNode} {body : List SNode} {privS privT : List Name}
    {pre : CSt} {st₁ : St}
    (hbody : ∀ (i : Idx) {res : Res (TCtl × CSt)},
      STree.ListExec tc body false {} body (iterStartT privT pre (pre.ρ.bindIdx mp.params i)) res →
      (∃ e, e.isUB = true ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart privS st₁) (.err e)) ∨
      (∃ e e', res = .err e ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart privS st₁) (.err e')) ∨
      ∃ st', OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id) (s.directChildren (some n.id))
        (iterStart privS st₁) (.ok st') ∧ ∃ cst', res = .ok (.normal, cst') ∧
          st'.log.dropNames privS = cst'.log.dropNames privT) :
    ∀ {pts : List Idx} {rT : Res (List Log)},
      STree.IterExec tc mp.params body privT pre pts rT →
      (∃ e, e.isUB = true ∧ IterExec ctx pre.ρ s n.id mp.params privS st₁ pts (.err e)) ∨
      (∃ e e', rT = .err e ∧ IterExec ctx pre.ρ s n.id mp.params privS st₁ pts (.err e')) ∨
      ∃ logs, rT = .ok logs ∧ IterExec ctx pre.ρ s n.id mp.params privS st₁ pts (.ok logs) := by
  intro pts
  induction pts with
  | nil =>
    intro rT hi
    cases hi
    exact .inr (.inr ⟨[], rfl, .nil⟩)
  | cons i rest ih =>
    intro rT hi
    cases hi with
    | headErr hb hlog =>
      rcases hbody i hb with ⟨e', hu, hx⟩ | ⟨e', e'', rfl, hx⟩ | ⟨st', hx, cst', rfl, -⟩
      · exact .inl ⟨e', hu, .headErr hx⟩
      · cases hlog; exact .inr (.inl ⟨_, _, rfl, .headErr hx⟩)
      · cases hlog
    | tailErr hr =>
      rcases ih hr with ⟨e', hu, hy⟩ | ⟨e', e'', he', hy⟩ | ⟨logs, hl', -⟩
      · exact .inl ⟨e', hu, .tailErr hy⟩
      · exact .inr (.inl ⟨e', e'', he', .tailErr hy⟩)
      · cases hl'
    | cons hb hlog hr =>
      rcases hbody i hb with ⟨e', hu, hx⟩ | ⟨e', e'', rfl, hx⟩ | ⟨st', hx, cst', rfl, hlg⟩
      · exact .inl ⟨e', hu, .headErr hx⟩
      · cases hlog
      · rcases ih hr with ⟨e', hu, hy⟩ | ⟨e', e'', he', hy⟩ | ⟨logs', hl', hy⟩
        · exact .inl ⟨e', hu, .tailErr hy⟩
        · cases he'
        · cases hl'
          simp only [iterLog, Res.ok.injEq] at hlog
          subst hlog
          refine .inr (.inr ⟨_, rfl, ?_⟩)
          rw [← hlg]
          exact IterExec.cons hx hy

include h in
/-- **The iterations of a map**: the tree's iterations, each matched by the
SDFG's run of the scope body along the tree's order. -/
theorem iter_matched {n : DNode} {mp : MapNode} {oM full : List NodeId} {body : List SNode}
    (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
    (hdir : full.filter (s.directChildren (some n.id)).contains = oM)
    (hpriv : tc.privateTransients body = ctx.privateTransients s n.id)
    (ihb : ∀ {cst : CSt} {r : Res (TCtl × CSt)} {st : St}, STree.ListExec tc body false {} (body ++ []) cst r →
      StRel st cst → SeqOut ctx s oM cst.ρ st tc body false [] r)
    {pre : CSt} {st₁ : St} (hm : st₁.mem = pre.mem) (hv : st₁.vals = []) :
    ∀ {pts : List Idx} {rT : Res (List Log)},
      STree.IterExec tc mp.params body (tc.privateTransients body) pre pts rT →
      (∃ e, e.isUB = true ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e)) ∨
      (∃ e e', rT = .err e ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e')) ∨
      ∃ logs, rT = .ok logs ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs) := by
  have hrel : ∀ i, StRel (iterStart (ctx.privateTransients s n.id) st₁) (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) :=
    fun i => ⟨by simp only [iterStart, iterStartT, hm, hpriv], rfl, hv⟩
  have hp : ∀ ρ', bindViews ctx ρ' s (s.viewOrder ctx (s.directChildren (some n.id)))
      (iterStart (ctx.privateTransients s n.id) st₁) = .ok (iterStart (ctx.privateTransients s n.id) st₁) :=
    fun ρ' => h n.id ρ' _
  intro pts rT hi
  refine iter_matched_of (n := n) (fun i {res} hb => ?_) hi
  have hb' : STree.ListExec tc body false {} (body ++ []) (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) res := by
    rw [List.append_nil]; exact hb
  rcases ihb hb' (hrel i) with ⟨e, hu, hs⟩ | ⟨e, e', hr, hs⟩ | ⟨cst', st', tr, hl, hs, -, hst⟩
  · rw [← hdir] at hs
    exact .inl ⟨e, hu, .seqErr hto (hp _) hs⟩
  · rw [← hdir] at hs
    exact .inr (.inl ⟨e, e', hr, .seqErr hto (hp _) hs⟩)
  · rw [← hdir] at hs
    have hres := STree.ListExec_nil (by rfl) hl
    cases hf : s.hbFree tr with
    | false => exact .inl ⟨_, rfl, .hbRace hto (hp _) hs hf⟩
    | true => exact .inr (.inr ⟨st', .run hto (hp _) hs hf, cst', hres, by rw [hst.2.1, hpriv]⟩)

omit h in
include he in
theorem scopeFinishT_eq (cst : CSt) (logs : List Log) :
    scopeFinishT tc cst logs = (scopeFinish ctx cst.st logs).map cst.withSt := by
  unfold scopeFinishT
  rw [Ctx.leaf_eq he scopeFinish (fun _ => rfl)]

/-- An access or exit node executes its copies. -/
theorem execSimple_copyTarget {ctx : Ctx} {ρ : Valuation} {n : DNode} (hk : DNode.isCopyTarget n = true) (X : St) :
    execSimple ctx ρ s n X = execCopiesInto ctx ρ s n X := by
  unfold execSimple
  unfold DNode.isCopyTarget at hk
  split at hk <;> simp_all

theorem DNode.isSimple_of_copyTarget {n : DNode} (hk : DNode.isCopyTarget n = true) : n.isSimple = true := by
  unfold DNode.isCopyTarget at hk
  unfold DNode.isSimple
  split at hk <;> simp_all

omit h he in
/-- Idle nodes run as no-ops. -/
theorem idle_run {ρ : Valuation} : ∀ {xs : List NodeId}, IdleNodes ctx s xs → ∀ st : St,
    SeqExec ctx ρ s xs st (.ok (st, xs.map fun x => (x, Log.empty)))
  | [], _, _ => .nil
  | x :: xs, hxs, st => by
    obtain ⟨xn, hxn, hk, hsh, hcp⟩ := hxs x List.mem_cons_self
    have hx : NodeExec ctx ρ s xn st.forNode (.ok st.forNode) := by
      have := NodeExec.simple (ctx := ctx) (ρ := ρ) (s := s) (st := st.forNode) (DNode.isSimple_of_copyTarget hk)
      rwa [execSimple_copyTarget hk, execCopiesInto_eq ρ hsh, hcp] at this
    have hst : st.afterNode st.forNode = st := by
      cases st; simp [St.afterNode, St.forNode, Log.seq_empty]
    have hr : SeqExec ctx ρ s xs (st.afterNode st.forNode) (.ok (st, xs.map fun x => (x, Log.empty))) := by
      rw [hst]; exact idle_run (fun y hy => hxs y (List.mem_cons_of_mem _ hy)) st
    exact SeqExec.cons hxn hx hr

include h he in
/-- **One iteration of a map whose body is a nested SDFG**: the tree runs
the inlined inner region from the iteration's start; the SDFG calls the inner
SDFG from a memory that differs only by its per-call allocation, which the
simulation (`SDFGExec.sim`, `π = 1`) shows harmless; the call's log, without
the per-call transients, is the tree's iteration log. -/
theorem nested_body (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {n : DNode} {mp : MapNode} {full : List NodeId}
    {body : List SNode} {nid : NodeId} {nd : DNode} {inner : SDFG} {mapping : List (Sym × Expr)} {ni : Bool}
    {conns : List (Name × Name)}
    {xs : List NodeId}
    (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
    (hdir : full.filter (s.directChildren (some n.id)).contains = nid :: xs) (hxs : IdleNodes ctx s xs)
    (hnd : s.node? nid = some nd) (hkn : nd.kind = .nested inner mapping ni)
    (hb : nestedBindings ctx s nd = .ok conns) (hmap : NestedMapOk ctx inner mapping)
    (hcode : ∀ c ∈ nestedConns nd, (nestedEndpoint s nd c).isData = true)
    (hout : s.DataOuts nd)
    (hpriv : ∀ x, x ∈ tc.privateTransients body ↔
      x ∈ ctx.privateTransients s n.id ++ inner.callTransients.map (ctx.enter inner nd.id conns).cname)
    {cT : Ctx} (hcI : cT.I = tc.ctx.I) (hct : cT.g.top = inner.top)
    (hnames : Region.Names 1 (fun _ => True) (ctx.enter inner nd.id conns) cT inner.top)
    (hinner : ∀ (cst : CSt) (r : Res (TCtl × CSt)), STree.ListExec tc body false {} body cst r →
      NestedOut (RegionAt cT inner.top cst inner.top.start) r)
    {pre : CSt} {st₁ : St} (hm : st₁.mem = pre.mem) (i : Idx) {res : Res (TCtl × CSt)}
    (hbd : STree.ListExec tc body false {} body (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) res) :
    (∃ e, e.isUB = true ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) (.err e)) ∨
      (∃ e e', res = .err e ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) (.err e')) ∨
      ∃ st', OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id) (s.directChildren (some n.id))
        (iterStart (ctx.privateTransients s n.id) st₁) (.ok st') ∧ ∃ cst', res = .ok (.normal, cst') ∧
          st'.log.dropNames (ctx.privateTransients s n.id) = cst'.log.dropNames (tc.privateTransients body) := by
  have hp : bindViews ctx (pre.ρ.bindIdx mp.params i) s (s.viewOrder ctx (s.directChildren (some n.id)))
      (iterStart (ctx.privateTransients s n.id) st₁) = .ok (iterStart (ctx.privateTransients s n.id) st₁) :=
    h n.id _ _
  have herr : ∀ e, NodeExec ctx (pre.ρ.bindIdx mp.params i) s nd (iterStart (ctx.privateTransients s n.id) st₁).forNode
      (.err e) → OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id) (s.directChildren (some n.id))
        (iterStart (ctx.privateTransients s n.id) st₁) (.err e) := by
    intro e hx
    exact .seqErr hto hp (by rw [hdir]; exact .headErr hnd hx)
  have hokk : ∀ y, NodeExec ctx (pre.ρ.bindIdx mp.params i) s nd (iterStart (ctx.privateTransients s n.id) st₁).forNode
      (.ok y) → (∃ e, e.isUB = true ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) (.err e)) ∨
      OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id) (s.directChildren (some n.id))
        (iterStart (ctx.privateTransients s n.id) st₁) (.ok ((iterStart (ctx.privateTransients s n.id) st₁).afterNode y)) := by
    intro y hx
    have hs : SeqExec ctx (pre.ρ.bindIdx mp.params i) s (full.filter (s.directChildren (some n.id)).contains)
        (iterStart (ctx.privateTransients s n.id) st₁)
        (.ok ((iterStart (ctx.privateTransients s n.id) st₁).afterNode y,
          (nid, y.log) :: xs.map fun x => (x, Log.empty))) := by
      rw [hdir]; exact .cons hnd hx (idle_run hxs _)
    cases hf : s.hbFree ((nid, y.log) :: xs.map fun x => (x, Log.empty)) with
    | false => exact .inl ⟨_, rfl, .hbRace hto hp hs hf⟩
    | true => exact .inr (.run hto hp hs hf)
  have hset := nestedSetup_eq (ρ := pre.ρ.bindIdx mp.params i) (inner := inner) (mapping := mapping)
    (st := (iterStart (ctx.privateTransients s n.id) st₁).forNode) hb hcode
  rcases nestedValuation_id hmap (pre.ρ.bindIdx mp.params i) (iterStart (ctx.privateTransients s n.id) st₁).forNode.mem with
    hval | ⟨e, hval, hu⟩
  rotate_left
  · rw [hval] at hset
    exact .inl ⟨e, hu, herr e (.nestedSetupErr hkn hset)⟩
  rw [hval] at hset
  simp only [Res.bind] at hset
  have hmem : Mem.Sim 1 (fun _ => True) (fun _ => True)
      ((iterStart (ctx.privateTransients s n.id) st₁).forNode.mem.allocAll
        (inner.callTransients.map (ctx.enter inner nd.id conns).cname))
      (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)).mem := by
    simp only [St.forNode, iterStart, iterStartT, hm, Mem.allocAll_append]
    refine (Mem.Sim.refl pre.mem).allocAll_congr fun x => ?_
    simp only [Equiv.Perm.coe_one, List.map_id_fun, id_eq]
    exact (hpriv x).symm
  have hc : CtxSim 1 (fun _ => True) (ctx.enter inner nd.id conns) cT :=
    ⟨by rw [hcI, I_eq he]; rfl, hct.symm⟩
  have hw' : ∀ op v, cT.I.wcr op .undef v = .undef := by
    rw [hcI, I_eq he]; exact hw
  have hcst : CSt.mk (inner.withConstants (pre.ρ.bindIdx mp.params i))
      (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)).mem Log.empty =
      iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i) := by
    simp only [hmap.consts]; rfl
  rcases hinner _ _ hbd with ⟨e, hu, hR⟩ | ⟨e₀, e, rfl, hR⟩ | ⟨ctl, c, hctl, rfl, hR⟩
  · have hx : SDFGExec cT (pre.ρ.bindIdx mp.params i)
        (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)).mem inner (.err e) :=
      .err (by rw [hcst]; exact .start hR)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    obtain ⟨e', rfl, hu'⟩ := ResSim.err_inv hr
    refine .inl ⟨e', ?_, herr e' (.nestedInnerErr hkn hset hx₂)⟩
    rcases hu' with hu' | rfl
    exacts [hu', hu]
  · have hx : SDFGExec cT (pre.ρ.bindIdx mp.params i)
        (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)).mem inner (.err e) :=
      .err (by rw [hcst]; exact .start hR)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    obtain ⟨e', rfl, hu'⟩ := ResSim.err_inv hr
    rcases hu' with hu' | rfl
    · exact .inl ⟨e', hu', herr e' (.nestedInnerErr hkn hset hx₂)⟩
    · exact .inr (.inl ⟨e₀, e', rfl, herr e' (.nestedInnerErr hkn hset hx₂)⟩)
  · have hx : SDFGExec cT (pre.ρ.bindIdx mp.params i)
        (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)).mem inner (.ok (c.ρ, c.mem, c.log)) :=
      .run (by rw [hcst]; exact .start hR) (by rcases hctl with rfl | rfl <;> rfl)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    rcases ResSim.ok_cases hr with ⟨e', rfl, hu'⟩ | ⟨⟨ρ₂, m₂, g₂⟩, rfl, -, -, hg₂, -⟩
    · exact .inl ⟨e', hu', herr e' (.nestedInnerErr hkn hset hx₂)⟩
    · have hnode := NodeExec.nested hkn hset hx₂
      rw [nestedFinish_data hout] at hnode
      rcases hokk _ hnode with ⟨e', hu', hy⟩ | hy
      · exact .inl ⟨e', hu', hy⟩
      · refine .inr (.inr ⟨_, hy, c, rfl, ?_⟩)
        simp only at hg₂
        rw [Log.ren_one] at hg₂
        simp only [St.afterNode, iterStart, St.forNode, hg₂, Log.empty_seq, Log.dropNames_dropNames]
        exact Log.dropNames_congr fun x => by rw [hpriv x, List.mem_append, List.mem_append, or_comm]

/-- An exact simulation under the identity renaming is equality. -/
theorem Mem.Sim.eq_one {m₂ m₁ : Mem} (hm : Mem.Sim 1 (fun _ => True) (fun _ => False) m₂ m₁) : m₂ = m₁ := by
  obtain ⟨ha, -, hc, hq⟩ := hm
  have hl : ∀ l : Loc, l.ren 1 = l := fun l => rfl
  cases m₂ with
  | mk c₂ q₂ a₂ =>
    cases m₁ with
    | mk c₁ q₁ a₁ =>
      simp only [Mem.mk.injEq]
      refine ⟨funext fun l => ?_, funext fun l => ?_, funext fun n => ?_⟩
      · rcases hc l trivial with ⟨h, -⟩ | h
        · exact h.elim
        · simpa [hl] using h
      · simpa [hl] using hq l trivial
      · have := ha n trivial
        simp only [Equiv.Perm.coe_one, id_eq] at this
        rw [this]
        cases a₁ n with
        | none => rfl
        | some b => rfl

include he in
/-- **A nested call outside map scopes**: the tree's run of the inlined region
from the call's memory (`hR`, matched by the inner region in the tree-named
context `cT`) is matched by the call, which ends in the inner memory and log
(no per-call transients, data connectors only). -/
theorem nested_call (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {nd : DNode} {inner : SDFG}
    {mapping : List (Sym × Expr)} {ni : Bool} {conns : List (Name × Name)} (hkn : nd.kind = .nested inner mapping ni)
    (hb : nestedBindings ctx s nd = .ok conns) (hmap : NestedMapOk ctx inner mapping)
    (hcode : ∀ c ∈ nestedConns nd, (nestedEndpoint s nd c).isData = true)
    (hout : s.DataOuts nd) (hctr : inner.callTransients = [])
    {cT : Ctx} (hcI : cT.I = tc.ctx.I) (hct : cT.g.top = inner.top)
    (hnames : Region.Names 1 (fun _ => True) (ctx.enter inner nd.id conns) cT inner.top)
    (ρ : Valuation) (st₀ : St) {res : Res (TCtl × CSt)}
    (hR : NestedOut (RegionAt cT inner.top { ρ, mem := st₀.mem } inner.top.start) res) :
    (∃ e, e.isUB = true ∧ NodeExec ctx ρ s nd st₀ (.err e)) ∨
    (∃ e e', res = .err e ∧ NodeExec ctx ρ s nd st₀ (.err e')) ∨
    ∃ c, res = .ok (.normal, c) ∧ NodeExec ctx ρ s nd st₀ (.ok { mem := c.mem, log := st₀.log.seq c.log, vals := st₀.vals }) := by
  have hset := nestedSetup_eq (ρ := ρ) (inner := inner) (mapping := mapping) (st := st₀) hb hcode
  rcases nestedValuation_id hmap ρ st₀.mem with hval | ⟨e, hval, hu⟩
  rotate_left
  · rw [hval] at hset
    exact .inl ⟨e, hu, .nestedSetupErr hkn hset⟩
  rw [hval] at hset
  simp only [Res.bind, hctr, List.map_nil] at hset
  have hst₀ : ({ st₀ with mem := st₀.mem.allocAll [] } : St) = st₀ := by cases st₀; rfl
  rw [hst₀] at hset
  have hmem : Mem.Sim 1 (fun _ => True) (fun _ => False) st₀.mem st₀.mem := Mem.Sim.refl _
  have hc : CtxSim 1 (fun _ => True) (ctx.enter inner nd.id conns) cT :=
    ⟨by rw [hcI, I_eq he]; rfl, hct.symm⟩
  have hw' : ∀ op v, cT.I.wcr op .undef v = .undef := by
    rw [hcI, I_eq he]; exact hw
  have hcst : ({ ρ := inner.withConstants ρ, mem := st₀.mem } : CSt) = { ρ, mem := st₀.mem } := by
    simp only [hmap.consts]
  rcases hR with ⟨e, hu, hR⟩ | ⟨e₀, e, rfl, hR⟩ | ⟨ctl, c, hctl, rfl, hR⟩
  · have hx : SDFGExec cT ρ st₀.mem inner (.err e) := .err (by rw [hcst]; exact .start hR)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    obtain ⟨e', rfl, hu'⟩ := ResSim.err_inv hr
    refine .inl ⟨e', ?_, .nestedInnerErr hkn hset hx₂⟩
    rcases hu' with hu' | rfl
    exacts [hu', hu]
  · have hx : SDFGExec cT ρ st₀.mem inner (.err e) := .err (by rw [hcst]; exact .start hR)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    obtain ⟨e', rfl, hu'⟩ := ResSim.err_inv hr
    rcases hu' with hu' | rfl
    · exact .inl ⟨e', hu', .nestedInnerErr hkn hset hx₂⟩
    · exact .inr (.inl ⟨e₀, e', rfl, .nestedInnerErr hkn hset hx₂⟩)
  · have hx : SDFGExec cT ρ st₀.mem inner (.ok (c.ρ, c.mem, c.log)) :=
      .run (by rw [hcst]; exact .start hR) (by rcases hctl with rfl | rfl <;> rfl)
    obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hc hw' hnames hx _ hmem
    rcases ResSim.ok_cases hr with ⟨e', rfl, hu'⟩ | ⟨⟨ρ₂, m₂, g₂⟩, rfl, -, hm₂, hg₂, -⟩
    · exact .inl ⟨e', hu', .nestedInnerErr hkn hset hx₂⟩
    · have hnode := NodeExec.nested hkn hset hx₂
      rw [nestedFinish_data hout] at hnode
      simp only at hm₂ hg₂
      rw [Mem.Sim.eq_one hm₂] at hnode
      rw [hg₂, Log.ren_one, Log.dropNames_nil'] at hnode
      exact .inr (.inr ⟨c, rfl, hnode⟩)

omit h in
include he in
theorem consumeCond_eq {ρ : Valuation} {c : ConsumeNode} {m₁ m₂ : Mem} (hm : m₁ = m₂) (l : Loc) (q : List Val) :
    consumeCond tc.ctx ρ c m₂ l q = consumeCond ctx ρ c m₁ l q := by
  subst hm
  exact Ctx.leaf_eq he (fun x => consumeCond x ρ c m₁ l q) (fun _ => rfl)

omit h in
include he in
/-- **The tasks of a consume scope**, given one task: the tree's tasks, each
matched by an SDFG task with the same log and deferred pushes. -/
theorem tasks_matched_of {n : DNode} {c : ConsumeNode} {d : Name} {body : List SNode} {privS privT : List Name}
    {ρ : Valuation} {l : Loc} {np : Int} {pre : CSt} {st₁ : St} (hm : st₁.mem = pre.mem)
    (hbody : ∀ (q : List Val) (p : Int) {res : Res (TCtl × CSt)},
      STree.ListExec (tc.consuming d) body false {} body
        (taskStartT tc d c l privT pre (Valuation.set ρ c.peIndex (.int p)) q) res →
      (∃ e, e.isUB = true ∧ OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (taskStart ctx s n.id c l privS st₁ q) (.err e)) ∨
      (∃ e e', res = .err e ∧ OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (taskStart ctx s n.id c l privS st₁ q) (.err e')) ∨
      ∃ st', OrderedExec ctx (Valuation.set ρ c.peIndex (.int p)) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (taskStart ctx s n.id c l privS st₁ q) (.ok st') ∧
        ∃ cst', res = .ok (.normal, cst') ∧ taskLog st' privS l = taskLog cst'.st privT l ∧
          taskPushed st' privS l = taskPushed cst'.st privT l) :
    ∀ {q : List Val} {rT : Res (List Log × List Val × List Val)},
      STree.ConsumeTasks tc ρ c d body l np privT pre q rT →
      (∃ e, e.isUB = true ∧ ConsumeTasks ctx ρ s n.id c l np privS st₁ q (.err e)) ∨
      (∃ e e', rT = .err e ∧ ConsumeTasks ctx ρ s n.id c l np privS st₁ q (.err e')) ∨
      ∃ res, rT = .ok res ∧ ConsumeTasks ctx ρ s n.id c l np privS st₁ q (.ok res)
  | _, _, .condErr hc => .inr (.inl ⟨_, _, rfl, .condErr ((consumeCond_eq he hm _ _).symm.trans hc)⟩)
  | _, _, .stop hc => .inr (.inr ⟨_, rfl, .stop ((consumeCond_eq he hm _ _).symm.trans hc)⟩)
  | _, _, .bodyErr hc hq hp hb ho => by
    have hc' := (consumeCond_eq he hm _ _).symm.trans hc
    rcases hbody _ _ hb with ⟨e', hu, hx⟩ | ⟨e', e'', rfl, hx⟩ | ⟨st', hx, cst', rfl, -, -⟩
    · exact .inl ⟨e', hu, .bodyErr hc' hq hp hx⟩
    · cases ho; exact .inr (.inl ⟨_, _, rfl, .bodyErr hc' hq hp hx⟩)
    · cases ho
  | _, _, .restErr hc hq hp hb ho hr => by
    have hc' := (consumeCond_eq he hm _ _).symm.trans hc
    rcases hbody _ _ hb with ⟨e', hu, hx⟩ | ⟨e', e'', rfl, hx⟩ | ⟨st', hx, cst', rfl, -, -⟩
    · exact .inl ⟨e', hu, .bodyErr hc' hq hp hx⟩
    · cases ho
    · rcases tasks_matched_of hm hbody hr with ⟨e', hu, hy⟩ | ⟨e', e'', he', hy⟩ | ⟨res, hres, -⟩
      · exact .inl ⟨e', hu, .restErr hc' hq hp hx hy⟩
      · exact .inr (.inl ⟨e', e'', he', .restErr hc' hq hp hx hy⟩)
      · cases hres
  | _, _, .step hc hq hp hb ho hr => by
    have hc' := (consumeCond_eq he hm _ _).symm.trans hc
    rcases hbody _ _ hb with ⟨e', hu, hx⟩ | ⟨e', e'', rfl, hx⟩ | ⟨st', hx, cst', rfl, hlg, hpu⟩
    · exact .inl ⟨e', hu, .bodyErr hc' hq hp hx⟩
    · cases ho
    · rcases tasks_matched_of hm hbody hr with ⟨e', hu, hy⟩ | ⟨e', e'', he', hy⟩ | ⟨res, hres, hy⟩
      · exact .inl ⟨e', hu, .restErr hc' hq hp hx hy⟩
      · cases he'
      · cases hres
        simp only [taskOutcome, Res.ok.injEq, Prod.mk.injEq] at ho
        obtain ⟨rfl, rfl⟩ := ho
        refine .inr (.inr ⟨_, rfl, ?_⟩)
        rw [← hlg, ← hpu]
        exact .step hc' hq hp hx hy

include h he in
/-- **The tasks of a consume scope**: the tree's tasks, each matched by the
SDFG's run of the scope body along the tree's order. -/
theorem tasks_matched {n : DNode} {c : ConsumeNode} {d : Name} {oM full : List NodeId} {body : List SNode}
    (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
    (hdir : full.filter (s.directChildren (some n.id)).contains = oM)
    (hpriv : tc.privateTransients body = ctx.privateTransients s n.id) (hd : s.consumedStream? n.id = some d)
    (ihb : ∀ {cst : CSt} {r : Res (TCtl × CSt)} {st : St},
      STree.ListExec (tc.consuming d) body false {} (body ++ []) cst r →
      StRel st cst → SeqOut ctx s oM cst.ρ st (tc.consuming d) body false [] r)
    {ρ : Valuation} {l : Loc} {np : Int} {pre : CSt} {st₁ : St} (hm : st₁.mem = pre.mem) (hv : st₁.vals = []) :
    ∀ {q : List Val} {rT : Res (List Log × List Val × List Val)},
      STree.ConsumeTasks tc ρ c d body l np (tc.privateTransients body) pre q rT →
      (∃ e, e.isUB = true ∧ ConsumeTasks ctx ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.err e)) ∨
      (∃ e e', rT = .err e ∧ ConsumeTasks ctx ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.err e')) ∨
      ∃ res, rT = .ok res ∧ ConsumeTasks ctx ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.ok res) := by
  have hcn : tc.ctx.cname = ctx.cname := by
    have := congrArg Ctx.cname he
    simpa [Ctx.eraseL] using this
  have hrel : ∀ ρ' q, StRel (taskStart ctx s n.id c l (ctx.privateTransients s n.id) st₁ q)
      (taskStartT tc d c l (tc.privateTransients body) pre ρ' q) := by
    intro ρ' q
    refine ⟨?_, rfl, hv⟩
    simp only [taskStart, taskStartT, chunkMemName, TCtx.chunkMemName, hd, Option.map_some, hm, hpriv, hcn]
  have hp : ∀ ρ' q, bindViews ctx ρ' s (s.viewOrder ctx (s.directChildren (some n.id)))
      (taskStart ctx s n.id c l (ctx.privateTransients s n.id) st₁ q) =
        .ok (taskStart ctx s n.id c l (ctx.privateTransients s n.id) st₁ q) :=
    fun ρ' q => h n.id ρ' _
  intro q rT ht
  refine tasks_matched_of he (n := n) hm (fun q p {res} hb => ?_) ht
  have hb' : STree.ListExec (tc.consuming d) body false {} (body ++ [])
      (taskStartT tc d c l (tc.privateTransients body) pre (Valuation.set ρ c.peIndex (.int p)) q) res := by
    rw [List.append_nil]; exact hb
  rcases ihb hb' (hrel _ q) with ⟨e, hu, hs⟩ | ⟨e, e', hr, hs⟩ | ⟨cst', st', tr, hl, hs, -, hst⟩
  · rw [← hdir] at hs
    exact .inl ⟨e, hu, .seqErr hto (hp _ _) hs⟩
  · rw [← hdir] at hs
    exact .inr (.inl ⟨e, e', hr, .seqErr hto (hp _ _) hs⟩)
  · rw [← hdir] at hs
    have hres := STree.ListExec_nil (by rfl) hl
    cases hf : s.hbFree tr with
    | false => exact .inl ⟨_, rfl, .hbRace hto (hp _ _) hs hf⟩
    | true =>
      refine .inr (.inr ⟨st', .run hto (hp _ _) hs hf, cst', hres, ?_, ?_⟩) <;>
        simp only [taskLog, taskPushed, CSt.st, hst.2.1, hpriv]

omit h in
include he in
/-- **A consume scope node** without dynamic inputs, given its tasks
(`htasks`) and the rest of the segment (`ih`). -/
theorem consume_node {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {c : ConsumeNode}
    {body : List SNode} {e : DEdge} {mIn : Memlet} (hn : s.node? id = some n) (hk : n.kind = .consumeEntry c)
    (hkeys : s.dynKeys? n = some [])
    (hin : (s.inEdges n.id).find? (·.dstConn = some "IN_stream") = some e)
    (hsrc : s.srcEndpoint e = .data mIn.data mIn.dataSubset)
    (htasks : ∀ {pre : CSt} {st₁ : St}, st₁.mem = pre.mem → st₁.vals = [] →
      ∀ {l : Loc} {np : Int} {q : List Val} {rT : Res (List Log × List Val × List Val)},
        STree.ConsumeTasks tc pre.ρ c mIn.data body l np (tc.privateTransients body) pre q rT →
        (∃ e, e.isUB = true ∧ ConsumeTasks ctx pre.ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.err e)) ∨
        (∃ e e', rT = .err e ∧ ConsumeTasks ctx pre.ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.err e')) ∨
        ∃ res, rT = .ok res ∧ ConsumeTasks ctx pre.ρ s n.id c l np (ctx.privateTransients s n.id) st₁ q (.ok res))
    (ih : ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St},
      STree.ListExec tc full gb {} (cs ++ rest) cst r → StRel st cst → SeqOut ctx s o cst.ρ st tc full gb rest r)
    {full' : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St}
    (hx : STree.ListExec tc full' gb {} ((.dynScopeCopy "IN_stream" mIn :: .consumeScope c body :: cs) ++ rest) cst r)
    (hrel : StRel st cst) :
    SeqOut ctx s (id :: o) cst.ρ st tc full' gb rest r := by
  apply SeqOut.cons hn
  simp only [List.cons_append] at hx
  have hshift : cst.st = st.forNode.shift cst.log := hrel.forNode_shift
  have hmem : st.forNode.mem = cst.mem := hrel.1
  have hv : st.forNode.vals = [] := hrel.2.2
  have hsetup := consumeSetup_in (ctx := ctx) (ρ := cst.ρ) (c := c) hkeys hin hsrc st.forNode
  rw [hmem] at hsetup
  have hpt : inStreamPt tc.ctx cst.ρ cst.mem mIn = inStreamPt ctx cst.ρ cst.mem mIn :=
    Ctx.leaf_eq he (fun x => inStreamPt x cst.ρ cst.mem mIn) (fun _ => rfl)
  rcases instream_inv hx with ⟨e', hp, rfl⟩ | ⟨p, hp, hx₂⟩
  · rw [hpt] at hp
    obtain ⟨e'', hs⟩ := hsetup.1 _ hp
    exact .inr (.inl ⟨_, _, rfl, .consumeSetupErr hk hs⟩)
  rw [hpt] at hp
  have hs := hsetup.2 _ hp
  have hsT := consumeSetupT_stream tc c mIn.data p cst
  have htail : consumeTail tc.ctx cst.ρ cst.mem mIn.data c p = consumeTail ctx cst.ρ cst.mem mIn.data c p :=
    Ctx.leaf_eq he (fun x => consumeTail x cst.ρ cst.mem mIn.data c p) (fun _ => rfl)
  rw [htail] at hsT
  cases hq : consumeTail ctx cst.ρ cst.mem mIn.data c p with
  | err e' =>
    rw [hq] at hs hsT
    rcases consume_inv hx₂ with ⟨e'', hs', rfl⟩ | ⟨_, _, _, _, _, hs', -, -⟩ | ⟨_, _, _, _, _, _, _, hs', -, -⟩ <;>
      rw [hsT] at hs' <;> cases hs'
    exact .inr (.inl ⟨_, _, rfl, .consumeSetupErr hk hs⟩)
  | ok lq =>
    obtain ⟨l, np⟩ := lq
    rw [hq] at hs hsT
    simp only [Res.map] at hs hsT
    have hfinT : ∀ logs rs pushed, consumeFinishT tc cst l logs rs pushed =
        (consumeFinish ctx st.forNode l logs rs pushed).map (fun y => cst.withSt (y.shift cst.log)) := by
      intro logs rs pushed
      unfold consumeFinishT
      rw [Ctx.leaf_eq he (fun x => consumeFinish x cst.st l logs rs pushed) (fun _ => rfl), hshift,
        consumeFinish_shift ctx cst.log st.forNode l logs rs pushed]
      cases consumeFinish ctx st.forNode l logs rs pushed <;> rfl
    rcases consume_inv hx₂ with ⟨e', hs', -⟩ | ⟨ρ', d, l', np', e', hs', ht, rfl⟩ |
      ⟨ρ', d, l', np', logs, rs, pushed, hs', ht, hfin⟩
    · rw [hsT] at hs'; cases hs'
    · rw [hsT] at hs'
      simp only [Res.ok.injEq, Prod.mk.injEq] at hs'
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hs'
      rw [← hmem] at ht
      rcases htasks hmem hv ht with ⟨e'', hu, hy⟩ | ⟨e'', e''', he', hy⟩ | ⟨res, hres, -⟩
      · exact .inl ⟨e'', hu, .consumeTasksErr hk hs hy⟩
      · cases he'; exact .inr (.inl ⟨_, _, rfl, .consumeTasksErr hk hs hy⟩)
      · cases hres
    · rw [hsT] at hs'
      simp only [Res.ok.injEq, Prod.mk.injEq] at hs'
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hs'
      rw [← hmem] at ht
      rcases htasks hmem hv ht with ⟨e'', hu, hy⟩ | ⟨e'', e''', he', -⟩ | ⟨res, hres, hy⟩
      · exact .inl ⟨e'', hu, .consumeTasksErr hk hs hy⟩
      · cases he'
      · cases hres
        rw [hfinT] at hfin
        cases hy₂ : consumeFinish ctx st.forNode l logs rs pushed with
        | err e'' =>
          rw [hy₂] at hfin
          rcases hfin with ⟨e₃, hf, rfl⟩ | ⟨cst', hf, -⟩
          · cases hf; exact .inr (.inl ⟨_, _, rfl, hy₂ ▸ NodeExec.consumeScope hk hs hy⟩)
          · cases hf
        | ok y₂ =>
          rw [hy₂] at hfin
          rcases hfin with ⟨e₃, hf, -⟩ | ⟨cst', hf, hx'⟩
          · cases hf
          · simp only [Res.map, Res.ok.injEq] at hf
            subst hf
            have hyv₂ : y₂.vals = [] := by
              unfold consumeFinish at hy₂
              split at hy₂ <;> cases hy₂
              exact hv
            exact .inr (.inr ⟨y₂, hy₂ ▸ NodeExec.consumeScope hk hs hy,
              ih (cst := cst.withSt (y₂.shift cst.log)) hx' (hrel.afterNode hyv₂ ⟨rfl, rfl⟩)⟩)

include he in
/-- **A map scope node**, given its iterations (`hiter`) and the rest of the
segment (`ih`). -/
theorem map_node {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {mp : MapNode} {body : List SNode}
    {keys : List InKey} {ds : List SNode} (hn : s.node? id = some n) (hk : n.kind = .mapEntry mp)
    (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (hds : DynNodes tc keys ds)
    (hiter : ∀ {pre : CSt} {st₁ : St}, st₁.mem = pre.mem → st₁.vals = [] →
      ∀ {pts : List Idx} {rT : Res (List Log)},
        STree.IterExec tc mp.params body (tc.privateTransients body) pre pts rT →
        (∃ e, e.isUB = true ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e)) ∨
        (∃ e e', rT = .err e ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e')) ∨
        ∃ logs, rT = .ok logs ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs))
    (ih : ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St},
      STree.ListExec tc full gb {} (cs ++ rest) cst r → StRel st cst → SeqOut ctx s o cst.ρ st tc full gb rest r)
    {full' : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St}
    (hx : STree.ListExec tc full' gb {} ((ds ++ .mapScope mp body :: cs) ++ rest) cst r) (hrel : StRel st cst) :
    SeqOut ctx s (id :: o) cst.ρ st tc full' gb rest r := by
    have hn' := List.mem_of_find?_eq_some hn
    apply SeqOut.cons hn
    have hx₁ : STree.ListExec tc full' gb (pendingSc []) (ds ++ (.mapScope mp body :: (cs ++ rest))) cst r := by
      simpa only [List.append_assoc, List.cons_append] using hx
    have hshift : cst.st = st.forNode.shift cst.log := hrel.forNode_shift
    rcases dscopies_run he hds (fun k hk' => (hok.2 k hk').1) hx₁ with
      ⟨e, rfl, hd⟩ | ⟨syms, y', hd, hyv', hx₂⟩
    · rw [hshift, dynReadAcc_shift] at hd
      cases hd₀ : dynReadAcc ctx cst.ρ [] keys st.forNode with
      | ok _ => rw [hd₀] at hd; cases hd
      | err e' =>
        rw [hd₀] at hd; cases hd
        exact .inr (.inl ⟨_, _, rfl, .mapSetupErr hk (by rw [mapSetup_dyn hkeys hok, hd₀]; rfl)⟩)
    · rw [hshift, dynReadAcc_shift] at hd
      cases hd₀ : dynReadAcc ctx cst.ρ [] keys st.forNode with
      | err _ => rw [hd₀] at hd; cases hd
      | ok p =>
        obtain ⟨syms₀, y⟩ := p
        rw [hd₀] at hd
        simp only [Res.map, St.shiftR, Prod.map, _root_.id, Res.ok.injEq, Prod.mk.injEq] at hd
        obtain ⟨rfl, rfl⟩ := hd
        have hyv : y.vals = [] := hyv'
        have hset := mapSetup_dyn (ρ := cst.ρ) (mp := mp) hkeys hok st.forNode
        rw [hd₀] at hset
        simp only [Res.ok_bind] at hset
        have hsetT := mapSetupT_dyn he (mp := mp) syms₀ (cst.withSt (y.shift cst.log))
        have hm₂ : (cst.withSt (y.shift cst.log)).mem = y.mem := rfl
        have hρ₂ : (cst.withSt (y.shift cst.log)).ρ = cst.ρ := rfl
        rw [hm₂, hρ₂] at hsetT
        rcases map_inv hx₂ with ⟨e, hs, rfl⟩ | ⟨ρ', pts, e, hs, hi, rfl⟩ | ⟨ρ', pts, logs, hs, hi, hfin⟩
        · rw [hsetT] at hs
          cases hp : ctx.rangePoints (Valuation.setAll cst.ρ syms₀) y.mem mp.range with
          | ok _ => rw [hp] at hs; cases hs
          | err e' =>
            rw [hp] at hs hset; cases hs
            exact .inr (.inl ⟨_, _, rfl, .mapSetupErr hk hset⟩)
        · rw [hsetT] at hs
          cases hp : ctx.rangePoints (Valuation.setAll cst.ρ syms₀) y.mem mp.range with
          | err _ => rw [hp] at hs; cases hs
          | ok pts' =>
            rw [hp] at hs hset
            simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at hs hset
            obtain ⟨rfl, rfl⟩ := hs
            rcases hiter (pre := { cst.withSt (y.shift cst.log) with ρ := Valuation.setAll cst.ρ syms₀ }) (st₁ := y)
                rfl hyv hi with ⟨e', hu, hy⟩ | ⟨e', e'', he', hy⟩ | ⟨logs, hl, -⟩
            · exact .inl ⟨e', hu, .mapIterErr hk hset hy⟩
            · cases he'; exact .inr (.inl ⟨_, _, rfl, .mapIterErr hk hset hy⟩)
            · cases hl
        · rw [hsetT] at hs
          cases hp : ctx.rangePoints (Valuation.setAll cst.ρ syms₀) y.mem mp.range with
          | err _ => rw [hp] at hs; cases hs
          | ok pts' =>
            rw [hp] at hs hset
            simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at hs hset
            obtain ⟨rfl, rfl⟩ := hs
            rcases hiter (pre := { cst.withSt (y.shift cst.log) with ρ := Valuation.setAll cst.ρ syms₀ }) (st₁ := y)
                rfl hyv hi with ⟨e', hu, hy⟩ | ⟨e', e'', he', -⟩ | ⟨logs', hl, hy⟩
            · exact .inl ⟨e', hu, .mapIterErr hk hset hy⟩
            · cases he'
            · cases hl
              have hst₂ : (cst.withSt (y.shift cst.log)).st = y.shift cst.log := by
                obtain ⟨m', l', v'⟩ := y
                simp only at hyv
                simp [CSt.withSt, CSt.st, St.shift, hyv]
              have hfinS := scopeFinish_shift ctx cst.log y logs
              rw [← hst₂] at hfinS
              rw [scopeFinishT_eq he, hfinS] at hfin
              cases hy' : scopeFinish ctx y logs with
              | err e' =>
                rw [hy'] at hfin
                rcases hfin with ⟨e'', hf, rfl⟩ | ⟨cst', hf, -⟩
                · cases hf; exact .inr (.inl ⟨_, _, rfl, hy' ▸ NodeExec.mapScope hk hset hy⟩)
                · cases hf
              | ok y₂ =>
                rw [hy'] at hfin
                rcases hfin with ⟨e'', hf, -⟩ | ⟨cst', hf, hx'⟩
                · cases hf
                · simp only [Res.map, Res.ok.injEq] at hf
                  subst hf
                  have hyv₂ : y₂.vals = [] := by
                    obtain ⟨-, hyd⟩ := scopeFinish_ok hy'
                    rw [hyd]; simpa [St.step] using hyv
                  have hc : ((cst.withSt (y.shift cst.log)).withSt (y₂.shift cst.log)) = cst.withSt (y₂.shift cst.log) := rfl
                  rw [hc] at hx'
                  exact .inr (.inr ⟨y₂, hy' ▸ NodeExec.mapScope hk hset hy,
                    ih (cst := cst.withSt (y₂.shift cst.log)) hx' (hrel.afterNode hyv₂ ⟨rfl, rfl⟩)⟩)


include h he in
/-- **Matching of a dataflow segment** (see the module documentation). -/
theorem SegCorr.matched (hw : ∀ op v, ctx.I.wcr op .undef v = .undef) {o : List NodeId} {cs : List SNode}
    (hc : SegCorr ctx tc s o cs) :
    ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St},
      STree.ListExec tc full gb {} (cs ++ rest) cst r → StRel st cst → SeqOut ctx s o cst.ρ st tc full gb rest r := by
  induction hc with
  | nil =>
    intro full gb rest cst r st hx hrel
    exact .inr (.inr ⟨cst, st, [], hx, .nil, rfl, hrel⟩)
  | @copies tc s id n o cs cps hn hk hsh hks _ ih =>
    intro full gb rest cst r st hx hrel
    have hn' := List.mem_of_find?_eq_some hn
    rw [List.append_assoc] at hx
    -- the SDFG node runs the same copies
    have hfold : ∀ X, execSimple ctx cst.ρ s n X = (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) X := by
      intro X
      rw [execSimple_copyTarget hk, execCopiesInto_eq cst.ρ hsh, foldlM_keyCorr hks]
      rfl
    have hshift : (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) cst.st =
        ((cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) st.forNode).map (St.shift cst.log) := by
      rw [hrel.forNode_shift]
      exact List.foldlM_res_comm _ _ (fun b k => copyData_shift ctx cst.ρ cst.log b _ _ _ _ _ _) _ _
    have hsimp := DNode.isSimple_of_copyTarget hk
    apply SeqOut.cons hn
    rcases copies_run he cps hx with ⟨e, rfl, hf⟩ | ⟨cst', hx', hf, hρ⟩
    · rw [hshift] at hf
      cases hy : (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) st.forNode with
      | ok y => rw [hy] at hf; cases hf
      | err e' =>
        rw [hy] at hf
        cases hf
        exact .inr (.inl ⟨_, _, rfl, (hfold st.forNode).trans hy ▸ NodeExec.simple hsimp⟩)
    · rw [hshift] at hf
      cases hy : (cps.map fun p => tc.copyKey p.1 p.2).foldlM (copyStep ctx cst.ρ) st.forNode with
      | err e' => rw [hy] at hf; cases hf
      | ok y =>
        rw [hy] at hf
        simp only [Res.map, Res.ok.injEq] at hf
        have hyv : y.vals = [] := by
          have hnc : n.isCode = false := by
            unfold DNode.isCopyTarget at hk; unfold DNode.isCode; split at hk <;> simp_all
          obtain ⟨-, -, -, hv⟩ := execSimple_steps (kv := true) ctx cst.ρ s n
            (fun _ hc => by rw [hnc] at hc; cases hc) st.forNode y ((hfold _).trans hy)
          rw [hv rfl]; simpa [St.forNode] using hrel.2.2
        refine .inr (.inr ⟨y, (hfold st.forNode).trans hy ▸ NodeExec.simple hsimp, ?_⟩)
        have := ih h he hx' (hrel.afterNode hyv ⟨?_, ?_⟩)
        · rw [hρ] at this; exact this
        · have := congrArg St.mem hf; simpa [CSt.st] using this.symm
        · have := congrArg St.log hf; simpa [CSt.st, hrel.2.1] using this.symm
  | @refSet tc s id n o cs d base sub m sd x hn hk hr hm hsub _ ih =>
    intro full gb rest cst r st hx hrel
    refine leaf_node hn (DNode.isSimple_of_copyTarget hk) (fun hc => ?_) ?_ hx hrel
      (fun hx' hrel' hρ => hρ ▸ ih h he hx' hrel')
    · unfold DNode.isCopyTarget at hk; unfold DNode.isCode at hc; split at hk <;> simp_all
    · have : ∀ X, execRefSetT tc cst.ρ d m (.data sd) X = execSimple ctx cst.ρ s n X := by
        intro X
        rw [execSimple_copyTarget hk, execCopiesInto_refSet hr]
        simp only [execRefSetT]
        rw [Ctx.leaf_eq he (fun c => aliasTo c cst.ρ X d m.data m.dataSubset) (fun _ => rfl), hm, aliasTo_subCorr hsub]
      simp only [stepLeaf, this]
  | @refSetCode tc s id n o cs d m g cls x hn hk hr _ ih =>
    intro full gb rest cst r st hx hrel
    apply SeqOut.cons hn
    obtain ⟨e, hy⟩ := execCopiesInto_refSetCode hr cst.ρ st.forNode
    have hstep : stepLeaf tc {} (.refSetNode d m (.codeNode g cls) x) cst =
        some (leafOut (cst.lift (.err (.unsupported s!"reference set of '{d}' from a {cls} (Decisions.md#D55)")))) := rfl
    rcases leaf_inv hstep hx with ⟨e', -, rfl⟩ | ⟨cst₁, hR, -⟩
    · refine .inr (.inl ⟨_, e, rfl, ?_⟩)
      have := NodeExec.simple (ctx := ctx) (ρ := cst.ρ) (s := s) (st := st.forNode) (DNode.isSimple_of_copyTarget hk)
      rwa [execSimple_copyTarget hk, hy] at this
    · cases hR
  | @tasklet tc s id n o cs t ins outs hn hk hsh hki hko hd hdT _ ih =>
    intro full gb rest cst r st hx hrel
    refine leaf_node hn (by simp [DNode.isSimple, hk]) (fun _ => hsh.outs) ?_ hx hrel
      (fun hx' hrel' hρ => hρ ▸ ih h he hx' hrel')
    have : ∀ X, execCodeT tc cst.ρ t.label ins outs X (tc.ctx.I.tasklet t cst.ρ) (.taskletUndefined t.label) =
        execSimple ctx cst.ρ s n X := by
      intro X
      rw [← execCode_eq he cst.ρ hsh hki hko hd hdT, I_eq he]
      simp [execSimple, hk]
    simp only [stepLeaf, this]
  | @library tc s id n o cs l l' props ins outs hn hk hl hsh hki hko hd hdT _ ih =>
    intro full gb rest cst r st hx hrel
    refine leaf_node hn (by simp [DNode.isSimple, hk]) (fun _ => hsh.outs) ?_ hx hrel
      (fun hx' hrel' hρ => hρ ▸ ih h he hx' hrel')
    have : ∀ X, execLibraryT tc cst.ρ l' ins outs X = execSimple ctx cst.ρ s n X := by
      intro X
      unfold execLibraryT
      rw [hl.2, hl.1]
      cases hr : l.reduce with
      | some props' =>
        simp only []
        rw [← execReduce_eq he cst.ρ hsh hki hko props' X]
        simp [execSimple, hk, hr]
      | none =>
        simp only []
        rw [← execCode_eq he cst.ρ hsh hki hko hd hdT, I_eq he]
        simp [execSimple, hk, hr]
    simp only [stepLeaf, this]
  | @map tc s id n o cs mp oM full body keys ds hn hk hkeys hok hds hto hdir hpriv _ _ ihb ih =>
    intro full' gb rest cst r st hx hrel
    exact map_node he hn hk hkeys hok hds
      (fun hm hv _ _ hi => iter_matched h hto hdir hpriv (fun hb hr => ihb h he hb hr) hm hv hi)
      (fun hx hr => ih h he hx hr) hx hrel
  | @consume tc s id n o cs c oM full body e mIn hn hk hkeys hin hsrc hd hto hdir hpriv _ _ ihb ih =>
    intro full' gb rest cst r st hx hrel
    exact consume_node he hn hk hkeys hin hsrc
      (fun hm hv _ _ _ _ ht => tasks_matched h he hto hdir hpriv hd (fun hb hr => ihb h he hb hr) hm hv ht)
      (fun hx hr => ih h he hx hr) hx hrel
  | @nestedMap tc s id n o cs mp full body keys ds nid nd inner mapping ni conns σ hn hk hkeys hok hds xs hto hdir hxs hnd
      hkn hb hmap hcode hout hpriv hnames hinner _ ih =>
    intro full' gb rest cst r st hx hrel
    have htc : tc.withI ctx.I = tc := by rw [← I_eq he]; rfl
    have htcR : (tc.renT σ).withI ctx.I = tc.renT σ := by rw [← I_eq he]; rfl
    have hnames' : Region.Names 1 (fun _ => True) (ctx.enter inner nd.id conns) ((tc.renT σ).innerCtx inner)
        inner.top := by
      have := hnames ctx.I
      rw [htcR] at this
      exact this
    have hinner' : ∀ (cst : CSt) (r : Res (TCtl × CSt)), STree.ListExec tc body false {} body cst r →
        NestedOut (RegionAt ((tc.renT σ).innerCtx inner) inner.top cst inner.top.start) r := by
      have := hinner ctx.I hw
      rw [htc] at this
      exact this
    exact map_node he hn hk hkeys hok hds
      (fun hm _ _ _ hi => iter_matched_of (n := n) (fun i {res} hbd => nested_body h he hw hto hdir hxs hnd hkn hb hmap
        hcode hout hpriv (cT := (tc.renT σ).innerCtx inner) rfl rfl hnames' hinner' hm i hbd) hi)
      (fun hx hr => ih h he hx hr) hx hrel
  | @nested tc s id nd o cs body inner mapping ni conns σ hn hkn hb hmap hcode hout hctr hshape hnames hinner _ ih =>
    intro full gb rest cst r st hx hrel
    rw [List.append_assoc] at hx
    have htc : tc.withI ctx.I = tc := by rw [← I_eq he]; rfl
    have htcR : (tc.renT σ).withI ctx.I = tc.renT σ := by rw [← I_eq he]; rfl
    have hnames' : Region.Names 1 (fun _ => True) (ctx.enter inner nd.id conns) ((tc.renT σ).innerCtx inner)
        inner.top := by
      have := hnames ctx.I
      rw [htcR] at this
      exact this
    have hinner' : ∀ (cst : CSt) (r : Res (TCtl × CSt)), STree.ListExec tc body false {} body cst r →
        NestedOut (RegionAt ((tc.renT σ).innerCtx inner) inner.top cst inner.top.start) r := by
      have := hinner ctx.I hw
      rw [htc] at this
      exact this
    have hlog : ∀ {rb : Res (TCtl × CSt)}, STree.ListExec tc body false {} body cst rb →
        ∃ r₀, STree.ListExec tc body false {} body { ρ := cst.ρ, mem := st.forNode.mem } r₀ ∧
          rb = r₀.map (shiftT cst.log) := by
      intro rb hb'
      have hb'' : STree.ListExec tc body false {} body { cst with log := cst.log } rb := hb'
      obtain ⟨r₀, h₀, hr₀⟩ := (STree.ListExec.log_iff cst.log).1 hb''
      refine ⟨r₀, ?_, hr₀⟩
      have hst : ({ cst with log := Log.empty } : CSt) = { ρ := cst.ρ, mem := st.forNode.mem } := by
        simp only [St.forNode, hrel.1]
      rwa [hst] at h₀
    apply SeqOut.cons hn
    rcases STree.ListExec.split_nested hshape hx with ⟨e, rfl, hbE⟩ | ⟨ctl, cJ, hj, rfl, hbJ⟩ | ⟨cst', hbO, hρ', hrest⟩
    · obtain ⟨r₀, h₀, hr₀⟩ := hlog (hbE body)
      rcases nested_call he hw hkn hb hmap hcode hout hctr (cT := (tc.renT σ).innerCtx inner) rfl rfl hnames' cst.ρ
        st.forNode (hinner' _ _ h₀) with
        ⟨e', hu, hx'⟩ | ⟨e₀, e', -, hx'⟩ | ⟨c, rfl, -⟩
      · exact .inl ⟨e', hu, hx'⟩
      · exact .inr (.inl ⟨e, e', rfl, hx'⟩)
      · cases hr₀
    · -- the body jumps out: only an undefined inner run matches
      obtain ⟨r₀, h₀, hr₀⟩ := hlog (hbJ body)
      rcases nested_call he hw hkn hb hmap hcode hout hctr (cT := (tc.renT σ).innerCtx inner) rfl rfl hnames' cst.ρ
        st.forNode (hinner' _ _ h₀) with
        ⟨e', hu, hx'⟩ | ⟨e₀, e', rfl, -⟩ | ⟨c, rfl, -⟩
      · exact .inl ⟨e', hu, hx'⟩
      · cases hr₀
      · simp only [Res.map_ok, shiftT, Res.ok_inj, Prod.mk.injEq] at hr₀
        rw [hr₀.1] at hj; cases hj
    · obtain ⟨r₀, h₀, hr₀⟩ := hlog (hbO body)
      rcases nested_call he hw hkn hb hmap hcode hout hctr (cT := (tc.renT σ).innerCtx inner) rfl rfl hnames' cst.ρ
        st.forNode (hinner' _ _ h₀) with
        ⟨e', hu, hx'⟩ | ⟨e₀, e', rfl, -⟩ | ⟨c, rfl, hx'⟩
      · exact .inl ⟨e', hu, hx'⟩
      · cases hr₀
      · simp only [Res.map_ok, shiftT, Res.ok_inj, Prod.mk.injEq] at hr₀
        obtain ⟨-, rfl⟩ := hr₀
        refine .inr (.inr ⟨_, hx', ?_⟩)
        have hrel' : StRel (st.afterNode { mem := c.mem, log := st.forNode.log.seq c.log, vals := st.forNode.vals })
            (c.shift cst.log) := by
          refine ⟨rfl, ?_, ?_⟩
          · simp only [St.afterNode, St.forNode, Log.empty_seq, CSt.shift, hrel.2.1]
          · simpa [St.afterNode, St.forNode] using hrel.2.2
        have := ih h he hrest hrel'
        rwa [hρ'] at this

end Matched

end Corr

end Dace
