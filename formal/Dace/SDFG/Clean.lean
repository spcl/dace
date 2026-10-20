-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Exec

/-!
# Races only arise from merges and happens-before checks

`Res.Clean r`: the outcome `r` is not a race error.  Every function of
`Dace.SDFG.Semantics.Leaf` other than the merging ones (`scopeFinish`,
`consumeFinish`, `stateFinish`) is clean, as are the helpers of control flow
(`trueEdges`, `applyAssigns`, `applyAssign?`, `Ctx.evalBoolR`).  Race errors
of an execution therefore originate in a merge of parallel logs or a
happens-before check (`OrderedExec.hbRace`); `Dace.Equiv.RaceFree` uses this.
-/

set_option autoImplicit false

namespace Dace

/-- The outcome is not a race error. -/
def Res.Clean {α : Type} (r : Res α) : Prop := ∀ w, r ≠ .err (.race w)

instance : LawfulMonad Res := LawfulMonad.mk'
  (id_map := by intro α x; cases x <;> rfl)
  (pure_bind := by intros; rfl)
  (bind_assoc := by intro α β γ x f g; cases x <;> rfl)

namespace Res.Clean

variable {α β : Type}

theorem ok (a : α) : Res.Clean (.ok a) := fun _ h => by cases h

theorem pure' (a : α) : Res.Clean (pure a : Res α) := fun _ h => by cases h

theorem err {e : ErrReason} (h : ∀ w, e ≠ .race w) : Res.Clean (.err e : Res α) := fun w hw => by
  cases hw; exact h w rfl

theorem bind {x : Res α} {f : α → Res β} (hx : Res.Clean x) (hf : ∀ a, x = .ok a → Res.Clean (f a)) :
    Res.Clean (x >>= f) := by
  intro w hw
  cases x with
  | ok a => exact hf a rfl w hw
  | err e => exact hx w (by simpa [Bind.bind, Res.bind] using hw)

theorem ofOption {o : Option α} {e : ErrReason} (h : ∀ w, e ≠ .race w) : Res.Clean (Res.ofOption o e) := by
  cases o
  · exact err h
  · exact ok _

theorem foldlM {γ : Type} {f : β → γ → Res β} :
    ∀ {l : List γ} {b : β}, (∀ b, ∀ c ∈ l, Res.Clean (f b c)) → Res.Clean (l.foldlM f b)
  | [], b, _ => pure' b
  | c :: cs, b, h => by
    simp only [List.foldlM_cons]
    exact bind (h b c (List.mem_cons_self ..)) fun a _ => foldlM fun b' c' hc => h b' c' (List.mem_cons_of_mem _ hc)

theorem mapM {γ : Type} {f : γ → Res β} :
    ∀ {l : List γ}, (∀ c ∈ l, Res.Clean (f c)) → Res.Clean (l.mapM f)
  | [], _ => pure' _
  | c :: cs, h => by
    rw [List.mapM_cons]
    exact bind (h c (List.mem_cons_self ..)) fun a _ =>
      bind (mapM fun c' hc => h c' (List.mem_cons_of_mem _ hc)) fun _ _ => pure' _

/-- An error of one type is clean when the same error of another is. -/
theorem err_of {x : Res α} {e : ErrReason} (h : x = .err e) (hx : Res.Clean x) : Res.Clean (.err e : Res β) :=
  err fun w hw => hx w (by rw [h, hw])

theorem filterMapM {γ : Type} {f : γ → Res (Option β)} :
    ∀ {l : List γ}, (∀ c ∈ l, Res.Clean (f c)) → Res.Clean (l.filterMapM f)
  | [], _ => pure' _
  | c :: cs, h => by
    rw [List.filterMapM_cons]
    refine bind (h c (List.mem_cons_self ..)) fun a _ => ?_
    have ih := filterMapM (l := cs) fun c' hc => h c' (List.mem_cons_of_mem _ hc)
    split
    · exact ih
    · exact bind ih fun _ _ => pure' _

end Res.Clean

/-- Non-race error constructors. -/
macro "clean_err" : tactic => `(tactic| (intro w; simp; done))

/-- Try the given lemmas (each applied to `..`). -/
syntax "clean_apply" " [" term,* "]" : tactic

macro_rules
  | `(tactic| clean_apply [$ts,*]) => do
    let mut tac ← `(tactic| fail "no clean lemma applies")
    for t in ts.getElems.reverse do
      tac ← `(tactic| first | with_reducible apply $t | $tac:tactic)
    return tac

/-- Decompose a `Res.Clean` goal along `bind`, `foldlM`, `mapM`, `ofOption`
and pattern matches, closing leaves by constructors and the given lemmas. -/
syntax "clean_tac" (" [" term,* "]")? : tactic

macro_rules
  | `(tactic| clean_tac) => `(tactic| clean_tac [])
  | `(tactic| clean_tac [$ts,*]) => `(tactic| repeat' (first
      | with_reducible exact Res.Clean.ok _
      | with_reducible exact Res.Clean.pure' _
      | (with_reducible refine Res.Clean.err ?_); clean_err
      | (with_reducible refine Res.Clean.ofOption ?_); clean_err
      | clean_apply [$ts,*]
      | with_reducible refine Res.Clean.bind ?_ fun _ _ => ?_
      | with_reducible refine Res.Clean.foldlM fun _ _ _ => ?_
      | with_reducible refine Res.Clean.mapM fun _ _ => ?_
      | split
      | dsimp only
      | clean_err))

section Basic

variable (ctx : Ctx) (ρ : Valuation)

theorem Ctx.addr_clean (m : Mem) (n : Name) (idx : Idx) : Res.Clean (ctx.addr ρ m n idx) := by
  unfold Ctx.addr; clean_tac

theorem Ctx.checkBounds_clean (n : Name) (idx : Idx) (oob : Bool) : Res.Clean (ctx.checkBounds ρ n idx oob) := by
  unfold Ctx.checkBounds; clean_tac

theorem Ctx.loc_clean (m : Mem) (n : Name) (idx : Idx) (oob : Bool) : Res.Clean (ctx.loc ρ m n idx oob) := by
  unfold Ctx.loc; clean_tac [ctx.addr_clean ρ, ctx.checkBounds_clean ρ]

theorem Ctx.pointsOf_clean (m : Mem) (n : Name) (sub : Option Subset) : Res.Clean (ctx.pointsOf ρ m n sub) := by
  unfold Ctx.pointsOf; clean_tac

theorem Ctx.rangePoints_clean (m : Mem) (r : Range) : Res.Clean (ctx.rangePoints ρ m r) := by
  unfold Ctx.rangePoints; clean_tac

theorem Ctx.evalR_clean (m : Mem) (e : Expr) : Res.Clean (ctx.evalR ρ m e) := by
  unfold Ctx.evalR; clean_tac

theorem Ctx.evalBoolR_clean (m : Mem) (e : Expr) : Res.Clean (ctx.evalBoolR ρ m e) := by
  unfold Ctx.evalBoolR; clean_tac

theorem Ctx.evalIntR_clean (m : Mem) (e : Expr) : Res.Clean (ctx.evalIntR ρ m e) := by
  unfold Ctx.evalIntR; clean_tac

theorem St.read_clean (st : St) (n : Name) (idx : Idx) (oob : Bool) : Res.Clean (st.read ctx ρ n idx oob) := by
  unfold St.read; clean_tac [ctx.loc_clean ρ]

theorem St.readMany_clean (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    Res.Clean (st.readMany ctx ρ n pts oob) := by
  unfold St.readMany; clean_tac [St.read_clean ctx ρ]

theorem St.write_clean (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    Res.Clean (st.write ctx ρ n idx oob v) := by
  unfold St.write; clean_tac [ctx.loc_clean ρ]

theorem St.wcr_clean (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    Res.Clean (st.wcr ctx ρ n idx oob op v) := by
  unfold St.wcr; clean_tac [ctx.loc_clean ρ]

theorem St.store_clean (st : St) (n : Name) (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    Res.Clean (st.store ctx ρ n idx oob w v) := by
  unfold St.store; clean_tac [St.write_clean ctx ρ, St.wcr_clean ctx ρ]

theorem St.queue_clean (st : St) (n : Name) (idx : Idx) : Res.Clean (st.queue ctx ρ n idx) := by
  unfold St.queue; clean_tac [ctx.loc_clean ρ]

theorem St.push_clean (st : St) (n : Name) (idx : Idx) (vs : List Val) : Res.Clean (st.push ctx ρ n idx vs) := by
  unfold St.push; clean_tac [ctx.loc_clean ρ]

theorem St.popN_clean (st : St) (n : Name) (idx : Idx) (k : Nat) : Res.Clean (st.popN ctx ρ n idx k) := by
  unfold St.popN; clean_tac [ctx.loc_clean ρ]

end Basic

section Leaf

variable (ctx : Ctx) (ρ : Valuation) (s : State)

theorem readScalarData_clean (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    Res.Clean (readScalarData ctx ρ d sub oob st) := by
  unfold readScalarData; clean_tac [ctx.pointsOf_clean ρ, St.popN_clean ctx ρ, St.read_clean ctx ρ]

theorem readEdgeScalar_clean (e : DEdge) (st : St) : Res.Clean (readEdgeScalar ctx ρ s e st) := by
  unfold readEdgeScalar; clean_tac [readScalarData_clean ctx ρ]

theorem streamPoint_clean (m : Mem) (d : Name) (sub : Option Subset) : Res.Clean (streamPoint ctx ρ m d sub) := by
  unfold streamPoint; clean_tac [ctx.pointsOf_clean ρ]

theorem Ctx.lowerCorner_clean (m : Mem) (n : Name) (sub : Option Subset) : Res.Clean (ctx.lowerCorner ρ m n sub) := by
  unfold Ctx.lowerCorner; clean_tac [ctx.pointsOf_clean ρ, ctx.evalIntR_clean ρ]

theorem copyData_clean (st : St) (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (w : Option WcrOp) (oob : Bool) : Res.Clean (copyData ctx ρ st sd ssub dd dsub w oob) := by
  unfold copyData
  clean_tac [ctx.pointsOf_clean ρ, streamPoint_clean ctx ρ, St.popN_clean ctx ρ, St.store_clean ctx ρ,
    St.readMany_clean ctx ρ, St.push_clean ctx ρ, St.read_clean ctx ρ]

theorem execCopy_clean (e : DEdge) (st : St) : Res.Clean (execCopy ctx ρ s e st) := by
  unfold execCopy; clean_tac [copyData_clean ctx ρ]

theorem bindInData_clean (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    Res.Clean (bindInData ctx ρ d sub oob st) := by
  unfold bindInData
  clean_tac [streamPoint_clean ctx ρ, St.queue_clean ctx ρ, ctx.pointsOf_clean ρ, St.read_clean ctx ρ,
    ctx.lowerCorner_clean ρ, St.readMany_clean ctx ρ]

theorem bindIn_clean (e : DEdge) (st : St) : Res.Clean (bindIn ctx ρ s e st) := by
  unfold bindIn; clean_tac [bindInData_clean ctx ρ]

theorem bindInputs_clean (n : DNode) (st : St) : Res.Clean (bindInputs ctx ρ s n st) := by
  unfold bindInputs; clean_tac [bindIn_clean ctx ρ s]

theorem storeScalar_clean (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool) (v : Val) :
    Res.Clean (storeScalar ctx ρ st d sub w oob v) := by
  unfold storeScalar
  clean_tac [streamPoint_clean ctx ρ, St.push_clean ctx ρ, ctx.pointsOf_clean ρ, St.store_clean ctx ρ]

theorem applyScalar_clean (n : DNode) (st : St) (c : Conn) (v : Val) :
    Res.Clean (applyScalar ctx ρ s n st c v) := by
  unfold applyScalar; clean_tac [storeScalar_clean ctx ρ]

theorem storeArrayWrite_clean (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool)
    (rel : Idx) (v : Val) : Res.Clean (storeArrayWrite ctx ρ st d sub w oob rel v) := by
  unfold storeArrayWrite
  clean_tac [ctx.lowerCorner_clean ρ, ctx.pointsOf_clean ρ, St.store_clean ctx ρ]

theorem applyArrayWrite_clean (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    Res.Clean (applyArrayWrite ctx ρ s n st c rel v) := by
  unfold applyArrayWrite; clean_tac [storeArrayWrite_clean ctx ρ]

theorem pushTo_clean (st : St) (d : Name) (sub : Option Subset) (vs : List Val) :
    Res.Clean (pushTo ctx ρ st d sub vs) := by
  unfold pushTo; clean_tac [streamPoint_clean ctx ρ, St.push_clean ctx ρ]

theorem applyPush_clean (n : DNode) (st : St) (c : Conn) (vs : List Val) :
    Res.Clean (applyPush ctx ρ s n st c vs) := by
  unfold applyPush; clean_tac [pushTo_clean ctx ρ]

theorem popFrom_clean (st : St) (d : Name) (sub : Option Subset) (k : Nat) :
    Res.Clean (popFrom ctx ρ st d sub k) := by
  unfold popFrom; clean_tac [streamPoint_clean ctx ρ, St.popN_clean ctx ρ]

theorem applyPop_clean (n : DNode) (st : St) (c : Conn) (k : Nat) : Res.Clean (applyPop ctx ρ s n st c k) := by
  unfold applyPop; clean_tac [popFrom_clean ctx ρ]

theorem applyOut_clean (n : DNode) (st : St) (out : TaskletOut) : Res.Clean (applyOut ctx ρ s n st out) := by
  unfold applyOut
  clean_tac [applyPop_clean ctx ρ s, applyScalar_clean ctx ρ s, applyArrayWrite_clean ctx ρ s,
    applyPush_clean ctx ρ s]

theorem execCode_clean (n : DNode) (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut)
    {err : ErrReason} (herr : ∀ w, err ≠ .race w) : Res.Clean (execCode ctx ρ s n st run err) := by
  unfold execCode
  clean_tac [bindInputs_clean ctx ρ s, applyOut_clean ctx ρ s, Res.Clean.err herr]

theorem reduceData_clean (st : St) (props : ReduceProps) (sd : Name) (ssub : Option Subset) (dd : Name)
    (dsub : Option Subset) (w : Option WcrOp) (soob doob : Bool) :
    Res.Clean (reduceData ctx ρ st props sd ssub dd dsub w soob doob) := by
  unfold reduceData
  clean_tac [ctx.pointsOf_clean ρ, St.readMany_clean ctx ρ, ctx.evalR_clean ρ, St.store_clean ctx ρ,
    St.wcr_clean ctx ρ]

theorem execReduce_clean (n : DNode) (props : ReduceProps) (st : St) : Res.Clean (execReduce ctx ρ s n props st) := by
  unfold execReduce; clean_tac [reduceData_clean ctx ρ]

theorem aliasTo_clean (st : St) (d base : Name) (sub : Option Subset) : Res.Clean (aliasTo ctx ρ st d base sub) := by
  unfold aliasTo; clean_tac [ctx.lowerCorner_clean ρ, ctx.addr_clean ρ]

theorem bindView_clean (v : DNode) (d : Name) (st : St) : Res.Clean (bindView ctx ρ s v d st) := by
  unfold bindView; clean_tac [aliasTo_clean ctx ρ]

theorem bindViews_clean (ids : List NodeId) (st : St) : Res.Clean (bindViews ctx ρ s ids st) := by
  unfold bindViews; clean_tac [bindView_clean ctx ρ s]

theorem refSet_clean (e : DEdge) (d : Name) (st : St) : Res.Clean (refSet ctx ρ s e d st) := by
  unfold refSet; clean_tac [aliasTo_clean ctx ρ]

theorem execCopiesInto_clean (n : DNode) (st : St) : Res.Clean (execCopiesInto ctx ρ s n st) := by
  unfold execCopiesInto; clean_tac [refSet_clean ctx ρ s, execCopy_clean ctx ρ s]

theorem execSimple_clean (n : DNode) (st : St) : Res.Clean (execSimple ctx ρ s n st) := by
  unfold execSimple
  clean_tac [execCopiesInto_clean ctx ρ s, execReduce_clean ctx ρ s, execCode_clean ctx ρ s]

theorem bindDynamicInputs_clean (n : DNode) (st : St) : Res.Clean (bindDynamicInputs ctx ρ s n st) := by
  unfold bindDynamicInputs; clean_tac [readEdgeScalar_clean ctx _ s]

theorem mapSetup_clean (n : DNode) (mp : MapNode) (st : St) : Res.Clean (mapSetup ctx ρ s n mp st) := by
  unfold mapSetup; clean_tac [bindDynamicInputs_clean ctx ρ s, ctx.rangePoints_clean]

end Leaf

section Control

variable (ctx : Ctx)

theorem trueEdges_clean (r : Region) (cst : CSt) (bid : BlockId) : Res.Clean (trueEdges ctx r cst bid) := by
  unfold trueEdges
  exact Res.Clean.filterMapM fun _ _ => Res.Clean.bind (ctx.evalBoolR_clean _ _ _) fun _ _ => Res.Clean.pure' _

theorem applyAssigns_clean (cst : CSt) (as : List (Sym × Expr)) : Res.Clean (applyAssigns ctx cst as) := by
  unfold applyAssigns; clean_tac [ctx.evalR_clean]

theorem applyAssign?_clean (cst : CSt) (a : Option (Sym × Expr)) : Res.Clean (applyAssign? ctx cst a) := by
  unfold applyAssign?; clean_tac [applyAssigns_clean ctx]

end Control

end Dace
