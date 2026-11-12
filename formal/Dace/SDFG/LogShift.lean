-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics
import Dace.Equiv.Frag

/-!
# Logs are write-only (SDFG)

No execution relation of the SDFG semantics reads the log it starts with: a
derivation from a state whose log is `L` corresponds exactly to a derivation
from the same state with any other log, the outcomes agreeing on everything
but the log, and the final log being the initial one followed (`Log.seq`) by
the same delta.  With `St.shift L st` (prefix `L` to the log of `st`):

* `NodeExec`, `OrderedExec` (`Res St`), `SeqExec` (`Res (St × trace)`, the
  trace unchanged), and the control-flow relations `StateExec`, `BlockExec`,
  `RegionAt`, `RegionExec`, `LoopExec`, `LoopGuard`, `LoopIter`, `LoopTail`,
  `CondExec` (`Res (Ctl × CSt)`):
  `R … (st.shift L) r ↔ ∃ r₀, R … st r₀ ∧ r = r₀.map (shift L)`
  (`*.shift_iff`); in the form "log `L` versus the empty log, final log
  `L.seq δ`": `NodeExec`, `SeqExec`, `OrderedExec`, `StateExec`,
  `BlockExec`, `RegionAt`, `RegionExec` `.log_iff`
  (`{st with log := L}` is `St.shift L {st with log := .empty}`,
  `St.withLog_eq_shift`);
* `IterExec` and `ConsumeTasks` do not depend on the log of the pre-scope
  state at all (every iteration and task starts from an empty log):
  `R … (pre.shift L) … r ↔ R … pre … r` (`*.shift_iff`);
* `CompsExec` and `SDFGExec` start from a memory, not a log (nothing to state).

No relation reads its incoming log, so no side condition is needed: every
consumer of a log (`scopeFinish`, `consumeFinish`, `nestedFinish`,
`stateFinish`, `St.afterNode`) only appends to it with `Log.seq`, which is
associative (`Log.seq_assoc`), and every log that *is* inspected
(`dropNames` of an iteration, `taskLog`, the trace deltas checked by
`State.hbFree`) is the log of a run that started from the empty log.

Errors are unaffected (`Res.map` keeps `.err e`).  The proofs commute every
leaf function of `Dace.SDFG.Semantics.Leaf` with `St.shift`, then go by
recursion on the derivation: the dataflow relations reuse their
sub-derivations unchanged (each node starts from `St.forNode`, each
iteration from `iterStart`), only the control-flow relations recurse
mutually.

Corollaries (`*.rel`): runs from two start states that differ only in their
logs, related by `Log.Le`, have related outcomes (`Res.Rel` with `St.Rel`,
`SeqRel`, or `CtlRel` built on `CSt.Rel`), in both directions.  For
dataflow states the start states must have equal `vals` lists (not merely
equal lookups as in `St.Rel`).
-/

set_option autoImplicit false

namespace Dace

/-! ## `Res` helpers -/

namespace Res

@[simp] theorem pure_eq {α : Type} (a : α) : (pure a : Res α) = .ok a := rfl
@[simp] theorem ok_bind {α β : Type} (a : α) (f : α → Res β) : (Res.ok a >>= f) = f a := rfl
@[simp] theorem err_bind {α β : Type} (e : ErrReason) (f : α → Res β) : (Res.err e >>= f) = Res.err e := rfl

/-- The composition lemma of the leaf proofs: a bind whose head commutes
with the shift (`sh₁`) commutes with `sh₂` if its continuation does. -/
theorem bind_shift {α β : Type} {sh₁ : α → α} {sh₂ : β → β} (r : Res α) {f g : α → Res β}
    (h : ∀ a, f (sh₁ a) = (g a).map sh₂) : (r.map sh₁ >>= f) = (r >>= g).map sh₂ := by
  cases r with
  | ok a => exact h a
  | err e => rfl

/-- The same for a head that does not depend on the log at all. -/
theorem bind_shift' {α β : Type} {sh₂ : β → β} (r : Res α) {f g : α → Res β}
    (h : ∀ a, f a = (g a).map sh₂) : (r >>= f) = (r >>= g).map sh₂ := by
  cases r with
  | ok a => exact h a
  | err e => rfl

theorem map_eq_ok {α β : Type} {r : Res α} {f : α → β} {b : β} :
    r.map f = .ok b ↔ ∃ a, r = .ok a ∧ f a = b := by
  cases r <;> simp [Res.map]

theorem map_eq_err {α β : Type} {r : Res α} {f : α → β} {e : ErrReason} :
    r.map f = .err e ↔ r = .err e := by
  cases r <;> simp [Res.map]

/-- Outcomes related by `R` when normal, equal when erroneous. -/
def Rel {α : Type} (R : α → α → Prop) : Res α → Res α → Prop
  | .ok a', .ok a => R a' a
  | .err e', .err e => e' = e
  | _, _ => False

theorem rel_map {α : Type} {R : α → α → Prop} {f g : α → α} (h : ∀ a, R (f a) (g a)) :
    ∀ r : Res α, Rel R (r.map f) (r.map g)
  | .ok a => h a
  | .err _ => rfl

/-- Transfer of a log-shift characterization (`hX`) to start states that
differ only in the prefixed log: every outcome from one has an `Rb`-related
outcome from the other, provided the two shifts are `Rb`-related. -/
theorem rel_of_shift {α β : Type} {X : α → Res β → Prop} {shα : Log → α → α} {sh : Log → β → β}
    {Rb : β → β → Prop} (hX : ∀ a L r, X (shα L a) r ↔ ∃ r₀, X a r₀ ∧ r = r₀.map (sh L))
    {g' g : Log} (hR : ∀ x, Rb (sh g' x) (sh g x)) (a₀ : α) :
    (∀ r, X (shα g a₀) r → ∃ r', X (shα g' a₀) r' ∧ Rel Rb r' r) ∧
    (∀ r', X (shα g' a₀) r' → ∃ r, X (shα g a₀) r ∧ Rel Rb r' r) := by
  constructor
  · intro r h
    obtain ⟨r₀, h₀, rfl⟩ := (hX _ _ _).1 h
    exact ⟨_, (hX _ _ _).2 ⟨r₀, h₀, rfl⟩, rel_map hR r₀⟩
  · intro r' h
    obtain ⟨r₀, h₀, rfl⟩ := (hX _ _ _).1 h
    exact ⟨_, (hX _ _ _).2 ⟨r₀, h₀, rfl⟩, rel_map hR r₀⟩

end Res

/-- `foldlM` in `Res` commutes with any map `sh` that commutes with every step. -/
theorem List.foldlM_res_comm {α β : Type} (sh : β → β) (f : β → α → Res β)
    (hf : ∀ b a, f (sh b) a = (f b a).map sh) (xs : List α) (b : β) :
    xs.foldlM f (sh b) = (xs.foldlM f b).map sh := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    rw [hf]
    cases f b x with
    | ok b' => exact ih b'
    | err e => rfl

/-! ## Shifting logs -/

namespace St

/-- Prefix the log `L` to the log of `st`. -/
def shift (L : Log) (st : St) : St := { st with log := L.seq st.log }

@[simp] theorem shift_mem (L : Log) (st : St) : (st.shift L).mem = st.mem := rfl
@[simp] theorem shift_log (L : Log) (st : St) : (st.shift L).log = L.seq st.log := rfl
@[simp] theorem shift_vals (L : Log) (st : St) : (st.shift L).vals = st.vals := rfl
@[simp] theorem shift_val? (L : Log) (st : St) (k : NodeId × Conn) : (st.shift L).val? k = st.val? k := rfl

/-- A state with log `L` is the state with the empty log, shifted by `L`. -/
theorem withLog_eq_shift (st : St) (L : Log) : { st with log := L } = St.shift L { st with log := Log.empty } := by
  simp [shift, Log.seq_empty]

@[simp] theorem shift_logged (L g : Log) (st : St) : (st.shift L).logged g = (st.logged g).shift L := by
  simp [shift, logged, Log.seq_assoc]

@[simp] theorem shift_setVal (L : Log) (st : St) (k : NodeId × Conn) (v : Val) :
    (st.shift L).setVal k v = (st.setVal k v).shift L := rfl

@[simp] theorem shift_bindAlias (L : Log) (st : St) (n : Name) (b : AliasBinding) :
    (st.shift L).bindAlias n b = (st.bindAlias n b).shift L := by
  simp [shift, bindAlias, Log.seq_assoc]

@[simp] theorem shift_forNode (L : Log) (st : St) : (st.shift L).forNode = st.forNode := rfl

@[simp] theorem shift_afterNode (L : Log) (st st' : St) :
    (st.shift L).afterNode st' = (st.afterNode st').shift L := by
  simp [shift, afterNode, Log.seq_assoc]

end St

namespace CSt

/-- Prefix the log `L` to the log of `cst`. -/
def shift (L : Log) (cst : CSt) : CSt := { cst with log := L.seq cst.log }

@[simp] theorem shift_ρ (L : Log) (cst : CSt) : (cst.shift L).ρ = cst.ρ := rfl
@[simp] theorem shift_mem (L : Log) (cst : CSt) : (cst.shift L).mem = cst.mem := rfl
@[simp] theorem shift_log (L : Log) (cst : CSt) : (cst.shift L).log = L.seq cst.log := rfl

theorem withLog_eq_shift (cst : CSt) (L : Log) :
    { cst with log := L } = CSt.shift L { cst with log := Log.empty } := by
  simp [shift, Log.seq_empty]

end CSt

/-- Shift the state of a control-flow outcome. -/
def shiftC (L : Log) : Ctl × CSt → Ctl × CSt := fun (c, cst) => (c, cst.shift L)

@[simp] theorem shiftC_apply (L : Log) (c : Ctl) (cst : CSt) : shiftC L (c, cst) = (c, cst.shift L) := rfl

/-! ## Primitive operations -/

namespace St

/-- Shift the state component of a pair. -/
abbrev shiftR {α : Type} (L : Log) : α × St → α × St := Prod.map id (St.shift L)

variable (ctx : Ctx) (ρ : Valuation) (L : Log)

theorem read_shift (st : St) (n : Name) (idx : Idx) (oob : Bool) :
    (st.shift L).read ctx ρ n idx oob = (st.read ctx ρ n idx oob).map (shiftR L) := by
  unfold read
  simp only [shift_mem]
  exact Res.bind_shift' _ fun l => by split <;> simp_all [Res.map]

theorem readMany_shift (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    (st.shift L).readMany ctx ρ n pts oob = (st.readMany ctx ρ n pts oob).map (shiftR L) := by
  unfold readMany
  exact List.foldlM_res_comm (shiftR L) _ (fun ⟨acc, st⟩ p => by
    simp only [shiftR, Prod.map_apply, id_eq, read_shift]
    exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl) pts ([], st)

theorem write_shift (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    (st.shift L).write ctx ρ n idx oob v = (st.write ctx ρ n idx oob v).map (St.shift L) := by
  unfold write
  simp only [shift_mem, shift_log, shift_vals]
  exact Res.bind_shift' _ fun l => by simp [Res.map, shift, Log.seq_assoc]

theorem wcr_shift (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    (st.shift L).wcr ctx ρ n idx oob op v = (st.wcr ctx ρ n idx oob op v).map (St.shift L) := by
  unfold wcr
  simp only [shift_mem, shift_log, shift_vals]
  exact Res.bind_shift' _ fun l => by simp [Res.map, shift, Log.seq_assoc]

theorem store_shift (st : St) (n : Name) (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    (st.shift L).store ctx ρ n idx oob w v = (st.store ctx ρ n idx oob w v).map (St.shift L) := by
  unfold store
  cases w with
  | none => exact write_shift ..
  | some op => exact wcr_shift ..

theorem queue_shift (st : St) (n : Name) (idx : Idx) :
    (st.shift L).queue ctx ρ n idx = st.queue ctx ρ n idx := rfl

theorem push_shift (st : St) (n : Name) (idx : Idx) (vs : List Val) :
    (st.shift L).push ctx ρ n idx vs = (st.push ctx ρ n idx vs).map (St.shift L) := by
  unfold push
  simp only [shift_mem, shift_log, shift_vals]
  exact Res.bind_shift' _ fun l => by by_cases h : vs.isEmpty <;> simp [h, Res.map, shift, Log.seq_assoc]

theorem popN_shift (st : St) (n : Name) (idx : Idx) (k : Nat) :
    (st.shift L).popN ctx ρ n idx k = (st.popN ctx ρ n idx k).map (shiftR L) := by
  unfold popN
  simp only [shift_mem, shift_log, shift_vals]
  refine Res.bind_shift' _ fun l => ?_
  by_cases h0 : k = 0
  · simp [h0, Res.map]
  · by_cases h1 : (st.mem.queues (st.mem.resolve l)).length < k <;> simp [h0, h1, Res.map, shift, Log.seq_assoc]

end St

/-! ## Leaf functions -/

section Leaf

open St (shiftR)

variable (ctx : Ctx) (ρ : Valuation) (L : Log)

theorem readScalarData_shift (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    readScalarData ctx ρ d sub oob (st.shift L) = (readScalarData ctx ρ d sub oob st).map (shiftR L) := by
  unfold readScalarData
  simp only [St.shift_mem]
  refine Res.bind_shift' _ fun pts => ?_
  split
  · split
    · rw [St.popN_shift]
      exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl
    · rw [St.read_shift]
      exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl
  · rfl

theorem copyData_shift (st : St) (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (w : Option WcrOp) (oob : Bool) :
    copyData ctx ρ (st.shift L) sd ssub dd dsub w oob = (copyData ctx ρ st sd ssub dd dsub w oob).map (St.shift L) := by
  unfold copyData
  simp only [St.shift_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun dpts => Res.bind_shift' _ fun sp => ?_
    rw [St.popN_shift]
    refine Res.bind_shift _ fun ⟨vs, st'⟩ => ?_
    exact List.foldlM_res_comm (St.shift L) _ (fun st ⟨q, v⟩ => St.store_shift ..) _ _
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dp => ?_
    rw [St.readMany_shift]
    exact Res.bind_shift _ fun ⟨vs, st'⟩ => St.push_shift ..
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dpts => ?_
    refine List.foldlM_res_comm (St.shift L) _ (fun st ⟨p, q⟩ => ?_) _ _
    simp only
    rw [St.read_shift]
    exact Res.bind_shift _ fun ⟨v, st'⟩ => St.store_shift ..

theorem execCopy_shift (s : State) (e : DEdge) (st : St) :
    execCopy ctx ρ s e (st.shift L) = (execCopy ctx ρ s e st).map (St.shift L) := by
  unfold execCopy
  split
  · rfl
  · split
    · exact copyData_shift ..
    · rfl

theorem bindInData_shift (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    bindInData ctx ρ d sub oob (st.shift L) = (bindInData ctx ρ d sub oob st).map (shiftR L) := by
  unfold bindInData
  simp only [St.shift_mem]
  split
  · exact Res.bind_shift' _ fun p => Res.bind_shift' _ fun q => rfl
  · refine Res.bind_shift' _ fun pts => ?_
    split
    · rw [St.read_shift]
      exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl
    · refine Res.bind_shift' _ fun lo => ?_
      rw [St.readMany_shift]
      exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl

theorem bindIn_shift (s : State) (e : DEdge) (st : St) :
    bindIn ctx ρ s e (st.shift L) = (bindIn ctx ρ s e st).map (shiftR L) := by
  unfold bindIn
  split
  · rfl
  · rw [bindInData_shift]
    exact Res.bind_shift _ fun ⟨c, st'⟩ => rfl
  · rfl

theorem bindInputs_shift (s : State) (n : DNode) (st : St) :
    bindInputs ctx ρ s n (st.shift L) = (bindInputs ctx ρ s n st).map (shiftR L) := by
  unfold bindInputs
  rw [show (([] : List (Conn × ConnIn)), st.shift L) = shiftR L ([], st) from rfl,
    List.foldlM_res_comm (shiftR L) _ ?hstep]
  · exact Res.bind_shift _ fun ⟨bs, st'⟩ => rfl
  case hstep =>
    intro ⟨acc, st⟩ e
    simp only [Prod.map_apply, id_eq]
    split
    · rfl
    · rw [bindIn_shift]
      refine Res.bind_shift _ fun ⟨ci, st'⟩ => ?_
      cases ci <;> rfl

theorem storeScalar_shift (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool) (v : Val) :
    storeScalar ctx ρ (st.shift L) d sub w oob v = (storeScalar ctx ρ st d sub w oob v).map (St.shift L) := by
  unfold storeScalar
  simp only [St.shift_mem]
  split
  · exact Res.bind_shift' _ fun p => St.push_shift ..
  · refine Res.bind_shift' _ fun pts => ?_
    split
    · exact St.store_shift ..
    · rfl
    · rfl

theorem applyScalar_shift (s : State) (n : DNode) (st : St) (c : Conn) (v : Val) :
    applyScalar ctx ρ s n (st.shift L) c v = (applyScalar ctx ρ s n st c v).map (St.shift L) := by
  unfold applyScalar
  refine List.foldlM_res_comm (St.shift L) _ (fun st e => ?_) _ _
  split
  · rfl
  · split
    · rfl
    · exact storeScalar_shift ..
    · rfl

theorem storeArrayWrite_shift (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool)
    (rel : Idx) (v : Val) :
    storeArrayWrite ctx ρ (st.shift L) d sub w oob rel v =
      (storeArrayWrite ctx ρ st d sub w oob rel v).map (St.shift L) := by
  unfold storeArrayWrite
  simp only [St.shift_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun lo => ?_
    split
    · refine Res.bind_shift' _ fun pts => ?_
      split
      · exact St.store_shift ..
      · rfl
    · rfl

theorem applyArrayWrite_shift (s : State) (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    applyArrayWrite ctx ρ s n (st.shift L) c rel v = (applyArrayWrite ctx ρ s n st c rel v).map (St.shift L) := by
  unfold applyArrayWrite
  refine List.foldlM_res_comm (St.shift L) _ (fun st e => ?_) _ _
  split
  · rfl
  · split
    · exact storeArrayWrite_shift ..
    · rfl

theorem pushTo_shift (st : St) (d : Name) (sub : Option Subset) (vs : List Val) :
    pushTo ctx ρ (st.shift L) d sub vs = (pushTo ctx ρ st d sub vs).map (St.shift L) := by
  unfold pushTo
  simp only [St.shift_mem]
  split
  · exact Res.bind_shift' _ fun p => St.push_shift ..
  · rfl

theorem applyPush_shift (s : State) (n : DNode) (st : St) (c : Conn) (vs : List Val) :
    applyPush ctx ρ s n (st.shift L) c vs = (applyPush ctx ρ s n st c vs).map (St.shift L) := by
  unfold applyPush
  refine List.foldlM_res_comm (St.shift L) _ (fun st e => ?_) _ _
  split
  · exact pushTo_shift ..
  · rfl

theorem popFrom_shift (st : St) (d : Name) (sub : Option Subset) (k : Nat) :
    popFrom ctx ρ (st.shift L) d sub k = (popFrom ctx ρ st d sub k).map (St.shift L) := by
  unfold popFrom
  simp only [St.shift_mem]
  split
  · refine Res.bind_shift' _ fun p => ?_
    rw [St.popN_shift]
    exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl
  · rfl

theorem applyPop_shift (s : State) (n : DNode) (st : St) (c : Conn) (k : Nat) :
    applyPop ctx ρ s n (st.shift L) c k = (applyPop ctx ρ s n st c k).map (St.shift L) := by
  unfold applyPop
  split
  · rfl
  · split
    · exact popFrom_shift ..
    · rfl

theorem applyOut_shift (s : State) (n : DNode) (st : St) (out : TaskletOut) :
    applyOut ctx ρ s n (st.shift L) out = (applyOut ctx ρ s n st out).map (St.shift L) := by
  unfold applyOut
  rw [List.foldlM_res_comm (St.shift L) _ (fun st ⟨c, k⟩ => applyPop_shift ..)]
  refine Res.bind_shift _ fun st => ?_
  rw [List.foldlM_res_comm (St.shift L) _ (fun st ⟨c, v⟩ => applyScalar_shift ..)]
  refine Res.bind_shift _ fun st => ?_
  rw [List.foldlM_res_comm (St.shift L) _ (fun st ⟨c, rel, v⟩ => applyArrayWrite_shift ..)]
  refine Res.bind_shift _ fun st => ?_
  exact List.foldlM_res_comm (St.shift L) _ (fun st ⟨c, vs⟩ => applyPush_shift ..) _ _

theorem execCode_shift (s : State) (n : DNode) (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut)
    (ue : ErrReason) :
    execCode ctx ρ s n (st.shift L) run ue = (execCode ctx ρ s n st run ue).map (St.shift L) := by
  unfold execCode
  split
  · rfl
  · rw [bindInputs_shift]
    refine Res.bind_shift _ fun ⟨ins, st'⟩ => ?_
    simp only [Prod.map_apply, id_eq]
    split
    · rfl
    · exact applyOut_shift ..

theorem reduceData_shift (st : St) (props : ReduceProps) (sd : Name) (ssub : Option Subset) (dd : Name)
    (dsub : Option Subset) (w : Option WcrOp) (soob doob : Bool) :
    reduceData ctx ρ (st.shift L) props sd ssub dd dsub w soob doob =
      (reduceData ctx ρ st props sd ssub dd dsub w soob doob).map (St.shift L) := by
  unfold reduceData
  simp only [St.shift_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dpts => ?_
    -- the shape check (and the grouping it depends on) does not read the state
    split <;> (try split) <;> first
      | rfl
      | rw [St.readMany_shift]
        refine Res.bind_shift _ fun ⟨vals, st'⟩ => ?_
        refine List.foldlM_res_comm (St.shift L) _ (fun st ⟨g, q⟩ => ?_) _ _
        simp only [St.shift_mem]
        repeat' split
        all_goals first
          | rfl
          | exact Res.bind_shift' _ fun i => St.store_shift ..
          | exact St.wcr_shift ..

theorem execReduce_shift (s : State) (n : DNode) (props : ReduceProps) (st : St) :
    execReduce ctx ρ s n props (st.shift L) = (execReduce ctx ρ s n props st).map (St.shift L) := by
  unfold execReduce
  dsimp only
  split
  · split
    · exact reduceData_shift ..
    · rfl
  · rfl

theorem aliasTo_shift (st : St) (d base : Name) (sub : Option Subset) :
    aliasTo ctx ρ (st.shift L) d base sub = (aliasTo ctx ρ st d base sub).map (St.shift L) := by
  unfold aliasTo
  simp only [St.shift_mem]
  split
  · rfl
  · repeat (refine Res.bind_shift' _ fun _ => ?_)
    simp [Res.map]

theorem refSet_shift (s : State) (e : DEdge) (d : Name) (st : St) :
    refSet ctx ρ s e d (st.shift L) = (refSet ctx ρ s e d st).map (St.shift L) := by
  unfold refSet
  split
  · exact aliasTo_shift ..
  · rfl
  · rfl

theorem execCopiesInto_shift (s : State) (n : DNode) (st : St) :
    execCopiesInto ctx ρ s n (st.shift L) = (execCopiesInto ctx ρ s n st).map (St.shift L) := by
  unfold execCopiesInto
  refine List.foldlM_res_comm (St.shift L) _ (fun st e => ?_) _ _
  split
  · rfl
  · split
    · rfl
    · split
      · exact refSet_shift ..
      · split
        · exact execCopy_shift ..
        · rfl

theorem execSimple_shift (s : State) (n : DNode) (st : St) :
    execSimple ctx ρ s n (st.shift L) = (execSimple ctx ρ s n st).map (St.shift L) := by
  unfold execSimple
  split <;> first
    | exact execCopiesInto_shift ..
    | exact execCode_shift ..
    | rfl
    | split
      · exact execReduce_shift ..
      · exact execCode_shift ..

theorem scopeFinish_shift (pre : St) (logs : List Log) :
    scopeFinish ctx (pre.shift L) logs = (scopeFinish ctx pre logs).map (St.shift L) := by
  unfold scopeFinish
  simp only [St.shift_mem]
  split <;> simp [Res.map, St.shift, Log.seq_assoc]

theorem consumeFinish_shift (pre : St) (l : Loc) (logs : List Log) (rest pushed : List Val) :
    consumeFinish ctx (pre.shift L) l logs rest pushed =
      (consumeFinish ctx pre l logs rest pushed).map (St.shift L) := by
  unfold consumeFinish
  simp only [St.shift_mem]
  split <;> simp [Res.map, St.shift, Log.seq_assoc]

@[simp] theorem iterStart_shift (priv : List Name) (pre : St) : iterStart priv (pre.shift L) = iterStart priv pre := rfl

@[simp] theorem taskStart_shift (s : State) (entry : NodeId) (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : St)
    (q : List Val) : taskStart ctx s entry c l priv (pre.shift L) q = taskStart ctx s entry c l priv pre q := rfl

theorem trueEdges_shift (rg : Region) (cst : CSt) (bid : BlockId) :
    trueEdges ctx rg (cst.shift L) bid = trueEdges ctx rg cst bid := rfl

theorem applyAssigns_shift (cst : CSt) (as : List (Sym × Expr)) :
    applyAssigns ctx (cst.shift L) as = (applyAssigns ctx cst as).map (CSt.shift L) := by
  unfold applyAssigns
  exact Res.bind_shift' _ fun vs => rfl

theorem applyAssign?_shift (cst : CSt) (a : Option (Sym × Expr)) :
    applyAssign? ctx (cst.shift L) a = (applyAssign? ctx cst a).map (CSt.shift L) := by
  cases a with
  | none => rfl
  | some a => exact applyAssigns_shift ..

theorem stateFinish_shift (cst : CSt) (logs : List Log) :
    stateFinish ctx (cst.shift L) logs = (stateFinish ctx cst logs).map (shiftC L) := by
  unfold stateFinish
  simp only [CSt.shift_mem]
  split <;> simp [Res.map, CSt.shift, Log.seq_assoc]

end Leaf

/-! ## Dataflow relations -/

section Dataflow

open St (shiftR)

variable {ctx : Ctx} {ρ : Valuation} {s : State}

theorem SeqExec.shift {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} (L : Log) :
    SeqExec ctx ρ s ids st r → SeqExec ctx ρ s ids (st.shift L) (r.map (Prod.map (St.shift L) id))
  | .nil => .nil
  | .missing h => .missing h
  | .headErr h hn => .headErr h hn
  | .tailErr h hn hr => .tailErr h hn (by rw [St.shift_afterNode]; exact SeqExec.shift L hr)
  | .cons h hn hr => .cons h hn (by rw [St.shift_afterNode]; exact SeqExec.shift L hr)

theorem SeqExec.unshift {ids : List NodeId} {st' st : St} {L : Log} {r : Res (St × List (NodeId × Log))}
    (heq : st' = st.shift L) : SeqExec ctx ρ s ids st' r →
      ∃ r₀, SeqExec ctx ρ s ids st r₀ ∧ r = r₀.map (Prod.map (St.shift L) id)
  | .nil => ⟨_, .nil, by rw [heq]; rfl⟩
  | .missing h => ⟨_, .missing h, rfl⟩
  | .headErr h hn => by
    rw [heq] at hn
    exact ⟨_, .headErr h hn, rfl⟩
  | .tailErr h hn hr => by
    rw [heq] at hn hr
    obtain ⟨r₀, h₀, he⟩ := SeqExec.unshift (St.shift_afterNode L st _) hr
    rw [eq_comm, Res.map_eq_err] at he
    subst he
    exact ⟨_, .tailErr h hn h₀, rfl⟩
  | .cons h hn hr => by
    rw [heq] at hn hr
    obtain ⟨r₀, h₀, he⟩ := SeqExec.unshift (St.shift_afterNode L st _) hr
    rw [eq_comm, Res.map_eq_ok] at he
    obtain ⟨⟨st₂, tr⟩, rfl, he⟩ := he
    cases he
    exact ⟨_, .cons h hn h₀, rfl⟩

/-- **Logs are write-only** (a sequential execution; the trace is unchanged). -/
theorem SeqExec.shift_iff {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} (L : Log) :
    SeqExec ctx ρ s ids (st.shift L) r ↔
      ∃ r₀, SeqExec ctx ρ s ids st r₀ ∧ r = r₀.map (Prod.map (St.shift L) id) :=
  ⟨SeqExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

end Dataflow

/-! ## Control-flow relations -/

section Control

variable {ctx : Ctx}

theorem shiftC_eq_ok {L : Log} {r₀ : Res (Ctl × CSt)} {c : Ctl} {x : CSt}
    (h : Res.ok (c, x) = r₀.map (shiftC L)) : ∃ y, r₀ = .ok (c, y) ∧ x = y.shift L := by
  cases r₀ with
  | err e => cases h
  | ok a =>
    obtain ⟨c', y⟩ := a
    simp only [Res.map_ok, shiftC_apply, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨y, rfl, rfl⟩

theorem shiftC_eq_err {L : Log} {r₀ : Res (Ctl × CSt)} {e : ErrReason}
    (h : Res.err e = r₀.map (shiftC L)) : r₀ = .err e :=
  Res.map_eq_err.1 h.symm

theorem StateExec.shift {cst : CSt} {s : State} {r : Res (Ctl × CSt)} (L : Log) (h : StateExec ctx cst s r) :
    StateExec ctx (cst.shift L) s (r.map (shiftC L)) := by
  cases h with
  | compsErr hc => exact .compsErr hc
  | run hc =>
    rw [← stateFinish_shift]
    exact .run hc

theorem StateExec.unshift {cst' cst : CSt} {L : Log} {s : State} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L)
    (h : StateExec ctx cst' s r) : ∃ r₀, StateExec ctx cst s r₀ ∧ r = r₀.map (shiftC L) := by
  subst heq
  cases h with
  | compsErr hc => exact ⟨_, .compsErr hc, rfl⟩
  | run hc => exact ⟨_, .run hc, stateFinish_shift ..⟩

mutual

theorem BlockExec.shift {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} (L : Log) :
    BlockExec ctx cst b r → BlockExec ctx (cst.shift L) b (r.map (shiftC L))
  | .state h => .state (h.shift L)
  | .region h => .region (RegionExec.shift L h)
  | .loop h => .loop (LoopExec.shift L h)
  | .cond h => .cond (CondExec.shift L h)
  | .brk => .brk
  | .cont => .cont
  | .ret => .ret

theorem RegionAt.shift {rg : Region} {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)} (L : Log) :
    RegionAt ctx rg cst bid r → RegionAt ctx rg (cst.shift L) bid (r.map (shiftC L))
  | .missing hb => .missing hb
  | .blockErr hb h => .blockErr hb (BlockExec.shift L h)
  | .exit hb h hc => .exit hb (BlockExec.shift L h) hc
  | .edgesErr hb h he => .edgesErr hb (BlockExec.shift L h) he
  | .stop hb h he => .stop hb (BlockExec.shift L h) he
  | .ambiguous hb h he => .ambiguous hb (BlockExec.shift L h) he
  | .assignErr hb h he ha =>
    .assignErr hb (BlockExec.shift L h) he ((applyAssigns_shift ..).trans (congrArg _ ha))
  | .step hb h he ha hn =>
    .step hb (BlockExec.shift L h) he ((applyAssigns_shift ..).trans (congrArg _ ha)) (RegionAt.shift L hn)

theorem RegionExec.shift {cst : CSt} {rg : Region} {r : Res (Ctl × CSt)} (L : Log) :
    RegionExec ctx cst rg r → RegionExec ctx (cst.shift L) rg (r.map (shiftC L))
  | .start h => .start (RegionAt.shift L h)

theorem LoopExec.shift {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} (L : Log) :
    LoopExec ctx cst l r → LoopExec ctx (cst.shift L) l (r.map (shiftC L))
  | .initErr hi => .initErr ((applyAssign?_shift ..).trans (congrArg _ hi))
  | .guardFirst hi hv hg => .guardFirst ((applyAssign?_shift ..).trans (congrArg _ hi)) hv (LoopGuard.shift L hg)
  | .bodyFirst hi hv hb => .bodyFirst ((applyAssign?_shift ..).trans (congrArg _ hi)) hv (LoopIter.shift L hb)

theorem LoopGuard.shift {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} (L : Log) :
    LoopGuard ctx cst l r → LoopGuard ctx (cst.shift L) l (r.map (shiftC L))
  | .condErr hc => .condErr hc
  | .exit hc => .exit hc
  | .enter hc hb => .enter hc (LoopIter.shift L hb)

theorem LoopIter.shift {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} (L : Log) :
    LoopIter ctx cst l r → LoopIter ctx (cst.shift L) l (r.map (shiftC L))
  | .bodyErr hb => .bodyErr (RegionExec.shift L hb)
  | .ret hb => .ret (RegionExec.shift L hb)
  | .brk hb => .brk (RegionExec.shift L hb)
  | .tail hb hc ht => .tail (RegionExec.shift L hb) hc (LoopTail.shift L ht)

theorem LoopTail.shift {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} (L : Log) :
    LoopTail ctx cst l r → LoopTail ctx (cst.shift L) l (r.map (shiftC L))
  | .updateErr hv hu => .updateErr hv ((applyAssign?_shift ..).trans (congrArg _ hu))
  | .updateGuard hv hu hg => .updateGuard hv ((applyAssign?_shift ..).trans (congrArg _ hu)) (LoopGuard.shift L hg)
  | .condFirstErr hv hc => .condFirstErr hv hc
  | .condFirstExit hv hc => .condFirstExit hv hc
  | .condFirstUpdateErr hv hc hu => .condFirstUpdateErr hv hc ((applyAssign?_shift ..).trans (congrArg _ hu))
  | .condFirstNext hv hc hu hb =>
    .condFirstNext hv hc ((applyAssign?_shift ..).trans (congrArg _ hu)) (LoopIter.shift L hb)

theorem CondExec.shift {cst : CSt} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)} (L : Log) :
    CondExec ctx cst brs r → CondExec ctx (cst.shift L) brs (r.map (shiftC L))
  | .nil => .nil
  | .els h => .els (RegionExec.shift L h)
  | .condErr hc => .condErr hc
  | .taken hc h => .taken hc (RegionExec.shift L h)
  | .skipped hc h => .skipped hc (CondExec.shift L h)

end

mutual

theorem BlockExec.unshift {cst' cst : CSt} {L : Log} {b : Block} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    BlockExec ctx cst' b r → ∃ r₀, BlockExec ctx cst b r₀ ∧ r = r₀.map (shiftC L)
  | .state h => let ⟨_, h₀, he⟩ := StateExec.unshift heq h; ⟨_, .state h₀, he⟩
  | .region h => let ⟨_, h₀, he⟩ := RegionExec.unshift heq h; ⟨_, .region h₀, he⟩
  | .loop h => let ⟨_, h₀, he⟩ := LoopExec.unshift heq h; ⟨_, .loop h₀, he⟩
  | .cond h => let ⟨_, h₀, he⟩ := CondExec.unshift heq h; ⟨_, .cond h₀, he⟩
  | .brk => ⟨_, .brk, by rw [heq]; rfl⟩
  | .cont => ⟨_, .cont, by rw [heq]; rfl⟩
  | .ret => ⟨_, .ret, by rw [heq]; rfl⟩

theorem RegionAt.unshift {rg : Region} {cst' cst : CSt} {L : Log} {bid : BlockId} {r : Res (Ctl × CSt)}
    (heq : cst' = cst.shift L) :
    RegionAt ctx rg cst' bid r → ∃ r₀, RegionAt ctx rg cst bid r₀ ∧ r = r₀.map (shiftC L)
  | .missing hb => ⟨_, .missing hb, rfl⟩
  | .blockErr hb h => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    cases shiftC_eq_err he
    exact ⟨_, .blockErr hb h₀, rfl⟩
  | .exit hb h hc => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .exit hb h₀ hc, rfl⟩
  | .edgesErr hb h he' => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .edgesErr hb h₀ he', rfl⟩
  | .stop hb h he' => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .stop hb h₀ he', rfl⟩
  | .ambiguous hb h he' => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .ambiguous hb h₀ he', rfl⟩
  | .assignErr hb h he' ha => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    rw [applyAssigns_shift, Res.map_eq_err] at ha
    exact ⟨_, .assignErr hb h₀ he' ha, rfl⟩
  | .step hb h he' ha hn => by
    obtain ⟨r₀, h₀, he⟩ := BlockExec.unshift heq h
    obtain ⟨y, rfl, hy⟩ := shiftC_eq_ok he
    subst hy
    rw [applyAssigns_shift, Res.map_eq_ok] at ha
    obtain ⟨cst₂, ha, h₂⟩ := ha
    obtain ⟨r₁, h₁, he₁⟩ := RegionAt.unshift h₂.symm hn
    exact ⟨_, .step hb h₀ he' ha h₁, he₁⟩

theorem RegionExec.unshift {cst' cst : CSt} {L : Log} {rg : Region} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    RegionExec ctx cst' rg r → ∃ r₀, RegionExec ctx cst rg r₀ ∧ r = r₀.map (shiftC L)
  | .start h => let ⟨_, h₀, he⟩ := RegionAt.unshift heq h; ⟨_, .start h₀, he⟩

theorem LoopExec.unshift {cst' cst : CSt} {L : Log} {l : LoopRegion} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    LoopExec ctx cst' l r → ∃ r₀, LoopExec ctx cst l r₀ ∧ r = r₀.map (shiftC L)
  | .initErr hi => by
    rw [heq, applyAssign?_shift, Res.map_eq_err] at hi
    exact ⟨_, .initErr hi, rfl⟩
  | .guardFirst hi hv hg => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hi
    obtain ⟨cst₁, hi, h₁⟩ := hi
    obtain ⟨_, h₀, he⟩ := LoopGuard.unshift h₁.symm hg
    exact ⟨_, .guardFirst hi hv h₀, he⟩
  | .bodyFirst hi hv hb => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hi
    obtain ⟨cst₁, hi, h₁⟩ := hi
    obtain ⟨_, h₀, he⟩ := LoopIter.unshift h₁.symm hb
    exact ⟨_, .bodyFirst hi hv h₀, he⟩

theorem LoopGuard.unshift {cst' cst : CSt} {L : Log} {l : LoopRegion} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    LoopGuard ctx cst' l r → ∃ r₀, LoopGuard ctx cst l r₀ ∧ r = r₀.map (shiftC L)
  | .condErr hc => by
    subst heq
    exact ⟨_, .condErr hc, rfl⟩
  | .exit hc => by
    subst heq
    exact ⟨_, .exit hc, rfl⟩
  | .enter hc hb => by
    obtain ⟨_, h₀, he⟩ := LoopIter.unshift heq hb
    subst heq
    exact ⟨_, .enter hc h₀, he⟩

theorem LoopIter.unshift {cst' cst : CSt} {L : Log} {l : LoopRegion} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    LoopIter ctx cst' l r → ∃ r₀, LoopIter ctx cst l r₀ ∧ r = r₀.map (shiftC L)
  | .bodyErr hb => by
    obtain ⟨r₀, h₀, he⟩ := RegionExec.unshift heq hb
    cases shiftC_eq_err he
    exact ⟨_, .bodyErr h₀, rfl⟩
  | .ret hb => by
    obtain ⟨r₀, h₀, he⟩ := RegionExec.unshift heq hb
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .ret h₀, rfl⟩
  | .brk hb => by
    obtain ⟨r₀, h₀, he⟩ := RegionExec.unshift heq hb
    obtain ⟨y, rfl, rfl⟩ := shiftC_eq_ok he
    exact ⟨_, .brk h₀, rfl⟩
  | .tail hb hc ht => by
    obtain ⟨r₀, h₀, he⟩ := RegionExec.unshift heq hb
    obtain ⟨y, rfl, hy⟩ := shiftC_eq_ok he
    obtain ⟨_, h₁, he₁⟩ := LoopTail.unshift hy ht
    exact ⟨_, .tail h₀ hc h₁, he₁⟩

theorem LoopTail.unshift {cst' cst : CSt} {L : Log} {l : LoopRegion} {r : Res (Ctl × CSt)} (heq : cst' = cst.shift L) :
    LoopTail ctx cst' l r → ∃ r₀, LoopTail ctx cst l r₀ ∧ r = r₀.map (shiftC L)
  | .updateErr hv hu => by
    rw [heq, applyAssign?_shift, Res.map_eq_err] at hu
    exact ⟨_, .updateErr hv hu, rfl⟩
  | .updateGuard hv hu hg => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hu
    obtain ⟨cst₁, hu, h₁⟩ := hu
    obtain ⟨_, h₀, he⟩ := LoopGuard.unshift h₁.symm hg
    exact ⟨_, .updateGuard hv hu h₀, he⟩
  | .condFirstErr hv hc => by
    subst heq
    exact ⟨_, .condFirstErr hv hc, rfl⟩
  | .condFirstExit hv hc => by
    subst heq
    exact ⟨_, .condFirstExit hv hc, rfl⟩
  | .condFirstUpdateErr hv hc hu => by
    subst heq
    rw [applyAssign?_shift, Res.map_eq_err] at hu
    exact ⟨_, .condFirstUpdateErr hv hc hu, rfl⟩
  | .condFirstNext hv hc hu hb => by
    rw [heq, applyAssign?_shift, Res.map_eq_ok] at hu
    obtain ⟨cst₁, hu, h₁⟩ := hu
    obtain ⟨_, h₀, he⟩ := LoopIter.unshift h₁.symm hb
    subst heq
    exact ⟨_, .condFirstNext hv hc hu h₀, he⟩

theorem CondExec.unshift {cst' cst : CSt} {L : Log} {brs : List (Option Expr × Region)} {r : Res (Ctl × CSt)}
    (heq : cst' = cst.shift L) :
    CondExec ctx cst' brs r → ∃ r₀, CondExec ctx cst brs r₀ ∧ r = r₀.map (shiftC L)
  | .nil => ⟨_, .nil, by rw [heq]; rfl⟩
  | .els h => let ⟨_, h₀, he⟩ := RegionExec.unshift heq h; ⟨_, .els h₀, he⟩
  | .condErr hc => by
    subst heq
    exact ⟨_, .condErr hc, rfl⟩
  | .taken hc h => by
    obtain ⟨_, h₀, he⟩ := RegionExec.unshift heq h
    subst heq
    exact ⟨_, .taken hc h₀, he⟩
  | .skipped hc h => by
    obtain ⟨_, h₀, he⟩ := CondExec.unshift heq h
    subst heq
    exact ⟨_, .skipped hc h₀, he⟩

end

/-! ### Logs are write-only (control flow) -/

variable {cst : CSt} {r : Res (Ctl × CSt)}

theorem BlockExec.shift_iff {b : Block} (L : Log) :
    BlockExec ctx (cst.shift L) b r ↔ ∃ r₀, BlockExec ctx cst b r₀ ∧ r = r₀.map (shiftC L) :=
  ⟨BlockExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

theorem RegionExec.shift_iff {rg : Region} (L : Log) :
    RegionExec ctx (cst.shift L) rg r ↔ ∃ r₀, RegionExec ctx cst rg r₀ ∧ r = r₀.map (shiftC L) :=
  ⟨RegionExec.unshift rfl, fun ⟨_, h, he⟩ => he ▸ h.shift L⟩

/-! ### Related start states (control flow) -/

/-- Related control-flow outcomes: same signal, `CSt.Rel` states. -/
def CtlRel (x' x : Ctl × CSt) : Prop := x'.1 = x.1 ∧ CSt.Rel x'.2 x.2

theorem CSt.Rel.shift {g' g : Log} (h : Log.Le g' g) (x : CSt) : CSt.Rel (x.shift g') (x.shift g) :=
  ⟨rfl, rfl, Log.Le.seq h (Log.Le.refl _)⟩

theorem CtlRel.shiftC {g' g : Log} (h : Log.Le g' g) (x : Ctl × CSt) : CtlRel (shiftC g' x) (shiftC g x) :=
  ⟨rfl, CSt.Rel.shift h x.2⟩

/-- Two `CSt.Rel`-related states are shifts of one state by related logs. -/
theorem CSt.Rel.eq_shift {a' a : CSt} (h : CSt.Rel a' a) :
    a' = CSt.shift a'.log { ρ := a.ρ, mem := a.mem } ∧ a = CSt.shift a.log { ρ := a.ρ, mem := a.mem } := by
  obtain ⟨hρ, hm, -⟩ := h
  cases a'; cases a
  simp only at hρ hm
  subst hρ hm
  exact ⟨by simp [CSt.shift, Log.seq_empty], by simp [CSt.shift, Log.seq_empty]⟩

/-- Generic transfer for relations over control-flow states: `CSt.Rel`-related
start states have `Rb`-related outcomes, in both directions. -/
theorem CSt.Rel.transferWith {β : Type} {X : CSt → Res β → Prop} {sh : Log → β → β} {Rb : β → β → Prop}
    (hX : ∀ a L r, X (a.shift L) r ↔ ∃ r₀, X a r₀ ∧ r = r₀.map (sh L))
    (hR : ∀ {g' g : Log}, Log.Le g' g → ∀ x, Rb (sh g' x) (sh g x)) {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, X a r → ∃ r', X a' r' ∧ Res.Rel Rb r' r) ∧ (∀ r', X a' r' → ∃ r, X a r ∧ Res.Rel Rb r' r) := by
  obtain ⟨h₁, h₂⟩ := h.eq_shift
  rw [h₁, h₂]
  exact Res.rel_of_shift (shα := CSt.shift) hX (hR h.2.2) _

/-- Generic transfer for the SDFG control-flow relations. -/
theorem CSt.Rel.transfer {X : CSt → Res (Ctl × CSt) → Prop}
    (hX : ∀ a L r, X (a.shift L) r ↔ ∃ r₀, X a r₀ ∧ r = r₀.map (shiftC L)) {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, X a r → ∃ r', X a' r' ∧ Res.Rel CtlRel r' r) ∧
    (∀ r', X a' r' → ∃ r, X a r ∧ Res.Rel CtlRel r' r) :=
  CSt.Rel.transferWith hX CtlRel.shiftC h

theorem BlockExec.rel {b : Block} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, BlockExec ctx a b r → ∃ r', BlockExec ctx a' b r' ∧ Res.Rel CtlRel r' r) ∧
    (∀ r', BlockExec ctx a' b r' → ∃ r, BlockExec ctx a b r ∧ Res.Rel CtlRel r' r) :=
  CSt.Rel.transfer (X := fun a r => BlockExec ctx a b r) (fun _ L _ => BlockExec.shift_iff L) h

theorem RegionExec.rel {rg : Region} {a' a : CSt} (h : CSt.Rel a' a) :
    (∀ r, RegionExec ctx a rg r → ∃ r', RegionExec ctx a' rg r' ∧ Res.Rel CtlRel r' r) ∧
    (∀ r', RegionExec ctx a' rg r' → ∃ r, RegionExec ctx a rg r ∧ Res.Rel CtlRel r' r) :=
  CSt.Rel.transfer (X := fun a r => RegionExec ctx a rg r) (fun _ L _ => RegionExec.shift_iff L) h

end Control

/-! ## Dataflow: states that differ only in their logs -/

section DataflowRel

variable {ctx : Ctx} {ρ : Valuation} {s : State}

/-- Two states with the same memory and code-edge values are shifts of one
state. -/
theorem St.eq_shift_of {a' a : St} (hm : a'.mem = a.mem) (hv : a'.vals = a.vals) :
    a' = St.shift a'.log { mem := a.mem, vals := a.vals } ∧ a = St.shift a.log { mem := a.mem, vals := a.vals } := by
  cases a'; cases a
  simp only at hm hv
  subst hm hv
  exact ⟨by simp [St.shift, Log.seq_empty], by simp [St.shift, Log.seq_empty]⟩

end DataflowRel

end Dace
