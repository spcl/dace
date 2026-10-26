-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Frame

/-!
# Coherence of every sequential run

Every normal run of a node, a node sequence or a scope body changes the
memory by exactly what it logs (`St.StepTo`: the final memory is the
initial one with a delta applied, and the final log is the initial one
followed by that delta), whatever the containers: plain cells, streams
(queue effects), views and references (alias bindings).  Code-edge values
(`St.vals`) are not logged and may change.

This is the part of `Dace.SDFG.Frame` the pair checker needs (coherence),
without its fragment: the frame property needs plain deltas and pure
expressions, coherence does not.  Nested SDFG calls and the allocation of
transients are the exceptions (allocation is not logged): nested calls are
covered when they allocate nothing (`DNode.NestedFlat`), allocation inside
map iterations and consume tasks is invisible from outside (their logs are
merged).
-/

set_option autoImplicit false

namespace Dace

/-- `a` is `st` changed by a logged delta; with `kv` the code-edge values are
kept too (they are not logged). -/
def St.StepTo (w : WcrSem) (kv : Bool) (st a : St) : Prop :=
  ∃ δ : Log, a.mem = δ.apply w st.mem ∧ a.log = st.log.seq δ ∧ (kv = true → a.vals = st.vals)

namespace St.StepTo

variable {w : WcrSem} {kv : Bool}

theorem refl (w : WcrSem) (st : St) : St.StepTo w kv st st := ⟨Log.empty, rfl, (Log.seq_empty _).symm, fun _ => rfl⟩

theorem trans {a b c : St} (h₁ : St.StepTo w kv a b) (h₂ : St.StepTo w kv b c) : St.StepTo w kv a c := by
  obtain ⟨δ₁, hm₁, hl₁, hv₁⟩ := h₁
  obtain ⟨δ₂, hm₂, hl₂, hv₂⟩ := h₂
  exact ⟨δ₁.seq δ₂, by rw [hm₂, hm₁, Log.apply_seq], by rw [hl₂, hl₁, Log.seq_assoc],
    fun h => (hv₂ h).trans (hv₁ h)⟩

theorem of_step (w : WcrSem) (δ : Log) (st : St) : St.StepTo w kv st (st.step w δ) := ⟨δ, rfl, rfl, fun _ => rfl⟩

theorem of_eq {st a : St} (hm : a.mem = st.mem) (hl : a.log = st.log) (hv : kv = true → a.vals = st.vals) :
    St.StepTo w kv st a :=
  ⟨Log.empty, hm, by rw [hl, Log.seq_empty], hv⟩

end St.StepTo

/-- Every normal run of `f` is a logged step. -/
def StepsA (w : WcrSem) (kv : Bool) {α : Type} (f : St → Res (α × St)) : Prop :=
  ∀ st a st', f st = .ok (a, st') → St.StepTo w kv st st'

/-- `StepsA` without a result. -/
def Steps (w : WcrSem) (kv : Bool) (f : St → Res St) : Prop :=
  ∀ st st', f st = .ok st' → St.StepTo w kv st st'

namespace StepsA

variable {w : WcrSem} {kv : Bool} {α β γ : Type}

theorem pure (a : α) : StepsA w kv (fun st => .ok (a, st)) := by
  intro st a' st' h
  simp only [Res.ok_inj, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact St.StepTo.refl w st

theorem err (e : ErrReason) : StepsA w kv (fun _ => (.err e : Res (α × St))) := by
  intro st a st' h; cases h

theorem ofFootA {f : St → Res (α × St)} (h : FootA w f) : StepsA w kv f := by
  intro st a st' hf
  obtain ⟨δ, -, rfl, -⟩ := h st a st' hf
  exact St.StepTo.of_step w δ st

theorem bind {f : St → Res (α × St)} {g : α × St → Res (β × St)} (hf : StepsA w kv f)
    (hg : ∀ a, StepsA w kv (fun st => g (a, st))) : StepsA w kv (fun st => f st >>= g) := by
  intro st b st'' h
  dsimp only at h
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok p =>
    obtain ⟨a, st'⟩ := p
    rw [hfs] at h
    exact (hf st a st' hfs).trans (hg a st' b st'' h)

/-- A head that does not change the state. -/
theorem bind_det {h : St → Res γ} {g : γ → St → Res (β × St)} (hg : ∀ c, StepsA w kv (g c)) :
    StepsA w kv (fun st => h st >>= fun c => g c st) := by
  intro st b st' hr
  dsimp only at hr
  cases hhs : h st with
  | err e => rw [hhs] at hr; cases hr
  | ok c => rw [hhs] at hr; exact hg c st b st' hr

theorem foldlM {f : α × St → β → Res (α × St)} :
    ∀ (xs : List β), (∀ b ∈ xs, ∀ a, StepsA w kv (fun st => f (a, st) b)) →
      ∀ (a₀ : α), StepsA w kv (fun st => xs.foldlM f (a₀, st))
  | [], _, a₀ => by simpa using (pure a₀ : StepsA w kv (fun st => .ok (a₀, st)))
  | x :: xs, hf, a₀ => by
    simp only [List.foldlM_cons]
    exact bind (hf x List.mem_cons_self a₀) (fun a => foldlM xs (fun b hb => hf b (List.mem_cons_of_mem _ hb)) a)

/-- A pure computation of a result, the state unchanged. -/
theorem pure_any (f : St → α × St) (hf : ∀ st, (f st).2 = st) :
    StepsA w kv (fun st => (Pure.pure (f st) : Res (α × St))) := by
  intro st a st' h
  simp only [Res.pure_eq, Res.ok_inj] at h
  have h2 := hf st
  rw [h] at h2
  dsimp only at h2
  subst h2
  exact St.StepTo.refl w _

/-- An action that only reads the code-edge values. -/
theorem ofVals (f : List ((NodeId × Conn) × Val) → α) : StepsA w kv (fun st => .ok (f st.vals, st)) := by
  intro st a st' h
  simp only [Res.ok_inj, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact St.StepTo.refl w st

end StepsA

namespace Steps

variable {w : WcrSem} {kv : Bool} {α β γ : Type}

theorem pure : Steps w kv (fun st => .ok st) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  exact St.StepTo.refl w st

theorem err (e : ErrReason) : Steps w kv (fun _ => .err e) := by
  intro st st' h; cases h

theorem ofFoot {f : St → Res St} (h : Foot w f) : Steps w kv f := by
  intro st st' hf
  obtain ⟨δ, -, rfl, -⟩ := h st st' hf
  exact St.StepTo.of_step w δ st

theorem bindA {f : St → Res (α × St)} {g : α × St → Res St} (hf : StepsA w kv f)
    (hg : ∀ a, Steps w kv (fun st => g (a, st))) : Steps w kv (fun st => f st >>= g) := by
  intro st st'' h
  dsimp only at h
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok p =>
    obtain ⟨a, st'⟩ := p
    rw [hfs] at h
    exact (hf st a st' hfs).trans (hg a st' st'' h)

theorem bind {f g : St → Res St} (hf : Steps w kv f) (hg : Steps w kv g) : Steps w kv (fun st => f st >>= g) := by
  intro st st'' h
  dsimp only at h
  cases hfs : f st with
  | err e => rw [hfs] at h; cases h
  | ok st' =>
    rw [hfs] at h
    exact (hf st st' hfs).trans (hg st' st'' h)

/-- A head that does not change the state. -/
theorem bind_det {h : St → Res γ} {g : γ → St → Res St} (hg : ∀ c, Steps w kv (g c)) :
    Steps w kv (fun st => h st >>= fun c => g c st) := by
  intro st st' hr
  dsimp only at hr
  cases hhs : h st with
  | err e => rw [hhs] at hr; cases hr
  | ok c => rw [hhs] at hr; exact hg c st st' hr

theorem foldlM {f : St → β → Res St} :
    ∀ (xs : List β), (∀ b ∈ xs, Steps w kv (fun st => f st b)) → Steps w kv (fun st => xs.foldlM f st)
  | [], _ => by simpa using (pure : Steps w kv (fun st => .ok st))
  | x :: xs, hf => by
    simp only [List.foldlM_cons]
    exact bind (hf x List.mem_cons_self) (foldlM xs fun b hb => hf b (List.mem_cons_of_mem _ hb))

/-- A pure computation leaving the state unchanged. -/
theorem pure_any (f : St → St) (hf : ∀ st, f st = st) : Steps w kv (fun st => (Pure.pure (f st) : Res St)) := by
  intro st st' h
  simp only [Res.pure_eq, Res.ok_inj] at h
  have h2 := hf st
  rw [h] at h2
  subst h2
  exact St.StepTo.refl w _

/-- Code-edge values are not logged. -/
theorem setVal (key : NodeId × Conn) (v : Val) : Steps w false (fun st => .ok (st.setVal key v)) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  exact St.StepTo.of_eq rfl rfl (fun h => by cases h)

end Steps

/-! ## Primitive accesses -/

namespace St

variable (ctx : Ctx) (ρ : Valuation) (w : WcrSem) {kv : Bool}

theorem pushMany_eq (l : Loc) : ∀ (vs : List Val) (m : Mem),
    m.pushMany l vs = { m with queues := Function.update m.queues (m.resolve l) (m.queues (m.resolve l) ++ vs) }
  | [], m => by
    cases m
    simp only [Mem.pushMany, List.foldl_nil, List.append_nil, Function.update_eq_self]
  | v :: vs, m => by
    have h := pushMany_eq l vs (m.push l v)
    simp only [Mem.pushMany, List.foldl_cons] at h ⊢
    rw [h]
    have hr : (m.push l v).resolve l = m.resolve l := Mem.resolve_congr rfl l
    rw [hr]
    simp only [Mem.push, Function.update_self, Function.update_idem, List.append_assoc, List.singleton_append]

theorem apply_queues_one (m : Mem) (l : Loc) (e : QueueEffect) :
    Log.apply ({ queues := [(l, e)] } : Log) w m =
      match e with
      | .pushOnly vs => { m with queues := Function.update m.queues l (m.queues l ++ vs) }
      | .exclusive final => { m with queues := Function.update m.queues l final } := by
  cases e <;> rfl

theorem push_steps (n : Name) (idx : Idx) (vs : List Val) : Steps w kv (fun st => st.push ctx ρ n idx vs) := by
  intro st st' h
  dsimp only at h
  unfold St.push at h
  cases hl : ctx.loc ρ st.mem n idx with
  | err e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Res.ok_bind] at h
    split at h
    · cases h; exact St.StepTo.refl w st
    · cases h
      refine ⟨{ queues := [(st.mem.resolve l, .pushOnly vs)] }, ?_, rfl, fun _ => rfl⟩
      rw [apply_queues_one, pushMany_eq]

theorem popN_steps (n : Name) (idx : Idx) (k : Nat) : StepsA w kv (fun st => st.popN ctx ρ n idx k) := by
  intro st a st' h
  dsimp only at h
  unfold St.popN at h
  cases hl : ctx.loc ρ st.mem n idx with
  | err e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Res.ok_bind] at h
    split at h
    · simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact St.StepTo.refl w st
    · split at h
      · cases h
      · simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
        obtain ⟨-, rfl⟩ := h
        exact ⟨{ queues := [(st.mem.resolve l, .exclusive ((st.mem.queues (st.mem.resolve l)).drop k))] },
          by rw [apply_queues_one], rfl, fun _ => rfl⟩

theorem queue_steps (n : Name) (idx : Idx) : StepsA w kv (fun st => do
    let q ← st.queue ctx ρ n idx
    pure (q, st)) := by
  intro st a st' h
  dsimp only at h
  cases hq : st.queue ctx ρ n idx with
  | err e => rw [hq] at h; cases h
  | ok q =>
    rw [hq] at h
    simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact St.StepTo.refl w st

theorem bindAlias_steps (n : Name) (b : AliasBinding) : Steps w kv (fun st => .ok (st.bindAlias n b)) := by
  intro st st' h
  simp only [Res.ok_inj] at h
  subst h
  exact ⟨{ aliasSets := [(n, b)] }, rfl, rfl, fun _ => rfl⟩

theorem read_steps (n : Name) (idx : Idx) (oob : Bool) : StepsA w kv (fun st => st.read ctx ρ n idx oob) :=
  StepsA.ofFootA (St.read_foot ctx ρ w n idx oob)

theorem readMany_steps (n : Name) (pts : List Idx) (oob : Bool) : StepsA w kv (fun st => st.readMany ctx ρ n pts oob) :=
  StepsA.ofFootA (St.readMany_foot ctx ρ w n pts oob)

theorem store_steps (n : Name) (idx : Idx) (oob : Bool) (wcrOp : Option WcrOp) (v : Val) :
    Steps ctx.I.wcr kv (fun st => st.store ctx ρ n idx oob wcrOp v) :=
  Steps.ofFoot (St.store_foot ctx ρ n idx oob wcrOp v)

theorem wcr_steps (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    Steps ctx.I.wcr kv (fun st => st.wcr ctx ρ n idx oob op v) :=
  Steps.ofFoot (St.wcr_foot ctx ρ n idx oob op v)

end St

/-! ## Leaf functions -/

section Leaf

variable (ctx : Ctx) (ρ : Valuation) (s : State) {kv : Bool}

theorem readScalarData_steps (d : Name) (sub : Option Subset) (oob : Bool) :
    StepsA ctx.I.wcr kv (fun st => readScalarData ctx ρ d sub oob st) := by
  unfold readScalarData
  refine StepsA.bind_det (fun pts => ?_)
  rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
  · exact StepsA.err _
  · dsimp only
    split
    · exact StepsA.bind (St.popN_steps ctx ρ _ d p 1) (fun vs => StepsA.pure_any _ (by intro; rfl))
    · exact StepsA.bind (St.read_steps ctx ρ _ d p oob) (fun v => StepsA.pure_any _ (by intro; rfl))
  · exact StepsA.err _

theorem readEdgeScalar_steps (e : DEdge) : StepsA ctx.I.wcr kv (fun st => readEdgeScalar ctx ρ s e st) := by
  unfold readEdgeScalar
  cases s.srcEndpoint e with
  | code key => exact StepsA.ofVals (fun vs => (vs.find? (·.1 = key)).map (·.2))
  | data d sub => exact readScalarData_steps ctx ρ d sub _
  | invalid => exact StepsA.err _

theorem copyData_steps (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (wcr : Option WcrOp) (oob : Bool) :
    Steps ctx.I.wcr kv (fun st => copyData ctx ρ st sd ssub dd dsub wcr oob) := by
  unfold copyData
  cases ctx.isStream sd <;> cases ctx.isStream dd <;> try dsimp only
  · refine Steps.bind_det (fun spts => ?_)
    refine Steps.bind_det (fun dpts => ?_)
    refine Steps.foldlM _ (fun ⟨p, q⟩ _ => ?_)
    exact Steps.bindA (St.read_steps ctx ρ _ sd p oob) (fun v => St.store_steps ctx ρ dd q oob wcr v)
  · refine Steps.bind_det (fun spts => ?_)
    refine Steps.bind_det (fun dp => ?_)
    exact Steps.bindA (St.readMany_steps ctx ρ _ sd spts oob) (fun vs => St.push_steps ctx ρ _ dd dp vs)
  · refine Steps.bind_det (fun dpts => ?_)
    refine Steps.bind_det (fun sp => ?_)
    refine Steps.bindA (St.popN_steps ctx ρ _ sd sp dpts.length) (fun vs => ?_)
    dsimp only
    exact Steps.foldlM _ (fun ⟨q, v⟩ _ => St.store_steps ctx ρ dd q oob wcr v)
  · exact Steps.err _

theorem execCopy_steps (e : DEdge) : Steps ctx.I.wcr kv (fun st => execCopy ctx ρ s e st) := by
  unfold execCopy
  cases e.data with
  | dep => exact Steps.pure
  | memlet m =>
    dsimp only
    cases s.srcEndpoint e <;> cases s.dstEndpoint e <;> first
      | exact Steps.err _
      | exact copyData_steps ctx ρ _ _ _ _ _ _

theorem bindInData_steps (d : Name) (sub : Option Subset) (oob : Bool) :
    StepsA ctx.I.wcr kv (fun st => bindInData ctx ρ d sub oob st) := by
  unfold bindInData
  cases ctx.isStream d <;> try dsimp only
  · refine StepsA.bind_det (fun pts => ?_)
    rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
    all_goals first
      | exact StepsA.bind (St.read_steps ctx ρ _ d p oob) (fun v => StepsA.pure_any _ (by intro; rfl))
      | (refine StepsA.bind_det (fun lo => ?_)
         exact StepsA.bind (St.readMany_steps ctx ρ _ d _ oob) (fun r => StepsA.pure_any _ (by intro; rfl)))
  · refine StepsA.bind_det (fun p => ?_)
    intro st a st' h
    dsimp only at h
    cases hq : st.queue ctx ρ d p with
    | err e => rw [hq] at h; cases h
    | ok q =>
      rw [hq] at h
      simp only [Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact St.StepTo.refl _ st

theorem bindIn_steps (e : DEdge) : StepsA ctx.I.wcr kv (fun st => bindIn ctx ρ s e st) := by
  unfold bindIn
  cases s.srcEndpoint e with
  | code key => exact StepsA.ofVals (fun vs => ((vs.find? (·.1 = key)).map (·.2)).map ConnIn.scalar)
  | data d sub => exact StepsA.bind (bindInData_steps ctx ρ d sub _) (fun c => StepsA.pure_any _ (by intro; rfl))
  | invalid => exact StepsA.err _

theorem bindInputs_steps (n : DNode) : StepsA ctx.I.wcr kv (fun st => bindInputs ctx ρ s n st) := by
  unfold bindInputs
  refine StepsA.bind (StepsA.foldlM _ (fun e _ acc => ?_) []) (fun bs => StepsA.pure_any _ (by intro; rfl))
  cases e.data with
  | dep => exact StepsA.pure_any _ (by intro; rfl)
  | memlet m =>
    refine StepsA.bind (bindIn_steps ctx ρ s e) (fun ci => ?_)
    cases ci with
    | none => exact StepsA.pure_any _ (by intro; rfl)
    | some c => exact StepsA.pure_any _ (by intro; rfl)

theorem storeScalar_steps (d : Name) (sub : Option Subset) (wcr : Option WcrOp) (oob : Bool) (v : Val) :
    Steps ctx.I.wcr kv (fun st => storeScalar ctx ρ st d sub wcr oob v) := by
  unfold storeScalar
  cases ctx.isStream d <;> try dsimp only
  · refine Steps.bind_det (fun pts => ?_)
    rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
    · exact Steps.err _
    · exact St.store_steps ctx ρ d p oob wcr v
    · exact Steps.err _
  · exact Steps.bind_det (fun p => St.push_steps ctx ρ _ d p [v])

/-- The memlet out-edges of a code node end at containers (no code-edge
values are produced). -/
def State.OutsData (s : State) (n : DNode) : Prop :=
  ∀ e ∈ s.outEdges n.id, (∃ m, e.data = .memlet m) → (s.dstEndpoint e).isData = true

theorem applyScalar_steps (n : DNode) (hk : kv = true → s.OutsData n) (c : Conn) (v : Val) :
    Steps ctx.I.wcr kv (fun st => applyScalar ctx ρ s n st c v) := by
  unfold applyScalar
  refine Steps.foldlM _ (fun e he => ?_)
  cases hd : e.data with
  | dep => exact Steps.pure_any _ (by intro; rfl)
  | memlet m =>
    try dsimp only
    cases hde : s.dstEndpoint e with
    | code key =>
      cases kv with
      | false => exact Steps.setVal key v
      | true =>
        have := hk rfl e (List.mem_filter.1 he).1 ⟨m, hd⟩
        rw [hde] at this; cases this
    | data d sub => exact storeScalar_steps ctx ρ d sub _ _ v
    | invalid => exact Steps.err _

theorem storeArrayWrite_steps (d : Name) (sub : Option Subset) (wcr : Option WcrOp) (oob : Bool) (rel : Idx)
    (v : Val) : Steps ctx.I.wcr kv (fun st => storeArrayWrite ctx ρ st d sub wcr oob rel v) := by
  unfold storeArrayWrite
  cases ctx.isStream d <;> try dsimp only
  · refine Steps.bind_det (fun lo => ?_)
    split
    · refine Steps.bind_det (fun pts => ?_)
      split
      · exact St.store_steps ctx ρ d _ oob wcr v
      · exact Steps.err _
    · exact Steps.err _
  · exact Steps.err _

theorem applyArrayWrite_steps (n : DNode) (c : Conn) (rel : Idx) (v : Val) :
    Steps ctx.I.wcr kv (fun st => applyArrayWrite ctx ρ s n st c rel v) := by
  unfold applyArrayWrite
  refine Steps.foldlM _ (fun e _ => ?_)
  cases e.data with
  | dep => exact Steps.pure_any _ (by intro; rfl)
  | memlet m =>
    try dsimp only
    cases s.dstEndpoint e with
    | data d sub => exact storeArrayWrite_steps ctx ρ d sub _ _ rel v
    | code key => exact Steps.err _
    | invalid => exact Steps.err _

theorem pushTo_steps (d : Name) (sub : Option Subset) (vs : List Val) :
    Steps ctx.I.wcr kv (fun st => pushTo ctx ρ st d sub vs) := by
  unfold pushTo
  cases ctx.isStream d <;> try dsimp only
  · exact Steps.err _
  · exact Steps.bind_det (fun p => St.push_steps ctx ρ _ d p vs)

theorem applyPush_steps (n : DNode) (c : Conn) (vs : List Val) :
    Steps ctx.I.wcr kv (fun st => applyPush ctx ρ s n st c vs) := by
  unfold applyPush
  refine Steps.foldlM _ (fun e _ => ?_)
  cases s.dstEndpoint e with
  | data d sub => exact pushTo_steps ctx ρ d sub vs
  | code key => exact Steps.err _
  | invalid => exact Steps.err _

theorem popFrom_steps (d : Name) (sub : Option Subset) (k : Nat) :
    Steps ctx.I.wcr kv (fun st => popFrom ctx ρ st d sub k) := by
  unfold popFrom
  cases ctx.isStream d <;> try dsimp only
  · exact Steps.err _
  · exact Steps.bind_det (fun p => Steps.bindA (St.popN_steps ctx ρ _ d p k) (fun r => Steps.pure_any _ (by intro; rfl)))

theorem applyPop_steps (n : DNode) (c : Conn) (k : Nat) :
    Steps ctx.I.wcr kv (fun st => applyPop ctx ρ s n st c k) := by
  unfold applyPop
  cases s.inEdgeOn n c with
  | none => exact Steps.err _
  | some e =>
    try dsimp only
    cases s.srcEndpoint e with
    | data d sub => exact popFrom_steps ctx ρ d sub k
    | code key => exact Steps.err _
    | invalid => exact Steps.err _

theorem applyOut_steps (n : DNode) (hk : kv = true → s.OutsData n) (out : TaskletOut) :
    Steps ctx.I.wcr kv (fun st => applyOut ctx ρ s n st out) := by
  unfold applyOut
  refine Steps.bind (Steps.foldlM _ fun ⟨c, k⟩ _ => applyPop_steps ctx ρ s n c k) ?_
  refine Steps.bind (Steps.foldlM _ fun ⟨c, v⟩ _ => applyScalar_steps ctx ρ s n hk c v) ?_
  refine Steps.bind (Steps.foldlM _ fun ⟨c, rel, v⟩ _ => applyArrayWrite_steps ctx ρ s n c rel v) ?_
  exact Steps.foldlM _ fun ⟨c, vs⟩ _ => applyPush_steps ctx ρ s n c vs

theorem execCode_steps (n : DNode) (hk : kv = true → s.OutsData n) (run : (Conn → Option ConnIn) → Option TaskletOut) (ue : ErrReason) :
    Steps ctx.I.wcr kv (fun st => execCode ctx ρ s n st run ue) := by
  unfold execCode
  split
  · exact Steps.err _
  · refine Steps.bindA (bindInputs_steps ctx ρ s n) (fun ins => ?_)
    dsimp only
    cases run ins with
    | none => exact Steps.err _
    | some out => exact applyOut_steps ctx ρ s n hk out

theorem reduceData_steps (props : ReduceProps) (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (outWcr : Option WcrOp) (soob doob : Bool) :
    Steps ctx.I.wcr kv (fun st => reduceData ctx ρ st props sd ssub dd dsub outWcr soob doob) := by
  unfold reduceData
  split
  · exact Steps.err _
  · refine Steps.bind_det (fun spts => ?_)
    refine Steps.bind_det (fun dpts => ?_)
    dsimp only
    split
    · exact Steps.err _
    · refine Steps.bindA (St.readMany_steps ctx ρ _ sd spts soob) (fun vals => ?_)
      dsimp only
      refine Steps.foldlM _ (fun x _ => ?_)
      split
      · exact Steps.bind_det (fun i => St.store_steps ctx ρ dd x.2 doob outWcr _)
      · split
        · exact Steps.pure
        · exact St.wcr_steps ctx ρ dd x.2 doob _ _

theorem execReduce_steps (n : DNode) (props : ReduceProps) :
    Steps ctx.I.wcr kv (fun st => execReduce ctx ρ s n props st) := by
  unfold execReduce
  dsimp only
  split
  · split
    · exact reduceData_steps ctx ρ props _ _ _ _ _ _ _
    · exact Steps.err _
  · exact Steps.err _

theorem aliasTo_steps (d base : Name) (sub : Option Subset) :
    Steps ctx.I.wcr kv (fun st => aliasTo ctx ρ st d base sub) := by
  unfold aliasTo
  split
  · exact Steps.err _
  · refine Steps.bind_det (fun lo => ?_)
    exact Steps.bind_det (fun l => St.bindAlias_steps _ _ _)

theorem bindView_steps (v : DNode) (d : Name) : Steps ctx.I.wcr kv (fun st => bindView ctx ρ s v d st) := by
  unfold bindView
  cases s.viewEdge? v with
  | none => exact Steps.err _
  | some p =>
    dsimp only
    split
    · exact Steps.err _
    · exact aliasTo_steps ctx ρ _ _ _

theorem bindViews_steps (ids : List NodeId) : Steps ctx.I.wcr kv (fun st => bindViews ctx ρ s ids st) := by
  unfold bindViews
  refine Steps.foldlM _ (fun id _ => ?_)
  cases s.node? id with
  | none => exact Steps.pure
  | some n =>
    dsimp only
    cases n.kind with
    | access d =>
      dsimp only
      split
      · exact bindView_steps ctx ρ s n d
      · exact Steps.pure
    | _ => exact Steps.pure

theorem refSet_steps (e : DEdge) (d : Name) : Steps ctx.I.wcr kv (fun st => refSet ctx ρ s e d st) := by
  unfold refSet
  cases s.srcEndpoint e with
  | data base sub => exact aliasTo_steps ctx ρ d base sub
  | code _ => exact Steps.err _
  | invalid => exact Steps.err _

theorem execCopiesInto_steps (n : DNode) : Steps ctx.I.wcr kv (fun st => execCopiesInto ctx ρ s n st) := by
  unfold execCopiesInto
  refine Steps.foldlM _ (fun e _ => ?_)
  cases e.data with
  | dep => exact Steps.pure
  | memlet m =>
    dsimp only
    split
    · exact Steps.pure
    · cases refSetTarget? ctx s e with
      | some d => exact refSet_steps ctx ρ s e d
      | none =>
        dsimp only
        split
        · exact execCopy_steps ctx ρ s e
        · exact Steps.pure

theorem execSimple_steps (n : DNode) (hk : kv = true → n.isCode = true → s.OutsData n) :
    Steps ctx.I.wcr kv (fun st => execSimple ctx ρ s n st) := by
  unfold execSimple
  cases hn : n.kind with
  | access d => exact execCopiesInto_steps ctx ρ s n
  | mapExit => exact execCopiesInto_steps ctx ρ s n
  | consumeExit => exact execCopiesInto_steps ctx ρ s n
  | tasklet t => exact execCode_steps ctx ρ s n (fun h => hk h (by simp [DNode.isCode, hn])) _ _
  | library l =>
    dsimp only
    cases l.reduce with
    | some props => exact execReduce_steps ctx ρ s n props
    | none => exact execCode_steps ctx ρ s n (fun h => hk h (by simp [DNode.isCode, hn])) _ _
  | mapEntry _ => exact Steps.err _
  | consumeEntry _ => exact Steps.err _
  | nested _ _ _ => exact Steps.err _

theorem bindDynamicInputs_steps (n : DNode) (ρ₀ : Valuation) :
    StepsA ctx.I.wcr kv (fun st => bindDynamicInputs ctx ρ₀ s n st) := by
  unfold bindDynamicInputs
  refine StepsA.foldlM _ (fun e _ ρ₁ => ?_) ρ₀
  dsimp only
  split
  · split
    · refine StepsA.bind (readEdgeScalar_steps ctx ρ₁ s e) (fun v? => ?_)
      cases v? with
      | none => exact StepsA.pure_any _ (by intro; rfl)
      | some v => exact StepsA.pure_any _ (by intro; rfl)
    · exact StepsA.pure_any _ (by intro; rfl)
  · exact StepsA.pure_any _ (by intro; rfl)

theorem mapSetup_stepTo {n : DNode} {mp : MapNode} {st st₁ : St} {ρ' : Valuation} {pts : List Idx}
    (hs : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts)) : St.StepTo ctx.I.wcr kv st st₁ := by
  unfold mapSetup at hs
  cases hb : bindDynamicInputs ctx ρ s n st with
  | err e => rw [hb] at hs; cases hs
  | ok p =>
    obtain ⟨ρ₁, st₁'⟩ := p
    rw [hb] at hs
    simp only [Res.ok_bind] at hs
    cases hrp : ctx.rangePoints ρ₁ st₁'.mem mp.range with
    | err e => rw [hrp] at hs; cases hs
    | ok q =>
      simp only [hrp, Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs
      obtain ⟨-, rfl, -⟩ := hs
      exact bindDynamicInputs_steps ctx s n ρ st ρ₁ st₁' hb

theorem consumeSetup_stepTo {n : DNode} {c : ConsumeNode} {st st₁ : St} {ρ' : Valuation} {l : Loc} {np : Int}
    (hs : consumeSetup ctx ρ s n c st = .ok (ρ', st₁, l, np)) : St.StepTo ctx.I.wcr kv st st₁ := by
  unfold consumeSetup at hs
  cases hb : bindDynamicInputs ctx ρ s n st with
  | err e => rw [hb] at hs; cases hs
  | ok p =>
    obtain ⟨ρ₁, st₁'⟩ := p
    rw [hb] at hs
    simp only [Res.ok_bind] at hs
    cases hl : consumeStream ctx ρ₁ s n st₁'.mem with
    | err e => rw [hl] at hs; cases hs
    | ok l' =>
      simp only [hl, Res.ok_bind] at hs
      cases hn : ctx.evalIntR ρ₁ st₁'.mem c.numPes with
      | err e => rw [hn] at hs; cases hs
      | ok k =>
        simp only [hn, Res.ok_bind, Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs
        obtain ⟨-, rfl, -, -⟩ := hs
        exact bindDynamicInputs_steps ctx s n ρ st ρ₁ st₁' hb

end Leaf

theorem scopeFinish_stepTo {kv : Bool} {ctx : Ctx} {pre a : St} {logs : List Log} (h : scopeFinish ctx pre logs = .ok a) :
    St.StepTo ctx.I.wcr kv pre a := by
  obtain ⟨-, rfl⟩ := scopeFinish_ok h
  exact St.StepTo.of_step _ _ _

theorem consumeFinish_stepTo {kv : Bool} {ctx : Ctx} {pre a : St} {l : Loc} {logs : List Log} {rest pushed : List Val}
    (h : consumeFinish ctx pre l logs rest pushed = .ok a) : St.StepTo ctx.I.wcr kv pre a := by
  unfold consumeFinish at h
  cases hm : merge ctx.I.wcr pre.mem logs with
  | error e => rw [hm] at h; cases h
  | ok m' =>
    rw [hm] at h
    simp only [Res.ok_inj] at h
    subst h
    have hf := merge_ok_free' hm
    rw [merge_eq_apply _ _ hf] at hm
    injection hm with hm
    refine ⟨(Log.unionAll logs).seq { queues := [(l, .exclusive (rest ++ pushed))] }, ?_, ?_, fun _ => rfl⟩
    · rw [Log.apply_seq, ← hm, St.apply_queues_one]
      rfl
    · simp only [Log.seq_assoc]

/-- Nodes whose normal runs are logged steps: flat nodes (`DNode.FlatIn`) and
consume scopes. -/
def DNode.CohIn (s : State) (n : DNode) : Prop := n.FlatIn s ∨ ∃ c, n.kind = .consumeEntry c

theorem St.StepTo.afterNode {w : WcrSem} {kv : Bool} {st a : St} (h : St.StepTo w kv st.forNode a) :
    St.StepTo w kv st (st.afterNode a) := by
  obtain ⟨δ, hm, hl, hv⟩ := h
  refine ⟨δ, hm, ?_, hv⟩
  simp only [St.afterNode, hl, St.forNode, Log.empty_seq]

section Nodes

variable {ctx : Ctx} {s : State}

/-- **Coherence of a node**: a normal run of a flat node or consume scope
changes the state by a logged delta. -/
theorem NodeExec.stepTo {ρ : Valuation} {n : DNode} {st a : St} (hk : n.CohIn s)
    (hx : NodeExec ctx ρ s n st (.ok a)) : St.StepTo ctx.I.wcr false st a := by
  suffices H : ∀ r, NodeExec ctx ρ s n st r → r = .ok a → St.StepTo ctx.I.wcr false st a from H _ hx rfl
  intro r hx hr
  cases hx with
  | simple _ => exact execSimple_steps ctx ρ s n (fun h => by cases h) st a hr
  | mapScope _ hs _ => exact (mapSetup_stepTo ctx ρ s hs).trans (scopeFinish_stepTo hr)
  | consumeScope _ hs _ => exact (consumeSetup_stepTo ctx ρ s hs).trans (consumeFinish_stepTo hr)
  | mapSetupErr => cases hr
  | mapIterErr => cases hr
  | consumeSetupErr => cases hr
  | consumeTasksErr => cases hr
  | nestedSetupErr => cases hr
  | nestedInnerErr => cases hr
  | @nested _ ctx' _ ρ' ρ₂ _ _ _ st₁ inner mapping _ aliased m₂ g₂ hk' hs hx =>
    rcases hk with ((hs' | ⟨mp, hm⟩) | ⟨inner', mapping', ni', hkn, hct, hcode, hout⟩) | ⟨c, hc⟩
    · simp [DNode.isSimple, hk'] at hs'
    · rw [hk'] at hm; cases hm
    · rw [hk'] at hkn
      cases hkn
      unfold nestedSetup at hs
      cases hb : nestedBindings ctx s n with
      | err _ => rw [hb] at hs; cases hs
      | ok conns =>
        cases hv : nestedValuation ctx ρ st.mem mapping with
        | err _ => rw [hb] at hs; simp only [Res.ok_bind] at hs; rw [hv] at hs; cases hs
        | ok ρ'' =>
          rw [hb] at hs
          simp only [Res.ok_bind] at hs
          rw [hv] at hs
          simp only [Res.ok_bind, hct, List.map_nil, bindCodeInputs_data hcode, List.append_nil] at hs
          simp only [Res.pure_eq, Res.ok_inj, Prod.mk.injEq] at hs
          obtain ⟨rfl, rfl, rfl, rfl⟩ := hs
          cases hx with
          | run hR _ =>
            cases hR with
            | start hR =>
              obtain ⟨δ, hm, hl⟩ := RegionAt.steps hR _ _ rfl
              rw [nestedFinish_data' hout] at hr
              simp only [Res.ok_inj] at hr
              subst hr
              refine ⟨δ, ?_, ?_, fun _ => rfl⟩
              · simp only [hm]; rfl
              · simp only [hl, Log.empty_seq, Log.dropNames_nil']
    · rw [hk'] at hc; cases hc

/-- A normal sequential run of such nodes is a logged step. -/
theorem SeqExec.stepTo {ρ : Valuation} :
    ∀ {ids : List NodeId} {st st₂ : St} {tr : List (NodeId × Log)},
      (∀ id ∈ ids, ∀ n, s.node? id = some n → n.CohIn s) →
      SeqExec ctx ρ s ids st (.ok (st₂, tr)) → St.StepTo ctx.I.wcr false st st₂
  | [], st, _, _, _, hx => by cases hx; exact St.StepTo.refl _ st
  | id :: _, st, _, _, hk, hx => by
    cases hx with
    | cons hnode hn hr =>
      exact (NodeExec.stepTo (hk id List.mem_cons_self _ hnode) hn).afterNode.trans
        (SeqExec.stepTo (fun i hi => hk i (List.mem_cons_of_mem _ hi)) hr)

/-- A normal run of a scope body or component whose direct nodes are such
nodes is a logged step (view bindings included). -/
theorem OrderedExec.stepTo {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hk : ∀ id ∈ direct, ∀ n, s.node? id = some n → n.CohIn s)
    (hx : OrderedExec ctx ρ s all direct st (.ok st')) : St.StepTo ctx.I.wcr false st st' := by
  suffices H : ∀ r, OrderedExec ctx ρ s all direct st r → r = .ok st' → St.StepTo ctx.I.wcr false st st' from
    H _ hx rfl
  intro r hx hr
  cases hx with
  | prologueErr => cases hr
  | seqErr => cases hr
  | hbRace => cases hr
  | run _ hp hs _ =>
    cases hr
    exact (bindViews_steps ctx ρ s _ st _ hp).trans
      (SeqExec.stepTo (fun i hi => hk i (by simpa using (List.mem_filter.1 hi).2)) hs)

/-- **Coherence** of a run from an empty log: the memory is the initial
memory with the log applied. -/
theorem OrderedExec.coherent' {ρ : Valuation} {all direct : List NodeId} {st st' : St}
    (hk : ∀ id ∈ direct, ∀ n, s.node? id = some n → n.CohIn s)
    (hx : OrderedExec ctx ρ s all direct { st with log := Log.empty } (.ok st')) :
    st'.mem = st'.log.apply ctx.I.wcr st.mem := by
  obtain ⟨δ, hm, hl, -⟩ := OrderedExec.stepTo hk hx
  rw [hl, Log.empty_seq]
  exact hm

end Nodes

/-! ## Views -/

/-- No container of the context is a view (the prologue of a sequential
execution binds nothing). -/
def Ctx.noViews? (ctx : Ctx) : Bool := ctx.g.arrays.all fun (_, d) => !d.isView

theorem Ctx.noViews?_sound {ctx : Ctx} (h : ctx.noViews? = true) (n : Name) : ctx.isView n = false := by
  unfold Ctx.isView
  cases hd : ctx.desc? n with
  | none => rfl
  | some d =>
    simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    have := List.all_eq_true.1 h x (List.mem_of_find?_eq_some hx)
    simpa using this

theorem bindViews_eq_of {ctx : Ctx} {ρ : Valuation} {s : State} (hv : ∀ n, ctx.isView n = false)
    (ids : List NodeId) (st : St) : bindViews ctx ρ s ids st = .ok st := by
  unfold bindViews
  simp only [hv, Bool.false_eq_true, ↓reduceIte]
  induction ids generalizing st with
  | nil => rfl
  | cons id ids ih =>
    simp only [List.foldlM_cons]
    split
    · split <;> exact ih st
    · exact ih st

end Dace
