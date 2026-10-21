-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrLeaf
import Dace.SDFG.Footprint

/-!
# SDFG and schedule tree: matching of a dataflow segment (M5a)

`SegCorr ctx tc s o cs`: the tree sibling list `cs` is the image of the
SDFG nodes `o` of state `s`, in that order: an access or exit node becomes
its copies (one `CopyNode` per copy edge into it, in edge order, possibly
none), a tasklet or library node the tree node with the same keys
(`Dace.Equiv.CorrLeaf`), a map entry a `MapScope` whose body is the image
of the scope's direct children in a topological order of the scope.

**Matching** (`SegCorr.matched`; a simulation relation in the sense of
refinement proofs, see the terminology note in `Dace.Equiv.Frag`): every run of the tree list (followed by any
`rest`) is matched by the sequential SDFG run of `o` from a state with the
same memory and log: the SDFG run is undefined behavior, or fails with the
same error, or ends normally in a state with the same memory and log while
the tree continues with `rest`.  The proof is by induction on `SegCorr`;
leaves are equal leaf functions (`execCode_eq`, …) up to the log prefix
(`*_shift`), a map's iterations are matched one by one, and an iteration
whose SDFG trace has a happens-before race is undefined behavior
(`OrderedExec.hbRace`).

The fragment is `SeqPureP` (`Dace.SDFG.Frame`): its leaf lemmas give that
SDFG nodes keep the (empty) code-edge values.
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
  | .access _ | .mapExit => true
  | _ => false

/-- The `DynScopeCopyNode`s of the dynamic inputs `keys`, in order. -/
def DynNodes (keys : List InKey) (ds : List SNode) : Prop :=
  List.Forall₂ (fun k n => ∃ m : Memlet, n = .dynScopeCopy k.conn m ∧ m.inKey k.conn = k) keys ds

/-- The tree list `cs` is the image of the SDFG nodes `o` of state `s`. -/
inductive SegCorr (ctx : Ctx) (tc : TCtx) (s : State) : List NodeId → List SNode → Prop where
  | nil : SegCorr ctx tc s [] []
  | copies {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {cps : List (Name × Memlet)}
      (hn : s.node? id = some n) (hk : DNode.isCopyTarget n = true) (hsh : s.CopyShape ctx n)
      (hks : s.copiesInto ctx n = cps.map fun p => p.2.copyKey p.1) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (cps.map (fun p => SNode.copyNode p.1 p.2) ++ cs)
  | tasklet {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {t : Tasklet} {ins outs : MemletMap}
      (hn : s.node? id = some n) (hk : n.kind = .tasklet t) (hsh : s.CodeShape n)
      (hki : s.codeIns n = ins.pairs.map fun p => Memlet.inKey p.1 p.2)
      (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2)
      (hd : s.hasDynamicWcrOut n = false) (hdT : outs.hasDynamicWcr = false) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.taskletNode t ins outs :: cs)
  | library {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {l l' : LibNode}
      {props : List (String × String)} {ins outs : MemletMap}
      (hn : s.node? id = some n) (hk : n.kind = .library l) (hl : l'.classpath = l.classpath ∧ l'.reduce = l.reduce)
      (hsh : s.CodeShape n)
      (hki : s.codeIns n = ins.pairs.map fun p => Memlet.inKey p.1 p.2)
      (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2)
      (hd : s.hasDynamicWcrOut n = false) (hdT : outs.hasDynamicWcr = false) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (.libraryCall l' props ins outs :: cs)
  | map {id : NodeId} {n : DNode} {o : List NodeId} {cs : List SNode} {mp : MapNode} {oM full : List NodeId}
      {body : List SNode} {keys : List InKey} {ds : List SNode}
      (hn : s.node? id = some n) (hk : n.kind = .mapEntry mp)
      (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (hds : DynNodes keys ds)
      (hto : s.isTopologicalOrder (s.scopeNodes n.id) full = true)
      (hdir : full.filter (s.directChildren (some n.id)).contains = oM)
      (hpriv : tc.privateTransients body = ctx.privateTransients s n.id)
      (hb : SegCorr ctx tc s oM body) (h : SegCorr ctx tc s o cs) :
      SegCorr ctx tc s (id :: o) (ds ++ .mapScope mp body :: cs)

/-- The outcomes of the sequential SDFG run of `o` from `st` that match a
tree run ending in `r` after the image of `o`, continuing with `rest`. -/
def SeqOut (ctx : Ctx) (s : State) (o : List NodeId) (ρ : Valuation) (st : St) (tc : TCtx) (full : List SNode)
    (gb : Bool) (rest : List SNode) (r : Res (TCtl × CSt)) : Prop :=
  (∃ e, e.isUB = true ∧ SeqExec ctx ρ s o st (.err e)) ∨
  (∃ e, r = .err e ∧ SeqExec ctx ρ s o st (.err e)) ∨
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
theorem dscopies_run {ctx : Ctx} {tc : TCtx} (he : tc.ctx.erase = ctx.erase) (hcons : tc.consumed = none)
    {full : List SNode} {gb : Bool} {rest : List SNode} {r : Res (TCtl × CSt)} :
    ∀ {keys : List InKey} {ds : List SNode}, DynNodes keys ds → (∀ k ∈ keys, k.conn ≠ "IN_stream") →
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
        simp only [execDynScopeCopy, hk', ↓reduceIte, TCtx.inData_none hcons]
        rw [Ctx.leaf_eq he readScalarData (fun _ => rfl)]
        rw [← hk]
        simp only [Memlet.inKey]
        cases readScalarData ctx cst.ρ m.data m.dataSubset m.allowOob cst.st with
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
          rcases dscopies_run he hcons hns (fun k' hk'' => hnot k' (List.mem_cons_of_mem _ hk'')) hr with
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
          rcases dscopies_run he hcons hns (fun k' hk'' => hnot k' (List.mem_cons_of_mem _ hk'')) hr with
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
theorem mapSetupT_dyn {ctx : Ctx} {tc : TCtx} (he : tc.ctx.erase = ctx.erase) {mp : MapNode}
    (syms : List (Sym × Val)) (cst : CSt) :
    mapSetupT tc (pendingSc syms) mp cst =
      (ctx.rangePoints (Valuation.setAll cst.ρ syms) cst.mem mp.range).map fun pts => (Valuation.setAll cst.ρ syms, pts) := by
  simp only [mapSetupT, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  rw [Ctx.leaf_eq he Ctx.rangePoints (fun _ => rfl)]
  cases ctx.rangePoints (Valuation.setAll cst.ρ syms) cst.mem mp.range <;> rfl

/-! ## Matching -/

/-- Prepend a node to the matched SDFG run. -/
theorem SeqOut.cons {ctx : Ctx} {s : State} {id : NodeId} {n : DNode} {o : List NodeId} {ρ : Valuation} {st : St}
    {tc : TCtx} {full : List SNode} {gb : Bool} {rest : List SNode} {r : Res (TCtl × CSt)}
    (hn : s.node? id = some n)
    (h : (∃ e, e.isUB = true ∧ NodeExec ctx ρ s n st.forNode (.err e)) ∨
      (∃ e, r = .err e ∧ NodeExec ctx ρ s n st.forNode (.err e)) ∨
      ∃ y, NodeExec ctx ρ s n st.forNode (.ok y) ∧ SeqOut ctx s o ρ (st.afterNode y) tc full gb rest r) :
    SeqOut ctx s (id :: o) ρ st tc full gb rest r := by
  rcases h with ⟨e, hu, hx⟩ | ⟨e, hr, hx⟩ | ⟨y, hx, ht⟩
  · exact .inl ⟨e, hu, .headErr hn hx⟩
  · exact .inr (.inl ⟨e, hr, .headErr hn hx⟩)
  · rcases ht with ⟨e, hu, ht⟩ | ⟨e, hr, ht⟩ | ⟨cst', st', tr, hl, ht, hρ, hrel⟩
    · exact .inl ⟨e, hu, .tailErr hn hx ht⟩
    · exact .inr (.inl ⟨e, hr, .tailErr hn hx ht⟩)
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

theorem CSt.st_withSt (cst : CSt) {y : St} (hy : y.vals = []) : (cst.withSt y).st = y := by
  obtain ⟨m, l, v⟩ := y
  simp only at hy
  simp [CSt.withSt, CSt.st, hy]

section Matched

variable {ctx : Ctx} {tc : TCtx} {s : State} (h : SeqPureP ctx s) (he : tc.ctx.erase = ctx.erase)
  (hcons : tc.consumed = none)

omit hcons in
include he in
theorem I_eq : tc.ctx.I = ctx.I := by
  have := congrArg Ctx.I he
  simpa [Ctx.erase] using this

include he hcons in
/-- The tree runs copy nodes one after the other: the fold of `copyData`
over their keys. -/
theorem copies_run {full : List SNode} {gb : Bool} :
    ∀ (cps : List (Name × Memlet)) {more : List SNode} {cst : CSt} {r : Res (TCtl × CSt)},
      STree.ListExec tc full gb {} (cps.map (fun p => SNode.copyNode p.1 p.2) ++ more) cst r →
      (∃ e, r = .err e ∧ (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) cst.st = .err e) ∨
      ∃ cst', STree.ListExec tc full gb {} more cst' r ∧
        (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) cst.st = .ok cst'.st ∧ cst'.ρ = cst.ρ
  | [], _, cst, _, hx => .inr ⟨cst, hx, rfl, rfl⟩
  | p :: cps, more, cst, r, hx => by
    simp only [List.map_cons, List.cons_append] at hx
    have hstep : stepLeaf tc {} (.copyNode p.1 p.2) cst =
        some (leafOut (cst.lift (execCopyT tc cst.ρ p.1 p.2 cst.st))) := rfl
    rw [execCopyT_eq he hcons] at hstep
    rcases leaf_inv hstep hx with ⟨e, hR, rfl⟩ | ⟨cst₁, hR, hx₁⟩
    · refine .inl ⟨e, rfl, ?_⟩
      simp only [CSt.lift] at hR
      simp only [List.map_cons, List.foldlM_cons, copyStep]
      cases hc : copyData ctx cst.ρ cst.st (p.2.copyKey p.1).src (p.2.copyKey p.1).ssub (p.2.copyKey p.1).dst
          (p.2.copyKey p.1).dsub (p.2.copyKey p.1).wcr (p.2.copyKey p.1).oob with
      | ok y => rw [hc] at hR; cases hR
      | err e' => rw [hc] at hR; cases hR; rfl
    · simp only [CSt.lift] at hR
      cases hc : copyData ctx cst.ρ cst.st (p.2.copyKey p.1).src (p.2.copyKey p.1).ssub (p.2.copyKey p.1).dst
          (p.2.copyKey p.1).dsub (p.2.copyKey p.1).wcr (p.2.copyKey p.1).oob with
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

include h in
/-- One SDFG leaf node that is one tree leaf. -/
theorem leaf_node {full : List SNode} {gb : Bool} {id : NodeId} {n : DNode} {o : List NodeId} {nT : SNode}
    {more rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St}
    (hn : s.node? id = some n) (hsimp : n.isSimple = true)
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
      exact .inr (.inl ⟨_, rfl, hy ▸ NodeExec.simple hsimp⟩)
    · cases hR
  | ok y =>
    rw [hy] at hstep
    have hyv : y.vals = [] := by
      obtain ⟨δ, -, hyd, -⟩ := execSimple_foot cst.ρ h hn' st.forNode y hy
      rw [hyd]; simpa [St.forNode] using hrel.2.2
    rcases leaf_inv hstep hx with ⟨e', hR, -⟩ | ⟨cst₁, hR, hx₁⟩
    · cases hR
    · simp only [Res.map, CSt.lift, Res.ok.injEq] at hR
      subst hR
      exact .inr (.inr ⟨y, hy ▸ NodeExec.simple hsimp, ih hx₁ (hrel.afterNode hyv ⟨rfl, rfl⟩) rfl⟩)

include h in
/-- **The iterations of a map**: the tree's iterations, each matched by
the SDFG's run of the scope body along the tree's order. -/
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
      (∃ e, rT = .err e ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.err e)) ∨
      ∃ logs, rT = .ok logs ∧ IterExec ctx pre.ρ s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs) := by
  have hrel : ∀ i, StRel (iterStart (ctx.privateTransients s n.id) st₁) (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) :=
    fun i => ⟨by simp only [iterStart, iterStartT, hm, hpriv], rfl, hv⟩
  have hp : ∀ ρ', bindViews ctx ρ' s (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) = .ok (iterStart (ctx.privateTransients s n.id) st₁) :=
    fun ρ' => bindViews_eq ρ' h _ _
  -- the body of one iteration
  have hbody : ∀ (i : Idx) {res : Res (TCtl × CSt)},
      STree.ListExec tc body false {} body (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) res →
      (∃ e, e.isUB = true ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) (.err e)) ∨
      (∃ e, res = .err e ∧ OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id)
        (s.directChildren (some n.id)) (iterStart (ctx.privateTransients s n.id) st₁) (.err e)) ∨
      ∃ st', OrderedExec ctx (pre.ρ.bindIdx mp.params i) s (s.scopeNodes n.id) (s.directChildren (some n.id))
        (iterStart (ctx.privateTransients s n.id) st₁) (.ok st') ∧ ∃ cst', res = .ok (.normal, cst') ∧ st'.log = cst'.log := by
    intro i res hb
    have hb' : STree.ListExec tc body false {} (body ++ []) (iterStartT (tc.privateTransients body) pre (pre.ρ.bindIdx mp.params i)) res := by
      rw [List.append_nil]; exact hb
    rcases ihb hb' (hrel i) with ⟨e, hu, hs⟩ | ⟨e, hr, hs⟩ | ⟨cst', st', tr, hl, hs, -, hst⟩
    · rw [← hdir] at hs
      exact .inl ⟨e, hu, .seqErr hto (hp _) hs⟩
    · rw [← hdir] at hs
      exact .inr (.inl ⟨e, hr, .seqErr hto (hp _) hs⟩)
    · rw [← hdir] at hs
      have hres := STree.ListExec_nil (by rfl) hl
      cases hf : s.hbFree tr with
      | false => exact .inl ⟨_, rfl, .hbRace hto (hp _) hs hf⟩
      | true => exact .inr (.inr ⟨st', .run hto (hp _) hs hf, cst', hres, hst.2.1⟩)
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
      rcases hbody i hb with ⟨e', hu, hx⟩ | ⟨e', rfl, hx⟩ | ⟨st', hx, cst', rfl, -⟩
      · exact .inl ⟨e', hu, .headErr hx⟩
      · cases hlog; exact .inr (.inl ⟨_, rfl, .headErr hx⟩)
      · cases hlog
    | tailErr hb hlog hr =>
      rcases hbody i hb with ⟨e', hu, hx⟩ | ⟨e', rfl, hx⟩ | ⟨st', hx, cst', rfl, hlg⟩
      · exact .inl ⟨e', hu, .headErr hx⟩
      · cases hlog
      · rcases ih hr with ⟨e', hu, hy⟩ | ⟨e', he', hy⟩ | ⟨logs, hl', -⟩
        · exact .inl ⟨e', hu, .tailErr hx hy⟩
        · exact .inr (.inl ⟨e', he', .tailErr hx hy⟩)
        · cases hl'
    | cons hb hlog hr =>
      rcases hbody i hb with ⟨e', hu, hx⟩ | ⟨e', rfl, hx⟩ | ⟨st', hx, cst', rfl, hlg⟩
      · exact .inl ⟨e', hu, .headErr hx⟩
      · cases hlog
      · rcases ih hr with ⟨e', hu, hy⟩ | ⟨e', he', hy⟩ | ⟨logs', hl', hy⟩
        · exact .inl ⟨e', hu, .tailErr hx hy⟩
        · cases he'
        · cases hl'
          simp only [iterLog, Res.ok.injEq] at hlog
          subst hlog
          refine .inr (.inr ⟨_, rfl, ?_⟩)
          rw [hpriv, ← hlg]
          exact IterExec.cons hx hy

omit h hcons in
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

include h he hcons in
/-- **Matching of a dataflow segment** (see the module documentation). -/
theorem SegCorr.matched {o : List NodeId} {cs : List SNode} (hc : SegCorr ctx tc s o cs) :
    ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} {st : St},
      STree.ListExec tc full gb {} (cs ++ rest) cst r → StRel st cst → SeqOut ctx s o cst.ρ st tc full gb rest r := by
  induction hc with
  | nil =>
    intro full gb rest cst r st hx hrel
    exact .inr (.inr ⟨cst, st, [], hx, .nil, rfl, hrel⟩)
  | @copies id n o cs cps hn hk hsh hks _ ih =>
    intro full gb rest cst r st hx hrel
    have hn' := List.mem_of_find?_eq_some hn
    rw [List.append_assoc] at hx
    -- the SDFG node runs the same copies
    have hfold : ∀ X, execSimple ctx cst.ρ s n X = (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) X := by
      intro X
      rw [execSimple_copyTarget hk, execCopiesInto_eq cst.ρ hsh, hks]
      rfl
    have hshift : (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) cst.st =
        ((cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) st.forNode).map (St.shift cst.log) := by
      rw [hrel.forNode_shift]
      exact List.foldlM_res_comm _ _ (fun b k => copyData_shift ctx cst.ρ cst.log b _ _ _ _ _ _) _ _
    have hsimp := DNode.isSimple_of_copyTarget hk
    apply SeqOut.cons hn
    rcases copies_run he hcons cps hx with ⟨e, rfl, hf⟩ | ⟨cst', hx', hf, hρ⟩
    · rw [hshift] at hf
      cases hy : (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) st.forNode with
      | ok y => rw [hy] at hf; cases hf
      | err e' =>
        rw [hy] at hf
        cases hf
        exact .inr (.inl ⟨_, rfl, (hfold st.forNode).trans hy ▸ NodeExec.simple hsimp⟩)
    · rw [hshift] at hf
      cases hy : (cps.map fun p => p.2.copyKey p.1).foldlM (copyStep ctx cst.ρ) st.forNode with
      | err e' => rw [hy] at hf; cases hf
      | ok y =>
        rw [hy] at hf
        simp only [Res.map, Res.ok.injEq] at hf
        have hyv : y.vals = [] := by
          obtain ⟨δ, -, hyd, -⟩ := execSimple_foot cst.ρ h hn' st.forNode y ((hfold _).trans hy)
          rw [hyd]; simpa [St.forNode] using hrel.2.2
        refine .inr (.inr ⟨y, (hfold st.forNode).trans hy ▸ NodeExec.simple hsimp, ?_⟩)
        have := ih hx' (hrel.afterNode hyv ⟨?_, ?_⟩)
        · rw [hρ] at this; exact this
        · have := congrArg St.mem hf; simpa [CSt.st] using this.symm
        · have := congrArg St.log hf; simpa [CSt.st, hrel.2.1] using this.symm
  | @tasklet id n o cs t ins outs hn hk hsh hki hko hd hdT _ ih =>
    intro full gb rest cst r st hx hrel
    refine leaf_node h hn (by simp [DNode.isSimple, hk]) ?_ hx hrel (fun hx' hrel' hρ => hρ ▸ ih hx' hrel')
    have : ∀ X, execCodeT tc cst.ρ t.label ins outs X (tc.ctx.I.tasklet t cst.ρ) (.taskletUndefined t.label) =
        execSimple ctx cst.ρ s n X := by
      intro X
      rw [← execCode_eq he hcons cst.ρ hsh hki hko hd hdT, I_eq he]
      simp [execSimple, hk]
    simp only [stepLeaf, this]
  | @library id n o cs l l' props ins outs hn hk hl hsh hki hko hd hdT _ ih =>
    intro full gb rest cst r st hx hrel
    refine leaf_node h hn (by simp [DNode.isSimple, hk]) ?_ hx hrel (fun hx' hrel' hρ => hρ ▸ ih hx' hrel')
    have : ∀ X, execLibraryT tc cst.ρ l' ins outs X = execSimple ctx cst.ρ s n X := by
      intro X
      unfold execLibraryT
      rw [hl.2, hl.1]
      cases hr : l.reduce with
      | some props' =>
        simp only []
        rw [← execReduce_eq he hcons cst.ρ hsh hki hko props' X]
        simp [execSimple, hk, hr]
      | none =>
        simp only []
        rw [← execCode_eq he hcons cst.ρ hsh hki hko hd hdT, I_eq he]
        simp [execSimple, hk, hr]
    simp only [stepLeaf, this]
  | @map id n o cs mp oM full body keys ds hn hk hkeys hok hds hto hdir hpriv _ _ ihb ih =>
    intro full' gb rest cst r st hx hrel
    have hn' := List.mem_of_find?_eq_some hn
    apply SeqOut.cons hn
    have hx₁ : STree.ListExec tc full' gb (pendingSc []) (ds ++ (.mapScope mp body :: (cs ++ rest))) cst r := by
      simpa only [List.append_assoc, List.cons_append] using hx
    have hshift : cst.st = st.forNode.shift cst.log := hrel.forNode_shift
    rcases dscopies_run he hcons hds (fun k hk' => (hok.2 k hk').1) hx₁ with
      ⟨e, rfl, hd⟩ | ⟨syms, y', hd, hyv', hx₂⟩
    · rw [hshift, dynReadAcc_shift] at hd
      cases hd₀ : dynReadAcc ctx cst.ρ [] keys st.forNode with
      | ok _ => rw [hd₀] at hd; cases hd
      | err e' =>
        rw [hd₀] at hd; cases hd
        exact .inr (.inl ⟨_, rfl, .mapSetupErr hk (by rw [mapSetup_dyn hkeys hok, hd₀]; rfl)⟩)
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
            exact .inr (.inl ⟨_, rfl, .mapSetupErr hk hset⟩)
        · rw [hsetT] at hs
          cases hp : ctx.rangePoints (Valuation.setAll cst.ρ syms₀) y.mem mp.range with
          | err _ => rw [hp] at hs; cases hs
          | ok pts' =>
            rw [hp] at hs hset
            simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at hs hset
            obtain ⟨rfl, rfl⟩ := hs
            rcases iter_matched h hto hdir hpriv (fun hb hr => ihb hb hr)
                (pre := { cst.withSt (y.shift cst.log) with ρ := Valuation.setAll cst.ρ syms₀ }) (st₁ := y)
                rfl hyv hi with ⟨e', hu, hy⟩ | ⟨e', he', hy⟩ | ⟨logs, hl, -⟩
            · exact .inl ⟨e', hu, .mapIterErr hk hset hy⟩
            · cases he'; exact .inr (.inl ⟨_, rfl, .mapIterErr hk hset hy⟩)
            · cases hl
        · rw [hsetT] at hs
          cases hp : ctx.rangePoints (Valuation.setAll cst.ρ syms₀) y.mem mp.range with
          | err _ => rw [hp] at hs; cases hs
          | ok pts' =>
            rw [hp] at hs hset
            simp only [Res.map, Res.ok.injEq, Prod.mk.injEq] at hs hset
            obtain ⟨rfl, rfl⟩ := hs
            rcases iter_matched h hto hdir hpriv (fun hb hr => ihb hb hr)
                (pre := { cst.withSt (y.shift cst.log) with ρ := Valuation.setAll cst.ρ syms₀ }) (st₁ := y)
                rfl hyv hi with ⟨e', hu, hy⟩ | ⟨e', he', -⟩ | ⟨logs', hl, hy⟩
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
                · cases hf; exact .inr (.inl ⟨_, rfl, hy' ▸ NodeExec.mapScope hk hset hy⟩)
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

end Matched

end Corr

end Dace
