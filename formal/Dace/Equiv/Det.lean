-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Order

/-!
# Determinism of race-free sequential dataflow

On the fragment `SeqPure` (`Dace.SDFG.Frame`), for a lawful WCR semantics
and states whose map scopes are convex (`ScopesOk`, decided by
`scopesOk?`), the relational semantics of a state is **deterministic up to
commuting reorderings of logs**: if *one* derivation of a node, a sequence
of nodes, a scope body or component, the iterations of a map, the
components of a state or the state itself is normal, then *every*
derivation from an equivalent state (`St.Eqv`: equal memory and code-edge
values, logs equal up to commuting reorderings) is normal with an
equivalent result (`NodeExec.det`, `SeqExec.det`, `OrderedExec.det`,
`IterExec.det`, `CompsExec.det`, `StateExec.det`).  In particular, the
topological orders chosen inside nested map scopes do not matter, and a
state with one normal outcome has no `race` outcome (`StateExec.no_err`).

The proof is by induction over the *other* derivation:

* a leaf node and the set-up of a map are footprinted actions, so they give
  the same delta from equivalent states (`execSimple_foot`,
  `mapSetup_frame`);
* the iterations of a map all start from `iterStart`, which forgets the log,
  so equivalent pre-map states start them from the *same* state;
* a scope body or component run along another topological order is first
  replaced by the run along that order that `OrderedExec.order_indep`
  builds from the normal one; the two runs along the same order then agree
  node by node, and the happens-before check gives the same answer on
  traces whose deltas are equivalent (`State.hbFree_eqv`);
* merges of equivalent logs are equal (`merge_le`).

Convexity of map scopes (paths between nodes of a scope stay in the scope)
is what makes every topological order of a scope respect reachability
(`State.topo_pairwise`); it holds for well-formed acyclic states and is
checked here directly (`convex?`, by reachability in the subgraph induced
by the scope), so no well-formedness proof is needed.
-/

set_option autoImplicit false

namespace Dace

/-! ## Equivalent logs and traces -/

/-- Logs equal up to commuting reorderings. -/
def Log.Eqv (a b : Log) : Prop := Log.Le a b ∧ Log.Le b a

namespace Log.Eqv

theorem refl (a : Log) : Log.Eqv a a := ⟨Log.Le.refl a, Log.Le.refl a⟩

theorem symm {a b : Log} (h : Log.Eqv a b) : Log.Eqv b a := ⟨h.2, h.1⟩

theorem trans {a b c : Log} (h₁ : Log.Eqv a b) (h₂ : Log.Eqv b c) : Log.Eqv a c :=
  ⟨h₁.1.trans h₂.1, h₂.2.trans h₁.2⟩

theorem seq {a' a b' b : Log} (ha : Log.Eqv a' a) (hb : Log.Eqv b' b) : Log.Eqv (a'.seq b') (a.seq b) :=
  ⟨ha.1.seq hb.1, ha.2.seq hb.2⟩

theorem dropNames {a' a : Log} (h : Log.Eqv a' a) (P : List Name) : Log.Eqv (a'.dropNames P) (a.dropNames P) :=
  ⟨h.1.dropNames P, h.2.dropNames P⟩

theorem unionAll {gs' gs : List Log} (h : List.Forall₂ Log.Eqv gs' gs) :
    Log.Eqv (Log.unionAll gs') (Log.unionAll gs) := by
  refine ⟨Log.Le.unionAll (h.imp fun _ _ => And.left), Log.Le.unionAll ?_⟩
  induction h with
  | nil => exact .nil
  | cons hab _ ih => exact .cons hab.2 ih

/-- Equivalent logs have the same conflicts. -/
theorem conflicts {a' a b' b : Log} (ha : Log.Eqv a' a) (hb : Log.Eqv b' b) :
    Log.conflicts a' b' = Log.conflicts a b := by
  cases h : Log.conflicts a b with
  | false => exact Log.conflicts_mono ha.1.sub hb.1.sub h
  | true =>
    cases h' : Log.conflicts a' b' with
    | false => rw [Log.conflicts_mono ha.2.sub hb.2.sub h'] at h; cases h
    | true => rfl

end Log.Eqv

/-- Trace entries of the same node with equivalent deltas. -/
def TrEqv (x y : NodeId × Log) : Prop := x.1 = y.1 ∧ Log.Eqv x.2 y.2

theorem State.hbConflict_eqv (s : State) {x' x y' y : NodeId × Log} (hx : TrEqv x' x) (hy : TrEqv y' y) :
    s.hbConflict x' y' = s.hbConflict x y := by
  simp only [State.hbConflict, hx.1, hy.1, hx.2.conflicts hy.2]

theorem List.all_forall₂ {α β : Type} {R : α → β → Prop} {p : α → Bool} {q : β → Bool}
    (hpq : ∀ a b, R a b → p a = q b) : ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → l.all p = l'.all q
  | _, _, .nil => rfl
  | _, _, .cons hab h => by
    simp only [List.all_cons, hpq _ _ hab, List.all_forall₂ hpq h]

/-- The happens-before check gives the same answer on traces of the same
nodes with equivalent deltas. -/
theorem State.hbFree_eqv (s : State) :
    ∀ {tr' tr : List (NodeId × Log)}, List.Forall₂ TrEqv tr' tr → s.hbFree tr' = s.hbFree tr
  | _, _, .nil => rfl
  | _, _, .cons (a := x') (b := x) hx h => by
    simp only [State.hbFree]
    rw [State.hbFree_eqv s h, List.all_forall₂ (p := fun y => !s.hbConflict x' y)
      (q := fun y => !s.hbConflict x y) (fun _ _ hy => by simp only [s.hbConflict_eqv hx hy]) h]

/-! ## Equivalent states -/

namespace St.Eqv

theorem of_eq {a b : St} (hm : a.mem = b.mem) (hv : a.vals = b.vals) (hl : Log.Eqv a.log b.log) : St.Eqv a b :=
  ⟨hm, hv, hl.1, hl.2⟩

theorem log {a b : St} (h : St.Eqv a b) : Log.Eqv a.log b.log := ⟨h.2.2.1, h.2.2.2⟩

theorem forNode {a b : St} (h : St.Eqv a b) : St.Eqv a.forNode b.forNode :=
  of_eq h.1 h.2.1 (Log.Eqv.refl _)

theorem afterNode {a b a' b' : St} (h : St.Eqv a b) (h' : St.Eqv a' b') :
    St.Eqv (a.afterNode a') (b.afterNode b') :=
  of_eq h'.1 h'.2.1 (h.log.seq h'.log)

theorem step {w : WcrSem} (hl : w.Lawful) {a b : St} {δ' δ : Log} (h : St.Eqv a b) (hδ : Log.Eqv δ' δ) :
    St.Eqv (a.step w δ') (b.step w δ) := by
  refine of_eq ?_ h.2.1 (h.log.seq hδ)
  simp only [St.step_mem, h.1, hδ.1.apply_eq w hl]

theorem iterStart {a b : St} (h : St.Eqv a b) : Dace.iterStart a = Dace.iterStart b := by
  simp only [Dace.iterStart, h.1, h.2.1]

end St.Eqv

/-! ## Convex scopes -/

/-- The subgraph of `s` induced by `U` (edges with both ends in `U`). -/
def State.induced (s : State) (U : List NodeId) : State :=
  { s with edges := s.edges.filter fun e => U.contains e.src && U.contains e.dst }

theorem State.induced_edge {s : State} {U : List NodeId} {a b : NodeId} :
    (s.induced U).Edge a b ↔ s.Edge a b ∧ a ∈ U ∧ b ∈ U := by
  simp only [State.Edge, State.induced, List.mem_filter, Bool.and_eq_true, List.contains_iff_mem]
  constructor
  · rintro ⟨e, ⟨he, hs, hd⟩, rfl, rfl⟩
    exact ⟨⟨e, he, rfl, rfl⟩, hs, hd⟩
  · rintro ⟨⟨e, he, rfl, rfl⟩, hs, hd⟩
    exact ⟨e, ⟨he, hs, hd⟩, rfl, rfl⟩

/-- Decision procedure for `State.Convex` (`convex?_sound`): every pair of
nodes of `U` that `s` orders is ordered inside `U`. -/
def State.convex? (s : State) (U : List NodeId) : Bool :=
  U.all fun x => U.all fun y => !s.reaches x y || (s.induced U).reaches x y

theorem State.convex?_sound {s : State} {U : List NodeId} (h : s.convex? U = true) : s.Convex U := by
  intro x hx y hy hr
  simp only [State.convex?, List.all_eq_true, Bool.or_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
  rcases h x hx y hy with hn | hi
  · rw [hn] at hr; cases hr
  · have hp := State.reaches_iff.1 hi
    clear hi hr hy
    induction hp with
    | single he => exact .single (State.induced_edge.1 he)
    | tail _ he ih => exact .tail ih (State.induced_edge.1 he)

/-- Every map scope of `s` lists its nodes once and is convex. -/
def ScopesOk (s : State) : Prop :=
  ∀ n ∈ s.nodes, ∀ mp, n.kind = .mapEntry mp → (s.scopeNodes n.id).Nodup ∧ s.Convex (s.scopeNodes n.id)

/-- Decision procedure for `ScopesOk` (`scopesOk?_sound`). -/
def scopesOk? (s : State) : Bool :=
  s.nodes.all fun n => match n.kind with
    | .mapEntry _ => decide (s.scopeNodes n.id).Nodup && s.convex? (s.scopeNodes n.id)
    | _ => true

theorem scopesOk?_sound {s : State} (h : scopesOk? s = true) : ScopesOk s := by
  intro n hn mp hk
  have := List.all_eq_true.1 h n hn
  simp only [hk, Bool.and_eq_true, decide_eq_true_eq] at this
  exact ⟨this.1, State.convex?_sound this.2⟩

/-! ## Inversion of normal node runs -/

/-- A normal run of a node of the fragment is a leaf run or a map scope. -/
theorem NodeExec.ok_inv {ctx : Ctx} {ρ : Valuation} {s : State} (h : SeqPure ctx s) {n : DNode} {st a : St}
    (hn : n ∈ s.nodes) (hx : NodeExec ctx ρ s n st (.ok a)) :
    (n.isSimple = true ∧ execSimple ctx ρ s n st = .ok a) ∨
      ∃ mp ρ' st₁ pts logs, n.kind = .mapEntry mp ∧ mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts) ∧
        IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs) ∧
        scopeFinish ctx st₁ logs = .ok a := by
  suffices H : ∀ r, NodeExec ctx ρ s n st r → r = .ok a → (n.isSimple = true ∧ execSimple ctx ρ s n st = .ok a) ∨
      ∃ mp ρ' st₁ pts logs, n.kind = .mapEntry mp ∧ mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts) ∧
        IterExec ctx ρ' s n.id mp.params (ctx.privateTransients s n.id) st₁ pts (.ok logs) ∧
        scopeFinish ctx st₁ logs = .ok a from H _ hx rfl
  intro r hx hr
  cases hx with
  | simple hs => exact .inl ⟨hs, hr⟩
  | mapSetupErr => cases hr
  | mapIterErr => cases hr
  | mapScope hk hs hi => exact .inr ⟨_, _, _, _, _, hk, hs, hi, hr⟩
  | consumeSetupErr => cases hr
  | consumeTasksErr => cases hr
  | consumeScope hk _ _ =>
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | nestedSetupErr => cases hr
  | nestedInnerErr => cases hr
  | nested hk _ _ =>
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm

/-! ## Determinism -/

section Det

variable {ctx : Ctx} {s : State} (h : SeqPure ctx s) (hl : ctx.I.wcr.Lawful) (hsc : ScopesOk s)
include h hl hsc

set_option linter.unusedSectionVars false

/-- The map part of `NodeExec.det`: the set-up of a map from an equivalent
state succeeds with the same valuation and points, from an equivalent
state. -/
theorem mapSetup_det {ρ : Valuation} {n : DNode} {mp : MapNode} (hn : n ∈ s.nodes) (hk : n.kind = .mapEntry mp)
    {st₂ st st₁ : St} {ρ' : Valuation} {pts : List Idx} (hs : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts))
    (he : St.Eqv st₂ st) :
    ∃ st₂', mapSetup ctx ρ s n mp st₂ = .ok (ρ', st₂', pts) ∧ St.Eqv st₂' st₁ := by
  obtain ⟨δ, -, rfl, hδ⟩ := mapSetup_frame ρ h (h.maps _ hn mp hk).1 hs
  exact ⟨_, hδ st₂ (he.1 ▸ Mem.Agree.refl _ _) he.2.1, he.step hl (Log.Eqv.refl δ)⟩

/-- The normal run of a map entry, seen from the set-up of another run of
it from an equivalent state: the iterations start from an equivalent state
with the same valuation and points. -/
theorem mapScope_inv {ρ ρ₂ : Valuation} {n : DNode} {mp : MapNode} (hn : n ∈ s.nodes) (hk : n.kind = .mapEntry mp)
    {st₂ st st₂' a : St} {pts₂ : List Idx} (hs : mapSetup ctx ρ s n mp st₂ = .ok (ρ₂, st₂', pts₂))
    (hx : NodeExec ctx ρ s n st (.ok a)) (he : St.Eqv st₂ st) :
    ∃ pre logs, IterExec ctx ρ₂ s n.id mp.params (ctx.privateTransients s n.id) pre pts₂ (.ok logs) ∧
      St.Eqv st₂' pre ∧ scopeFinish ctx pre logs = .ok a := by
  rcases NodeExec.ok_inv h hn hx with ⟨hsim, -⟩ | ⟨mp', ρ', st₁, pts, logs, hk', hs', hi', hf⟩
  · simp [DNode.isSimple, hk] at hsim
  · rw [hk] at hk'; cases hk'
    obtain ⟨_, hs₂, he₁⟩ := mapSetup_det h hl hsc hn hk hs' he
    rw [hs₂] at hs
    simp only [Res.ok_inj, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl, rfl⟩ := hs
    exact ⟨_, _, hi', he₁, hf⟩

omit hl hsc in
theorem prologue_eqv {ρ : Valuation} {direct : List NodeId} {st₂ st₁ st : St}
    (hp : bindViews ctx ρ s direct st₂ = .ok st₁) (he : St.Eqv st₂ st) : St.Eqv st₁ st := by
  rw [bindViews_eq ρ h] at hp
  simp only [Res.ok_inj] at hp
  rw [← hp]; exact he

omit h hl hsc in
theorem sameNode {ρ : Valuation} {id : NodeId} {n n' : DNode} {st : St} {r : Res St} (hnode : s.node? id = some n)
    (hnode' : s.node? id = some n') (hn' : NodeExec ctx ρ s n' st r) : NodeExec ctx ρ s n st r := by
  rw [hnode] at hnode'; cases hnode'; exact hn'

mutual

/-- **Determinism of a node**: a normal run from `st` makes every run from
an equivalent state normal, with an equivalent result. -/
theorem NodeExec.det {ρ : Valuation} {n : DNode} {st₂ st a : St} {r : Res St} (hn : n ∈ s.nodes) :
    NodeExec ctx ρ s n st₂ r → NodeExec ctx ρ s n st (.ok a) → St.Eqv st₂ st → ∃ a', r = .ok a' ∧ St.Eqv a' a
  | .simple hsim, hx, he => by
    rcases NodeExec.ok_inv h hn hx with ⟨-, hx⟩ | ⟨mp, -, -, -, -, hk, -⟩
    · obtain ⟨δ, -, rfl, hδ⟩ := execSimple_foot ρ h hn st a hx
      exact ⟨_, hδ st₂ (he.1 ▸ Mem.Agree.refl _ _) he.2.1, he.step hl (Log.Eqv.refl δ)⟩
    · simp [DNode.isSimple, hk] at hsim
  | .mapSetupErr hk hs, hx, he => by
    rcases NodeExec.ok_inv h hn hx with ⟨hsim, -⟩ | ⟨mp, ρ', st₁, pts, logs, hk', hs', -, -⟩
    · simp [DNode.isSimple, hk] at hsim
    · rw [hk] at hk'; cases hk'
      obtain ⟨_, hs₂, -⟩ := mapSetup_det h hl hsc hn hk hs' he
      rw [hs₂] at hs; cases hs
  | .mapIterErr hk hs hi, hx, he => by
    obtain ⟨_, _, hi', he₁, -⟩ := mapScope_inv h hl hsc hn hk hs hx he
    obtain ⟨_, hr, -⟩ := IterExec.det (hsc n hn _ hk) hi hi' he₁
    cases hr
  | .mapScope hk hs hi, hx, he => by
    obtain ⟨_, _, hi', he₁, hf⟩ := mapScope_inv h hl hsc hn hk hs hx he
    obtain ⟨logs', hr, hlogs⟩ := IterExec.det (hsc n hn _ hk) hi hi' he₁
    cases hr
    obtain ⟨hpf, rfl⟩ := scopeFinish_ok hf
    have hpf' := Log.pairwiseFree_of_sub (hlogs.imp fun _ _ h => h.1.sub) hpf
    exact ⟨_, scopeFinish_eq_step ctx _ hpf', he₁.step hl (Log.Eqv.unionAll hlogs)⟩
  | .consumeSetupErr hk _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .consumeTasksErr hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .consumeScope hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .nestedSetupErr hk _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .nestedInnerErr hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm
  | .nested hk _ _, _, _ => by
    exfalso
    rcases h.kinds _ hn with hs | ⟨mp, hm⟩
    · simp [DNode.isSimple, hk] at hs
    · rw [hk] at hm; cases hm

/-- **Determinism of a sequential execution** along a fixed order: the
same trace nodes, equivalent deltas and results. -/
theorem SeqExec.det {ρ : Valuation} {ids : List NodeId} {st₂ st a : St} {tr : List (NodeId × Log)}
    {r : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s ids st₂ r → SeqExec ctx ρ s ids st (.ok (a, tr)) → St.Eqv st₂ st →
      ∃ a' tr', r = .ok (a', tr') ∧ St.Eqv a' a ∧ List.Forall₂ TrEqv tr' tr
  | .nil, hx, he => by
    cases hx
    exact ⟨_, _, rfl, he, .nil⟩
  | .missing hm, hx, _ => by
    cases hx with
    | cons hnode _ _ => rw [hm] at hnode; cases hnode
  | .headErr hnode hn, hx, he => by
    cases hx with
    | cons hnode' hn' _ =>
      have hn'' := sameNode hnode hnode' hn'
      obtain ⟨_, hr, -⟩ := NodeExec.det (List.mem_of_find?_eq_some hnode) hn hn'' he.forNode
      cases hr
  | .tailErr hnode hn hr, hx, he => by
    cases hx with
    | cons hnode' hn' hr' =>
      have hn'' := sameNode hnode hnode' hn'
      obtain ⟨_, hr₁, he₁⟩ := NodeExec.det (List.mem_of_find?_eq_some hnode) hn hn'' he.forNode
      cases hr₁
      obtain ⟨_, _, hr₂, -⟩ := SeqExec.det hr hr' (he.afterNode he₁)
      cases hr₂
  | .cons hnode hn hr, hx, he => by
    cases hx with
    | cons hnode' hn' hr' =>
      have hn'' := sameNode hnode hnode' hn'
      obtain ⟨_, hr₁, he₁⟩ := NodeExec.det (List.mem_of_find?_eq_some hnode) hn hn'' he.forNode
      cases hr₁
      obtain ⟨_, _, hr₂, he₂, htr⟩ := SeqExec.det hr hr' (he.afterNode he₁)
      cases hr₂
      exact ⟨_, _, rfl, he₂, .cons ⟨rfl, he₁.log⟩ htr⟩

/-- **Determinism of a scope body or component**: a normal run along one
topological order makes the run along every topological order (and with
every choice inside nested scopes) normal, with an equivalent result. -/
theorem OrderedExec.det {ρ : Valuation} {all direct : List NodeId} {st₂ st a : St} {r : Res St}
    (hnd : all.Nodup) (hconv : s.Convex all) :
    OrderedExec ctx ρ s all direct st₂ r → OrderedExec ctx ρ s all direct st (.ok a) → St.Eqv st₂ st →
      ∃ a', r = .ok a' ∧ St.Eqv a' a
  | .prologueErr hp, _, _ => by rw [bindViews_eq ρ h] at hp; cases hp
  | .seqErr ho hp hs, hx, he => by
    obtain ⟨b, trb, hb, -, -⟩ := OrderedExec.order_indep h hl hnd hconv hx ho
    obtain ⟨_, _, hr, -⟩ := SeqExec.det hs hb (prologue_eqv h hp he)
    cases hr
  | .hbRace ho hp hs hf, hx, he => by
    obtain ⟨b, trb, hb, hfb, -⟩ := OrderedExec.order_indep h hl hnd hconv hx ho
    obtain ⟨_, _, hr, -, htr⟩ := SeqExec.det hs hb (prologue_eqv h hp he)
    simp only [Res.ok_inj, Prod.mk.injEq] at hr
    obtain ⟨-, rfl⟩ := hr
    rw [s.hbFree_eqv htr, hfb] at hf
    cases hf
  | .run ho hp hs _, hx, he => by
    obtain ⟨b, trb, hb, -, hba⟩ := OrderedExec.order_indep h hl hnd hconv hx ho
    obtain ⟨_, _, hr, he', -⟩ := SeqExec.det hs hb (prologue_eqv h hp he)
    simp only [Res.ok_inj, Prod.mk.injEq] at hr
    obtain ⟨rfl, -⟩ := hr
    exact ⟨_, rfl, he'.trans hba⟩

/-- **Determinism of the iterations of a map**: equivalent logs. -/
theorem IterExec.det {ρ : Valuation} {entry : NodeId} {params : List Sym} {priv : List Name} {pre₂ pre : St}
    {pts : List Idx} {logs : List Log} {r : Res (List Log)}
    (hscope : (s.scopeNodes entry).Nodup ∧ s.Convex (s.scopeNodes entry)) :
    IterExec ctx ρ s entry params priv pre₂ pts r → IterExec ctx ρ s entry params priv pre pts (.ok logs) →
      St.Eqv pre₂ pre → ∃ logs', r = .ok logs' ∧ List.Forall₂ Log.Eqv logs' logs
  | .nil, hx, _ => by
    cases hx
    exact ⟨_, rfl, .nil⟩
  | .headErr hb, hx, he => by
    cases hx with
    | cons hb' _ =>
      obtain ⟨_, hr, -⟩ := OrderedExec.det hscope.1 hscope.2 hb hb' (by rw [he.iterStart]; exact St.Eqv.refl _)
      cases hr
  | .tailErr hb hr, hx, he => by
    cases hx with
    | cons hb' hr' =>
      obtain ⟨_, _, -⟩ := OrderedExec.det hscope.1 hscope.2 hb hb' (by rw [he.iterStart]; exact St.Eqv.refl _)
      obtain ⟨_, hr₂, -⟩ := IterExec.det hscope hr hr' he
      cases hr₂
  | .cons hb hr, hx, he => by
    cases hx with
    | cons hb' hr' =>
      obtain ⟨_, hr₁, he₁⟩ := OrderedExec.det hscope.1 hscope.2 hb hb' (by rw [he.iterStart]; exact St.Eqv.refl _)
      cases hr₁
      obtain ⟨_, hr₂, hlogs⟩ := IterExec.det hscope hr hr' he
      cases hr₂
      exact ⟨_, rfl, .cons (he₁.log.dropNames _) hlogs⟩

end

/-- **Determinism of the components of a state**: equivalent logs. -/
theorem CompsExec.det {ρ : Valuation} {m : Mem} :
    ∀ {comps : List (List NodeId)} {logs : List Log} {r : Res (List Log)},
      (∀ c ∈ comps, c.Nodup ∧ s.Convex c) →
      CompsExec ctx ρ s m comps r → CompsExec ctx ρ s m comps (.ok logs) →
        ∃ logs', r = .ok logs' ∧ List.Forall₂ Log.Eqv logs' logs
  | [], _, _, _, hy, hx => by cases hy; cases hx; exact ⟨_, rfl, .nil⟩
  | c :: _, _, _, hcs, hy, hx => by
    have hc := hcs c List.mem_cons_self
    have hcs' := fun c' hc' => hcs c' (List.mem_cons_of_mem _ hc')
    cases hx with
    | cons hc' hr' =>
      cases hy with
      | headErr hc₁ =>
        obtain ⟨_, hr, -⟩ := OrderedExec.det h hl hsc hc.1 hc.2 hc₁ hc' (St.Eqv.refl _)
        cases hr
      | tailErr _ hr =>
        obtain ⟨_, hr₂, -⟩ := CompsExec.det hcs' hr hr'
        cases hr₂
      | cons hc₁ hr =>
        obtain ⟨_, hr₁, he₁⟩ := OrderedExec.det h hl hsc hc.1 hc.2 hc₁ hc' (St.Eqv.refl _)
        cases hr₁
        obtain ⟨_, hr₂, hlogs⟩ := CompsExec.det hcs' hr hr'
        cases hr₂
        exact ⟨_, rfl, .cons he₁.log hlogs⟩

/-- **Determinism of a state**: if one execution of the state is normal,
every execution is normal, with the same control signal, valuation and
memory and an equivalent log. -/
theorem StateExec.det {cst : CSt} {x : Ctl × CSt} {r : Res (Ctl × CSt)}
    (hy : StateExec ctx cst s r) (hx : StateExec ctx cst s (.ok x)) :
    ∃ x', r = .ok x' ∧ x'.1 = x.1 ∧ x'.2.ρ = x.2.ρ ∧ x'.2.mem = x.2.mem ∧ Log.Eqv x'.2.log x.2.log := by
  have hcs : ∀ c ∈ s.components, c.Nodup ∧ s.Convex c :=
    fun c hc => ⟨State.components_nodup hc, State.components_convex hc⟩
  suffices H : ∀ rx, StateExec ctx cst s rx → rx = .ok x → ∃ x', r = .ok x' ∧ x'.1 = x.1 ∧ x'.2.ρ = x.2.ρ ∧
      x'.2.mem = x.2.mem ∧ Log.Eqv x'.2.log x.2.log from H _ hx rfl
  intro rx hx hrx
  cases hx with
  | compsErr => cases hrx
  | run hc =>
    rename_i logs
    cases hy with
    | compsErr hc' =>
      obtain ⟨_, hr, -⟩ := CompsExec.det h hl hsc hcs hc' hc
      cases hr
    | run hc' =>
      obtain ⟨logs', hr, hlogs⟩ := CompsExec.det h hl hsc hcs hc' hc
      cases hr
      unfold stateFinish at hrx ⊢
      cases hm : merge ctx.I.wcr cst.mem logs with
      | error e => rw [hm] at hrx; cases hrx
      | ok m' =>
        rw [hm] at hrx
        simp only [Res.ok_inj] at hrx
        subst hrx
        have hpf := (merge_isOk_iff _ _ _).1 ⟨m', hm⟩
        obtain ⟨-, hm'⟩ := merge_le ctx.I.wcr hl cst.mem (hlogs.imp fun _ _ h => h.1) hpf
        rw [hm', hm]
        exact ⟨_, rfl, rfl, rfl, rfl, (Log.Eqv.refl _).seq (Log.Eqv.unionAll hlogs)⟩

/-- A state of the fragment with one normal execution has no erroneous
(e.g. racy) execution. -/
theorem StateExec.no_err {cst : CSt} {x : Ctl × CSt} {e : ErrReason}
    (hx : StateExec ctx cst s (.ok x)) : ¬ StateExec ctx cst s (.err e) := fun hy => by
  obtain ⟨_, hr, -⟩ := StateExec.det h hl hsc hy hx
  cases hr

end Det

end Dace
