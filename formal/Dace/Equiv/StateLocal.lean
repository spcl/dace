-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.ScopeBody
import Dace.SDFG.BEq

/-!
# Locality of state executions

Two states `s'` and `s` that agree on a set `X` of node ids — the same nodes,
the same adjacent edges seen the same way by the leaf functions
(`EdgeSame`), the same reachability between nodes of
`X`, and corresponding map scopes inside `X` (`LocalFacts`) — execute every
node, sequence, scope body or component made of nodes of `X` with **the same
outcomes** (`NodeExec.local`, `SeqExec.local`, `OrderedExec.local`,
`IterExec.local`, `CompsExec.local`).  Scope lists may be permutations of
each other (`ListSame`): a sequential execution runs its nodes in any
topological order, and only the view prologue reads the order of the direct
nodes: `ListSame.views` requires the same sequence of view bindings
(`viewKey`: the view and what it views).

Uses (`Dace.Equiv.SdfgCorr`):

* reordering the node list of a state (`X` = every node);
* merging two independent states into one (`X` = the nodes of either part).

Consume scopes also need the same consumed stream (`LocalFacts.consumes`).  `localSame?`
decides `LocalFacts` (`localSame?_spec`).
-/

set_option autoImplicit false

namespace Dace

/-- What binding the views of node `id` does: the view and what it views
(`bindView_eq`); `none` for nodes that bind nothing. -/
def viewKey (ctx : Ctx) (s : State) (id : NodeId) : Option (Name × Option (Option Name × Option Subset)) :=
  match s.node? id with
  | some n =>
    match n.kind with
    | .access d => if ctx.isView d then some (d, s.viewBase? n) else none
    | _ => none
  | none => none

/-- Bind one view from its key. -/
def viewBind (ctx : Ctx) (ρ : Valuation) (st : St) : Name × Option (Option Name × Option Subset) → Res St
  | (d, none) => .err (.malformed s!"ambiguous or invalid view edge for '{d}'")
  | (d, some (none, _)) => .err (.malformed s!"view edge of '{d}' does not reach an access node")
  | (d, some (some base, sub)) => aliasTo ctx ρ st d base sub

/-- The view prologue depends only on the sequence of view keys. -/
theorem bindViews_keys {ctx : Ctx} {ρ : Valuation} {s : State} :
    ∀ (ids : List NodeId) (st : St),
      bindViews ctx ρ s ids st = (ids.filterMap (viewKey ctx s)).foldlM (viewBind ctx ρ) st
  | [], _ => rfl
  | id :: ids, st => by
    unfold bindViews at *
    simp only [List.foldlM_cons, List.filterMap_cons]
    unfold viewKey
    cases hn : s.node? id with
    | none => exact bindViews_keys ids st
    | some n =>
      simp only
      cases hk : n.kind with
      | access d =>
        simp only
        by_cases hv : ctx.isView d = true
        · simp only [hv, ↓reduceIte, List.foldlM_cons]
          rw [bindView_eq]
          rcases s.viewBase? n with _ | ⟨_ | base, sub⟩
          · rfl
          · rfl
          · simp only [viewBind]
            cases aliasTo ctx ρ st d base sub with
            | ok st' => exact bindViews_keys ids st'
            | err e => rfl
        · simp only [hv, Bool.false_eq_true, ↓reduceIte]
          exact bindViews_keys ids st
      | _ => exact bindViews_keys ids st

/-- The node lists of one sequential execution in the two states: a scope's
(or component's) nodes and its direct nodes, up to permutation, inside `X`,
with the same view prologue. -/
structure ListSame (ctx : Ctx) (s' s : State) (X : List NodeId) (all' all direct' direct : List NodeId) : Prop where
  perm : all'.Perm all
  nodup : all.Nodup
  allX : ∀ x ∈ all, x ∈ X
  dperm : direct'.Perm direct
  directX : ∀ x ∈ direct, x ∈ X
  views : (s'.viewOrder ctx direct').filterMap (viewKey ctx s') = (s.viewOrder ctx direct).filterMap (viewKey ctx s)

/-- What the executions of the nodes of `X` read from the two states. -/
structure LocalFacts (ctx : Ctx) (s' s : State) (X : List NodeId) : Prop where
  node : ∀ id ∈ X, s'.node? id = s.node? id
  inE : ∀ id ∈ X, s'.inEdges id = s.inEdges id
  outE : ∀ id ∈ X, s'.outEdges id = s.outEdges id
  edgeIn : ∀ id ∈ X, ∀ e ∈ s.inEdges id, EdgeSame ctx s' s e
  edgeOut : ∀ id ∈ X, ∀ e ∈ s.outEdges id, EdgeSame ctx s' s e
  reach : ∀ a ∈ X, ∀ b ∈ X, s'.reaches a b = s.reaches a b
  consumes : ∀ id ∈ X, ∀ n c, s.node? id = some n → n.kind = .consumeEntry c →
    ListSame ctx s' s X (s'.scopeNodes id) (s.scopeNodes id) (s'.directChildren (some id))
      (s.directChildren (some id)) ∧
    ctx.privateTransients s' id = ctx.privateTransients s id ∧ s'.consumedStream? id = s.consumedStream? id
  scopes : ∀ id ∈ X, ∀ n mp, s.node? id = some n → n.kind = .mapEntry mp →
    ListSame ctx s' s X (s'.scopeNodes id) (s.scopeNodes id) (s'.directChildren (some id))
      (s.directChildren (some id)) ∧
    ctx.privateTransients s' id = ctx.privateTransients s id

theorem NodeLocal.consumeStream {ctx : Ctx} {s' s : State} {n : DNode} (h : NodeLocal ctx s' s n) (ρ : Valuation)
    (m : Mem) : Dace.consumeStream ctx ρ s' n m = Dace.consumeStream ctx ρ s n m := by
  unfold Dace.consumeStream
  rw [h.inE]
  cases hf : (s.inEdges n.id).find? (·.dstConn = some "IN_stream") with
  | none => rfl
  | some e => simp only [(h.edgeIn e (List.mem_of_find?_eq_some hf)).src]

theorem NodeLocal.consumeSetup {ctx : Ctx} {s' s : State} {n : DNode} (h : NodeLocal ctx s' s n) (ρ : Valuation)
    (c : ConsumeNode) (st : St) : Dace.consumeSetup ctx ρ s' n c st = Dace.consumeSetup ctx ρ s n c st := by
  unfold Dace.consumeSetup
  simp only [h.bindDynamicInputs, h.consumeStream]

theorem taskStart_of_stream {ctx : Ctx} {s' s : State} {entry : NodeId} (h : s'.consumedStream? entry = s.consumedStream? entry)
    (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : St) (q : List Val) :
    taskStart ctx s' entry c l priv pre q = taskStart ctx s entry c l priv pre q := by
  simp only [taskStart, chunkMemName, h]

namespace LocalFacts

variable {ctx : Ctx} {s' s : State} {X : List NodeId} (hF : LocalFacts ctx s' s X)
include hF

theorem nodeLocal {n : DNode} (hn : n.id ∈ X) : NodeLocal ctx s' s n :=
  ⟨hF.inE _ hn, hF.outE _ hn, hF.edgeIn _ hn, hF.edgeOut _ hn⟩

omit hF in
theorem prologue {ρ : Valuation} {direct' direct : List NodeId}
    (hv : (s'.viewOrder ctx direct').filterMap (viewKey ctx s') = (s.viewOrder ctx direct).filterMap (viewKey ctx s))
    (st : St) :
    Dace.bindViews ctx ρ s' (s'.viewOrder ctx direct') st = Dace.bindViews ctx ρ s (s.viewOrder ctx direct) st := by
  rw [bindViews_keys, bindViews_keys, hv]

theorem respectsEdges {o : List NodeId} (hnd : o.Nodup) (hX : ∀ x ∈ o, x ∈ X)
    (h : s'.respectsEdges o = true) : s.respectsEdges o = true := by
  rw [State.respectsEdges_iff hnd] at h ⊢
  intro e he hs hd
  have he' : e ∈ s.outEdges e.src := List.mem_filter.2 ⟨he, by simp⟩
  rw [← hF.outE _ (hX _ hs)] at he'
  exact h e (List.mem_filter.1 he').1 hs hd

theorem topo {all' all direct' direct o : List NodeId} (hl : ListSame ctx s' s X all' all direct' direct)
    (ho : s'.isTopologicalOrder all' o = true) : s.isTopologicalOrder all o = true := by
  have hp : o.Perm all' := State.isTopologicalOrder_perm (hl.perm.nodup_iff.2 hl.nodup) ho
  have ho' := ho
  simp only [State.isTopologicalOrder, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at ho' ⊢
  obtain ⟨⟨⟨hlen, -⟩, hnd⟩, hr⟩ := ho'
  refine ⟨⟨⟨by rw [hlen, hl.perm.length_eq], fun x hx => ?_⟩, hnd⟩, ?_⟩
  · simpa using (hp.trans hl.perm).mem_iff.2 hx
  · exact hF.respectsEdges hnd (fun x hx => hl.allX x ((hp.trans hl.perm).mem_iff.1 hx)) hr

theorem hbConflict {x y : NodeId × Log} (hx : x.1 ∈ X) (hy : y.1 ∈ X) :
    s'.hbConflict x y = s.hbConflict x y := by
  simp only [State.hbConflict, hF.reach _ hx _ hy, hF.reach _ hy _ hx]

theorem hbFree : ∀ {tr : List (NodeId × Log)}, (∀ x ∈ tr, x.1 ∈ X) →
    s'.hbFree tr = s.hbFree tr ∧ s'.hbWitness tr = s.hbWitness tr
  | [], _ => ⟨rfl, rfl⟩
  | x :: xs, h => by
    have hx := h x (List.mem_cons_self ..)
    have hxs : ∀ y ∈ xs, y.1 ∈ X := fun y hy => h y (List.mem_cons_of_mem _ hy)
    obtain ⟨ih₁, ih₂⟩ := hbFree hxs
    have hc : ∀ y ∈ xs, s'.hbConflict x y = s.hbConflict x y := fun y hy => hF.hbConflict hx (hxs y hy)
    have hall : xs.all (fun y => !s'.hbConflict x y) = xs.all (fun y => !s.hbConflict x y) := by
      rw [Bool.eq_iff_iff, List.all_eq_true, List.all_eq_true]
      exact forall₂_congr fun y hy => by rw [hc y hy]
    have hfind : xs.find? (fun y => s'.hbConflict x y) = xs.find? (fun y => s.hbConflict x y) :=
      List.find?_congr fun y hy => hc y hy
    constructor
    · simp only [State.hbFree, hall, ih₁]
    · simp only [State.hbWitness, hfind, ih₂]

omit hF in
theorem filter_X {all' all direct' direct : List NodeId} (hl : ListSame ctx s' s X all' all direct' direct)
    (o : List NodeId) : ∀ id ∈ o.filter direct'.contains, id ∈ X := fun id hid =>
  hl.directX id (hl.dperm.mem_iff.1 (by simpa using (List.mem_filter.1 hid).2))

omit hF in
theorem filter_eq {o direct' direct : List NodeId} (hp : direct'.Perm direct) :
    o.filter direct'.contains = o.filter direct.contains :=
  List.filter_congr fun x _ => by simp only [List.contains_eq_mem, hp.mem_iff]

set_option linter.unusedSectionVars false

mutual

theorem _root_.Dace.NodeExec.local {ρ : Valuation} {n : DNode} {st : St} {r : Res St} :
    NodeExec ctx ρ s' n st r → n.id ∈ X → s.node? n.id = some n → NodeExec ctx ρ s n st r
  | .simple hs, hn, _ => by
    rw [(hF.nodeLocal hn).execSimple]
    exact .simple hs
  | .mapSetupErr hk hs, hn, _ => .mapSetupErr hk (by rw [← (hF.nodeLocal hn).mapSetup]; exact hs)
  | .mapIterErr hk hs hi, hn, hnode => by
    obtain ⟨hl, hpriv⟩ := hF.scopes _ hn _ _ hnode hk
    have hi' := IterExec.local hi hl
    rw [hpriv] at hi'
    exact .mapIterErr hk (by rw [← (hF.nodeLocal hn).mapSetup]; exact hs) hi'
  | .mapScope hk hs hi, hn, hnode => by
    obtain ⟨hl, hpriv⟩ := hF.scopes _ hn _ _ hnode hk
    have hi' := IterExec.local hi hl
    rw [hpriv] at hi'
    exact .mapScope hk (by rw [← (hF.nodeLocal hn).mapSetup]; exact hs) hi'
  | .consumeSetupErr hk hs, hn, _ => .consumeSetupErr hk (by rw [← (hF.nodeLocal hn).consumeSetup]; exact hs)
  | .consumeTasksErr hk hs ht, hn, hnode => by
    obtain ⟨hl, hpriv, hstr⟩ := hF.consumes _ hn _ _ hnode hk
    have ht' := ConsumeTasks.local ht hl hstr
    rw [hpriv] at ht'
    exact .consumeTasksErr hk (by rw [← (hF.nodeLocal hn).consumeSetup]; exact hs) ht'
  | .consumeScope hk hs ht, hn, hnode => by
    obtain ⟨hl, hpriv, hstr⟩ := hF.consumes _ hn _ _ hnode hk
    have ht' := ConsumeTasks.local ht hl hstr
    rw [hpriv] at ht'
    exact .consumeScope hk (by rw [← (hF.nodeLocal hn).consumeSetup]; exact hs) ht'
  | .nestedSetupErr hk hs, hn, _ => .nestedSetupErr hk (by rw [← (hF.nodeLocal hn).nestedSetup]; exact hs)
  | .nestedInnerErr hk hs hr, hn, _ => .nestedInnerErr hk (by rw [← (hF.nodeLocal hn).nestedSetup]; exact hs) hr
  | .nested hk hs hr, hn, _ => by
    rw [(hF.nodeLocal hn).nestedFinish]
    exact .nested hk (by rw [← (hF.nodeLocal hn).nestedSetup]; exact hs) hr

theorem _root_.Dace.SeqExec.local {ρ : Valuation} {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s' ids st r → (∀ id ∈ ids, id ∈ X) → SeqExec ctx ρ s ids st r
  | .nil, _ => .nil
  | .missing h, hids => .missing (by rw [← hF.node _ (hids _ (List.mem_cons_self ..))]; exact h)
  | .headErr h hn, hids => by
    have hX := hids _ (List.mem_cons_self ..)
    have h' := (hF.node _ hX).symm.trans h
    obtain ⟨-, hid⟩ := State.node?_some h
    exact .headErr h' (NodeExec.local hn (hid ▸ hX) (hid ▸ h'))
  | .tailErr h hn hr, hids => by
    have hX := hids _ (List.mem_cons_self ..)
    have h' := (hF.node _ hX).symm.trans h
    obtain ⟨-, hid⟩ := State.node?_some h
    exact .tailErr h' (NodeExec.local hn (hid ▸ hX) (hid ▸ h'))
      (SeqExec.local hr fun id hid => hids id (List.mem_cons_of_mem _ hid))
  | .cons h hn hr, hids => by
    have hX := hids _ (List.mem_cons_self ..)
    have h' := (hF.node _ hX).symm.trans h
    obtain ⟨-, hid⟩ := State.node?_some h
    exact .cons h' (NodeExec.local hn (hid ▸ hX) (hid ▸ h'))
      (SeqExec.local hr fun id hid => hids id (List.mem_cons_of_mem _ hid))

theorem _root_.Dace.OrderedExec.local {ρ : Valuation} {all' all direct' direct : List NodeId} {st : St} {r : Res St} :
    OrderedExec ctx ρ s' all' direct' st r → ListSame ctx s' s X all' all direct' direct →
      OrderedExec ctx ρ s all direct st r
  | .prologueErr hp, hl => .prologueErr (by rw [← prologue hl.views]; exact hp)
  | .seqErr ho hp hs, hl => by
    refine .seqErr (hF.topo hl ho) (by rw [← prologue hl.views]; exact hp) ?_
    rw [← filter_eq hl.dperm]
    exact SeqExec.local hs (filter_X hl _)
  | .hbRace (order := o) (st₁ := st₁) (st₂ := st₂) (tr := tr) ho hp hs hf, hl => by
    have hids := filter_X hl o
    obtain ⟨hb₁, hb₂⟩ := hF.hbFree fun x hx => hids _ (SeqExec.trace_mem hs x hx).1
    have hs' : SeqExec ctx ρ s (o.filter direct.contains) st₁ (.ok (st₂, tr)) := by
      rw [← filter_eq hl.dperm]
      exact SeqExec.local hs hids
    have hp' : Dace.bindViews ctx ρ s (s.viewOrder ctx direct) st = .ok st₁ := by
      rw [← prologue hl.views]; exact hp
    rw [hb₂]
    exact .hbRace (hF.topo hl ho) hp' hs' (hb₁ ▸ hf)
  | .run (order := o) (st₁ := st₁) (st₂ := st₂) (tr := tr) ho hp hs hf, hl => by
    have hids := filter_X hl o
    obtain ⟨hb₁, hb₂⟩ := hF.hbFree fun x hx => hids _ (SeqExec.trace_mem hs x hx).1
    have hs' : SeqExec ctx ρ s (o.filter direct.contains) st₁ (.ok (st₂, tr)) := by
      rw [← filter_eq hl.dperm]
      exact SeqExec.local hs hids
    have hp' : Dace.bindViews ctx ρ s (s.viewOrder ctx direct) st = .ok st₁ := by
      rw [← prologue hl.views]; exact hp
    exact .run (hF.topo hl ho) hp' hs' (hb₁ ▸ hf)

theorem _root_.Dace.IterExec.local {ρ : Valuation} {P : NodeId} {params : List Sym} {priv : List Name} {pre : St}
    {pts : List Idx} {r : Res (List Log)} :
    IterExec ctx ρ s' P params priv pre pts r →
      ListSame ctx s' s X (s'.scopeNodes P) (s.scopeNodes P) (s'.directChildren (some P)) (s.directChildren (some P)) →
      IterExec ctx ρ s P params priv pre pts r
  | .nil, _ => .nil
  | .headErr hb, hl => .headErr (OrderedExec.local hb hl)
  | .tailErr hr, hl => .tailErr (IterExec.local hr hl)
  | .cons hb hr, hl => .cons (OrderedExec.local hb hl) (IterExec.local hr hl)

theorem _root_.Dace.ConsumeTasks.local {ρ : Valuation} {P : NodeId} {c : ConsumeNode} {l : Loc} {np : Int}
    {priv : List Name} {pre : St} {q : List Val} {r : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx ρ s' P c l np priv pre q r →
      ListSame ctx s' s X (s'.scopeNodes P) (s.scopeNodes P) (s'.directChildren (some P)) (s.directChildren (some P)) →
      s'.consumedStream? P = s.consumedStream? P →
      ConsumeTasks ctx ρ s P c l np priv pre q r
  | .condErr hc, _, _ => .condErr hc
  | .stop hc, _, _ => .stop hc
  | .bodyErr hc hq hp hb, hl, hs => by
    have hb' := OrderedExec.local hb hl
    rw [taskStart_of_stream hs] at hb'
    exact .bodyErr hc hq hp hb'
  | .restErr hc hq hp hb hr, hl, hs => by
    have hb' := OrderedExec.local hb hl
    rw [taskStart_of_stream hs] at hb'
    exact .restErr hc hq hp hb' (ConsumeTasks.local hr hl hs)
  | .step hc hq hp hb hr, hl, hs => by
    have hb' := OrderedExec.local hb hl
    rw [taskStart_of_stream hs] at hb'
    exact .step hc hq hp hb' (ConsumeTasks.local hr hl hs)

end

/-- Components in corresponding order. -/
theorem _root_.Dace.CompsExec.local {ρ : Valuation} {m : Mem} :
    ∀ {comps' comps : List (List NodeId)} {r : Res (List Log)}, CompsExec ctx ρ s' m comps' r →
      List.Forall₂ (fun c' c => ListSame ctx s' s X c' c (s'.topLevelOf c') (s.topLevelOf c)) comps' comps →
      CompsExec ctx ρ s m comps r
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .headErr hc, .cons hl _ => .headErr (OrderedExec.local hF hc hl)
  | _, _, _, .tailErr hr, .cons _ hrest => .tailErr (CompsExec.local hr hrest)
  | _, _, _, .cons hc hr, .cons hl hrest => .cons (OrderedExec.local hF hc hl) (CompsExec.local hr hrest)

end LocalFacts

end Dace

namespace Dace

/-! ## States -/

/-- **A state that agrees with `s` on its nodes runs like `s`**: the same
components in the same order (each up to a permutation of its nodes), the
same transients. -/
theorem StateExec.local {ctx : Ctx} {s' s : State} {X : List NodeId} (hF : LocalFacts ctx s' s X)
    (hc : List.Forall₂ (fun c' c => ListSame ctx s' s X c' c (s'.topLevelOf c') (s.topLevelOf c))
      s'.components s.components)
    (ht : ctx.stateTransients s' = ctx.stateTransients s) {cst : CSt} {r : Res (Ctl × CSt)} :
    StateExec ctx cst s' r → StateExec ctx cst s r
  | .compsErr h => .compsErr (ht ▸ CompsExec.local hF h hc)
  | .run h => .run (ht ▸ CompsExec.local hF h hc)

/-! ## The check -/

/-- Optional nodes compared with the sound `DNode.beq`. -/
def optNodeBEq : Option DNode → Option DNode → Bool
  | some a, some b => DNode.beq a b
  | none, none => true
  | _, _ => false

theorem optNodeBEq_sound : ∀ {a b : Option DNode}, optNodeBEq a b = true → a = b
  | some _, some _, h => by rw [DNode.beq_sound h]
  | none, none, _ => rfl
  | some _, none, h => by cases h
  | none, some _, h => by cases h

/-- Decision procedure for `ListSame` (in the check context). -/
def listSame? (g : SDFG) (s' s : State) (X all' all direct' direct : List NodeId) : Bool :=
  all'.isPerm all && decide all.Nodup && all.all X.contains && direct'.isPerm direct && direct.all X.contains &&
  (s'.viewOrder (checkCtx g) direct').filterMap (viewKey (checkCtx g) s') ==
    (s.viewOrder (checkCtx g) direct).filterMap (viewKey (checkCtx g) s)

theorem listSame?_sound {g : SDFG} {s' s : State} {X all' all direct' direct : List NodeId}
    (h : listSame? g s' s X all' all direct' direct = true) (I : Interp) :
    ListSame { I, g } s' s X all' all direct' direct := by
  unfold listSame? at h
  simp only [Bool.and_eq_true, List.isPerm_iff, decide_eq_true_eq, List.all_eq_true, List.contains_eq_mem,
    beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩, h₅⟩, h₆⟩ := h
  exact ⟨h₁, h₂, fun x hx => by simpa using h₃ x hx, h₄, fun x hx => by simpa using h₅ x hx, h₆⟩

/-- Decision procedure for `LocalFacts` (in the check context). -/
def localSame? (g : SDFG) (s' s : State) (X : List NodeId) : Bool :=
  let ctx := checkCtx g
  X.all fun id =>
    optNodeBEq (s'.node? id) (s.node? id) &&
    s'.inEdges id == s.inEdges id && s'.outEdges id == s.outEdges id &&
    (s.inEdges id ++ s.outEdges id).all (fun e => edgeFp ctx s' e == edgeFp ctx s e) &&
    X.all (fun b => s'.reaches id b == s.reaches id b) &&
    match s.node? id with
    | some n => match n.kind with
      | .consumeEntry _ =>
        listSame? g s' s X (s'.scopeNodes id) (s.scopeNodes id) (s'.directChildren (some id))
          (s.directChildren (some id)) &&
        ctx.privateTransients s' id == ctx.privateTransients s id && s'.consumedStream? id == s.consumedStream? id
      | .mapEntry _ =>
        listSame? g s' s X (s'.scopeNodes id) (s.scopeNodes id) (s'.directChildren (some id))
          (s.directChildren (some id)) &&
        ctx.privateTransients s' id == ctx.privateTransients s id
      | _ => true
    | none => true

theorem localSame?_spec {g : SDFG} {s' s : State} {X : List NodeId} (h : localSame? g s' s X = true) (I : Interp) :
    LocalFacts { I, g } s' s X := by
  unfold localSame? at h
  simp only [List.all_eq_true] at h
  have hx : ∀ id ∈ X, _ := fun id hid => by
    have := h id hid
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_append] at this
    exact this
  have edge : ∀ id ∈ X, ∀ e, e ∈ s.inEdges id ∨ e ∈ s.outEdges id → EdgeSame { I, g } s' s e := by
    intro id hid e he
    have := (hx id hid).1.1.2 e he
    simp only [edgeFp, Prod.mk.injEq] at this
    obtain ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩ := this
    exact ⟨h₁, h₂, h₃, h₄, h₅, h₆⟩
  refine
    { node := fun id hid => optNodeBEq_sound (hx id hid).1.1.1.1.1
      inE := fun id hid => (hx id hid).1.1.1.1.2
      outE := fun id hid => (hx id hid).1.1.1.2
      edgeIn := fun id hid e he => edge id hid e (.inl he)
      edgeOut := fun id hid e he => edge id hid e (.inr he)
      reach := fun a ha b hb => by simpa using (hx a ha).1.2 b hb
      consumes := fun id hid n c hn hk => ?_
      scopes := fun id hid n mp hn hk => ?_ }
  · have := (hx id hid).2
    rw [hn] at this
    simp only [hk, Bool.and_eq_true, beq_iff_eq] at this
    exact ⟨listSame?_sound this.1.1 I, this.1.2, this.2⟩
  · have := (hx id hid).2
    rw [hn] at this
    simp only [hk, Bool.and_eq_true, beq_iff_eq] at this
    exact ⟨listSame?_sound this.1 I, this.2⟩

/-- The components correspond in order. -/
def compsSame? (g : SDFG) (s' s : State) (X : List NodeId) : Bool :=
  s'.components.length == s.components.length &&
  (s'.components.zip s.components).all fun (c', c) => listSame? g s' s X c' c (s'.topLevelOf c') (s.topLevelOf c)

theorem compsSame?_sound {g : SDFG} {s' s : State} {X : List NodeId} (h : compsSame? g s' s X = true) (I : Interp) :
    List.Forall₂ (fun c' c => ListSame { I, g } s' s X c' c (s'.topLevelOf c') (s.topLevelOf c))
      s'.components s.components := by
  unfold compsSame? at h
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
  exact List.forall₂_iff_zip.2 ⟨h.1, fun {c' c} hz => listSame?_sound (h.2 (c', c) hz) I⟩

/-- **The check of the state replacement rule**: `s'` and `s` run alike in
both directions (same nodes, components, transients and mention counts). -/
def stateSame? (g : SDFG) (s' s : State) : Bool :=
  let X := s.nodes.map (·.id)
  localSame? g s' s X && localSame? g s s' X && compsSame? g s' s X && compsSame? g s s' X &&
  (checkCtx g).stateTransients s' == (checkCtx g).stateTransients s &&
  sameTransientMentions? g (.state s') (.state s)

/-- **State replacement rule**: replacing a state by one that `stateSame?`
accepts gives an equivalent SDFG. -/
theorem Congr.replaceState {g : SDFG} {p : List PathStep} {i : BlockId} {s s' : State}
    (hat : g.top.blockAt? p i = some (.state s)) (hchk : stateSame? g s' s = true) (hsup : Supported g.prog) :
    Equiv (g.setBlockAt p i (.state s')).prog g.prog := by
  unfold stateSame? at hchk
  simp only [Bool.and_eq_true, beq_iff_eq] at hchk
  obtain ⟨⟨⟨⟨⟨h₁, h₂⟩, h₃⟩, h₄⟩, h₅⟩, hm⟩ := hchk
  refine Congr.blockEquiv hat hm (fun I _ cst r' h => ?_) (fun I _ cst r h => ?_) hsup
  · cases h with
    | state h =>
      exact MatchedE.of_rel (.state (StateExec.local (localSame?_spec h₁ I) (compsSame?_sound h₃ I) h₅ h))
        (Res.Rel.refl CtlRel.refl _)
  · cases h with
    | state h =>
      exact MatchedE.of_rel (.state (StateExec.local (localSame?_spec h₂ I) (compsSame?_sound h₄ I) h₅.symm h))
        (Res.Rel.refl CtlRel.refl _)

end Dace
