-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.ScopeBody
import Dace.Equiv.EvalCongr
import Dace.STree.Semantics

/-!
# SDFG and schedule tree: corresponding leaves (M5a)

A schedule tree node and the SDFG node it came from execute the same
container-level code (`Dace.STree.Semantics.Leaf` reuses the leaf functions
of the SDFG semantics).  This file states when the two calls coincide:

* the **keys** of an SDFG node are what its memlet edges contribute to its
  execution (`InKey`, `OutKey`: connector, container endpoint, subset, WCR,
  `allow_oob`; `CopyKey`: the two endpoints of a copy), computed from the
  structural endpoints of the leaf edges (`State.codeIns`, `codeOuts`,
  `copiesInto`); the keys of a tree node are read off its normalized
  memlets (`Memlet.inKey`, `outKey`, `copyKey`);
* when the keys agree (and the edges are well shaped: `State.CodeShape`,
  `State.CopyShape`), the SDFG leaf function equals the tree leaf function
  on every state (`execCode_eq`, `execReduce_eq`, `execCopiesInto_eq`,
  `mapSetup_eq`), in contexts that agree on what leaf functions read
  (`Ctx.eraseL`).
-/

set_option autoImplicit false

namespace Dace

/-! ## Keys -/

/-- What an input memlet of a code node determines. -/
structure InKey where
  conn : Conn
  data : Name
  sub : Option Subset
  oob : Bool
  deriving DecidableEq, Repr

/-- What an output memlet of a code node determines. -/
structure OutKey where
  conn : Conn
  data : Name
  sub : Option Subset
  wcr : Option WcrOp
  oob : Bool
  deriving DecidableEq, Repr

/-- What a copy determines: `dst[dsub] = src[ssub]`. -/
structure CopyKey where
  dst : Name
  src : Name
  ssub : Option Subset
  dsub : Option Subset
  wcr : Option WcrOp
  oob : Bool
  deriving DecidableEq, Repr

namespace Memlet

def inKey (c : Conn) (m : Memlet) : InKey := ⟨c, m.data, m.dataSubset, m.allowOob⟩
def outKey (c : Conn) (m : Memlet) : OutKey := ⟨c, m.data, m.dataSubset, m.wcr, m.allowOob⟩
def copyKey (target : Name) (m : Memlet) : CopyKey :=
  ⟨target, m.data, m.dataSubset, m.otherSubset, m.wcr, m.allowOob⟩

end Memlet

/-- The key of an input memlet in the tree context: inside a consume scope,
a memlet on the consumed stream reads the chunk container
(`TCtx.inData`, `Decisions.md#D38`). -/
def TCtx.inKey (tc : TCtx) (c : Conn) (m : Memlet) : InKey := ⟨c, (tc.inData m).1, (tc.inData m).2, m.allowOob⟩

/-- The key of a copy in the tree context (its source through `TCtx.inData`). -/
def TCtx.copyKey (tc : TCtx) (target : Name) (m : Memlet) : CopyKey :=
  ⟨target, (tc.inData m).1, (tc.inData m).2, m.otherSubset, m.wcr, m.allowOob⟩

namespace State

/-- The key of a memlet in-edge of a code node with a container source. -/
def inKey? (s : State) (e : DEdge) : Option InKey :=
  match e.data, s.srcEndpoint e with
  | .memlet _, .data d sub => some ⟨e.dstConn.getD "", d, sub, e.allowOob⟩
  | _, _ => none

/-- The keys of the dynamic inputs of a scope entry, in in-edge order: each
from a container element into a connector. -/
def dynKeys? (s : State) (n : DNode) : Option (List InKey) :=
  ((s.inEdges n.id).filter s.isDynamicInput).mapM fun e =>
    match e.data, e.dstConn, s.srcEndpoint e with
    | .memlet _, some c, .data d sub => some ⟨c, d, sub, e.allowOob⟩
    | _, _, _ => none

/-- The key of a memlet out-edge of a code node with a container destination. -/
def outKey? (s : State) (e : DEdge) : Option OutKey :=
  match e.data, s.dstEndpoint e with
  | .memlet m, .data d sub => some ⟨e.srcConn.getD "", d, sub, m.wcr, m.allowOob⟩
  | _, _ => none

/-- Input keys of a code node, in edge order. -/
def codeIns (s : State) (n : DNode) : List InKey := (s.inEdges n.id).filterMap s.inKey?

/-- Output keys of a code node, in edge order. -/
def codeOuts (s : State) (n : DNode) : List OutKey := (s.outEdges n.id).filterMap s.outKey?

/-- Is `e` a copy executed at its destination (`execCopiesInto`)? -/
def isCopyEdge (ctx : Ctx) (s : State) (e : DEdge) : Bool :=
  match e.data with
  | .dep => false
  | .memlet _ => !(ctx.isViewDef s e || s.isRouting e) && (s.srcEndpoint e).isData

/-- The key of a copy edge. -/
def copyKey? (ctx : Ctx) (s : State) (e : DEdge) : Option CopyKey :=
  if s.isCopyEdge ctx e then
    match e.data, s.srcEndpoint e, s.dstEndpoint e with
    | .memlet m, .data sd ssub, .data dd dsub => some ⟨dd, sd, ssub, dsub, m.wcr, m.allowOob⟩
    | _, _, _ => none
  else none

/-- The copies executed at node `n`, in edge order. -/
def copiesInto (ctx : Ctx) (s : State) (n : DNode) : List CopyKey := (s.inEdges n.id).filterMap (s.copyKey? ctx)

/-- Well-shaped edges of a code node: memlet in-edges read containers,
memlet out-edges write containers, input connectors are distinct. -/
structure CodeShape (s : State) (n : DNode) : Prop where
  ins : ∀ e ∈ s.inEdges n.id, (∃ m, e.data = .memlet m) → (s.srcEndpoint e).isData = true
  outs : ∀ e ∈ s.outEdges n.id, (∃ m, e.data = .memlet m) → (s.dstEndpoint e).isData = true
  conns : ((s.codeIns n).map (·.conn)).Nodup

/-- Decision procedure for `CodeShape`. -/
def codeShape? (s : State) (n : DNode) : Bool :=
  (s.inEdges n.id).all (fun e => match e.data with | .memlet _ => (s.srcEndpoint e).isData | .dep => true) &&
  (s.outEdges n.id).all (fun e => match e.data with | .memlet _ => (s.dstEndpoint e).isData | .dep => true) &&
  decide ((s.codeIns n).map (·.conn)).Nodup

theorem codeShape?_sound {s : State} {n : DNode} (h : s.codeShape? n = true) : s.CodeShape n := by
  simp only [codeShape?, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  refine ⟨fun e he ⟨m, hm⟩ => ?_, fun e he ⟨m, hm⟩ => ?_, h.2⟩
  · have := h.1.1 e he; simp only [hm] at this; exact this
  · have := h.1.2 e he; simp only [hm] at this; exact this

/-- Well-shaped copies into a node: no reference is set, and every copy
edge has a container destination. -/
structure CopyShape (ctx : Ctx) (s : State) (n : DNode) : Prop where
  noRef : ∀ e ∈ s.inEdges n.id, ∀ m, e.data = .memlet m → (ctx.isViewDef s e || s.isRouting e) = false →
    refSetTarget? ctx s e = none
  dst : ∀ e ∈ s.inEdges n.id, s.isCopyEdge ctx e = true → (s.dstEndpoint e).isData = true

def copyShape? (ctx : Ctx) (s : State) (n : DNode) : Bool :=
  (s.inEdges n.id).all fun e => match e.data with
    | .dep => true
    | .memlet _ => (ctx.isViewDef s e || s.isRouting e) ||
        ((refSetTarget? ctx s e).isNone && (!(s.srcEndpoint e).isData || (s.dstEndpoint e).isData))

theorem copyShape?_sound {ctx : Ctx} {s : State} {n : DNode} (h : s.copyShape? ctx n = true) :
    s.CopyShape ctx n := by
  simp only [copyShape?, List.all_eq_true] at h
  refine ⟨fun e he m hm hv => ?_, fun e he hc => ?_⟩
  · have := h e he
    simp only [hm, hv, Bool.false_or, Bool.and_eq_true, Option.isNone_iff_eq_none] at this
    exact this.1
  · have := h e he
    simp only [isCopyEdge] at hc
    cases hm : e.data with
    | dep => rw [hm] at hc; cases hc
    | memlet m =>
      rw [hm] at hc this
      simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hc
      simp only [hc.1, Bool.false_or, Bool.and_eq_true, hc.2, Bool.not_true,
        ] at this
      exact this.2

end State

/-! ## Folds over keyed lists -/

/-- A fold over `l` whose step on an element is the identity when the
element has no key and a step on the key otherwise is the fold over the
keys. -/
theorem List.foldlM_filterMap {α κ β : Type} (k : α → Option κ) (f : β → α → Res β) (g : β → κ → Res β) :
    ∀ (l : List α), (∀ a ∈ l, ∀ b, (k a = none → f b a = .ok b) ∧ ∀ x, k a = some x → f b a = g b x) →
      ∀ b, l.foldlM f b = (l.filterMap k).foldlM g b
  | [], _, _ => rfl
  | a :: l, h, b => by
    obtain ⟨h₀, h₁⟩ := h a List.mem_cons_self b
    have ih := List.foldlM_filterMap k f g l (fun a' ha' => h a' (List.mem_cons_of_mem _ ha'))
    cases hk : k a with
    | none => simp only [List.foldlM_cons, h₀ hk, Res.ok_bind, List.filterMap_cons, hk, ih]
    | some x =>
      simp only [List.foldlM_cons, h₁ x hk, List.filterMap_cons, hk]
      cases g b x with
      | ok b' => simp only [Res.ok_bind, ih]
      | err e => rfl

/-- A fold over `l` whose step factors through a key function is the fold
over the keys. -/
theorem List.foldlM_map_eq {α κ β : Type} (k : α → κ) (f : β → α → Res β) (g : β → κ → Res β)
    (l : List α) (h : ∀ a ∈ l, ∀ b, f b a = g b (k a)) (b : β) : l.foldlM f b = (l.map k).foldlM g b := by
  rw [List.foldlM_map]
  exact List.foldlM_congr_mem h b

/-- Filtering before or after a partial map, when the filter is decided by
the key. -/
theorem List.filter_filterMap_of {α κ : Type} (P : α → Bool) (k : α → Option κ) (q : κ → Bool) :
    ∀ (l : List α), (∀ a ∈ l, P a = (k a).any q) → (l.filter P).filterMap k = (l.filterMap k).filter q
  | [], _ => rfl
  | a :: l, h => by
    have ih := List.filter_filterMap_of P k q l (fun a' ha' => h a' (List.mem_cons_of_mem _ ha'))
    have ha := h a List.mem_cons_self
    cases hk : k a with
    | none =>
      rw [hk] at ha
      simp only [Option.any_none] at ha
      simp [ha, hk, ih]
    | some x =>
      rw [hk] at ha
      simp only [Option.any_some] at ha
      cases hq : q x with
      | false => simp [ha, hq, hk, ih]
      | true => simp [ha, hq, hk, ih]

/-- The first key of the elements satisfying `P` is the key of the first
element satisfying `P`, when those all have keys. -/
theorem List.head?_filterMap_filter {α κ : Type} (P : α → Bool) (k : α → Option κ) :
    ∀ (l : List α), (∀ a ∈ l, P a = true → (k a).isSome = true) →
      ((l.filter P).filterMap k).head? = (l.find? P).bind k
  | [], _ => rfl
  | a :: l, h => by
    have ih := List.head?_filterMap_filter P k l (fun a' ha' => h a' (List.mem_cons_of_mem _ ha'))
    cases hP : P a with
    | false => simp [hP, ih]
    | true =>
      obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 (h a List.mem_cons_self hP)
      simp [hP, hx]

/-- At most one element of a list with distinct keys has a given key. -/
theorem List.length_filter_le_one {α κ : Type} [DecidableEq κ] (f : α → κ) (c : κ) :
    ∀ (l : List α), (l.map f).Nodup → (l.filter (fun a => f a = c)).length ≤ 1
  | [], _ => by simp
  | a :: l, h => by
    rw [List.map_cons, List.nodup_cons] at h
    have ih := List.length_filter_le_one f c l h.2
    by_cases ha : f a = c
    · have : l.filter (fun a => f a = c) = [] := by
        rw [List.filter_eq_nil_iff]
        intro b hb hc
        exact h.1 (List.mem_map.2 ⟨b, hb, by simpa [ha] using hc⟩)
      simp [ha, this]
    · simp [ha, ih]

/-- The memlets of a map on a connector, as pairs. -/
theorem MemletMap.on_eq (mm : MemletMap) (c : Conn) : mm.on c = ((mm.pairs.filter (·.1 = c)).map Prod.snd) := by
  cases mm with
  | dict items => rfl
  | set items =>
    simp only [MemletMap.on, MemletMap.pairs, List.filter_map, Function.comp_def, List.map_map]
    split
    · rename_i h
      have : ∀ x : Memlet, decide (("", x).1 = c) = true := fun _ => by simp [h]
      simp only [this, List.filter_true]
      exact (List.map_id' items).symm.trans (by rfl)
    · rename_i h
      have : ∀ x : Memlet, decide (("", x).1 = c) = false := fun _ => by simp [Ne.symm h]
      simp only [this, List.filter_false, List.map_nil]

theorem MemletMap.memlets_eq (mm : MemletMap) : mm.memlets = mm.pairs.map Prod.snd := by
  cases mm with
  | dict items => rfl
  | set items => simp [MemletMap.memlets, MemletMap.pairs]

/-! ## Contexts that agree on what leaf functions read -/

/-- The part of a context the leaf functions read: interpretation, memory
names and descriptor repository; unlike `Ctx.erase`, not the fresh-name scope
(read by nested invocations only). -/
def Ctx.eraseL (ctx : Ctx) : Ctx :=
  { I := ctx.I, g := { name := "", arrays := ctx.g.arrays, top := {} }, cname := ctx.cname }

theorem Ctx.leaf_eq {α : Sort _} {c₁ c₂ : Ctx} (he : c₁.eraseL = c₂.eraseL) (f : Ctx → α)
    (hf : ∀ c, f c = f c.eraseL) : f c₁ = f c₂ := by
  rw [hf c₁, he, ← hf c₂]

namespace Corr

section Leaf

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL)
  (ρ : Valuation) {s : State}

theorem TCtx.inData_none {tc : TCtx} (hcons : tc.consumed = none) (m : Memlet) :
    tc.inData m = (m.data, m.dataSubset) := by
  simp [TCtx.inData, TCtx.isChunk, hcons]

theorem TCtx.inKey_none {tc : TCtx} (hcons : tc.consumed = none) (c : Conn) (m : Memlet) :
    tc.inKey c m = m.inKey c := by
  simp [TCtx.inKey, TCtx.inData_none hcons, Memlet.inKey]

theorem TCtx.copyKey_none {tc : TCtx} (hcons : tc.consumed = none) (t : Name) (m : Memlet) :
    tc.copyKey t m = m.copyKey t := by
  simp [TCtx.copyKey, TCtx.inData_none hcons, Memlet.copyKey]

include he

/-- Input binding: the SDFG binds the keys of the memlet in-edges, the tree
those of its input memlets. -/
theorem bindInputs_eq {n : DNode} {ins : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2) (st : St) :
    bindInputs ctx ρ s n st = bindInputsT tc ρ ins st := by
  let g : List (Conn × ConnIn) × St → InKey → Res (List (Conn × ConnIn) × St) := fun (acc, st) k => do
    let (ci, st) ← bindInData ctx ρ k.data k.sub k.oob st
    pure (acc ++ [(k.conn, ci)], st)
  unfold bindInputs bindInputsT
  rw [List.foldlM_filterMap s.inKey? _ g (s.inEdges n.id) ?hs,
    List.foldlM_map_eq (fun p => tc.inKey p.1 p.2) _ g ins.pairs ?ht, ← hk]
  · rfl
  case hs =>
    intro e he' ⟨acc, st⟩
    refine ⟨fun hn => ?_, fun x hx => ?_⟩
    · cases hd : e.data with
      | dep => rfl
      | memlet m =>
        have := hsh.ins e he' ⟨m, hd⟩
        simp only [State.inKey?, hd] at hn
        cases hep : s.srcEndpoint e <;> simp_all [State.Endpoint.isData]
    · cases hd : e.data with
      | dep => simp [State.inKey?, hd] at hx
      | memlet m =>
        simp only [State.inKey?, hd] at hx
        cases hep : s.srcEndpoint e with
        | data d sub =>
          simp only [hep, Option.some.injEq] at hx
          subst hx
          simp only [bindIn, hep, g]
          cases bindInData ctx ρ d sub e.allowOob st with
          | ok p => rfl
          | err e => rfl
        | _ => simp [hep] at hx
  case ht =>
    intro p _ ⟨acc, st⟩
    simp only [bindMemletIn, TCtx.inKey, g]
    rcases hid : tc.inData p.2 with ⟨d, sub⟩
    simp only
    rw [Ctx.leaf_eq he bindInData (fun _ => rfl)]

omit he in
/-- The output keys of an SDFG code node on connector `c` are those of the
tree's output memlets on `c`. -/
theorem outKeys_on {n : DNode} {outs : MemletMap} (hsh : s.CodeShape n)
    (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2) (c : Conn) :
    (s.outEdgesOn n c).filterMap s.outKey? = (outs.on c).map (Memlet.outKey c) := by
  rw [MemletMap.on_eq, List.map_map]
  have h₁ : (outs.pairs.filter (·.1 = c)).map (Memlet.outKey c ∘ Prod.snd) =
      (outs.pairs.map fun p => Memlet.outKey p.1 p.2).filter (·.conn = c) := by
    rw [List.filter_map]
    refine List.map_congr_left fun p hp => ?_
    simp only [List.mem_filter, decide_eq_true_eq] at hp
    simp [Memlet.outKey, hp.2]
  rw [h₁, ← hko]
  unfold State.outEdgesOn State.codeOuts
  refine List.filter_filterMap_of _ _ _ _ fun e he => ?_
  cases hd : e.data with
  | dep => simp [State.outKey?, hd]
  | memlet m =>
    have := hsh.outs e he ⟨m, hd⟩
    cases hep : s.dstEndpoint e <;> simp_all [State.outKey?, State.Endpoint.isData]

omit he in
/-- A fold over the memlet out-edges of a code node on connector `c` whose
step depends only on the output key is the fold over the tree's memlets on
`c`. -/
theorem outs_fold {n : DNode} {outs : MemletMap} {c : Conn} (hsh : s.CodeShape n)
    (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2) (F : St → DEdge → Res St)
    (G : St → OutKey → Res St) (hF : ∀ e ∈ s.outEdgesOn n c, ∀ k, s.outKey? e = some k → ∀ st, F st e = G st k)
    (st : St) : (s.outEdgesOn n c).foldlM F st = ((outs.on c).map (Memlet.outKey c)).foldlM G st := by
  rw [← outKeys_on hsh hko c]
  refine List.foldlM_filterMap _ _ _ _ (fun e he st => ⟨fun hn => ?_, fun k hk => hF e he k hk st⟩) st
  exfalso
  simp only [State.outEdgesOn, List.mem_filter, Bool.and_eq_true] at he
  obtain ⟨he, -, hm⟩ := he
  cases hd : e.data with
  | dep => rw [hd] at hm; cases hm
  | memlet m =>
    have := hsh.outs e he ⟨m, hd⟩
    cases hep : s.dstEndpoint e <;> simp_all [State.outKey?, State.Endpoint.isData]

omit he in
/-- The input keys on connector `c`: the first one is the key of the SDFG's
in-edge on `c`, and there is at most one. -/
theorem inKeys_on {n : DNode} {ins : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2) (c : Conn) :
    ((ins.on c).map (tc.inKey c)).head? = (s.inEdgeOn n c).bind s.inKey? ∧ (ins.on c).length ≤ 1 := by
  have h₁ : (ins.on c).map (tc.inKey c) = (s.codeIns n).filter (·.conn = c) := by
    rw [MemletMap.on_eq, List.map_map, hk, List.filter_map]
    refine List.map_congr_left fun p hp => ?_
    simp only [List.mem_filter, decide_eq_true_eq] at hp
    simp [Memlet.inKey, hp.2]
  have hP : ∀ e ∈ s.inEdges n.id, (e.dstConn.getD "" = c && (e.data matches .memlet _)) =
      (s.inKey? e).any (·.conn = c) := by
    intro e he
    cases hd : e.data with
    | dep => simp [State.inKey?, hd]
    | memlet m =>
      have := hsh.ins e he ⟨m, hd⟩
      cases hep : s.srcEndpoint e <;> simp_all [State.inKey?, State.Endpoint.isData]
  refine ⟨?_, ?_⟩
  · rw [h₁, State.codeIns, ← List.filter_filterMap_of _ _ _ _ hP, State.inEdgeOn]
    refine List.head?_filterMap_filter _ _ _ fun e he hp => ?_
    simp only [Bool.and_eq_true] at hp
    cases hd : e.data with
    | dep => rw [hd] at hp; cases hp.2
    | memlet m =>
      have := hsh.ins e he ⟨m, hd⟩
      cases hep : s.srcEndpoint e <;> simp_all [State.inKey?, State.Endpoint.isData]
  · have := congrArg List.length h₁
    rw [List.length_map] at this
    rw [this]
    exact List.length_filter_le_one (fun k : InKey => k.conn) c _ hsh.conns

/-- Popping through an input connector. -/
theorem applyPop_eq {n : DNode} {ins : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2) (st : St) (c : Conn) (k : Nat) :
    applyPop ctx ρ s n st c k = match ins.on c with
      | [] => .err (.malformed s!"pop through unknown connector '{c}'")
      | ms => ms.foldlM (fun st m => let (d, sub) := tc.inData m; popFrom tc.ctx ρ st d sub k) st := by
  obtain ⟨hh, hl⟩ := inKeys_on hsh hk c
  unfold applyPop
  cases hon : ins.on c with
  | nil =>
    rw [hon] at hh
    cases he' : s.inEdgeOn n c with
    | none => rfl
    | some e =>
      exfalso
      rw [he'] at hh
      have hmem : e ∈ s.inEdges n.id := List.mem_of_find?_eq_some he'
      have hp := List.find?_some he'
      simp only [Bool.and_eq_true] at hp
      cases hd : e.data with
      | dep => rw [hd] at hp; cases hp.2
      | memlet m =>
        have := hsh.ins e hmem ⟨m, hd⟩
        cases hep : s.srcEndpoint e <;> simp_all [State.inKey?, State.Endpoint.isData]
  | cons m ms =>
    rw [hon] at hh hl
    rw [List.length_cons] at hl
    have hms : ms = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst hms
    cases he' : s.inEdgeOn n c with
    | none => rw [he'] at hh; simp at hh
    | some e =>
      rw [he'] at hh
      simp only [List.map_cons, List.map_nil, List.head?_cons, Option.bind_some] at hh
      cases hd : e.data with
      | dep => simp [State.inKey?, hd] at hh
      | memlet mm =>
        cases hep : s.srcEndpoint e with
        | data d sub =>
          simp only [State.inKey?, hd, hep, Option.some.injEq, TCtx.inKey, InKey.mk.injEq] at hh
          obtain ⟨-, hd1, hd2, -⟩ := hh
          have hid : tc.inData m = (d, sub) := Prod.ext hd1 hd2
          simp only [hep, List.foldlM_cons, List.foldlM_nil, hid]
          rw [Ctx.leaf_eq he popFrom (fun _ => rfl)]
          cases popFrom ctx ρ st d sub k <;> rfl
        | _ => simp [State.inKey?, hd, hep] at hh

/-- Effects of a code node: the SDFG applies them through its edges, the tree
through its memlets. -/
theorem applyOut_eq {n : DNode} {ins outs : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2)
    (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2) (st : St) (out : TaskletOut) :
    applyOut ctx ρ s n st out = applyOutT tc ρ ins outs st out := by
  have hst : ∀ (G : St → OutKey → Res St) (F : St → DEdge → Res St) (Gt : St → Memlet → Res St) (c : Conn),
      (∀ e ∈ s.outEdgesOn n c, ∀ k, s.outKey? e = some k → ∀ st, F st e = G st k) →
      (∀ m st, Gt st m = G st (Memlet.outKey c m)) →
      ∀ st, (s.outEdgesOn n c).foldlM F st = (outs.on c).foldlM Gt st := by
    intro G F Gt c hF hG st
    rw [outs_fold hsh hko F G hF st, List.foldlM_map]
    exact List.foldlM_congr_mem (fun m _ st => (hG m st).symm) st
  unfold applyOut applyOutT
  congr 1
  · refine List.foldlM_congr_mem (fun ⟨c, k⟩ _ st => ?_) st
    exact applyPop_eq he ρ hsh hk st c k
  · funext st
    congr 1
    · refine List.foldlM_congr_mem (fun ⟨c, v⟩ _ st => ?_) st
      unfold applyScalar
      refine hst (fun st k => storeScalar ctx ρ st k.data k.sub k.wcr k.oob v) _ _ c (fun e _ k hk st => ?_)
        (fun m st => ?_) st
      · cases hd : e.data with
        | dep => simp [State.outKey?, hd] at hk
        | memlet m =>
          cases hep : s.dstEndpoint e with
          | data d sub =>
            simp only [State.outKey?, hd, hep, Option.some.injEq] at hk
            subst hk; simp []
          | _ => simp [State.outKey?, hd, hep] at hk
      · simp only [Memlet.outKey]
        rw [Ctx.leaf_eq he storeScalar (fun _ => rfl)]
    · funext st
      congr 1
      · refine List.foldlM_congr_mem (fun ⟨c, rel, v⟩ _ st => ?_) st
        unfold applyArrayWrite
        refine hst (fun st k => storeArrayWrite ctx ρ st k.data k.sub k.wcr k.oob rel v) _ _ c
          (fun e _ k hk st => ?_) (fun m st => ?_) st
        · cases hd : e.data with
          | dep => simp [State.outKey?, hd] at hk
          | memlet m =>
            cases hep : s.dstEndpoint e with
            | data d sub =>
              simp only [State.outKey?, hd, hep, Option.some.injEq] at hk
              subst hk; simp []
            | _ => simp [State.outKey?, hd, hep] at hk
        · simp only [Memlet.outKey]
          rw [Ctx.leaf_eq he storeArrayWrite (fun _ => rfl)]
      · funext st
        refine List.foldlM_congr_mem (fun ⟨c, vs⟩ _ st => ?_) st
        unfold applyPush
        refine hst (fun st k => pushTo ctx ρ st k.data k.sub vs) _ _ c (fun e _ k hk st => ?_) (fun m st => ?_) st
        · cases hd : e.data with
          | dep => simp [State.outKey?, hd] at hk
          | memlet m =>
            cases hep : s.dstEndpoint e with
            | data d sub =>
              simp only [State.outKey?, hd, hep, Option.some.injEq] at hk
              subst hk; simp []
            | _ => simp [State.outKey?, hd, hep] at hk
        · simp only [Memlet.outKey]
          rw [Ctx.leaf_eq he pushTo (fun _ => rfl)]

/-- **Code nodes**: an SDFG tasklet or library node and the tree node with
the same keys execute the same code. -/
theorem execCode_eq {n : DNode} {ins outs : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2)
    (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2)
    (hdyn : s.hasDynamicWcrOut n = false) (hdynT : outs.hasDynamicWcr = false) (label : String) (st : St)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (err : ErrReason) :
    execCode ctx ρ s n st run err = execCodeT tc ρ label ins outs st run err := by
  unfold execCode execCodeT
  rw [hdyn, hdynT, bindInputs_eq he ρ hsh hk]
  simp only [Bool.false_eq_true, ↓reduceIte]
  cases bindInputsT tc ρ ins st with
  | err e => rfl
  | ok p =>
    simp only [Res.ok_bind]
    cases run p.1 with
    | none => rfl
    | some out => exact applyOut_eq he ρ hsh hk hko p.2 out

omit he in
/-- The keyed memlet edges of a code node: dependency edges have no key,
memlet edges do. -/
theorem filterMap_memlet {α : Type} {l : List DEdge} {k : DEdge → Option α}
    (hd : ∀ e ∈ l, e.data = .dep → k e = none) (hm : ∀ e ∈ l, (∃ m, e.data = .memlet m) → (k e).isSome = true) :
    l.filterMap k = (l.filter fun e => e.data matches .memlet _).filterMap k ∧
      ∀ e ∈ l.filter (fun e => e.data matches .memlet _), (k e).isSome = true := by
  refine ⟨?_, fun e he => ?_⟩
  · induction l with
    | nil => rfl
    | cons e l ih =>
      have ih := ih (fun e' he' => hd e' (List.mem_cons_of_mem _ he')) (fun e' he' => hm e' (List.mem_cons_of_mem _ he'))
      cases h : e.data with
      | dep => simp [h, hd e List.mem_cons_self h, ih]
      | memlet m =>
        simp only [List.filter_cons, h, ↓reduceIte, List.filterMap_cons]
        split <;> simp [ih]
  · obtain ⟨he, hp⟩ := List.mem_filter.1 he
    cases h : e.data with
    | dep => rw [h] at hp; cases hp
    | memlet m => exact hm e he ⟨m, h⟩

omit he in
theorem filterMap_length_of_isSome {α β : Type} {l : List α} {k : α → Option β} (h : ∀ a ∈ l, (k a).isSome = true) :
    (l.filterMap k).length = l.length := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 (h a List.mem_cons_self)
    simp only [List.filterMap_cons, hx, List.length_cons, ih fun a' ha' => h a' (List.mem_cons_of_mem _ ha')]

/-- **Reduce**: the SDFG's single memlet in- and out-edge and the tree's
single input and output memlet. -/
theorem execReduce_eq {n : DNode} {ins outs : MemletMap} (hsh : s.CodeShape n)
    (hk : s.codeIns n = ins.pairs.map fun p => tc.inKey p.1 p.2)
    (hko : s.codeOuts n = outs.pairs.map fun p => Memlet.outKey p.1 p.2) (props : ReduceProps) (st : St) :
    execReduce ctx ρ s n props st = execReduceT tc ρ props ins outs st := by
  have hi := filterMap_memlet (l := s.inEdges n.id) (k := s.inKey?)
    (fun e _ hd => by simp [State.inKey?, hd])
    (fun e he hm => by
      obtain ⟨m, hd⟩ := hm
      have := hsh.ins e he ⟨m, hd⟩
      cases hep : s.srcEndpoint e <;> simp_all [State.inKey?, State.Endpoint.isData])
  have ho := filterMap_memlet (l := s.outEdges n.id) (k := s.outKey?)
    (fun e _ hd => by simp [State.outKey?, hd])
    (fun e he hm => by
      obtain ⟨m, hd⟩ := hm
      have := hsh.outs e he ⟨m, hd⟩
      cases hep : s.dstEndpoint e <;> simp_all [State.outKey?, State.Endpoint.isData])
  have hlI : ins.memlets.length = ((s.inEdges n.id).filter fun e => e.data matches .memlet _).length := by
    rw [MemletMap.memlets_eq, List.length_map, ← List.length_map (f := fun p => tc.inKey p.1 p.2), ← hk,
      State.codeIns, hi.1, filterMap_length_of_isSome hi.2]
  have hlO : outs.memlets.length = ((s.outEdges n.id).filter fun e => e.data matches .memlet _).length := by
    rw [MemletMap.memlets_eq, List.length_map, ← List.length_map (f := fun p => Memlet.outKey p.1 p.2), ← hko,
      State.codeOuts, ho.1, filterMap_length_of_isSome ho.2]
  have hsingle : ∀ {β : Type} (mm : MemletMap) (x : Memlet) (key : Conn → Memlet → β), mm.memlets = [x] →
      ∃ c, mm.pairs.map (fun p => key p.1 p.2) = [key c x] := by
    intro β mm x key h
    rw [MemletMap.memlets_eq] at h
    cases hp : mm.pairs with
    | nil => rw [hp] at h; cases h
    | cons p rest =>
      rw [hp] at h
      simp only [List.map_cons, List.cons.injEq, List.map_eq_nil_iff] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨p.1, rfl⟩
  unfold execReduce execReduceT
  simp only []
  generalize hIx : List.filter _ (s.inEdges n.id) = I
  generalize hOx : List.filter _ (s.outEdges n.id) = O
  have hI2 : ((s.inEdges n.id).filter fun e => e.data matches .memlet _) = I := hIx
  have hO2 : ((s.outEdges n.id).filter fun e => e.data matches .memlet _) = O := hOx
  rw [hI2] at hlI hi
  rw [hO2] at hlO ho
  by_cases hs : (∃ ie', I = [ie']) ∧ ∃ oe', O = [oe']
  · obtain ⟨⟨ie', rfl⟩, ⟨oe', rfl⟩⟩ := hs
    obtain ⟨ie, hI⟩ := List.length_eq_one_iff.1 (by rw [hlI]; rfl)
    obtain ⟨oe, hO⟩ := List.length_eq_one_iff.1 (by rw [hlO]; rfl)
    rw [hI, hO]
    obtain ⟨c, hc⟩ := hsingle ins ie tc.inKey hI
    obtain ⟨c', hc'⟩ := hsingle outs oe Memlet.outKey hO
    have hki : s.inKey? ie' = some (tc.inKey c ie) := by
      have := hi.1
      rw [← State.codeIns, hk, hc] at this
      obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.1 (hi.2 ie' (List.mem_singleton_self _))
      simp only [List.filterMap_cons, hk', List.filterMap_nil] at this
      rw [hk', (List.singleton_inj.1 this).symm]
    have hko' : s.outKey? oe' = some (Memlet.outKey c' oe) := by
      have := ho.1
      rw [← State.codeOuts, hko, hc'] at this
      obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.1 (ho.2 oe' (List.mem_singleton_self _))
      simp only [List.filterMap_cons, hk', List.filterMap_nil] at this
      rw [hk', (List.singleton_inj.1 this).symm]
    unfold State.inKey? at hki
    unfold State.outKey? at hko'
    split at hki
    · rename_i hdi hepi
      split at hko'
      · rename_i om hdo hepo
        simp only [Option.some.injEq, TCtx.inKey, Memlet.outKey, InKey.mk.injEq, OutKey.mk.injEq] at hki hko'
        obtain ⟨-, hd1, hd2, hoi⟩ := hki
        have hid : tc.inData ie = (_, _) := Prod.ext hd1.symm hd2.symm
        obtain ⟨-, rfl, rfl, hw, hoo⟩ := hko'
        simp only [hepi, hepo, hdo, EdgeData.memlet?, hid, hw, hoo, hoi]
        rw [Ctx.leaf_eq he reduceData (fun _ => rfl)]
      · cases hko'
    · cases hki
  · have hno : ¬ ((∃ ie, ins.memlets = [ie]) ∧ ∃ oe, outs.memlets = [oe]) := by
      rintro ⟨⟨ie, hI⟩, ⟨oe, hO⟩⟩
      exact hs ⟨List.length_eq_one_iff.1 (by rw [← hlI, hI]; rfl),
        List.length_eq_one_iff.1 (by rw [← hlO, hO]; rfl)⟩
    split
    · exact absurd ⟨⟨_, rfl⟩, ⟨_, rfl⟩⟩ hs
    · split
      · rename_i h₁ h₂
        exact absurd ⟨⟨_, h₁⟩, ⟨_, h₂⟩⟩ hno
      · rfl

omit he in
/-- **Copies**: the copies into an access or exit node, in edge order. -/
theorem execCopiesInto_eq {n : DNode} (hcs : s.CopyShape ctx n) (st : St) :
    execCopiesInto ctx ρ s n st =
      (s.copiesInto ctx n).foldlM (fun st k => copyData ctx ρ st k.src k.ssub k.dst k.dsub k.wcr k.oob) st := by
  unfold execCopiesInto State.copiesInto
  refine List.foldlM_filterMap _ _ _ _ (fun e he' st => ⟨fun hn => ?_, fun k hk => ?_⟩) st
  · cases hd : e.data with
    | dep => rfl
    | memlet m =>
      cases hv : (ctx.isViewDef s e || s.isRouting e) with
      | true => simp []
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte, hcs.noRef e he' m hd hv]
        cases hep : s.srcEndpoint e with
        | data sd ssub =>
          exfalso
          have hc : s.isCopyEdge ctx e = true := by simp [State.isCopyEdge, hd, hv, hep, State.Endpoint.isData]
          have hdst := hcs.dst e he' hc
          unfold State.copyKey? at hn
          rw [hc] at hn
          cases hdp : s.dstEndpoint e <;> simp_all [State.Endpoint.isData]
        | _ => rfl
  · unfold State.copyKey? at hk
    split at hk
    · rename_i hc
      split at hk
      · rename_i m sd ssub dd dsub hd hep hdp
        simp only [Option.some.injEq] at hk
        subst hk
        simp only [State.isCopyEdge, hd, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hc
        simp only [hd, hc.1, Bool.false_eq_true, ↓reduceIte, hcs.noRef e he' m hd hc.1, hep, execCopy, hdp]
      · cases hk
    · cases hk

/-- A tree copy node. -/
theorem execCopyT_eq (target : Name) (m : Memlet) (st : St) :
    execCopyT tc ρ target m st =
      copyData ctx ρ st (tc.copyKey target m).src (tc.copyKey target m).ssub (tc.copyKey target m).dst
        (tc.copyKey target m).dsub (tc.copyKey target m).wcr (tc.copyKey target m).oob := by
  simp only [execCopyT, TCtx.copyKey]
  rcases hid : tc.inData m with ⟨d, sub⟩
  simp only
  rw [Ctx.leaf_eq he copyData (fun _ => rfl)]

omit he in
/-- **Map set-up** without dynamic inputs. -/
theorem mapSetup_eq {n : DNode} {mp : MapNode} (hdyn : ∀ e ∈ s.inEdges n.id, s.isDynamicInput e = false)
    (st : St) : mapSetup ctx ρ s n mp st = (ctx.rangePoints ρ st.mem mp.range).map fun pts => (ρ, st, pts) := by
  unfold mapSetup bindDynamicInputs
  rw [List.foldlM_fixed (fun e he' => by simp [hdyn e he'])]
  simp only [Res.ok_bind]
  cases ctx.rangePoints ρ st.mem mp.range <;> rfl

omit ρ in
/-- **Map set-up** in the tree, without pending dynamic inputs. -/
theorem mapSetupT_eq {mp : MapNode} (cst : CSt) :
    mapSetupT tc {} mp cst = (ctx.rangePoints cst.ρ cst.mem mp.range).map fun pts => (cst.ρ, pts) := by
  have hρ : cst.ρ.setAll [] = cst.ρ := rfl
  simp only [mapSetupT, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, hρ]
  rw [Ctx.leaf_eq he Ctx.rangePoints (fun _ => rfl)]
  cases ctx.rangePoints cst.ρ cst.mem mp.range <;> rfl

/-! ### Dynamic map inputs

The SDFG binds the dynamic inputs of a map one by one, reading each under
the valuation extended by the previous ones (`bindDynamicInputs`); the tree
reads them under the scope's valuation into pending symbols and binds them
all at the map (`mapSetupT`).  The two agree when the connectors are
distinct and neither a descriptor nor an input's subset mentions a connector
(`DynOk`); both then equal `dynReadAcc` followed by `Valuation.setAll`. -/

end Leaf

/-- Dynamic inputs that may all be read before any is bound. -/
def DynOk (ctx : Ctx) (keys : List InKey) : Prop :=
  (keys.map (·.conn)).Nodup ∧ ∀ k ∈ keys, k.conn ≠ "IN_stream" ∧ k.conn ∉ ctx.g.descSyms ∧
    ∀ k' ∈ keys, k.conn ∉ (k'.sub.map Subset.freeSyms).getD []

def dynOk? (ctx : Ctx) (keys : List InKey) : Bool :=
  (keys.map (·.conn)).Nodup && keys.all fun k => k.conn != "IN_stream" && !ctx.g.descSyms.contains k.conn &&
    keys.all fun k' => !((k'.sub.map Subset.freeSyms).getD []).contains k.conn

theorem dynOk?_sound {ctx : Ctx} {keys : List InKey} (h : dynOk? ctx keys = true) : DynOk ctx keys := by
  simp only [dynOk?, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, bne_iff_ne, ne_eq,
    Bool.not_eq_eq_eq_not, Bool.not_true, List.contains_eq_mem, decide_eq_false_iff_not] at h
  exact ⟨h.1, fun k hk => ⟨(h.2 k hk).1.1, (h.2 k hk).1.2, (h.2 k hk).2⟩⟩

/-- Read the dynamic inputs `ks` under the fixed valuation `ρ`, appending
the defined values to `acc`. -/
def dynReadAcc (ctx : Ctx) (ρ : Valuation) : List (Sym × Val) → List InKey → St → Res (List (Sym × Val) × St)
  | acc, [], st => .ok (acc, st)
  | acc, k :: ks, st =>
    readScalarData ctx ρ k.data k.sub k.oob st >>= fun (v?, st) =>
      dynReadAcc ctx ρ (acc ++ (v?.map fun v => (k.conn, v)).toList) ks st

theorem dynReadAcc_shift (ctx : Ctx) (ρ : Valuation) (L : Log) :
    ∀ (acc : List (Sym × Val)) (ks : List InKey) (st : St),
      dynReadAcc ctx ρ acc ks (st.shift L) = (dynReadAcc ctx ρ acc ks st).map (St.shiftR L)
  | acc, [], st => rfl
  | acc, k :: ks, st => by
    simp only [dynReadAcc]
    rw [readScalarData_shift]
    cases readScalarData ctx ρ k.data k.sub k.oob st with
    | err e => rfl
    | ok p =>
      obtain ⟨v?, st'⟩ := p
      exact dynReadAcc_shift ctx ρ L _ ks st'

theorem setAll_snoc (ρ : Valuation) (acc : List (Sym × Val)) (c : Sym) (v : Val) (hc : c ∉ acc.map Prod.fst) :
    Valuation.set (Valuation.setAll ρ acc) c v = Valuation.setAll ρ (acc ++ [(c, v)]) := by
  funext y
  simp only [Valuation.set, Valuation.setAll, List.find?_append]
  by_cases hy : y = c
  · subst hy
    have : acc.find? (fun q => decide (q.1 = y)) = none := by
      rw [List.find?_eq_none]
      intro q hq hqy
      exact hc (List.mem_map.2 ⟨q, hq, by simpa using hqy⟩)
    simp [this]
  · cases h : acc.find? (fun q => decide (q.1 = y)) <;> simp [hy, Ne.symm hy]

theorem foldlM_filter {α β : Type} (f : β → α → Res β) (p : α → Bool) (hp : ∀ b x, p x = false → f b x = .ok b) :
    ∀ (l : List α) (b : β), l.foldlM f b = (l.filter p).foldlM f b
  | [], _ => rfl
  | x :: xs, b => by
    rw [List.foldlM_cons]
    cases hx : p x with
    | false => rw [hp b x hx, List.filter_cons_of_neg (by simp [hx])]; exact foldlM_filter f p hp xs b
    | true =>
      rw [List.filter_cons_of_pos hx, List.foldlM_cons]
      cases f b x with
      | err e => rfl
      | ok b' => exact foldlM_filter f p hp xs b'

/-- **Dynamic inputs** of the SDFG: under `DynOk`, binding them one by one is
reading them under the scope's valuation and binding them at once. -/
theorem bindDynamicInputs_eq {ctx : Ctx} {ρ : Valuation} {s : State} {n : DNode} {keys : List InKey}
    (hkeys : s.dynKeys? n = some keys) (hok : DynOk ctx keys) (st : St) :
    bindDynamicInputs ctx ρ s n st = (dynReadAcc ctx ρ [] keys st).map fun p => (Valuation.setAll ρ p.1, p.2) := by
  unfold bindDynamicInputs
  rw [foldlM_filter _ s.isDynamicInput (fun b x hx => by simp [hx])]
  unfold State.dynKeys? at hkeys
  rw [List.mapM_option_some_iff] at hkeys
  -- invariant: the accumulated symbols are connectors of earlier keys
  suffices H : ∀ (E : List DEdge) (ks : List InKey), List.Forall₂ (fun e k => (match e.data, e.dstConn, s.srcEndpoint e with
      | .memlet _, some c, .data d sub => some (⟨c, d, sub, e.allowOob⟩ : InKey)
      | _, _, _ => none) = some k) E ks → (∀ e ∈ E, s.isDynamicInput e = true) → ks ⊆ keys →
      ((ks.map (·.conn)).Nodup) →
      ∀ (acc : List (Sym × Val)) (st : St), (∀ x ∈ acc.map Prod.fst, x ∈ keys.map (·.conn) ∧ x ∉ ks.map (·.conn)) →
        E.foldlM (fun (p : Valuation × St) e =>
          if s.isDynamicInput e then
            match e.data, e.dstConn with
            | .memlet _, some c => do
              let (v?, st) ← readEdgeScalar ctx p.1 s e p.2
              match v? with
              | some v => pure (Valuation.set p.1 c v, st)
              | none => pure (p.1, st)
            | _, _ => pure (p.1, p.2)
          else pure (p.1, p.2)) (Valuation.setAll ρ acc, st) =
        (dynReadAcc ctx ρ acc ks st).map fun p => (Valuation.setAll ρ p.1, p.2) by
    have := H _ _ hkeys (fun e he => (List.mem_filter.1 he).2) (fun _ h => h) hok.1 [] st (by simp)
    exact this
  intro E
  induction E with
  | nil => intro ks hks; cases hks; intros; rfl
  | cons e E ih =>
    intro ks hks hdynE hsub hnd acc st hacc
    cases hks with
    | cons hk hks' =>
      rename_i k ks'
      have hde := hdynE e List.mem_cons_self
      rw [List.foldlM_cons]
      simp only [hde, ↓reduceIte]
      split at hk
      · rename_i m c d sub hdata hconn hsrc
        simp only [Option.some.injEq] at hk
        subst hk
        simp only [hdata, hconn]
        have hkmem : (⟨c, d, sub, e.allowOob⟩ : InKey) ∈ keys := hsub List.mem_cons_self
        -- the read does not see the accumulated bindings
        have hagree : ∀ x, x ∉ acc.map Prod.fst → Valuation.setAll ρ acc x = ρ x := fun x hx => Valuation.setAll_of_not_mem hx
        have hread : readEdgeScalar ctx (Valuation.setAll ρ acc) s e st = readScalarData ctx ρ d sub e.allowOob st := by
          unfold readEdgeScalar
          rw [hsrc]
          apply readScalarData_congr
          · intro x hx
            apply hagree
            intro hxa
            obtain ⟨k₀, hk₀, rfl⟩ := List.mem_map.1 (hacc _ hxa).1
            exact (hok.2 k₀ hk₀).2.1 hx
          · intro x hx
            apply hagree
            intro hxa
            obtain ⟨k₀, hk₀, rfl⟩ := List.mem_map.1 (hacc _ hxa).1
            exact (hok.2 k₀ hk₀).2.2 _ hkmem hx
        rw [hread]
        simp only [dynReadAcc]
        cases hrd : readScalarData ctx ρ d sub e.allowOob st with
        | err er => rfl
        | ok p =>
          obtain ⟨v?, st'⟩ := p
          have hnd' : (ks'.map (·.conn)).Nodup := (List.nodup_cons.1 hnd).2
          have hcnot : c ∉ ks'.map (·.conn) := (List.nodup_cons.1 hnd).1
          have hcacc : c ∉ acc.map Prod.fst := fun h => (hacc c h).2 List.mem_cons_self
          cases v? with
          | some v =>
            show List.foldlM _ (Valuation.set (Valuation.setAll ρ acc) c v, st') E = _
            rw [setAll_snoc ρ acc c v hcacc]
            refine ih ks' hks' (fun e' he' => hdynE e' (List.mem_cons_of_mem _ he'))
              (fun k' hk' => hsub (List.mem_cons_of_mem _ hk')) hnd' _ st' ?_
            intro x hx
            rw [List.map_append, List.mem_append] at hx
            rcases hx with hx | hx
            · exact ⟨(hacc x hx).1, fun h => (hacc x hx).2 (List.mem_cons_of_mem _ h)⟩
            · simp only [List.map_cons, List.map_nil, List.mem_singleton] at hx
              subst hx
              exact ⟨List.mem_map.2 ⟨_, hkmem, rfl⟩, hcnot⟩
          | none =>
            show List.foldlM _ (Valuation.setAll ρ acc, st') E = (dynReadAcc ctx ρ (acc ++ []) ks' st').map _
            rw [List.append_nil]
            refine ih ks' hks' (fun e' he' => hdynE e' (List.mem_cons_of_mem _ he'))
              (fun k' hk' => hsub (List.mem_cons_of_mem _ hk')) hnd' _ st' ?_
            intro x hx
            exact ⟨(hacc x hx).1, fun h => (hacc x hx).2 (List.mem_cons_of_mem _ h)⟩
      · cases hk

end Corr

end Dace
