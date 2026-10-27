-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Sim

/-!
# Relabeling the nodes of a state

Node ids carry no meaning of their own: renumbering the nodes of a state by a
permutation `φ` of the ids (`State.relabel`: node ids, `scopeEntry`
references and edge endpoints, the lists keeping their order) does not change
what the state does.  This is the first rule of the SDFG–SDFG checker
(`Dace.Equiv.SdfgCorr`): DaCe renumbers the nodes of a state whenever a
transformation adds or removes one (node ids are positions in the node list).

## What depends on node ids

* The derived structure of a state (`Dace.SDFG.Scope`, the memlet-tree
  walks, endpoints, views, components, happens-before) is *equivariant*: a
  function of the relabeled state at relabeled arguments is the relabeled
  result (`State.node?_relabel`, `State.srcEndpoint_relabel`,
  `State.components_relabel`, …).  The fuel of the bounded walks is the
  number of nodes or edges, which relabeling keeps.
* Code→code values are keyed by their producing node (`St.vals`):
  `St.relabel` renames the keys.  Functions on memory and logs never look at
  `vals` (`St.mapVals`).
* The race witness of a happens-before race names the two nodes; races are
  undefined behavior, so the outcomes are related by `ResSim`, which lets a
  run end in any undefined behavior the other run has.
* The memory names of a nested SDFG's containers contain the id of its node
  (`Ctx.enter`): v1 requires `φ` to fix the ids of nested SDFG nodes
  (`State.FixesNested`).

## Main results

* `NodeExec.relabel`, `SeqExec.relabel`, `OrderedExec.relabel`,
  `IterExec.relabel`, `ConsumeTasks.relabel`, `CompsExec.relabel`,
  `StateExec.relabel`: a run of the state is matched by a run of the relabeled
  state with relabeled arguments and the relabeled (or an undefined) outcome.
* `StateExec.relabel_iff`-style use in both directions through `φ.symm`
  (`State.relabel_symm`).
-/

set_option autoImplicit false

namespace Dace

/-! ## Closures under an injective map -/

theorem appendNew_map {α β : Type} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] (f : α → β)
    (hf : Function.Injective f) : ∀ (acc ys : List α), appendNew (acc.map f) (ys.map f) = (appendNew acc ys).map f
  | acc, [] => rfl
  | acc, y :: ys => by
    simp only [appendNew, List.map_cons, List.foldl_cons]
    have hc : (acc.map f).contains (f y) = acc.contains y := by
      simp [hf.eq_iff]
    have step : (if (acc.map f).contains (f y) then acc.map f else acc.map f ++ [f y]) =
        (if acc.contains y then acc else acc ++ [y]).map f := by
      rw [hc]; split <;> simp
    rw [step]
    exact appendNew_map f hf _ ys

theorem closureFrom_map {α β : Type} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] (f : α → β)
    (hf : Function.Injective f) {step : α → List α} {step' : β → List β} (hs : ∀ x, step' (f x) = (step x).map f) :
    ∀ (fuel : Nat) (seen : List α) (i : Nat),
      closureFrom step' fuel (seen.map f) i = (closureFrom step fuel seen i).map f
  | 0, seen, i => rfl
  | fuel + 1, seen, i => by
    simp only [closureFrom, List.getElem?_map]
    cases seen[i]? with
    | none => rfl
    | some x =>
      simp only [Option.map_some]
      rw [hs, appendNew_map f hf]
      exact closureFrom_map f hf hs fuel _ _

theorem closure_map {α β : Type} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] (f : α → β)
    (hf : Function.Injective f) {step : α → List α} {step' : β → List β} (hs : ∀ x, step' (f x) = (step x).map f)
    (bound : Nat) (x : α) : closure step' bound (f x) = (closure step bound x).map f :=
  closureFrom_map f hf hs _ [x] 0

/-- `foldlM` in `Res` with two step functions related by a map `sh` of the
accumulator. -/
theorem List.foldlM_res_comm₂ {α β : Type} (sh : β → β) {f' f : β → α → Res β}
    (hf : ∀ b a, f' (sh b) a = (f b a).map sh) (xs : List α) (b : β) :
    xs.foldlM f' (sh b) = (xs.foldlM f b).map sh := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    rw [hf]
    cases f b x with
    | ok b' => exact ih b'
    | err e => rfl

/-- A `foldlM` followed by a continuation, both commuting with maps of the
accumulator and the result. -/
theorem List.bind_foldlM_res_comm {α β γ : Type} (sh : β → β) (sh₂ : γ → γ) {f' f : β → α → Res β}
    {k' k : β → Res γ} (hf : ∀ b a, f' (sh b) a = (f b a).map sh) (hk : ∀ b, k' (sh b) = (k b).map sh₂)
    (xs : List α) (b : β) : (xs.foldlM f' (sh b) >>= k') = (xs.foldlM f b >>= k).map sh₂ := by
  rw [List.foldlM_res_comm₂ sh hf]
  exact Res.bind_shift _ hk

/-! ## Relabeling syntax -/

variable (φ : Equiv.Perm NodeId)

/-- Relabel the endpoints of an edge. -/
def DEdge.relabel (e : DEdge) : DEdge := { e with src := φ e.src, dst := φ e.dst }

/-- Relabel a node: its id and its scope entry. -/
def DNode.relabel (n : DNode) : DNode := { n with id := φ n.id, scopeEntry := n.scopeEntry.map φ }

/-- Relabel the nodes of a state (lists keep their order). -/
def State.relabel (s : State) : State :=
  { s with nodes := s.nodes.map (DNode.relabel φ), edges := s.edges.map (DEdge.relabel φ) }

/-- Relabel the producer of a code value. -/
def State.Endpoint.relabel : State.Endpoint → State.Endpoint
  | .code (id, c) => .code (φ id, c)
  | e => e

/-- Replace the code values of a sequential state. -/
def St.mapVals (f : List ((NodeId × Conn) × Val) → List ((NodeId × Conn) × Val)) (st : St) : St :=
  { st with vals := f st.vals }

/-- Relabel the keys of the code values. -/
def St.relabelVals (vs : List ((NodeId × Conn) × Val)) : List ((NodeId × Conn) × Val) :=
  vs.map fun x => ((φ x.1.1, x.1.2), x.2)

/-- Relabel the producers of the code values. -/
def St.relabel (st : St) : St := st.mapVals (St.relabelVals φ)

namespace DEdge
variable {φ}
@[simp] theorem relabel_src (e : DEdge) : (e.relabel φ).src = φ e.src := rfl
@[simp] theorem relabel_dst (e : DEdge) : (e.relabel φ).dst = φ e.dst := rfl
@[simp] theorem relabel_srcConn (e : DEdge) : (e.relabel φ).srcConn = e.srcConn := rfl
@[simp] theorem relabel_dstConn (e : DEdge) : (e.relabel φ).dstConn = e.dstConn := rfl
@[simp] theorem relabel_data (e : DEdge) : (e.relabel φ).data = e.data := rfl
@[simp] theorem relabel_allowOob (e : DEdge) : (e.relabel φ).allowOob = e.allowOob := rfl

theorem relabel_injective : Function.Injective (DEdge.relabel φ) := by
  intro e f h
  cases e; cases f
  simp only [DEdge.relabel, DEdge.mk.injEq] at h ⊢
  obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := h
  exact ⟨φ.injective h₁, h₂, φ.injective h₃, h₄, h₅⟩

theorem relabel_symm (e : DEdge) : (e.relabel φ).relabel φ.symm = e := by
  cases e; simp [DEdge.relabel]

end DEdge

namespace DNode
variable {φ}
@[simp] theorem relabel_id (n : DNode) : (n.relabel φ).id = φ n.id := rfl
@[simp] theorem relabel_kind (n : DNode) : (n.relabel φ).kind = n.kind := rfl
@[simp] theorem relabel_label (n : DNode) : (n.relabel φ).label = n.label := rfl
@[simp] theorem relabel_inConns (n : DNode) : (n.relabel φ).inConns = n.inConns := rfl
@[simp] theorem relabel_outConns (n : DNode) : (n.relabel φ).outConns = n.outConns := rfl
@[simp] theorem relabel_scopeEntry (n : DNode) : (n.relabel φ).scopeEntry = n.scopeEntry.map φ := rfl
@[simp] theorem relabel_isEntry (n : DNode) : (n.relabel φ).isEntry = n.isEntry := rfl
@[simp] theorem relabel_isExit (n : DNode) : (n.relabel φ).isExit = n.isExit := rfl
@[simp] theorem relabel_isAccess (n : DNode) : (n.relabel φ).isAccess = n.isAccess := rfl
@[simp] theorem relabel_isCodeNode (n : DNode) : (n.relabel φ).isCodeNode = n.isCodeNode := rfl
@[simp] theorem relabel_isSimple (n : DNode) : (n.relabel φ).isSimple = n.isSimple := rfl
@[simp] theorem relabel_accessData? (n : DNode) : (n.relabel φ).accessData? = n.accessData? := rfl
@[simp] theorem relabel_mentions (n : DNode) (t : Name) : (n.relabel φ).mentions t = n.mentions t := rfl

theorem relabel_symm (n : DNode) : (n.relabel φ).relabel φ.symm = n := by
  rcases n with ⟨id, label, ic, oc, se, kind⟩
  cases se <;> simp [DNode.relabel]

end DNode

namespace State.Endpoint
variable {φ}
@[simp] theorem relabel_data (n : Name) (sub : Option Subset) : (State.Endpoint.data n sub).relabel φ = .data n sub := rfl
@[simp] theorem relabel_invalid : State.Endpoint.invalid.relabel φ = .invalid := rfl
@[simp] theorem relabel_code (k : NodeId × Conn) : (State.Endpoint.code k).relabel φ = .code (φ k.1, k.2) := rfl
@[simp] theorem relabel_isData (e : State.Endpoint) : (e.relabel φ).isData = e.isData := by
  cases e <;> rfl
end State.Endpoint

namespace State
variable {φ}

@[simp] theorem relabel_nodes (s : State) : (s.relabel φ).nodes = s.nodes.map (DNode.relabel φ) := rfl
@[simp] theorem relabel_edges (s : State) : (s.relabel φ).edges = s.edges.map (DEdge.relabel φ) := rfl

theorem relabel_symm (s : State) : (s.relabel φ).relabel φ.symm = s := by
  rcases s with ⟨label, nodes, edges, nosync⟩
  simp only [State.relabel, List.map_map, State.mk.injEq, true_and, and_true]
  constructor
  · conv => rhs; rw [← List.map_id nodes]
    exact List.map_congr_left fun n _ => DNode.relabel_symm n
  · conv => rhs; rw [← List.map_id edges]
    exact List.map_congr_left fun e _ => DEdge.relabel_symm e

theorem node?_relabel (s : State) (id : NodeId) :
    (s.relabel φ).node? (φ id) = (s.node? id).map (DNode.relabel φ) := by
  simp only [node?, relabel_nodes, List.find?_map]
  congr 1
  apply List.find?_congr
  intro n _
  simp [Function.comp, φ.injective.eq_iff]

theorem inEdges_relabel (s : State) (id : NodeId) :
    (s.relabel φ).inEdges (φ id) = (s.inEdges id).map (DEdge.relabel φ) := by
  simp only [inEdges, relabel_edges, List.filter_map]
  congr 1
  apply List.filter_congr
  intro e _
  simp [Function.comp, φ.injective.eq_iff]

theorem outEdges_relabel (s : State) (id : NodeId) :
    (s.relabel φ).outEdges (φ id) = (s.outEdges id).map (DEdge.relabel φ) := by
  simp only [outEdges, relabel_edges, List.filter_map]
  congr 1
  apply List.filter_congr
  intro e _
  simp [Function.comp, φ.injective.eq_iff]

theorem isScopeNode_relabel (s : State) (id : NodeId) : (s.relabel φ).isScopeNode (φ id) = s.isScopeNode id := by
  simp only [isScopeNode, node?_relabel]
  cases s.node? id <;> rfl

theorem isEntryNode_relabel (s : State) (id : NodeId) : (s.relabel φ).isEntryNode (φ id) = s.isEntryNode id := by
  simp only [isEntryNode, node?_relabel]
  cases s.node? id <;> rfl

theorem isExitNode_relabel (s : State) (id : NodeId) : (s.relabel φ).isExitNode (φ id) = s.isExitNode id := by
  simp only [isExitNode, node?_relabel]
  cases s.node? id <;> rfl

theorem isAccessNode_relabel (s : State) (id : NodeId) : (s.relabel φ).isAccessNode (φ id) = s.isAccessNode id := by
  simp only [isAccessNode, node?_relabel]
  cases s.node? id <;> rfl

theorem isConsumeEntry_relabel (s : State) (id : NodeId) :
    (s.relabel φ).isConsumeEntry (φ id) = s.isConsumeEntry id := by
  simp only [isConsumeEntry, node?_relabel]
  cases s.node? id <;> rfl

theorem prevEdges_relabel (s : State) (e : DEdge) :
    (s.relabel φ).prevEdges (e.relabel φ) = (s.prevEdges e).map (DEdge.relabel φ) := by
  simp only [prevEdges, DEdge.relabel_src, isScopeNode_relabel, DEdge.relabel_srcConn]
  split
  · split
    · rw [inEdges_relabel, List.filter_map]
      rfl
    · rfl
  · rfl

theorem nextEdges_relabel (s : State) (e : DEdge) :
    (s.relabel φ).nextEdges (e.relabel φ) = (s.nextEdges e).map (DEdge.relabel φ) := by
  simp only [nextEdges, DEdge.relabel_dst, isScopeNode_relabel, DEdge.relabel_dstConn]
  split
  · split
    · rw [outEdges_relabel, List.filter_map]
      rfl
    · rfl
  · rfl

theorem isRouting_relabel (s : State) (e : DEdge) : (s.relabel φ).isRouting (e.relabel φ) = s.isRouting e := by
  simp [isRouting, isEntryNode_relabel, isExitNode_relabel]

theorem isLeaf_relabel (s : State) (e : DEdge) : (s.relabel φ).isLeaf (e.relabel φ) = s.isLeaf e := by
  simp [isLeaf, isRouting_relabel]

theorem isDynamicInput_relabel (s : State) (e : DEdge) :
    (s.relabel φ).isDynamicInput (e.relabel φ) = s.isDynamicInput e := by
  simp [isDynamicInput, isEntryNode_relabel]

theorem walkPrev_relabel (s : State) : ∀ (k : Nat) (e : DEdge),
    (s.relabel φ).walkPrev k (e.relabel φ) = (s.walkPrev k e).relabel φ
  | 0, e => rfl
  | k + 1, e => by
    simp only [walkPrev, DEdge.relabel_src, isScopeNode_relabel, DEdge.relabel_srcConn, prevEdges_relabel]
    split
    · cases s.prevEdges e with
      | nil => rfl
      | cons f _ => exact walkPrev_relabel s k f
    · rfl

theorem walkNext_relabel (s : State) (oe : Bool) : ∀ (k : Nat) (e : DEdge),
    (s.relabel φ).walkNext oe k (e.relabel φ) = (s.walkNext oe k e).relabel φ
  | 0, e => rfl
  | k + 1, e => by
    simp only [walkNext, DEdge.relabel_dst, DEdge.relabel_dstConn, nextEdges_relabel, isScopeNode_relabel,
      isExitNode_relabel]
    by_cases hb : ((if oe = true then s.isExitNode e.dst else s.isScopeNode e.dst) && connPrefix e.dstConn "IN_") = true
    · simp only [hb, ↓reduceIte]
      cases s.nextEdges e with
      | nil => rfl
      | cons f _ => exact walkNext_relabel s oe k f
    · simp only [hb, ↓reduceIte, Bool.false_eq_true]

theorem pathStart_relabel (s : State) (e : DEdge) : (s.relabel φ).pathStart (e.relabel φ) = (s.pathStart e).relabel φ := by
  simp only [pathStart, relabel_edges, List.length_map]
  exact walkPrev_relabel s _ e

theorem pathEnd_relabel (s : State) (e : DEdge) : (s.relabel φ).pathEnd (e.relabel φ) = (s.pathEnd e).relabel φ := by
  simp only [pathEnd, relabel_edges, List.length_map]
  exact walkNext_relabel s _ _ e

theorem dstRootEdge_relabel (s : State) (e : DEdge) :
    (s.relabel φ).dstRootEdge (e.relabel φ) = (s.dstRootEdge e).relabel φ := by
  simp only [dstRootEdge, relabel_edges, List.length_map]
  exact walkNext_relabel s _ _ e

theorem treeRootFuel_relabel (s : State) : ∀ (k : Nat) (e : DEdge),
    (s.relabel φ).treeRootFuel k (e.relabel φ) = (s.treeRootFuel k e).relabel φ
  | 0, e => rfl
  | k + 1, e => by
    simp only [treeRootFuel, DEdge.relabel_src, DEdge.relabel_dst, isEntryNode_relabel, isExitNode_relabel,
      DEdge.relabel_srcConn, DEdge.relabel_dstConn, prevEdges_relabel, nextEdges_relabel]
    split
    · cases s.prevEdges e with
      | nil => rfl
      | cons f _ => exact treeRootFuel_relabel s k f
    · split
      · cases s.nextEdges e with
        | nil => rfl
        | cons f _ => exact treeRootFuel_relabel s k f
      · rfl

theorem treeRoot_relabel (s : State) (e : DEdge) : (s.relabel φ).treeRoot (e.relabel φ) = (s.treeRoot e).relabel φ := by
  simp only [treeRoot, relabel_edges, List.length_map]
  exact treeRootFuel_relabel s _ e

theorem treeEdges_relabel (s : State) (e : DEdge) :
    (s.relabel φ).treeEdges (e.relabel φ) = (s.treeEdges e).map (DEdge.relabel φ) := by
  simp only [treeEdges, relabel_edges, List.length_map]
  exact closure_map _ DEdge.relabel_injective (fun f => by rw [prevEdges_relabel, nextEdges_relabel, List.map_append])
    _ e

theorem treeLeaves_relabel (s : State) (e : DEdge) :
    (s.relabel φ).treeLeaves (e.relabel φ) = (s.treeLeaves e).map (DEdge.relabel φ) := by
  simp only [treeLeaves, treeEdges_relabel, List.filter_map]
  congr 1
  apply List.filter_congr
  intro f _
  simp [Function.comp, isLeaf_relabel]

theorem chunkSource_relabel (s : State) : ∀ (k : Nat) (e : DEdge),
    (s.relabel φ).chunkSource k (e.relabel φ) = (s.chunkSource k e).map φ
  | 0, e => rfl
  | k + 1, e => by
    show (if ((s.relabel φ).isScopeNode (φ e.src) && connPrefix e.srcConn "OUT_") = true then
        if ((s.relabel φ).isConsumeEntry (φ e.src) && decide (e.srcConn = some "OUT_stream")) = true then
          some (φ e.src)
        else
          match (s.relabel φ).prevEdges (e.relabel φ) with
          | f :: _ => (s.relabel φ).chunkSource k f
          | [] => none
      else none) = _
    simp only [chunkSource, isScopeNode_relabel, isConsumeEntry_relabel, prevEdges_relabel]
    by_cases h₁ : (s.isScopeNode e.src && connPrefix e.srcConn "OUT_") = true
    · by_cases h₂ : (s.isConsumeEntry e.src && decide (e.srcConn = some "OUT_stream")) = true
      · simp [h₁, h₂]
      · cases hp : s.prevEdges e with
        | nil => simp [h₁, h₂]
        | cons f _ => simp [h₁, h₂, chunkSource_relabel s k f]
    · simp [h₁]

theorem consumedStream?_relabel (s : State) (entry : NodeId) :
    (s.relabel φ).consumedStream? (φ entry) = s.consumedStream? entry := by
  simp only [consumedStream?, inEdges_relabel, List.find?_map]
  have : ((fun e : DEdge => decide (e.dstConn = some "IN_stream")) ∘ DEdge.relabel φ) =
      fun e => decide (e.dstConn = some "IN_stream") := rfl
  rw [this]
  cases (s.inEdges entry).find? (fun e => decide (e.dstConn = some "IN_stream")) with
  | none => rfl
  | some e =>
    simp only [Option.map_some, pathStart_relabel, DEdge.relabel_src, node?_relabel]
    cases s.node? (s.pathStart e).src <;> rfl

theorem children_relabel (s : State) (entry : Option NodeId) :
    (s.relabel φ).children (entry.map φ) = (s.children entry).map (DNode.relabel φ) := by
  simp only [children, relabel_nodes, List.filter_map]
  congr 1
  apply List.filter_congr
  intro n _
  cases entry <;> cases h : n.scopeEntry <;> simp [Function.comp, h, φ.injective.eq_iff]

theorem directChildren_relabel (s : State) (entry : Option NodeId) :
    (s.relabel φ).directChildren (entry.map φ) = (s.directChildren entry).map φ := by
  simp [directChildren, children_relabel, List.map_map, Function.comp]

theorem directChildren_some_relabel (s : State) (entry : NodeId) :
    (s.relabel φ).directChildren (some (φ entry)) = (s.directChildren (some entry)).map φ :=
  directChildren_relabel s (some entry)

theorem scopeNodesFuel_relabel (s : State) : ∀ (k : Nat) (entry : NodeId),
    (s.relabel φ).scopeNodesFuel k (φ entry) = (s.scopeNodesFuel k entry).map φ
  | 0, _ => rfl
  | k + 1, entry => by
    simp only [scopeNodesFuel]
    have := children_relabel (φ := φ) s (some entry)
    simp only [Option.map_some] at this
    rw [this, List.flatMap_map, List.map_flatMap]
    apply List.flatMap_congr
    intro n _
    by_cases h : n.isEntry = true <;> simp [h, scopeNodesFuel_relabel s k n.id]

theorem scopeNodes_relabel (s : State) (entry : NodeId) :
    (s.relabel φ).scopeNodes (φ entry) = (s.scopeNodes entry).map φ := by
  simp only [scopeNodes, relabel_nodes, List.length_map]
  exact scopeNodesFuel_relabel s _ entry

theorem topLevelOf_relabel (s : State) (ids : List NodeId) :
    (s.relabel φ).topLevelOf (ids.map φ) = (s.topLevelOf ids).map φ := by
  simp only [topLevelOf, List.filter_map]
  congr 1
  apply List.filter_congr
  intro id _
  simp only [Function.comp, node?_relabel]
  cases s.node? id with
  | none => rfl
  | some n => cases h : n.scopeEntry <;> simp [h]

theorem neighbors_relabel (s : State) (id : NodeId) :
    (s.relabel φ).neighbors (φ id) = (s.neighbors id).map φ := by
  simp only [neighbors, relabel_edges, List.filterMap_map, List.map_filterMap]
  congr 1
  funext e
  simp only [Function.comp, DEdge.relabel_src, DEdge.relabel_dst, φ.injective.eq_iff]
  split <;> (try split) <;> simp_all

theorem component_relabel (s : State) (id : NodeId) :
    (s.relabel φ).component (φ id) = (s.component id).map φ := by
  simp only [component, relabel_edges, List.length_map]
  exact closure_map φ φ.injective (neighbors_relabel s) _ id

theorem contains_map_perm (l : List NodeId) (id : NodeId) : (l.map φ).contains (φ id) = l.contains id := by
  rw [Bool.eq_iff_iff]
  simp [φ.injective.eq_iff]

theorem componentsFrom_relabel (s : State) : ∀ (seen : List NodeId) (ns : List DNode),
    (s.relabel φ).componentsFrom (seen.map φ) (ns.map (DNode.relabel φ)) =
      (s.componentsFrom seen ns).map (List.map φ)
  | _, [] => rfl
  | seen, n :: rest => by
    simp only [componentsFrom, List.map_cons, DNode.relabel_id, contains_map_perm]
    split
    · exact componentsFrom_relabel s seen rest
    · rw [component_relabel, ← List.map_append, componentsFrom_relabel s _ rest]
      rfl

theorem components_relabel (s : State) : (s.relabel φ).components = s.components.map (List.map φ) := by
  simp only [components, relabel_nodes]
  exact componentsFrom_relabel s [] s.nodes

theorem idxOf?_map_perm (l : List NodeId) (id : NodeId) : (l.map φ).idxOf? (φ id) = l.idxOf? id := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons, List.idxOf?_cons, φ.injective.eq_iff, beq_iff_eq]
    split <;> simp_all

theorem respectsEdges_relabel (s : State) (order : List NodeId) :
    (s.relabel φ).respectsEdges (order.map φ) = s.respectsEdges order := by
  simp only [respectsEdges, relabel_edges, List.all_map]
  congr 1
  funext e
  simp [Function.comp, idxOf?_map_perm]

theorem isTopologicalOrder_relabel (s : State) (ids order : List NodeId) :
    (s.relabel φ).isTopologicalOrder (ids.map φ) (order.map φ) = s.isTopologicalOrder ids order := by
  simp only [isTopologicalOrder, List.length_map, List.all_map, respectsEdges_relabel]
  congr 2
  · congr 1
    congr 1
    funext id
    simp [Function.comp]
  · simp [List.nodup_map_iff φ.injective]

theorem succs_relabel (s : State) (id : NodeId) : (s.relabel φ).succs (φ id) = (s.succs id).map φ := by
  simp [succs, outEdges_relabel, List.map_map, Function.comp]

theorem reaches_relabel (s : State) (a b : NodeId) : (s.relabel φ).reaches (φ a) (φ b) = s.reaches a b := by
  simp only [reaches, succs_relabel, List.any_map, relabel_edges, List.length_map]
  congr 1
  funext c
  simp only [Function.comp]
  rw [closure_map φ φ.injective (succs_relabel s), contains_map_perm]

end State

/-! ## Endpoints, views and per-node helpers -/

namespace State
variable {φ}

theorem srcEndpoint_relabel (s : State) (e : DEdge) :
    (s.relabel φ).srcEndpoint (e.relabel φ) = (s.srcEndpoint e).relabel φ := by
  unfold srcEndpoint
  rw [show (s.relabel φ).edges.length + 1 = s.edges.length + 1 by simp, chunkSource_relabel]
  cases s.chunkSource (s.edges.length + 1) e with
  | some entry =>
    simp only [Option.map_some, consumedStream?_relabel]
    cases s.consumedStream? entry <;> rfl
  | none =>
    simp only [Option.map_none, pathStart_relabel, DEdge.relabel_src, node?_relabel]
    cases s.node? (s.pathStart e).src with
    | none => rfl
    | some n =>
      simp only [Option.map_some, DNode.relabel_kind]
      cases n.kind <;> rfl

theorem dstEndpoint_relabel (s : State) (e : DEdge) :
    (s.relabel φ).dstEndpoint (e.relabel φ) = (s.dstEndpoint e).relabel φ := by
  unfold dstEndpoint
  simp only [DEdge.relabel_dst, node?_relabel, DEdge.relabel_src, DEdge.relabel_srcConn]
  cases s.node? e.dst with
  | none => rfl
  | some n =>
    simp only [Option.map_some, DNode.relabel_kind]
    cases n.kind with
    | mapExit | consumeExit =>
      simp only [dstRootEdge_relabel, DEdge.relabel_dst, node?_relabel]
      cases s.node? (s.dstRootEdge e).dst with
      | none => rfl
      | some rn =>
        simp only [Option.map_some, DNode.relabel_kind]
        cases rn.kind <;> rfl
    | _ => rfl

theorem pathStartsAtAccess_relabel (s : State) (e : DEdge) :
    (s.relabel φ).pathStartsAtAccess (e.relabel φ) = s.pathStartsAtAccess e := by
  simp [pathStartsAtAccess, pathStart_relabel, isAccessNode_relabel]

theorem pathEndsAtAccess_relabel (s : State) (e : DEdge) :
    (s.relabel φ).pathEndsAtAccess (e.relabel φ) = s.pathEndsAtAccess e := by
  simp [pathEndsAtAccess, pathEnd_relabel, isAccessNode_relabel]

@[simp] theorem memletData?_relabel (e : DEdge) : memletData? (e.relabel φ) = memletData? e := rfl
@[simp] theorem leafSrcSubset_relabel (e : DEdge) : leafSrcSubset (e.relabel φ) = leafSrcSubset e := rfl
@[simp] theorem leafDstSubset_relabel (e : DEdge) : leafDstSubset (e.relabel φ) = leafDstSubset e := rfl

/-- Relabel an edge choice. -/
abbrev relabelEB (x : DEdge × Bool) : DEdge × Bool := (x.1.relabel φ, x.2)

theorem viewEdgeOneOne_relabel (s : State) (v : DNode) (i : DEdge) (o? : Option DEdge) :
    (s.relabel φ).viewEdgeOneOne (v.relabel φ) (i.relabel φ) (o?.map (DEdge.relabel φ)) =
      (s.viewEdgeOneOne v i o?).map (relabelEB (φ := φ)) := by
  unfold viewEdgeOneOne
  have h1 : (o?.map (DEdge.relabel φ)).any ((s.relabel φ).pathEndsAtAccess ·) = o?.any (s.pathEndsAtAccess ·) := by
    cases o? <;> simp [pathEndsAtAccess_relabel]
  simp only [pathStartsAtAccess_relabel, h1]
  split
  · rfl
  · split
    · cases o? <;> rfl
    · split
      · rfl
      · cases o? with
        | none => rfl
        | some o =>
          simp only [Option.map_some, DEdge.relabel_dstConn, DEdge.relabel_srcConn, memletData?_relabel,
            DNode.relabel_accessData?, DEdge.relabel_src, DEdge.relabel_dst, isEntryNode_relabel,
            isExitNode_relabel]
          repeat' split
          all_goals simp_all [relabelEB]

/-- The body of `State.viewEdge?` for given incoming memlet edges and outgoing edges. -/
private def viewEdgeOf (s : State) (v : DNode) (ins outs : List DEdge) : Option (DEdge × Bool) :=
  if ins.isEmpty && outs.isEmpty then none
  else if ins.length = 1 && outs.length ≠ 1 then
    match ins with
    | i :: _ => if s.pathStartsAtAccess i then some (i, true) else none
    | [] => none
  else if outs.length = 1 && ins.length ≠ 1 then
    match outs with
    | o :: _ => if s.pathEndsAtAccess o then some (o, false) else none
    | [] => none
  else if ins.length = outs.length && outs.length ≠ 1 then none
  else
    match ins with
    | i :: _ => viewEdgeOneOne s v i outs.head?
    | [] => none

private theorem viewEdge?_eq (s : State) (v : DNode) :
    s.viewEdge? v = viewEdgeOf s v ((s.inEdges v.id).filter fun e => e.data matches .memlet _) (s.outEdges v.id) :=
  rfl

private theorem viewEdgeOf_relabel (s : State) (v : DNode) (ins outs : List DEdge) :
    viewEdgeOf (s.relabel φ) (v.relabel φ) (ins.map (DEdge.relabel φ)) (outs.map (DEdge.relabel φ)) =
      (viewEdgeOf s v ins outs).map (relabelEB (φ := φ)) := by
  unfold viewEdgeOf
  simp only [List.length_map, List.isEmpty_map]
  split
  · rfl
  · split
    · cases ins with
      | nil => rfl
      | cons i _ => simp only [List.map_cons, pathStartsAtAccess_relabel]; split <;> rfl
    · split
      · cases outs with
        | nil => rfl
        | cons o _ => simp only [List.map_cons, pathEndsAtAccess_relabel]; split <;> rfl
      · split
        · rfl
        · cases ins with
          | nil => rfl
          | cons i _ =>
            simp only [List.map_cons, List.head?_map]
            exact viewEdgeOneOne_relabel s v i _

theorem viewEdge?_relabel (s : State) (v : DNode) :
    (s.relabel φ).viewEdge? (v.relabel φ) = (s.viewEdge? v).map (relabelEB (φ := φ)) := by
  rw [viewEdge?_eq, viewEdge?_eq, DNode.relabel_id, inEdges_relabel, outEdges_relabel]
  have hins : ((s.inEdges v.id).map (DEdge.relabel φ)).filter (fun e => e.data matches .memlet _) =
      ((s.inEdges v.id).filter fun e => e.data matches .memlet _).map (DEdge.relabel φ) := by
    rw [List.filter_map]
    rfl
  rw [hins]
  exact viewEdgeOf_relabel s v _ _

theorem viewedData?_relabel (s : State) (v : DNode) :
    (s.relabel φ).viewedData? (v.relabel φ) = s.viewedData? v := by
  unfold viewedData?
  rw [viewEdge?_relabel]
  cases s.viewEdge? v with
  | none => rfl
  | some x =>
    obtain ⟨e, b⟩ := x
    cases b <;> simp [relabelEB, pathStart_relabel, pathEnd_relabel, node?_relabel]
    · cases s.node? (s.pathEnd e).dst <;> rfl
    · cases s.node? (s.pathStart e).src <;> rfl

theorem viewBinding?_relabel (s : State) (v : DNode) :
    (s.relabel φ).viewBinding? (v.relabel φ) = s.viewBinding? v := by
  unfold viewBinding?
  rw [viewEdge?_relabel]
  cases s.viewEdge? v with
  | none => rfl
  | some x =>
    obtain ⟨e, b⟩ := x
    cases b <;> simp [relabelEB, pathStart_relabel, pathEnd_relabel, node?_relabel]
    · cases s.node? (s.pathEnd e).dst <;> rfl
    · cases s.node? (s.pathStart e).src <;> rfl

theorem viewSig_relabel (s : State) (id : NodeId) : (s.relabel φ).viewSig (φ id) = s.viewSig id := by
  unfold viewSig
  rw [node?_relabel]
  cases s.node? id with
  | none => rfl
  | some n => simp [viewBinding?_relabel]

theorem isViewNode_relabel (ctx : Ctx) (s : State) (id : NodeId) :
    (s.relabel φ).isViewNode ctx (φ id) = s.isViewNode ctx id := by
  unfold isViewNode
  rw [node?_relabel]
  cases s.node? id <;> rfl

end State

/-- The views of `pending` whose viewed container is no pending view. -/
private def vgReady (d b : NodeId → Option Name) (pending : List NodeId) : List NodeId :=
  pending.filter fun id => match b id with
    | some x => !(pending.filterMap d).contains x
    | none => true

private theorem viewGensWith_succ_cons (d b : NodeId → Option Name) (f : Nat) (p : NodeId) (ps : List NodeId) :
    viewGensWith d b (f + 1) (p :: ps) =
      if (vgReady d b (p :: ps)).isEmpty then p :: ps
      else vgReady d b (p :: ps) ++
        viewGensWith d b f ((p :: ps).filter fun id => !(vgReady d b (p :: ps)).contains id) := rfl

private theorem vgReady_map {d₁ b₁ d₂ b₂ : NodeId → Option Name} (hd : ∀ id, d₂ (φ id) = d₁ id)
    (hb : ∀ id, b₂ (φ id) = b₁ id) (l : List NodeId) : vgReady d₂ b₂ (l.map φ) = (vgReady d₁ b₁ l).map φ := by
  unfold vgReady
  have hpd : (l.map φ).filterMap d₂ = l.filterMap d₁ := by
    rw [List.filterMap_map]
    exact List.filterMap_congr fun id _ => hd id
  rw [List.filter_map, hpd]
  congr 1
  apply List.filter_congr
  intro id _
  simp only [Function.comp, hb]

theorem viewGensWith_map {d₁ b₁ d₂ b₂ : NodeId → Option Name} (hd : ∀ id, d₂ (φ id) = d₁ id)
    (hb : ∀ id, b₂ (φ id) = b₁ id) :
    ∀ (f : Nat) (pending : List NodeId), viewGensWith d₂ b₂ f (pending.map φ) = (viewGensWith d₁ b₁ f pending).map φ
  | 0, _ => rfl
  | _ + 1, [] => rfl
  | f + 1, p :: ps => by
    rw [List.map_cons, viewGensWith_succ_cons, viewGensWith_succ_cons, ← List.map_cons, vgReady_map φ hd hb,
      List.isEmpty_map]
    split
    · rfl
    · rw [List.map_append, List.filter_map]
      congr 1
      have : ((fun id => !((vgReady d₁ b₁ (p :: ps)).map φ).contains id) ∘ φ) =
          fun id => !(vgReady d₁ b₁ (p :: ps)).contains id := by
        funext id
        simp [Function.comp]
      rw [this]
      exact viewGensWith_map hd hb f _

theorem keepFirstSig_map {α : Type} [DecidableEq α] {sig₁ sig₂ : NodeId → Option α} (h : ∀ id, sig₂ (φ id) = sig₁ id) :
    ∀ (seen : List α) (l : List NodeId), keepFirstSig sig₂ seen (l.map φ) = (keepFirstSig sig₁ seen l).map φ
  | _, [] => rfl
  | seen, x :: xs => by
    simp only [List.map_cons, keepFirstSig, h]
    split
    · split
      · exact keepFirstSig_map h seen xs
      · rw [keepFirstSig_map h _ xs]; rfl
    · rw [keepFirstSig_map h seen xs]; rfl

namespace State
variable {φ}

theorem viewGens_relabel (s : State) (f : Nat) (pending : List NodeId) :
    (s.relabel φ).viewGens f (pending.map φ) = (s.viewGens f pending).map φ := by
  unfold viewGens
  refine viewGensWith_map φ (fun id => ?_) (fun id => ?_) f pending
  · rw [node?_relabel]; cases s.node? id <;> rfl
  · rw [node?_relabel]
    cases s.node? id with
    | none => rfl
    | some n => exact viewedData?_relabel s n

theorem viewOrder_relabel (ctx : Ctx) (s : State) (ids : List NodeId) :
    (s.relabel φ).viewOrder ctx (ids.map φ) = (s.viewOrder ctx ids).map φ := by
  unfold viewOrder
  have hf : (ids.map φ).filter ((s.relabel φ).isViewNode ctx) = (ids.filter (s.isViewNode ctx)).map φ := by
    rw [List.filter_map]
    congr 1
    apply List.filter_congr
    intro id _
    simp [Function.comp, isViewNode_relabel]
  simp only
  rw [hf, keepFirstSig_map φ (viewSig_relabel s), List.length_map, viewGens_relabel]

@[simp] theorem nestedConns_relabel (n : DNode) : nestedConns (n.relabel φ) = nestedConns n := rfl

theorem outEdgesOn_relabel (s : State) (n : DNode) (c : Conn) :
    (s.relabel φ).outEdgesOn (n.relabel φ) c = (s.outEdgesOn n c).map (DEdge.relabel φ) := by
  simp only [outEdgesOn, DNode.relabel_id, outEdges_relabel, List.filter_map]
  rfl

theorem inEdgeOn_relabel (s : State) (n : DNode) (c : Conn) :
    (s.relabel φ).inEdgeOn (n.relabel φ) c = (s.inEdgeOn n c).map (DEdge.relabel φ) := by
  simp only [inEdgeOn, DNode.relabel_id, inEdges_relabel, List.find?_map]
  rfl

theorem hasDynamicWcrOut_relabel (s : State) (n : DNode) :
    (s.relabel φ).hasDynamicWcrOut (n.relabel φ) = s.hasDynamicWcrOut n := by
  simp only [hasDynamicWcrOut, DNode.relabel_id, outEdges_relabel, List.any_map]
  rfl

theorem nestedEndpoint_relabel (s : State) (n : DNode) (c : Conn) :
    nestedEndpoint (s.relabel φ) (n.relabel φ) c = (nestedEndpoint s n c).relabel φ := by
  unfold nestedEndpoint
  rw [inEdgeOn_relabel]
  cases s.inEdgeOn n c with
  | some e => exact srcEndpoint_relabel s e
  | none =>
    simp only [Option.map_none, outEdgesOn_relabel]
    cases s.outEdgesOn n c with
    | nil => rfl
    | cons e _ => exact dstEndpoint_relabel s e

theorem hbConflict_relabel (s : State) (x y : NodeId × Log) :
    (s.relabel φ).hbConflict (φ x.1, x.2) (φ y.1, y.2) = s.hbConflict x y := by
  simp [hbConflict, reaches_relabel]

/-- Relabel the nodes of a trace. -/
def relabelTrace (φ : Equiv.Perm NodeId) (tr : List (NodeId × Log)) : List (NodeId × Log) :=
  tr.map fun x => (φ x.1, x.2)

theorem hbFree_relabel (s : State) : ∀ tr : List (NodeId × Log),
    (s.relabel φ).hbFree (relabelTrace φ tr) = s.hbFree tr
  | [] => rfl
  | x :: xs => by
    show ((relabelTrace φ xs).all (fun y => !(s.relabel φ).hbConflict (φ x.1, x.2) y) &&
      (s.relabel φ).hbFree (relabelTrace φ xs)) = _
    rw [hbFree_relabel s xs, hbFree]
    congr 1
    simp only [relabelTrace, List.all_map]
    congr 1
    funext y
    simp [Function.comp, hbConflict_relabel]

end State

namespace Ctx
variable {φ}

theorem refSetTarget?_relabel (ctx : Ctx) (s : State) (e : DEdge) :
    refSetTarget? ctx (s.relabel φ) (e.relabel φ) = refSetTarget? ctx s e := by
  unfold refSetTarget?
  simp only [State.dstRootEdge_relabel, DEdge.relabel_dstConn, DEdge.relabel_dst, State.node?_relabel]
  split
  · cases s.node? (s.dstRootEdge e).dst <;> rfl
  · rfl

/-- One side of `Ctx.isViewDef`: the node `id` is a view access node whose
view edge is `(r, b)`. -/
private def viewDefAt (ctx : Ctx) (s : State) (id : NodeId) (r : DEdge) (b : Bool) : Bool :=
  match (s.node? id).filter (fun n => match n.kind with
      | .access d => ctx.isView d
      | _ => false) with
  | some v => decide (s.viewEdge? v = some (r, b))
  | none => false

private theorem isViewDef_eq (ctx : Ctx) (s : State) (e : DEdge) :
    ctx.isViewDef s e = (viewDefAt ctx s (s.pathStart e).src (s.pathStart e) false ||
      viewDefAt ctx s (s.dstRootEdge e).dst (s.dstRootEdge e) true) := rfl

private theorem viewDefAt_relabel (ctx : Ctx) (s : State) (id : NodeId) (r : DEdge) (b : Bool) :
    viewDefAt ctx (s.relabel φ) (φ id) (r.relabel φ) b = viewDefAt ctx s id r b := by
  unfold viewDefAt
  rw [State.node?_relabel]
  cases s.node? id with
  | none => rfl
  | some n =>
    have hk : (n.relabel φ).kind = n.kind := rfl
    simp only [Option.map_some, Option.filter_some, hk]
    cases n.kind with
    | access d =>
      by_cases hv : ctx.isView d = true
      · simp only [hv, ↓reduceIte, State.viewEdge?_relabel]
        cases s.viewEdge? n with
        | none => simp
        | some x =>
          obtain ⟨f, c⟩ := x
          simp [State.relabelEB, DEdge.relabel_injective.eq_iff]
      · simp [hv]
    | _ => rfl

theorem isViewDef_relabel (ctx : Ctx) (s : State) (e : DEdge) :
    ctx.isViewDef (s.relabel φ) (e.relabel φ) = ctx.isViewDef s e := by
  rw [isViewDef_eq, isViewDef_eq, State.pathStart_relabel, State.dstRootEdge_relabel, DEdge.relabel_src,
    DEdge.relabel_dst, viewDefAt_relabel, viewDefAt_relabel]

theorem privateTransients_relabel (ctx : Ctx) (s : State) (entry : NodeId) :
    ctx.privateTransients (s.relabel φ) (φ entry) = ctx.privateTransients s entry := by
  unfold privateTransients
  dsimp only
  rw [State.scopeNodes_relabel]
  simp only [State.relabel_nodes, List.filter_map, List.length_map, Function.comp_def, DNode.relabel_mentions,
    DNode.relabel_id, State.contains_map_perm]

theorem stateTransients_relabel (ctx : Ctx) (s : State) :
    ctx.stateTransients (s.relabel φ) = ctx.stateTransients s := by
  unfold stateTransients
  simp only [State.relabel_nodes, List.filter_map, List.length_map]
  rfl

end Ctx

/-! ## Sequential states -/

namespace St
variable {φ}

@[simp] theorem relabel_mem (st : St) : (st.relabel φ).mem = st.mem := rfl
@[simp] theorem relabel_log (st : St) : (st.relabel φ).log = st.log := rfl
@[simp] theorem relabel_vals (st : St) : (st.relabel φ).vals = relabelVals φ st.vals := rfl

theorem relabel_symm (st : St) : (st.relabel φ).relabel φ.symm = st := by
  rcases st with ⟨m, g, vs⟩
  simp only [St.relabel, St.mapVals, relabelVals, List.map_map, St.mk.injEq, true_and]
  conv => rhs; rw [← List.map_id vs]
  exact List.map_congr_left fun x _ => by simp

@[simp] theorem relabel_logged (st : St) (g : Log) : (st.relabel φ).logged g = (st.logged g).relabel φ := rfl
@[simp] theorem relabel_forNode (st : St) : (st.relabel φ).forNode = st.forNode.relabel φ := rfl
@[simp] theorem relabel_afterNode (st st' : St) :
    (st.relabel φ).afterNode (st'.relabel φ) = (st.afterNode st').relabel φ := rfl
@[simp] theorem relabel_bindAlias (st : St) (n : Name) (b : AliasBinding) :
    (st.relabel φ).bindAlias n b = (st.bindAlias n b).relabel φ := rfl

theorem relabel_val? (st : St) (k : NodeId × Conn) : (st.relabel φ).val? (φ k.1, k.2) = st.val? k := by
  unfold val?
  simp only [relabel_vals, relabelVals, List.find?_map]
  have : ((fun x : (NodeId × Conn) × Val => decide (x.1 = (φ k.1, k.2))) ∘
      fun x : (NodeId × Conn) × Val => ((φ x.1.1, x.1.2), x.2)) = fun x => decide (x.1 = k) := by
    funext x
    obtain ⟨⟨a, c⟩, v⟩ := x
    obtain ⟨a', c'⟩ := k
    simp [φ.injective.eq_iff]
  rw [this]
  cases st.vals.find? (fun x => decide (x.1 = k)) <;> rfl

theorem relabel_setVal (st : St) (k : NodeId × Conn) (v : Val) :
    (st.relabel φ).setVal (φ k.1, k.2) v = (st.setVal k v).relabel φ := rfl

/-- Relabel the state component of a pair. -/
abbrev relabelR {α : Type} (φ : Equiv.Perm NodeId) : α × St → α × St := Prod.map id (St.relabel φ)

variable (ctx : Ctx) (ρ : Valuation)

theorem read_relabel (st : St) (n : Name) (idx : Idx) (oob : Bool) :
    (st.relabel φ).read ctx ρ n idx oob = (st.read ctx ρ n idx oob).map (relabelR φ) := by
  unfold read
  simp only [relabel_mem]
  exact Res.bind_shift' _ fun l => by split <;> simp_all [Res.map]

theorem readMany_relabel (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    (st.relabel φ).readMany ctx ρ n pts oob = (st.readMany ctx ρ n pts oob).map (relabelR φ) := by
  unfold readMany
  exact List.foldlM_res_comm (relabelR φ) _ (fun ⟨acc, st⟩ p => by
    simp only [Prod.map_apply, id_eq, read_relabel]
    exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl) pts ([], st)

theorem write_relabel (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    (st.relabel φ).write ctx ρ n idx oob v = (st.write ctx ρ n idx oob v).map (St.relabel φ) := by
  unfold write
  simp only [relabel_mem, relabel_log]
  exact Res.bind_shift' _ fun l => rfl

theorem wcr_relabel (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    (st.relabel φ).wcr ctx ρ n idx oob op v = (st.wcr ctx ρ n idx oob op v).map (St.relabel φ) := by
  unfold wcr
  simp only [relabel_mem, relabel_log]
  exact Res.bind_shift' _ fun l => rfl

theorem store_relabel (st : St) (n : Name) (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    (st.relabel φ).store ctx ρ n idx oob w v = (st.store ctx ρ n idx oob w v).map (St.relabel φ) := by
  unfold store
  cases w with
  | none => exact write_relabel ..
  | some op => exact wcr_relabel ..

theorem queue_relabel (st : St) (n : Name) (idx : Idx) :
    (st.relabel φ).queue ctx ρ n idx = st.queue ctx ρ n idx := rfl

theorem push_relabel (st : St) (n : Name) (idx : Idx) (vs : List Val) :
    (st.relabel φ).push ctx ρ n idx vs = (st.push ctx ρ n idx vs).map (St.relabel φ) := by
  unfold push
  simp only [relabel_mem, relabel_log]
  exact Res.bind_shift' _ fun l => by by_cases h : vs.isEmpty <;> simp [h, Res.map]; rfl

theorem popN_relabel (st : St) (n : Name) (idx : Idx) (k : Nat) :
    (st.relabel φ).popN ctx ρ n idx k = (st.popN ctx ρ n idx k).map (relabelR φ) := by
  unfold popN
  simp only [relabel_mem, relabel_log]
  refine Res.bind_shift' _ fun l => ?_
  by_cases h0 : k = 0
  · simp [h0, Res.map]
  · by_cases h1 : (st.mem.queues (st.mem.resolve l)).length < k <;> simp [h0, h1, Res.map]; rfl

end St

/-! ## Leaf functions -/

section Leaf

open St (relabelR)

variable {φ} (ctx : Ctx) (ρ : Valuation)

theorem readScalarData_relabel (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    readScalarData ctx ρ d sub oob (st.relabel φ) = (readScalarData ctx ρ d sub oob st).map (relabelR φ) := by
  unfold readScalarData
  simp only [St.relabel_mem]
  refine Res.bind_shift' _ fun pts => ?_
  split
  · split
    · rw [St.popN_relabel]
      exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl
    · rw [St.read_relabel]
      exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl
  · rfl

theorem readEdgeScalar_relabel (s : State) (e : DEdge) (st : St) :
    readEdgeScalar ctx ρ (s.relabel φ) (e.relabel φ) (st.relabel φ) =
      (readEdgeScalar ctx ρ s e st).map (relabelR φ) := by
  unfold readEdgeScalar
  rw [State.srcEndpoint_relabel]
  cases s.srcEndpoint e with
  | code key =>
    simp only [State.Endpoint.relabel_code]
    rw [show (φ key.1, key.2) = (φ key.1, key.2) from rfl, St.relabel_val?]
    rfl
  | data d sub => exact readScalarData_relabel ..
  | invalid => rfl

theorem copyData_relabel (st : St) (sd : Name) (ssub : Option Subset) (dd : Name) (dsub : Option Subset)
    (w : Option WcrOp) (oob : Bool) :
    copyData ctx ρ (st.relabel φ) sd ssub dd dsub w oob = (copyData ctx ρ st sd ssub dd dsub w oob).map (St.relabel φ) := by
  unfold copyData
  simp only [St.relabel_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun dpts => Res.bind_shift' _ fun sp => ?_
    rw [St.popN_relabel]
    refine Res.bind_shift _ fun ⟨vs, st'⟩ => ?_
    exact List.foldlM_res_comm (St.relabel φ) _ (fun st ⟨q, v⟩ => St.store_relabel ..) _ _
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dp => ?_
    rw [St.readMany_relabel]
    exact Res.bind_shift _ fun ⟨vs, st'⟩ => St.push_relabel ..
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dpts => ?_
    refine List.foldlM_res_comm (St.relabel φ) _ (fun st ⟨p, q⟩ => ?_) _ _
    simp only
    rw [St.read_relabel]
    exact Res.bind_shift _ fun ⟨v, st'⟩ => St.store_relabel ..

theorem execCopy_relabel (s : State) (e : DEdge) (st : St) :
    execCopy ctx ρ (s.relabel φ) (e.relabel φ) (st.relabel φ) = (execCopy ctx ρ s e st).map (St.relabel φ) := by
  unfold execCopy
  simp only [DEdge.relabel_data, State.srcEndpoint_relabel, State.dstEndpoint_relabel]
  split
  · rfl
  · cases s.srcEndpoint e <;> cases s.dstEndpoint e <;> first | rfl | exact copyData_relabel ..

theorem bindInData_relabel (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    bindInData ctx ρ d sub oob (st.relabel φ) = (bindInData ctx ρ d sub oob st).map (relabelR φ) := by
  unfold bindInData
  simp only [St.relabel_mem]
  split
  · exact Res.bind_shift' _ fun p => Res.bind_shift' _ fun q => rfl
  · refine Res.bind_shift' _ fun pts => ?_
    split
    · rw [St.read_relabel]
      exact Res.bind_shift _ fun ⟨v, st'⟩ => rfl
    · refine Res.bind_shift' _ fun lo => ?_
      rw [St.readMany_relabel]
      exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl

theorem bindIn_relabel (s : State) (e : DEdge) (st : St) :
    bindIn ctx ρ (s.relabel φ) (e.relabel φ) (st.relabel φ) = (bindIn ctx ρ s e st).map (relabelR φ) := by
  unfold bindIn
  rw [State.srcEndpoint_relabel]
  cases s.srcEndpoint e with
  | code key =>
    simp only [State.Endpoint.relabel_code]
    rw [St.relabel_val?]
    rfl
  | data d sub =>
    simp only [State.Endpoint.relabel_data, DEdge.relabel_allowOob]
    rw [bindInData_relabel]
    exact Res.bind_shift _ fun ⟨c, st'⟩ => rfl
  | invalid => rfl

theorem bindInputs_relabel (s : State) (n : DNode) (st : St) :
    bindInputs ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) = (bindInputs ctx ρ s n st).map (relabelR φ) := by
  unfold bindInputs
  rw [DNode.relabel_id, State.inEdges_relabel, List.foldlM_map,
    show (([] : List (Conn × ConnIn)), st.relabel φ) = relabelR φ ([], st) from rfl,
    List.foldlM_res_comm₂ (relabelR φ) ?hstep]
  · exact Res.bind_shift _ fun ⟨bs, st'⟩ => rfl
  case hstep =>
    intro ⟨acc, st⟩ e
    simp only [Prod.map_apply, id_eq, DEdge.relabel_data, DEdge.relabel_dstConn]
    split
    · rfl
    · rw [bindIn_relabel]
      refine Res.bind_shift _ fun ⟨ci, st'⟩ => ?_
      cases ci <;> rfl

theorem storeScalar_relabel (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool) (v : Val) :
    storeScalar ctx ρ (st.relabel φ) d sub w oob v = (storeScalar ctx ρ st d sub w oob v).map (St.relabel φ) := by
  unfold storeScalar
  simp only [St.relabel_mem]
  split
  · exact Res.bind_shift' _ fun p => St.push_relabel ..
  · refine Res.bind_shift' _ fun pts => ?_
    split
    · exact St.store_relabel ..
    · rfl
    · rfl

theorem applyScalar_relabel (s : State) (n : DNode) (st : St) (c : Conn) (v : Val) :
    applyScalar ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) c v =
      (applyScalar ctx ρ s n st c v).map (St.relabel φ) := by
  unfold applyScalar
  rw [State.outEdgesOn_relabel, List.foldlM_map]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st e => ?_) _ _
  simp only [DEdge.relabel_data, State.dstEndpoint_relabel]
  split
  · rfl
  · cases s.dstEndpoint e with
    | code key => rfl
    | data d sub => exact storeScalar_relabel ..
    | invalid => rfl

theorem storeArrayWrite_relabel (st : St) (d : Name) (sub : Option Subset) (w : Option WcrOp) (oob : Bool)
    (rel : Idx) (v : Val) :
    storeArrayWrite ctx ρ (st.relabel φ) d sub w oob rel v =
      (storeArrayWrite ctx ρ st d sub w oob rel v).map (St.relabel φ) := by
  unfold storeArrayWrite
  simp only [St.relabel_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun lo => ?_
    split
    · refine Res.bind_shift' _ fun pts => ?_
      split
      · exact St.store_relabel ..
      · rfl
    · rfl

theorem applyArrayWrite_relabel (s : State) (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    applyArrayWrite ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) c rel v =
      (applyArrayWrite ctx ρ s n st c rel v).map (St.relabel φ) := by
  unfold applyArrayWrite
  rw [State.outEdgesOn_relabel, List.foldlM_map]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st e => ?_) _ _
  simp only [DEdge.relabel_data, State.dstEndpoint_relabel]
  split
  · rfl
  · cases s.dstEndpoint e with
    | data d sub => exact storeArrayWrite_relabel ..
    | _ => rfl

theorem pushTo_relabel (st : St) (d : Name) (sub : Option Subset) (vs : List Val) :
    pushTo ctx ρ (st.relabel φ) d sub vs = (pushTo ctx ρ st d sub vs).map (St.relabel φ) := by
  unfold pushTo
  simp only [St.relabel_mem]
  split
  · exact Res.bind_shift' _ fun p => St.push_relabel ..
  · rfl

theorem applyPush_relabel (s : State) (n : DNode) (st : St) (c : Conn) (vs : List Val) :
    applyPush ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) c vs =
      (applyPush ctx ρ s n st c vs).map (St.relabel φ) := by
  unfold applyPush
  rw [State.outEdgesOn_relabel, List.foldlM_map]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st e => ?_) _ _
  simp only [State.dstEndpoint_relabel]
  cases s.dstEndpoint e with
  | data d sub => exact pushTo_relabel ..
  | _ => rfl

theorem popFrom_relabel (st : St) (d : Name) (sub : Option Subset) (k : Nat) :
    popFrom ctx ρ (st.relabel φ) d sub k = (popFrom ctx ρ st d sub k).map (St.relabel φ) := by
  unfold popFrom
  simp only [St.relabel_mem]
  split
  · refine Res.bind_shift' _ fun p => ?_
    rw [St.popN_relabel]
    exact Res.bind_shift _ fun ⟨vs, st'⟩ => rfl
  · rfl

theorem applyPop_relabel (s : State) (n : DNode) (st : St) (c : Conn) (k : Nat) :
    applyPop ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) c k = (applyPop ctx ρ s n st c k).map (St.relabel φ) := by
  unfold applyPop
  rw [State.inEdgeOn_relabel]
  cases s.inEdgeOn n c with
  | none => rfl
  | some e =>
    simp only [Option.map_some, State.srcEndpoint_relabel]
    cases s.srcEndpoint e with
    | data d sub => exact popFrom_relabel ..
    | _ => rfl

theorem applyOut_relabel (s : State) (n : DNode) (st : St) (out : TaskletOut) :
    applyOut ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) out = (applyOut ctx ρ s n st out).map (St.relabel φ) := by
  unfold applyOut
  refine List.bind_foldlM_res_comm _ _ (fun st x => ?_) (fun st => ?_) _ _
  · obtain ⟨c, k⟩ := x
    exact applyPop_relabel ..
  refine List.bind_foldlM_res_comm _ _ (fun st x => ?_) (fun st => ?_) _ _
  · obtain ⟨c, v⟩ := x
    exact applyScalar_relabel ..
  refine List.bind_foldlM_res_comm _ _ (fun st x => ?_) (fun st => ?_) _ _
  · obtain ⟨c, rel, v⟩ := x
    exact applyArrayWrite_relabel ..
  refine List.foldlM_res_comm₂ _ (fun st x => ?_) _ _
  obtain ⟨c, vs⟩ := x
  exact applyPush_relabel ..

theorem execCode_relabel (s : State) (n : DNode) (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut)
    (ue : ErrReason) :
    execCode ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) run ue = (execCode ctx ρ s n st run ue).map (St.relabel φ) := by
  unfold execCode
  rw [State.hasDynamicWcrOut_relabel]
  split
  · rfl
  · rw [bindInputs_relabel]
    refine Res.bind_shift _ fun ⟨ins, st'⟩ => ?_
    simp only [Prod.map_apply, id_eq]
    split
    · rfl
    · exact applyOut_relabel ..

theorem reduceData_relabel (st : St) (props : ReduceProps) (sd : Name) (ssub : Option Subset) (dd : Name)
    (dsub : Option Subset) (w : Option WcrOp) (soob doob : Bool) :
    reduceData ctx ρ (st.relabel φ) props sd ssub dd dsub w soob doob =
      (reduceData ctx ρ st props sd ssub dd dsub w soob doob).map (St.relabel φ) := by
  unfold reduceData
  simp only [St.relabel_mem]
  split
  · rfl
  · refine Res.bind_shift' _ fun spts => Res.bind_shift' _ fun dpts => ?_
    split <;> (try split) <;> first
      | rfl
      | rw [St.readMany_relabel]
        refine Res.bind_shift _ fun ⟨vals, st'⟩ => ?_
        refine List.foldlM_res_comm (St.relabel φ) _ (fun st ⟨g, q⟩ => ?_) _ _
        simp only [St.relabel_mem]
        repeat' split
        all_goals first
          | rfl
          | exact Res.bind_shift' _ fun i => St.store_relabel ..
          | exact St.wcr_relabel ..

/-- The body of `execReduce` for given memlet in- and out-edges. -/
private def execReduceOf (ctx : Ctx) (ρ : Valuation) (s : State) (props : ReduceProps) (st : St)
    (ins outs : List DEdge) : Res St :=
  match ins, outs with
  | [ie], [oe] =>
    match s.srcEndpoint ie, s.dstEndpoint oe, oe.data.memlet? with
    | .data sd ssub, .data dd dsub, some om =>
      reduceData ctx ρ st props sd ssub dd dsub om.wcr ie.allowOob om.allowOob
    | _, _, _ => .err (.malformed "reduce node not between containers")
  | _, _ => .err (.malformed "reduce node must have exactly one input and one output")

private theorem execReduce_eq (s : State) (n : DNode) (props : ReduceProps) (st : St) :
    execReduce ctx ρ s n props st = execReduceOf ctx ρ s props st
      ((s.inEdges n.id).filter fun e => e.data matches .memlet _)
      ((s.outEdges n.id).filter fun e => e.data matches .memlet _) := rfl

theorem execReduce_relabel (s : State) (n : DNode) (props : ReduceProps) (st : St) :
    execReduce ctx ρ (s.relabel φ) (n.relabel φ) props (st.relabel φ) =
      (execReduce ctx ρ s n props st).map (St.relabel φ) := by
  rw [execReduce_eq, execReduce_eq, DNode.relabel_id, State.inEdges_relabel, State.outEdges_relabel]
  have hf : ∀ l : List DEdge, (l.map (DEdge.relabel φ)).filter (fun e => e.data matches .memlet _) =
      (l.filter fun e => e.data matches .memlet _).map (DEdge.relabel φ) := by
    intro l
    rw [List.filter_map]
    rfl
  rw [hf, hf]
  generalize (s.inEdges n.id).filter (fun e => e.data matches .memlet _) = ins
  generalize (s.outEdges n.id).filter (fun e => e.data matches .memlet _) = outs
  unfold execReduceOf
  match ins, outs with
  | [ie], [oe] =>
    simp only [List.map_cons, List.map_nil, State.srcEndpoint_relabel, State.dstEndpoint_relabel,
      DEdge.relabel_data, DEdge.relabel_allowOob]
    cases s.srcEndpoint ie <;> cases s.dstEndpoint oe <;> cases oe.data.memlet? <;>
      first | rfl | exact reduceData_relabel ..
  | [], _ => rfl
  | _ :: _ :: _, _ => rfl
  | [_], [] => rfl
  | [_], _ :: _ :: _ => rfl

theorem aliasTo_relabel (st : St) (d base : Name) (sub : Option Subset) :
    aliasTo ctx ρ (st.relabel φ) d base sub = (aliasTo ctx ρ st d base sub).map (St.relabel φ) := by
  unfold aliasTo
  simp only [St.relabel_mem]
  split
  · rfl
  · repeat (refine Res.bind_shift' _ fun _ => ?_)
    rfl

theorem bindView_relabel (s : State) (v : DNode) (d : Name) (st : St) :
    bindView ctx ρ (s.relabel φ) (v.relabel φ) d (st.relabel φ) = (bindView ctx ρ s v d st).map (St.relabel φ) := by
  unfold bindView
  rw [State.viewEdge?_relabel]
  cases s.viewEdge? v with
  | none => rfl
  | some x =>
    obtain ⟨e, b⟩ := x
    have key : ∀ (o : Option DNode) (sub : Option Subset),
        (match (o.map (DNode.relabel φ)).bind DNode.accessData? with
          | none => (Res.err (.malformed s!"view edge of '{d}' does not reach an access node") : Res St)
          | some base => aliasTo ctx ρ (st.relabel φ) d base sub) =
        (match o.bind DNode.accessData? with
          | none => (Res.err (.malformed s!"view edge of '{d}' does not reach an access node") : Res St)
          | some base => aliasTo ctx ρ st d base sub).map (St.relabel φ) := by
      intro o sub
      cases o with
      | none => rfl
      | some n =>
        simp only [Option.map_some, Option.bind_some, DNode.relabel_accessData?]
        cases n.accessData? with
        | none => rfl
        | some base => exact aliasTo_relabel ..
    cases b <;>
    · simp only [Option.map_some, State.relabelEB, State.pathStart_relabel, State.pathEnd_relabel,
        DEdge.relabel_src, DEdge.relabel_dst, State.node?_relabel, State.leafSrcSubset_relabel,
        State.leafDstSubset_relabel, Bool.false_eq_true, ↓reduceIte]
      exact key _ _

theorem bindViews_relabel (s : State) (ids : List NodeId) (st : St) :
    bindViews ctx ρ (s.relabel φ) (ids.map φ) (st.relabel φ) = (bindViews ctx ρ s ids st).map (St.relabel φ) := by
  unfold bindViews
  rw [List.foldlM_map]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st id => ?_) _ _
  simp only [State.node?_relabel]
  cases s.node? id with
  | none => rfl
  | some n =>
    simp only [Option.map_some, DNode.relabel_kind]
    cases n.kind with
    | access d =>
      simp only
      split
      · exact bindView_relabel ..
      · rfl
    | _ => rfl

theorem refSet_relabel (s : State) (e : DEdge) (d : Name) (st : St) :
    refSet ctx ρ (s.relabel φ) (e.relabel φ) d (st.relabel φ) = (refSet ctx ρ s e d st).map (St.relabel φ) := by
  unfold refSet
  rw [State.srcEndpoint_relabel]
  cases s.srcEndpoint e with
  | data base sub => exact aliasTo_relabel ..
  | _ => rfl

theorem execCopiesInto_relabel (s : State) (n : DNode) (st : St) :
    execCopiesInto ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) =
      (execCopiesInto ctx ρ s n st).map (St.relabel φ) := by
  unfold execCopiesInto
  rw [DNode.relabel_id, State.inEdges_relabel, List.foldlM_map]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st e => ?_) _ _
  simp only [DEdge.relabel_data, Ctx.isViewDef_relabel, State.isRouting_relabel, Ctx.refSetTarget?_relabel]
  split
  · rfl
  · split
    · rfl
    · split
      · exact refSet_relabel ..
      · rw [State.srcEndpoint_relabel]
        cases s.srcEndpoint e with
        | data _ _ => exact execCopy_relabel ..
        | _ => rfl

theorem execSimple_relabel (s : State) (n : DNode) (st : St) :
    execSimple ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) = (execSimple ctx ρ s n st).map (St.relabel φ) := by
  unfold execSimple
  rw [DNode.relabel_kind]
  split <;> first
    | exact execCopiesInto_relabel ..
    | exact execCode_relabel ..
    | rfl
    | split
      · exact execReduce_relabel ..
      · exact execCode_relabel ..

theorem bindDynamicInputs_relabel (s : State) (n : DNode) (st : St) :
    bindDynamicInputs ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) =
      (bindDynamicInputs ctx ρ s n st).map (relabelR φ) := by
  unfold bindDynamicInputs
  rw [DNode.relabel_id, State.inEdges_relabel, List.foldlM_map]
  refine List.foldlM_res_comm₂ (relabelR φ) (fun ⟨ρ, st⟩ e => ?_) _ (ρ, st)
  simp only [Prod.map_apply, id_eq, State.isDynamicInput_relabel, DEdge.relabel_data, DEdge.relabel_dstConn]
  split
  · split
    · rw [readEdgeScalar_relabel]
      refine Res.bind_shift _ fun ⟨v?, st'⟩ => ?_
      cases v? <;> rfl
    · rfl
  · rfl

/-- Relabel the state of a scope set-up result. -/
abbrev relabelSetup {α : Type} (φ : Equiv.Perm NodeId) : Valuation × St × α → Valuation × St × α :=
  Prod.map id (Prod.map (St.relabel φ) id)

theorem mapSetup_relabel (s : State) (n : DNode) (mp : MapNode) (st : St) :
    mapSetup ctx ρ (s.relabel φ) (n.relabel φ) mp (st.relabel φ) = (mapSetup ctx ρ s n mp st).map (relabelSetup φ) := by
  unfold mapSetup
  rw [bindDynamicInputs_relabel]
  exact Res.bind_shift _ fun ⟨ρ', st'⟩ => Res.bind_shift' _ fun pts => rfl

theorem scopeFinish_relabel (pre : St) (logs : List Log) :
    scopeFinish ctx (pre.relabel φ) logs = (scopeFinish ctx pre logs).map (St.relabel φ) := by
  unfold scopeFinish
  simp only [St.relabel_mem]
  split <;> rfl

theorem consumeStream_relabel (s : State) (n : DNode) (m : Mem) :
    consumeStream ctx ρ (s.relabel φ) (n.relabel φ) m = consumeStream ctx ρ s n m := by
  unfold consumeStream
  rw [DNode.relabel_id, State.inEdges_relabel, List.find?_map]
  have : ((fun e : DEdge => decide (e.dstConn = some "IN_stream")) ∘ DEdge.relabel φ) =
      fun e => decide (e.dstConn = some "IN_stream") := rfl
  rw [this]
  cases (s.inEdges n.id).find? (fun e => decide (e.dstConn = some "IN_stream")) with
  | none => rfl
  | some e =>
    simp only [Option.map_some, State.srcEndpoint_relabel]
    cases s.srcEndpoint e <;> rfl

theorem consumeSetup_relabel (s : State) (n : DNode) (c : ConsumeNode) (st : St) :
    consumeSetup ctx ρ (s.relabel φ) (n.relabel φ) c (st.relabel φ) =
      (consumeSetup ctx ρ s n c st).map (relabelSetup φ) := by
  unfold consumeSetup
  rw [bindDynamicInputs_relabel]
  refine Res.bind_shift _ fun ⟨ρ', st'⟩ => ?_
  simp only [consumeStream_relabel]
  exact Res.bind_shift' _ fun l => Res.bind_shift' _ fun np => rfl

theorem chunkMemName_relabel (s : State) (entry : NodeId) :
    chunkMemName ctx (s.relabel φ) (φ entry) = chunkMemName ctx s entry := by
  simp [chunkMemName, State.consumedStream?_relabel]

theorem taskStart_relabel (s : State) (entry : NodeId) (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : St)
    (q : List Val) :
    taskStart ctx (s.relabel φ) (φ entry) c l priv (pre.relabel φ) q = (taskStart ctx s entry c l priv pre q).relabel φ := by
  simp only [taskStart, chunkMemName_relabel]
  rfl

theorem consumeFinish_relabel (pre : St) (l : Loc) (logs : List Log) (rest pushed : List Val) :
    consumeFinish ctx (pre.relabel φ) l logs rest pushed = (consumeFinish ctx pre l logs rest pushed).map (St.relabel φ) := by
  unfold consumeFinish
  simp only [St.relabel_mem]
  split <;> rfl

@[simp] theorem iterStart_relabel (priv : List Name) (pre : St) :
    iterStart priv (pre.relabel φ) = (iterStart priv pre).relabel φ := rfl

@[simp] theorem taskLog_relabel (st : St) (priv : List Name) (l : Loc) : taskLog (st.relabel φ) priv l = taskLog st priv l :=
  rfl

@[simp] theorem taskPushed_relabel (st : St) (priv : List Name) (l : Loc) :
    taskPushed (st.relabel φ) priv l = taskPushed st priv l := rfl

theorem nestedBindings_relabel (s : State) (n : DNode) :
    nestedBindings ctx (s.relabel φ) (n.relabel φ) = nestedBindings ctx s n := by
  unfold nestedBindings
  rw [State.nestedConns_relabel]
  congr 1
  funext acc c
  rw [State.nestedEndpoint_relabel]
  cases nestedEndpoint s n c <;> rfl

theorem bindCodeInputs_relabel (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (st : St) :
    bindCodeInputs ctx' ρ' (s.relabel φ) (n.relabel φ) (st.relabel φ) =
      (bindCodeInputs ctx' ρ' s n st).map (Prod.map (St.relabel φ) id) := by
  unfold bindCodeInputs
  rw [State.nestedConns_relabel]
  refine List.foldlM_res_comm₂ (Prod.map (St.relabel φ) id) (fun ⟨st, codes⟩ c => ?_) _ (st, [])
  simp only [Prod.map_apply, id_eq, State.nestedEndpoint_relabel]
  cases nestedEndpoint s n c with
  | code key =>
    simp only [State.Endpoint.relabel_code]
    rw [St.relabel_val?]
    cases st.val? key with
    | some v => exact Res.bind_shift' _ fun l => rfl
    | none => rfl
  | _ => rfl

/-- Relabel the state of a nested set-up result. -/
abbrev relabelNested (φ : Equiv.Perm NodeId) : Ctx × Valuation × St × List Name → Ctx × Valuation × St × List Name :=
  Prod.map id (Prod.map id (Prod.map (St.relabel φ) id))

theorem nestedSetup_relabel (s : State) (n : DNode) (hn : φ n.id = n.id) (inner : SDFG) (mapping : List (Sym × Expr))
    (st : St) :
    nestedSetup ctx ρ (s.relabel φ) (n.relabel φ) inner mapping (st.relabel φ) =
      (nestedSetup ctx ρ s n inner mapping st).map (relabelNested φ) := by
  unfold nestedSetup
  rw [nestedBindings_relabel, DNode.relabel_id, hn]
  refine Res.bind_shift' _ fun conns => ?_
  simp only [St.relabel_mem]
  refine Res.bind_shift' _ fun ρ' => ?_
  have : ({ st.relabel φ with mem := st.mem.allocAll (inner.callTransients.map (ctx.enter inner n.id conns).cname) } : St) =
      St.relabel φ { st with mem := st.mem.allocAll (inner.callTransients.map (ctx.enter inner n.id conns).cname) } := rfl
  rw [this, bindCodeInputs_relabel]
  exact Res.bind_shift _ fun ⟨st', codes⟩ => rfl

theorem nestedFinish_relabel (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (aliased : List Name) (pre : St)
    (m₂ : Mem) (g₂ : Log) :
    nestedFinish ctx' ρ' (s.relabel φ) (n.relabel φ) aliased (pre.relabel φ) m₂ g₂ =
      (nestedFinish ctx' ρ' s n aliased pre m₂ g₂).map (St.relabel φ) := by
  unfold nestedFinish
  dsimp only
  rw [DNode.relabel_id, State.outEdges_relabel, List.foldlM_map,
    show ({ mem := m₂, log := (pre.relabel φ).log.seq (g₂.dropNames aliased), vals := (pre.relabel φ).vals } : St) =
      St.relabel φ { mem := m₂, log := pre.log.seq (g₂.dropNames aliased), vals := pre.vals } from rfl]
  refine List.foldlM_res_comm₂ (St.relabel φ) (fun st e => ?_) _ _
  simp only [DEdge.relabel_data, DEdge.relabel_srcConn, State.dstEndpoint_relabel]
  split
  · cases s.dstEndpoint e with
    | code key => exact Res.bind_shift' _ fun l => by split <;> rfl
    | _ => rfl
  · rfl

end Leaf

/-! ## Execution relations -/

/-- `φ` fixes the ids of the nested SDFG nodes of `s` (their memory-name
prefixes, `Ctx.enter`). -/
def State.FixesNested (φ : Equiv.Perm NodeId) (s : State) : Prop := ∀ n ∈ s.nodes, n.isNested = true → φ n.id = n.id

theorem ResSim.of_map {α : Type} (g : α → α) (r : Res α) : ResSim (fun a₂ a => a₂ = g a) (r.map g) r := by
  cases r with
  | ok a => exact ResSim.ok rfl
  | err e => exact ResSim.err e

theorem List.filter_contains_map_perm (φ : Equiv.Perm NodeId) (order direct : List NodeId) :
    (order.map φ).filter (direct.map φ).contains = (order.filter direct.contains).map φ := by
  rw [List.filter_map]
  congr 1
  apply List.filter_congr
  intro x _
  simp [Function.comp]

section Relations

variable {φ} {ctx : Ctx}

/-- Relabel a sequential result: state and trace. -/
abbrev relabelSeq (φ : Equiv.Perm NodeId) (x : St × List (NodeId × Log)) : St × List (NodeId × Log) :=
  (x.1.relabel φ, State.relabelTrace φ x.2)

mutual

theorem NodeExec.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {n : DNode} (hn : n ∈ s.nodes)
    {st : St} {r : Res St} :
    NodeExec ctx ρ s n st r →
      ∃ r₂, NodeExec ctx ρ (s.relabel φ) (n.relabel φ) (st.relabel φ) r₂ ∧ ResSim (fun a₂ a => a₂ = a.relabel φ) r₂ r
  | .simple hs => ⟨_, .simple (by rw [DNode.relabel_isSimple]; exact hs), by
      rw [execSimple_relabel]; exact ResSim.of_map _ _⟩
  | .mapSetupErr hk he => ⟨_, .mapSetupErr (by rw [DNode.relabel_kind]; exact hk)
      (by rw [mapSetup_relabel, he]; rfl), ResSim.err _⟩
  | .mapIterErr (st₁ := st₁) hk hs hi => by
    obtain ⟨r₂, hi₂, hr⟩ := IterExec.relabel hfix hi
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .mapIterErr (st₁ := st₁.relabel φ) (by rw [DNode.relabel_kind]; exact hk)
      (by rw [mapSetup_relabel, hs]; rfl) ?_, ResSim.of_err hu⟩
    rw [DNode.relabel_id, Ctx.privateTransients_relabel]
    exact hi₂
  | .mapScope (st₁ := st₁) (logs := logs) (mp := mp) (ρ' := ρ') (pts := pts) hk hs hi => by
    obtain ⟨r₂, hi₂, hr⟩ := IterExec.relabel hfix hi
    have hk' : (n.relabel φ).kind = .mapEntry mp := by rw [DNode.relabel_kind]; exact hk
    have hs' : mapSetup ctx ρ (s.relabel φ) (n.relabel φ) mp (st.relabel φ) = .ok (ρ', st₁.relabel φ, pts) := by
      rw [mapSetup_relabel, hs]; rfl
    rw [← Ctx.privateTransients_relabel (φ := φ), ← DNode.relabel_id] at hi₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs₂, rfl, rfl⟩
    · exact ⟨_, .mapIterErr hk' hs' hi₂, ResSim.ub hu _⟩
    · refine ⟨_, .mapScope hk' hs' hi₂, ?_⟩
      rw [scopeFinish_relabel]
      exact ResSim.of_map _ _
  | .consumeSetupErr hk he => ⟨_, .consumeSetupErr (by rw [DNode.relabel_kind]; exact hk)
      (by rw [consumeSetup_relabel, he]; rfl), ResSim.err _⟩
  | .consumeTasksErr (st₁ := st₁) hk hs ht => by
    obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.relabel hfix ht
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .consumeTasksErr (st₁ := st₁.relabel φ) (by rw [DNode.relabel_kind]; exact hk)
      (by rw [consumeSetup_relabel, hs]; rfl) ?_, ResSim.of_err hu⟩
    rw [DNode.relabel_id, Ctx.privateTransients_relabel]
    exact ht₂
  | .consumeScope (st₁ := st₁) (c := c) (ρ' := ρ') (l := l) (np := np) hk hs ht => by
    obtain ⟨r₂, ht₂, hr⟩ := ConsumeTasks.relabel hfix ht
    have hk' : (n.relabel φ).kind = .consumeEntry c := by rw [DNode.relabel_kind]; exact hk
    have hs' : consumeSetup ctx ρ (s.relabel φ) (n.relabel φ) c (st.relabel φ) = .ok (ρ', st₁.relabel φ, l, np) := by
      rw [consumeSetup_relabel, hs]; rfl
    rw [← Ctx.privateTransients_relabel (φ := φ), ← DNode.relabel_id] at ht₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨x, rfl, rfl⟩
    · exact ⟨_, .consumeTasksErr hk' hs' ht₂, ResSim.ub hu _⟩
    · refine ⟨_, .consumeScope hk' hs' ht₂, ?_⟩
      rw [consumeFinish_relabel]
      exact ResSim.of_map _ _
  | .nestedSetupErr hk he => by
    have hfx := hfix n hn (by simp [DNode.isNested, hk])
    exact ⟨_, .nestedSetupErr (by rw [DNode.relabel_kind]; exact hk)
      (by rw [nestedSetup_relabel _ _ _ _ hfx, he]; rfl), ResSim.err _⟩
  | .nestedInnerErr (st₁ := st₁) hk hs hr => by
    have hfx := hfix n hn (by simp [DNode.isNested, hk])
    exact ⟨_, .nestedInnerErr (st₁ := st₁.relabel φ) (by rw [DNode.relabel_kind]; exact hk)
      (by rw [nestedSetup_relabel _ _ _ _ hfx, hs]; rfl) hr, ResSim.err _⟩
  | .nested (st₁ := st₁) hk hs hr => by
    have hfx := hfix n hn (by simp [DNode.isNested, hk])
    refine ⟨_, .nested (st₁ := st₁.relabel φ) (by rw [DNode.relabel_kind]; exact hk)
      (by rw [nestedSetup_relabel _ _ _ _ hfx, hs]; rfl) hr, ?_⟩
    rw [nestedFinish_relabel]
    exact ResSim.of_map _ _

theorem SeqExec.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {ids : List NodeId} {st : St}
    {r : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s ids st r →
      ∃ r₂, SeqExec ctx ρ (s.relabel φ) (ids.map φ) (st.relabel φ) r₂ ∧
        ResSim (fun a₂ a => a₂ = relabelSeq φ a) r₂ r
  | .nil => ⟨_, .nil, ResSim.ok rfl⟩
  | .missing hm => ⟨_, .missing (by rw [State.node?_relabel, hm]; rfl), ResSim.err _⟩
  | .headErr hn hx => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.relabel hfix (State.node?_mem hn) hx
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    exact ⟨_, .headErr (by rw [State.node?_relabel, hn]; rfl) hx₂, ResSim.of_err hu⟩
  | .tailErr (id := id) (n := n) hn hx hrest => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.relabel hfix (State.node?_mem hn) hx
    have hn' : (s.relabel φ).node? (φ id) = some (n.relabel φ) := by rw [State.node?_relabel, hn]; rfl
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hn' hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.relabel hfix hrest
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hn' hx₂ hrest₂, ResSim.of_err hu⟩
  | .cons (id := id) (n := n) hn hx hrest => by
    obtain ⟨r₂, hx₂, hr⟩ := NodeExec.relabel hfix (State.node?_mem hn) hx
    have hn' : (s.relabel φ).node? (φ id) = some (n.relabel φ) := by rw [State.node?_relabel, hn]; rfl
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hn' hx₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hrest₂, hr₃⟩ := SeqExec.relabel hfix hrest
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
      · exact ⟨_, .tailErr hn' hx₂ hrest₂, ResSim.ub hu _⟩
      · exact ⟨_, .cons hn' hx₂ hrest₂, ResSim.ok rfl⟩

theorem OrderedExec.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {all direct : List NodeId}
    {st : St} {r : Res St} :
    OrderedExec ctx ρ s all direct st r →
      ∃ r₂, OrderedExec ctx ρ (s.relabel φ) (all.map φ) (direct.map φ) (st.relabel φ) r₂ ∧
        ResSim (fun a₂ a => a₂ = a.relabel φ) r₂ r
  | .prologueErr hp => ⟨_, .prologueErr (by rw [State.viewOrder_relabel, bindViews_relabel, hp]; rfl),
      ResSim.err _⟩
  | .seqErr (order := order) (st₁ := st₁) ho hp hs => by
    obtain ⟨r₂, hs₂, hr⟩ := SeqExec.relabel hfix hs
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .seqErr (order := order.map φ) (st₁ := st₁.relabel φ)
      (by rw [State.isTopologicalOrder_relabel]; exact ho)
      (by rw [State.viewOrder_relabel, bindViews_relabel, hp]; rfl) ?_, ResSim.of_err hu⟩
    rw [List.filter_contains_map_perm]
    exact hs₂
  | .hbRace (order := order) (st₁ := st₁) ho hp hs hf => by
    obtain ⟨r₂, hs₂, hr⟩ := SeqExec.relabel hfix hs
    have ho' := (State.isTopologicalOrder_relabel (φ := φ) s _ order).trans ho
    have hp' : bindViews ctx ρ (s.relabel φ) ((s.relabel φ).viewOrder ctx (direct.map φ)) (st.relabel φ) =
        .ok (st₁.relabel φ) := by rw [State.viewOrder_relabel, bindViews_relabel, hp]; rfl
    rw [← List.filter_contains_map_perm] at hs₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
    · exact ⟨_, .seqErr ho' hp' hs₂, ResSim.ub hu _⟩
    · exact ⟨_, .hbRace ho' hp' hs₂ (by rw [State.hbFree_relabel]; exact hf), ResSim.ub rfl _⟩
  | .run (order := order) (st₁ := st₁) ho hp hs hf => by
    obtain ⟨r₂, hs₂, hr⟩ := SeqExec.relabel hfix hs
    have ho' := (State.isTopologicalOrder_relabel (φ := φ) s _ order).trans ho
    have hp' : bindViews ctx ρ (s.relabel φ) ((s.relabel φ).viewOrder ctx (direct.map φ)) (st.relabel φ) =
        .ok (st₁.relabel φ) := by rw [State.viewOrder_relabel, bindViews_relabel, hp]; rfl
    rw [← List.filter_contains_map_perm] at hs₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
    · exact ⟨_, .seqErr ho' hp' hs₂, ResSim.ub hu _⟩
    · exact ⟨_, .run ho' hp' hs₂ (by rw [State.hbFree_relabel]; exact hf), ResSim.ok rfl⟩

theorem IterExec.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {entry : NodeId} {params : List Sym}
    {priv : List Name} {pre : St} {pts : List Idx} {r : Res (List Log)} :
    IterExec ctx ρ s entry params priv pre pts r →
      ∃ r₂, IterExec ctx ρ (s.relabel φ) (φ entry) params priv (pre.relabel φ) pts r₂ ∧ ResSim Eq r₂ r
  | .nil => ⟨_, .nil, ResSim.ok rfl⟩
  | .headErr hb => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .headErr ?_, ResSim.of_err hu⟩
    rw [State.scopeNodes_relabel, State.directChildren_some_relabel, iterStart_relabel]
    exact hb₂
  | .tailErr hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    rw [← State.scopeNodes_relabel, ← State.directChildren_some_relabel, ← iterStart_relabel] at hb₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := IterExec.relabel hfix hrest
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hb₂ hr₂, ResSim.of_err hu⟩
  | .cons hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    rw [← State.scopeNodes_relabel, ← State.directChildren_some_relabel, ← iterStart_relabel] at hb₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := IterExec.relabel hfix hrest
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
      · exact ⟨_, .tailErr hb₂ hr₂, ResSim.ub hu _⟩
      · exact ⟨_, .cons hb₂ hr₂, ResSim.ok rfl⟩

theorem ConsumeTasks.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {entry : NodeId} {c : ConsumeNode}
    {l : Loc} {np : Int} {priv : List Name} {pre : St} {q : List Val} {r : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx ρ s entry c l np priv pre q r →
      ∃ r₂, ConsumeTasks ctx ρ (s.relabel φ) (φ entry) c l np priv (pre.relabel φ) q r₂ ∧ ResSim Eq r₂ r
  | .condErr hc => ⟨_, .condErr hc, ResSim.err _⟩
  | .stop hc => ⟨_, .stop hc, ResSim.ok rfl⟩
  | .bodyErr hc hq hp hb => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .bodyErr hc hq hp ?_, ResSim.of_err hu⟩
    rw [State.scopeNodes_relabel, State.directChildren_some_relabel, taskStart_relabel]
    exact hb₂
  | .restErr hc hq hp hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    rw [← State.scopeNodes_relabel, ← State.directChildren_some_relabel, ← taskStart_relabel] at hb₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .bodyErr hc hq hp hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := ConsumeTasks.relabel hfix hrest
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .restErr hc hq hp hb₂ hr₂, ResSim.of_err hu⟩
  | .step hc hq hp hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := OrderedExec.relabel hfix hb
    rw [← State.scopeNodes_relabel, ← State.directChildren_some_relabel, ← taskStart_relabel] at hb₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .bodyErr hc hq hp hb₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := ConsumeTasks.relabel hfix hrest
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
      · exact ⟨_, .restErr hc hq hp hb₂ hr₂, ResSim.ub hu _⟩
      · exact ⟨_, .step hc hq hp hb₂ hr₂, ResSim.ok rfl⟩

end

theorem CompsExec.relabel {ρ : Valuation} {s : State} (hfix : s.FixesNested φ) {m : Mem} :
    ∀ {comps : List (List NodeId)} {r : Res (List Log)}, CompsExec ctx ρ s m comps r →
      ∃ r₂, CompsExec ctx ρ (s.relabel φ) m (comps.map (List.map φ)) r₂ ∧ ResSim Eq r₂ r
  | _, _, .nil => ⟨_, .nil, ResSim.ok rfl⟩
  | _, _, .headErr hc => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.relabel hfix hc
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .headErr ?_, ResSim.of_err hu⟩
    rw [State.topLevelOf_relabel]
    exact hc₂
  | _, _, .tailErr hc hrest => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.relabel hfix hc
    rw [← State.topLevelOf_relabel] at hc₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hc₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := CompsExec.relabel hfix hrest
      obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr₃
      exact ⟨_, .tailErr hc₂ hr₂, ResSim.of_err hu⟩
  | _, _, .cons hc hrest => by
    obtain ⟨r₂, hc₂, hr⟩ := OrderedExec.relabel hfix hc
    rw [← State.topLevelOf_relabel] at hc₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨a₂, rfl, rfl⟩
    · exact ⟨_, .headErr hc₂, ResSim.ub hu _⟩
    · obtain ⟨r₃, hr₂, hr₃⟩ := CompsExec.relabel hfix hrest
      rcases ResSim.ok_cases hr₃ with ⟨e, rfl, hu⟩ | ⟨b₂, rfl, rfl⟩
      · exact ⟨_, .tailErr hc₂ hr₂, ResSim.ub hu _⟩
      · exact ⟨_, .cons hc₂ hr₂, ResSim.ok rfl⟩

/-- **A relabeled state runs like the state**: every run of `s` is matched by a
run of `s.relabel φ` from the same control state with the same outcome, or the
relabeled run is undefined behavior (a race whose witness names the
relabeled nodes). -/
theorem StateExec.relabel {cst : CSt} {s : State} (hfix : s.FixesNested φ) {r : Res (Ctl × CSt)}
    (h : StateExec ctx cst s r) : ∃ r₂, StateExec ctx cst (s.relabel φ) r₂ ∧ ResSim Eq r₂ r := by
  cases h with
  | compsErr hc =>
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.relabel hfix hc
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    refine ⟨_, .compsErr ?_, ResSim.of_err hu⟩
    rw [State.components_relabel, Ctx.stateTransients_relabel]
    exact hc₂
  | run hc =>
    obtain ⟨r₂, hc₂, hr⟩ := CompsExec.relabel hfix hc
    rw [← Ctx.stateTransients_relabel (φ := φ), ← State.components_relabel] at hc₂
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨logs, rfl, rfl⟩
    · exact ⟨_, .compsErr hc₂, ResSim.ub hu _⟩
    · exact ⟨_, .run hc₂, ResSim.of_eq fun _ => rfl⟩

theorem State.FixesNested.relabel {s : State} (h : s.FixesNested φ) : (s.relabel φ).FixesNested φ.symm := by
  intro n hn hnest
  simp only [State.relabel_nodes, List.mem_map] at hn
  obtain ⟨n₀, hn₀, rfl⟩ := hn
  have := h n₀ hn₀ hnest
  simp only [DNode.relabel_id, this]
  exact φ.symm_apply_eq.2 this.symm

/-- The converse direction: every run of the relabeled state is matched by a
run of `s`. -/
theorem StateExec.of_relabel {cst : CSt} {s : State} (hfix : s.FixesNested φ) {r : Res (Ctl × CSt)}
    (h : StateExec ctx cst (s.relabel φ) r) : ∃ r₂, StateExec ctx cst s r₂ ∧ ResSim Eq r₂ r := by
  have := StateExec.relabel (φ := φ.symm) hfix.relabel h
  rwa [State.relabel_symm] at this

end Relations

end Dace
