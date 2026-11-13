-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Runs.Invariance.Relabel
import Dace.SDFG.Names
import Dace.Runs.Invariance.EvalCongr
import Dace.Runs.Invariance.TreeRename

/-!
# Renaming containers

`SDFG.ren σ` renames the containers of an SDFG by a permutation `σ` of
names: the keys of its descriptors, the data of its access nodes and the
data of its memlets.  Expressions, connectors and nested SDFGs (their own
namespace) are left as they are, so the renaming is only meaningful when
`σ` fixes every name read by an expression (`Rename.lean` of
`Dace.Equiv`; a bare name may denote a scalar container, `Decisions.md#D33`,
and symbols shadow containers).

This file proves that every structural function of a state commutes with
the renaming (`State.*_ren`).  Node ids are unchanged; edges and endpoints
are renamed.
-/

set_option autoImplicit false

namespace Dace

variable (σ : Equiv.Perm Name)

/-! ## Renaming syntax -/

def EdgeData.ren : EdgeData → EdgeData
  | .memlet m => .memlet (m.ren σ)
  | .dep => .dep

def DEdge.ren (e : DEdge) : DEdge := { e with data := e.data.ren σ }

def NodeKind.ren : NodeKind → NodeKind
  | .access d => .access (σ d)
  | k => k

def DNode.ren (n : DNode) : DNode := { n with kind := n.kind.ren σ }

def State.ren (s : State) : State := { s with nodes := s.nodes.map (DNode.ren σ), edges := s.edges.map (DEdge.ren σ) }

def State.Endpoint.ren : State.Endpoint → State.Endpoint
  | .data d sub => .data (σ d) sub
  | e => e

mutual

def LoopRegion.ren : LoopRegion → LoopRegion
  | ⟨l, v, i, c, u, inv, ubc, body⟩ => ⟨l, v, i, c, u, inv, ubc, Region.ren body⟩

def Block.ren : Block → Block
  | .state s => .state (s.ren σ)
  | .region r => .region (Region.ren r)
  | .loop l => .loop (LoopRegion.ren l)
  | .cond lb brs => .cond lb (Branches.ren brs)
  | b => b

def Blocks.ren : List Block → List Block
  | [] => []
  | b :: bs => Block.ren b :: Blocks.ren bs

def Branches.ren : List (Option Expr × Region) → List (Option Expr × Region)
  | [] => []
  | (c, r) :: rest => (c, Region.ren r) :: Branches.ren rest

def Region.ren : Region → Region
  | ⟨l, k, blocks, start, edges⟩ => ⟨l, k, Blocks.ren blocks, start, edges⟩

end

def SDFG.ren (g : SDFG) : SDFG :=
  { g with arrays := g.arrays.map (fun (n, d) => (σ n, d)), top := g.top.ren σ }

/-! ## Basic facts -/

namespace Memlet
variable {σ}
@[simp] theorem ren_srcSubset (m : Memlet) : (m.ren σ).srcSubset = m.srcSubset := rfl
@[simp] theorem ren_dstSubset (m : Memlet) : (m.ren σ).dstSubset = m.dstSubset := rfl
theorem ren_symm (m : Memlet) : (m.ren σ).ren σ.symm = m := by cases m; simp [Memlet.ren]
end Memlet

namespace EdgeData
variable {σ}
@[simp] theorem ren_memlet (m : Memlet) : (EdgeData.memlet m).ren σ = .memlet (m.ren σ) := rfl
@[simp] theorem ren_dep : EdgeData.dep.ren σ = .dep := rfl
@[simp] theorem ren_memlet? (d : EdgeData) : (d.ren σ).memlet? = d.memlet?.map (Memlet.ren σ) := by cases d <;> rfl
theorem ren_symm (d : EdgeData) : (d.ren σ).ren σ.symm = d := by cases d <;> simp [EdgeData.ren, Memlet.ren_symm]
end EdgeData

namespace DEdge
variable {σ}
@[simp] theorem ren_src (e : DEdge) : (e.ren σ).src = e.src := rfl
@[simp] theorem ren_dst (e : DEdge) : (e.ren σ).dst = e.dst := rfl
@[simp] theorem ren_srcConn (e : DEdge) : (e.ren σ).srcConn = e.srcConn := rfl
@[simp] theorem ren_dstConn (e : DEdge) : (e.ren σ).dstConn = e.dstConn := rfl
@[simp] theorem ren_data (e : DEdge) : (e.ren σ).data = e.data.ren σ := rfl
@[simp] theorem ren_allowOob (e : DEdge) : (e.ren σ).allowOob = e.allowOob := by
  simp [DEdge.allowOob, EdgeData.ren_memlet?, Option.any_map]
theorem ren_symm (e : DEdge) : (e.ren σ).ren σ.symm = e := by cases e; simp [DEdge.ren, EdgeData.ren_symm]
theorem ren_injective : Function.Injective (DEdge.ren σ) := fun a b h => by
  rw [← ren_symm a, h, ren_symm]
end DEdge

namespace DNode
variable {σ}
@[simp] theorem ren_id (n : DNode) : (n.ren σ).id = n.id := rfl
@[simp] theorem ren_kind (n : DNode) : (n.ren σ).kind = n.kind.ren σ := rfl
@[simp] theorem ren_label (n : DNode) : (n.ren σ).label = n.label := rfl
@[simp] theorem ren_scopeEntry (n : DNode) : (n.ren σ).scopeEntry = n.scopeEntry := rfl
@[simp] theorem ren_isEntry (n : DNode) : (n.ren σ).isEntry = n.isEntry := by
  cases n; rename_i k; cases k <;> rfl
@[simp] theorem ren_isExit (n : DNode) : (n.ren σ).isExit = n.isExit := by
  cases n; rename_i k; cases k <;> rfl
@[simp] theorem ren_isAccess (n : DNode) : (n.ren σ).isAccess = n.isAccess := by
  cases n; rename_i k; cases k <;> rfl
@[simp] theorem ren_isSimple (n : DNode) : (n.ren σ).isSimple = n.isSimple := by
  cases n; rename_i k; cases k <;> rfl
@[simp] theorem ren_accessData? (n : DNode) : (n.ren σ).accessData? = n.accessData?.map σ := by
  cases n; rename_i k; cases k <;> rfl
theorem ren_symm (n : DNode) : (n.ren σ).ren σ.symm = n := by
  rcases n with ⟨id, label, ic, oc, se, kind⟩
  cases kind <;> simp [DNode.ren, NodeKind.ren]
end DNode

namespace State.Endpoint
variable {σ}
@[simp] theorem ren_data' (n : Name) (sub : Option Subset) : (State.Endpoint.data n sub).ren σ = .data (σ n) sub := rfl
@[simp] theorem ren_invalid : State.Endpoint.invalid.ren σ = .invalid := rfl
@[simp] theorem ren_code (k : NodeId × Conn) : (State.Endpoint.code k).ren σ = .code k := rfl
end State.Endpoint

/-! ## Structural functions of a state -/

namespace State
variable {σ}

@[simp] theorem ren_nodes (s : State) : (s.ren σ).nodes = s.nodes.map (DNode.ren σ) := rfl
@[simp] theorem ren_edges (s : State) : (s.ren σ).edges = s.edges.map (DEdge.ren σ) := rfl
@[simp] theorem ren_label (s : State) : (s.ren σ).label = s.label := rfl

theorem ren_symm (s : State) : (s.ren σ).ren σ.symm = s := by
  rcases s with ⟨label, nodes, edges, nosync⟩
  simp only [State.ren, List.map_map, State.mk.injEq, true_and, and_true]
  constructor
  · conv => rhs; rw [← List.map_id nodes]
    exact List.map_congr_left fun n _ => DNode.ren_symm n
  · conv => rhs; rw [← List.map_id edges]
    exact List.map_congr_left fun e _ => DEdge.ren_symm e

theorem node?_ren (s : State) (id : NodeId) : (s.ren σ).node? id = (s.node? id).map (DNode.ren σ) := by
  simp only [node?, ren_nodes, List.find?_map]
  rfl

theorem inEdges_ren (s : State) (id : NodeId) : (s.ren σ).inEdges id = (s.inEdges id).map (DEdge.ren σ) := by
  simp only [inEdges, ren_edges, List.filter_map]
  rfl

theorem outEdges_ren (s : State) (id : NodeId) : (s.ren σ).outEdges id = (s.outEdges id).map (DEdge.ren σ) := by
  simp only [outEdges, ren_edges, List.filter_map]
  rfl

theorem isScopeNode_ren (s : State) (id : NodeId) : (s.ren σ).isScopeNode id = s.isScopeNode id := by
  simp only [isScopeNode, node?_ren]
  cases s.node? id <;> simp

theorem isEntryNode_ren (s : State) (id : NodeId) : (s.ren σ).isEntryNode id = s.isEntryNode id := by
  simp only [isEntryNode, node?_ren]
  cases s.node? id <;> simp

theorem isExitNode_ren (s : State) (id : NodeId) : (s.ren σ).isExitNode id = s.isExitNode id := by
  simp only [isExitNode, node?_ren]
  cases s.node? id <;> simp

theorem isAccessNode_ren (s : State) (id : NodeId) : (s.ren σ).isAccessNode id = s.isAccessNode id := by
  simp only [isAccessNode, node?_ren]
  cases s.node? id <;> simp

theorem isConsumeEntry_ren (s : State) (id : NodeId) : (s.ren σ).isConsumeEntry id = s.isConsumeEntry id := by
  simp only [isConsumeEntry, node?_ren]
  cases s.node? id with
  | none => rfl
  | some n => rcases n with ⟨_, _, _, _, _, k⟩; cases k <;> rfl

theorem prevEdges_ren (s : State) (e : DEdge) : (s.ren σ).prevEdges (e.ren σ) = (s.prevEdges e).map (DEdge.ren σ) := by
  simp only [prevEdges, DEdge.ren_src, isScopeNode_ren, DEdge.ren_srcConn]
  split
  · split
    · rw [inEdges_ren, List.filter_map]
      rfl
    · rfl
  · rfl

theorem nextEdges_ren (s : State) (e : DEdge) : (s.ren σ).nextEdges (e.ren σ) = (s.nextEdges e).map (DEdge.ren σ) := by
  simp only [nextEdges, DEdge.ren_dst, isScopeNode_ren, DEdge.ren_dstConn]
  split
  · split
    · rw [outEdges_ren, List.filter_map]
      rfl
    · rfl
  · rfl

theorem isRouting_ren (s : State) (e : DEdge) : (s.ren σ).isRouting (e.ren σ) = s.isRouting e := by
  simp [isRouting, isEntryNode_ren, isExitNode_ren]

theorem isDynamicInput_ren (s : State) (e : DEdge) : (s.ren σ).isDynamicInput (e.ren σ) = s.isDynamicInput e := by
  simp [isDynamicInput, isEntryNode_ren]

theorem walkPrev_ren (s : State) : ∀ (k : Nat) (e : DEdge),
    (s.ren σ).walkPrev k (e.ren σ) = (s.walkPrev k e).ren σ
  | 0, e => rfl
  | k + 1, e => by
    simp only [walkPrev, DEdge.ren_src, isScopeNode_ren, DEdge.ren_srcConn, prevEdges_ren]
    split
    · cases s.prevEdges e with
      | nil => rfl
      | cons f _ => exact walkPrev_ren s k f
    · rfl

theorem walkNext_ren (s : State) (oe : Bool) : ∀ (k : Nat) (e : DEdge),
    (s.ren σ).walkNext oe k (e.ren σ) = (s.walkNext oe k e).ren σ
  | 0, e => rfl
  | k + 1, e => by
    simp only [walkNext, DEdge.ren_dst, DEdge.ren_dstConn, nextEdges_ren, isScopeNode_ren, isExitNode_ren]
    by_cases hb : ((if oe = true then s.isExitNode e.dst else s.isScopeNode e.dst) && connPrefix e.dstConn "IN_") = true
    · simp only [hb, ↓reduceIte]
      cases s.nextEdges e with
      | nil => rfl
      | cons f _ => exact walkNext_ren s oe k f
    · simp only [hb, ↓reduceIte, Bool.false_eq_true]

theorem pathStart_ren (s : State) (e : DEdge) : (s.ren σ).pathStart (e.ren σ) = (s.pathStart e).ren σ := by
  simp only [pathStart, ren_edges, List.length_map]
  exact walkPrev_ren s _ e

theorem pathEnd_ren (s : State) (e : DEdge) : (s.ren σ).pathEnd (e.ren σ) = (s.pathEnd e).ren σ := by
  simp only [pathEnd, ren_edges, List.length_map]
  exact walkNext_ren s _ _ e

theorem dstRootEdge_ren (s : State) (e : DEdge) : (s.ren σ).dstRootEdge (e.ren σ) = (s.dstRootEdge e).ren σ := by
  simp only [dstRootEdge, ren_edges, List.length_map]
  exact walkNext_ren s _ _ e

theorem chunkSource_ren (s : State) : ∀ (k : Nat) (e : DEdge),
    (s.ren σ).chunkSource k (e.ren σ) = s.chunkSource k e
  | 0, e => rfl
  | k + 1, e => by
    show (if ((s.ren σ).isScopeNode e.src && connPrefix e.srcConn "OUT_") = true then
        if ((s.ren σ).isConsumeEntry e.src && decide (e.srcConn = some "OUT_stream")) = true then some e.src
        else
          match (s.ren σ).prevEdges (e.ren σ) with
          | f :: _ => (s.ren σ).chunkSource k f
          | [] => none
      else none) = _
    simp only [chunkSource, isScopeNode_ren, isConsumeEntry_ren, prevEdges_ren]
    by_cases h₁ : (s.isScopeNode e.src && connPrefix e.srcConn "OUT_") = true
    · by_cases h₂ : (s.isConsumeEntry e.src && decide (e.srcConn = some "OUT_stream")) = true
      · simp [h₁, h₂]
      · cases hp : s.prevEdges e with
        | nil => simp [h₁, h₂]
        | cons f _ => simp [h₁, h₂, chunkSource_ren s k f]
    · simp [h₁]

theorem consumedStream?_ren (s : State) (entry : NodeId) :
    (s.ren σ).consumedStream? entry = (s.consumedStream? entry).map σ := by
  simp only [consumedStream?, inEdges_ren, List.find?_map]
  have : ((fun e : DEdge => decide (e.dstConn = some "IN_stream")) ∘ DEdge.ren σ) =
      fun e => decide (e.dstConn = some "IN_stream") := rfl
  rw [this]
  cases (s.inEdges entry).find? (fun e => decide (e.dstConn = some "IN_stream")) with
  | none => rfl
  | some e =>
    simp only [Option.map_some, pathStart_ren, DEdge.ren_src, node?_ren]
    cases s.node? (s.pathStart e).src with
    | none => rfl
    | some n => simp [DNode.ren_accessData?]

theorem children_ren (s : State) (entry : Option NodeId) :
    (s.ren σ).children entry = (s.children entry).map (DNode.ren σ) := by
  simp only [children, ren_nodes, List.filter_map]
  rfl

theorem directChildren_ren (s : State) (entry : Option NodeId) :
    (s.ren σ).directChildren entry = s.directChildren entry := by
  simp [directChildren, children_ren, List.map_map, Function.comp_def]

theorem scopeNodesFuel_ren (s : State) : ∀ (k : Nat) (entry : NodeId),
    (s.ren σ).scopeNodesFuel k entry = s.scopeNodesFuel k entry
  | 0, _ => rfl
  | k + 1, entry => by
    simp only [scopeNodesFuel, children_ren, List.flatMap_map]
    apply List.flatMap_congr
    intro n _
    simp [scopeNodesFuel_ren s k n.id]

theorem scopeNodes_ren (s : State) (entry : NodeId) : (s.ren σ).scopeNodes entry = s.scopeNodes entry := by
  simp only [scopeNodes, ren_nodes, List.length_map]
  exact scopeNodesFuel_ren s _ entry

theorem topLevelOf_ren (s : State) (ids : List NodeId) : (s.ren σ).topLevelOf ids = s.topLevelOf ids := by
  simp only [topLevelOf]
  apply List.filter_congr
  intro id _
  simp only [node?_ren]
  cases s.node? id <;> rfl

theorem neighbors_ren (s : State) (id : NodeId) : (s.ren σ).neighbors id = s.neighbors id := by
  simp only [neighbors, ren_edges, List.filterMap_map]
  rfl

theorem component_ren (s : State) (id : NodeId) : (s.ren σ).component id = s.component id := by
  simp only [component, ren_edges, List.length_map]
  have : (s.ren σ).neighbors = s.neighbors := funext (neighbors_ren s)
  rw [this]

theorem componentsFrom_ren (s : State) : ∀ (seen : List NodeId) (ns : List DNode),
    (s.ren σ).componentsFrom seen (ns.map (DNode.ren σ)) = s.componentsFrom seen ns
  | _, [] => rfl
  | seen, n :: rest => by
    simp only [componentsFrom, List.map_cons, DNode.ren_id, component_ren]
    split
    · exact componentsFrom_ren s seen rest
    · rw [componentsFrom_ren s _ rest]

theorem components_ren (s : State) : (s.ren σ).components = s.components := by
  simp only [components, ren_nodes]
  exact componentsFrom_ren s [] s.nodes

theorem respectsEdges_ren (s : State) (order : List NodeId) : (s.ren σ).respectsEdges order = s.respectsEdges order := by
  simp only [respectsEdges, ren_edges, List.all_map]
  rfl

theorem isTopologicalOrder_ren (s : State) (ids order : List NodeId) :
    (s.ren σ).isTopologicalOrder ids order = s.isTopologicalOrder ids order := by
  simp only [isTopologicalOrder, respectsEdges_ren]

theorem succs_ren (s : State) (id : NodeId) : (s.ren σ).succs id = s.succs id := by
  simp [succs, outEdges_ren, List.map_map, Function.comp_def]

theorem reaches_ren (s : State) (a b : NodeId) : (s.ren σ).reaches a b = s.reaches a b := by
  have : (s.ren σ).succs = s.succs := funext (succs_ren s)
  simp only [reaches, this, ren_edges, List.length_map]

theorem hbConflict_ren (s : State) (x y : NodeId × Log) : (s.ren σ).hbConflict x y = s.hbConflict x y := by
  simp [hbConflict, reaches_ren]

theorem hbFree_sren (s : State) (tr : List (NodeId × Log)) : (s.ren σ).hbFree tr = s.hbFree tr := by
  have : (s.ren σ).hbConflict = s.hbConflict := funext fun x => funext (hbConflict_ren s x)
  induction tr with
  | nil => rfl
  | cons x xs ih => simp [hbFree, this, ih]

theorem hbWitness_sren (s : State) (tr : List (NodeId × Log)) : (s.ren σ).hbWitness tr = s.hbWitness tr := by
  have : (s.ren σ).hbConflict = s.hbConflict := funext fun x => funext (hbConflict_ren s x)
  induction tr with
  | nil => rfl
  | cons x xs ih => simp [hbWitness, this, ih]

end State

/-! ## Outcomes up to error messages

Error messages name containers, so a renamed run fails with another
message.  Outcomes are compared after erasing errors to their class
(undefined behavior or not, `ErrReason.cls`). -/

/-- The class of an error: a fixed undefined-behavior error, or a fixed
non-UB error. -/
def ErrReason.cls (e : ErrReason) : ErrReason := if e.isUB then .ambiguousTransition 0 else .malformed ""

theorem ErrReason.isUB_cls (e : ErrReason) : e.cls.isUB = e.isUB := by
  unfold ErrReason.cls; split <;> simp_all [ErrReason.isUB]

namespace Res

/-- An outcome with its error erased to its class. -/
def er {α : Type} : Res α → Res α
  | ok a => ok a
  | err e => err e.cls

variable {α β : Type}

@[simp] theorem er_ok (a : α) : (ok a : Res α).er = ok a := rfl
@[simp] theorem er_err (e : ErrReason) : (err e : Res α).er = err e.cls := rfl

theorem er_bind (r : Res α) (f : α → Res β) : (r >>= f).er = (r.er >>= fun a => (f a).er) := by
  cases r <;> rfl

theorem er_pure (a : α) : (pure a : Res α).er = pure a := rfl

theorem er_bind_congr {r' r : Res α} {f' f : α → Res β} (h : r'.er = r.er) (hf : ∀ a, (f' a).er = (f a).er) :
    (r' >>= f').er = (r >>= f).er := by
  rw [er_bind, er_bind, h]
  congr 1
  funext a
  exact hf a

theorem er_map (f : α → β) (r : Res α) : (r.map f).er = r.er.map f := by cases r <;> rfl

theorem er_ofOption_congr {o : Option α} {e' e : ErrReason} (h : e'.isUB = e.isUB) :
    (Res.ofOption o e').er = (Res.ofOption o e).er := by
  cases o <;> simp [Res.ofOption, ErrReason.cls, h]

theorem er_err_congr {e' e : ErrReason} (h : e'.isUB = e.isUB) : (err e' : Res α).er = (err e).er := by
  simp [ErrReason.cls, h]

theorem er_foldlM_congr {γ : Type} {l : List γ} {f' f : α → γ → Res α}
    (hf : ∀ x ∈ l, ∀ b, (f' b x).er = (f b x).er) (b : α) : (l.foldlM f' b).er = (l.foldlM f b).er := by
  induction l generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    exact er_bind_congr (hf x List.mem_cons_self b) fun b' => ih (fun y hy => hf y (List.mem_cons_of_mem _ hy)) b'

end Res

/-! ## Expressions under interpretations agreeing on the names they read -/

namespace Expr

mutual

theorem eval_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) :
    (e : Expr) → (∀ d ∈ e.readData, I.readData d = I'.readData d) → (∀ x ∈ e.freeSyms, ρ x = ρ' x) →
      e.eval I ρ = e.eval I' ρ'
  | .lit (.int _), _, _ => rfl
  | .lit (.float _ _), _, _ => rfl
  | .lit (.bool _), _, _ => rfl
  | .sym x, _, hs => hs x (by simp [freeSyms])
  | .un op a, hr, hs => by
    simp only [eval]
    rw [eval_congr_rd happ hattr a hr hs]
  | .bin op a b, hr, hs => by
    simp only [eval]
    rw [eval_congr_rd happ hattr a (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_rd happ hattr b (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx]))]
  | .ite c t f, hr, hs => by
    simp only [eval]
    rw [eval_congr_rd happ hattr c (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_rd happ hattr t (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx])),
      eval_congr_rd happ hattr f (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx]))]
  | .app fn args, hr, hs => by
    simp only [eval]
    rw [evalList_congr_rd happ hattr args (fun d hd => hr d (by simp [readData, hd]))
      (fun x hx => hs x (by simp [freeSyms, hx])), happ]
  | .subscript d idx, hr, hs => by
    simp only [eval]
    rw [evalList_congr_rd happ hattr idx (fun d' hd => hr d' (by simp [readData, hd]))
      (fun x hx => hs x (by simp [freeSyms, hx])), hr d (by simp [readData])]
  | .attr b _, hr, hs => by
    simp only [eval]
    rw [eval_congr_rd happ hattr b hr hs, hattr]
  | .approx a o, hr, hs => by
    simp only [eval]
    exact eval_congr_rd happ hattr a (fun d hd => hr d (by simp [readData, hd])) (fun x hx => hs x (by simp [freeSyms, hx]))

theorem evalList_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) :
    (es : Exprs) → (∀ d ∈ readDataList es, I.readData d = I'.readData d) → (∀ x ∈ freeSymsList es, ρ x = ρ' x) →
      evalList I ρ es = evalList I' ρ' es
  | .nil, _, _ => rfl
  | .cons e es, hr, hs => by
    simp only [evalList]
    rw [eval_congr_rd happ hattr e (fun d hd => hr d (by simp [readDataList, hd]))
        (fun x hx => hs x (by simp [freeSymsList, hx])),
      evalList_congr_rd happ hattr es (fun d hd => hr d (by simp [readDataList, hd]))
        (fun x hx => hs x (by simp [freeSymsList, hx]))]

end

theorem evalInt_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) {e : Expr} (hr : ∀ d ∈ e.readData, I.readData d = I'.readData d)
    (hs : ∀ x ∈ e.freeSyms, ρ x = ρ' x) : e.evalInt I ρ = e.evalInt I' ρ' := by
  unfold evalInt; rw [eval_congr_rd happ hattr e hr hs]

theorem evalBool_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) {e : Expr} (hr : ∀ d ∈ e.readData, I.readData d = I'.readData d)
    (hs : ∀ x ∈ e.freeSyms, ρ x = ρ' x) : e.evalBool I ρ = e.evalBool I' ρ' := by
  unfold evalBool; rw [eval_congr_rd happ hattr e hr hs]

end Expr

theorem Dim.elems_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) {d : Dim} (hn : ∀ x ∈ d.names, (I.readData x = I'.readData x) ∧ ρ x = ρ' x) :
    d.elems I ρ = d.elems I' ρ' := by
  have h : ∀ e : Expr, (∀ x ∈ e.names, x ∈ d.names) → e.evalInt I ρ = e.evalInt I' ρ' := fun e he =>
    Expr.evalInt_congr_rd happ hattr (fun x hx => (hn x (he x (by simp [Expr.names, hx]))).1)
      (fun x hx => (hn x (he x (by simp [Expr.names, hx]))).2)
  unfold Dim.elems
  rw [h d.lo (fun x hx => by simp [Dim.names, hx]), h d.hi (fun x hx => by simp [Dim.names, hx]),
    h d.step (fun x hx => by simp [Dim.names, hx])]

theorem Range.points_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) {r : Range} (hn : ∀ x ∈ r.names, (I.readData x = I'.readData x) ∧ ρ x = ρ' x) :
    r.points I ρ = r.points I' ρ' := by
  unfold Range.points
  rw [mapM_congr_mem r.dims fun d hd => Dim.elems_congr_rd happ hattr fun x hx =>
    hn x (List.mem_flatMap.2 ⟨d, hd, hx⟩)]

theorem Subset.points_congr_rd {I I' : ExprInterp} {ρ ρ' : Valuation} (happ : I.app = I'.app)
    (hattr : I.readAttr = I'.readAttr) {S : Subset} (hn : ∀ x ∈ S.names, (I.readData x = I'.readData x) ∧ ρ x = ρ' x) :
    S.points I ρ = S.points I' ρ' := by
  cases S with
  | range r => exact Range.points_congr_rd happ hattr hn
  | union rs =>
    simp only [Subset.points]
    rw [mapM_congr_mem rs fun r hr => Range.points_congr_rd happ hattr fun x hx =>
      hn x (List.mem_flatMap.2 ⟨r, hr, hx⟩)]

/-! ## Contexts related by a renaming -/

/-- `ctx'` interprets the name `σ n` like `ctx` interprets `n`; descriptor
shapes read only names `σ` fixes. -/
structure RenCtx (ctx' ctx : Ctx) : Prop where
  I : ctx'.I = ctx.I
  desc : ∀ n, ctx'.desc? (σ n) = ctx.desc? n
  cname : ∀ n, ctx'.cname (σ n) = ctx.cname n
  /-- Fresh names are read only under a call prefix (`Ctx.enter`). -/
  scope : ∀ p x : Name, ctx'.scope (p ++ "/" ++ x) = ctx.scope (p ++ "/" ++ x)
  shapes : ∀ n d, ctx.desc? n = some d → ∀ x ∈ d.fullRange.names, σ x = x

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation}
include h

theorem desc_fix {x : Name} (hx : σ x = x) : ctx'.desc? x = ctx.desc? x := by
  have := h.desc x; rwa [hx] at this

theorem cname_fix {x : Name} (hx : σ x = x) : ctx'.cname x = ctx.cname x := by
  have := h.cname x; rwa [hx] at this

theorem isView (n : Name) : ctx'.isView (σ n) = ctx.isView n := by simp [Ctx.isView, h.desc]
theorem isStream (n : Name) : ctx'.isStream (σ n) = ctx.isStream n := by simp [Ctx.isStream, h.desc]
theorem isReference (n : Name) : ctx'.isReference (σ n) = ctx.isReference n := by simp [Ctx.isReference, h.desc]

theorem elemOffset (d : DataDesc) (idx : Idx) : ctx'.elemOffset ρ d idx = ctx.elemOffset ρ d idx := by
  simp [Ctx.elemOffset, h.I]

theorem addr (m : Mem) (n : Name) (idx : Idx) : (ctx'.addr ρ m (σ n) idx).er = (ctx.addr ρ m n idx).er := by
  unfold Ctx.addr
  rw [h.desc, h.cname]
  simp only [h.elemOffset]
  cases ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    split
    · rfl
    · cases ctx.elemOffset ρ d idx <;> rfl

theorem addr_fix (m : Mem) {x : Name} (hx : σ x = x) (idx : Idx) : ctx'.addr ρ m x idx = ctx.addr ρ m x idx := by
  unfold Ctx.addr
  rw [h.desc_fix hx, h.cname_fix hx]
  simp only [h.elemOffset]

theorem checkBounds (n : Name) (idx : Idx) (oob : Bool) :
    (ctx'.checkBounds ρ (σ n) idx oob).er = (ctx.checkBounds ρ n idx oob).er := by
  unfold Ctx.checkBounds
  rw [h.desc, h.I]
  cases ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    split
    · rfl
    · refine Res.er_bind_congr rfl fun shape => ?_
      split <;> rfl

theorem checkBounds_fix {x : Name} (hx : σ x = x) (idx : Idx) (oob : Bool) :
    ctx'.checkBounds ρ x idx oob = ctx.checkBounds ρ x idx oob := by
  unfold Ctx.checkBounds
  rw [h.desc_fix hx, h.I]

theorem loc (m : Mem) (n : Name) (idx : Idx) (oob : Bool) :
    (ctx'.loc ρ m (σ n) idx oob).er = (ctx.loc ρ m n idx oob).er := by
  unfold Ctx.loc
  refine Res.er_bind_congr (h.addr m n idx) fun l => ?_
  exact Res.er_bind_congr (h.checkBounds n idx oob) fun _ => rfl

theorem loc_fix (m : Mem) {x : Name} (hx : σ x = x) (idx : Idx) (oob : Bool) :
    ctx'.loc ρ m x idx oob = ctx.loc ρ m x idx oob := by
  unfold Ctx.loc
  rw [h.addr_fix m hx, h.checkBounds_fix hx]

theorem exprInterp_app (m : Mem) : (ctx'.exprInterp ρ m).app = (ctx.exprInterp ρ m).app := by
  simp [Ctx.exprInterp, h.I]

theorem exprInterp_attr (m : Mem) : (ctx'.exprInterp ρ m).readAttr = (ctx.exprInterp ρ m).readAttr := by
  simp [Ctx.exprInterp, h.I]

theorem exprInterp_fix (m : Mem) {x : Name} (hx : σ x = x) :
    (ctx'.exprInterp ρ m).readData x = (ctx.exprInterp ρ m).readData x := by
  simp only [Ctx.exprInterp]
  funext idx
  rw [h.loc_fix m hx]

theorem val_fix (m : Mem) {x : Name} (hx : σ x = x) : ctx'.val ρ m x = ctx.val ρ m x := by
  simp only [Ctx.val, h.desc_fix hx, h.loc_fix m hx]

theorem eval_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) : ctx'.eval ρ m e = ctx.eval ρ m e := by
  unfold Ctx.eval
  exact (Expr.eval_congr_rd (h.exprInterp_app m) (h.exprInterp_attr m) e
    (fun d hd => h.exprInterp_fix m (he d (by simp [Expr.names, hd])))
    (fun x hx => h.val_fix m (he x (by simp [Expr.names, hx]))))

theorem evalInt_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) : ctx'.evalInt ρ m e = ctx.evalInt ρ m e := by
  unfold Ctx.evalInt
  exact Expr.evalInt_congr_rd (h.exprInterp_app m) (h.exprInterp_attr m)
    (fun d hd => h.exprInterp_fix m (he d (by simp [Expr.names, hd])))
    (fun x hx => h.val_fix m (he x (by simp [Expr.names, hx])))

theorem evalBool_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) : ctx'.evalBool ρ m e = ctx.evalBool ρ m e := by
  unfold Ctx.evalBool
  exact Expr.evalBool_congr_rd (h.exprInterp_app m) (h.exprInterp_attr m)
    (fun d hd => h.exprInterp_fix m (he d (by simp [Expr.names, hd])))
    (fun x hx => h.val_fix m (he x (by simp [Expr.names, hx])))

theorem evalR_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) : ctx'.evalR ρ m e = ctx.evalR ρ m e := by
  unfold Ctx.evalR; rw [h.eval_fix m he]

theorem evalIntR_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) : ctx'.evalIntR ρ m e = ctx.evalIntR ρ m e := by
  unfold Ctx.evalIntR; rw [h.evalInt_fix m he]

theorem evalBoolR_fix (m : Mem) {e : Expr} (he : ∀ x ∈ e.names, σ x = x) :
    ctx'.evalBoolR ρ m e = ctx.evalBoolR ρ m e := by
  unfold Ctx.evalBoolR; rw [h.evalBool_fix m he]

theorem subsetPoints_fix (m : Mem) {S : Subset} (hS : ∀ x ∈ S.names, σ x = x) :
    S.points (ctx'.exprInterp ρ m) (ctx'.val ρ m) = S.points (ctx.exprInterp ρ m) (ctx.val ρ m) :=
  Subset.points_congr_rd (h.exprInterp_app m) (h.exprInterp_attr m)
    fun x hx => ⟨h.exprInterp_fix m (hS x hx), h.val_fix m (hS x hx)⟩

theorem rangePoints_fix (m : Mem) {r : Range} (hr : ∀ x ∈ r.names, σ x = x) :
    ctx'.rangePoints ρ m r = ctx.rangePoints ρ m r := by
  unfold Ctx.rangePoints
  rw [Range.points_congr_rd (h.exprInterp_app m) (h.exprInterp_attr m)
    fun x hx => ⟨h.exprInterp_fix m (hr x hx), h.val_fix m (hr x hx)⟩]

theorem pointsOf (m : Mem) (n : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) :
    (ctx'.pointsOf ρ m (σ n) sub).er = (ctx.pointsOf ρ m n sub).er := by
  unfold Ctx.pointsOf
  rw [h.desc]
  cases hd : ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    have hS : ∀ x ∈ (sub.getD (.range d.fullRange)).names, σ x = x := by
      cases sub with
      | none => exact fun x hx => h.shapes n d hd x hx
      | some S => exact fun x hx => hs x (by simpa [optSubsetNames] using hx)
    rw [h.subsetPoints_fix m hS]

end RenCtx

/-! ## Sequential-state operations -/

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation}
include h

theorem read (st : St) (n : Name) (idx : Idx) (oob : Bool) :
    (st.read ctx' ρ (σ n) idx oob).er = (st.read ctx ρ n idx oob).er := by
  unfold St.read
  exact Res.er_bind_congr (h.loc _ n idx oob) fun l => by simp only; split <;> rfl

theorem readMany (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    (st.readMany ctx' ρ (σ n) pts oob).er = (st.readMany ctx ρ n pts oob).er := by
  unfold St.readMany
  exact Res.er_foldlM_congr (fun p _ ⟨acc, st⟩ => Res.er_bind_congr (h.read st n p oob) fun _ => rfl) _

theorem write (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    (st.write ctx' ρ (σ n) idx oob v).er = (st.write ctx ρ n idx oob v).er := by
  unfold St.write
  exact Res.er_bind_congr (h.loc _ n idx oob) fun _ => rfl

theorem wcr (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    (st.wcr ctx' ρ (σ n) idx oob op v).er = (st.wcr ctx ρ n idx oob op v).er := by
  unfold St.wcr
  rw [h.I]
  exact Res.er_bind_congr (h.loc _ n idx oob) fun _ => rfl

theorem store (st : St) (n : Name) (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    (st.store ctx' ρ (σ n) idx oob w v).er = (st.store ctx ρ n idx oob w v).er := by
  unfold St.store
  cases w
  · exact h.write st n idx oob v
  · exact h.wcr st n idx oob _ v

theorem queue (st : St) (n : Name) (idx : Idx) : (st.queue ctx' ρ (σ n) idx).er = (st.queue ctx ρ n idx).er := by
  unfold St.queue
  exact Res.er_bind_congr (h.loc _ n idx false) fun _ => rfl

theorem push (st : St) (n : Name) (idx : Idx) (vs : List Val) :
    (st.push ctx' ρ (σ n) idx vs).er = (st.push ctx ρ n idx vs).er := by
  unfold St.push
  exact Res.er_bind_congr (h.loc _ n idx false) fun _ => rfl

theorem popN (st : St) (n : Name) (idx : Idx) (k : Nat) :
    (st.popN ctx' ρ (σ n) idx k).er = (st.popN ctx ρ n idx k).er := by
  unfold St.popN
  exact Res.er_bind_congr (h.loc _ n idx false) fun _ => rfl

end RenCtx

@[simp] theorem State.leafSrcSubset_ren (e : DEdge) : State.leafSrcSubset (e.ren σ) = State.leafSrcSubset e := by
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

@[simp] theorem State.leafDstSubset_ren (e : DEdge) : State.leafDstSubset (e.ren σ) = State.leafDstSubset e := by
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

/-! ## States whose expressions read only fixed names -/

/-- Every name read by an expression of `s` (and every consumed stream with
its chunk container) is fixed by `σ`. -/
structure RenState (s : State) : Prop where
  expr : ∀ x ∈ s.exprNames, σ x = x
  streams : ∀ entry d, s.consumedStream? entry = some d → σ d = d ∧ σ (consumeChunkName d) = consumeChunkName d

namespace RenState

variable {σ} {s : State} (hS : RenState σ s)
include hS

theorem srcSub {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset} (h : s.srcEndpoint e = .data d sub) :
    ∀ x ∈ optSubsetNames sub, σ x = x := by
  rcases (State.srcEndpoint_names h).2 with rfl | rfl
  · simp [optSubsetNames]
  · exact fun x hx => hS.expr x (State.leafSrcSubset_names he x hx)

theorem dstSub {e : DEdge} (he : e ∈ s.edges) {d : Name} {sub : Option Subset} (h : s.dstEndpoint e = .data d sub) :
    ∀ x ∈ optSubsetNames sub, σ x = x := by
  rw [(State.dstEndpoint_names h).2]
  exact fun x hx => hS.expr x (State.leafDstSubset_names he x hx)

theorem srcEndpoint (e : DEdge) : (s.ren σ).srcEndpoint (e.ren σ) = (s.srcEndpoint e).ren σ := by
  unfold State.srcEndpoint
  rw [show (s.ren σ).edges.length + 1 = s.edges.length + 1 by simp, State.chunkSource_ren]
  cases s.chunkSource (s.edges.length + 1) e with
  | some entry =>
    simp only [State.consumedStream?_ren]
    cases hc : s.consumedStream? entry with
    | none => rfl
    | some d =>
      obtain ⟨h₁, h₂⟩ := hS.streams entry d hc
      simp [h₁, h₂]
  | none =>
    simp only [State.pathStart_ren, DEdge.ren_src, State.node?_ren]
    cases s.node? (s.pathStart e).src with
    | none => rfl
    | some n =>
      rcases n with ⟨_, _, _, _, _, k⟩
      cases k <;> simp [DNode.ren, NodeKind.ren]

end RenState

theorem State.dstEndpoint_ren (s : State) (e : DEdge) : (s.ren σ).dstEndpoint (e.ren σ) = (s.dstEndpoint e).ren σ := by
  unfold State.dstEndpoint
  simp only [DEdge.ren_dst, DEdge.ren_src, DEdge.ren_srcConn, State.node?_ren]
  cases s.node? e.dst with
  | none => rfl
  | some n =>
    rcases n with ⟨_, _, _, _, _, k⟩
    cases k with
    | mapExit | consumeExit =>
      simp only [Option.map_some, DNode.ren, NodeKind.ren, State.dstRootEdge_ren, DEdge.ren_dst]
      cases s.node? (s.dstRootEdge e).dst with
      | none => rfl
      | some rn =>
        rcases rn with ⟨_, _, _, _, _, k'⟩
        cases k' <;> simp [NodeKind.ren]
    | _ => simp [DNode.ren, NodeKind.ren]

/-! ## Reading and copying data -/

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation}
include h

theorem readScalarData (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) (oob : Bool)
    (st : St) : (Dace.readScalarData ctx' ρ (σ d) sub oob st).er = (Dace.readScalarData ctx ρ d sub oob st).er := by
  unfold Dace.readScalarData
  refine Res.er_bind_congr (h.pointsOf _ d hs) fun pts => ?_
  split
  · rw [h.isStream]
    split
    · exact Res.er_bind_congr (h.popN st d _ 1) fun _ => rfl
    · exact Res.er_bind_congr (h.read st d _ oob) fun _ => rfl
  · rfl

theorem streamPoint (m : Mem) (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) :
    (Dace.streamPoint ctx' ρ m (σ d) sub).er = (Dace.streamPoint ctx ρ m d sub).er := by
  unfold Dace.streamPoint
  exact Res.er_bind_congr (h.pointsOf m d hs) fun pts => Res.er_ofOption_congr rfl

theorem lowerCorner (m : Mem) (n : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) :
    (ctx'.lowerCorner ρ m (σ n) sub).er = (ctx.lowerCorner ρ m n sub).er := by
  unfold Ctx.lowerCorner
  refine Res.er_bind_congr (h.pointsOf m n hs) fun _ => ?_
  cases sub with
  | none => simp only [h.desc]; cases ctx.desc? n <;> rfl
  | some S =>
    simp only
    split
    · rfl
    · rename_i r hr
      have hrn : ∀ x ∈ r.names, σ x = x := by
        intro x hx
        apply hs x
        cases S with
        | range r' => cases hr; simpa [optSubsetNames, Subset.names] using hx
        | union rs =>
          simp only [List.head?_eq_some_iff] at hr
          obtain ⟨tl, rfl⟩ := hr
          simp only [optSubsetNames, Option.map_some, Option.getD_some, Subset.names, List.flatMap_cons,
            List.mem_append]
          exact .inl hx
      congr 1
      refine mapM_congr_mem r.dims fun dim hdim => ?_
      rw [h.evalInt_fix m fun x hx => hrn x (List.mem_flatMap.2 ⟨dim, hdim, by
        simp only [Dim.names, List.mem_append]; exact .inl (.inl (.inl hx))⟩)]

theorem copyData (st : St) (sd : Name) {ssub : Option Subset} (hss : ∀ x ∈ optSubsetNames ssub, σ x = x)
    (dd : Name) {dsub : Option Subset} (hds : ∀ x ∈ optSubsetNames dsub, σ x = x) (w : Option WcrOp) (oob : Bool) :
    (Dace.copyData ctx' ρ st (σ sd) ssub (σ dd) dsub w oob).er = (Dace.copyData ctx ρ st sd ssub dd dsub w oob).er := by
  unfold Dace.copyData
  rw [h.isStream, h.isStream]
  split
  · rfl
  · refine Res.er_bind_congr (h.pointsOf _ dd hds) fun dpts => ?_
    refine Res.er_bind_congr (h.streamPoint _ sd hss) fun sp => ?_
    refine Res.er_bind_congr (h.popN st sd sp _) fun ⟨vs, st'⟩ => ?_
    exact Res.er_foldlM_congr (fun ⟨q, v⟩ _ b => h.store b dd q oob w v) _
  · refine Res.er_bind_congr (h.pointsOf _ sd hss) fun spts => ?_
    refine Res.er_bind_congr (h.streamPoint _ dd hds) fun dp => ?_
    refine Res.er_bind_congr (h.readMany st sd spts oob) fun ⟨vs, st'⟩ => ?_
    exact h.push st' dd dp vs
  · refine Res.er_bind_congr (h.pointsOf _ sd hss) fun spts => ?_
    refine Res.er_bind_congr (h.pointsOf _ dd hds) fun dpts => ?_
    exact Res.er_foldlM_congr (fun ⟨p, q⟩ _ b =>
      Res.er_bind_congr (h.read b sd p oob) fun ⟨v, b'⟩ => h.store b' dd q oob w v) _

end RenCtx

theorem State.outEdgesOn_ren (s : State) (n : DNode) (c : Conn) :
    (s.ren σ).outEdgesOn (n.ren σ) c = (s.outEdgesOn n c).map (DEdge.ren σ) := by
  simp only [State.outEdgesOn, DNode.ren_id, State.outEdges_ren, List.filter_map]
  congr 1
  apply List.filter_congr
  intro e _
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

theorem State.inEdgeOn_ren (s : State) (n : DNode) (c : Conn) :
    (s.ren σ).inEdgeOn (n.ren σ) c = (s.inEdgeOn n c).map (DEdge.ren σ) := by
  simp only [State.inEdgeOn, DNode.ren_id, State.inEdges_ren, List.find?_map]
  congr 2
  funext e
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

theorem State.hasDynamicWcrOut_ren (s : State) (n : DNode) : (s.ren σ).hasDynamicWcrOut (n.ren σ) = s.hasDynamicWcrOut n := by
  simp only [State.hasDynamicWcrOut, DNode.ren_id, State.outEdges_ren, List.any_map]
  congr 1
  funext e
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation} {s : State} (hS : RenState σ s)
include h hS

theorem readEdgeScalar {e : DEdge} (he : e ∈ s.edges) (st : St) :
    (Dace.readEdgeScalar ctx' ρ (s.ren σ) (e.ren σ) st).er = (Dace.readEdgeScalar ctx ρ s e st).er := by
  unfold Dace.readEdgeScalar
  rw [hS.srcEndpoint]
  cases hse : s.srcEndpoint e with
  | data d sub => simpa using h.readScalarData d (hS.srcSub he hse) e.allowOob st
  | code k => rfl
  | invalid => rfl

theorem execCopy {e : DEdge} (he : e ∈ s.edges) (st : St) :
    (Dace.execCopy ctx' ρ (s.ren σ) (e.ren σ) st).er = (Dace.execCopy ctx ρ s e st).er := by
  unfold Dace.execCopy
  rw [hS.srcEndpoint, State.dstEndpoint_ren]
  rcases e with ⟨_, _, _, _, dt⟩
  cases dt with
  | dep => rfl
  | memlet m =>
    simp only [DEdge.ren_data, EdgeData.ren_memlet]
    cases hse : s.srcEndpoint _ <;> cases hde : s.dstEndpoint _ <;> try rfl
    simpa using h.copyData st _ (hS.srcSub he hse) _ (hS.dstSub he hde) m.wcr m.allowOob

omit hS in
theorem bindInData (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) (oob : Bool) (st : St) :
    (Dace.bindInData ctx' ρ (σ d) sub oob st).er = (Dace.bindInData ctx ρ d sub oob st).er := by
  unfold Dace.bindInData
  rw [h.isStream]
  split
  · refine Res.er_bind_congr (h.streamPoint _ d hs) fun p => ?_
    exact Res.er_bind_congr (h.queue st d p) fun _ => rfl
  · refine Res.er_bind_congr (h.pointsOf _ d hs) fun pts => ?_
    split
    · exact Res.er_bind_congr (h.read st d _ oob) fun _ => rfl
    · refine Res.er_bind_congr (h.lowerCorner _ d hs) fun lo => ?_
      exact Res.er_bind_congr (h.readMany st d pts oob) fun _ => rfl

theorem bindIn {e : DEdge} (he : e ∈ s.edges) (st : St) :
    (Dace.bindIn ctx' ρ (s.ren σ) (e.ren σ) st).er = (Dace.bindIn ctx ρ s e st).er := by
  unfold Dace.bindIn
  rw [hS.srcEndpoint]
  cases hse : s.srcEndpoint e with
  | data d sub =>
    simp only [State.Endpoint.ren_data', DEdge.ren_allowOob]
    exact Res.er_bind_congr (h.bindInData d (hS.srcSub he hse) e.allowOob st) fun _ => rfl
  | code k => rfl
  | invalid => rfl

theorem bindInputs (n : DNode) (st : St) :
    (Dace.bindInputs ctx' ρ (s.ren σ) (n.ren σ) st).er = (Dace.bindInputs ctx ρ s n st).er := by
  unfold Dace.bindInputs
  simp only [DNode.ren_id, State.inEdges_ren, List.foldlM_map]
  refine Res.er_bind_congr (Res.er_foldlM_congr (fun e he ⟨acc, st⟩ => ?_) _) fun _ => rfl
  rcases e with ⟨_, _, _, _, dt⟩
  cases dt with
  | dep => rfl
  | memlet m =>
    exact Res.er_bind_congr (h.bindIn hS (State.mem_inEdges he) st) fun _ => rfl

omit hS in
theorem storeScalar (st : St) (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x)
    (w : Option WcrOp) (oob : Bool) (v : Val) :
    (Dace.storeScalar ctx' ρ st (σ d) sub w oob v).er = (Dace.storeScalar ctx ρ st d sub w oob v).er := by
  unfold Dace.storeScalar
  rw [h.isStream]
  split
  · exact Res.er_bind_congr (h.streamPoint _ d hs) fun p => h.push st d p [v]
  · refine Res.er_bind_congr (h.pointsOf _ d hs) fun pts => ?_
    split
    · exact h.store st d _ oob w v
    · rfl
    · rfl

theorem applyScalar (n : DNode) (st : St) (c : Conn) (v : Val) :
    (Dace.applyScalar ctx' ρ (s.ren σ) (n.ren σ) st c v).er = (Dace.applyScalar ctx ρ s n st c v).er := by
  unfold Dace.applyScalar
  rw [State.outEdgesOn_ren, List.foldlM_map]
  refine Res.er_foldlM_congr (fun e he st => ?_) st
  have hmem := State.edge_of_outEdgesOn he
  rcases e with ⟨_, _, _, _, dt⟩
  cases dt with
  | dep => rfl
  | memlet m =>
    simp only [DEdge.ren_data, EdgeData.ren_memlet]
    rw [State.dstEndpoint_ren]
    cases hde : s.dstEndpoint _ with
    | data d sub => simpa using h.storeScalar st d (hS.dstSub hmem hde) m.wcr m.allowOob v
    | code k => rfl
    | invalid => rfl

omit hS in
theorem storeArrayWrite (st : St) (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x)
    (w : Option WcrOp) (oob : Bool) (rel : Idx) (v : Val) :
    (Dace.storeArrayWrite ctx' ρ st (σ d) sub w oob rel v).er = (Dace.storeArrayWrite ctx ρ st d sub w oob rel v).er := by
  unfold Dace.storeArrayWrite
  rw [h.isStream]
  split
  · rfl
  · refine Res.er_bind_congr (h.lowerCorner _ d hs) fun lo => ?_
    split
    · refine Res.er_bind_congr (h.pointsOf _ d hs) fun pts => ?_
      simp only
      split
      · exact h.store st d _ oob w v
      · rfl
    · rfl

theorem applyArrayWrite (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    (Dace.applyArrayWrite ctx' ρ (s.ren σ) (n.ren σ) st c rel v).er = (Dace.applyArrayWrite ctx ρ s n st c rel v).er := by
  unfold Dace.applyArrayWrite
  rw [State.outEdgesOn_ren, List.foldlM_map]
  refine Res.er_foldlM_congr (fun e he st => ?_) st
  have hmem := State.edge_of_outEdgesOn he
  rcases e with ⟨_, _, _, _, dt⟩
  cases dt with
  | dep => rfl
  | memlet m =>
    simp only [DEdge.ren_data, EdgeData.ren_memlet]
    rw [State.dstEndpoint_ren]
    cases hde : s.dstEndpoint _ with
    | data d sub => simpa using h.storeArrayWrite st d (hS.dstSub hmem hde) m.wcr m.allowOob rel v
    | code k => rfl
    | invalid => rfl

omit hS in
theorem pushTo (st : St) (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) (vs : List Val) :
    (Dace.pushTo ctx' ρ st (σ d) sub vs).er = (Dace.pushTo ctx ρ st d sub vs).er := by
  unfold Dace.pushTo
  rw [h.isStream]
  split
  · exact Res.er_bind_congr (h.streamPoint _ d hs) fun p => h.push st d p vs
  · rfl

theorem applyPush (n : DNode) (st : St) (c : Conn) (vs : List Val) :
    (Dace.applyPush ctx' ρ (s.ren σ) (n.ren σ) st c vs).er = (Dace.applyPush ctx ρ s n st c vs).er := by
  unfold Dace.applyPush
  rw [State.outEdgesOn_ren, List.foldlM_map]
  refine Res.er_foldlM_congr (fun e he st => ?_) st
  have hmem := State.edge_of_outEdgesOn he
  rw [State.dstEndpoint_ren]
  cases hde : s.dstEndpoint e with
  | data d sub => simpa using h.pushTo st d (hS.dstSub hmem hde) vs
  | code k => rfl
  | invalid => rfl

omit hS in
theorem popFrom (st : St) (d : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) (k : Nat) :
    (Dace.popFrom ctx' ρ st (σ d) sub k).er = (Dace.popFrom ctx ρ st d sub k).er := by
  unfold Dace.popFrom
  rw [h.isStream]
  split
  · refine Res.er_bind_congr (h.streamPoint _ d hs) fun p => ?_
    exact Res.er_bind_congr (h.popN st d p k) fun _ => rfl
  · rfl

theorem applyPop (n : DNode) (st : St) (c : Conn) (k : Nat) :
    (Dace.applyPop ctx' ρ (s.ren σ) (n.ren σ) st c k).er = (Dace.applyPop ctx ρ s n st c k).er := by
  unfold Dace.applyPop
  rw [State.inEdgeOn_ren]
  cases hi : s.inEdgeOn n c with
  | none => rfl
  | some e =>
    have hmem := State.edge_of_inEdgeOn hi
    simp only [Option.map_some]
    rw [hS.srcEndpoint]
    cases hse : s.srcEndpoint e with
    | data d sub => simpa using h.popFrom st d (hS.srcSub hmem hse) k
    | code k => rfl
    | invalid => rfl

theorem applyOut (n : DNode) (st : St) (out : TaskletOut) :
    (Dace.applyOut ctx' ρ (s.ren σ) (n.ren σ) st out).er = (Dace.applyOut ctx ρ s n st out).er := by
  unfold Dace.applyOut
  refine Res.er_bind_congr (Res.er_foldlM_congr (fun ⟨c, k⟩ _ st => h.applyPop hS n st c k) _) fun st => ?_
  refine Res.er_bind_congr (Res.er_foldlM_congr (fun ⟨c, v⟩ _ st => h.applyScalar hS n st c v) _) fun st => ?_
  refine Res.er_bind_congr (Res.er_foldlM_congr (fun ⟨c, rel, v⟩ _ st => h.applyArrayWrite hS n st c rel v) _)
    fun st => ?_
  exact Res.er_foldlM_congr (fun ⟨c, vs⟩ _ st => h.applyPush hS n st c vs) _

theorem execCode (n : DNode) (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut) (err : ErrReason) :
    (Dace.execCode ctx' ρ (s.ren σ) (n.ren σ) st run err).er = (Dace.execCode ctx ρ s n st run err).er := by
  unfold Dace.execCode
  rw [State.hasDynamicWcrOut_ren]
  split
  · rfl
  · refine Res.er_bind_congr (h.bindInputs hS n st) fun ⟨ins, st'⟩ => ?_
    simp only
    split
    · rfl
    · exact h.applyOut hS n st' _

end RenCtx

/-! ## Views -/

namespace State

variable {σ}

theorem memletData?_ren (e : DEdge) : State.memletData? (e.ren σ) = (State.memletData? e).map σ := by
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

theorem pathStartsAtAccess_ren (s : State) (e : DEdge) :
    (s.ren σ).pathStartsAtAccess (e.ren σ) = s.pathStartsAtAccess e := by
  simp [pathStartsAtAccess, pathStart_ren, isAccessNode_ren]

theorem pathEndsAtAccess_ren (s : State) (e : DEdge) :
    (s.ren σ).pathEndsAtAccess (e.ren σ) = s.pathEndsAtAccess e := by
  simp [pathEndsAtAccess, pathEnd_ren, isAccessNode_ren]

theorem map_eq_map_iff {a b : Option Name} : a.map σ = b.map σ ↔ a = b :=
  ⟨fun h => Option.map_injective σ.injective h, fun h => h ▸ rfl⟩

theorem viewEdgeOneOne_ren (s : State) (v : DNode) (i : DEdge) (o? : Option DEdge) :
    (s.ren σ).viewEdgeOneOne (v.ren σ) (i.ren σ) (o?.map (DEdge.ren σ)) =
      (s.viewEdgeOneOne v i o?).map fun (e, b) => (e.ren σ, b) := by
  unfold viewEdgeOneOne
  have ho : (o?.map (DEdge.ren σ)).any ((s.ren σ).pathEndsAtAccess ·) = o?.any (s.pathEndsAtAccess ·) := by
    cases o? <;> simp [pathEndsAtAccess_ren]
  simp only [pathStartsAtAccess_ren, ho]
  split
  · rfl
  · split
    · cases o? <;> rfl
    · split
      · rfl
      · cases o? with
        | none => rfl
        | some o =>
          simp only [Option.map_some, DEdge.ren_dstConn, DEdge.ren_srcConn, DNode.ren_accessData?,
            memletData?_ren, map_eq_map_iff, DEdge.ren_src, DEdge.ren_dst, isEntryNode_ren, isExitNode_ren]
          repeat' split
          all_goals rfl

/-- The memlet in-edges of a node. -/
def memletIns (s : State) (v : DNode) : List DEdge := (s.inEdges v.id).filter fun e => e.data matches .memlet _

/-- The body of `viewEdge?` for given in- and out-edges. -/
def viewEdgeCore (s : State) (v : DNode) (ins outs : List DEdge) : Option (DEdge × Bool) :=
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

theorem viewEdge?_eq (s : State) (v : DNode) : s.viewEdge? v = s.viewEdgeCore v (s.memletIns v) (s.outEdges v.id) := rfl

theorem memletIns_ren (s : State) (v : DNode) : (s.ren σ).memletIns (v.ren σ) = (s.memletIns v).map (DEdge.ren σ) := by
  rw [memletIns, DNode.ren_id, inEdges_ren, List.filter_map]
  congr 1
  apply List.filter_congr
  intro e _
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

theorem viewEdge?_ren (s : State) (v : DNode) :
    (s.ren σ).viewEdge? (v.ren σ) = (s.viewEdge? v).map fun (e, b) => (e.ren σ, b) := by
  rw [viewEdge?_eq, viewEdge?_eq, memletIns_ren, DNode.ren_id, outEdges_ren]
  generalize s.memletIns v = ins
  generalize s.outEdges v.id = outs
  have key : ∀ (i : DEdge) (o? : Option DEdge),
      (s.ren σ).viewEdgeOneOne (v.ren σ) (i.ren σ) (o?.map (DEdge.ren σ)) =
        (s.viewEdgeOneOne v i o?).map fun (e, b) => (e.ren σ, b) := viewEdgeOneOne_ren s v
  unfold viewEdgeCore
  rcases ins with _ | ⟨i, ins⟩ <;> rcases outs with _ | ⟨o, outs⟩ <;>
    simp only [List.map_nil, List.map_cons, List.isEmpty_nil, List.isEmpty_cons, List.length_map, List.head?_cons,
      List.head?_nil, Bool.true_and, Bool.false_and, Bool.and_false, Bool.false_eq_true, ↓reduceIte,
      pathStartsAtAccess_ren, pathEndsAtAccess_ren] <;>
    (repeat' split) <;> simp_all <;> first | exact key i none | exact key i (some o)

end State

/-- The ready views of a generation (`viewGensWith`). -/
def vgReadyR (d b : NodeId → Option Name) (pending : List NodeId) : List NodeId :=
  pending.filter fun id => match b id with
    | some x => !(pending.filterMap d).contains x
    | none => true

theorem viewGensWith_succ_consR (d b : NodeId → Option Name) (f : Nat) (p : NodeId) (ps : List NodeId) :
    viewGensWith d b (f + 1) (p :: ps) =
      if (vgReadyR d b (p :: ps)).isEmpty then p :: ps
      else vgReadyR d b (p :: ps) ++
        viewGensWith d b f ((p :: ps).filter fun id => !(vgReadyR d b (p :: ps)).contains id) := rfl

theorem viewGensWith_ren {d₁ b₁ d₂ b₂ : NodeId → Option Name} (hd : ∀ id, d₂ id = (d₁ id).map σ)
    (hb : ∀ id, b₂ id = (b₁ id).map σ) :
    ∀ (f : Nat) (pending : List NodeId), viewGensWith d₂ b₂ f pending = viewGensWith d₁ b₁ f pending
  | 0, _ => rfl
  | _ + 1, [] => rfl
  | f + 1, p :: ps => by
    have hpd : (p :: ps).filterMap d₂ = ((p :: ps).filterMap d₁).map σ := by
      rw [List.map_filterMap]
      exact List.filterMap_congr fun id _ => hd id
    have hready : vgReadyR d₂ b₂ (p :: ps) = vgReadyR d₁ b₁ (p :: ps) := by
      unfold vgReadyR
      refine List.filter_congr fun id _ => ?_
      rw [hb, hpd]
      cases b₁ id with
      | none => rfl
      | some b => simp [σ.injective.eq_iff]
    rw [viewGensWith_succ_consR, viewGensWith_succ_consR, hready, viewGensWith_ren hd hb f]

theorem keepFirstSig_ren {α β : Type} [DecidableEq α] [DecidableEq β] {F : α → β} (hF : Function.Injective F)
    {sig₁ : NodeId → Option α} {sig₂ : NodeId → Option β} (h : ∀ id, sig₂ id = (sig₁ id).map F) :
    ∀ (seen : List α) (l : List NodeId), keepFirstSig sig₂ (seen.map F) l = keepFirstSig sig₁ seen l
  | _, [] => rfl
  | seen, x :: xs => by
    simp only [keepFirstSig, h]
    cases sig₁ x with
    | none => simp only [Option.map_none]; rw [keepFirstSig_ren hF h seen xs]
    | some a =>
      simp only [Option.map_some]
      have : (seen.map F).contains (F a) = seen.contains a := by simp [hF.eq_iff]
      rw [this]
      split
      · exact keepFirstSig_ren hF h seen xs
      · rw [← List.map_cons, keepFirstSig_ren hF h _ xs]

namespace State

variable {σ}

theorem viewedData?_ren (s : State) (v : DNode) : (s.ren σ).viewedData? (v.ren σ) = (s.viewedData? v).map σ := by
  unfold viewedData?
  rw [viewEdge?_ren]
  cases s.viewEdge? v with
  | none => rfl
  | some p =>
    obtain ⟨e, b⟩ := p
    cases b <;> simp [pathStart_ren, pathEnd_ren, node?_ren] <;>
      first | (cases s.node? (s.pathEnd e).dst <;> simp) | (cases s.node? (s.pathStart e).src <;> simp)

theorem viewBinding?_ren (s : State) (v : DNode) :
    (s.ren σ).viewBinding? (v.ren σ) = (s.viewBinding? v).map fun (o, sub) => (o.map σ, sub) := by
  unfold viewBinding?
  rw [viewEdge?_ren]
  cases s.viewEdge? v with
  | none => rfl
  | some p =>
    obtain ⟨e, b⟩ := p
    cases b <;> simp [pathStart_ren, pathEnd_ren, node?_ren] <;>
      first | (cases s.node? (s.pathEnd e).dst <;> simp) | (cases s.node? (s.pathStart e).src <;> simp)

/-- The renaming of a view signature. -/
def sigRen (σ : Equiv.Perm Name) (x : Name × Option (Option Name × Option Subset)) :
    Name × Option (Option Name × Option Subset) :=
  (σ x.1, x.2.map fun (o, sub) => (o.map σ, sub))

theorem sigRen_injective : Function.Injective (sigRen σ) := by
  intro ⟨a, b⟩ ⟨c, d⟩ h
  simp only [sigRen, Prod.mk.injEq] at h
  obtain ⟨h₁, h₂⟩ := h
  have h₁ := σ.injective h₁
  subst h₁
  cases b <;> cases d <;> simp_all
  rename_i x y
  obtain ⟨o, sub⟩ := x
  obtain ⟨o', sub'⟩ := y
  simp only [Prod.mk.injEq] at h₂ ⊢
  exact ⟨Option.map_injective σ.injective h₂.1, h₂.2⟩

theorem viewSig_ren (s : State) (id : NodeId) : (s.ren σ).viewSig id = (s.viewSig id).map (sigRen σ) := by
  unfold viewSig
  rw [node?_ren]
  cases s.node? id with
  | none => rfl
  | some n =>
    simp only [Option.map_some, Option.bind_some, DNode.ren_accessData?, viewBinding?_ren]
    cases n.accessData? <;> rfl

end State

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx)
include h

theorem isViewNode (s : State) (id : NodeId) : (s.ren σ).isViewNode ctx' id = s.isViewNode ctx id := by
  unfold State.isViewNode
  rw [State.node?_ren]
  cases s.node? id with
  | none => rfl
  | some n =>
    rcases n with ⟨_, _, _, _, _, k⟩
    cases k <;> simp [DNode.ren, NodeKind.ren, h.isView]

theorem viewOrder (s : State) (ids : List NodeId) : (s.ren σ).viewOrder ctx' ids = s.viewOrder ctx ids := by
  unfold State.viewOrder State.viewGens
  have hf : ids.filter ((s.ren σ).isViewNode ctx') = ids.filter (s.isViewNode ctx) :=
    List.filter_congr fun id _ => h.isViewNode s id
  simp only
  rw [hf, show ([] : List (Name × Option (Option Name × Option Subset))) = ([] : List _).map (State.sigRen σ) from rfl,
    keepFirstSig_ren State.sigRen_injective (State.viewSig_ren s)]
  refine viewGensWith_ren σ (fun id => ?_) (fun id => ?_) _ _
  · rw [State.node?_ren]; cases s.node? id <;> simp
  · rw [State.node?_ren]
    cases s.node? id with
    | none => rfl
    | some n => exact State.viewedData?_ren s n

end RenCtx

/-! ## Aliasing, references, reduce, simple nodes -/

namespace State

/-- The memlet out-edges of a node. -/
def memletOuts (s : State) (v : DNode) : List DEdge := (s.outEdges v.id).filter fun e => e.data matches .memlet _

theorem memletOuts_ren (s : State) (v : DNode) : (s.ren σ).memletOuts (v.ren σ) = (s.memletOuts v).map (DEdge.ren σ) := by
  rw [memletOuts, DNode.ren_id, outEdges_ren, List.filter_map]
  congr 1
  apply List.filter_congr
  intro e _
  rcases e with ⟨_, _, _, _, d⟩; cases d <;> rfl

end State

/-- The body of `execReduce` for given memlet in- and out-edges. -/
def execReduceCore (ctx : Ctx) (ρ : Valuation) (s : State) (props : ReduceProps) (ins outs : List DEdge) (st : St) :
    Res St :=
  match ins, outs with
  | [ie], [oe] =>
    match s.srcEndpoint ie, s.dstEndpoint oe, oe.data.memlet? with
    | .data sd ssub, .data dd dsub, some om =>
      reduceData ctx ρ st props sd ssub dd dsub om.wcr ie.allowOob om.allowOob
    | _, _, _ => .err (.malformed "reduce node not between containers")
  | _, _ => .err (.malformed "reduce node must have exactly one input and one output")

theorem execReduce_eq (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (props : ReduceProps) (st : St) :
    execReduce ctx ρ s n props st = execReduceCore ctx ρ s props (s.memletIns n) (s.memletOuts n) st := rfl

/-- The view access node with id `id`, if any. -/
def viewNodeOf (ctx : Ctx) (s : State) (id : NodeId) : Option DNode :=
  (s.node? id).filter fun n => match n.kind with
    | .access d => ctx.isView d
    | _ => false

/-- The body of `Ctx.isViewDef`. -/
def isViewDefCore (ctx : Ctx) (s : State) (e : DEdge) : Bool :=
  (match viewNodeOf ctx s (s.pathStart e).src with
   | some v => decide (s.viewEdge? v = some (s.pathStart e, false))
   | none => false) ||
  (match viewNodeOf ctx s (s.dstRootEdge e).dst with
   | some v => decide (s.viewEdge? v = some (s.dstRootEdge e, true))
   | none => false)

theorem isViewDef_eq (ctx : Ctx) (s : State) (e : DEdge) : ctx.isViewDef s e = isViewDefCore ctx s e := rfl

theorem RenCtx.viewNodeOf {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (s : State) (id : NodeId) :
    viewNodeOf ctx' (s.ren σ) id = (viewNodeOf ctx s id).map (DNode.ren σ) := by
  unfold Dace.viewNodeOf
  rw [State.node?_ren]
  cases s.node? id with
  | none => rfl
  | some n =>
    rcases n with ⟨_, _, _, _, _, k⟩
    cases k <;> simp [Option.filter, DNode.ren, NodeKind.ren, h.isView]

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation}
include h

theorem reinterprets (d b : Name) : ctx'.reinterprets (σ d) (σ b) = ctx.reinterprets d b := by
  simp only [Ctx.reinterprets, h.desc]

theorem aliasTo (st : St) (d base : Name) {sub : Option Subset} (hs : ∀ x ∈ optSubsetNames sub, σ x = x) :
    (Dace.aliasTo ctx' ρ st (σ d) (σ base) sub).er = (Dace.aliasTo ctx ρ st d base sub).er := by
  unfold Dace.aliasTo
  rw [h.reinterprets]
  split
  · rfl
  · refine Res.er_bind_congr (h.lowerCorner _ base hs) fun lo => ?_
    refine Res.er_bind_congr (h.addr _ base lo) fun l => ?_
    simp only [h.cname]

theorem reduceData (st : St) (props : ReduceProps) (hid : ∀ x ∈ (props.identity.map Expr.names).getD [], σ x = x)
    (sd : Name) {ssub : Option Subset} (hss : ∀ x ∈ optSubsetNames ssub, σ x = x)
    (dd : Name) {dsub : Option Subset} (hds : ∀ x ∈ optSubsetNames dsub, σ x = x) (w : Option WcrOp) (soob doob : Bool) :
    (Dace.reduceData ctx' ρ st props (σ sd) ssub (σ dd) dsub w soob doob).er =
      (Dace.reduceData ctx ρ st props sd ssub dd dsub w soob doob).er := by
  unfold Dace.reduceData
  rw [h.isStream, h.isStream]
  split
  · rfl
  · refine Res.er_bind_congr (h.pointsOf _ sd hss) fun spts => ?_
    refine Res.er_bind_congr (h.pointsOf _ dd hds) fun dpts => ?_
    simp only
    split
    · rfl
    · refine Res.er_bind_congr (h.readMany st sd spts soob) fun ⟨vals, st'⟩ => ?_
      simp only [h.I]
      refine Res.er_foldlM_congr (fun ⟨g, q⟩ _ b => ?_) _
      simp only
      cases hi : props.identity with
      | some ide =>
        simp only
        have hide : ∀ x ∈ ide.names, σ x = x := fun x hx => hid x (by simp [hi, hx])
        rw [h.evalR_fix _ hide]
        exact Res.er_bind_congr rfl fun i => h.store b dd q doob w _
      | none =>
        simp only
        split
        · rfl
        · exact h.wcr b dd q doob _ _

end RenCtx

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation} {s : State} (hS : RenState σ s)
include h hS

theorem execReduce {n : DNode} (props : ReduceProps) (hid : ∀ x ∈ (props.identity.map Expr.names).getD [], σ x = x)
    (st : St) : (Dace.execReduce ctx' ρ (s.ren σ) (n.ren σ) props st).er = (Dace.execReduce ctx ρ s n props st).er := by
  rw [execReduce_eq, execReduce_eq, State.memletIns_ren, State.memletOuts_ren]
  have hinm : ∀ e ∈ s.memletIns n, e ∈ s.edges := fun e he => State.mem_inEdges (List.mem_of_mem_filter he)
  have houtm : ∀ e ∈ s.memletOuts n, e ∈ s.edges := fun e he => State.mem_outEdges (List.mem_of_mem_filter he)
  generalize s.memletIns n = ins at hinm
  generalize s.memletOuts n = outs at houtm
  unfold execReduceCore
  rcases ins with _ | ⟨ie, _ | ⟨_, _⟩⟩ <;> rcases outs with _ | ⟨oe, _ | ⟨_, _⟩⟩ <;> try rfl
  simp only [List.map_cons, List.map_nil]
  rw [hS.srcEndpoint, State.dstEndpoint_ren]
  rcases oe with ⟨_, _, _, _, od⟩
  cases hse : s.srcEndpoint ie <;> cases hde : s.dstEndpoint _ <;> cases od <;> try rfl
  dsimp only [State.Endpoint.ren, DEdge.ren_data, EdgeData.ren, EdgeData.memlet?]
  rw [DEdge.ren_allowOob]
  exact h.reduceData _ props hid _ (hS.srcSub (hinm ie (by simp)) hse) _ (hS.dstSub (houtm _ (by simp)) hde) _
    ie.allowOob _

theorem bindView {v : DNode} (d : Name) (st : St) :
    (Dace.bindView ctx' ρ (s.ren σ) (v.ren σ) (σ d) st).er = (Dace.bindView ctx ρ s v d st).er := by
  unfold Dace.bindView
  rw [State.viewEdge?_ren]
  cases hv : s.viewEdge? v with
  | none => rfl
  | some p =>
    obtain ⟨e, b⟩ := p
    have he := State.viewEdge?_mem hv
    cases b
    · simp only [Option.map_some, Bool.false_eq_true, ↓reduceIte, State.pathEnd_ren, DEdge.ren_dst, State.node?_ren,
        State.leafDstSubset_ren]
      cases s.node? (s.pathEnd e).dst with
      | none => rfl
      | some n =>
        simp only [Option.map_some, Option.bind_some, DNode.ren_accessData?]
        cases n.accessData? with
        | none => rfl
        | some base =>
          exact h.aliasTo st d base fun x hx => hS.expr x (State.leafDstSubset_names he x hx)
    · simp only [Option.map_some, ↓reduceIte, State.pathStart_ren, DEdge.ren_src, State.node?_ren,
        State.leafSrcSubset_ren]
      cases s.node? (s.pathStart e).src with
      | none => rfl
      | some n =>
        simp only [Option.map_some, Option.bind_some, DNode.ren_accessData?]
        cases n.accessData? with
        | none => rfl
        | some base =>
          exact h.aliasTo st d base fun x hx => hS.expr x (State.leafSrcSubset_names he x hx)

theorem bindViews (ids : List NodeId) (st : St) :
    (Dace.bindViews ctx' ρ (s.ren σ) ids st).er = (Dace.bindViews ctx ρ s ids st).er := by
  unfold Dace.bindViews
  refine Res.er_foldlM_congr (fun id _ st => ?_) st
  rw [State.node?_ren]
  cases s.node? id with
  | none => rfl
  | some n =>
    simp only [Option.map_some, DNode.ren_kind]
    cases hk : n.kind with
    | access d =>
      simp only [NodeKind.ren, h.isView]
      split
      · exact h.bindView hS d st
      · rfl
    | _ => rfl

omit hS in
theorem refSetTarget? (e : DEdge) : refSetTarget? ctx' (s.ren σ) (e.ren σ) = (refSetTarget? ctx s e).map σ := by
  unfold Dace.refSetTarget?
  simp only [State.dstRootEdge_ren, DEdge.ren_dstConn, DEdge.ren_dst, State.node?_ren]
  split
  · cases s.node? (s.dstRootEdge e).dst with
    | none => rfl
    | some n =>
      simp only [Option.map_some, Option.bind_some, DNode.ren_accessData?]
      cases n.accessData? with
      | none => rfl
      | some d => simp [Option.filter, h.isReference]
  · rfl

omit hS in
theorem isViewDef (e : DEdge) : ctx'.isViewDef (s.ren σ) (e.ren σ) = ctx.isViewDef s e := by
  rw [isViewDef_eq, isViewDef_eq]
  unfold isViewDefCore
  simp only [State.pathStart_ren, State.dstRootEdge_ren, DEdge.ren_src, DEdge.ren_dst, h.viewNodeOf]
  have hv : ∀ (o : Option DNode) (b : Bool) (r : DEdge),
      (match o.map (DNode.ren σ) with
       | some v => decide ((s.ren σ).viewEdge? v = some (r.ren σ, b))
       | none => false) =
      (match o with
       | some v => decide (s.viewEdge? v = some (r, b))
       | none => false) := by
    intro o b r
    cases o with
    | none => rfl
    | some v =>
      simp only [Option.map_some, State.viewEdge?_ren]
      congr 1
      apply propext
      constructor
      · intro hx
        cases hw : s.viewEdge? v with
        | none => rw [hw] at hx; cases hx
        | some p =>
          rw [hw] at hx
          obtain ⟨e₀, b₀⟩ := p
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hx
          rw [DEdge.ren_injective hx.1, hx.2]
      · intro hx; rw [hx]; rfl
  rw [hv, hv]

end RenCtx

/-! ## Simple nodes, scopes, nested SDFGs -/

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation} {s : State} (hS : RenState σ s)
include h hS

theorem refSet {e : DEdge} (he : e ∈ s.edges) (d : Name) (st : St) :
    (Dace.refSet ctx' ρ (s.ren σ) (e.ren σ) (σ d) st).er = (Dace.refSet ctx ρ s e d st).er := by
  unfold Dace.refSet
  rw [hS.srcEndpoint]
  cases hse : s.srcEndpoint e with
  | data b sub => exact h.aliasTo st d b (hS.srcSub he hse)
  | code k => rfl
  | invalid => rfl

theorem execCopiesInto (n : DNode) (st : St) :
    (Dace.execCopiesInto ctx' ρ (s.ren σ) (n.ren σ) st).er = (Dace.execCopiesInto ctx ρ s n st).er := by
  unfold Dace.execCopiesInto
  rw [DNode.ren_id, State.inEdges_ren, List.foldlM_map]
  refine Res.er_foldlM_congr (fun e he st => ?_) st
  have hmem := State.mem_inEdges he
  rcases e with ⟨a, b, c, d, dt⟩
  cases dt with
  | dep => rfl
  | memlet m =>
    simp only [DEdge.ren_data, EdgeData.ren_memlet]
    rw [h.isViewDef, State.isRouting_ren]
    split
    · rfl
    · rw [h.refSetTarget?]
      cases hr : Dace.refSetTarget? ctx s _ with
      | some d => exact h.refSet hS hmem d st
      | none =>
        simp only [Option.map_none]
        rw [hS.srcEndpoint]
        cases s.srcEndpoint _ with
        | data _ _ => exact h.execCopy hS hmem st
        | code _ => rfl
        | invalid => rfl

theorem execSimple {n : DNode} (hn : n ∈ s.nodes) (st : St) :
    (Dace.execSimple ctx' ρ (s.ren σ) (n.ren σ) st).er = (Dace.execSimple ctx ρ s n st).er := by
  unfold Dace.execSimple
  rw [DNode.ren_kind]
  cases hk : n.kind with
  | access d => exact h.execCopiesInto hS n st
  | mapExit => exact h.execCopiesInto hS n st
  | consumeExit => exact h.execCopiesInto hS n st
  | tasklet t => simp only [NodeKind.ren, h.I]; exact h.execCode hS n st _ _
  | library l =>
    simp only [NodeKind.ren]
    cases hr : l.reduce with
    | some props =>
      refine h.execReduce hS props (fun x hx => hS.expr x (State.mem_exprNames_node hn ?_)) st
      simp [DNode.exprNames, hk, hr, hx]
    | none => simp only [h.I]; exact h.execCode hS n st _ _
  | mapEntry _ => rfl
  | consumeEntry _ => rfl
  | nested _ _ _ => rfl

theorem bindDynamicInputs (n : DNode) (st : St) :
    (Dace.bindDynamicInputs ctx' ρ (s.ren σ) (n.ren σ) st).er = (Dace.bindDynamicInputs ctx ρ s n st).er := by
  unfold Dace.bindDynamicInputs
  rw [DNode.ren_id, State.inEdges_ren, List.foldlM_map]
  refine Res.er_foldlM_congr (fun e he ⟨ρ₁, st⟩ => ?_) _
  have hmem := State.mem_inEdges he
  rw [State.isDynamicInput_ren]
  by_cases hd : s.isDynamicInput e = true
  · simp only [hd, ↓reduceIte]
    rcases e with ⟨a, b, c, d, dt⟩
    cases dt <;> cases d <;> simp only [DEdge.ren_data, EdgeData.ren, DEdge.ren_dstConn] <;> try rfl
    exact Res.er_bind_congr (h.readEdgeScalar hS hmem st) fun _ => rfl
  · simp only [hd, Bool.false_eq_true, ↓reduceIte]

theorem mapSetup {n : DNode} (hn : n ∈ s.nodes) {mp : MapNode} (hk : n.kind = .mapEntry mp) (st : St) :
    (Dace.mapSetup ctx' ρ (s.ren σ) (n.ren σ) mp st).er = (Dace.mapSetup ctx ρ s n mp st).er := by
  unfold Dace.mapSetup
  refine Res.er_bind_congr (h.bindDynamicInputs hS n st) fun ⟨ρ', st'⟩ => ?_
  simp only
  rw [h.rangePoints_fix _ fun x hx => hS.expr x (State.mem_exprNames_node hn (by simp [DNode.exprNames, hk, hx]))]

theorem consumeStream (n : DNode) (m : Mem) :
    (Dace.consumeStream ctx' ρ (s.ren σ) (n.ren σ) m).er = (Dace.consumeStream ctx ρ s n m).er := by
  unfold Dace.consumeStream
  rw [DNode.ren_id, State.inEdges_ren, List.find?_map]
  have : ((fun e : DEdge => decide (e.dstConn = some "IN_stream")) ∘ DEdge.ren σ) =
      fun e => decide (e.dstConn = some "IN_stream") := rfl
  rw [this]
  cases hf : (s.inEdges n.id).find? (fun e => decide (e.dstConn = some "IN_stream")) with
  | none => rfl
  | some e =>
    have hmem := State.mem_inEdges (List.mem_of_find?_eq_some hf)
    simp only [Option.map_some]
    rw [hS.srcEndpoint]
    cases hse : s.srcEndpoint e with
    | data d sub =>
      simp only [State.Endpoint.ren_data', h.isStream]
      split
      · refine Res.er_bind_congr (h.streamPoint m d (hS.srcSub hmem hse)) fun p => ?_
        exact Res.er_bind_congr (h.loc m d p false) fun _ => rfl
      · rfl
    | code _ => rfl
    | invalid => rfl

theorem consumeSetup {n : DNode} (hn : n ∈ s.nodes) {c : ConsumeNode} (hk : n.kind = .consumeEntry c) (st : St) :
    (Dace.consumeSetup ctx' ρ (s.ren σ) (n.ren σ) c st).er = (Dace.consumeSetup ctx ρ s n c st).er := by
  unfold Dace.consumeSetup
  refine Res.er_bind_congr (h.bindDynamicInputs hS n st) fun ⟨ρ', st'⟩ => ?_
  refine Res.er_bind_congr (h.consumeStream hS n st'.mem) fun l => ?_
  rw [h.evalIntR_fix _ fun x hx => hS.expr x (State.mem_exprNames_node hn (by simp [DNode.exprNames, hk, hx]))]

theorem chunkMemName (entry : NodeId) : Dace.chunkMemName ctx' (s.ren σ) entry = Dace.chunkMemName ctx s entry := by
  unfold Dace.chunkMemName
  rw [State.consumedStream?_ren]
  cases hc : s.consumedStream? entry with
  | none => rfl
  | some d =>
    obtain ⟨h₁, h₂⟩ := hS.streams entry d hc
    simp only [Option.map_some, h₁]
    rw [← h₂, h.cname, h₂]

theorem taskStart (entry : NodeId) (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : St) (q : List Val) :
    Dace.taskStart ctx' (s.ren σ) entry c l priv pre q = Dace.taskStart ctx s entry c l priv pre q := by
  simp only [Dace.taskStart, h.chunkMemName hS]

omit hS in
theorem consumeCond {c : ConsumeNode} (hc : ∀ x ∈ (c.condition.map Expr.names).getD [], σ x = x) (m : Mem) (l : Loc)
    (q : List Val) : Dace.consumeCond ctx' ρ c m l q = Dace.consumeCond ctx ρ c m l q := by
  unfold Dace.consumeCond
  cases hcond : c.condition with
  | none => rfl
  | some e =>
    simp only
    rw [h.evalBoolR_fix _ fun x hx => hc x (by simp [hcond, hx])]

omit h in
theorem nestedEndpoint (n : DNode) (c : Conn) :
    Dace.nestedEndpoint (s.ren σ) (n.ren σ) c = (Dace.nestedEndpoint s n c).ren σ := by
  unfold Dace.nestedEndpoint
  rw [State.inEdgeOn_ren, State.outEdgesOn_ren]
  cases s.inEdgeOn n c with
  | some e => exact hS.srcEndpoint e
  | none =>
    cases s.outEdgesOn n c with
    | nil => rfl
    | cons e _ => exact State.dstEndpoint_ren σ s e

theorem nestedBindings (n : DNode) : Dace.nestedBindings ctx' (s.ren σ) (n.ren σ) = Dace.nestedBindings ctx s n := by
  unfold Dace.nestedBindings
  have hc : nestedConns (n.ren σ) = nestedConns n := rfl
  rw [hc]
  congr 1
  funext acc c
  rw [RenCtx.nestedEndpoint hS]
  cases Dace.nestedEndpoint s n c with
  | data b _ => simp [h.cname]
  | code _ => rfl
  | invalid => rfl

omit hS in
theorem enter (inner : SDFG) (conns : List (Name × Name)) : ctx'.enter inner conns = ctx.enter inner conns := by
  have : (fun n => ctx'.scope (inner.name ++ "/" ++ n)) = fun n => ctx.scope (inner.name ++ "/" ++ n) :=
    funext fun n => h.scope _ n
  simp only [Ctx.enter, this, h.I, h.scope]

end RenCtx

/-! ## Nested calls -/

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) {ρ : Valuation} {s : State} (hS : RenState σ s)
include h

theorem nestedValuation (m : Mem) {mapping : List (Sym × Expr)} (hm : ∀ p ∈ mapping, ∀ x ∈ p.2.names, σ x = x) :
    Dace.nestedValuation ctx' ρ m mapping = Dace.nestedValuation ctx ρ m mapping := by
  unfold Dace.nestedValuation
  congr 1
  apply mapM_congr_mem
  rintro ⟨x, e⟩ hp
  simp only
  rw [h.evalR_fix m (hm _ hp)]

omit h in
theorem bindCodeInputs (ctx₂ : Ctx) (ρ' : Valuation) (n : DNode) (st : St) (hS : RenState σ s) :
    Dace.bindCodeInputs ctx₂ ρ' (s.ren σ) (n.ren σ) st = Dace.bindCodeInputs ctx₂ ρ' s n st := by
  unfold Dace.bindCodeInputs
  have hc : nestedConns (n.ren σ) = nestedConns n := rfl
  rw [hc]
  congr 1
  funext acc c
  rw [RenCtx.nestedEndpoint hS]
  cases Dace.nestedEndpoint s n c <;> rfl

include hS in
theorem nestedSetup {n : DNode} (hn : n ∈ s.nodes) {inner : SDFG} {mapping : List (Sym × Expr)} {ni : Bool}
    (hk : n.kind = .nested inner mapping ni) (st : St) :
    Dace.nestedSetup ctx' ρ (s.ren σ) (n.ren σ) inner mapping st = Dace.nestedSetup ctx ρ s n inner mapping st := by
  unfold Dace.nestedSetup
  have hm : ∀ p ∈ mapping, ∀ x ∈ p.2.names, σ x = x := fun p hp x hx =>
    hS.expr x (State.mem_exprNames_node hn (by
      simp only [DNode.exprNames, hk, List.mem_flatMap, List.mem_cons]
      exact ⟨p, hp, .inr hx⟩))
  simp only [h.nestedBindings hS, h.nestedValuation _ hm, h.enter, bindCodeInputs _ _ _ _ hS]

omit h in
theorem nestedFinish (ctx₂ : Ctx) (ρ' : Valuation) (n : DNode) (dropped : List Name) (pre : St) (m₂ : Mem) (g₂ : Log) :
    Dace.nestedFinish ctx₂ ρ' (s.ren σ) (n.ren σ) dropped pre m₂ g₂ = Dace.nestedFinish ctx₂ ρ' s n dropped pre m₂ g₂ := by
  unfold Dace.nestedFinish
  rw [DNode.ren_id, State.outEdges_ren, List.foldlM_map]
  congr 1
  funext st e
  rcases e with ⟨a, b, c, d, dt⟩
  cases dt <;> cases b <;> simp only [DEdge.ren_data, EdgeData.ren, DEdge.ren_srcConn] <;> try rfl
  rw [State.dstEndpoint_ren]
  cases s.dstEndpoint _ <;> rfl

end RenCtx

/-! ## Mentions and transients -/

theorem DNode.mentions_ren {s : State} (hS : RenState σ s) {n : DNode} (hn : n ∈ s.nodes) (t : Name) :
    (n.ren σ).mentions (σ t) = n.mentions t := by
  by_cases ha : ∃ d, n.kind = .access d
  · obtain ⟨d, hd⟩ := ha
    simp [DNode.mentions, DNode.ren_kind, hd, NodeKind.ren, σ.injective.eq_iff]
  · have hr : n.ren σ = n := by
      rcases n with ⟨_, _, _, _, _, k⟩
      cases k <;> simp_all [DNode.ren, NodeKind.ren]
    rw [hr]
    have hk : ∀ x, n.mentions x = true → σ x = x := fun x hx => by
      rcases DNode.mentions_names hx with h | h
      · unfold DNode.dataNames at h
        split at h
        · exact absurd ⟨_, ‹_›⟩ ha
        · simp at h
      · exact hS.expr x (State.mem_exprNames_node hn h)
    by_cases ht : σ t = t
    · rw [ht]
    · cases h₁ : n.mentions t
      · cases h₂ : n.mentions (σ t)
        · rfl
        · exact absurd (σ.injective (hk _ h₂)) ht
      · exact absurd (hk _ h₁) ht

/-- The SDFGs of two related contexts: the same container names up to `σ`,
the same mention counts. -/
structure RenG (ctx' ctx : Ctx) : Prop where
  names : ctx'.containerNames = ctx.containerNames.map σ
  count : ∀ t, ctx'.g.countMentions (σ t) = ctx.g.countMentions t

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (hG : RenG σ ctx' ctx) {s : State} (hS : RenState σ s)
include h hG hS

omit h hG in
theorem countHere (t : Name) (p : NodeId → Bool) :
    ((s.ren σ).nodes.filter fun n => n.mentions (σ t) && p n.id).length =
      (s.nodes.filter fun n => n.mentions t && p n.id).length := by
  rw [State.ren_nodes, List.filter_map, List.length_map]
  congr 1
  apply List.filter_congr
  intro n hn
  simp only [Function.comp_def, DNode.ren_id, DNode.mentions_ren σ hS hn]

theorem privateTransients (entry : NodeId) :
    ctx'.privateTransients (s.ren σ) entry = ctx.privateTransients s entry := by
  unfold Ctx.privateTransients
  rw [State.scopeNodes_ren, hG.names, List.filterMap_map]
  apply List.filterMap_congr
  intro t _
  simp only [Function.comp_def, h.desc]
  cases ctx.desc? t with
  | none => rfl
  | some d =>
    simp only
    split
    · rw [RenCtx.countHere hS t, hG.count, h.cname]
    · rfl

theorem stateTransients : ctx'.stateTransients (s.ren σ) = (ctx.stateTransients s).map σ := by
  unfold Ctx.stateTransients
  rw [hG.names, List.filterMap_map, List.map_filterMap]
  apply List.filterMap_congr
  intro t _
  simp only [Function.comp_def, h.desc]
  cases ctx.desc? t with
  | none => rfl
  | some d =>
    simp only
    split
    · have := RenCtx.countHere hS t (fun _ => true)
      simp only [Bool.and_true] at this
      rw [this, hG.count]
      split <;> rfl
    · rfl

end RenCtx

theorem RenCtx.allocate {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (ts : List Name) (m : Mem) :
    ctx'.allocate (ts.map σ) m = ctx.allocate ts m := by
  unfold Ctx.allocate
  rw [List.foldl_map]
  congr 1
  funext m t
  rw [h.isStream, h.cname]

/-! ## Execution relations -/

namespace Res

variable {α : Type}

theorem er_err_inv {r' : Res α} {e : ErrReason} (h : r'.er = (err e).er) : ∃ e', r' = err e' ∧ e'.isUB = e.isUB := by
  cases r' with
  | ok a => simp [er] at h
  | err e' =>
    refine ⟨e', rfl, ?_⟩
    simp only [er, err.injEq] at h
    rw [← ErrReason.isUB_cls e', h, ErrReason.isUB_cls]

theorem er_ok_inv {r' : Res α} {a : α} (h : r'.er = (ok a).er) : r' = ok a := by
  cases r' with
  | ok b => simp only [er, ok.injEq] at h; rw [h]
  | err e => simp [er] at h

end Res

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (hG : RenG σ ctx' ctx) {s : State} (hS : RenState σ s)
include h hG hS

omit hG hS in
theorem scopeFinish (pre : St) (logs : List Log) : Dace.scopeFinish ctx' pre logs = Dace.scopeFinish ctx pre logs := by
  simp [Dace.scopeFinish, h.I]

omit hG hS in
theorem consumeFinish (pre : St) (l : Loc) (logs : List Log) (rest pushed : List Val) :
    Dace.consumeFinish ctx' pre l logs rest pushed = Dace.consumeFinish ctx pre l logs rest pushed := by
  simp [Dace.consumeFinish, h.I]

set_option linter.unusedSectionVars false

mutual

theorem nodeExec {ρ : Valuation} {n : DNode} {st : St} {r : Res St} :
    NodeExec ctx ρ s n st r → n ∈ s.nodes → ∃ r', NodeExec ctx' ρ (s.ren σ) (n.ren σ) st r' ∧ r'.er = r.er
  | .simple hs, hn => ⟨_, .simple (by rw [DNode.ren_isSimple]; exact hs), h.execSimple hS hn st⟩
  | .mapSetupErr (mp := mp) hk he, hn => by
    have hx := h.mapSetup (ρ := ρ) hS hn hk st
    rw [he] at hx
    obtain ⟨e', he', hu⟩ := Res.er_err_inv hx
    exact ⟨_, .mapSetupErr (by rw [DNode.ren_kind, hk]; rfl) he', Res.er_err_congr hu⟩
  | .mapIterErr (mp := mp) hk hs hi, hn => by
    have hx := h.mapSetup (ρ := ρ) hS hn hk st
    rw [hs] at hx
    obtain ⟨r₂, hi₂, hr⟩ := iterExec hi
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    rw [← h.privateTransients hG hS] at hi₂
    exact ⟨_, .mapIterErr (by rw [DNode.ren_kind, hk]; rfl) (Res.er_ok_inv hx) hi₂, Res.er_err_congr hu⟩
  | .mapScope (mp := mp) (logs := logs) hk hs hi, hn => by
    have hx := h.mapSetup (ρ := ρ) hS hn hk st
    rw [hs] at hx
    obtain ⟨r₂, hi₂, hr⟩ := iterExec hi
    rw [← h.privateTransients hG hS] at hi₂
    have hk' : (n.ren σ).kind = .mapEntry mp := by rw [DNode.ren_kind, hk]; rfl
    cases r₂ with
    | err e => exact ⟨_, .mapIterErr hk' (Res.er_ok_inv hx) hi₂, by simp [Res.er] at hr⟩
    | ok logs' =>
      cases Res.er_ok_inv hr
      exact ⟨_, .mapScope hk' (Res.er_ok_inv hx) hi₂, by rw [h.scopeFinish]⟩
  | .consumeSetupErr (c := c) hk he, hn => by
    have hx := h.consumeSetup (ρ := ρ) hS hn hk st
    rw [he] at hx
    obtain ⟨e', he', hu⟩ := Res.er_err_inv hx
    exact ⟨_, .consumeSetupErr (by rw [DNode.ren_kind, hk]; rfl) he', Res.er_err_congr hu⟩
  | .consumeTasksErr (c := c) hk hs ht, hn => by
    have hx := h.consumeSetup (ρ := ρ) hS hn hk st
    rw [hs] at hx
    have hc : ∀ x ∈ (c.condition.map Expr.names).getD [], σ x = x := fun x hx =>
      hS.expr x (State.mem_exprNames_node hn (by simp only [DNode.exprNames, hk, List.mem_append]; exact .inr hx))
    obtain ⟨r₂, ht₂, hr⟩ := consumeTasks hc ht
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    rw [← h.privateTransients hG hS] at ht₂
    exact ⟨_, .consumeTasksErr (by rw [DNode.ren_kind, hk]; rfl) (Res.er_ok_inv hx) ht₂, Res.er_err_congr hu⟩
  | .consumeScope (c := c) hk hs ht, hn => by
    have hx := h.consumeSetup (ρ := ρ) hS hn hk st
    rw [hs] at hx
    have hc : ∀ x ∈ (c.condition.map Expr.names).getD [], σ x = x := fun x hx =>
      hS.expr x (State.mem_exprNames_node hn (by simp only [DNode.exprNames, hk, List.mem_append]; exact .inr hx))
    obtain ⟨r₂, ht₂, hr⟩ := consumeTasks hc ht
    rw [← h.privateTransients hG hS] at ht₂
    have hk' : (n.ren σ).kind = .consumeEntry c := by rw [DNode.ren_kind, hk]; rfl
    cases r₂ with
    | err e => exact ⟨_, .consumeTasksErr hk' (Res.er_ok_inv hx) ht₂, by simp [Res.er] at hr⟩
    | ok x =>
      cases Res.er_ok_inv hr
      exact ⟨_, .consumeScope hk' (Res.er_ok_inv hx) ht₂, by rw [h.consumeFinish]⟩
  | .nestedSetupErr hk he, hn =>
    ⟨_, .nestedSetupErr (by rw [DNode.ren_kind, hk]; rfl) (by rw [h.nestedSetup hS hn hk]; exact he), rfl⟩
  | .nestedInnerErr hk hs hr, hn =>
    ⟨_, .nestedInnerErr (by rw [DNode.ren_kind, hk]; rfl) (by rw [h.nestedSetup hS hn hk]; exact hs) hr, rfl⟩
  | .nested hk hs hr, hn => by
    refine ⟨_, .nested (by rw [DNode.ren_kind, hk]; rfl) (by rw [h.nestedSetup hS hn hk]; exact hs) hr, ?_⟩
    rw [RenCtx.nestedFinish]

theorem seqExec {ρ : Valuation} {ids : List NodeId} {st : St} {r : Res (St × List (NodeId × Log))} :
    SeqExec ctx ρ s ids st r → ∃ r', SeqExec ctx' ρ (s.ren σ) ids st r' ∧ r'.er = r.er
  | .nil => ⟨_, .nil, rfl⟩
  | .missing hm => ⟨_, .missing (by rw [State.node?_ren, hm]; rfl), rfl⟩
  | .headErr hn hx => by
    obtain ⟨r₂, hx₂, hr⟩ := nodeExec hx (State.node?_mem hn)
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    exact ⟨_, .headErr (by rw [State.node?_ren, hn]; rfl) hx₂, Res.er_err_congr hu⟩
  | .tailErr (id := id) (n := n) hn hx hrest => by
    obtain ⟨r₂, hx₂, hr⟩ := nodeExec hx (State.node?_mem hn)
    have hn' : (s.ren σ).node? id = some (n.ren σ) := by rw [State.node?_ren, hn]; rfl
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hrest₂, hr₃⟩ := seqExec hrest
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr₃
    exact ⟨_, .tailErr hn' hx₂ hrest₂, Res.er_err_congr hu⟩
  | .cons (id := id) (n := n) hn hx hrest => by
    obtain ⟨r₂, hx₂, hr⟩ := nodeExec hx (State.node?_mem hn)
    have hn' : (s.ren σ).node? id = some (n.ren σ) := by rw [State.node?_ren, hn]; rfl
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hrest₂, hr₃⟩ := seqExec hrest
    cases Res.er_ok_inv hr₃
    exact ⟨_, .cons hn' hx₂ hrest₂, rfl⟩

theorem orderedExec {ρ : Valuation} {all direct : List NodeId} {st : St} {r : Res St} :
    OrderedExec ctx ρ s all direct st r → ∃ r', OrderedExec ctx' ρ (s.ren σ) all direct st r' ∧ r'.er = r.er
  | .prologueErr (e := e) hp => by
    have hx := h.bindViews hS (ρ := ρ) (s.viewOrder ctx direct) st
    rw [hp] at hx
    obtain ⟨e', he', hu⟩ := Res.er_err_inv hx
    exact ⟨_, .prologueErr (by rw [h.viewOrder]; exact he'), Res.er_err_congr hu⟩
  | .seqErr ho hp hs => by
    have hx := h.bindViews hS (ρ := ρ) (s.viewOrder ctx direct) st
    rw [hp] at hx
    obtain ⟨r₂, hs₂, hr⟩ := seqExec hs
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    exact ⟨_, .seqErr (by rw [State.isTopologicalOrder_ren]; exact ho) (by rw [h.viewOrder]; exact Res.er_ok_inv hx)
      hs₂, Res.er_err_congr hu⟩
  | .hbRace (tr := tr) ho hp hs hf => by
    have hx := h.bindViews hS (ρ := ρ) (s.viewOrder ctx direct) st
    rw [hp] at hx
    obtain ⟨r₂, hs₂, hr⟩ := seqExec hs
    cases Res.er_ok_inv hr
    refine ⟨_, .hbRace (by rw [State.isTopologicalOrder_ren]; exact ho) (by rw [h.viewOrder]; exact Res.er_ok_inv hx)
      hs₂ (by rw [State.hbFree_sren]; exact hf), ?_⟩
    rw [State.hbWitness_sren]
  | .run ho hp hs hf => by
    have hx := h.bindViews hS (ρ := ρ) (s.viewOrder ctx direct) st
    rw [hp] at hx
    obtain ⟨r₂, hs₂, hr⟩ := seqExec hs
    cases Res.er_ok_inv hr
    exact ⟨_, .run (by rw [State.isTopologicalOrder_ren]; exact ho) (by rw [h.viewOrder]; exact Res.er_ok_inv hx)
      hs₂ (by rw [State.hbFree_sren]; exact hf), rfl⟩

theorem iterExec {ρ : Valuation} {P : NodeId} {params : List Sym} {priv : List Name} {pre : St} {pts : List Idx}
    {r : Res (List Log)} :
    IterExec ctx ρ s P params priv pre pts r → ∃ r', IterExec ctx' ρ (s.ren σ) P params priv pre pts r' ∧ r'.er = r.er
  | .nil => ⟨_, .nil, rfl⟩
  | .headErr hb => by
    obtain ⟨r₂, hb₂, hr⟩ := orderedExec hb
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    rw [← State.scopeNodes_ren (σ := σ), ← State.directChildren_ren (σ := σ)] at hb₂
    exact ⟨_, .headErr hb₂, Res.er_err_congr hu⟩
  | .tailErr hrest => by
    obtain ⟨r₂, hr₂, hr⟩ := iterExec hrest
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    exact ⟨_, .tailErr hr₂, Res.er_err_congr hu⟩
  | .cons hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := orderedExec hb
    rw [← State.scopeNodes_ren (σ := σ), ← State.directChildren_ren (σ := σ)] at hb₂
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hr₃, hr'⟩ := iterExec hrest
    cases r₃ with
    | err e => exact ⟨_, .tailErr hr₃, by simp [Res.er] at hr'⟩
    | ok l => cases Res.er_ok_inv hr'; exact ⟨_, .cons hb₂ hr₃, rfl⟩

theorem consumeTasks {ρ : Valuation} {P : NodeId} {c : ConsumeNode}
    (hc : ∀ x ∈ (c.condition.map Expr.names).getD [], σ x = x) {l : Loc} {np : Int} {priv : List Name} {pre : St}
    {q : List Val} {r : Res (List Log × List Val × List Val)} :
    ConsumeTasks ctx ρ s P c l np priv pre q r →
      ∃ r', ConsumeTasks ctx' ρ (s.ren σ) P c l np priv pre q r' ∧ r'.er = r.er
  | .condErr hcond => ⟨_, .condErr (by rw [h.consumeCond hc]; exact hcond), rfl⟩
  | .stop hcond => ⟨_, .stop (by rw [h.consumeCond hc]; exact hcond), rfl⟩
  | .bodyErr hcond hq hp hb => by
    obtain ⟨r₂, hb₂, hr⟩ := orderedExec hb
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    rw [← State.scopeNodes_ren (σ := σ), ← State.directChildren_ren (σ := σ), ← h.taskStart hS] at hb₂
    exact ⟨_, .bodyErr (by rw [h.consumeCond hc]; exact hcond) hq hp hb₂, Res.er_err_congr hu⟩
  | .restErr hcond hq hp hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := orderedExec hb
    rw [← State.scopeNodes_ren (σ := σ), ← State.directChildren_ren (σ := σ), ← h.taskStart hS] at hb₂
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hr₃, hr'⟩ := consumeTasks hc hrest
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr'
    exact ⟨_, .restErr (by rw [h.consumeCond hc]; exact hcond) hq hp hb₂ hr₃, Res.er_err_congr hu⟩
  | .step hcond hq hp hb hrest => by
    obtain ⟨r₂, hb₂, hr⟩ := orderedExec hb
    rw [← State.scopeNodes_ren (σ := σ), ← State.directChildren_ren (σ := σ), ← h.taskStart hS] at hb₂
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hr₃, hr'⟩ := consumeTasks hc hrest
    cases r₃ with
    | err e => exact ⟨_, .restErr (by rw [h.consumeCond hc]; exact hcond) hq hp hb₂ hr₃, by simp [Res.er] at hr'⟩
    | ok x =>
      cases Res.er_ok_inv hr'
      exact ⟨_, .step (by rw [h.consumeCond hc]; exact hcond) hq hp hb₂ hr₃, rfl⟩

end

end RenCtx

/-! ## States -/

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (hG : RenG σ ctx' ctx) {s : State} (hS : RenState σ s)
include h hG hS

theorem compsExec {ρ : Valuation} {m : Mem} :
    ∀ {comps : List (List NodeId)} {r : Res (List Log)}, CompsExec ctx ρ s m comps r →
      ∃ r', CompsExec ctx' ρ (s.ren σ) m comps r' ∧ r'.er = r.er
  | _, _, .nil => ⟨_, .nil, rfl⟩
  | _, _, .headErr hc => by
    obtain ⟨r₂, hc₂, hr⟩ := h.orderedExec hG hS hc
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    rw [← State.topLevelOf_ren (σ := σ)] at hc₂
    exact ⟨_, .headErr hc₂, Res.er_err_congr hu⟩
  | _, _, .tailErr hrest => by
    obtain ⟨r₂, hr₂, hr⟩ := compsExec hrest
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    exact ⟨_, .tailErr hr₂, Res.er_err_congr hu⟩
  | _, _, .cons hc hrest => by
    obtain ⟨r₂, hc₂, hr⟩ := h.orderedExec hG hS hc
    rw [← State.topLevelOf_ren (σ := σ)] at hc₂
    cases Res.er_ok_inv hr
    obtain ⟨r₃, hr₃, hr'⟩ := compsExec hrest
    cases r₃ with
    | err e => exact ⟨_, .tailErr hr₃, by simp [Res.er] at hr'⟩
    | ok l => cases Res.er_ok_inv hr'; exact ⟨_, .cons hc₂ hr₃, rfl⟩

theorem stateExec {cst : CSt} {r : Res (Ctl × CSt)} :
    StateExec ctx cst s r → ∃ r', StateExec ctx' cst (s.ren σ) r' ∧ r'.er = r.er
  | .compsErr hc => by
    rw [← h.allocate, ← h.stateTransients hG hS, ← State.components_ren (σ := σ)] at hc
    obtain ⟨r₂, hc₂, hr⟩ := h.compsExec hG hS hc
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hr
    exact ⟨_, .compsErr hc₂, Res.er_err_congr hu⟩
  | .run hc => by
    rw [← h.allocate, ← h.stateTransients hG hS, ← State.components_ren (σ := σ)] at hc
    obtain ⟨r₂, hc₂, hr⟩ := h.compsExec hG hS hc
    cases r₂ with
    | err e => exact ⟨_, .compsErr hc₂, by simp [Res.er] at hr⟩
    | ok logs =>
      cases Res.er_ok_inv hr
      exact ⟨_, .run hc₂, by simp [stateFinish, h.I]⟩

end RenCtx

/-! ## Control flow -/

/-- The names an interstate edge reads. -/
def IEdge.exprNamesR (e : IEdge) : List Name := e.cond.names ++ e.assigns.flatMap fun (_, x) => x.names

section RenOk

mutual

/-- Every expression of a region reads only names `σ` fixes, and its states
satisfy `RenState`. -/
def Region.RenOk : Region → Prop
  | ⟨_, _, blocks, _, edges⟩ => Blocks.RenOk blocks ∧ ∀ e ∈ edges, ∀ x ∈ e.exprNamesR, σ x = x

def Blocks.RenOk : List Block → Prop
  | [] => True
  | b :: bs => Block.RenOk b ∧ Blocks.RenOk bs

def Block.RenOk : Block → Prop
  | .state s => RenState σ s
  | .region r => Region.RenOk r
  | .loop ⟨_, _, init, cond, update, _, _, body⟩ =>
    (∀ a ∈ init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ cond.names, σ x = x) ∧
      (∀ a ∈ update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk body
  | .cond _ brs => Branches.RenOk brs
  | .breakBlock | .continueBlock | .returnBlock => True

def Branches.RenOk : List (Option Expr × Region) → Prop
  | [] => True
  | (c, r) :: rest => (∀ e ∈ c, ∀ x ∈ e.names, σ x = x) ∧ Region.RenOk r ∧ Branches.RenOk rest

end

end RenOk

theorem Blocks.RenOk.get : ∀ {bs : List Block} {i : Nat} {b : Block}, Blocks.RenOk σ bs → bs[i]? = some b →
    Block.RenOk σ b
  | [], _, _, _, h => by simp at h
  | b :: bs, 0, b', hb, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact hb.1
  | b :: bs, i + 1, b', hb, h => by
    simp only [List.getElem?_cons_succ] at h
    exact Blocks.RenOk.get hb.2 h

theorem Region.RenOk.block {rg : Region} (h : Region.RenOk σ rg) {bid : BlockId} {b : Block}
    (hb : rg.block? bid = some b) : Block.RenOk σ b := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg
  exact Blocks.RenOk.get σ h.1 hb

theorem Region.RenOk.edges {rg : Region} (h : Region.RenOk σ rg) : ∀ e ∈ rg.edges, ∀ x ∈ e.exprNamesR, σ x = x := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg
  exact h.2

theorem Blocks.ren_get : ∀ (bs : List Block) (i : Nat), (Blocks.ren σ bs)[i]? = (bs[i]?).map (Block.ren σ)
  | [], _ => rfl
  | b :: bs, 0 => rfl
  | b :: bs, i + 1 => Blocks.ren_get bs i

theorem Region.block?_ren (rg : Region) (bid : BlockId) : (rg.ren σ).block? bid = (rg.block? bid).map (Block.ren σ) := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg
  exact Blocks.ren_get σ blocks bid

@[simp] theorem Region.ren_edges (rg : Region) : (rg.ren σ).edges = rg.edges := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg; rfl

@[simp] theorem Region.ren_start (rg : Region) : (rg.ren σ).start = rg.start := by
  obtain ⟨_, _, blocks, _, edges⟩ := rg; rfl

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx)
include h

theorem trueEdges {rg : Region} (hr : ∀ e ∈ rg.edges, ∀ x ∈ e.exprNamesR, σ x = x) (cst : CSt) (bid : BlockId) :
    Dace.trueEdges ctx' (rg.ren σ) cst bid = Dace.trueEdges ctx rg cst bid := by
  unfold Dace.trueEdges Region.outEdges
  rw [Region.ren_edges]
  have key : ∀ l : List IEdge, (∀ e ∈ l, e ∈ rg.edges) →
      l.filterMapM (fun e => do
        let b ← ctx'.evalBoolR cst.ρ cst.mem e.cond
        pure (if b then some e else none)) =
      l.filterMapM (fun e => do
        let b ← ctx.evalBoolR cst.ρ cst.mem e.cond
        pure (if b then some e else none)) := by
    intro l hl
    induction l with
    | nil => rfl
    | cons e es ih =>
      simp only [List.filterMapM_cons]
      rw [h.evalBoolR_fix _ fun x hx => hr e (hl e List.mem_cons_self) x (by simp [IEdge.exprNamesR, hx]),
        ih fun x hx => hl x (List.mem_cons_of_mem _ hx)]
  exact key _ fun e he => (List.mem_filter.1 he).1

theorem applyAssigns (cst : CSt) {as : List (Sym × Expr)} (ha : ∀ a ∈ as, ∀ x ∈ a.2.names, σ x = x) :
    Dace.applyAssigns ctx' cst as = Dace.applyAssigns ctx cst as := by
  unfold Dace.applyAssigns
  congr 1
  apply mapM_congr_mem
  rintro ⟨x, e⟩ hp
  simp only
  rw [h.evalR_fix _ (ha _ hp)]

theorem applyAssign? (cst : CSt) {a : Option (Sym × Expr)} (ha : ∀ p ∈ a, ∀ x ∈ p.2.names, σ x = x) :
    Dace.applyAssign? ctx' cst a = Dace.applyAssign? ctx cst a := by
  cases a with
  | none => rfl
  | some p =>
    refine h.applyAssigns cst fun q hq => ?_
    simp only [List.mem_singleton] at hq
    subst hq
    exact ha _ rfl

end RenCtx

@[simp] theorem LoopRegion.ren_init (l : LoopRegion) : (l.ren σ).init = l.init := by cases l; rfl
@[simp] theorem LoopRegion.ren_cond (l : LoopRegion) : (l.ren σ).cond = l.cond := by cases l; rfl
@[simp] theorem LoopRegion.ren_update (l : LoopRegion) : (l.ren σ).update = l.update := by cases l; rfl
@[simp] theorem LoopRegion.ren_inverted (l : LoopRegion) : (l.ren σ).inverted = l.inverted := by cases l; rfl
@[simp] theorem LoopRegion.ren_updateBeforeCond (l : LoopRegion) : (l.ren σ).updateBeforeCond = l.updateBeforeCond := by
  cases l; rfl
@[simp] theorem LoopRegion.ren_body (l : LoopRegion) : (l.ren σ).body = l.body.ren σ := by cases l; rfl

theorem Block.RenOk.loop' {l : LoopRegion} (h : Block.RenOk σ (.loop l)) :
    (∀ a ∈ l.init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ l.cond.names, σ x = x) ∧
      (∀ a ∈ l.update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk σ l.body := by
  cases l; exact h

namespace RenCtx

variable {σ} {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (hG : RenG σ ctx' ctx)
include h hG

set_option linter.unusedSectionVars false

mutual

theorem blockExec {cst : CSt} {b : Block} {r : Res (Ctl × CSt)} :
    BlockExec ctx cst b r → Block.RenOk σ b → ∃ r', BlockExec ctx' cst (b.ren σ) r' ∧ r'.er = r.er
  | .state hx, hb => by
    obtain ⟨r', hx', hr⟩ := h.stateExec hG hb hx
    exact ⟨r', .state hx', hr⟩
  | .region hx, hb => by
    obtain ⟨r', hx', hr⟩ := regionExec hx hb
    exact ⟨r', .region hx', hr⟩
  | .loop hx, hb => by
    obtain ⟨r', hx', hr⟩ := loopExec hx (Block.RenOk.loop' σ hb)
    exact ⟨r', .loop hx', hr⟩
  | .cond hx, hb => by
    obtain ⟨r', hx', hr⟩ := condExec hx hb
    exact ⟨r', .cond hx', hr⟩
  | .brk, _ => ⟨_, .brk, rfl⟩
  | .cont, _ => ⟨_, .cont, rfl⟩
  | .ret, _ => ⟨_, .ret, rfl⟩

theorem regionAt {rg : Region} {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)} :
    RegionAt ctx rg cst bid r → Region.RenOk σ rg → ∃ r', RegionAt ctx' (rg.ren σ) cst bid r' ∧ r'.er = r.er
  | .missing hb, _ => ⟨_, .missing (by rw [Region.block?_ren, hb]; rfl), rfl⟩
  | .blockErr hb hx, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hres
    exact ⟨_, .blockErr (by rw [Region.block?_ren, hb]; rfl) hx', Res.er_err_congr hu⟩
  | .exit hb hx hc, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    exact ⟨_, .exit (by rw [Region.block?_ren, hb]; rfl) hx' hc, rfl⟩
  | .edgesErr hb hx he, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    exact ⟨_, .edgesErr (by rw [Region.block?_ren, hb]; rfl) hx' (by rw [h.trueEdges (hr.edges σ)]; exact he), rfl⟩
  | .stop hb hx he, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    exact ⟨_, .stop (by rw [Region.block?_ren, hb]; rfl) hx' (by rw [h.trueEdges (hr.edges σ)]; exact he), rfl⟩
  | .ambiguous hb hx he, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    exact ⟨_, .ambiguous (by rw [Region.block?_ren, hb]; rfl) hx' (by rw [h.trueEdges (hr.edges σ)]; exact he), rfl⟩
  | .assignErr (ie := ie) hb hx he ha, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    have hie : ie ∈ rg.edges := trueEdges_cons_mem he
    refine ⟨_, .assignErr (by rw [Region.block?_ren, hb]; rfl) hx' (by rw [h.trueEdges (hr.edges σ)]; exact he)
      (by rw [h.applyAssigns _ fun a ha' x hx => hr.edges σ ie hie x (by
        simp only [IEdge.exprNamesR, List.mem_append, List.mem_flatMap]; exact .inr ⟨a, ha', hx⟩)]; exact ha), rfl⟩
  | .step (ie := ie) hb hx he ha hn, hr => by
    obtain ⟨r', hx', hres⟩ := blockExec hx (hr.block σ hb)
    cases Res.er_ok_inv hres
    have hie : ie ∈ rg.edges := trueEdges_cons_mem he
    obtain ⟨r₂, hn', hr₂⟩ := regionAt hn hr
    refine ⟨_, .step (by rw [Region.block?_ren, hb]; rfl) hx' (by rw [h.trueEdges (hr.edges σ)]; exact he)
      (by rw [h.applyAssigns _ fun a ha' x hx => hr.edges σ ie hie x (by
        simp only [IEdge.exprNamesR, List.mem_append, List.mem_flatMap]; exact .inr ⟨a, ha', hx⟩)]; exact ha) hn', hr₂⟩

theorem regionExec {rg : Region} {cst : CSt} {r : Res (Ctl × CSt)} :
    RegionExec ctx cst rg r → Region.RenOk σ rg → ∃ r', RegionExec ctx' cst (rg.ren σ) r' ∧ r'.er = r.er
  | .start hx, hr => by
    obtain ⟨r', hx', hres⟩ := regionAt hx hr
    rw [← Region.ren_start σ] at hx'
    exact ⟨r', .start hx', hres⟩

theorem loopExec {l : LoopRegion} {cst : CSt} {r : Res (Ctl × CSt)} :
    LoopExec ctx cst l r →
      ((∀ a ∈ l.init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ l.cond.names, σ x = x) ∧
        (∀ a ∈ l.update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk σ l.body) →
      ∃ r', LoopExec ctx' cst (l.ren σ) r' ∧ r'.er = r.er
  | .initErr hi, hl => ⟨_, .initErr (by rw [LoopRegion.ren_init, h.applyAssign? _ hl.1]; exact hi), rfl⟩
  | .guardFirst hi hv hg, hl => by
    obtain ⟨r', hg', hres⟩ := loopGuard hg hl
    exact ⟨r', .guardFirst (by rw [LoopRegion.ren_init, h.applyAssign? _ hl.1]; exact hi)
      (by rw [LoopRegion.ren_inverted]; exact hv) hg', hres⟩
  | .bodyFirst hi hv hb, hl => by
    obtain ⟨r', hb', hres⟩ := loopIter hb hl
    exact ⟨r', .bodyFirst (by rw [LoopRegion.ren_init, h.applyAssign? _ hl.1]; exact hi)
      (by rw [LoopRegion.ren_inverted]; exact hv) hb', hres⟩

theorem loopGuard {l : LoopRegion} {cst : CSt} {r : Res (Ctl × CSt)} :
    LoopGuard ctx cst l r →
      ((∀ a ∈ l.init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ l.cond.names, σ x = x) ∧
        (∀ a ∈ l.update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk σ l.body) →
      ∃ r', LoopGuard ctx' cst (l.ren σ) r' ∧ r'.er = r.er
  | .condErr hc, hl => ⟨_, .condErr (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc), rfl⟩
  | .exit hc, hl => ⟨_, .exit (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc), rfl⟩
  | .enter hc hb, hl => by
    obtain ⟨r', hb', hres⟩ := loopIter hb hl
    exact ⟨r', .enter (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc) hb', hres⟩

theorem loopIter {l : LoopRegion} {cst : CSt} {r : Res (Ctl × CSt)} :
    LoopIter ctx cst l r →
      ((∀ a ∈ l.init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ l.cond.names, σ x = x) ∧
        (∀ a ∈ l.update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk σ l.body) →
      ∃ r', LoopIter ctx' cst (l.ren σ) r' ∧ r'.er = r.er
  | .bodyErr hb, hl => by
    obtain ⟨r', hb', hres⟩ := regionExec hb hl.2.2.2
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hres
    exact ⟨_, .bodyErr (by rw [LoopRegion.ren_body]; exact hb'), Res.er_err_congr hu⟩
  | .ret hb, hl => by
    obtain ⟨r', hb', hres⟩ := regionExec hb hl.2.2.2
    cases Res.er_ok_inv hres
    exact ⟨_, .ret (by rw [LoopRegion.ren_body]; exact hb'), rfl⟩
  | .brk hb, hl => by
    obtain ⟨r', hb', hres⟩ := regionExec hb hl.2.2.2
    cases Res.er_ok_inv hres
    exact ⟨_, .brk (by rw [LoopRegion.ren_body]; exact hb'), rfl⟩
  | .tail hb hc ht, hl => by
    obtain ⟨r', hb', hres⟩ := regionExec hb hl.2.2.2
    cases Res.er_ok_inv hres
    obtain ⟨r₂, ht', hr₂⟩ := loopTail ht hl
    exact ⟨_, .tail (by rw [LoopRegion.ren_body]; exact hb') hc ht', hr₂⟩

theorem loopTail {l : LoopRegion} {cst : CSt} {r : Res (Ctl × CSt)} :
    LoopTail ctx cst l r →
      ((∀ a ∈ l.init, ∀ x ∈ a.2.names, σ x = x) ∧ (∀ x ∈ l.cond.names, σ x = x) ∧
        (∀ a ∈ l.update, ∀ x ∈ a.2.names, σ x = x) ∧ Region.RenOk σ l.body) →
      ∃ r', LoopTail ctx' cst (l.ren σ) r' ∧ r'.er = r.er
  | .updateErr hv hu, hl => ⟨_, .updateErr (by simpa using hv)
      (by rw [LoopRegion.ren_update, h.applyAssign? _ hl.2.2.1]; exact hu), rfl⟩
  | .updateGuard hv hu hg, hl => by
    obtain ⟨r', hg', hres⟩ := loopGuard hg hl
    exact ⟨r', .updateGuard (by simpa using hv) (by rw [LoopRegion.ren_update, h.applyAssign? _ hl.2.2.1]; exact hu)
      hg', hres⟩
  | .condFirstErr hv hc, hl => ⟨_, .condFirstErr (by simpa using hv)
      (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc), rfl⟩
  | .condFirstExit hv hc, hl => ⟨_, .condFirstExit (by simpa using hv)
      (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc), rfl⟩
  | .condFirstUpdateErr hv hc hu, hl => ⟨_, .condFirstUpdateErr (by simpa using hv)
      (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc)
      (by rw [LoopRegion.ren_update, h.applyAssign? _ hl.2.2.1]; exact hu), rfl⟩
  | .condFirstNext hv hc hu hb, hl => by
    obtain ⟨r', hb', hres⟩ := loopIter hb hl
    exact ⟨r', .condFirstNext (by simpa using hv) (by rw [LoopRegion.ren_cond, h.evalBoolR_fix _ hl.2.1]; exact hc)
      (by rw [LoopRegion.ren_update, h.applyAssign? _ hl.2.2.1]; exact hu) hb', hres⟩

theorem condExec {brs : List (Option Expr × Region)} {cst : CSt} {r : Res (Ctl × CSt)} :
    CondExec ctx cst brs r → Branches.RenOk σ brs → ∃ r', CondExec ctx' cst (Branches.ren σ brs) r' ∧ r'.er = r.er
  | .nil, _ => ⟨_, .nil, rfl⟩
  | .els hx, hb => by
    obtain ⟨r', hx', hres⟩ := regionExec hx hb.2.1
    exact ⟨r', .els hx', hres⟩
  | .condErr hc, hb => ⟨_, .condErr (by rw [h.evalBoolR_fix _ (hb.1 _ rfl)]; exact hc), rfl⟩
  | .taken hc hx, hb => by
    obtain ⟨r', hx', hres⟩ := regionExec hx hb.2.1
    exact ⟨r', .taken (by rw [h.evalBoolR_fix _ (hb.1 _ rfl)]; exact hc) hx', hres⟩
  | .skipped hc hx, hb => by
    obtain ⟨r', hx', hres⟩ := condExec hx hb.2.2
    exact ⟨r', .skipped (by rw [h.evalBoolR_fix _ (hb.1 _ rfl)]; exact hc) hx', hres⟩

end

end RenCtx

/-! ## SDFGs -/

theorem RenCtx.sdfgExec {ctx' ctx : Ctx} (h : RenCtx σ ctx' ctx) (hG : RenG σ ctx' ctx) {g : SDFG}
    (hg : Region.RenOk σ g.top) {ρ : Valuation} {m : Mem} {r : Res (Valuation × Mem × Log)} :
    SDFGExec ctx ρ m g r → ∃ r', SDFGExec ctx' ρ m (g.ren σ) r' ∧ r'.er = r.er
  | .err hx => by
    obtain ⟨r', hx', hres⟩ := h.regionExec hG hx hg
    obtain ⟨e', rfl, hu⟩ := Res.er_err_inv hres
    exact ⟨_, .err hx', Res.er_err_congr hu⟩
  | .escape hx hc => by
    obtain ⟨r', hx', hres⟩ := h.regionExec hG hx hg
    cases Res.er_ok_inv hres
    exact ⟨_, .escape hx' hc, rfl⟩
  | .run hx hc => by
    obtain ⟨r', hx', hres⟩ := h.regionExec hG hx hg
    cases Res.er_ok_inv hres
    exact ⟨_, .run hx' hc, rfl⟩

theorem List.eraseDups_map_inj {α β : Type} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] {f : α → β}
    (hf : Function.Injective f) : ∀ (l : List α), (l.map f).eraseDups = l.eraseDups.map f
  | [] => rfl
  | a :: as => by
    rw [List.map_cons, List.eraseDups_cons, List.eraseDups_cons, List.map_cons]
    congr 1
    have : (as.map f).filter (fun b => !b == f a) = (as.filter fun b => !b == a).map f := by
      rw [List.filter_map]
      congr 1
      apply List.filter_congr
      intro x _
      simp [Function.comp_def, hf.eq_iff]
    rw [this]
    have : (as.filter fun b => !b == a).length < as.length + 1 := Nat.lt_add_one_of_le (List.length_filter_le _ as)
    exact List.eraseDups_map_inj hf _
termination_by l => l.length

theorem SDFG.array?_ren (g : SDFG) (n : Name) : (g.ren σ).array? (σ n) = g.array? n := by
  simp only [SDFG.array?, SDFG.ren]
  induction g.arrays with
  | nil => rfl
  | cons a as ih =>
    obtain ⟨k, d⟩ := a
    by_cases hk : k = n
    · subst hk; simp
    · have : ¬ σ k = σ n := fun h => hk (σ.injective h)
      simp only [List.map_cons, List.find?_cons, this, hk, decide_false]
      exact ih

theorem SDFG.containerNames_ren (g : SDFG) :
    ((g.ren σ).arrays.map Prod.fst).eraseDups = (g.arrays.map Prod.fst).eraseDups.map σ := by
  simp only [SDFG.ren, List.map_map]
  rw [← List.eraseDups_map_inj σ.injective, List.map_map]
  rfl

/-- A Boolean property of names that holds only for names `σ` fixes takes
the same value at `σ t` and `t`. -/
theorem mention_fix {p : Name → Bool} (hp : ∀ x, p x = true → σ x = x) (t : Name) : p (σ t) = p t := by
  by_cases ht : σ t = t
  · rw [ht]
  · cases h₁ : p t
    · cases h₂ : p (σ t)
      · rfl
      · exact absurd (σ.injective (hp _ h₂)) ht
    · exact absurd (hp _ h₁) ht

theorem IEdge.mentions_fix {e : IEdge} (he : ∀ x ∈ e.exprNamesR, σ x = x) (t : Name) :
    e.mentions (σ t) = e.mentions t := by
  refine mention_fix σ (p := e.mentions) (fun x hx => he x ?_) t
  simp only [IEdge.mentions, Bool.or_eq_true, List.any_eq_true] at hx
  simp only [IEdge.exprNamesR, List.mem_append, List.mem_flatMap]
  rcases hx with hx | ⟨a, ha, hx⟩
  · exact .inl (Expr.mem_names_of_mentions hx)
  · exact .inr ⟨a, ha, Expr.mem_names_of_mentions hx⟩

mutual

theorem Region.countMentions_ren : (rg : Region) → Region.RenOk σ rg → ∀ t,
    Region.countMentions (σ t) (rg.ren σ) = Region.countMentions t rg
  | ⟨_, _, blocks, _, edges⟩, h, t => by
    simp only [Region.ren, Region.countMentions]
    rw [Blocks.countMentions_ren blocks h.1 t]
    congr 1
    congr 1
    exact List.filter_congr fun e he => IEdge.mentions_fix σ (h.2 e he) t

theorem Blocks.countMentions_ren : (bs : List Block) → Blocks.RenOk σ bs → ∀ t,
    Blocks.countMentions (σ t) (Blocks.ren σ bs) = Blocks.countMentions t bs
  | [], _, _ => rfl
  | b :: bs, h, t => by
    simp only [Blocks.ren, Blocks.countMentions]
    rw [Block.countMentions_ren b h.1 t, Blocks.countMentions_ren bs h.2 t]

theorem Block.countMentions_ren : (b : Block) → Block.RenOk σ b → ∀ t,
    Block.countMentions (σ t) (b.ren σ) = Block.countMentions t b
  | .state s, h, t => by
    simp only [Block.ren, Block.countMentions]
    have := RenCtx.countHere (σ := σ) h t (fun _ => true)
    simpa using this
  | .region r, h, t => by
    simp only [Block.ren, Block.countMentions]
    exact Region.countMentions_ren r h t
  | .loop ⟨l, v, init, cond, update, inv, ubc, body⟩, h, t => by
    simp only [Block.ren, LoopRegion.ren, Block.countMentions]
    obtain ⟨hi, hc, hu, hb⟩ := h
    rw [Region.countMentions_ren body hb t]
    have hp : (cond.mentions (σ t) || init.any (·.2.mentions (σ t)) || update.any (·.2.mentions (σ t))) =
        (cond.mentions t || init.any (·.2.mentions t) || update.any (·.2.mentions t)) := by
      refine mention_fix σ (p := fun t => cond.mentions t || init.any (·.2.mentions t) || update.any (·.2.mentions t))
        (fun x hx => ?_) t
      simp only [Bool.or_eq_true, Option.any_eq_true] at hx
      rcases hx with (hx | ⟨a, ha, hx⟩) | ⟨a, ha, hx⟩
      · exact hc x (Expr.mem_names_of_mentions hx)
      · exact hi a ha x (Expr.mem_names_of_mentions hx)
      · exact hu a ha x (Expr.mem_names_of_mentions hx)
    rw [hp]
  | .cond lb brs, h, t => by
    simp only [Block.ren, Block.countMentions]
    exact Branches.countMentions_ren brs h t
  | .breakBlock, _, _ => rfl
  | .continueBlock, _, _ => rfl
  | .returnBlock, _, _ => rfl

theorem Branches.countMentions_ren : (brs : List (Option Expr × Region)) → Branches.RenOk σ brs → ∀ t,
    Branches.countMentions (σ t) (Branches.ren σ brs) = Branches.countMentions t brs
  | [], _, _ => rfl
  | (c, r) :: rest, h, t => by
    simp only [Branches.ren, Branches.countMentions]
    rw [Region.countMentions_ren r h.2.1 t, Branches.countMentions_ren rest h.2.2 t]
    have hp : c.any (·.mentions (σ t)) = c.any (·.mentions t) := by
      refine mention_fix σ (p := fun t => c.any (·.mentions t)) (fun x hx => ?_) t
      simp only [Option.any_eq_true] at hx
      obtain ⟨e, he, hx⟩ := hx
      exact h.1 e he x (Expr.mem_names_of_mentions hx)
    rw [hp]

end

/-! ## Undoing a renaming -/

theorem perm_symm_fix {π : _root_.Equiv.Perm Name} {x : Name} (h : π x = x) : π.symm x = x := by
  rw [_root_.Equiv.symm_apply_eq]; exact h.symm

theorem State.exprNames_ren (s : State) : (s.ren σ).exprNames = s.exprNames := by
  simp only [State.exprNames, State.ren_nodes, State.ren_edges, List.flatMap_map]
  congr 1
  · congr 1; funext n
    rcases n with ⟨id, label, ic, oc, se, kind⟩
    cases kind <;> rfl
  · congr 1; funext e
    rcases e with ⟨src, sc, dst, dc, data⟩
    cases data <;> rfl

mutual

theorem LoopRegion.ren_symm : (l : LoopRegion) → (l.ren σ).ren σ.symm = l
  | ⟨_, _, _, _, _, _, _, body⟩ => by simp only [LoopRegion.ren, Region.ren_symm body]

theorem Block.ren_symm : (b : Block) → (b.ren σ).ren σ.symm = b
  | .state s => by simp only [Block.ren, State.ren_symm]
  | .region r => by simp only [Block.ren, Region.ren_symm r]
  | .loop l => by simp only [Block.ren, LoopRegion.ren_symm l]
  | .cond lb brs => by simp only [Block.ren, Branches.ren_symm brs]
  | .breakBlock | .continueBlock | .returnBlock => rfl

theorem Blocks.ren_symm : (bs : List Block) → Blocks.ren σ.symm (Blocks.ren σ bs) = bs
  | [] => rfl
  | b :: bs => by simp only [Blocks.ren, Block.ren_symm b, Blocks.ren_symm bs]

theorem Branches.ren_symm : (brs : List (Option Expr × Region)) → Branches.ren σ.symm (Branches.ren σ brs) = brs
  | [] => rfl
  | (c, r) :: rest => by simp only [Branches.ren, Region.ren_symm r, Branches.ren_symm rest]

theorem Region.ren_symm : (rg : Region) → (rg.ren σ).ren σ.symm = rg
  | ⟨_, _, blocks, _, _⟩ => by simp only [Region.ren, Blocks.ren_symm blocks]

end

theorem SDFG.ren_symm (g : SDFG) : (g.ren σ).ren σ.symm = g := by
  rcases g with ⟨name, arrays, symbols, constants, argNames, top, facts⟩
  simp only [SDFG.ren, List.map_map, Region.ren_symm, SDFG.mk.injEq, and_true, true_and]
  conv => rhs; rw [← List.map_id arrays]
  exact List.map_congr_left fun a _ => by simp

theorem RenState.symm {s : State} (h : RenState σ s) : RenState σ.symm (s.ren σ) := by
  refine ⟨fun x hx => ?_, fun entry d hd => ?_⟩
  · rw [State.exprNames_ren] at hx
    exact perm_symm_fix (h.expr x hx)
  · rw [State.consumedStream?_ren] at hd
    obtain ⟨d₀, hd₀, rfl⟩ := Option.map_eq_some_iff.1 hd
    obtain ⟨h₁, h₂⟩ := h.streams entry d₀ hd₀
    rw [h₁]
    exact ⟨perm_symm_fix h₁, perm_symm_fix h₂⟩

mutual

theorem Region.RenOk.symm : (rg : Region) → Region.RenOk σ rg → Region.RenOk σ.symm (rg.ren σ)
  | ⟨_, _, blocks, _, _⟩, h =>
    ⟨Blocks.RenOk.symm blocks h.1, fun e he x hx => perm_symm_fix (h.2 e he x hx)⟩

theorem Blocks.RenOk.symm : (bs : List Block) → Blocks.RenOk σ bs → Blocks.RenOk σ.symm (Blocks.ren σ bs)
  | [], _ => trivial
  | b :: bs, h => ⟨Block.RenOk.symm b h.1, Blocks.RenOk.symm bs h.2⟩

theorem Block.RenOk.symm : (b : Block) → Block.RenOk σ b → Block.RenOk σ.symm (b.ren σ)
  | .state _, h => RenState.symm σ h
  | .region r, h => Region.RenOk.symm r h
  | .loop ⟨_, _, _, _, _, _, _, body⟩, ⟨hi, hc, hu, hb⟩ =>
    ⟨fun a ha x hx => perm_symm_fix (hi a ha x hx),
      fun x hx => perm_symm_fix (hc x hx),
      fun a ha x hx => perm_symm_fix (hu a ha x hx), Region.RenOk.symm body hb⟩
  | .cond _ brs, h => Branches.RenOk.symm brs h
  | .breakBlock, _ => trivial
  | .continueBlock, _ => trivial
  | .returnBlock, _ => trivial

theorem Branches.RenOk.symm : (brs : List (Option Expr × Region)) → Branches.RenOk σ brs →
    Branches.RenOk σ.symm (Branches.ren σ brs)
  | [], _ => trivial
  | (_, r) :: rest, h =>
    ⟨fun e he x hx => perm_symm_fix (h.1 e he x hx), Region.RenOk.symm r h.2.1,
      Branches.RenOk.symm rest h.2.2⟩

end

theorem SDFG.array?_ren' (g : SDFG) (n : Name) : (g.ren σ).array? n = g.array? (σ.symm n) := by
  rw [← SDFG.array?_ren σ g (σ.symm n), Equiv.apply_symm_apply]

theorem SDFG.keepAtStart_ren (g : SDFG) (n : Name) : (g.ren σ).keepAtStart n = g.keepAtStart (σ.symm n) := by
  simp only [SDFG.keepAtStart, SDFG.ren, List.any_map]
  congr 1
  funext a
  rcases a with ⟨k, d⟩
  simp only [Function.comp_def]
  congr 1
  exact Bool.eq_iff_iff.2 (by simp only [beq_iff_eq]; exact (_root_.Equiv.eq_symm_apply σ).symm)

end Dace
