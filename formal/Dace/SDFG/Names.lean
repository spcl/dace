-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics

/-!
# Names a program mentions

The execution relations consult the context (descriptors, memory names) only
at names the program syntax contains: memlet data, access nodes, the chunk
containers of streams, names in expressions (subscripted data and bare
names, `Decisions.md#D33`), tasklet code identifiers, and the names in the
control-flow expressions.  This file collects them:

* `State.dataNames`: names whose memory name is taken (memlet data, access
  nodes and their consume chunks);
* `State.exprNames`: names evaluated in expressions (subsets, ranges,
  consume parameters, symbol mappings, `Reduce` identities) and tasklet code
  identifiers;
* `Region.Names D E`: every state of a control-flow hierarchy has its data
  names in `D` and expression names in `E`, and the control-flow
  expressions have their names in `E`.

`DNode.mentions_names`: every name a node mentions (`DNode.mentions`, the
private-transient count, `Decisions.md#D39`) is among them.  The
endpoint lemmas (`State.srcEndpoint_names`, …) show the names the structural
functions return come from the state.  Used by the partial-agreement
simulation (`Dace.SDFG.Sim`).
-/

set_option autoImplicit false

namespace Dace

/-! ## Expressions and subsets -/

/-- The names an expression reads: subscripted data and bare names. -/
def Expr.names (e : Expr) : List Name := e.readData ++ e.freeSyms

def Dim.names (d : Dim) : List Name := d.lo.names ++ d.hi.names ++ d.step.names ++ d.tile.names

def Range.names (r : Range) : List Name := r.dims.flatMap Dim.names

def Subset.names : Subset → List Name
  | .range r => r.names
  | .union rs => rs.flatMap Range.names

/-- Names of an optional subset. -/
def optSubsetNames (o : Option Subset) : List Name := (o.map Subset.names).getD []

def Memlet.exprNames (m : Memlet) : List Name :=
  optSubsetNames m.srcSubset ++ optSubsetNames m.dstSubset ++ m.volume.names

theorem Expr.mem_names_of_mentions {e : Expr} {t : Name} (h : e.mentions t = true) : t ∈ e.names := by
  simp only [Expr.mentions, Bool.or_eq_true, List.contains_iff_mem] at h
  simp only [Expr.names, List.mem_append]
  exact h

theorem Range.mem_names_of_mentions {r : Range} {t : Name} (h : r.mentions t = true) : t ∈ r.names := by
  simp only [Range.mentions, List.any_eq_true, Bool.or_eq_true] at h
  obtain ⟨d, hd, h⟩ := h
  simp only [Range.names, List.mem_flatMap]
  refine ⟨d, hd, ?_⟩
  simp only [Dim.names, List.mem_append]
  rcases h with ((h | h) | h) | h
  · exact .inl (.inl (.inl (Expr.mem_names_of_mentions h)))
  · exact .inl (.inl (.inr (Expr.mem_names_of_mentions h)))
  · exact .inl (.inr (Expr.mem_names_of_mentions h))
  · exact .inr (Expr.mem_names_of_mentions h)

/-! ## Nodes, edges, states -/

/-- Names whose memory name a node takes. -/
def DNode.dataNames (n : DNode) : List Name :=
  match n.kind with
  | .access d => [d, consumeChunkName d]
  | _ => []

/-- Names a node evaluates in expressions, and tasklet code identifiers. -/
def DNode.exprNames (n : DNode) : List Name :=
  match n.kind with
  | .tasklet tk => codeIdents tk.code.code
  | .mapEntry mp => mp.range.names
  | .consumeEntry c => c.numPes.names ++ (c.condition.map Expr.names).getD []
  | .nested _ mapping _ => mapping.flatMap fun (x, e) => x :: e.names
  | .library l => ((l.reduce.bind (·.identity)).map Expr.names).getD []
  | .access _ | .mapExit | .consumeExit => []

def DEdge.dataNames (e : DEdge) : List Name :=
  match e.data with
  | .memlet m => [m.data, consumeChunkName m.data]
  | .dep => []

def DEdge.exprNames (e : DEdge) : List Name :=
  match e.data with
  | .memlet m => m.exprNames
  | .dep => []

def State.dataNames (s : State) : List Name := s.nodes.flatMap DNode.dataNames ++ s.edges.flatMap DEdge.dataNames

def State.exprNames (s : State) : List Name := s.nodes.flatMap DNode.exprNames ++ s.edges.flatMap DEdge.exprNames

theorem DNode.mentions_names {n : DNode} {t : Name} (h : n.mentions t = true) :
    t ∈ n.dataNames ∨ t ∈ n.exprNames := by
  unfold DNode.mentions at h
  unfold DNode.dataNames DNode.exprNames
  cases hk : n.kind with
  | access d =>
    rw [hk] at h
    simp only [decide_eq_true_eq] at h
    exact .inl (by simp [h])
  | tasklet tk =>
    rw [hk] at h
    simp only [Tasklet.namesInCode, Bool.and_eq_true, List.contains_iff_mem] at h
    exact .inr h.2
  | mapEntry mp =>
    rw [hk] at h
    exact .inr (Range.mem_names_of_mentions h)
  | consumeEntry c =>
    rw [hk] at h
    simp only [Bool.or_eq_true, Option.any_eq_true] at h
    rcases h with h | ⟨e, he, h⟩
    · exact .inr (by simp [Expr.mem_names_of_mentions h])
    · exact .inr (by simp [he, Expr.mem_names_of_mentions h])
  | nested inner mapping ni =>
    rw [hk] at h
    simp only [List.any_eq_true] at h
    obtain ⟨⟨x, e⟩, hx, h⟩ := h
    exact .inr (List.mem_flatMap.2 ⟨(x, e), hx, List.mem_cons_of_mem _ (Expr.mem_names_of_mentions h)⟩)
  | library l => rw [hk] at h; cases h
  | mapExit => rw [hk] at h; cases h
  | consumeExit => rw [hk] at h; cases h

theorem State.mem_dataNames_node {s : State} {n : DNode} (hn : n ∈ s.nodes) {t : Name} (ht : t ∈ n.dataNames) :
    t ∈ s.dataNames :=
  List.mem_append_left _ (List.mem_flatMap.2 ⟨n, hn, ht⟩)

theorem State.mem_dataNames_edge {s : State} {e : DEdge} (he : e ∈ s.edges) {t : Name} (ht : t ∈ e.dataNames) :
    t ∈ s.dataNames :=
  List.mem_append_right _ (List.mem_flatMap.2 ⟨e, he, ht⟩)

theorem State.mem_exprNames_node {s : State} {n : DNode} (hn : n ∈ s.nodes) {t : Name} (ht : t ∈ n.exprNames) :
    t ∈ s.exprNames :=
  List.mem_append_left _ (List.mem_flatMap.2 ⟨n, hn, ht⟩)

theorem State.mem_exprNames_edge {s : State} {e : DEdge} (he : e ∈ s.edges) {t : Name} (ht : t ∈ e.exprNames) :
    t ∈ s.exprNames :=
  List.mem_append_right _ (List.mem_flatMap.2 ⟨e, he, ht⟩)

/-- A node mentioning a name has it among the state's names. -/
theorem State.mentions_names {s : State} {n : DNode} (hn : n ∈ s.nodes) {t : Name} (h : n.mentions t = true) :
    t ∈ s.dataNames ∨ t ∈ s.exprNames :=
  (DNode.mentions_names h).imp (State.mem_dataNames_node hn) (State.mem_exprNames_node hn)

/-! ## Endpoints -/

namespace State

variable {s : State}

theorem node?_mem {id : NodeId} {n : DNode} (h : s.node? id = some n) : n ∈ s.nodes :=
  List.mem_of_find?_eq_some h

theorem mem_inEdges {id : NodeId} {e : DEdge} (h : e ∈ s.inEdges id) : e ∈ s.edges := (List.mem_filter.1 h).1

theorem mem_outEdges {id : NodeId} {e : DEdge} (h : e ∈ s.outEdges id) : e ∈ s.edges := (List.mem_filter.1 h).1

theorem accessData_names {id : NodeId} {n : DNode} {d : Name} (h : s.node? id = some n)
    (hk : n.kind = .access d) : d ∈ s.dataNames ∧ consumeChunkName d ∈ s.dataNames := by
  have hn := node?_mem h
  refine ⟨mem_dataNames_node hn ?_, mem_dataNames_node hn ?_⟩ <;> simp [DNode.dataNames, hk]

theorem consumedStream?_names {entry : NodeId} {d : Name} (h : s.consumedStream? entry = some d) :
    d ∈ s.dataNames ∧ consumeChunkName d ∈ s.dataNames := by
  unfold consumedStream? at h
  split at h
  · rename_i e _
    obtain ⟨n, hn, hd⟩ := Option.bind_eq_some_iff.1 h
    unfold DNode.accessData? at hd
    split at hd
    · rename_i d' hk
      cases hd
      exact accessData_names hn hk
    · cases hd
  · cases h

theorem srcEndpoint_names {e : DEdge} {d : Name} {sub : Option Subset} (h : s.srcEndpoint e = .data d sub) :
    d ∈ s.dataNames ∧ (sub = none ∨ sub = leafSrcSubset e) := by
  unfold srcEndpoint at h
  split at h
  · split at h
    · rename_i d' hd'
      cases h
      exact ⟨(consumedStream?_names hd').2, .inl rfl⟩
    · cases h
  · simp only at h
    split at h
    · cases h
    · rename_i n hn
      split at h
      · rename_i d' hk
        cases h
        exact ⟨(accessData_names hn hk).1, .inr rfl⟩
      all_goals cases h

theorem dstEndpoint_names {e : DEdge} {d : Name} {sub : Option Subset} (h : s.dstEndpoint e = .data d sub) :
    d ∈ s.dataNames ∧ sub = leafDstSubset e := by
  unfold dstEndpoint at h
  cases hn : s.node? e.dst with
  | none => rw [hn] at h; cases h
  | some n =>
    rw [hn] at h
    simp only at h
    cases hk : n.kind with
    | access d' =>
      rw [hk] at h
      cases h
      exact ⟨(accessData_names hn hk).1, rfl⟩
    | mapExit | consumeExit =>
      rw [hk] at h
      simp only at h
      cases hrn : s.node? (s.dstRootEdge e).dst with
      | none => rw [hrn] at h; cases h
      | some rn =>
        rw [hrn] at h
        simp only at h
        cases hk' : rn.kind with
        | access d' =>
          rw [hk'] at h
          cases h
          exact ⟨(accessData_names hrn hk').1, rfl⟩
        | _ => rw [hk'] at h; cases h
    | _ => rw [hk] at h; cases h

theorem leafSrcSubset_names {e : DEdge} (he : e ∈ s.edges) :
    ∀ x ∈ optSubsetNames (leafSrcSubset e), x ∈ s.exprNames := by
  intro x hx
  refine mem_exprNames_edge he ?_
  unfold leafSrcSubset at hx
  unfold DEdge.exprNames
  cases hd : e.data with
  | dep => rw [hd] at hx; simp [EdgeData.memlet?, optSubsetNames] at hx
  | memlet m =>
    rw [hd] at hx
    simp only [EdgeData.memlet?, Option.bind_some] at hx
    simp [Memlet.exprNames, hx]

theorem leafDstSubset_names {e : DEdge} (he : e ∈ s.edges) :
    ∀ x ∈ optSubsetNames (leafDstSubset e), x ∈ s.exprNames := by
  intro x hx
  refine mem_exprNames_edge he ?_
  unfold leafDstSubset at hx
  unfold DEdge.exprNames
  cases hd : e.data with
  | dep => rw [hd] at hx; simp [EdgeData.memlet?, optSubsetNames] at hx
  | memlet m =>
    rw [hd] at hx
    simp only [EdgeData.memlet?, Option.bind_some] at hx
    simp [Memlet.exprNames, hx]

end State

namespace State

variable {s : State}

theorem edge_of_outEdgesOn {n : DNode} {c : Conn} {e : DEdge} (h : e ∈ s.outEdgesOn n c) : e ∈ s.edges :=
  mem_outEdges (List.mem_filter.1 h).1

theorem edge_of_inEdgeOn {n : DNode} {c : Conn} {e : DEdge} (h : s.inEdgeOn n c = some e) : e ∈ s.edges :=
  mem_inEdges (List.mem_of_find?_eq_some h)

theorem viewEdgeOneOne_mem {v : DNode} {i : DEdge} {o? : Option DEdge} {e : DEdge} {b : Bool}
    (h : s.viewEdgeOneOne v i o? = some (e, b)) : e = i ∨ o? = some e := by
  unfold viewEdgeOneOne at h
  cases o? with
  | none => simp only [Option.any_none, Option.map_none] at h; split_ifs at h <;> simp_all
  | some o => simp only [Option.any_some, Option.map_some] at h; split_ifs at h <;> simp_all
theorem viewEdge?_mem {v : DNode} {e : DEdge} {b : Bool} (h : s.viewEdge? v = some (e, b)) : e ∈ s.edges := by
  unfold viewEdge? at h
  simp only at h
  have hin : ∀ x ∈ (s.inEdges v.id).filter (fun e => e.data matches .memlet _), x ∈ s.edges :=
    fun x hx => mem_inEdges (List.mem_filter.1 hx).1
  have hout : ∀ x ∈ s.outEdges v.id, x ∈ s.edges := fun x hx => mem_outEdges hx
  revert hin hout h
  generalize (s.inEdges v.id).filter (fun e => e.data matches .memlet _) = ins
  generalize s.outEdges v.id = outs
  intro h hin hout
  split_ifs at h
  all_goals first
    | (cases ins with
        | nil => simp_all
        | cons i rest =>
          simp only at h
          first
            | (split_ifs at h <;> simp_all)
            | (rcases viewEdgeOneOne_mem h with rfl | ho
               · exact hin _ List.mem_cons_self
               · exact hout e (List.mem_of_mem_head? ho)))
    | (cases outs with
        | nil => simp_all
        | cons o rest => simp only at h; split_ifs at h <;> simp_all)
    | simp_all
end State

end Dace
