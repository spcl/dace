-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Scope

/-!
# Well-formedness of SDFGs

A decidable transcription of the structural rules of `dace/sdfg/validation.py`
(`validate_sdfg`, `validate_control_flow_region`, `validate_state`) plus the
nested-SDFG contract (`dace/sdfg/nodes.py:861-968`).  The checker returns a
report of **errors** (the SDFG is rejected) and **warnings** (facts DaCe
would establish with symbolic simplification that we only check
syntactically, e.g. descriptor equivalence of nested-SDFG connectors).

`WF g` is the proposition `(check g).errors = []`.  Rules that are about
runtime state (reference set before use, view-edge resolution) are not here:
the semantics reports them as error outcomes.
-/

namespace Dace

structure WFReport where
  errors : List String := []
  warnings : List String := []
  deriving Inhabited, Repr

namespace WFReport
def append (a b : WFReport) : WFReport := ⟨a.errors ++ b.errors, a.warnings ++ b.warnings⟩
instance : Append WFReport := ⟨append⟩
def err (s : String) : WFReport := ⟨[s], []⟩
def warn (s : String) : WFReport := ⟨[], [s]⟩
def ofList (l : List WFReport) : WFReport := l.foldl (· ++ ·) {}
def ok (r : WFReport) : Bool := r.errors.isEmpty
end WFReport

namespace WF

open WFReport (err warn)

/-- Root container name of a (possibly dotted) data name. -/
def rootName (n : Name) : Name := (n.splitOn ".").headD n

def arrayOf (g : SDFG) (n : Name) : Option DataDesc :=
  match g.array? n with
  | some d => some d
  | none => g.array? (rootName n)

/-- Every node's scope for edge-legality purposes: entries belong to their
parent scope, exits to their own entry's scope. -/
def scopeOf (n : DNode) : Option NodeId := n.scopeEntry

def checkMap (ctx : String) (m : MapNode) : WFReport :=
  (if m.params.isEmpty then err s!"{ctx}: map without parameters" else {}) ++
  (if m.params.length ≠ m.range.ndim then
    err s!"{ctx}: map has {m.params.length} parameters but {m.range.ndim} range dimensions" else {}) ++
  WFReport.ofList (m.range.dims.map fun d =>
    match d.step with
    | .lit (.int k) => if k ≤ 0 then err s!"{ctx}: non-positive map step {k}" else {}
    | _ => {})

def checkState (g : SDFG) (s : State) : WFReport :=
  let ctx := s!"state '{s.label}'"
  let ids := s.nodes.map (·.id)
  let uniqueIds := if ids.Nodup then {} else err s!"{ctx}: duplicate node ids"
  let nodeChecks := WFReport.ofList (s.nodes.map fun n =>
    let nctx := s!"{ctx}, node {n.id} '{n.label}'"
    let scopeOk := match n.scopeEntry with
      | none => {}
      | some p =>
        if n.isExit && p = n.id then err s!"{nctx}: exit node is its own scope"
        else if s.isEntryNode p then {} else err s!"{nctx}: scope_entry {p} is not an entry node"
    let kindChecks := match n.kind with
      | .access d => if (arrayOf g d).isNone then err s!"{nctx}: unknown data '{d}'" else {}
      | .mapEntry m =>
        checkMap nctx m ++
        (match s.exitOf? n.id with
         | none => err s!"{nctx}: entry without exit"
         | some _ => {}) ++
        -- IN_x ↔ OUT_x pairing on the entry
        WFReport.ofList (n.inConns.filterMap fun (c, _) =>
          match State.passSuffix? (some c) "IN_" with
          | some x => if (n.outConns.map Prod.fst).contains ("OUT_" ++ x) then none
                      else some (err s!"{nctx}: connector {c} without OUT_{x}")
          | none => none)
      | .consumeEntry _ =>
        (match s.exitOf? n.id with
         | none => err s!"{nctx}: consume entry without exit"
         | some _ => {}) ++
        (if (n.inConns.map Prod.fst).contains "IN_stream" && (n.outConns.map Prod.fst).contains "OUT_stream"
         then {} else err s!"{nctx}: consume entry needs IN_stream/OUT_stream")
      | .mapExit | .consumeExit =>
        (match n.scopeEntry with
         | some e => if s.isEntryNode e then {} else err s!"{nctx}: exit without entry"
         | none => err s!"{nctx}: exit without entry") ++
        WFReport.ofList (n.inConns.filterMap fun (c, _) =>
          match State.passSuffix? (some c) "IN_" with
          | some x => if (n.outConns.map Prod.fst).contains ("OUT_" ++ x) then none
                      else some (err s!"{nctx}: connector {c} without OUT_{x}")
          | none => none)
      | _ => {}
    -- dangling connectors
    let inUsed := (s.inEdges n.id).filterMap (·.dstConn)
    let outUsed := (s.outEdges n.id).filterMap (·.srcConn)
    let dangling :=
      WFReport.ofList (n.inConns.filterMap fun (c, _) =>
        if inUsed.contains c then none else some (err s!"{nctx}: dangling in-connector {c}")) ++
      WFReport.ofList (n.outConns.filterMap fun (c, _) =>
        if outUsed.contains c then none else some (err s!"{nctx}: dangling out-connector {c}"))
    -- at most one edge per in-connector except on exit nodes
    let multi := if n.isExit then {} else
      WFReport.ofList (n.inConns.filterMap fun (c, _) =>
        if (inUsed.filter (· = c)).length > 1 then some (err s!"{nctx}: several edges into connector {c}") else none)
    scopeOk ++ kindChecks ++ dangling ++ multi)
  let edgeChecks := WFReport.ofList (s.edges.map fun e =>
    let ectx := s!"{ctx}, edge {e.src}→{e.dst}"
    match s.node? e.src, s.node? e.dst with
    | some sn, some dn =>
      let connOk :=
        (match e.srcConn with
         | some c => if (sn.outConns.map Prod.fst).contains c then {} else err s!"{ectx}: unknown source connector {c}"
         | none => if sn.isCodeNode && !sn.isLibrary then err s!"{ectx}: code node edge without source connector" else {}) ++
        (match e.dstConn with
         | some c => if (dn.inConns.map Prod.fst).contains c then {} else err s!"{ectx}: unknown destination connector {c}"
         | none => if dn.isCodeNode && !dn.isLibrary && e.data matches .memlet _
                   then err s!"{ectx}: code node edge without destination connector" else {})
      -- scope legality
      let sS := scopeOf sn
      let sD := scopeOf dn
      let scopeLegal :=
        sS = sD || (sn.isEntry && sD = some sn.id) || (dn.isExit && sS = some dn.id) ||
        (sn.isExit && (match s.node? (sn.scopeEntry.getD 0) with | some en => en.scopeEntry = sD | none => false))
      let scopeOk := if scopeLegal then {} else err s!"{ectx}: crosses scopes illegally"
      let dataOk := match e.data with
        | .dep =>
          -- empty memlets: tasklet→tasklet ordering, or locking a node into a scope
          if (sn.isCodeNode && dn.isCodeNode) || sn.isEntry || dn.isExit then {}
          else err s!"{ectx}: empty memlet between non-code nodes"
        | .memlet m =>
          let arr := arrayOf g m.data
          (match arr with
           | none => err s!"{ectx}: memlet refers to unknown data '{m.data}'"
           | some d =>
             (if m.hasWcr && d.isStream then err s!"{ectx}: WCR on a stream" else {}) ++
             (match m.dataSubset with
              | some sub => if sub.ndim = d.ndim || d.isScalar then {}
                            else err s!"{ectx}: subset has {sub.ndim} dims, '{m.data}' has {d.ndim}"
              | none => {})) ++
          -- the data name must match an access-node endpoint of this edge, of its memlet path
          -- (`validation.py:1013-1047` checks `memlet_path(e)[0].src` / `[-1].dst`) or of its tree root
          (let r := s.treeRoot e
           let endpoints := [e.src, e.dst, (s.pathStart e).src, (s.pathEnd e).dst, r.src, r.dst].filterMap
             fun id => (s.node? id).bind DNode.accessData?
           if endpoints.isEmpty || endpoints.any (fun d => rootName d = rootName m.data) then {}
           else err s!"{ectx}: memlet data '{m.data}' matches no access-node endpoint of its tree") ++
          -- tree root has an access-node endpoint, or the edge is code→code
          (let r := s.treeRoot e
           if s.isAccessNode r.src || s.isAccessNode r.dst || (s.isCodeNode e.src && s.isCodeNode e.dst) then {}
           else err s!"{ectx}: memlet tree not rooted at an access node")
      connOk ++ scopeOk ++ dataOk
    | _, _ => err s!"{ectx}: edge endpoint does not exist")
  let acyclic := match s.topoOrder ids with
    | some _ => {}
    | none => err s!"{ctx}: dataflow graph is cyclic"
  uniqueIds ++ nodeChecks ++ edgeChecks ++ acyclic

mutual

partial def checkRegion (g : SDFG) (r : Region) : WFReport :=
  let ctx := s!"region '{r.label}'"
  let n := r.blocks.length
  (if n = 0 then err s!"{ctx}: no blocks" else {}) ++
  (if r.start < n || n = 0 then {} else err s!"{ctx}: start block {r.start} out of range") ++
  WFReport.ofList (r.edges.map fun e =>
    let ectx := s!"{ctx}, edge {e.src}→{e.dst}"
    (if e.src < n && e.dst < n then {} else err s!"{ectx}: endpoint out of range") ++
    (let lhs := e.assigns.map Prod.fst
     WFReport.ofList (e.assigns.map fun (x, rhs) =>
       (if (arrayOf g x).isSome then err s!"{ectx}: assignment to data container '{x}'" else {}) ++
       (if (rhs.freeSyms.filter lhs.contains).any (· ≠ x) || (rhs.freeSyms.contains x && lhs.length > 1 && false)
        then err s!"{ectx}: assignment to '{x}' reads another symbol assigned on the same edge" else {})))) ++
  WFReport.ofList (r.blocks.map (checkBlock g))

partial def checkBlock (g : SDFG) : Block → WFReport
  | .state s => checkState g s ++ WFReport.ofList (s.nodes.map fun n =>
      match n.kind with
      | .nested inner sm _ => checkNested g s n inner sm
      | _ => {})
  | .region r => checkRegion g r
  | .loop l => checkRegion g l.body
  | .cond label bs =>
    let elses := bs.filter (·.1.isNone)
    (if elses.length > 1 then err s!"conditional '{label}': several else branches" else {}) ++
    (match bs.getLast? with
     | some (none, _) => {}
     | _ => if elses.isEmpty then {} else err s!"conditional '{label}': else branch is not last") ++
    WFReport.ofList (bs.map fun (_, r) => checkRegion g r)
  | _ => {}

/-- The nested-SDFG contract: connectors ↔ non-transient inner arrays,
descriptor equivalence under the symbol mapping (warning only, syntactic). -/
partial def checkNested (g : SDFG) (s : State) (n : DNode) (inner : SDFG) (sm : List (Sym × Expr)) : WFReport :=
  let ctx := s!"state '{s.label}', nested SDFG '{inner.name}'"
  let conns := (n.inConns ++ n.outConns).map Prod.fst
  let σ : Sym → Option Expr := fun x => (sm.find? (·.1 = x)).map (·.2)
  let connChecks := WFReport.ofList (conns.map fun c =>
    match inner.array? c with
    | none => err s!"{ctx}: connector '{c}' is not an inner array"
    | some d =>
      (if d.transient then err s!"{ctx}: connector '{c}' is a transient inner array" else {}) ++
      -- outer array through the connected edge
      (let outerName : Option Name :=
         ((s.inEdges n.id).find? (·.dstConn = some c)).bind (fun e => (e.data.memlet?).map (·.data)) <|>
         ((s.outEdges n.id).find? (·.srcConn = some c)).bind (fun e => (e.data.memlet?).map (·.data))
       match outerName.bind (arrayOf g) with
       | none => {}
       | some od =>
         let di : DataDesc := { d with shape := d.shape.map (fun e => Expr.subst σ e),
                                       strides := d.strides.map (fun e => Expr.subst σ e) }
         if di.isEquivalent od then {}
         else warn s!"{ctx}: connector '{c}' descriptor differs syntactically from outer '{outerName.getD ""}'"))
  let missing := WFReport.ofList (inner.arrays.filterMap fun (a, d) =>
    if d.transient || conns.contains a then none else some (err s!"{ctx}: non-transient inner array '{a}' has no connector"))
  let symChecks := WFReport.ofList (inner.symbols.filterMap fun (x, _) =>
    if (σ x).isSome then none else some (warn s!"{ctx}: inner symbol '{x}' not in symbol mapping"))
  connChecks ++ missing ++ symChecks ++ checkSDFG inner

partial def checkSDFG (g : SDFG) : WFReport :=
  let names := g.arrays.map Prod.fst
  let syms := g.symbols.map Prod.fst
  (if names.Nodup then {} else err s!"sdfg '{g.name}': duplicate array names") ++
  WFReport.ofList (names.filterMap fun a =>
    if syms.contains a then some (err s!"sdfg '{g.name}': '{a}' is both an array and a symbol") else none) ++
  checkRegion g g.top

end

end WF

/-- Run the checker. -/
def SDFG.check (g : SDFG) : WFReport := WF.checkSDFG g

/-- Well-formedness proposition. -/
def SDFG.WF (g : SDFG) : Prop := (g.check).errors = []

instance (g : SDFG) : Decidable g.WF := by unfold SDFG.WF; infer_instance

end Dace
