-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.WF
import Dace.STree.Syntax

/-!
# Well-formedness of schedule trees

A decidable checker for the structural invariants that
`sdfg_to_tree.as_schedule_tree` establishes and `tree_to_sdfg` relies on
(`Decisions.md#D58`):

* only the node classes of the catalog occur and parents/children align
  (`validate_has_no_other_node_types`, `validate_children_and_parents_align`:
  both hold by construction of `SNode`);
* `ElifScope` / `ElseScope` immediately follow an `IfScope` / `ElifScope`
  (`tree_to_sdfg._process_next_conditional_block`);
* a `DynScopeCopyNode` is immediately followed by another one or by the
  `MapScope` / `ConsumeScope` it feeds (`tree_to_sdfg._dynamic_scope_inputs`);
  a `ConsumeScope` is fed its `IN_stream`;
* `StateLabel` names are unique; `GotoNode` targets resolve to a label of
  the enclosing `GBlock` or to a later sibling of an enclosing scope
  (`tree_to_sdfg._find_label`, `_lower_forward_gotos`) without leaving a
  dataflow scope; exit gotos do not occur inside dataflow scopes
  (`tree_to_sdfg.visit_GotoNode`);
* `break` / `continue` occur inside a loop of the same dataflow scope;
* memlets and copy/view/reference targets name declared containers, code
  node memlets use declared connectors, views are view descriptors,
  references are reference descriptors, no WCR on streams (`#D6`);
* maps are well-formed (`WF.checkMap`), container and symbol names are
  distinct, assignments do not target containers.

`STree.WF t` is the proposition `(t.check).errors = []`.
-/

set_option autoImplicit false

namespace Dace

namespace STree.WF

open WFReport (err warn)

/-- Context threaded through the traversal. -/
structure Env where
  /-- Labels a goto may target from here. -/
  visibleLabels : List String := []
  /-- Inside a loop (of the current dataflow scope). -/
  inLoop : Bool := false
  /-- Inside a map or consume scope. -/
  inDataflow : Bool := false
  /-- Inside a general block. -/
  inGBlock : Bool := false

def checkMemlet (t : STree) (ctx : String) (m : Memlet) : WFReport :=
  match t.container? m.data with
  | none => err s!"{ctx}: memlet refers to unknown data '{m.data}'"
  | some d =>
    (if m.hasWcr && d.isStream then err s!"{ctx}: WCR on a stream" else {}) ++
    (match m.dataSubset with
     | some sub =>
       if sub.ndim = d.ndim || d.isScalar then {}
       else err s!"{ctx}: subset has {sub.ndim} dims, '{m.data}' has {d.ndim}"
     | none => {})

def checkContainer (t : STree) (ctx : String) (n : Name) : WFReport :=
  if (t.container? n).isSome then {} else err s!"{ctx}: unknown data '{n}'"

def checkMemletMap (t : STree) (ctx : String) (conns : List Conn) (mm : MemletMap) : WFReport :=
  WFReport.ofList (mm.memlets.map (checkMemlet t ctx)) ++
  (match mm with
   | .dict items => WFReport.ofList (items.map fun (c, _) =>
       if conns.contains c then {} else err s!"{ctx}: memlet on undeclared connector '{c}'")
   | .set _ => if conns.isEmpty then {} else err s!"{ctx}: connector-less memlet set on a node with connectors")

/-- Checks of a node that do not depend on its position. -/
def checkOwn (t : STree) (n : SNode) : WFReport :=
  let ctx := n.typeName
  match n with
  | .assign name _ =>
    if (t.container? name).isSome then err s!"{ctx}: assignment to data container '{name}'" else {}
  | .mapScope mp _ => WF.checkMap s!"{ctx} '{mp.label}'" mp
  | .taskletNode tk ins outs =>
    checkMemletMap t s!"{ctx} '{tk.label}'" (tk.inConns.map Prod.fst) ins ++
    checkMemletMap t s!"{ctx} '{tk.label}'" (tk.outConns.map Prod.fst) outs
  | .libraryCall l _ ins outs =>
    -- library nodes may have no connectors at all (`Decisions.md#D57`)
    WFReport.ofList ((ins.memlets ++ outs.memlets).map (checkMemlet t s!"{ctx} '{l.label}'"))
  | .copyNode target m =>
    checkMemlet t s!"{ctx} to '{target}'" m ++ checkContainer t s!"{ctx}" target ++
    (match t.container? m.data, t.container? target with
     | some sd, some dd => if sd.isStream && dd.isStream then err s!"{ctx}: stream-to-stream copy" else {}
     | _, _ => {})
  | .dynScopeCopy target m => checkMemlet t s!"{ctx} '{target}'" m
  | .viewNode target source m _ _ =>
    checkMemlet t s!"{ctx} '{target}'" m ++ checkContainer t ctx source ++
    (match t.container? target with
     | none => err s!"{ctx}: unknown view '{target}'"
     | some d => if d.isView then {} else err s!"{ctx}: '{target}' is not a view") ++
    (if m.data = source then {} else err s!"{ctx}: memlet of view '{target}' is not on its source '{source}'")
  | .refSetNode target m _ _ =>
    checkMemlet t s!"{ctx} '{target}'" m ++
    (match t.container? target with
     | none => err s!"{ctx}: unknown reference '{target}'"
     | some d => if d.isReference then {} else err s!"{ctx}: '{target}' is not a reference")
  | _ => {}

/-- Labels directly in a list of siblings. -/
def labelsOf (cs : List SNode) : List String := cs.filterMap SNode.label?

/-- Labels among the siblings after position `k`. -/
def labelsAfter (cs : List SNode) (k : Nat) : List String := labelsOf (cs.drop (k + 1))

mutual

/-- Check a node in its environment. -/
partial def checkNode (t : STree) (env : Env) (n : SNode) : WFReport :=
  checkOwn t n ++
  match n with
  | .goto none => if env.inDataflow then err "GotoNode: exit goto inside a dataflow scope" else {}
  | .goto (some l) =>
    if env.visibleLabels.contains l then {} else err s!"GotoNode: target label '{l}' is not visible"
  | .breakNode => if env.inLoop then {} else err "BreakNode outside of a loop"
  | .continueNode => if env.inLoop then {} else err "ContinueNode outside of a loop"
  | .gblock cs => checkChildren t { env with inGBlock := true } cs true
  | .mapScope _ cs | .consumeScope _ cs =>
    checkChildren t { visibleLabels := [], inLoop := false, inDataflow := true, inGBlock := false } cs false
  | _ =>
    match n.children? with
    | some cs =>
      let env' := if n.isLoop then { env with inLoop := true, inGBlock := false } else { env with inGBlock := false }
      checkChildren t env' cs false
    | none => {}

/-- Check a sibling list: positional rules plus the children themselves. -/
partial def checkChildren (t : STree) (env : Env) (cs : List SNode) (isGBlock : Bool) : WFReport :=
  let allLabels := labelsOf cs
  let positional := WFReport.ofList (cs.zipIdx.map fun (c, k) =>
    let prev := if k = 0 then none else cs[k - 1]?
    let next := cs[k + 1]?
    (match c with
     | .elifScope .. | .elseScope .. =>
       (match prev with
        | some (.ifScope ..) | some (.elifScope ..) => {}
        | _ => err s!"{c.typeName} does not follow an IfScope/ElifScope")
     | .dynScopeCopy target m =>
       (match next with
        | some (.dynScopeCopy ..) | some (.mapScope ..) | some (.consumeScope ..) => {}
        | _ => err s!"DynScopeCopyNode '{target}' is not followed by a dataflow scope") ++
       (if target = "IN_stream" then
          match t.container? m.data with
          | some d => if d.isStream then {} else err s!"DynScopeCopyNode IN_stream of non-stream '{m.data}'"
          | none => {}
        else {})
     | .consumeScope c _ =>
       -- the consumed stream is given by the IN_stream dscopy directly before the scope
       let rec feeding (j : Nat) : Bool :=
         match j with
         | 0 => false
         | j + 1 =>
           match cs[j]? with
           | some (.dynScopeCopy target _) => target = "IN_stream" || feeding j
           | _ => false
       if feeding k then {} else err s!"ConsumeScope '{c.label}' without an IN_stream DynScopeCopyNode"
     | _ => {}))
  let children := WFReport.ofList (cs.zipIdx.map fun (c, k) =>
    let visible := (if isGBlock then allLabels else labelsAfter cs k) ++ env.visibleLabels
    checkNode t { env with visibleLabels := visible } c)
  positional ++ children

end

/-- Tree-level checks: repository names, label uniqueness. -/
def checkTree (t : STree) : WFReport :=
  let names := t.containers.map Prod.fst
  let syms := t.symbols.map Prod.fst
  let labels := t.preorder.filterMap SNode.label?
  (if names.Nodup then {} else err s!"tree '{t.name}': duplicate container names") ++
  WFReport.ofList (names.filterMap fun a =>
    if syms.contains a then some (err s!"tree '{t.name}': '{a}' is both a container and a symbol") else none) ++
  (if labels.Nodup then {} else err s!"tree '{t.name}': duplicate state labels") ++
  checkChildren t {} t.children false

end STree.WF

/-- Run the checker. -/
def STree.check (t : STree) : WFReport := STree.WF.checkTree t

/-- Well-formedness proposition. -/
def STree.WF (t : STree) : Prop := (t.check).errors = []

instance (t : STree) : Decidable t.WF := by unfold STree.WF; infer_instance

end Dace
