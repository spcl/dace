-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Leaf
import Dace.STree.WF

/-!
# Schedule tree semantics: basic definitions

Shared infrastructure of the schedule tree semantics (`Dace.STree.Semantics`):

* the control signals of tree nodes (`TCtl`: the SDFG signals plus `goto`);
* the context threaded through a *sibling list* (`SibCtx`): the dynamic
  scope inputs pending for the next dataflow scope and the state of an open
  `if`/`elif`/`else` chain (`Decisions.md#D50`, `#D52`);
* the execution context `TCtx`: the SDFG context `Ctx` built from the
  tree's descriptor repository (nested SDFGs are already flattened, so the
  container-name prefix is empty), the tree itself (for the private
  transients of a scope, `#D54`) and the stream consumed by the innermost
  consume scope (`#D50`);
* conversions between the control-flow state `CSt` and the dataflow state
  `St` of the SDFG semantics, so that its leaf functions are reused.
-/

set_option autoImplicit false

namespace Dace

/-- Control signal produced by a schedule tree node: the SDFG signals plus
a pending jump to a label (`Decisions.md#D51`). -/
inductive TCtl where
  | normal | brk | cont | ret | goto (label : String)
  deriving DecidableEq, Repr, Inhabited

namespace TCtl

/-- Does control proceed to the loop tail after the body? -/
def continuesLoop : TCtl → Bool
  | normal | cont => true
  | brk | ret | goto _ => false

/-- A structured jump that a sibling list passes on unchanged. -/
def isJump : TCtl → Bool
  | brk | cont | ret => true
  | normal | goto _ => false

end TCtl

/-- Dynamic scope inputs collected by `DynScopeCopyNode`s for the next
dataflow scope (`Decisions.md#D50`): symbol bindings, and the stream a
consume scope pops from (its `IN_stream` input) with the element index. -/
structure Pending where
  syms : List (Sym × Val) := []
  stream : Option (Name × Idx) := none

def Pending.isEmpty (p : Pending) : Bool := p.syms.isEmpty && p.stream.isNone

/-- Context threaded along a list of siblings. -/
structure SibCtx where
  pending : Pending := {}
  /-- The `if`/`elif`/`else` chain: `none` = no open chain, `some taken`. -/
  chain : Option Bool := none

/-- Execution context of a schedule tree. -/
structure TCtx where
  /-- The SDFG context over the tree's descriptor repository (`pfx = ""`). -/
  ctx : Ctx
  tree : STree
  /-- The stream consumed by the innermost enclosing consume scope: reads
  of it inside the scope denote the consumed chunk (`Decisions.md#D50`). -/
  consumed : Option Name := none

namespace TCtx

def ofTree (I : Interp) (t : STree) : TCtx := { ctx := { I, g := t.toSDFG, pfx := "" }, tree := t }

/-- Enter the body of a consume scope popping from `d`. -/
def consuming (tc : TCtx) (d : Name) : TCtx := { tc with consumed := some d }

/-- Does the container name denote the consumed chunk here? -/
def isChunk (tc : TCtx) (d : Name) : Bool := tc.consumed = some d

/-- Memory name of the chunk consumed from stream `d` (cf. `chunkName`). -/
def chunkName (tc : TCtx) (d : Name) : Name := tc.ctx.cname ("__consume_chunk_" ++ d)

/-- Transient containers all of whose mentions in the tree lie inside the
scope with children `cs` (`Decisions.md#D54`, the tree analogue of `#D39`);
their effects are private to one iteration/task. -/
def privateTransients (tc : TCtx) (cs : List SNode) : List Name :=
  tc.tree.containers.filterMap fun (t, d) =>
    if d.transient && SNodes.countData t cs = tc.tree.countData t then some (tc.ctx.cname t) else none

end TCtx

namespace CSt

/-- The dataflow view of a control-flow state (no code-edge values). -/
def st (cst : CSt) : St := { mem := cst.mem, log := cst.log }

/-- Take memory and log back from a dataflow state. -/
def withSt (cst : CSt) (s : St) : CSt := { cst with mem := s.mem, log := s.log }

/-- Lift the outcome of a leaf function. -/
def lift (cst : CSt) (r : Res St) : Res CSt := r.map cst.withSt

@[simp] theorem withSt_st (cst : CSt) : cst.withSt cst.st = cst := by
  cases cst; rfl

end CSt

namespace SNode

/-- Nodes that take the pending dynamic scope inputs: dataflow scopes
consume them, `DynScopeCopyNode`s extend them. -/
def consumesPending (n : SNode) : Bool := n.isDataflowScope || n.isDynScopeCopy

end SNode

/-- Pending dynamic scope inputs reaching a node that cannot take them
(`Decisions.md#D50`). -/
def pendingError : ErrReason := .malformed "dynamic scope inputs not followed by a dataflow scope"

/-- The suffix of a sibling list after the first `StateLabel l`. -/
def afterLabel (l : String) : List SNode → Option (List SNode)
  | [] => none
  | .stateLabel name :: rest => if name = l then some rest else afterLabel l rest
  | _ :: rest => afterLabel l rest

/-- Where a `goto l` raised by a child resumes in its sibling list
(`Decisions.md#D51`): after a later sibling label (forward goto), or, inside
a general block, after any label of the block (backward gotos allowed). -/
def resumeAt (gb : Bool) (full rest : List SNode) (l : String) : Option (List SNode) :=
  match afterLabel l rest with
  | some s => some s
  | none => if gb then afterLabel l full else none

end Dace
