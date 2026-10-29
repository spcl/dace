-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Semantics.Basic

/-!
# SDFG semantics: leaf-level functions

Everything that happens *within* one node of a state is a plain function of
the execution state (`St`): copies along leaf memlet edges, tasklet and
library-node execution, the direct semantics of `Reduce`, view bindings,
reference sets, dynamic scope inputs, and the set-up/tear-down of scopes
and nested invocations.  Only the parts that recurse into the syntax (scope
bodies, nested SDFGs, control flow) are relations; they live in
`Dace.SDFG.Semantics.Exec` and use the functions below as premises.

Conventions (`Decisions.md#D29`–`#D32`, `#D38`, `#D40`):

* a **copy** executes at its destination node (access node or scope exit);
  a code node reads its in-edges and writes its out-edges when it executes;
* every element access goes through `Ctx.loc` (bounds-checked unless its
  memlet has `allow_oob`, `#D65`) and is logged resolved;
* the consume chunk is an ordinary transient container
  (`consumeChunkName`, declared by the importer, `#D38`), private to each task
  by `#D39`; only its initial content is written outside the logs.
-/

set_option autoImplicit false

namespace Dace

open State (Endpoint)

/-! ## Consume chunk -/

namespace Mem

/-- Write the elements of a chunk at the unit-stride offsets `0, 1, …` of
container memory name `name` (the layout of `DataDesc.consumeChunk`),
bypassing aliases and logs.  Elements past the end of a short last chunk
keep their previous content (codegen passes the actual count separately,
`cpu.py:2221-2230`, and the model gives tasklets no access to it). -/
def writeChunk (m : Mem) (name : Name) (vs : List Val) : Mem :=
  vs.zipIdx.foldl (fun acc (v, k) => { acc with cells := Function.update acc.cells (Loc.elem name k) v }) m

/-- Replace the content of the queue at a (resolved) location. -/
def setQueue (m : Mem) (l : Loc) (q : List Val) : Mem :=
  { m with queues := Function.update m.queues l q }

end Mem

/-- Linear offset of an element location. -/
def Loc.offset (l : Loc) : Int :=
  match l.path with
  | [.lin o] => o
  | _ => 0

/-! ## Edge helpers -/

namespace State

/-- Memlet out-edges of `n` on connector `c` (`""` = no connector). -/
def outEdgesOn (s : State) (n : DNode) (c : Conn) : List DEdge :=
  (s.outEdges n.id).filter fun e => e.srcConn.getD "" = c && (e.data matches .memlet _)

/-- The memlet in-edge of `n` on connector `c`. -/
def inEdgeOn (s : State) (n : DNode) (c : Conn) : Option DEdge :=
  (s.inEdges n.id).find? fun e => e.dstConn.getD "" = c && (e.data matches .memlet _)

/-- Ids of the direct children of a scope (`none` = top level). -/
def directChildren (s : State) (entry : Option NodeId) : List NodeId := (s.children entry).map (·.id)

/-- The top-level nodes among `ids`. -/
def topLevelOf (s : State) (ids : List NodeId) : List NodeId :=
  ids.filter fun id => (s.node? id).any (·.scopeEntry.isNone)

end State

/-! ## Reading single values -/

/-- The `allow_oob` flag of an edge's memlet (`Decisions.md#D65`): accesses
through it are not bounds-checked. -/
def DEdge.allowOob (e : DEdge) : Bool := e.data.memlet?.any (·.allowOob)

/-- The scalar at a single-element subset of container `d`: one popped
element for a stream, the element itself otherwise.  `oob` is the
`allow_oob` flag of the memlet (`Decisions.md#D65`). -/
def readScalarData (ctx : Ctx) (ρ : Valuation) (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    Res (Option Val × St) := do
  let pts ← ctx.pointsOf ρ st.mem d sub
  match pts with
  | [p] =>
    if ctx.isStream d then do
      let (vs, st) ← st.popN ctx ρ d p 1
      pure (vs.head?, st)
    else do
      let (v, st) ← st.read ctx ρ d p oob
      pure (some v, st)
  | _ => .err (.malformed s!"scalar read of '{d}' is not a single element")

/-- The scalar flowing along a leaf edge into a scope entry or code node
(`Decisions.md#D42`): a single element of a container, one popped element
of a stream, or a code value (the chunk is a container).  `none` = no value
flows (unassigned dynamic output upstream, `Decisions.md#D13`). -/
def readEdgeScalar (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res (Option Val × St) :=
  match s.srcEndpoint e with
  | .code key => .ok (st.val? key, st)
  | .data d sub => readScalarData ctx ρ d sub e.allowOob st
  | .invalid => .err (.malformed "leaf edge without a valid source")

/-- Location of the (first) element of a stream endpoint. -/
def streamPoint (ctx : Ctx) (ρ : Valuation) (m : Mem) (d : Name) (sub : Option Subset) : Res Idx := do
  let pts ← ctx.pointsOf ρ m d sub
  Res.ofOption pts.head? (.malformed s!"empty subset of stream '{d}'")

/-- Lower corner of a subset of container `n` (`Range.min_element`, the
point a pointer to the subset addresses: `cpp_offset_expr` at index `0`,
`Decisions.md#D65`): the evaluated per-dimension lower bounds, whether or not
the subset is empty; zeros for the full range (`Decisions.md#D9`).  The
subset must be enumerable (`Ctx.pointsOf`, else `undefinedSubset`); for a
union, the lower corner of its first range. -/
def Ctx.lowerCorner (ctx : Ctx) (ρ : Valuation) (m : Mem) (n : Name) (sub : Option Subset) : Res Idx := do
  let _ ← ctx.pointsOf ρ m n sub
  match sub with
  | none =>
    match ctx.desc? n with
    | some d => pure (d.shape.map fun _ => 0)
    | none => .err (.malformed s!"unknown container '{n}'")
  | some s =>
    let r? : Option Range := match s with
      | .range r => some r
      | .union rs => rs.head?
    match r? with
    | none => .err (.undefinedSubset s)
    | some r =>
      r.dims.mapM fun dim => Res.ofOption (ctx.evalInt ρ m dim.lo) (.undefinedSubset s)

/-! ## Copies -/

/-- Copy between two container endpoints (`Decisions.md#D7`, `#D10`):
elementwise over the zipped point lists; a stream source pops as many
elements as the destination has points, a stream destination receives all
source elements in order.  `oob` is the memlet's `allow_oob` flag
(`Decisions.md#D65`). -/
def copyData (ctx : Ctx) (ρ : Valuation) (st : St) (sd : Name) (ssub : Option Subset) (dd : Name)
    (dsub : Option Subset) (wcr : Option WcrOp) (oob : Bool) : Res St :=
  match ctx.isStream sd, ctx.isStream dd with
  | true, true => .err (.malformed "stream-to-stream copy")
  | true, false => do
    let dpts ← ctx.pointsOf ρ st.mem dd dsub
    let sp ← streamPoint ctx ρ st.mem sd ssub
    let (vs, st) ← st.popN ctx ρ sd sp dpts.length
    (dpts.zip vs).foldlM (fun st (q, v) => st.store ctx ρ dd q oob wcr v) st
  | false, true => do
    let spts ← ctx.pointsOf ρ st.mem sd ssub
    let dp ← streamPoint ctx ρ st.mem dd dsub
    let (vs, st) ← st.readMany ctx ρ sd spts oob
    st.push ctx ρ dd dp vs
  | false, false => do
    let spts ← ctx.pointsOf ρ st.mem sd ssub
    let dpts ← ctx.pointsOf ρ st.mem dd dsub
    (spts.zip dpts).foldlM (fun st (p, q) => do
      let (v, st) ← st.read ctx ρ sd p oob
      st.store ctx ρ dd q oob wcr v) st

/-- Execute a leaf copy edge (`Decisions.md#D7`, `#D10`): `copyData` between
the structural endpoints. -/
def execCopy (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res St :=
  match e.data with
  | .dep => .ok st
  | .memlet m =>
    match s.srcEndpoint e, s.dstEndpoint e with
    | .data sd ssub, .data dd dsub => copyData ctx ρ st sd ssub dd dsub m.wcr m.allowOob
    | _, _ => .err (.malformed "copy between non-container endpoints")

/-! ## Code nodes (tasklets and library nodes) -/

/-- Bind a container endpoint to an input connector (`Decisions.md#D40`): a
single-point subset is a scalar, any other subset a pointer connector
addressed relative to the subset's lower corner (`Ctx.lowerCorner`; its
whole footprint is logged as read, and only its points are readable, so an
empty subset binds a pointer that reads nothing, `Decisions.md#D65`), a
stream connector sees the current queue.  `oob` is the memlet's `allow_oob`
flag. -/
def bindInData (ctx : Ctx) (ρ : Valuation) (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    Res (ConnIn × St) :=
  if ctx.isStream d then do
    let p ← streamPoint ctx ρ st.mem d sub
    let q ← st.queue ctx ρ d p
    pure (.stream q, st)
  else do
    let pts ← ctx.pointsOf ρ st.mem d sub
    match pts with
    | [p] => do
      let (v, st) ← st.read ctx ρ d p oob
      pure (.scalar v, st)
    | _ => do
      let lo ← ctx.lowerCorner ρ st.mem d sub
      let (_, st) ← st.readMany ctx ρ d pts oob
      let m := st.mem
      let get : Idx → Option Val := fun rel =>
        if rel.length = lo.length then
          let abs := List.zipWith (· + ·) lo rel
          if pts.contains abs then
            match ctx.loc ρ m d abs oob with
            | .ok l => some (m.read l)
            | .err _ => none
          else none
        else none
      pure (.array get, st)

/-- Bind one input connector of a code node from its leaf edge; code values
come from `St.vals`. -/
def bindIn (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res (Option ConnIn × St) :=
  match s.srcEndpoint e with
  | .code key => .ok ((st.val? key).map .scalar, st)
  | .data d sub => do
    let (c, st) ← bindInData ctx ρ d sub e.allowOob st
    pure (some c, st)
  | .invalid => .err (.malformed "input edge without a valid source")

/-- Bind all input connectors of a code node. -/
def bindInputs (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) :
    Res ((Conn → Option ConnIn) × St) := do
  let (bs, st) ← (s.inEdges n.id).foldlM (fun (acc, st) e =>
      match e.data with
      | .dep => pure (acc, st)
      | .memlet _ => do
        let (ci, st) ← bindIn ctx ρ s e st
        match ci with
        | some c => pure (acc ++ [(e.dstConn.getD "", c)], st)
        | none => pure (acc, st)) (([] : List (Conn × ConnIn)), st)
  pure (fun c => (bs.find? (·.1 = c)).map (·.2), st)

/-- Deliver a scalar output to a container endpoint: a plain/WCR write of
a single element, or a push to a stream.  `oob` is the memlet's `allow_oob`
flag (`Decisions.md#D65`). -/
def storeScalar (ctx : Ctx) (ρ : Valuation) (st : St) (d : Name) (sub : Option Subset) (wcr : Option WcrOp)
    (oob : Bool) (v : Val) : Res St :=
  if ctx.isStream d then do
    let p ← streamPoint ctx ρ st.mem d sub
    st.push ctx ρ d p [v]
  else do
    let pts ← ctx.pointsOf ρ st.mem d sub
    match pts with
    | [p] => st.store ctx ρ d p oob wcr v
    | [] => .err (.malformed s!"scalar output to an empty subset of '{d}'")
    | _ => .err (.malformed s!"scalar output to several elements of '{d}'")

/-- Deliver a scalar output: `storeScalar` for containers, or a code value
for a code/entry consumer. -/
def applyScalar (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (v : Val) : Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet m =>
      match s.dstEndpoint e with
      | .code key => pure (st.setVal key v)
      | .data d sub => storeScalar ctx ρ st d sub m.wcr m.allowOob v
      | _ => .err (.malformed s!"output connector '{c}' without a valid destination")) st

/-- Deliver an element write through a pointer output connector to a
container endpoint, relative to the lower corner of the subset
(`Ctx.lowerCorner`, also for an empty subset, `Decisions.md#D65`).  The
written element must be a point of the memlet's subset; a write outside it
is the undefined behavior `outsideSubset` (`Decisions.md#D67`), like a read
outside an input subset is undefined for the tasklet.  `oob` is the
memlet's `allow_oob` flag. -/
def storeArrayWrite (ctx : Ctx) (ρ : Valuation) (st : St) (d : Name) (sub : Option Subset) (wcr : Option WcrOp)
    (oob : Bool) (rel : Idx) (v : Val) : Res St :=
  if ctx.isStream d then .err (.malformed s!"array write to stream '{d}'")
  else do
    let lo ← ctx.lowerCorner ρ st.mem d sub
    if rel.length = lo.length then do
      let pts ← ctx.pointsOf ρ st.mem d sub
      let abs := List.zipWith (· + ·) lo rel
      if pts.contains abs then st.store ctx ρ d abs oob wcr v
      else .err (.outsideSubset d abs)
    else .err (.malformed s!"array write index rank mismatch on '{d}'")

/-- Deliver an element write through a pointer output connector. -/
def applyArrayWrite (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet m =>
      match s.dstEndpoint e with
      | .data d sub => storeArrayWrite ctx ρ st d sub m.wcr m.allowOob rel v
      | _ => .err (.malformed s!"array write through connector '{c}' to a non-container")) st

/-- Push values to a stream endpoint. -/
def pushTo (ctx : Ctx) (ρ : Valuation) (st : St) (d : Name) (sub : Option Subset) (vs : List Val) : Res St :=
  if ctx.isStream d then do
    let p ← streamPoint ctx ρ st.mem d sub
    st.push ctx ρ d p vs
  else .err (.malformed s!"push to non-stream '{d}'")

/-- Deliver pushes through a stream output connector. -/
def applyPush (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (vs : List Val) : Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match s.dstEndpoint e with
    | .data d sub => pushTo ctx ρ st d sub vs
    | _ => .err (.malformed s!"push through connector '{c}' to a non-container")) st

/-- Pop `k` elements from a stream endpoint. -/
def popFrom (ctx : Ctx) (ρ : Valuation) (st : St) (d : Name) (sub : Option Subset) (k : Nat) : Res St :=
  if ctx.isStream d then do
    let p ← streamPoint ctx ρ st.mem d sub
    let (_, st) ← st.popN ctx ρ d p k
    pure st
  else .err (.malformed s!"pop from non-stream '{d}'")

/-- Pop `k` elements from the stream behind an input connector (`Decisions.md#D14`). -/
def applyPop (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (k : Nat) : Res St :=
  match s.inEdgeOn n c with
  | none => .err (.malformed s!"pop through unknown connector '{c}'")
  | some e =>
    match s.srcEndpoint e with
    | .data d sub => popFrom ctx ρ st d sub k
    | _ => .err (.malformed s!"pop through connector '{c}' from a non-container")

/-- Apply the effects of one code-node execution: pops, then scalar outputs,
element writes and pushes. -/
def applyOut (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (out : TaskletOut) : Res St := do
  let st ← out.pops.foldlM (fun st (c, k) => applyPop ctx ρ s n st c k) st
  let st ← out.scalars.foldlM (fun st (c, v) => applyScalar ctx ρ s n st c v) st
  let st ← out.arrayWrites.foldlM (fun st (c, rel, v) => applyArrayWrite ctx ρ s n st c rel v) st
  out.pushes.foldlM (fun st (c, vs) => applyPush ctx ρ s n st c vs) st

/-- Does the code node have a dynamic WCR out-memlet (a fetch-and-add,
`Decisions.md#D47`)? -/
def State.hasDynamicWcrOut (s : State) (n : DNode) : Bool :=
  (s.outEdges n.id).any fun e => match e.data with
    | .memlet m => m.dynamic && m.hasWcr
    | .dep => false

/-- Execute a code node through an uninterpreted body `run`. -/
def execCode (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (undefinedErr : ErrReason) : Res St :=
  if s.hasDynamicWcrOut n then
    .err (.unsupported s!"dynamic WCR output of '{n.label}' (fetch-and-add, Decisions.md#D47)")
  else do
    let (ins, st) ← bindInputs ctx ρ s n st
    match run ins with
    | none => .err undefinedErr
    | some out => applyOut ctx ρ s n st out

/-! ## Reduce -/

/-- Coordinates of a point that are not reduced. -/
def reduceKeep (axes : List Nat) (p : Idx) : Idx :=
  p.zipIdx.filterMap fun (x, k) => if axes.contains k then none else some x

/-- The groups of a reduction, paired in order with the output points: the
distinct non-reduced coordinates of the input points; an empty input (e.g. an
empty view, `Decisions.md#D65`) gives every output point an empty group. -/
def reduceGroups (axes : List Nat) (spts dpts : List Idx) : List Idx :=
  if spts.isEmpty then dpts.map fun _ => [] else (spts.map (reduceKeep axes)).eraseDups

/-- Direct semantics of `Reduce` between two container endpoints
(`Decisions.md#D37`, mirroring `ExpandReducePure`,
`dace/libraries/standard/nodes/reduce.py:35-166`): input points are grouped
by their non-reduced coordinates, in the order of the output points; with an
`identity` the fold starts there and the result is written (plainly, or
through the out-memlet's WCR); without one the fold of the group is
contributed to the existing output through the reduction operator (the pure
expansion has no initialization state).  An empty input reduces every output
point over an empty group (the identity, or nothing; the expansion's
`reduce_init` map writes the identity to the whole output, `reduce.py:98-110`).
`soob`/`doob` are the `allow_oob`
flags of the input and output memlets (`Decisions.md#D65`). -/
def reduceData (ctx : Ctx) (ρ : Valuation) (st : St) (props : ReduceProps) (sd : Name) (ssub : Option Subset)
    (dd : Name) (dsub : Option Subset) (outWcr : Option WcrOp) (soob doob : Bool) : Res St :=
  if ctx.isStream sd || ctx.isStream dd then .err (.malformed "reduce over a stream")
  else do
    let spts ← ctx.pointsOf ρ st.mem sd ssub
    let dpts ← ctx.pointsOf ρ st.mem dd dsub
    let ndim := (spts.head?.map List.length).getD 0
    let axes := props.axes.getD (List.range ndim)
    let groups := reduceGroups axes spts dpts
    if groups.length ≠ dpts.length then .err (.malformed "reduce output shape mismatch")
    else do
      let (vals, st) ← st.readMany ctx ρ sd spts soob
      let pairs := spts.zip vals
      (groups.zip dpts).foldlM (fun st (g, q) =>
        let gv := (pairs.filter fun (p, _) => reduceKeep axes p = g).map (·.2)
        match props.identity with
        | some ide => do
          let i ← ctx.evalR ρ st.mem ide
          st.store ctx ρ dd q doob outWcr (gv.foldl (ctx.I.wcr props.wcr) i)
        | none =>
          match gv with
          | [] => pure st
          | v :: rest =>
            st.wcr ctx ρ dd q doob (outWcr.getD props.wcr) (rest.foldl (ctx.I.wcr props.wcr) v)) st

/-- The `Reduce` library node: `reduceData` between its single input and
output endpoints. -/
def execReduce (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (props : ReduceProps) (st : St) : Res St :=
  let ins := (s.inEdges n.id).filter fun e => e.data matches .memlet _
  let outs := (s.outEdges n.id).filter fun e => e.data matches .memlet _
  match ins, outs with
  | [ie], [oe] =>
    match s.srcEndpoint ie, s.dstEndpoint oe, oe.data.memlet? with
    | .data sd ssub, .data dd dsub, some om =>
      reduceData ctx ρ st props sd ssub dd dsub om.wcr ie.allowOob om.allowOob
    | _, _, _ => .err (.malformed "reduce node not between containers")
  | _, _ => .err (.malformed "reduce node must have exactly one input and one output")

/-! ## Views and references -/

/-- Does aliasing `d` to `base` reinterpret the element type
(`Decisions.md#D55`)?  Structures and container arrays (whose descriptors
carry no element type of their own) never do. -/
def Ctx.reinterprets (ctx : Ctx) (d base : Name) : Bool :=
  match ctx.desc? d, ctx.desc? base with
  | some dd, some bd =>
    let plain (x : DataDesc) : Bool := !(x.kind matches .structure _ _ || x.kind matches .containerArray _) &&
      !(x.kind matches .view .structure || x.kind matches .view .containerArray) &&
      !(x.kind matches .reference .structure || x.kind matches .reference .containerArray)
    plain dd && plain bd && dd.dtype ≠ bd.dtype
  | _, _ => false

/-- Alias binding of container `d` to the lower corner of `sub` in `base`
(`Ctx.lowerCorner`, its address `Ctx.addr`: binding is not an access, so
the corner of an empty subset may lie outside `base`; such a view is valid,
and element accesses through it are checked against the view's own shape,
`Decisions.md#D65`).  A view or reference whose element type differs from
its base's is not modeled (`unsupported`, `Decisions.md#D55`, `#D59`). -/
def aliasTo (ctx : Ctx) (ρ : Valuation) (st : St) (d base : Name) (sub : Option Subset) : Res St :=
  if ctx.reinterprets d base then
    let size (n : Name) := (ctx.desc? n).bind (·.dtype.bytes?)
    let how := if size d == size base then "a bitcast of equal element size" else "a different element size"
    .err (.unsupported s!"reinterpretation of '{base}' as '{d}' ({how}, Decisions.md#D55)")
  else do
    let lo ← ctx.lowerCorner ρ st.mem base sub
    let l ← ctx.addr ρ st.mem base lo
    pure (st.bindAlias (ctx.cname d) { base := l.name, offset := l.offset })

/-- Bind a view access node to its viewed data (`Decisions.md#D31`): the
access node at the start (read view) or end (write view) of the memlet path
through the view edge, offset by the lower corner of the viewed subset. -/
def bindView (ctx : Ctx) (ρ : Valuation) (s : State) (v : DNode) (d : Name) (st : St) : Res St :=
  match s.viewEdge? v with
  | none => .err (.malformed s!"ambiguous or invalid view edge for '{d}'")
  | some (e, isIn) =>
    let base? := if isIn then (s.node? (s.pathStart e).src).bind DNode.accessData?
                 else (s.node? (s.pathEnd e).dst).bind DNode.accessData?
    let sub := if isIn then State.leafSrcSubset e else State.leafDstSubset e
    match base? with
    | none => .err (.malformed s!"view edge of '{d}' does not reach an access node")
    | some base => aliasTo ctx ρ st d base sub

/-- The container a view access node views (the access node at the other end
of its view edge, `bindView`). -/
def State.viewedData? (s : State) (v : DNode) : Option Name :=
  match s.viewEdge? v with
  | none => none
  | some (e, isIn) =>
    if isIn then (s.node? (s.pathStart e).src).bind DNode.accessData?
    else (s.node? (s.pathEnd e).dst).bind DNode.accessData?

/-- Is node `id` an access node of a view? -/
def State.isViewNode (ctx : Ctx) (s : State) (id : NodeId) : Bool :=
  match s.node? id with
  | some n =>
    match n.kind with
    | .access d => ctx.isView d
    | _ => false
  | none => false

/-- Generations of the pending views (`dataOf`: the container of a node,
`baseOf`: the container it views): those whose viewed container is no
pending view first, in list order, then the next generation. -/
def viewGensWith (dataOf baseOf : NodeId → Option Name) : Nat → List NodeId → List NodeId
  | 0, pending => pending
  | _ + 1, [] => []
  | f + 1, pending =>
    let pendingData := pending.filterMap dataOf
    let ready := pending.filter fun id =>
      match baseOf id with
      | some b => !pendingData.contains b
      | none => true
    if ready.isEmpty then pending
    else ready ++ viewGensWith dataOf baseOf f (pending.filter fun id => !ready.contains id)

theorem viewGensWith_congr {d₁ d₂ b₁ b₂ : NodeId → Option Name} : ∀ (f : Nat) (pending : List NodeId),
    (∀ id ∈ pending, d₁ id = d₂ id ∧ b₁ id = b₂ id) → viewGensWith d₁ b₁ f pending = viewGensWith d₂ b₂ f pending
  | 0, _, _ => rfl
  | _ + 1, [], _ => rfl
  | f + 1, p :: ps, h => by
    have hpd : (p :: ps).filterMap d₁ = (p :: ps).filterMap d₂ := List.filterMap_congr fun id hid => (h id hid).1
    have hready : ((p :: ps).filter fun id => match b₁ id with
        | some b => !((p :: ps).filterMap d₁).contains b
        | none => true) = ((p :: ps).filter fun id => match b₂ id with
        | some b => !((p :: ps).filterMap d₂).contains b
        | none => true) := List.filter_congr fun id hid => by rw [(h id hid).2, hpd]
    unfold viewGensWith
    simp only
    rw [hready]
    split
    · rfl
    · rw [viewGensWith_congr f _ fun id hid => h id (List.mem_of_mem_filter hid)]

theorem viewGensWith_sub {d b : NodeId → Option Name} : ∀ (f : Nat) (pending : List NodeId) {id : NodeId},
    id ∈ viewGensWith d b f pending → id ∈ pending
  | 0, _, _, h => h
  | _ + 1, [], _, h => by simp [viewGensWith] at h
  | f + 1, p :: ps, id, h => by
    unfold viewGensWith at h
    simp only at h
    split at h
    · exact h
    · rcases List.mem_append.1 h with h | h
      · exact List.mem_of_mem_filter h
      · exact List.mem_of_mem_filter (viewGensWith_sub f _ h)

/-- `viewGensWith` for the views of a state. -/
def State.viewGens (s : State) (f : Nat) (pending : List NodeId) : List NodeId :=
  viewGensWith (fun id => (s.node? id).bind DNode.accessData?) (fun id => (s.node? id).bind s.viewedData?) f pending

/-- What a view access node binds (`bindView`): the container it views and
the viewed subset. -/
def State.viewBinding? (s : State) (v : DNode) : Option (Option Name × Option Subset) :=
  (s.viewEdge? v).map fun (e, isIn) =>
    (if isIn then (s.node? (s.pathStart e).src).bind DNode.accessData?
     else (s.node? (s.pathEnd e).dst).bind DNode.accessData?,
     if isIn then State.leafSrcSubset e else State.leafDstSubset e)

/-- The signature of an access node: its container and what it binds. -/
def State.viewSig (s : State) (id : NodeId) : Option (Name × Option (Option Name × Option Subset)) :=
  (s.node? id).bind fun n => n.accessData?.map fun d => (d, s.viewBinding? n)

/-- The nodes of `l` without those whose signature is that of an earlier
kept node (`seen`); nodes without a signature are kept. -/
def keepFirstSig {α : Type} [DecidableEq α] (sig : NodeId → Option α) : List α → List NodeId → List NodeId
  | _, [] => []
  | seen, x :: xs =>
    match sig x with
    | some a => if seen.contains a then keepFirstSig sig seen xs else x :: keepFirstSig sig (a :: seen) xs
    | none => x :: keepFirstSig sig seen xs

theorem keepFirstSig_sub {α : Type} [DecidableEq α] {sig : NodeId → Option α} {x : NodeId} :
    ∀ {seen : List α} {l : List NodeId}, x ∈ keepFirstSig sig seen l → x ∈ l
  | _, [], h => by simp [keepFirstSig] at h
  | seen, y :: ys, h => by
    unfold keepFirstSig at h
    split at h
    · split at h
      · exact List.mem_cons_of_mem _ (keepFirstSig_sub h)
      · rcases List.mem_cons.1 h with rfl | h
        · exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (keepFirstSig_sub h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (keepFirstSig_sub h)

theorem keepFirstSig_congr {α : Type} [DecidableEq α] {sig₁ sig₂ : NodeId → Option α} :
    ∀ {seen : List α} {l : List NodeId}, (∀ x ∈ l, sig₁ x = sig₂ x) →
      keepFirstSig sig₁ seen l = keepFirstSig sig₂ seen l
  | _, [], _ => rfl
  | seen, y :: ys, h => by
    have hy := h y List.mem_cons_self
    have hys : ∀ x ∈ ys, sig₁ x = sig₂ x := fun x hx => h x (List.mem_cons_of_mem _ hx)
    simp only [keepFirstSig, hy]
    split
    · split
      · exact keepFirstSig_congr hys
      · rw [keepFirstSig_congr hys]
    · rw [keepFirstSig_congr hys]

/-- The order in which the views among `ids` are bound (`Decisions.md#D31`):
every view after the view it views, like code generation allocating the
viewed data first (`cpu.py`, `allocate_view`); otherwise in node order.  A
binding of several view nodes (the read and the write side of one view) is
made once, by the first of them. -/
def State.viewOrder (ctx : Ctx) (s : State) (ids : List NodeId) : List NodeId :=
  let views := keepFirstSig s.viewSig [] (ids.filter (s.isViewNode ctx))
  s.viewGens views.length views

/-- The views bound are among the given nodes. -/
theorem State.viewOrder_sub {ctx : Ctx} {s : State} {ids : List NodeId} {id : NodeId}
    (h : id ∈ s.viewOrder ctx ids) : id ∈ ids :=
  List.mem_of_mem_filter (keepFirstSig_sub (viewGensWith_sub _ _ h))

/-- Bind every view among the given nodes, in list order (the prologue of a
sequential execution binds `State.viewOrder` of its direct nodes). -/
def bindViews (ctx : Ctx) (ρ : Valuation) (s : State) (ids : List NodeId) (st : St) : Res St :=
  ids.foldlM (fun st id =>
    match s.node? id with
    | some n =>
      match n.kind with
      | .access d => if ctx.isView d then bindView ctx ρ s n d st else pure st
      | _ => pure st
    | none => pure st) st

/-- Reference `set` through the in-edge `e` of reference `d` (`Decisions.md#D11`, `#D32`). -/
def refSet (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (d : Name) (st : St) : Res St :=
  match s.srcEndpoint e with
  | .data base sub => aliasTo ctx ρ st d base sub
  | .code _ => .err (.unsupported s!"reference set of '{d}' from a code node (Decisions.md#D55)")
  | _ => .err (.malformed s!"reference '{d}' set from a non-container source")

/-! ## Access and exit nodes -/

/-- The reference whose `set` connector the memlet path from `e` reaches
through exit nodes, if any. -/
def refSetTarget? (ctx : Ctx) (s : State) (e : DEdge) : Option Name :=
  let r := s.dstRootEdge e
  if r.dstConn = some "set" then
    ((s.node? r.dst).bind DNode.accessData?).filter ctx.isReference
  else none

/-- Execute the copies into a node (`Decisions.md#D29`, `#D30`): every
memlet in-edge that is the *leaf* of its memlet tree and whose source is a
container, except view definitions.  Routing edges (`State.isRouting`: here,
edges leaving a scope exit through `OUT_`) carry no copy of their own: the
copy they route executes at the leaf edge into the innermost exit, per
iteration; entry and exit nodes have no effect of their own.  An edge whose path ends at the
`set` connector of a reference binds the reference instead (`#D32`). -/
def execCopiesInto (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) : Res St :=
  (s.inEdges n.id).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet _ =>
      if ctx.isViewDef s e || s.isRouting e then pure st
      else
        match refSetTarget? ctx s e with
        | some d => refSet ctx ρ s e d st
        | none =>
          match s.srcEndpoint e with
          | .data _ _ => execCopy ctx ρ s e st
          | _ => pure st) st

/-- Nodes whose execution is a function of the state (no recursion into the
syntax). -/
def DNode.isSimple (n : DNode) : Bool :=
  match n.kind with
  | .access _ | .tasklet _ | .library _ | .mapExit | .consumeExit => true
  | .mapEntry _ | .consumeEntry _ | .nested _ _ _ => false

/-- Execution of a simple node. -/
def execSimple (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) : Res St :=
  match n.kind with
  | .access _ | .mapExit | .consumeExit => execCopiesInto ctx ρ s n st
  | .tasklet t => execCode ctx ρ s n st (ctx.I.tasklet t ρ) (.taskletUndefined t.label)
  | .library l =>
    match l.reduce with
    | some props => execReduce ctx ρ s n props st
    | none => execCode ctx ρ s n st (ctx.I.library l.classpath) (.libraryUndefined l.classpath)
  | .mapEntry _ | .consumeEntry _ | .nested _ _ _ =>
    .err (.malformed "scope entry or nested SDFG executed as a simple node")

/-! ## Scopes -/

/-- Bind the dynamic inputs of a scope entry as symbols (`Decisions.md#D12`). -/
def bindDynamicInputs (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) : Res (Valuation × St) :=
  (s.inEdges n.id).foldlM (fun (ρ, st) e =>
    if s.isDynamicInput e then
      match e.data, e.dstConn with
      | .memlet _, some c => do
        let (v?, st) ← readEdgeScalar ctx ρ s e st
        match v? with
        | some v => pure (Valuation.set ρ c v, st)
        | none => pure (ρ, st)
      | _, _ => pure (ρ, st)
    else pure (ρ, st)) (ρ, st)

/-- Set-up of a map scope: dynamic inputs, then the iteration space. -/
def mapSetup (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (mp : MapNode) (st : St) :
    Res (Valuation × St × List Idx) := do
  let (ρ', st) ← bindDynamicInputs ctx ρ s n st
  let pts ← ctx.rangePoints ρ' st.mem mp.range
  pure (ρ', st, pts)

/-- Initial state of one iteration/task: the pre-scope memory, an empty log. -/
def iterStart (priv : List Name) (pre : St) : St :=
  { mem := pre.mem.allocAll priv, log := Log.empty, vals := pre.vals }

/-- Merge the logs of the parallel executions of a scope onto the pre-scope
state (`Decisions.md#D3`). -/
def scopeFinish (ctx : Ctx) (pre : St) (logs : List Log) : Res St :=
  match merge ctx.I.wcr pre.mem logs with
  | .ok m' => .ok { mem := m', log := pre.log.seq (Log.unionAll logs), vals := pre.vals }
  | .error e => .err (.race e)

/-! ### Consume -/

/-- The stream consumed by a consume entry (root of its `IN_stream` edge),
as a resolved queue location. -/
def consumeStream (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (m : Mem) : Res Loc :=
  match (s.inEdges n.id).find? (·.dstConn = some "IN_stream") with
  | none => .err (.malformed "consume entry without IN_stream")
  | some e =>
    match s.srcEndpoint e with
    | .data d sub =>
      if ctx.isStream d then do
        let p ← streamPoint ctx ρ m d sub
        let l ← ctx.loc ρ m d p
        pure (m.resolve l)
      else .err (.malformed s!"consume of non-stream '{d}'")
    | _ => .err (.malformed "consume entry not fed by a stream")

/-- Set-up of a consume scope: dynamic inputs, the stream, the number of
processing elements. -/
def consumeSetup (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (c : ConsumeNode) (st : St) :
    Res (Valuation × St × Loc × Int) := do
  let (ρ', st) ← bindDynamicInputs ctx ρ s n st
  let l ← consumeStream ctx ρ' s n st.mem
  let np ← ctx.evalIntR ρ' st.mem c.numPes
  pure (ρ', st, l, np)

/-- Whether a consume scope runs another task, evaluated with the current
queue content: the scope's condition is a quiescence (stop) condition,
`while (!quiescence())` (`stream.h` `consume_cond`); without one, while the
stream is not empty (`stream.h` `consume`). -/
def consumeCond (ctx : Ctx) (ρ : Valuation) (c : ConsumeNode) (m : Mem) (l : Loc) (q : List Val) : Res Bool :=
  match c.condition with
  | none => .ok (!q.isEmpty)
  | some e => (ctx.evalBoolR ρ (m.setQueue l q) e).map (!·)

/-- Memory name of the chunk container of the consume scope `entry`
(`consumeChunkName` of its stream, `Decisions.md#D38`). -/
def chunkMemName (ctx : Ctx) (s : State) (entry : NodeId) : Option Name :=
  (s.consumedStream? entry).map fun d => ctx.cname (consumeChunkName d)

/-- Initial state of a consume task: the pre-scope memory with the queue
advanced past the chunk and the chunk written to its container
(`Decisions.md#D38`). -/
def taskStart (ctx : Ctx) (s : State) (entry : NodeId) (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : St)
    (q : List Val) : St :=
  let m := (pre.mem.allocAll priv).setQueue l (q.drop c.chunksize)
  { mem := match chunkMemName ctx s entry with
      | some name => m.writeChunk name (q.take c.chunksize)
      | none => m,
    log := Log.empty, vals := pre.vals }

namespace Log

/-- Remove the queue effects on `l`. -/
def stripQueue (g : Log) (l : Loc) : Log := { g with queues := g.queues.filter fun (l', _) => l' ≠ l }

/-- Values pushed to `l` by this log, in order. -/
def pushedTo (g : Log) (l : Loc) : List Val :=
  g.queues.flatMap fun (l', eff) =>
    if l' = l then
      match eff with
      | .pushOnly vs => vs
      | .exclusive _ => []
    else []

end Log

/-- Log contributed by one consume task: private transients dropped, effects
on the consumed stream handled by the scope. -/
def taskLog (st : St) (priv : List Name) (l : Loc) : Log := (st.log.dropNames priv).stripQueue l

/-- Values a task pushed to the consumed stream (deferred until after the scope). -/
def taskPushed (st : St) (priv : List Name) (l : Loc) : List Val := (st.log.dropNames priv).pushedTo l

/-- Finish a consume scope: merge the task logs, then set the consumed
queue to the remaining elements followed by the deferred pushes. -/
def consumeFinish (ctx : Ctx) (pre : St) (l : Loc) (logs : List Log) (rest pushed : List Val) : Res St :=
  match merge ctx.I.wcr pre.mem logs with
  | .ok m' =>
    let final := rest ++ pushed
    .ok { mem := m'.setQueue l final,
          log := (pre.log.seq (Log.unionAll logs)).seq { queues := [(l, .exclusive final)] },
          vals := pre.vals }
  | .error e => .err (.race e)

/-! ## Nested SDFGs -/

/-- Inner valuation of a nested SDFG (`Decisions.md#D34`): mapped symbols
are evaluated in the outer context, unmapped ones are inherited. -/
def nestedValuation (ctx : Ctx) (ρ : Valuation) (m : Mem) (mapping : List (Sym × Expr)) : Res Valuation := do
  let vs ← mapping.mapM fun (x, e) => do
    let v ← ctx.evalR ρ m e
    pure (x, v)
  pure (Valuation.setAll ρ vs)

/-- Connector names of a nested SDFG node (in-out connectors once). -/
def nestedConns (n : DNode) : List Conn := ((n.inConns ++ n.outConns).map Prod.fst).eraseDups

/-- The outer end of the edge on connector `c` of a nested SDFG node. -/
def nestedEndpoint (s : State) (n : DNode) (c : Conn) : Endpoint :=
  match s.inEdgeOn n c with
  | some e => s.srcEndpoint e
  | none =>
    match s.outEdgesOn n c with
    | e :: _ => s.dstEndpoint e
    | [] => .invalid

/-- The connector bindings of a nested SDFG node (`Decisions.md#D15`,
`#D34`): a connector on a data edge names the memory of the outer container
(offset `0`).  Connectors on code edges are not bound. -/
def nestedBindings (ctx : Ctx) (s : State) (n : DNode) : Res (List (Name × Name)) :=
  (nestedConns n).foldlM (fun acc c =>
    match nestedEndpoint s n c with
    | .data base _ => pure (acc ++ [(c, ctx.cname base)])
    | .code _ => pure acc
    | .invalid => .err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) []

/-- Copy the scalar values on code→nested edges into the inner scalars.
Returns the memory names of the connectors on code edges. -/
def bindCodeInputs (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (st : St) : Res (St × List Name) :=
  (nestedConns n).foldlM (fun (st, codes) c =>
    match nestedEndpoint s n c with
    | .code key =>
      match st.val? key with
      | some v => do
        let l ← ctx'.loc ρ' st.mem c []
        pure ({ st with mem := st.mem.write l v }, codes ++ [ctx'.cname c])
      | none => pure (st, codes ++ [ctx'.cname c])
    | _ => pure (st, codes)) (st, [])

/-- The transients of an SDFG allocated per call (`Decisions.md#D68`): all
but `Persistent`, `Global` and `External` ones. -/
def SDFG.callTransients (g : SDFG) : List Name :=
  (g.arrays.filter fun (_, d) =>
    d.transient && !(d.lifetime == .persistent || d.lifetime == .global || d.lifetime == .external)).map Prod.fst

/-- Allocate containers of the current SDFG (`Decisions.md#D68`): streams
get an empty queue, other containers uninitialized cells. -/
def Ctx.allocate (ctx : Ctx) (ts : List Name) (m : Mem) : Mem :=
  ts.foldl (fun m t => if ctx.isStream t then m.alloc (ctx.cname t) else m.allocCells (ctx.cname t)) m

@[simp] theorem Ctx.allocate_aliases (ctx : Ctx) (ts : List Name) (m : Mem) :
    (ctx.allocate ts m).aliases = m.aliases := by
  induction ts generalizing m with
  | nil => rfl
  | cons t ts ih =>
    simp only [Ctx.allocate, List.foldl_cons] at ih ⊢
    rw [ih]
    split <;> rfl

/-- A nested SDFG may own only per-call transients (`Decisions.md#D63`,
`#D68`): canonical SDFGs hoist `Persistent`, `External` and `Global`
transients to the top-level SDFG (`DataDesc.sharedAcrossCalls`). -/
def nestedCheck (inner : SDFG) : Res Unit :=
  if inner.ownsShared then
    .err (.malformed
      s!"nested SDFG '{inner.name}' owns a Persistent, External or Global transient (hoist it to the top level)")
  else .ok ()

/-- Set-up of a nested invocation: connector bindings, inner context,
valuation, the per-call transients allocated (fresh containers: uninitialized
cells and empty queues, like map-private transients), scalar code inputs
copied.
Returns the memory names whose log entries stay inside the call: the
connectors on code edges and the per-call transients. -/
def nestedSetup (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (inner : SDFG) (mapping : List (Sym × Expr))
    (st : St) : Res (Ctx × Valuation × St × List Name) := do
  nestedCheck inner
  let conns ← nestedBindings ctx s n
  let ctx' := ctx.enter inner conns
  let ρ' ← nestedValuation ctx ρ st.mem mapping
  let st := { st with mem := st.mem.allocAll (inner.callTransients.map ctx'.cname) }
  let (st, codes) ← bindCodeInputs ctx' ρ' s n st
  pure (ctx', ρ', st, codes ++ inner.callTransients.map ctx'.cname)

/-- Tear-down of a nested invocation: keep the memory, but free the names
private to the call (`dropped`, from `nestedSetup`: its per-call transients
and code-connector scalars, which become uninitialized again, like DaCe
deallocating them on return, `Decisions.md#D68`), contribute the inner log
without them, and pass scalar outputs, read from the inner memory, on to code
consumers.  After a call nothing is left at its private names, so their
memory names (which contain the node id, `Ctx.enter`) are unobservable. -/
def nestedFinish (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (dropped : List Name) (pre : St)
    (m₂ : Mem) (g₂ : Log) : Res St :=
  let st : St := { mem := m₂.allocAll dropped, log := pre.log.seq (g₂.dropNames dropped), vals := pre.vals }
  (s.outEdges n.id).foldlM (fun st e =>
    match e.data, e.srcConn with
    | .memlet _, some c =>
      match s.dstEndpoint e with
      | .code key => do
        let l ← ctx'.loc ρ' m₂ c []
        match (m₂.read l).defined? with
        | some v => pure (st.setVal key v)
        | none => .err (.uninitRead c [])
      | _ => pure st
    | _, _ => pure st) st

/-! ## Control flow -/

/-- The out-edges of a block whose condition holds (all conditions are
evaluated, `Decisions.md#D35`). -/
def trueEdges (ctx : Ctx) (r : Region) (cst : CSt) (bid : BlockId) : Res (List IEdge) :=
  (r.outEdges bid).filterMapM fun e => do
    let b ← ctx.evalBoolR cst.ρ cst.mem e.cond
    pure (if b then some e else none)

/-- Simultaneous symbol assignments (right-hand sides in the old valuation). -/
def applyAssigns (ctx : Ctx) (cst : CSt) (assigns : List (Sym × Expr)) : Res CSt := do
  let vs ← assigns.mapM fun (x, e) => do
    let v ← ctx.evalR cst.ρ cst.mem e
    pure (x, v)
  pure { cst with ρ := Valuation.setAll cst.ρ vs }

/-- An optional single assignment (loop init/update). -/
def applyAssign? (ctx : Ctx) (cst : CSt) : Option (Sym × Expr) → Res CSt
  | none => .ok cst
  | some a => applyAssigns ctx cst [a]

/-- Does control proceed to the loop tail after the body? -/
def Ctl.continuesLoop : Ctl → Bool
  | .normal | .cont => true
  | .brk | .ret => false

/-- Finish a state: merge the component logs (`Decisions.md#D4`). -/
def stateFinish (ctx : Ctx) (cst : CSt) (logs : List Log) : Res (Ctl × CSt) :=
  match merge ctx.I.wcr cst.mem logs with
  | .ok m' => .ok (.normal, { cst with mem := m', log := cst.log.seq (Log.unionAll logs) })
  | .error e => .err (.race e)

/-! ## Sequential executions and happens-before races (`Decisions.md#D61`)

The nodes of one sequential execution (a scope body or a state component)
run one after the other in some topological order, each from the current
local view with an **empty** log; its own log (*delta*) is recorded in the
execution's *trace* and appended to the accumulated log with `Log.seq`.
Two nodes of the trace that no path of the state orders (`State.reaches`)
must not have conflicting deltas (`Log.conflicts`, the conflict notion of
`merge`): such a pair is a race (`doc/sdfg/synchronization.rst`, Execution
Model), whatever order the execution happened to pick.  The delta of a scope
entry contains its whole scope execution, so nodes of nested scopes are
covered by their entry. -/

/-- The state a node of a sequential execution starts from: the current
local view and code→code values, with an empty log. -/
def St.forNode (st : St) : St := { st with log := Log.empty }

/-- Continue a sequential execution at `st` after a node that ended in
`st'` (whose log is the node's delta): accumulate the delta. -/
def St.afterNode (st st' : St) : St := { st' with log := st.log.seq st'.log }

namespace State

/-- Two trace entries race: their deltas conflict and neither node reaches
the other. -/
def hbConflict (s : State) (x y : NodeId × Log) : Bool :=
  Log.conflicts x.2 y.2 && !s.reaches x.1 y.1 && !s.reaches y.1 x.1

/-- No two entries of the trace race (`Log.pairwiseFree` restricted to the
pairs that the state leaves unordered). -/
def hbFree (s : State) : List (NodeId × Log) → Bool
  | [] => true
  | x :: xs => xs.all (fun y => !s.hbConflict x y) && s.hbFree xs

/-- The first racing pair of a trace, as the reported error
(`MergeError.unordered a b w` with a witness `w` of the conflict). -/
def hbWitness (s : State) : List (NodeId × Log) → MergeError
  | [] => .race default
  | x :: xs =>
    match xs.find? (fun y => s.hbConflict x y) with
    | some y => .unordered x.1 y.1 (Merge.witness x.2 y.2)
    | none => s.hbWitness xs

end State

end Dace
