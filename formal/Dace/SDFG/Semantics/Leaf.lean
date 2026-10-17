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
* every element access goes through `Ctx.loc` and is logged resolved;
* the consume chunk is a task-private pseudo-container named by
  `chunkName`, never logged.
-/

set_option autoImplicit false

namespace Dace

open State (Endpoint)

/-! ## Consume chunk pseudo-container -/

/-- Memory name of the chunk consumed by the consume scope `entry`
(`__dace_<label>_element(s)` in `cpu.py:2273-2291`). -/
def chunkName (ctx : Ctx) (entry : NodeId) : Name :=
  ctx.cname ("__consume_chunk_" ++ toString entry)

namespace Mem

/-- Number of elements of a chunk (stored at the whole-container location). -/
def chunkCount (m : Mem) (name : Name) : Nat := ((m.cells (Loc.whole name)).toInt?.getD 0).toNat

/-- Elements of a chunk in order. -/
def chunkVals (m : Mem) (name : Name) : List Val :=
  (List.range (m.chunkCount name)).map fun k => m.cells (Loc.elem name k)

/-- Write a chunk (count and elements), bypassing aliases and logs. -/
def writeChunk (m : Mem) (name : Name) (vs : List Val) : Mem :=
  let m := { m with cells := Function.update m.cells (Loc.whole name) (.int vs.length) }
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

/-- The scalar flowing along a leaf edge into a scope entry or code node
(`Decisions.md#D42`): a single element of a container, one popped element
of a stream, a code value, or the first chunk element.  `none` = no value
flows (unassigned dynamic output upstream, `Decisions.md#D13`). -/
def readEdgeScalar (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res (Option Val × St) :=
  match s.srcEndpoint e with
  | .code key => .ok (st.val? key, st)
  | .chunk entry => .ok ((st.mem.chunkVals (chunkName ctx entry)).head?, st)
  | .data d sub => do
    let pts ← ctx.pointsOf ρ st.mem d sub
    match pts with
    | [p] =>
      if ctx.isStream d then do
        let (vs, st) ← st.popN ctx ρ d p 1
        pure (vs.head?, st)
      else do
        let (v, st) ← st.read ctx ρ d p
        pure (some v, st)
    | _ => .err (.malformed s!"scalar read of '{d}' is not a single element")
  | .invalid => .err (.malformed "leaf edge without a valid source")

/-- Location of the (first) element of a stream endpoint. -/
def streamPoint (ctx : Ctx) (ρ : Valuation) (m : Mem) (d : Name) (sub : Option Subset) : Res Idx := do
  let pts ← ctx.pointsOf ρ m d sub
  Res.ofOption pts.head? (.malformed s!"empty subset of stream '{d}'")

/-! ## Copies -/

/-- Execute a leaf copy edge (`Decisions.md#D7`, `#D10`): elementwise over
the zipped point lists; a stream source pops as many elements as the
destination has points, a stream destination receives all source elements
in order; the consume chunk copies like an array source. -/
def execCopy (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res St :=
  match e.data with
  | .dep => .ok st
  | .memlet m =>
    match s.srcEndpoint e, s.dstEndpoint e with
    | .data sd ssub, .data dd dsub =>
      match ctx.isStream sd, ctx.isStream dd with
      | true, true => .err (.malformed "stream-to-stream copy")
      | true, false => do
        let dpts ← ctx.pointsOf ρ st.mem dd dsub
        let sp ← streamPoint ctx ρ st.mem sd ssub
        let (vs, st) ← st.popN ctx ρ sd sp dpts.length
        (dpts.zip vs).foldlM (fun st (q, v) => st.store ctx ρ dd q m.wcr v) st
      | false, true => do
        let spts ← ctx.pointsOf ρ st.mem sd ssub
        let dp ← streamPoint ctx ρ st.mem dd dsub
        let (vs, st) ← st.readMany ctx ρ sd spts
        st.push ctx ρ dd dp vs
      | false, false => do
        let spts ← ctx.pointsOf ρ st.mem sd ssub
        let dpts ← ctx.pointsOf ρ st.mem dd dsub
        (spts.zip dpts).foldlM (fun st (p, q) => do
          let (v, st) ← st.read ctx ρ sd p
          st.store ctx ρ dd q m.wcr v) st
    | .chunk entry, .data dd dsub => do
      let vs := st.mem.chunkVals (chunkName ctx entry)
      if ctx.isStream dd then do
        let dp ← streamPoint ctx ρ st.mem dd dsub
        st.push ctx ρ dd dp vs
      else do
        let dpts ← ctx.pointsOf ρ st.mem dd dsub
        (dpts.zip vs).foldlM (fun st (q, v) => st.store ctx ρ dd q m.wcr v) st
    | _, _ => .err (.malformed "copy between non-container endpoints")

/-! ## Code nodes (tasklets and library nodes) -/

/-- Bind one input connector (`Decisions.md#D40`): a single-point subset is
a scalar, a multi-point subset a pointer connector addressed relative to the
subset's lower corner (its whole footprint is logged as read), a stream
connector sees the current queue; code values come from `St.vals`. -/
def bindIn (ctx : Ctx) (ρ : Valuation) (s : State) (e : DEdge) (st : St) : Res (Option ConnIn × St) :=
  match s.srcEndpoint e with
  | .code key => .ok ((st.val? key).map .scalar, st)
  | .chunk entry =>
    let vs := st.mem.chunkVals (chunkName ctx entry)
    match vs with
    | [v] => .ok (some (.scalar v), st)
    | _ => .ok (some (.array fun rel =>
        match rel with
        | [k] => if 0 ≤ k then vs[k.toNat]? else none
        | _ => none), st)
  | .data d sub =>
    if ctx.isStream d then do
      let p ← streamPoint ctx ρ st.mem d sub
      let q ← st.queue ctx ρ d p
      pure (some (.stream q), st)
    else do
      let pts ← ctx.pointsOf ρ st.mem d sub
      match pts with
      | [] => .err (.malformed s!"empty input subset of '{d}'")
      | [p] => do
        let (v, st) ← st.read ctx ρ d p
        pure (some (.scalar v), st)
      | lo :: _ => do
        let (_, st) ← st.readMany ctx ρ d pts
        let m := st.mem
        let get : Idx → Option Val := fun rel =>
          if rel.length = lo.length then
            let abs := List.zipWith (· + ·) lo rel
            if pts.contains abs then
              match ctx.loc ρ m d abs with
              | .ok l => some (m.read l)
              | .err _ => none
            else none
          else none
        pure (some (.array get), st)
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

/-- Deliver a scalar output: a plain/WCR write of a single element, a push
to a stream, or a code value for a code/entry consumer. -/
def applyScalar (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (v : Val) : Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet m =>
      match s.dstEndpoint e with
      | .code key => pure (st.setVal key v)
      | .data d sub => do
        if ctx.isStream d then do
          let p ← streamPoint ctx ρ st.mem d sub
          st.push ctx ρ d p [v]
        else do
          let pts ← ctx.pointsOf ρ st.mem d sub
          match pts with
          | [p] => st.store ctx ρ d p m.wcr v
          | [] => .err (.malformed s!"scalar output to an empty subset of '{d}'")
          | _ => .err (.malformed s!"scalar output to several elements of '{d}'")
      | _ => .err (.malformed s!"output connector '{c}' without a valid destination")) st

/-- Deliver an element write through a pointer output connector, relative
to the lower corner of the destination subset. -/
def applyArrayWrite (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (rel : Idx) (v : Val) :
    Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet m =>
      match s.dstEndpoint e with
      | .data d sub =>
        if ctx.isStream d then .err (.malformed s!"array write to stream '{d}'")
        else do
          let pts ← ctx.pointsOf ρ st.mem d sub
          let lo ← Res.ofOption pts.head? (.malformed s!"array write to an empty subset of '{d}'")
          if rel.length = lo.length then st.store ctx ρ d (List.zipWith (· + ·) lo rel) m.wcr v
          else .err (.malformed s!"array write index rank mismatch on '{d}'")
      | _ => .err (.malformed s!"array write through connector '{c}' to a non-container")) st

/-- Deliver pushes through a stream output connector. -/
def applyPush (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (vs : List Val) : Res St :=
  (s.outEdgesOn n c).foldlM (fun st e =>
    match s.dstEndpoint e with
    | .data d sub =>
      if ctx.isStream d then do
        let p ← streamPoint ctx ρ st.mem d sub
        st.push ctx ρ d p vs
      else .err (.malformed s!"push to non-stream '{d}'")
    | _ => .err (.malformed s!"push through connector '{c}' to a non-container")) st

/-- Pop `k` elements from the stream behind an input connector (`Decisions.md#D14`). -/
def applyPop (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (c : Conn) (k : Nat) : Res St :=
  match s.inEdgeOn n c with
  | none => .err (.malformed s!"pop through unknown connector '{c}'")
  | some e =>
    match s.srcEndpoint e with
    | .data d sub =>
      if ctx.isStream d then do
        let p ← streamPoint ctx ρ st.mem d sub
        let (_, st) ← st.popN ctx ρ d p k
        pure st
      else .err (.malformed s!"pop from non-stream '{d}'")
    | _ => .err (.malformed s!"pop through connector '{c}' from a non-container")

/-- Apply the effects of one code-node execution: pops, then scalar outputs,
element writes and pushes. -/
def applyOut (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) (out : TaskletOut) : Res St := do
  let st ← out.pops.foldlM (fun st (c, k) => applyPop ctx ρ s n st c k) st
  let st ← out.scalars.foldlM (fun st (c, v) => applyScalar ctx ρ s n st c v) st
  let st ← out.arrayWrites.foldlM (fun st (c, rel, v) => applyArrayWrite ctx ρ s n st c rel v) st
  out.pushes.foldlM (fun st (c, vs) => applyPush ctx ρ s n st c vs) st

/-- Execute a code node through an uninterpreted body `run`. -/
def execCode (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (undefinedErr : ErrReason) : Res St := do
  let (ins, st) ← bindInputs ctx ρ s n st
  match run ins with
  | none => .err undefinedErr
  | some out => applyOut ctx ρ s n st out

/-! ## Reduce -/

/-- Coordinates of a point that are not reduced. -/
def reduceKeep (axes : List Nat) (p : Idx) : Idx :=
  p.zipIdx.filterMap fun (x, k) => if axes.contains k then none else some x

/-- Direct semantics of the `Reduce` library node (`Decisions.md#D37`,
mirroring `ExpandReducePure`, `dace/libraries/standard/nodes/reduce.py:35-166`):
input points are grouped by their non-reduced coordinates, in the order of
the output points; with an `identity` the fold starts there and the result
is written (plainly, or through the out-memlet's WCR); without one the fold
of the group is contributed to the existing output through the reduction
operator (the pure expansion has no initialization state). -/
def execReduce (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (props : ReduceProps) (st : St) : Res St :=
  let ins := (s.inEdges n.id).filter fun e => e.data matches .memlet _
  let outs := (s.outEdges n.id).filter fun e => e.data matches .memlet _
  match ins, outs with
  | [ie], [oe] =>
    match s.srcEndpoint ie, s.dstEndpoint oe, oe.data.memlet? with
    | .data sd ssub, .data dd dsub, some om =>
      if ctx.isStream sd || ctx.isStream dd then .err (.malformed "reduce over a stream")
      else do
        let spts ← ctx.pointsOf ρ st.mem sd ssub
        let dpts ← ctx.pointsOf ρ st.mem dd dsub
        let ndim := (spts.head?.map List.length).getD 0
        let axes := props.axes.getD (List.range ndim)
        let groups := (spts.map (reduceKeep axes)).eraseDups
        if groups.length ≠ dpts.length then .err (.malformed "reduce output shape mismatch")
        else do
          let (vals, st) ← st.readMany ctx ρ sd spts
          let pairs := spts.zip vals
          (groups.zip dpts).foldlM (fun st (g, q) =>
            let gv := (pairs.filter fun (p, _) => reduceKeep axes p = g).map (·.2)
            match props.identity with
            | some ide => do
              let i ← ctx.evalR ρ st.mem ide
              st.store ctx ρ dd q om.wcr (gv.foldl (ctx.I.wcr props.wcr) i)
            | none =>
              match gv with
              | [] => pure st
              | v :: rest => st.wcr ctx ρ dd q (om.wcr.getD props.wcr) (rest.foldl (ctx.I.wcr props.wcr) v)) st
    | _, _, _ => .err (.malformed "reduce node not between containers")
  | _, _ => .err (.malformed "reduce node must have exactly one input and one output")

/-! ## Views and references -/

/-- Alias binding of container `d` to the lower corner of `sub` in `base`. -/
def aliasTo (ctx : Ctx) (ρ : Valuation) (st : St) (d base : Name) (sub : Option Subset) : Res St := do
  let pts ← ctx.pointsOf ρ st.mem base sub
  let lo ← Res.ofOption pts.head? (.malformed s!"empty subset of '{base}' viewed by '{d}'")
  let l ← ctx.loc ρ st.mem base lo
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

/-- Prologue of a sequential execution: bind every view among the given
(direct) nodes, in node order. -/
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
  | _ => .err (.malformed s!"reference '{d}' set from a non-container source")

/-! ## Access and exit nodes -/

/-- Execute the copies into a node: every memlet in-edge whose source is a
container (or the chunk), except view definitions.  For a reference access
node an edge into connector `set` binds the reference instead. -/
def execCopiesInto (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (st : St) : Res St :=
  (s.inEdges n.id).foldlM (fun st e =>
    match e.data with
    | .dep => pure st
    | .memlet _ =>
      if ctx.isViewDef s e then pure st
      else
        match n.accessData? with
        | some d =>
          if e.dstConn = some "set" && ctx.isReference d then refSet ctx ρ s e d st
          else
            match s.srcEndpoint e with
            | .data _ _ | .chunk _ => execCopy ctx ρ s e st
            | _ => pure st
        | none =>
          match s.srcEndpoint e with
          | .data _ _ | .chunk _ => execCopy ctx ρ s e st
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
  | .tasklet t => execCode ctx ρ s n st (ctx.I.tasklet t) (.taskletUndefined t.label)
  | .library l =>
    match l.reduce with
    | some props => execReduce ctx ρ s n props st
    | none => execCode ctx ρ s n st (ctx.I.library l.classpath) (.libraryUnsupported l.classpath)
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
def iterStart (pre : St) : St := { mem := pre.mem, log := Log.empty, vals := pre.vals }

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

/-- Quiescence condition of a consume scope before a task, evaluated with
the current queue content (`stream.h:347,370`; default: non-empty). -/
def consumeCond (ctx : Ctx) (ρ : Valuation) (c : ConsumeNode) (m : Mem) (l : Loc) (q : List Val) : Res Bool :=
  match c.condition with
  | none => .ok (!q.isEmpty)
  | some e => ctx.evalBoolR ρ (m.setQueue l q) e

/-- Initial state of a consume task: the pre-scope memory with the queue
advanced past the chunk and the chunk written (`Decisions.md#D38`). -/
def taskStart (ctx : Ctx) (entry : NodeId) (c : ConsumeNode) (l : Loc) (pre : St) (q : List Val) : St :=
  { mem := (pre.mem.setQueue l (q.drop c.chunksize)).writeChunk (chunkName ctx entry) (q.take c.chunksize),
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

/-- Bind the connectors of a nested SDFG node (`Decisions.md#D15`, `#D34`):
containers are aliased with offset `0`, the consume chunk likewise, scalar
code values are copied into the inner scalar.  Returns the aliased inner
names. -/
def bindConnectors (ctx ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (st : St) : Res (St × List Name) :=
  (nestedConns n).foldlM (fun (st, aliased) c =>
    let ep : Endpoint :=
      match s.inEdgeOn n c with
      | some e => s.srcEndpoint e
      | none =>
        match s.outEdgesOn n c with
        | e :: _ => s.dstEndpoint e
        | [] => .invalid
    match ep with
    | .data base _ =>
      let inner := ctx'.cname c
      pure ({ st with mem := st.mem.setAlias inner { base := ctx.cname base } }, inner :: aliased)
    | .chunk entry =>
      let inner := ctx'.cname c
      pure ({ st with mem := st.mem.setAlias inner { base := chunkName ctx entry } }, inner :: aliased)
    | .code key =>
      match st.val? key with
      | some v => do
        let l ← ctx'.loc ρ' st.mem c []
        pure ({ st with mem := st.mem.write l v }, aliased)
      | none => pure (st, aliased)
    | .invalid => .err (.malformed s!"nested SDFG connector '{c}' without a valid edge")) (st, [])

/-- Set-up of a nested invocation: inner context, valuation, connectors. -/
def nestedSetup (ctx : Ctx) (ρ : Valuation) (s : State) (n : DNode) (inner : SDFG) (mapping : List (Sym × Expr))
    (st : St) : Res (Ctx × Valuation × St × List Name) := do
  let ctx' := ctx.enter inner n.id
  let ρ' ← nestedValuation ctx ρ st.mem mapping
  let (st, aliased) ← bindConnectors ctx ctx' ρ' s n st
  pure (ctx', ρ', st, aliased)

/-- Names whose log entries stay inside a nested invocation: its connectors
and its transients (prefixed). -/
def nestedDropped (ctx' : Ctx) (n : DNode) : List Name :=
  (nestedConns n).map ctx'.cname ++
    ctx'.g.arrays.filterMap fun (a, d) => if d.transient then some (ctx'.cname a) else none

/-- Tear-down of a nested invocation: clear the connector aliases, keep the
memory, contribute the inner log without private names, and pass scalar
outputs on to code consumers. -/
def nestedFinish (ctx' : Ctx) (ρ' : Valuation) (s : State) (n : DNode) (aliased : List Name) (pre : St)
    (m₂ : Mem) (g₂ : Log) : Res St :=
  let st : St := { mem := aliased.foldl Mem.clearAlias m₂,
                   log := pre.log.seq (g₂.dropNames (nestedDropped ctx' n)), vals := pre.vals }
  (s.outEdges n.id).foldlM (fun st e =>
    match e.data, e.srcConn with
    | .memlet _, some c =>
      match s.dstEndpoint e with
      | .code key => do
        let l ← ctx'.loc ρ' m₂ c []
        pure (st.setVal key (m₂.read l))
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

end Dace
