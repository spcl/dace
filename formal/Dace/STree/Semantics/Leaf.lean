-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics.Basic

/-!
# Schedule tree semantics: leaf-level functions

Everything a single tree node does to the state is a plain function here;
only the parts that recurse into the children (scopes) are relations
(`Dace.STree.Semantics.Exec`).  The container-level primitives are those of
the SDFG semantics (`Dace.SDFG.Semantics.Leaf`: `bindInData`, `storeScalar`,
`copyData`, `reduceData`, `aliasTo`, `scopeFinish`, `consumeFinish`, …), so
that a tree node and the SDFG node it came from execute the same code.

Conventions (`Decisions.md#D50`, `#D53`, `#D55`, `#D57`):

* a code node's `in_memlets` bind its connectors like leaf in-edges
  (`#D40`); `out_memlets` receive scalar outputs, element writes and pushes;
  a dynamic WCR output (fetch-and-add) is `unsupported` (`#D47`, `#D59`);
* a `CopyNode` reads `memlet.data[memlet.subset]` and writes
  `target[other_subset]` (full range when absent, `#D9`), elementwise;
* inside a consume scope, the consumed stream read through a memlet is the
  chunk container `consumeChunkName` (whole range, `TCtx.inData`); pushes to
  it are deferred as in `#D38`;
* `AssignNode`s are sequential symbol assignments (`#D49`).
-/

set_option autoImplicit false

namespace Dace

/-! ## Memlet endpoints -/

/-- Bind an input memlet to a connector value (`bindInData` on `TCtx.inData`:
the chunk container for the consumed stream). -/
def bindMemletIn (tc : TCtx) (ρ : Valuation) (m : Memlet) (st : St) : Res (ConnIn × St) :=
  let (d, sub) := tc.inData m
  bindInData tc.ctx ρ d sub m.allowOob st

/-- Bind all input connectors of a code node, in memlet order. -/
def bindInputsT (tc : TCtx) (ρ : Valuation) (ins : MemletMap) (st : St) : Res ((Conn → Option ConnIn) × St) := do
  let (bs, st) ← ins.pairs.foldlM (fun (acc, st) (c, m) => do
      let (ci, st) ← bindMemletIn tc ρ m st
      pure (acc ++ [(c, ci)], st)) (([] : List (Conn × ConnIn)), st)
  pure (fun c => (bs.find? (·.1 = c)).map (·.2), st)

/-- Apply the effects of one code-node execution through its memlets: pops
on the input memlets, then scalar outputs, element writes and pushes on the
output memlets of each connector. -/
def applyOutT (tc : TCtx) (ρ : Valuation) (ins outs : MemletMap) (st : St) (out : TaskletOut) : Res St := do
  let st ← out.pops.foldlM (fun st (c, k) =>
    match ins.on c with
    | [] => .err (.malformed s!"pop through unknown connector '{c}'")
    | ms => ms.foldlM (fun st m => let (d, sub) := tc.inData m; popFrom tc.ctx ρ st d sub k) st) st
  let st ← out.scalars.foldlM (fun st (c, v) =>
    (outs.on c).foldlM (fun st m => storeScalar tc.ctx ρ st m.data m.dataSubset m.wcr m.allowOob v) st) st
  let st ← out.arrayWrites.foldlM (fun st (c, rel, v) =>
    (outs.on c).foldlM (fun st m =>
      storeArrayWrite tc.ctx ρ st m.data m.dataSubset m.wcr m.allowOob rel v) st) st
  out.pushes.foldlM (fun st (c, vs) =>
    (outs.on c).foldlM (fun st m => pushTo tc.ctx ρ st m.data m.dataSubset vs) st) st

/-- Does the node have a dynamic WCR out-memlet (a fetch-and-add, `Decisions.md#D47`)? -/
def MemletMap.hasDynamicWcr (mm : MemletMap) : Bool := mm.memlets.any fun m => m.dynamic && m.hasWcr

/-- Execute a code node through an uninterpreted body `run`. -/
def execCodeT (tc : TCtx) (ρ : Valuation) (label : String) (ins outs : MemletMap) (st : St)
    (run : (Conn → Option ConnIn) → Option TaskletOut) (undefinedErr : ErrReason) : Res St :=
  if outs.hasDynamicWcr then
    .err (.unsupported s!"dynamic WCR output of '{label}' (fetch-and-add, Decisions.md#D47)")
  else do
    let (b, st) ← bindInputsT tc ρ ins st
    match run b with
    | none => .err undefinedErr
    | some out => applyOutT tc ρ ins outs st out

/-- The `Reduce` library node between its single input and output memlets
(`reduceData`, `Decisions.md#D37`). -/
def execReduceT (tc : TCtx) (ρ : Valuation) (props : ReduceProps) (ins outs : MemletMap) (st : St) : Res St :=
  match ins.memlets, outs.memlets with
  | [im], [om] =>
    let (d, sub) := tc.inData im
    reduceData tc.ctx ρ st props d sub om.data om.dataSubset om.wcr im.allowOob om.allowOob
  | _, _ => .err (.malformed "reduce node must have exactly one input and one output")

/-- A `LibraryCall`: `Reduce` directly, everything else through `Interp.library`. -/
def execLibraryT (tc : TCtx) (ρ : Valuation) (l : LibNode) (ins outs : MemletMap) (st : St) : Res St :=
  match l.reduce with
  | some props => execReduceT tc ρ props ins outs st
  | none => execCodeT tc ρ l.label ins outs st (tc.ctx.I.library l.classpath) (.libraryUndefined l.classpath)

/-- A `CopyNode` (`Decisions.md#D53`): `target[other_subset] = data[subset]`. -/
def execCopyT (tc : TCtx) (ρ : Valuation) (target : Name) (m : Memlet) (st : St) : Res St :=
  let (d, sub) := tc.inData m
  copyData tc.ctx ρ st d sub target m.otherSubset m.wcr m.allowOob

/-- A `ViewNode`: alias `target` to the lower corner of the viewed subset of
`source` (`Decisions.md#D31`, `#D55`). -/
def execViewT (tc : TCtx) (ρ : Valuation) (target source : Name) (m : Memlet) (st : St) : Res St :=
  if m.data = source then aliasTo tc.ctx ρ st target source m.dataSubset
  else .err (.malformed s!"memlet of view '{target}' is not on its source '{source}'")

/-- A `RefSetNode` from a data source (`Decisions.md#D32`, `#D55`); a code
node source is not modeled. -/
def execRefSetT (tc : TCtx) (ρ : Valuation) (target : Name) (m : Memlet) (src : RefSetSource) (st : St) : Res St :=
  match src with
  | .data _ => aliasTo tc.ctx ρ st target m.data m.dataSubset
  | .codeNode _ cls => .err (.unsupported s!"reference set of '{target}' from a {cls} (Decisions.md#D55)")

/-- A `DynScopeCopyNode` (`Decisions.md#D50`): the `IN_stream` of a consume
scope designates the stream (and is not popped); any other target reads one
element (popping a stream, `#D42`) into the pending bindings.  An absent
value leaves the symbol unbound. -/
def execDynScopeCopy (tc : TCtx) (ρ : Valuation) (p : Pending) (target : Sym) (m : Memlet) (st : St) :
    Res (Pending × St) :=
  if target = "IN_stream" then
    if tc.ctx.isStream m.data then do
      let pt ← streamPoint tc.ctx ρ st.mem m.data m.dataSubset
      pure ({ p with stream := some (m.data, pt) }, st)
    else .err (.malformed s!"IN_stream of non-stream '{m.data}'")
  else do
    let (d, sub) := tc.inData m
    let (v?, st) ← readScalarData tc.ctx ρ d sub m.allowOob st
    match v? with
    | some v => pure ({ p with syms := p.syms ++ [(target, v)] }, st)
    | none => pure (p, st)

/-! ## Scopes -/

/-- Set-up of a map scope: the pending dynamic inputs are bound (`#D12`),
then the iteration space is evaluated. -/
def mapSetupT (tc : TCtx) (sc : SibCtx) (mp : MapNode) (cst : CSt) : Res (Valuation × List Idx) :=
  if sc.pending.stream.isSome then .err (.malformed "IN_stream given to a map scope")
  else do
    let ρ' := cst.ρ.setAll sc.pending.syms
    let pts ← tc.ctx.rangePoints ρ' cst.mem mp.range
    pure (ρ', pts)

/-- Initial state of one iteration/task: the pre-scope memory, an empty log. -/
def iterStartT (priv : List Name) (pre : CSt) (ρ : Valuation) : CSt :=
  { ρ, mem := pre.mem.allocAll priv, log := Log.empty }

/-- The log contributed by one map iteration: its effects without the
scope's private transients; control flow may not leave the iteration. -/
def iterLog (priv : List Name) : Res (TCtl × CSt) → Res Log
  | .err e => .err e
  | .ok (.normal, cst) => .ok (cst.log.dropNames priv)
  | .ok (_, _) => .err (.malformed "control flow leaving a dataflow scope")

/-- Merge the iteration logs onto the pre-scope state (`Decisions.md#D3`). -/
def scopeFinishT (tc : TCtx) (pre : CSt) (logs : List Log) : Res CSt :=
  (scopeFinish tc.ctx pre.st logs).map pre.withSt

/-- Set-up of a consume scope: the dynamic inputs, the consumed stream
(from the `IN_stream` input) as a resolved queue location, the number of
processing elements. -/
def consumeSetupT (tc : TCtx) (sc : SibCtx) (c : ConsumeNode) (cst : CSt) : Res (Valuation × Name × Loc × Int) := do
  let ρ' := cst.ρ.setAll sc.pending.syms
  match sc.pending.stream with
  | none => .err (.malformed s!"consume scope '{c.label}' without an IN_stream input")
  | some (d, p) => do
    let l ← tc.ctx.loc ρ' cst.mem d p
    let np ← tc.ctx.evalIntR ρ' cst.mem c.numPes
    pure (ρ', d, cst.mem.resolve l, np)

/-- Initial state of a consume task (`Decisions.md#D38`): the queue advanced
past the chunk, the chunk written to its container (`taskStart`). -/
def taskStartT (tc : TCtx) (d : Name) (c : ConsumeNode) (l : Loc) (priv : List Name) (pre : CSt) (ρ : Valuation)
    (q : List Val) : CSt :=
  { ρ, mem := ((pre.mem.allocAll priv).setQueue l (q.drop c.chunksize)).writeChunk (tc.chunkMemName d) (q.take c.chunksize),
    log := Log.empty }

/-- The log and deferred pushes contributed by one consume task. -/
def taskOutcome (priv : List Name) (l : Loc) : Res (TCtl × CSt) → Res (Log × List Val)
  | .err e => .err e
  | .ok (.normal, cst) => .ok (taskLog cst.st priv l, taskPushed cst.st priv l)
  | .ok (_, _) => .err (.malformed "control flow leaving a dataflow scope")

/-- Finish a consume scope (`consumeFinish`). -/
def consumeFinishT (tc : TCtx) (pre : CSt) (l : Loc) (logs : List Log) (rest pushed : List Val) : Res CSt :=
  (consumeFinish tc.ctx pre.st l logs rest pushed).map pre.withSt

/-! ## Leaf step -/

/-- Outcome of one node: the sibling context for the next sibling, the
control signal and the state. -/
abbrev NodeOut := SibCtx × TCtl × CSt

/-- A plain leaf: closes any open `if` chain, proceeds normally. -/
def leafOut (r : Res CSt) : Res NodeOut := r.map fun cst => ({}, .normal, cst)

/-- The nodes whose execution is a function of the state (everything but
scopes).  `none` for scopes. -/
def stepLeaf (tc : TCtx) (sc : SibCtx) (n : SNode) (cst : CSt) : Option (Res NodeOut) :=
  match n with
  | .stateLabel _ => some (.ok ({}, .normal, cst))
  | .goto none => some (.ok ({}, .ret, cst))
  | .goto (some l) => some (.ok ({}, .goto l, cst))
  | .breakNode => some (.ok ({}, .brk, cst))
  | .continueNode => some (.ok ({}, .cont, cst))
  | .assign name v => some (leafOut (applyAssigns tc.ctx cst [(name, v.expr)]))
  | .taskletNode t ins outs =>
    some (leafOut (cst.lift (execCodeT tc cst.ρ t.label ins outs cst.st (tc.ctx.I.tasklet t cst.ρ)
      (.taskletUndefined t.label))))
  | .libraryCall l _ ins outs => some (leafOut (cst.lift (execLibraryT tc cst.ρ l ins outs cst.st)))
  | .copyNode target m => some (leafOut (cst.lift (execCopyT tc cst.ρ target m cst.st)))
  | .dynScopeCopy target m =>
    some ((execDynScopeCopy tc cst.ρ sc.pending target m cst.st).map fun (p, st) =>
      ({ pending := p }, .normal, cst.withSt st))
  | .viewNode target source m _ _ => some (leafOut (cst.lift (execViewT tc cst.ρ target source m cst.st)))
  | .refSetNode target m src _ => some (leafOut (cst.lift (execRefSetT tc cst.ρ target m src cst.st)))
  | .namedRegion .. | .gblock .. | .loopScope .. | .forScope .. | .whileScope .. | .doWhileScope ..
  | .ifScope .. | .stateIfScope .. | .elifScope .. | .elseScope .. | .mapScope .. | .consumeScope .. => none

/-- Outcome of a scope that closes the `if` chain. -/
def scopeOut (r : Res (TCtl × CSt)) : Res NodeOut := r.map fun (ctl, cst) => ({}, ctl, cst)

/-- Outcome of a taken `if`/`elif` branch: the chain stays open as taken. -/
def chainOut (r : Res (TCtl × CSt)) : Res NodeOut := r.map fun (ctl, cst) => ({ chain := some true }, ctl, cst)

/-- The outcome of a dataflow scope: pending inputs consumed, chain closed. -/
def dataflowOut (r : Res CSt) : Res NodeOut := r.map fun cst => ({}, .normal, cst)

/-- Outcome of the whole program (same observable as `SDFG.Exec`). -/
def programOutcome : Res (TCtl × CSt) → Res Mem
  | .err e => .err e
  | .ok (.normal, cst) => .ok cst.mem
  | .ok (.ret, cst) => .ok cst.mem
  | .ok (.brk, _) => .err (.malformed "break/continue escaping the program")
  | .ok (.cont, _) => .err (.malformed "break/continue escaping the program")
  | .ok (.goto l, _) => .err (.malformed s!"goto '{l}' escaping the schedule tree")

/-- The memory names defined at the start of a run of `t`: its arguments
and `External` transients (`Decisions.md#D68`). -/
def STree.keepAtStart (t : STree) (n : Name) : Bool := t.containers.any fun (a, d) => a == n && d.definedAtStart

/-- The state a run of `t` starts from: the constants in the valuation,
everything but the arguments and `External` transients uninitialized. -/
def rootStart (t : STree) (ρ₀ : Valuation) (m₀ : Mem) : CSt :=
  { ρ := t.toSDFG.withConstants ρ₀, mem := m₀.restrict t.keepAtStart }

end Dace
