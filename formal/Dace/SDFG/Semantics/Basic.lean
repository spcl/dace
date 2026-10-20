-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Merge
import Dace.SDFG.WF

/-!
# SDFG semantics: basic definitions

Shared infrastructure of the operational semantics (`Dace.SDFG.Semantics`):

* outcomes (`ErrReason`, `Res`) and control signals (`Ctl`);
* the execution context `Ctx` (interpretation, current SDFG, container-name
  prefix of the current nested invocation);
* element locations (`Ctx.loc`), expression evaluation that may read data
  (`Ctx.eval`), subset enumeration (`Ctx.pointsOf`);
* the sequential execution state `St` (local memory view, access log, scalar
  values on code→code edges) and its logged primitive operations;
* the structural **endpoints** of leaf memlet edges (`State.srcEndpoint`,
  `State.dstEndpoint`, `Decisions.md#D29`), view-edge resolution
  (`State.viewEdge?`, mirroring `dace/sdfg/utils.py:1141-1226`) and the
  private transients of a scope (`Decisions.md#D39`).

Nothing here is recursive over the SDFG syntax except the mention count
used by `Ctx.privateTransients`.
-/

set_option autoImplicit false

namespace Dace

/-! ## Outcomes -/

/-- Reasons for an erroneous outcome.  Three classes (`Decisions.md#D59`):

* **undefined behavior** (`race`, `stuckPop`, `undefinedExpr`,
  `undefinedSubset`, `unsetRef`, `outOfBounds`, `taskletUndefined`): the program is valid
  but this execution has no defined result;
* `malformed`: situations that well-formedness (`SDFG.WF`, `STree.check`)
  excludes; it keeps the semantics total over the syntax;
* `unsupported`: valid DaCe whose meaning is not modeled (fetch-and-add,
  pointer-valued code outputs, library nodes without a model, reinterpreting
  views of a different element size).  It is not undefined behavior:
  equivalence makes no claim about a program that may end in it. -/
inductive ErrReason where
  /-- Conflicting parallel accesses found by `merge`. -/
  | race (e : MergeError)
  /-- Pop from an empty (or too short) queue: the generated code would block. -/
  | stuckPop (l : Loc)
  /-- A symbolic expression is undefined under the current valuation. -/
  | undefinedExpr (e : Expr)
  /-- A subset cannot be enumerated under the current valuation. -/
  | undefinedSubset (sub : Subset)
  /-- Access to a reference container before it was `set` (`Decisions.md#D11`). -/
  | unsetRef (n : Name)
  /-- Element access outside the shape of the accessed container
  (`Decisions.md#D65`). -/
  | outOfBounds (n : Name) (idx : Idx)
  /-- The tasklet interpretation is undefined for these inputs. -/
  | taskletUndefined (label : String)
  /-- The library-node interpretation (`Interp.library`) is undefined for
  this node and its inputs.  Like `taskletUndefined` this depends on the
  interpretation, which equivalence quantifies over, hence undefined
  behavior rather than `unsupported` (`Decisions.md#D59`). -/
  | libraryUndefined (classpath : String)
  /-- Valid DaCe that the semantics does not model (`Decisions.md#D59`). -/
  | unsupported (msg : String)
  | malformed (msg : String)
  deriving Inhabited

/-- Human-readable rendering (for tools and test output).  `unsupported`
messages start with `unsupported` (`scripts/difftest.py` reports them as
SKIP). -/
def ErrReason.toString : ErrReason → String
  | .race (.unordered a b w) => s!"data race between the unordered nodes {a} and {b}: {repr w}"
  | .race e => s!"data race: {repr e}"
  | .stuckPop l => s!"pop from an empty queue at {repr l}"
  | .undefinedExpr e => s!"undefined expression '{e}'"
  | .undefinedSubset sub => s!"undefined subset {repr sub}"
  | .unsetRef n => s!"reference '{n}' accessed before it was set"
  | .outOfBounds n idx => s!"out-of-bounds access {idx} of '{n}'"
  | .taskletUndefined label => s!"tasklet '{label}' is undefined for its inputs"
  | .libraryUndefined cp => s!"library node '{cp}' is undefined for its inputs"
  | .unsupported msg => s!"unsupported {msg}"
  | .malformed msg => s!"malformed SDFG: {msg}"

/-- The class of an error (`Decisions.md#D59`): `"ub"`, `"malformed"`,
`"unsupported"`, or `"uninterpreted"` (undefined behavior because the
interpretation of a tasklet or library node is undefined; for the
executable interpretation this usually means a gap of the executable
subset, so `scripts/difftest.py` reports it as SKIP). -/
def ErrReason.kind : ErrReason → String
  | .unsupported _ => "unsupported"
  | .taskletUndefined _ | .libraryUndefined _ => "uninterpreted"
  | .malformed _ => "malformed"
  | _ => "ub"

/-- The constructor name of an error (printed by `dace-interp` as
`reason`, matched by the `expect_error` entries of `scripts/difftest.py`). -/
def ErrReason.tag : ErrReason → String
  | .race _ => "race"
  | .stuckPop _ => "stuckPop"
  | .undefinedExpr _ => "undefinedExpr"
  | .undefinedSubset _ => "undefinedSubset"
  | .unsetRef _ => "unsetRef"
  | .outOfBounds _ _ => "outOfBounds"
  | .taskletUndefined _ => "taskletUndefined"
  | .libraryUndefined _ => "libraryUndefined"
  | .unsupported _ => "unsupported"
  | .malformed _ => "malformed"

/-- Is the error undefined behavior (anything refines it)?  False exactly for
`malformed` and `unsupported` (`Decisions.md#D59`). -/
def ErrReason.isUB : ErrReason → Bool
  | .unsupported _ | .malformed _ => false
  | _ => true

instance : ToString ErrReason := ⟨ErrReason.toString⟩

/-- Outcome of an execution: a result or an error. -/
inductive Res (α : Type) where
  | ok (a : α)
  | err (e : ErrReason)

namespace Res

def map {α β : Type} (f : α → β) : Res α → Res β
  | ok a => ok (f a)
  | err e => err e

def bind {α β : Type} : Res α → (α → Res β) → Res β
  | ok a, f => f a
  | err e, _ => err e

instance : Monad Res where
  pure := ok
  bind := bind

/-- Lift an `Option`, naming the error for `none`. -/
def ofOption {α : Type} (o : Option α) (e : ErrReason) : Res α :=
  match o with
  | some a => ok a
  | none => err e

/-- Lift the result of `merge`. -/
def ofMerge {α : Type} : Except MergeError α → Res α
  | .ok a => ok a
  | .error e => err (.race e)

def isOk {α : Type} : Res α → Bool
  | ok _ => true
  | err _ => false

@[simp] theorem ok_inj {α : Type} {a b : α} : (ok a : Res α) = ok b ↔ a = b :=
  ⟨fun h => by cases h; rfl, fun h => h ▸ rfl⟩

@[simp] theorem ok_ne_err {α : Type} {a : α} {e : ErrReason} : (ok a : Res α) ≠ err e := by
  intro h; cases h

@[simp] theorem map_ok {α β : Type} (f : α → β) (a : α) : (ok a : Res α).map f = ok (f a) := rfl
@[simp] theorem map_err {α β : Type} (f : α → β) (e : ErrReason) : (err e : Res α).map f = err e := rfl

end Res

/-- Control signal produced by a control-flow block. -/
inductive Ctl where
  | normal | brk | cont | ret
  deriving DecidableEq, Repr, Inhabited

/-! ## Valuations -/

namespace Valuation

/-- Bind one symbol. -/
def set (ρ : Valuation) (x : Sym) (v : Val) : Valuation :=
  fun y => if y = x then some v else ρ y

/-- Bind several symbols (earlier entries win). -/
def setAll (ρ : Valuation) (xs : List (Sym × Val)) : Valuation :=
  fun y => match xs.find? (·.1 = y) with
    | some (_, v) => some v
    | none => ρ y

/-- Bind map parameters to an iteration index. -/
def bindIdx (ρ : Valuation) (params : List Sym) (i : Idx) : Valuation :=
  ρ.setAll ((params.zip i).map fun (p, k) => (p, Val.int k))

end Valuation

/-- Extend a valuation by the scalar constants of an SDFG (`SDFG.constants`). -/
def SDFG.withConstants (g : SDFG) (ρ : Valuation) : Valuation :=
  fun x => match g.constants.find? (·.1 = x) with
    | some (_, _, some v) => some v
    | _ => ρ x

/-! ## Execution context -/

/-- Context of one (possibly nested) SDFG invocation. -/
structure Ctx where
  I : Interp
  g : SDFG
  /-- Container-name prefix of the current nested invocation (`""` at top level). -/
  pfx : String := ""

namespace Ctx

/-- Memory name of a container of the current SDFG. -/
def cname (ctx : Ctx) (n : Name) : Name := ctx.pfx ++ n

def desc? (ctx : Ctx) (n : Name) : Option DataDesc := ctx.g.array? n

/-- Context of a nested SDFG invoked by node `node`. -/
def enter (ctx : Ctx) (inner : SDFG) (node : NodeId) : Ctx :=
  { ctx with g := inner, pfx := ctx.pfx ++ inner.name ++ "#" ++ toString node ++ "/" }

def isView (ctx : Ctx) (n : Name) : Bool := (ctx.desc? n).any (·.isView)
def isReference (ctx : Ctx) (n : Name) : Bool := (ctx.desc? n).any (·.isReference)
def isStream (ctx : Ctx) (n : Name) : Bool := (ctx.desc? n).any (·.isStream)

/-- Linear element offset: scalars live at their start offset whatever the
index (`[]` or `[0]`); everything else goes through the strides
(`DataDesc.linearOffset`). -/
def elemOffset (ctx : Ctx) (ρ : Valuation) (d : DataDesc) (idx : Idx) : Option Int :=
  if d.isScalar then d.startOffset.evalInt ctx.I.expr ρ else d.linearOffset ctx.I.expr ρ idx

/-- Address of element `idx` of container `n`, **without** a bounds check:
the location a view or reference is bound to (the lower corner of a subset,
which may lie outside the container when the subset is empty,
`Decisions.md#D65`).  Fails with `unsetRef` for a reference that is not bound
(`Decisions.md#D11`). -/
def addr (ctx : Ctx) (ρ : Valuation) (m : Mem) (n : Name) (idx : Idx) : Res Loc :=
  match ctx.desc? n with
  | none => .err (.malformed s!"unknown container '{n}'")
  | some d =>
    if d.isReference && !m.isAliased (ctx.cname n) then .err (.unsetRef n)
    else match ctx.elemOffset ρ d idx with
      | some o => .ok (Loc.elem (ctx.cname n) o)
      | none => .err (.malformed s!"undefined element offset {idx} in '{n}'")

/-- Bounds check of an element access (`Decisions.md#D65`): every coordinate
of `idx` lies in `0 ≤ idx_k < shape_k` of the accessed container's own
descriptor (a view, reference or nested connector is checked against its
own shape).  Skipped for accesses through an `allow_oob` memlet (`oob`), for
scalars (codegen ignores their index) and for streams (unbounded queues). -/
def checkBounds (ctx : Ctx) (ρ : Valuation) (n : Name) (idx : Idx) (oob : Bool) : Res Unit :=
  match ctx.desc? n with
  | none => .err (.malformed s!"unknown container '{n}'")
  | some d =>
    if oob || d.isScalar || d.isStream then .ok ()
    else do
      let shape ← d.shape.mapM fun s => Res.ofOption (s.evalInt ctx.I.expr ρ) (.undefinedExpr s)
      if shape.length = idx.length && (idx.zip shape).all (fun (i, s) => decide (0 ≤ i ∧ i < s)) then .ok ()
      else .err (.outOfBounds n idx)

/-- Location of element `idx` of container `n` accessed by a data access
(read, write, WCR, queue): its address (`addr`) after the bounds check
(`checkBounds`, `Decisions.md#D65`); `oob` is the `allow_oob` flag of the
memlet the access goes through. -/
def loc (ctx : Ctx) (ρ : Valuation) (m : Mem) (n : Name) (idx : Idx) (oob : Bool := false) : Res Loc := do
  let l ← ctx.addr ρ m n idx
  ctx.checkBounds ρ n idx oob
  pure l

/-- Expression interpretation reading data from `m` (interstate conditions,
assignments and dynamic ranges may read `A[i]`).  Such reads are not logged
(`Decisions.md#D33`). -/
def exprInterp (ctx : Ctx) (ρ : Valuation) (m : Mem) : ExprInterp :=
  { ctx.I.expr with
    readData := fun n idx =>
      match ctx.loc ρ m n idx with
      | .ok l => some (m.read l)
      | .err _ => none }

/-- Valuation extended by scalar containers named bare in expressions
(`Decisions.md#D33`): a symbol shadows a container of the same name. -/
def val (ctx : Ctx) (ρ : Valuation) (m : Mem) : Valuation :=
  fun x => match ρ x with
    | some v => some v
    | none =>
      match ctx.desc? x with
      | some d =>
        if d.isScalar then
          match ctx.loc ρ m x [] with
          | .ok l => some (m.read l)
          | .err _ => none
        else none
      | none => none

def eval (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Option Val :=
  e.eval (ctx.exprInterp ρ m) (ctx.val ρ m)

def evalBool (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Option Bool :=
  e.evalBool (ctx.exprInterp ρ m) (ctx.val ρ m)

def evalInt (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Option Int :=
  e.evalInt (ctx.exprInterp ρ m) (ctx.val ρ m)

/-- Evaluate to a value or fail with `undefinedExpr`. -/
def evalR (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Res Val :=
  Res.ofOption (ctx.eval ρ m e) (.undefinedExpr e)

def evalBoolR (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Res Bool :=
  Res.ofOption (ctx.evalBool ρ m e) (.undefinedExpr e)

def evalIntR (ctx : Ctx) (ρ : Valuation) (m : Mem) (e : Expr) : Res Int :=
  Res.ofOption (ctx.evalInt ρ m e) (.undefinedExpr e)

/-- Points of a subset of container `n` in lexicographic order
(`Decisions.md#D10`); a missing subset is the full range (`Decisions.md#D9`). -/
def pointsOf (ctx : Ctx) (ρ : Valuation) (m : Mem) (n : Name) (sub : Option Subset) : Res (List Idx) :=
  match ctx.desc? n with
  | none => .err (.malformed s!"unknown container '{n}'")
  | some d =>
    let sub := sub.getD (.range d.fullRange)
    Res.ofOption (sub.points (ctx.exprInterp ρ m) (ctx.val ρ m)) (.undefinedSubset sub)

/-- Points of a map range under the current valuation. -/
def rangePoints (ctx : Ctx) (ρ : Valuation) (m : Mem) (r : Range) : Res (List Idx) :=
  Res.ofOption (r.points (ctx.exprInterp ρ m) (ctx.val ρ m)) (.undefinedSubset (.range r))

end Ctx

/-! ## Sequential execution state -/

/-- State of one sequential execution (a state component, a map iteration, a
consume task, a nested invocation).  `mem` is the *local view*: the initial
memory plus this execution's own effects; `log` records the effects for the
parallel merge; `vals` holds scalar values flowing along code→code edges,
keyed by the producing `(node, out-connector)` (`Decisions.md#D29`). -/
structure St where
  mem : Mem
  log : Log := Log.empty
  vals : List ((NodeId × Conn) × Val) := []

namespace St

/-- Compose a new log fragment sequentially. -/
def logged (st : St) (g : Log) : St := { st with log := st.log.seq g }

def val? (st : St) (key : NodeId × Conn) : Option Val :=
  (st.vals.find? (·.1 = key)).map (·.2)

def setVal (st : St) (key : NodeId × Conn) (v : Val) : St :=
  { st with vals := (key, v) :: st.vals }

/-- Read one element, logging the (resolved) location.  `oob` is the
`allow_oob` flag of the memlet the read goes through (`Decisions.md#D65`). -/
def read (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (oob : Bool) : Res (Val × St) := do
  let l ← ctx.loc ρ st.mem n idx oob
  pure (st.mem.read l, st.logged { reads := [st.mem.resolve l] })

/-- Read several elements in order. -/
def readMany (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    Res (List Val × St) :=
  pts.foldlM (fun (acc, st) p => do
      let (v, st) ← st.read ctx ρ n p oob
      pure (acc ++ [v], st)) ([], st)

/-- Plain write. -/
def write (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) : Res St := do
  let l ← ctx.loc ρ st.mem n idx oob
  pure { st with mem := st.mem.write l v, log := st.log.seq { writes := [(st.mem.resolve l, v)] } }

/-- WCR contribution: the local view is updated immediately with the
interpretation's operator; the log records the contribution. -/
def wcr (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    Res St := do
  let l ← ctx.loc ρ st.mem n idx oob
  let old := st.mem.read l
  pure { st with mem := st.mem.write l (ctx.I.wcr op old v),
                 log := st.log.seq { wcrs := [(st.mem.resolve l, op, v)] } }

/-- Plain write or WCR contribution according to the memlet. -/
def store (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (oob : Bool) (wcrOp : Option WcrOp)
    (v : Val) : Res St :=
  match wcrOp with
  | none => st.write ctx ρ n idx oob v
  | some op => st.wcr ctx ρ n idx oob op v

/-- Current content of the queue at element `idx` of stream `n`. -/
def queue (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) : Res (List Val) := do
  let l ← ctx.loc ρ st.mem n idx
  pure (st.mem.queues (st.mem.resolve l))

/-- Push values in order (`Decisions.md#D5`, `#D25`). -/
def push (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (vs : List Val) : Res St := do
  let l ← ctx.loc ρ st.mem n idx
  if vs.isEmpty then pure st
  else pure { st with mem := st.mem.pushMany l vs,
                      log := st.log.seq { queues := [(st.mem.resolve l, .pushOnly vs)] } }

/-- Pop exactly `k` values; `stuckPop` if fewer are available. -/
def popN (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (k : Nat) : Res (List Val × St) := do
  let l ← ctx.loc ρ st.mem n idx
  let r := st.mem.resolve l
  let q := st.mem.queues r
  if k = 0 then pure ([], st)
  else if q.length < k then .err (.stuckPop r)
  else
    let rest := q.drop k
    pure (q.take k,
      { st with mem := { st.mem with queues := Function.update st.mem.queues r rest },
                log := st.log.seq { queues := [(r, .exclusive rest)] } })

/-- Bind an alias (view definition or reference `set`). -/
def bindAlias (st : St) (n : Name) (b : AliasBinding) : St :=
  { st with mem := st.mem.setAlias n b, log := st.log.seq { aliasSets := [(n, b)] } }

end St

/-- State threaded through control flow: valuation, memory and the log of
the current (sequential) SDFG execution. -/
structure CSt where
  ρ : Valuation
  mem : Mem
  log : Log := Log.empty

/-! ## Memlet-tree walks and endpoints -/

namespace State

def isConsumeEntry (s : State) (id : NodeId) : Bool :=
  match s.node? id with
  | some n => (n.kind matches .consumeEntry _)
  | none => false

/-- Outermost edge reached from `e` through exit nodes only. -/
def dstRootEdge (s : State) (e : DEdge) : DEdge := s.walkNext true (s.edges.length + 1) e

/-- The consume entry whose `OUT_stream` connector the path into `e`
crosses, if any (the data then is the consumed chunk container,
`Decisions.md#D38`). -/
def chunkSource (s : State) : Nat → DEdge → Option NodeId
  | 0, _ => none
  | fuel + 1, e =>
    if s.isScopeNode e.src && connPrefix e.srcConn "OUT_" then
      if s.isConsumeEntry e.src && e.srcConn = some "OUT_stream" then some e.src
      else match s.prevEdges e with
        | f :: _ => s.chunkSource fuel f
        | [] => none
    else none

/-- Structural endpoint of a leaf memlet edge (`Decisions.md#D29`). -/
inductive Endpoint where
  /-- A container side with its subset on the leaf memlet (`none` = full range). -/
  | data (n : Name) (sub : Option Subset)
  /-- A scalar value on a code→code edge, keyed by the producer. -/
  | code (key : NodeId × Conn)
  | invalid
  deriving DecidableEq, Repr, Inhabited

namespace Endpoint
def isData : Endpoint → Bool
  | data _ _ => true
  | _ => false
def data? : Endpoint → Option Name
  | data n _ => some n
  | _ => none
end Endpoint

def leafSrcSubset (e : DEdge) : Option Subset := e.data.memlet?.bind (·.srcSubset)
def leafDstSubset (e : DEdge) : Option Subset := e.data.memlet?.bind (·.dstSubset)

/-- Source endpoint of a leaf edge: its source access node, the access node
at the start of its memlet path (through entry nodes), the producing code
node, or the chunk container of a consume scope whose `OUT_stream` the path
crosses (`consumeChunkName`, whole range: the memlet's subset on the stream
is not used, `Decisions.md#D38`). -/
def srcEndpoint (s : State) (e : DEdge) : Endpoint :=
  match s.chunkSource (s.edges.length + 1) e with
  | some entry =>
    match s.consumedStream? entry with
    | some d => .data (consumeChunkName d) none
    | none => .invalid
  | none =>
    let r := s.pathStart e
    match s.node? r.src with
    | none => .invalid
    | some n =>
      match n.kind with
      | .access d => .data d (leafSrcSubset e)
      | .tasklet _ | .nested _ _ _ | .library _ => .code (r.src, r.srcConn.getD "")
      | _ => .invalid

/-- Destination endpoint of a leaf edge: its destination access node, the
access node reached through exit nodes, or a code value (code nodes and
entry nodes consume the value produced by the leaf's source). -/
def dstEndpoint (s : State) (e : DEdge) : Endpoint :=
  let key : NodeId × Conn := (e.src, e.srcConn.getD "")
  match s.node? e.dst with
  | none => .invalid
  | some n =>
    match n.kind with
    | .access d => .data d (leafDstSubset e)
    | .tasklet _ | .nested _ _ _ | .library _ | .mapEntry _ | .consumeEntry _ => .code key
    | .mapExit | .consumeExit =>
      let r := s.dstRootEdge e
      match s.node? r.dst with
      | none => .invalid
      | some rn =>
        match rn.kind with
        | .access d => .data d (leafDstSubset e)
        | .tasklet _ | .nested _ _ _ | .library _ => .code key
        | _ => .invalid

/-- Does the memlet path through `e` start at an access node? -/
def pathStartsAtAccess (s : State) (e : DEdge) : Bool := s.isAccessNode (s.pathStart e).src

/-- Does the memlet path through `e` end at an access node? -/
def pathEndsAtAccess (s : State) (e : DEdge) : Bool := s.isAccessNode (s.pathEnd e).dst

def memletData? (e : DEdge) : Option Name := e.data.memlet?.map (·.data)

/-- The one-in/one-out tail of `get_view_edge` (`dace/sdfg/utils.py:1181-1226`). -/
private def viewEdgeOneOne (s : State) (v : DNode) (i : DEdge) (o? : Option DEdge) : Option (DEdge × Bool) :=
  let srcIsData := s.pathStartsAtAccess i
  let dstIsData := o?.any (s.pathEndsAtAccess ·)
  if srcIsData && !dstIsData then some (i, true)
  else if !srcIsData && dstIsData then o?.map (·, false)
  else if !srcIsData && !dstIsData then none
  else
    match o? with
    | none => some (i, true)
    | some o =>
      if i.dstConn = some "views" then some (i, true)
      else if o.srcConn = some "views" then some (o, false)
      else
        let vd := v.accessData?
        let inData := memletData? i = vd
        let outData := memletData? o = vd
        if inData && !outData then some (o, false)
        else if !inData && outData then some (i, true)
        else if inData && outData then none
        else if s.isEntryNode i.src then some (i, true)
        else if s.isExitNode o.dst then some (o, false)
        else some (i, true)  -- ambiguous: DaCe warns and takes the input edge (D17)

/-- The edge through which a view access node views its data, with `true`
for an incoming edge (read view) and `false` for an outgoing one (write
view).  Mirrors `dace/sdfg/utils.py:get_view_edge` step by step
(`Decisions.md#D17`): empty incoming memlets are ignored; a single edge on one
side wins if its path reaches an access node and the other side has a
different count; otherwise the one-in/one-out rules apply. -/
def viewEdge? (s : State) (v : DNode) : Option (DEdge × Bool) :=
  let ins := (s.inEdges v.id).filter fun e => e.data matches .memlet _
  let outs := s.outEdges v.id
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

end State

namespace Ctx

/-- Is the leaf edge `e` (part of) a view definition rather than a copy
(`Decisions.md#D31`)?  Either its memlet path starts at a view access node
whose view edge is that outgoing edge, or it ends (through exit nodes) at a
view access node whose view edge is that incoming edge. -/
def isViewDef (ctx : Ctx) (s : State) (e : DEdge) : Bool :=
  let viewNode? (id : NodeId) : Option DNode :=
    (s.node? id).filter fun n => match n.kind with
      | .access d => ctx.isView d
      | _ => false
  let rs := s.pathStart e
  let rd := s.dstRootEdge e
  (match viewNode? rs.src with
   | some v => s.viewEdge? v = some (rs, false)
   | none => false) ||
  (match viewNode? rd.dst with
   | some v => s.viewEdge? v = some (rd, true)
   | none => false)

end Ctx

/-! ## Private transients of a scope -/

/-- Does node `n` mention container `t` (`Decisions.md#D39`): an access node
of `t`, a tasklet naming `t` in its code (other than through a connector),
or a map range, consume parameter or nested-SDFG symbol mapping mentioning
`t` (`Expr.mentions`).  Mirrors the uses `determine_allocation_lifetime`
collects (`framecode.py:676-696,803-850`). -/
def DNode.mentions (n : DNode) (t : Name) : Bool :=
  match n.kind with
  | .access d => d = t
  | .tasklet tk => tk.namesInCode t
  | .mapEntry mp => mp.range.mentions t
  | .consumeEntry c => c.numPes.mentions t || c.condition.any (·.mentions t)
  | .nested _ mapping _ => mapping.any (·.2.mentions t)
  | .library _ | .mapExit | .consumeExit => false

/-- Does an interstate edge mention container `t` (condition or assignment
right-hand sides)? -/
def IEdge.mentions (e : IEdge) (t : Name) : Bool := e.cond.mentions t || e.assigns.any (·.2.mentions t)

mutual

/-- Number of mentions of container `t` in a region: nodes mentioning it
(`DNode.mentions`, all states, nested regions included; nested SDFGs have
their own namespace) plus interstate edges, loop and branch expressions
mentioning it. -/
def Region.countMentions (t : Name) : Region → Nat
  | ⟨_, _, blocks, _, edges⟩ => Blocks.countMentions t blocks + (edges.filter (·.mentions t)).length

def Blocks.countMentions (t : Name) : List Block → Nat
  | [] => 0
  | b :: bs => Block.countMentions t b + Blocks.countMentions t bs

def Block.countMentions (t : Name) : Block → Nat
  | .state s => (s.nodes.filter (·.mentions t)).length
  | .region r => Region.countMentions t r
  | .loop ⟨_, _, init, cond, update, _, _, body⟩ =>
    (if cond.mentions t || init.any (·.2.mentions t) || update.any (·.2.mentions t) then 1 else 0) +
      Region.countMentions t body
  | .cond _ brs => Branches.countMentions t brs
  | .breakBlock | .continueBlock | .returnBlock => 0

def Branches.countMentions (t : Name) : List (Option Expr × Region) → Nat
  | [] => 0
  | (c, r) :: rest =>
    (if c.any (·.mentions t) then 1 else 0) + Region.countMentions t r + Branches.countMentions t rest

end

def SDFG.countMentions (g : SDFG) (t : Name) : Nat := Region.countMentions t g.top

namespace Ctx

/-- Transient containers of the current SDFG with `Scope` lifetime all of
whose mentions (over the whole SDFG, `Region.countMentions`) are nodes
inside the scope of `entry` in state `s`; they are allocated per
iteration/task, so their effects are private to it (`Decisions.md#D39`).
Any other lifetime (`State`, `SDFG`, `Global`, `Persistent`, `External`)
allocates the container once outside the scope, shared by all iterations.
Returned with the memory-name prefix. -/
def privateTransients (ctx : Ctx) (s : State) (entry : NodeId) : List Name :=
  let inside := s.scopeNodes entry
  ctx.g.arrays.filterMap fun (t, d) =>
    if d.transient && d.lifetime = .scope then
      let here := (s.nodes.filter fun n => n.mentions t && inside.contains n.id).length
      if here = ctx.g.countMentions t then some (ctx.cname t) else none
    else none

end Ctx

end Dace
