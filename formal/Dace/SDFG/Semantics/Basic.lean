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

Nothing here is recursive over the SDFG syntax except the access-node count
used by `Ctx.privateTransients`.
-/

set_option autoImplicit false

namespace Dace

/-! ## Outcomes -/

/-- Reasons for an erroneous outcome.  `malformed` covers situations that
well-formedness (`SDFG.WF`) excludes; it keeps the semantics total over the
syntax. -/
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
  /-- The tasklet interpretation is undefined for these inputs. -/
  | taskletUndefined (label : String)
  /-- The library-node interpretation is undefined for this class path. -/
  | libraryUnsupported (classpath : String)
  | malformed (msg : String)
  deriving Inhabited

/-- Human-readable rendering (for tools and test output). -/
def ErrReason.toString : ErrReason → String
  | .race e => s!"data race: {repr e}"
  | .stuckPop l => s!"pop from an empty queue at {repr l}"
  | .undefinedExpr e => s!"undefined expression '{e}'"
  | .undefinedSubset sub => s!"undefined subset {repr sub}"
  | .unsetRef n => s!"reference '{n}' accessed before it was set"
  | .taskletUndefined label => s!"tasklet '{label}' is undefined for its inputs"
  | .libraryUnsupported classpath => s!"unsupported library node '{classpath}'"
  | .malformed msg => s!"malformed SDFG: {msg}"

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

/-- Location of element `idx` of container `n`.  Fails with `unsetRef` for a
reference that is not bound (`Decisions.md#D11`). -/
def loc (ctx : Ctx) (ρ : Valuation) (m : Mem) (n : Name) (idx : Idx) : Res Loc :=
  match ctx.desc? n with
  | none => .err (.malformed s!"unknown container '{n}'")
  | some d =>
    if d.isReference && !m.isAliased (ctx.cname n) then .err (.unsetRef n)
    else match ctx.elemOffset ρ d idx with
      | some o => .ok (Loc.elem (ctx.cname n) o)
      | none => .err (.malformed s!"undefined element offset {idx} in '{n}'")

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

/-- Read one element, logging the (resolved) location. -/
def read (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) : Res (Val × St) := do
  let l ← ctx.loc ρ st.mem n idx
  pure (st.mem.read l, st.logged { reads := [st.mem.resolve l] })

/-- Read several elements in order. -/
def readMany (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (pts : List Idx) : Res (List Val × St) :=
  pts.foldlM (fun (acc, st) p => do
      let (v, st) ← st.read ctx ρ n p
      pure (acc ++ [v], st)) ([], st)

/-- Plain write. -/
def write (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (v : Val) : Res St := do
  let l ← ctx.loc ρ st.mem n idx
  pure { st with mem := st.mem.write l v, log := st.log.seq { writes := [(st.mem.resolve l, v)] } }

/-- WCR contribution: the local view is updated immediately with the
interpretation's operator; the log records the contribution. -/
def wcr (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (op : WcrOp) (v : Val) : Res St := do
  let l ← ctx.loc ρ st.mem n idx
  let old := st.mem.read l
  pure { st with mem := st.mem.write l (ctx.I.wcr op old v),
                 log := st.log.seq { wcrs := [(st.mem.resolve l, op, v)] } }

/-- Plain write or WCR contribution according to the memlet. -/
def store (ctx : Ctx) (ρ : Valuation) (st : St) (n : Name) (idx : Idx) (wcrOp : Option WcrOp) (v : Val) : Res St :=
  match wcrOp with
  | none => st.write ctx ρ n idx v
  | some op => st.wcr ctx ρ n idx op v

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

/-- Follow `prevEdges` while the source is a scope node entered through an
`OUT_` connector (`memlet_path(e)[0]`). -/
def walkPrev (s : State) : Nat → DEdge → DEdge
  | 0, e => e
  | fuel + 1, e =>
    if s.isScopeNode e.src && connPrefix e.srcConn "OUT_" then
      match s.prevEdges e with
      | f :: _ => s.walkPrev fuel f
      | [] => e
    else e

/-- Follow `nextEdges` while the destination is a scope node (only exit nodes
when `onlyExits`) entered through an `IN_` connector (`memlet_path(e)[-1]`). -/
def walkNext (s : State) (onlyExits : Bool) : Nat → DEdge → DEdge
  | 0, e => e
  | fuel + 1, e =>
    if (if onlyExits then s.isExitNode e.dst else s.isScopeNode e.dst) && connPrefix e.dstConn "IN_" then
      match s.nextEdges e with
      | f :: _ => s.walkNext onlyExits fuel f
      | [] => e
    else e

/-- First edge of the memlet path through `e`. -/
def pathStart (s : State) (e : DEdge) : DEdge := s.walkPrev (s.edges.length + 1) e

/-- Last edge of the memlet path through `e`. -/
def pathEnd (s : State) (e : DEdge) : DEdge := s.walkNext false (s.edges.length + 1) e

/-- Outermost edge reached from `e` through exit nodes only. -/
def dstRootEdge (s : State) (e : DEdge) : DEdge := s.walkNext true (s.edges.length + 1) e

/-- The consume entry whose `OUT_stream` connector the path into `e`
crosses, if any (the data then is the consumed chunk, `Decisions.md#D38`). -/
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
  /-- The chunk consumed by a consume scope. -/
  | chunk (entry : NodeId)
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
node, or the consumed chunk. -/
def srcEndpoint (s : State) (e : DEdge) : Endpoint :=
  match s.chunkSource (s.edges.length + 1) e with
  | some entry => .chunk entry
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

mutual

/-- Number of access nodes of container `t` in a region (all states, nested
regions included; nested SDFGs have their own namespace). -/
def Region.countAccess (t : Name) : Region → Nat
  | ⟨_, _, blocks, _, _⟩ => Blocks.countAccess t blocks

def Blocks.countAccess (t : Name) : List Block → Nat
  | [] => 0
  | b :: bs => Block.countAccess t b + Blocks.countAccess t bs

def Block.countAccess (t : Name) : Block → Nat
  | .state s => (s.nodes.filter fun n => n.accessData? = some t).length
  | .region r => Region.countAccess t r
  | .loop ⟨_, _, _, _, _, _, _, body⟩ => Region.countAccess t body
  | .cond _ brs => Branches.countAccess t brs
  | .breakBlock | .continueBlock | .returnBlock => 0

def Branches.countAccess (t : Name) : List (Option Expr × Region) → Nat
  | [] => 0
  | (_, r) :: rest => Region.countAccess t r + Branches.countAccess t rest

end

def SDFG.countAccess (g : SDFG) (t : Name) : Nat := Region.countAccess t g.top

namespace Ctx

/-- Transient containers of the current SDFG all of whose access nodes (over
every state of the SDFG) lie inside the scope of `entry` in state `s`; their
effects are private to one iteration/task (`Decisions.md#D39`).  Returned
with the memory-name prefix. -/
def privateTransients (ctx : Ctx) (s : State) (entry : NodeId) : List Name :=
  let inside := s.scopeNodes entry
  ctx.g.arrays.filterMap fun (t, d) =>
    if d.transient then
      let here := (s.nodes.filter fun n => n.accessData? = some t && inside.contains n.id).length
      if here = ctx.g.countAccess t then some (ctx.cname t) else none
    else none

end Ctx

end Dace
