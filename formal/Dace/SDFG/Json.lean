-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Lean.Data.Json
import Dace.SDFG.Scope

/-!
# Importing `.sdfg` files

Reads the JSON produced by `dace.serialize` (`SDFG.save`) into the `Dace`
syntax.  Only semantic fields are read; everything else is ignored.  Memlets
are normalized to source/destination subsets (`Core/Memlet`), and a copy
whose other side has no subset gets the full range of that container
(`Decisions.md#D9`).  Empty memlets become dependency edges
(`Decisions.md#D27`).  Structure members are flattened into the descriptor
repository under dotted names (`Decisions.md#D21`).

This module is executable tooling: it is not reasoned about.
-/

namespace Dace.SDFG.Json

open Lean (Json JsonNumber)

abbrev P := Except String

/-- Field lookup treating JSON `null` as absent. -/
def field? (j : Json) (k : String) : Option Json :=
  match j.getObjVal? k with
  | .ok v => if v.isNull then none else some v
  | .error _ => none

def field (j : Json) (k : String) : P Json :=
  match field? j k with
  | some v => pure v
  | none => throw s!"missing field '{k}'"

def attrs (j : Json) : Json := (field? j "attributes").getD (Json.mkObj [])

def attr? (j : Json) (k : String) : Option Json := field? (attrs j) k

def str (j : Json) : P String := j.getStr?

def strField (j : Json) (k : String) : P String := do str (← field j k)

def strField? (j : Json) (k : String) : Option String :=
  (field? j k).bind fun v => (v.getStr?).toOption

def boolField (j : Json) (k : String) (dflt : Bool := false) : Bool :=
  ((field? j k).bind fun v => v.getBool?.toOption).getD dflt

/-- Node/block ids appear both as numbers and as decimal strings. -/
def natOf (j : Json) : P Nat :=
  match j with
  | .num n => if n.exponent = 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw "non-natural id"
  | .str s => match s.toNat? with | some n => pure n | none => throw s!"bad id '{s}'"
  | _ => throw "bad id"

def natField? (j : Json) (k : String) : P (Option Nat) :=
  match field? j k with
  | none => pure none
  | some v => some <$> natOf v

/-- Entries of a JSON object, in key order. -/
def entries (j : Json) : P (List (String × Json)) := do
  let o ← j.getObj?
  pure o.toList

/-- An expression: a string in DaCe syntax, a number, or a boolean. -/
def expr (j : Json) : P Expr :=
  match j with
  | .str s =>
    match Expr.parse? s with
    | some e => pure e
    | none => throw s!"cannot parse expression '{s}'"
  | .num n =>
    if n.exponent = 0 then pure (.lit (.int n.mantissa))
    else pure (.lit (.float n.mantissa (-(n.exponent : Int))))
  | .bool b => pure (.lit (.bool b))
  | _ => throw "bad expression"

def exprField (j : Json) (k : String) : P Expr := do expr (← field j k)

def exprField? (j : Json) (k : String) : P (Option Expr) :=
  match field? j k with
  | none => pure none
  | some v => some <$> expr v

def exprList (j : Json) : P (List Expr) := do
  let a ← j.getArr?
  a.toList.mapM expr

def exprListField (j : Json) (k : String) : P (List Expr) :=
  match field? j k with
  | none => pure []
  | some v => exprList v

/-- A code block `{string_data, language}`; bare strings are accepted. -/
def codeBlock (j : Json) : P CodeBlock :=
  match j with
  | .str s => pure { code := s }
  | _ => do
    let code ← strField j "string_data"
    let lang := match strField? j "language" with
      | some "Python" => Language.python
      | some "CPP" => .cpp
      | some "MLIR" => .mlir
      | some l => .other l
      | none => .python
    pure { code, language := lang }

/-- A Python code block as an expression (interstate conditions). -/
def codeExpr (j : Json) : P Expr := do
  let cb ← codeBlock j
  match Expr.parse? cb.code with
  | some e => pure e
  | none => throw s!"cannot parse condition '{cb.code}'"

/-- A Python assignment statement `x = e`. -/
def codeAssign (j : Json) : P (Sym × Expr) := do
  let cb ← codeBlock j
  match Expr.parseAssign? cb.code with
  | some a => pure a
  | none => throw s!"cannot parse assignment '{cb.code}'"

def dtypeOfString (s : String) : Dtype :=
  match s with
  | "float16" => .float 16 | "float32" => .float 32 | "float64" => .float 64
  | "int8" => .int 8 true | "int16" => .int 16 true | "int32" => .int 32 true | "int64" => .int 64 true
  | "uint8" => .int 8 false | "uint16" => .int 16 false | "uint32" => .int 32 false | "uint64" => .int 64 false
  | "bool" | "bool_" => .bool
  | other => .opaque other

def dtype (j : Json) : Dtype :=
  match j with
  | .str s => dtypeOfString s
  | other => .opaque other.compress

/-- Connector table `{name: dtype-or-null}`. -/
def conns (j : Json) (k : String) : P Conns :=
  match field? (attrs j) k with
  | none => pure []
  | some v => do
    let es ← entries v
    pure (es.map fun (c, t) => (c, if t.isNull then none else some (dtype t)))

/-! ## Subsets and memlets -/

def dim (j : Json) : P Dim := do
  let lo ← exprField j "start"
  let hi ← exprField j "end"
  let step ← (exprField? j "step").map (·.getD 1)
  let tile ← (exprField? j "tile").map (·.getD 1)
  pure { lo, hi, step, tile }

def range (j : Json) : P Range := do
  let rs ← (← field j "ranges").getArr?
  pure ⟨← rs.toList.mapM dim⟩

def subset (j : Json) : P Subset := do
  match strField? j "type" with
  | some "Range" => .range <$> range j
  | some "Indices" => do
    let is ← exprListField j "indices"
    pure (.range (Range.ofIndices is))
  | some "SubsetUnion" => do
    let rs ← (← field j "subset_list").getArr?
    pure (.union (← rs.toList.mapM range))
  | some t => throw s!"unknown subset type '{t}'"
  | none => throw "subset without type"

def subset? (j : Json) (k : String) : P (Option Subset) :=
  match field? j k with
  | none => pure none
  | some v => some <$> subset v

/-- A memlet or, for empty memlets, `none`. -/
def memlet? (j : Json) : P (Option Memlet) := do
  let a := attrs j
  match strField? a "data" with
  | none => pure none
  | some data => do
    let sub ← subset? a "subset"
    let other ← subset? a "other_subset"
    let srcJ ← subset? a "src_subset"
    let dstJ ← subset? a "dst_subset"
    let isDataSrc := match field? a "is_data_src" with
      | some v => v.getBool?.toOption.getD true
      | none => true
    let hasExplicit := (field? a "src_subset").isSome || (field? a "dst_subset").isSome
    let (srcS, dstS) :=
      if hasExplicit then (srcJ, dstJ)
      else if isDataSrc then (sub, other) else (other, sub)
    let volume ← (exprField? a "volume").map (·.getD 0)
    let wcr ← match strField? a "wcr" with
      | none => pure none
      | some lam =>
        match WcrOp.ofLambda? lam with
        | some op => pure (some op)
        | none => throw s!"cannot parse wcr '{lam}'"
    pure (some { data, isDataSrc, srcSubset := srcS, dstSubset := dstS, volume,
                 dynamic := boolField a "dynamic", wcr,
                 wcrNonatomic := boolField a "wcr_nonatomic", allowOob := boolField a "allow_oob" })

/-! ## Data descriptors -/

/-- `AllocationLifetime` by its serialized name (`serialize.py:142-144`); an
absent field is the default `Scope` (`dace/data/core.py`).  Unknown names are
rejected, since the lifetime decides privacy (`Decisions.md#D39`). -/
def lifetime (s : Option String) : P Lifetime :=
  match s with
  | none | some "Scope" => pure .scope
  | some "State" => pure .state | some "SDFG" => pure .sdfg | some "Global" => pure .global
  | some "Persistent" => pure .persistent | some "External" => pure .external
  | some other => throw s!"unknown allocation lifetime '{other}'"

def baseKindOf (t : String) : BaseKind :=
  if t.startsWith "Structure" then .structure
  else if t.startsWith "ContainerArray" || t = "ContainerView" then .containerArray
  else .array

/-- Parse a descriptor under `name`; returns it and the flattened member
descriptors of structures / the element descriptor of container arrays. -/
partial def dataDesc (name : Name) (j : Json) : P (List (Name × DataDesc)) := do
  let t ← strField j "type"
  let a := attrs j
  let shape ← exprListField a "shape"
  let strides ← exprListField a "strides"
  let offset ← exprListField a "offset"
  let totalSize ← (exprField? a "total_size").map (·.getD (shape.foldl (· * ·) (1 : Expr)))
  let startOffset ← (exprField? a "start_offset").map (·.getD 0)
  let lt ← lifetime (strField? a "lifetime")
  let base : DataDesc :=
    { dtype := (field? a "dtype").map dtype |>.getD (.opaque "void")
      shape, strides, offset, totalSize, startOffset
      transient := boolField a "transient"
      storage := (strField? a "storage").getD "Default"
      lifetime := lt
      mayAlias := boolField a "may_alias"
      allowConflicts := boolField a "allow_conflicts" }
  match t with
  | "Array" => pure [(name, { base with kind := .array })]
  | "Scalar" => pure [(name, { base with kind := .scalar, shape := [1], strides := [1], totalSize := 1 })]
  | "Stream" => do
    let bs ← (exprField? a "buffer_size").map (·.getD 0)
    pure [(name, { base with kind := .stream bs })]
  | "View" | "ArrayView" => pure [(name, { base with kind := .view .array, transient := true })]
  | "StructureView" => pure [(name, { base with kind := .view .structure, transient := true })]
  | "ContainerView" => pure [(name, { base with kind := .view .containerArray, transient := true })]
  | "Reference" | "ArrayReference" => pure [(name, { base with kind := .reference .array, transient := true })]
  | "StructureReference" => pure [(name, { base with kind := .reference .structure, transient := true })]
  | "ContainerArrayReference" =>
    pure [(name, { base with kind := .reference .containerArray, transient := true })]
  | "Structure" => do
    let memberEntries ← match field? a "members" with
      | none => pure []
      | some m => entries m
    let members ← memberEntries.mapM fun (mn, mj) => dataDesc s!"{name}.{mn}" mj
    let typeName := (strField? a "name").getD name
    pure ((name, { base with kind := .structure typeName (memberEntries.map Prod.fst) }) :: members.flatten)
  | "ContainerArray" => do
    let elemName := s!"{name}.__elem"
    let elem ← match field? a "stype" with
      | some sj => dataDesc elemName sj
      | none => pure []
    pure ((name, { base with kind := .containerArray elemName }) :: elem)
  | other => throw s!"unknown data descriptor type '{other}'"

/-! ## Nodes, states, regions -/

def scopeEntry? (j : Json) : P (Option NodeId) := natField? j "scope_entry"

def reduceProps (a : Json) : P ReduceProps := do
  let axes ← match field? a "axes" with
    | none => pure none
    | some v => do
      let arr ← v.getArr?
      some <$> arr.toList.mapM natOf
  let wcr ← match strField? a "wcr" with
    | some lam => match WcrOp.ofLambda? lam with
      | some op => pure op
      | none => throw s!"cannot parse reduce wcr '{lam}'"
    | none => pure .exchange
  let identity ← exprField? a "identity"
  pure { axes, wcr, identity }

/-! The node payloads below are shared with the schedule-tree importer
(`Dace.STree.Json`), where they appear without an enclosing graph. -/

/-- A `Tasklet` from its label, attributes and connector tables. -/
def taskletOf (label : String) (a : Json) (inC outC : Conns) : P Tasklet := do
  let code ← match field? a "code" with
    | some c => codeBlock c
    | none => pure { code := "" }
  let se := (field? a "side_effects").bind fun v => v.getBool?.toOption
  pure { label, code, inConns := inC, outConns := outC, sideEffects := se }

/-- A `Map` from its attributes (`nodes.Map` / `MapEntry`). -/
def mapOf (label : String) (a : Json) : P MapNode := do
  let params ← (← field a "params").getArr? >>= fun ps => ps.toList.mapM str
  let rng ← range (← field a "range")
  let collapse ← (natField? a "collapse").map (·.getD 1)
  pure { label := (strField? a "label").getD label, params, range := rng,
         schedule := (strField? a "schedule").getD "Default", unroll := boolField a "unroll", collapse }

/-- A `Consume` from its attributes (`nodes.Consume` / `ConsumeEntry`). -/
def consumeOf (label : String) (a : Json) : P ConsumeNode := do
  let peIndex ← strField a "pe_index"
  let numPes ← (exprField? a "num_pes").map (·.getD 1)
  let condition ← match field? a "condition" with
    | some c => some <$> codeExpr c
    | none => pure none
  let chunksize ← (natField? a "chunksize").map (·.getD 1)
  pure { label := (strField? a "label").getD label, peIndex, numPes, condition, chunksize,
         schedule := (strField? a "schedule").getD "Default" }

/-- A `LibraryNode` from its JSON object (`classpath` at the top level). -/
def libNodeOf (j : Json) (label : String) : P LibNode := do
  let a := attrs j
  let classpath := (strField? j "classpath").getD ""
  let reduce ← if classpath.endsWith ".Reduce" then some <$> reduceProps a else pure none
  pure { classpath, label, implementation := strField? a "implementation", reduce }

/-- Declare the chunk containers of the consume scopes of an SDFG
(`Decisions.md#D38`): for every consume entry popping from stream `d`, a
transient `consumeChunkName d` described by `DataDesc.consumeChunk` with the
stream's element type, as codegen declares it (`cpu.py:2273-2291`).  A
consume entry whose `IN_stream` path does not start at a stream is skipped
(the semantics reports it as `malformed`).  A chunk name that is already
taken (two consume scopes on one stream, or a user container of that name)
is rejected as unsupported. -/
def declareConsumeChunks (g : SDFG) : P SDFG := do
  let consumes := g.top.allStates.flatMap fun s => s.nodes.filterMap fun n =>
    match n.kind with
    | .consumeEntry c => (s.consumedStream? n.id).map fun d => (n.label, d, c.chunksize)
    | _ => none
  let chunks ← consumes.foldlM (fun acc (label, d, chunksize) => do
    match g.arrays.find? (·.1 = d) with
    | some (_, sd) =>
      if sd.isStream then
        let name := consumeChunkName d
        if g.arrays.any (·.1 = name) || acc.any (·.1 = name) then
          throw s!"unsupported: the chunk container '{name}' of consume scope '{label}' is already declared (several consume scopes on one stream, or a container of that name; Decisions.md#D38)"
        else pure (acc ++ [(name, DataDesc.consumeChunk sd.dtype chunksize)])
      else pure acc
    | none => pure acc) []
  pure { g with arrays := g.arrays ++ chunks }

/-- A relation `{"kind", "lhs", "rhs"}`; `none` for scale relations (`MUCH_LT`). -/
def relation (j : Json) : P (Option Relation) := do
  match RelKind.ofString? (← strField j "kind") with
  | none => pure none
  | some kind => pure (some { kind, lhs := ← exprField j "lhs", rhs := ← exprField j "rhs" })

/-- The facts implied by declared symbol types alone (`Decisions.md#D60`):
symbols of an integer type other than `bool` are integers, unsigned ones are
nonnegative.  Used when an SDFG has no symbol repository. -/
def typeFacts (types : List (Sym × Dtype)) : Constraints :=
  { intSyms := types.filterMap fun (s, t) => match t with
      | .int _ _ => some s
      | _ => none
    sign := types.filterMap fun (s, t) => match t with
      | .int _ false => some (s, Pred.nonneg)
      | _ => none }

/-- The facts of `attributes.symbol_repo` (`symbol_repo.py`, `SymbolRepo.to_json`)
and the declared parameter types.  Only `params` is read: the scale facts
(`small`, `MUCH_LT`), `runtime_assumptions` and the facts of inner scopes are
ignored (`Core/Facts`, `Decisions.md#D60`).  As in `symbol_facts`, integer
symbols are those of an integer type other than `bool`, and an unsigned type
is a `NONNEGATIVE` fact of its own. -/
def symbolRepo (j : Json) : P (Constraints × List (Sym × Dtype)) := do
  let params := (field? j "params").getD (Json.mkObj [])
  let types ← match field? params "types" with
    | none => pure []
    | some m => do
      let es ← entries m
      pure (es.map fun (s, t) => (s, dtype t))
  let intSyms := types.filterMap fun (s, t) => match t with
    | .int _ _ => some s
    | _ => none
  let unsigned := types.filterMap fun (s, t) => match t with
    | .int _ false => some (s, Pred.nonneg)
    | _ => none
  let preds ← match field? params "predicates" with
    | none => pure []
    | some m => do
      let es ← entries m
      es.flatMapM fun (s, ps) => do
        let names ← (← ps.getArr?).toList.mapM str
        names.mapM fun n => match Pred.ofString? n with
          | some p => pure (s, p)
          | none => throw s!"unknown predicate '{n}'"
  let rels ← match field? params "relations" with
    | none => pure []
    | some v => do (← v.getArr?).toList.filterMapM relation
  pure ({ intSyms, sign := preds ++ unsigned, rels }, types)

mutual

partial def node (j : Json) : P DNode := do
  let t ← strField j "type"
  let id ← natOf (← field j "id")
  let label := (strField? j "label").getD ""
  let a := attrs j
  let inC ← conns j "in_connectors"
  let outC ← conns j "out_connectors"
  let se ← scopeEntry? j
  let kind ← match t with
    | "AccessNode" => NodeKind.access <$> strField a "data"
    | "Tasklet" => NodeKind.tasklet <$> taskletOf label a inC outC
    | "MapEntry" => NodeKind.mapEntry <$> mapOf label a
    | "MapExit" => pure .mapExit
    | "ConsumeEntry" => NodeKind.consumeEntry <$> consumeOf label a
    | "ConsumeExit" => pure .consumeExit
    | "NestedSDFG" => do
      let inner ← sdfg (← field a "sdfg")
      let sm ← match field? a "symbol_mapping" with
        | none => pure []
        | some m => do
          let es ← entries m
          es.mapM fun (k, v) => do pure (k, ← expr v)
      pure (.nested inner sm (boolField a "no_inline"))
    | "LibraryNode" => NodeKind.library <$> libNodeOf j label
    | other => throw s!"unknown node type '{other}'"
  pure { id, label, inConns := inC, outConns := outC, scopeEntry := se, kind }

partial def edge (j : Json) : P DEdge := do
  let src ← natOf (← field j "src")
  let dst ← natOf (← field j "dst")
  let srcConn := strField? j "src_connector"
  let dstConn := strField? j "dst_connector"
  let data ← match field? (attrs j) "data" with
    | none => pure EdgeData.dep
    | some m => do
      match ← memlet? m with
      | none => pure .dep
      | some mm => pure (.memlet mm)
  pure { src, srcConn, dst, dstConn, data }

partial def state (arrays : List (Name × DataDesc)) (j : Json) : P State := do
  let label := (strField? j "label").getD ""
  let nodesJ ← (← field j "nodes").getArr?
  let edgesJ ← ((field? j "edges").getD (Json.arr #[])).getArr?
  let nodes ← nodesJ.toList.mapM node
  let edges ← edgesJ.toList.mapM edge
  -- D9: an access→access copy with a missing side gets the full range.
  let fullOf (id : NodeId) : Option Subset := do
    let n ← nodes.find? (·.id = id)
    let d ← n.accessData?
    let desc ← (arrays.find? (·.1 = d)).map (·.2)
    pure (.range desc.fullRange)
  let edges := edges.map fun e =>
    match e.data with
    | .memlet m =>
      let srcIsAccess := (nodes.find? (·.id = e.src)).map DNode.isAccess |>.getD false
      let dstIsAccess := (nodes.find? (·.id = e.dst)).map DNode.isAccess |>.getD false
      if srcIsAccess && dstIsAccess then
        let m := if m.srcSubset.isNone then { m with srcSubset := fullOf e.src } else m
        let m := if m.dstSubset.isNone then { m with dstSubset := fullOf e.dst } else m
        { e with data := .memlet m }
      else e
    | .dep => e
  pure { label, nodes, edges, nosync := boolField (attrs j) "nosync" }

partial def iedge (j : Json) : P IEdge := do
  let src ← natOf (← field j "src")
  let dst ← natOf (← field j "dst")
  let a := attrs ((field? (attrs j) "data").getD (Json.mkObj []))
  let cond ← match field? a "condition" with
    | some c => codeExpr c
    | none => pure (.lit (.bool true))
  let assigns ← match field? a "assignments" with
    | none => pure []
    | some m => do
      let es ← entries m
      es.mapM fun (k, v) => do pure (k, ← expr v)
  pure { src, dst, cond, assigns }

partial def regionBody (arrays : List (Name × DataDesc)) (j : Json) (kind : RegionKind) : P Region := do
  let label := (strField? j "label").getD ""
  let blocksJ ← ((field? j "nodes").getD (Json.arr #[])).getArr?
  let edgesJ ← ((field? j "edges").getD (Json.arr #[])).getArr?
  let blocks ← blocksJ.toList.mapM (block arrays)
  let edges ← edgesJ.toList.mapM iedge
  -- `ControlFlowRegion.start_block` (`dace/sdfg/state.py:3459-3476`): the unique
  -- source block if there is one, else the explicitly stored `start_block`
  -- (`Decisions.md#D45`).
  let explicit ← natField? j "start_block"
  let sources := (List.range blocks.length).filter fun b => edges.all (·.dst ≠ b)
  let start ← match sources, explicit with
    | [s], _ => pure s
    | _, some s => pure s
    | _, none =>
      if blocks.isEmpty then pure 0
      else throw s!"ambiguous or undefined start block of region '{label}'"
  pure { label, kind, blocks, start, edges }

partial def block (arrays : List (Name × DataDesc)) (j : Json) : P Block := do
  let t ← strField j "type"
  match t with
  | "SDFGState" => Block.state <$> state arrays j
  | "ControlFlowRegion" => Block.region <$> regionBody arrays j .plain
  | "NamedRegion" => Block.region <$> regionBody arrays j (.named ((strField? j "label").getD ""))
  | "FunctionCallRegion" => Block.region <$> regionBody arrays j (.functionCall ((strField? j "label").getD ""))
  | "UnstructuredControlFlow" => Block.region <$> regionBody arrays j .unstructured
  | "LoopRegion" => do
    let a := attrs j
    let body ← regionBody arrays j .plain
    let init ← match field? a "init_statement" with
      | some c => some <$> codeAssign c
      | none => pure none
    let update ← match field? a "update_statement" with
      | some c => some <$> codeAssign c
      | none => pure none
    let cond ← match field? a "loop_condition" with
      | some c => codeExpr c
      | none => pure (.lit (.bool true))
    pure (.loop { label := body.label, loopVar := strField? a "loop_variable", init, cond, update,
                  inverted := boolField a "inverted",
                  updateBeforeCond := boolField a "update_before_condition" (dflt := true), body })
  | "ConditionalBlock" => do
    let brs ← ((field? j "branches").getD (Json.arr #[])).getArr?
    let branches ← brs.toList.mapM fun b => do
      let pair ← b.getArr?
      match pair.toList with
      | [c, r] => do
        let cond ← if c.isNull then pure none else some <$> codeExpr c
        let reg ← regionBody arrays r .plain
        pure (cond, reg)
      | _ => throw "malformed conditional branch"
    pure (.cond ((strField? j "label").getD "") branches)
  | "BreakBlock" => pure .breakBlock
  | "ContinueBlock" => pure .continueBlock
  | "ReturnBlock" => pure .returnBlock
  | other => throw s!"unknown control-flow block type '{other}'"

partial def sdfg (j : Json) : P SDFG := do
  let a := attrs j
  let name := (strField? a "name").getD "sdfg"
  let arrayEntries ← match field? a "_arrays" with
    | none => pure []
    | some m => entries m
  let arrays ← arrayEntries.mapM fun (n, d) => dataDesc n d
  let arrays := arrays.flatten
  let symbols ← match field? a "symbols" with
    | none => pure []
    | some m => do
      let es ← entries m
      pure (es.map fun (s, t) => (s, dtype t))
  let constants ← match field? a "constants_prop" with
    | none => pure []
    | some m => do
      let es ← entries m
      es.filterMapM fun (n, v) => do
        match v.getArr? with
        | .ok pair =>
          match pair.toList with
          | [dj, vj] =>
            let ds ← dataDesc n dj
            let val : Option Val := match vj with
              | .num k => if k.exponent = 0 then some (.int k.mantissa) else some (.float (Lit.floatVal k.mantissa (-(k.exponent : Int))))
              | .bool b => some (.bool b)
              | _ => none
            pure (ds.head?.map fun (_, d) => (n, d, val))
          | _ => pure none
        | .error _ => pure none
  let argNames ← match field? a "arg_names" with
    | none => pure []
    | some v => do (← v.getArr?).toList.mapM str
  let top ← regionBody arrays j .plain
  let (facts, repoTypes) ← match field? a "symbol_repo" with
    | none => pure (typeFacts symbols, [])
    | some r => symbolRepo r
  -- with a symbol repository, `SDFG.symbols` is a view of `params.types`
  let symbols := if symbols.isEmpty then repoTypes else symbols
  declareConsumeChunks
    { name, arrays, symbols, constants, argNames, top := { top with label := name }, facts }

end

/-- Parse an SDFG from JSON text. -/
def parseSDFG (text : String) : P SDFG := do
  let j ← Lean.Json.parse text
  sdfg j

/-- Read and parse a `.sdfg` / `.sdfg.json` file. -/
def readSDFG (path : System.FilePath) : IO (P SDFG) := do
  let text ← IO.FS.readFile path
  pure (parseSDFG text)

end Dace.SDFG.Json
