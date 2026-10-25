-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Json
import Dace.STree.Syntax

/-!
# Importing `.stree.json` files

Reads the JSON written by `dace.sdfg.analysis.schedule_tree.serialize`
(`to_json` / `save`) into `STree`.  Every node is `{"type": "<ClassName>",
…}`; scopes carry `"children"`; the embedded DaCe objects (memlets,
descriptors, `Map`/`Consume`/`Tasklet`/`LibraryNode`, code blocks, loop
metadata) use the `.sdfg` encodings, so the parsers of `Dace.SDFG.Json` are
reused.

Rejected at import (`Decisions.md#D48`): `StateBoundaryNode`s (`#D19`),
unknown node types, conditions / assignment values the expression parser does
not accept, and empty memlets where a node needs data (views, reference sets,
dynamic-scope copies).  Empty memlets of code nodes and copies are dropped
(`#D57`): a code node keeps its other memlets, a `CopyNode` with an empty
memlet moves no data and is dropped altogether.

This module is executable tooling: it is not reasoned about.
-/

set_option autoImplicit false

namespace Dace.STree.Json

open Lean (Json)
open Dace.SDFG.Json

/-- A code block with its parsed expression. -/
def pyExpr (j : Json) : P PyExpr := do
  let cb ← codeBlock j
  match Expr.parse? cb.code with
  | some e => pure { code := cb, expr := e }
  | none => throw s!"cannot parse expression '{cb.code}'"

/-- A code block with its parsed assignment. -/
def pyAssign (j : Json) : P PyAssign := do
  let cb ← codeBlock j
  match Expr.parseAssign? cb.code with
  | some (lhs, rhs) => pure { code := cb, lhs, rhs }
  | none => throw s!"cannot parse assignment '{cb.code}'"

/-- A non-empty memlet. -/
def memletOf (j : Json) : P Memlet := do
  match ← memlet? j with
  | some m => pure m
  | none => throw "empty memlet where a data memlet is required"

/-- `in_memlets` / `out_memlets`: `{"kind": "dict"|"set", "items": [...]}`.
Empty memlets are dropped (`Decisions.md#D57`): in a sequential sibling list
they carry no ordering beyond the list's own. -/
def memletMap (j : Json) : P MemletMap := do
  let kind ← strField j "kind"
  let items ← (← field j "items").getArr?
  match kind with
  | "dict" =>
    let entries ← items.toList.mapM fun it => do
      match (← it.getArr?).toList with
      | [c, m] => do pure ((← str c, ·) <$> (← memlet? m))
      | _ => throw "malformed memlet dictionary entry"
    pure (.dict entries.reduceOption)
  | "set" => do pure (.set (← items.toList.mapM memlet?).reduceOption)
  | other => throw s!"unknown memlet mapping kind '{other}'"

/-- Loop metadata (`serialize._loop_to_json`). -/
def loopMeta (j : Json) : P LoopMeta := do
  let cond ← match field? j "loop_condition" with
    | some c => pyExpr c
    | none => pure { code := { code := "True" }, expr := .lit (.bool true) }
  let init ← match field? j "init_statement" with
    | some c => some <$> pyAssign c
    | none => pure none
  let update ← match field? j "update_statement" with
    | some c => some <$> pyAssign c
    | none => pure none
  pure { label := (strField? j "label").getD "", loopVar := strField? j "loop_variable", init, cond, update,
         inverted := boolField j "inverted", updateBeforeCond := boolField j "update_before_condition" (dflt := true) }

/-- A data descriptor (the first entry of the flattened repository). -/
def singleDesc (name : String) (j : Json) : P DataDesc := do
  match (← dataDesc name j).head? with
  | some (_, d) => pure d
  | none => throw s!"empty data descriptor for '{name}'"

/-- The library node's own properties as DaCe prints them
(`LibraryCall.own_properties`, `Decisions.md#D56`): `[name, text]` pairs in
declaration order, stored by `serialize.py` next to the node because the
node's attributes omit default values and JSON objects lose the declaration
order.  Kept as source text only; the semantics reads the node's attributes. -/
def libraryProps (j : Json) : P (List (String × String)) := do
  match field? j "own_properties" with
  | none => pure []
  | some ps =>
    (← ps.getArr?).toList.mapM fun p => do
      match (← p.getArr?).toList with
      | [k, v] => do pure (← str k, ← str v)
      | _ => throw "malformed library node property"

/-- Guids and class names of all code nodes (for `RefSetNode` sources). -/
partial def codeNodeGuids (j : Json) : List (String × String) :=
  let own := match field? j "node" with
    | some n =>
      -- a library node's Python class is the last component of its class path
      let cls := match strField? n "classpath" with
        | some cp => some ((cp.splitOn ".").getLast?.getD cp)
        | none => strField? n "type"
      match strField? (attrs n) "guid", cls with
      | some g, some t => [(g, t)]
      | _, _ => []
    | none => []
  let children := match field? j "children" with
    | some (.arr cs) => cs.toList.flatMap codeNodeGuids
    | _ => []
  own ++ children

/-- Import one node; `none` for a node that is dropped (a `CopyNode` with an
empty memlet, `Decisions.md#D57`). -/
partial def node (guids : List (String × String)) (j : Json) : P (Option SNode) := do
  let t ← strField j "type"
  if t = "CopyNode" then
    let target ← strField j "target"
    return (← memlet? (← field j "memlet")).map (SNode.copyNode target)
  let children ← match field? j "children" with
    | some (.arr cs) => List.reduceOption <$> cs.toList.mapM (node guids)
    | some _ => throw "children is not a list"
    | none => pure []
  some <$> match t with
  | "NamedRegionScope" => pure (.namedRegion ((strField? j "label").getD "") children)
  | "GBlock" => pure (.gblock children)
  | "StateLabel" => SNode.stateLabel <$> strField j "state"
  | "GotoNode" => pure (.goto (strField? j "target"))
  | "AssignNode" => do pure (.assign (← strField j "name") (← pyExpr (← field j "value")))
  | "LoopScope" => do pure (.loopScope (← loopMeta (← field j "loop")) children)
  | "ForScope" => do pure (.forScope (← loopMeta (← field j "loop")) children)
  | "WhileScope" => do pure (.whileScope (← loopMeta (← field j "loop")) children)
  | "DoWhileScope" => do pure (.doWhileScope (← loopMeta (← field j "loop")) children)
  | "IfScope" => do pure (.ifScope (← pyExpr (← field j "condition")) children)
  | "StateIfScope" => do pure (.stateIfScope (← pyExpr (← field j "condition")) children)
  | "BreakNode" => pure .breakNode
  | "ContinueNode" => pure .continueNode
  | "ElifScope" => do pure (.elifScope (← pyExpr (← field j "condition")) children)
  | "ElseScope" => pure (.elseScope children)
  | "MapScope" => do
    let n ← field j "node"
    pure (.mapScope (← mapOf ((strField? n "label").getD "") (attrs n)) children)
  | "ConsumeScope" => do
    let n ← field j "node"
    pure (.consumeScope (← consumeOf ((strField? n "label").getD "") (attrs n)) children)
  | "TaskletNode" => do
    let n ← field j "node"
    let label := (strField? n "label").getD ""
    let tk ← taskletOf label (attrs n) (← conns n "in_connectors") (← conns n "out_connectors")
    pure (.taskletNode tk (← memletMap (← field j "in_memlets")) (← memletMap (← field j "out_memlets")))
  | "LibraryCall" => do
    let n ← field j "node"
    let lib ← libNodeOf n ((strField? n "label").getD "")
    pure (.libraryCall lib (← libraryProps j)
            (← memletMap (← field j "in_memlets")) (← memletMap (← field j "out_memlets")))
  | "DynScopeCopyNode" => do pure (.dynScopeCopy (← strField j "target") (← memletOf (← field j "memlet")))
  | "ViewNode" => do
    let target ← strField j "target"
    pure (.viewNode target (← strField j "source") (← memletOf (← field j "memlet"))
            (← singleDesc "src" (← field j "src_desc")) (← singleDesc target (← field j "view_desc")))
  | "RefSetNode" => do
    let target ← strField j "target"
    let srcJ ← field j "src_desc"
    let src ← match strField? srcJ "kind" with
      | some "code_node" => do
        let g ← strField srcJ "guid"
        pure (RefSetSource.codeNode g (((guids.find? (·.1 = g)).map (·.2)).getD "CodeNode"))
      | _ => RefSetSource.data <$> singleDesc "src" (← field srcJ "desc")
    pure (.refSetNode target (← memletOf (← field j "memlet")) src (← singleDesc target (← field j "ref_desc")))
  | "StateBoundaryNode" => throw "StateBoundaryNode is a conversion artifact and has no schedule-tree syntax (D19)"
  | other => throw s!"unknown schedule tree node type '{other}'"

/-- The consume scopes of a sibling list and its descendants with the stream
each pops from (the memlet of the `IN_stream` `DynScopeCopyNode` before it,
`Decisions.md#D50`) and its chunk size. -/
partial def consumeStreams (cs : List SNode) : List (String × Name × Nat) :=
  let rec go (inStream : Option Name) : List SNode → List (String × Name × Nat)
    | [] => []
    | n :: rest =>
      let here := match n, inStream with
        | .consumeScope c _, some d => [(c.label, d, c.chunksize)]
        | _, _ => []
      let inner := (n.children?.map consumeStreams).getD []
      let next := match n with
        | .dynScopeCopy "IN_stream" m => some m.data
        | .dynScopeCopy _ _ => inStream
        | _ => none
      here ++ inner ++ go next rest
  go none cs

/-- Declare the chunk containers of the consume scopes (`Decisions.md#D38`),
exactly as `Dace.SDFG.Json.declareConsumeChunks` does for SDFGs. -/
def declareConsumeChunks (containers : List (Name × DataDesc)) (children : List SNode) :
    P (List (Name × DataDesc)) := do
  let chunks ← (consumeStreams children).foldlM (fun acc (label, d, chunksize) => do
    match containers.find? (·.1 = d) with
    | some (_, sd) =>
      if sd.isStream then
        let name := consumeChunkName d
        if containers.any (·.1 = name) || acc.any (·.1 = name) then
          throw s!"unsupported: the chunk container '{name}' of consume scope '{label}' is already declared (several consume scopes on one stream, or a container of that name; Decisions.md#D38)"
        else pure (acc ++ [(name, DataDesc.consumeChunk sd.dtype chunksize)])
      else pure acc
    | none => pure acc) []
  pure (containers ++ chunks)

/-- The root (`ScheduleTreeRoot`). -/
def root (j : Json) : P STree := do
  if strField? j "type" ≠ some "ScheduleTreeRoot" then throw "expected a ScheduleTreeRoot"
  let name := (strField? j "name").getD "stree"
  let containerEntries ← match field? j "containers" with
    | none => pure []
    | some m => entries m
  let containers ← containerEntries.mapM fun (n, d) => dataDesc n d
  let symbols ← match field? j "symbols" with
    | none => pure []
    | some m => do
      let es ← entries m
      pure (es.map fun (s, t) => (s, dtype t))
  let constants ← match field? j "constants" with
    | none => pure []
    | some m => do
      let es ← entries m
      es.mapM fun (n, c) => do
        let d ← singleDesc n (← field c "desc")
        let val : Option Val := match field? c "value" with
          | some (.num k) =>
            if k.exponent = 0 then some (.int k.mantissa)
            else some (.float (Lit.floatVal k.mantissa (-(k.exponent : Int))))
          | some (.bool b) => some (.bool b)
          | _ => none
        pure (n, d, val)
  let argNames ← match field? j "arg_names" with
    | none => pure []
    | some v => do (← v.getArr?).toList.mapM str
  let callbackMapping ← match field? j "callback_mapping" with
    | none => pure []
    | some m => do
      let es ← entries m
      es.mapM fun (k, v) => do pure (k, ← str v)
  let guids := codeNodeGuids j
  let children ← match field? j "children" with
    | some (.arr cs) => List.reduceOption <$> cs.toList.mapM (node guids)
    | some _ => throw "children is not a list"
    | none => pure []
  let containers ← declareConsumeChunks containers.flatten children
  pure { name, containers, symbols, constants, argNames, callbackMapping, children }

/-- Parse a schedule tree from JSON text. -/
def parseSTree (text : String) : P STree := do
  let j ← Lean.Json.parse text
  root j

end Dace.STree.Json

namespace Dace

/-- Read and parse a `.stree.json` file. -/
def readSTree (path : System.FilePath) : IO (Except String STree) := do
  let text ← IO.FS.readFile path
  pure (STree.Json.parseSTree text)

end Dace
