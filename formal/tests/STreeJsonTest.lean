-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Corpus-independent checks of the schedule-tree importer (`Dace.STree.Json`).
Run with `lake env lean tests/STreeJsonTest.lean`; every `#guard` must pass.

Empty memlets (`Decisions.md#D57`): the tree below (written by
`dace.sdfg.analysis.schedule_tree.serialize`, guids dropped) has a tasklet with
an empty memlet on its connector `__dep` and a `CopyNode` with an empty memlet.
The importer drops the empty memlet of the tasklet and the copy altogether. -/

set_option autoImplicit false

open Dace

def emptyMemlets : String := r#"{"arg_names": [], "callback_mapping": {}, "children": [
  {"in_memlets": {"items": [
     ["__in", {"attributes": {"data": "A", "dst_subset": null, "is_data_src": null, "num_accesses": "1",
               "src_subset": {"ranges": [{"end": "0", "start": "0", "step": "1", "tile": "1"}], "type": "Range"},
               "subset": {"ranges": [{"end": "0", "start": "0", "step": "1", "tile": "1"}], "type": "Range"},
               "volume": "1"}, "type": "Memlet"}],
     ["__dep", {"attributes": {"dst_subset": null, "is_data_src": null, "num_accesses": "1", "src_subset": null,
               "volume": "1"}, "type": "Memlet"}]], "kind": "dict"},
   "node": {"attributes": {"code": {"language": "Python", "string_data": "__out = (__in + 1)"},
            "in_connectors": {"__dep": null, "__in": null}, "label": "t", "out_connectors": {"__out": null}},
            "label": "t", "type": "Tasklet"},
   "out_memlets": {"items": [
     ["__out", {"attributes": {"data": "A", "dst_subset": null, "is_data_src": null, "num_accesses": "1",
                "src_subset": {"ranges": [{"end": "1", "start": "1", "step": "1", "tile": "1"}], "type": "Range"},
                "subset": {"ranges": [{"end": "1", "start": "1", "step": "1", "tile": "1"}], "type": "Range"},
                "volume": "1"}, "type": "Memlet"}]], "kind": "dict"},
   "type": "TaskletNode"},
  {"memlet": {"attributes": {"dst_subset": null, "is_data_src": null, "num_accesses": "1", "src_subset": null,
              "volume": "1"}, "type": "Memlet"}, "target": "A", "type": "CopyNode"}],
 "constants": {},
 "containers": {"A": {"attributes": {"dtype": "float64", "offset": ["0"], "shape": ["4"], "strides": ["1"],
                                     "total_size": "4"}, "type": "Array"}},
 "name": "empty_memlets", "symbols": {}, "type": "ScheduleTreeRoot"}"#

def imported : Except String STree := STree.Json.parseSTree emptyMemlets

/-- Connector names of the single remaining tasklet's inputs, if that is the tree's shape. -/
def taskletInputs : Option (List String) :=
  match imported with
  | .ok t =>
    match t.children with
    | [.taskletNode _ (.dict ins) _] => some (ins.map (·.1))
    | _ => none
  | .error _ => none

#eval imported.map STree.prettyPrint
#guard taskletInputs == some ["__in"]
#guard (imported.map STree.prettyPrint).toOption == some "  A[1] = tasklet(A[0])"

/-- Library-node properties are printed from the stored `own_properties` text
(`Decisions.md#D56`), in their stored order. -/
def libraryCall : String := r#"{"type": "ScheduleTreeRoot", "name": "lib", "containers": {}, "children": [
  {"type": "LibraryCall", "own_properties": [["transB", "False"], ["alpha", "1"]],
   "in_memlets": {"kind": "set", "items": []}, "out_memlets": {"kind": "set", "items": []},
   "node": {"type": "MatMul", "label": "mm", "classpath": "dace.libraries.blas.nodes.matmul.MatMul",
            "attributes": {"in_connectors": {}, "out_connectors": {}}}}]}"#

#eval (STree.Json.parseSTree libraryCall).map STree.prettyPrint
#guard ((STree.Json.parseSTree libraryCall).map STree.prettyPrint).toOption ==
  some "   = library MatMul[transB=False, alpha=1]()"
