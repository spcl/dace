# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for callback batching and outlining in the next-generation frontend:
adjacent interpreter statements coalesce into one callback, merged I/O sets
chain dataflow through the run, and every callback carries outlined
function/call scaffolding registered in the tree's callback mapping.
"""
import ast

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_adjacent_opaque_merged():

    @dace.program
    def two_prints(A: dace.float64[N]):
        print(A)
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(two_prints)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    # Both statements survive, in order, in the merged code
    assert callbacks[0].code.as_string.count('print') == 2


def test_io_chaining():

    @dace.program
    def chained(A: dace.float64[N]):
        box = open('/dev/null')
        box.close()
        A[0] = 1.0

    tree = nextgen.parse_program(chained)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    # `box` is produced inside the run: an output, not an input
    assert 'box' in callbacks[0].output_names
    assert 'box' not in callbacks[0].input_names


def test_no_merge_across_supported_stmt():

    @dace.program
    def interleaved(A: dace.float64[N]):
        print(A)
        A[0] = 1.0
        print(A)

    tree = nextgen.parse_program(interleaved)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 2


def test_no_merge_across_scope_boundary():

    @dace.program
    def scoped(A: dace.float64[N], flag: dace.int32):
        if flag > 0:
            print(A)
        print(A)

    tree = nextgen.parse_program(scoped)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 2
    assert len(_nodes_of_type(tree, tn.IfScope)) == 1


def test_outlined_fields_populated():

    @dace.program
    def outlined(A: dace.float64[N]):
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(outlined)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.outlined_function_name
    assert callback.outlined_function_code is not None
    assert callback.outlined_call_code is not None
    # The scaffolding is valid Python with the callback inputs as parameters
    function_def = ast.parse(callback.outlined_function_code.as_string).body[0]
    assert isinstance(function_def, ast.FunctionDef)
    assert function_def.name == callback.outlined_function_name
    assert [argument.arg for argument in function_def.args.args] == list(callback.input_names)
    ast.parse(callback.outlined_call_code.as_string)


def test_outlined_body_uses_repository_names():

    @dace.program
    def renamed(A: dace.float64[N]):
        b = A  # Alias: source name "b" refers to repository container "A"
        print(b)
        A[0] = 1.0

    tree = nextgen.parse_program(renamed)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.input_names == ['A']
    # The outlined body references the repository container, not the alias
    outlined_body = callback.outlined_function_code.as_string
    assert 'print(A)' in outlined_body
    # The original code text stays source-level
    assert 'print(b)' in callback.code.as_string


def test_callback_mapping_contains_outlined_names():

    @dace.program
    def mapped(A: dace.float64[N]):
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(mapped)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert callbacks[0].outlined_function_name in tree.callback_mapping


def test_detected_callable_reconstitutes_callee_only():
    # Preprocessing embeds detected callables in callee position with the
    # *full call expression* as their qualified name; reconstitution must
    # restore only the callee, not produce a double call.

    def read_box(box):
        return box['value']

    @dace.program
    def uses_callable(A: dace.float64[N]):
        box = {'value': 1.0}
        y = read_box(box)
        A[0] = y

    tree = nextgen.parse_program(uses_callable)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert 'read_box(box)' in callbacks[0].code.as_string
    assert 'read_box(box)(box)' not in callbacks[0].code.as_string
    # The unknown return is an opaque Python object, not a typed container
    assert str(tree.containers['box'].dtype) == 'pyobject'
    assert str(tree.containers['y'].dtype) == 'pyobject'


def test_pretyped_callback_target_stays_typed():

    def read_box(box):
        return box['value']

    @dace.program
    def typed_target(A: dace.float64[N]):
        y = 0.0
        box = {'value': 1.0}
        y = read_box(box)
        A[0] = y

    tree = nextgen.parse_program(typed_target)
    # A pre-bound typed container is reused: the callback returns a typed value
    assert str(tree.containers['y'].dtype) == 'double'
    assert str(tree.containers['box'].dtype) == 'pyobject'


def test_annotated_callback_target_typed():
    # Classic-parity: `y: dace.float64 = call(...)` types the callback result
    # even when descriptor inference is unavailable.

    def read_box(box):
        return box['value']

    @dace.program
    def annotated_target(A: dace.float64[N]):
        box = {'value': 1.0}
        y: dace.float64 = read_box(box)
        A[0] = y

    tree = nextgen.parse_program(annotated_target)
    assert str(tree.containers['y'].dtype) == 'double'
    assert str(tree.containers['box'].dtype) == 'pyobject'


def test_merged_reason_reports_both():

    @dace.program
    def reasons(A: dace.float64[N]):
        print(A)
        A.sort()

    tree = nextgen.parse_program(reasons)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    # Two distinct fallback reasons merged: both survive
    assert ';' in callbacks[0].reason
    assert 'print' in callbacks[0].code.as_string
    assert 'sort' in callbacks[0].code.as_string


if __name__ == '__main__':
    test_adjacent_opaque_merged()
    test_io_chaining()
    test_no_merge_across_supported_stmt()
    test_no_merge_across_scope_boundary()
    test_detected_callable_reconstitutes_callee_only()
    test_pretyped_callback_target_stays_typed()
    test_annotated_callback_target_typed()
    test_outlined_fields_populated()
    test_outlined_body_uses_repository_names()
    test_callback_mapping_contains_outlined_names()
    test_merged_reason_reports_both()
