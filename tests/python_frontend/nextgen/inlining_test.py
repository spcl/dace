# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for nested ``@dace.program`` call inlining in the next-generation
frontend: structural inlining into FunctionCallScope with a shared repository,
return-value binding, recursion/early-return fallback, and explicit SDFG calls.
"""
import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


@dace.program
def _scale_into(X: dace.float64[N], Y: dace.float64[N], factor: dace.float64):
    Y[:] = X * factor


@dace.program
def _add_one(X: dace.float64[N]):
    return X + 1.0


def test_bare_call_inlines():

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        _scale_into(A, B, 2.0)

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 1
    assert scopes[0].call.callee_name.endswith('_scale_into')
    assert scopes[0].call.arguments['X'] == 'A'
    # The callee body references the caller's containers directly
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert any(memlet.data == 'A' for memlet in tasklets[0].in_memlets.values())
    assert tasklets[0].out_memlets['__out'].data == 'B'
    # The constant argument specialized into the tasklet code
    assert '2.0' in tasklets[0].node.code.as_string
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_return_value_binding():

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        y = _add_one(A)
        B[:] = y

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 1
    # The return value materialized into a prefixed transient container
    return_names = [name for name in tree.containers if '_add_one_ret' in name]
    assert len(return_names) == 1
    assert tree.containers[return_names[0]].transient is True
    # The caller consumes it through a copy into B
    copies = _nodes_of_type(tree, tn.CopyNode)
    assert any(copy.memlet.data == return_names[0] and copy.target == 'B' for copy in copies)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # No inlined ReturnNode leaks out of the call scope
    assert not _nodes_of_type(tree, tn.ReturnNode)


def test_two_calls_unique_names():

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        y = _add_one(A)
        z = _add_one(B)
        C[:] = y + z

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 2
    return_names = [name for name in tree.containers if '_add_one_ret' in name]
    assert len(return_names) == 2
    assert len(set(return_names)) == 2
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_recursion_falls_back():

    @dace.program
    def recursive(X: dace.float64[N]):
        recursive(X)

    @dace.program
    def caller(A: dace.float64[N]):
        recursive(A)

    # Parsing must terminate (no infinite inlining); the recursive call is
    # deferred to the interpreter. Depending on where the cycle is broken
    # (closure capture, preprocessing, or the inline stack), the callback
    # reason differs — termination plus a callback is the contract.
    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) <= 1
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) >= 1


def test_early_return_inlines():

    @dace.program
    def maybe(X: dace.float64[N]):
        if X[0] > 0.0:
            return 1.0
        return 2.0

    @dace.program
    def caller(A: dace.float64[N]):
        y = maybe(A)
        A[0] = y

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 1
    # Early returns are restructured into tail positions (the fall-through
    # return moves into the else branch) and stripped: exiting the callee
    # coincides with falling off the scope end, so no ReturnNode remains.
    assert not _nodes_of_type(tree, tn.ReturnNode)
    assert len(_nodes_of_type(tree, tn.IfScope)) == 1
    assert len(_nodes_of_type(tree, tn.ElseScope)) == 1
    # Both returns materialize into the same return container
    return_names = [name for name in tree.containers if 'maybe_ret' in name]
    assert len(return_names) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_mixed_return_arity_falls_back():

    @dace.program
    def mixed(X: dace.float64[N]):
        if X[0] > 0.0:
            return 1.0, 2.0
        return 3.0

    @dace.program
    def caller(A: dace.float64[N]):
        y = mixed(A)
        A[0] = 1.0

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.FunctionCallScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert any('inconsistent return arities' in callback.reason for callback in callbacks)


def test_fallthrough_return_falls_back():

    @dace.program
    def sometimes(X: dace.float64[N]):
        if X[0] > 0.0:
            return 1.0

    @dace.program
    def caller(A: dace.float64[N]):
        y = sometimes(A)
        A[0] = 1.0

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.FunctionCallScope)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert any('fall through' in callback.reason for callback in callbacks)


def test_default_argument_specializes():

    @dace.program
    def scale_default(X: dace.float64[N], Y: dace.float64[N], factor: dace.float64 = 3.0):
        Y[:] = X * factor

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        scale_default(A, B)

    tree = nextgen.parse_program(caller)
    assert len(_nodes_of_type(tree, tn.FunctionCallScope)) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert any('3.0' in t.node.code.as_string for t in tasklets)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_two_level_inlining():

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        y = _add_one(A)
        _scale_into(y, B, 2.0)

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 2
    # The second callee reads the first callee's return container
    return_names = [name for name in tree.containers if '_add_one_ret' in name]
    assert len(return_names) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert any(any(memlet.data == return_names[0] for memlet in tasklet.in_memlets.values()) for tasklet in tasklets)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_nested_callee_calls_callee():

    @dace.program
    def middle(X: dace.float64[N], Y: dace.float64[N]):
        _scale_into(X, Y, 4.0)

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        middle(A, B)

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 2
    # The inner scope is nested inside the outer one
    outer = next(scope for scope in scopes if scope.call.callee_name.endswith('middle'))
    assert any(isinstance(child, tn.FunctionCallScope) for child in outer.children)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert any(memlet.data == 'A' for memlet in tasklets[0].in_memlets.values())
    assert tasklets[0].out_memlets['__out'].data == 'B'
    assert '4.0' in tasklets[0].node.code.as_string
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_sdfg_valued_callee():

    @dace.program
    def inner(X: dace.float64[10]):
        X[:] = 5.0

    inner_sdfg = inner.to_sdfg()

    @dace.program
    def caller(A: dace.float64[10]):
        inner_sdfg(A)

    tree = nextgen.parse_program(caller)
    sdfg_calls = _nodes_of_type(tree, tn.SDFGCallNode)
    assert len(sdfg_calls) == 1
    assert sdfg_calls[0].sdfg is not None
    assert 'A' in sdfg_calls[0].call.arguments.values()
    assert not _nodes_of_type(tree, tn.FunctionCallScope)


def test_compiletime_constant_arguments():
    # Mirrors tests/python_frontend/constant_and_keyword_args_test.py::
    # test_constant_proper_use: dace.compiletime arguments specialize the
    # caller and propagate through nested-program calls.
    import numpy as np

    @dace.program
    def callee(scal: dace.compiletime, scal2: dace.compiletime, arr):
        a_bool = scal == 1
        if a_bool:
            arr[:] = arr[:] + scal2

    @dace.program
    def prog(arr, scal: dace.compiletime):
        arr[:] = arr[:] * scal
        callee(scal, 3.0, arr)

    arr = np.ones((12, ), np.float64)
    tree = nextgen.parse_program(prog, arr, 2)
    # Fully specialized: no callbacks, and the dead branch (scal == 1 is
    # False) leaves no conditional computation behind
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(_nodes_of_type(tree, tn.FunctionCallScope)) == 1

    result = np.ones((12, ), np.float64)
    tree.as_sdfg()(arr=result)
    assert np.allclose(result, 2.0)


if __name__ == '__main__':
    test_bare_call_inlines()
    test_return_value_binding()
    test_two_calls_unique_names()
    test_recursion_falls_back()
    test_early_return_inlines()
    test_mixed_return_arity_falls_back()
    test_fallthrough_return_falls_back()
    test_default_argument_specializes()
    test_two_level_inlining()
    test_nested_callee_calls_callee()
    test_sdfg_valued_callee()
    test_compiletime_constant_arguments()
