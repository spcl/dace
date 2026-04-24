# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_local_lambda_array_call_devirtualizes():

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):
        f = lambda a, b: a + b
        return f(A, B)

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert any(isinstance(node, tn.MapScope) for node in stree.children)


def test_global_lambda_array_call_devirtualizes():
    f = lambda a, b: a + b

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):
        return f(A, B)

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert any(isinstance(node, tn.MapScope) for node in stree.children)


def test_lambda_capture_devirtualizes():
    offset = 3.0

    @dace.program
    def prog(A: dace.float64[4]):
        f = lambda a: a + offset
        return f(A)

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert any(isinstance(node, tn.MapScope) for node in stree.children)


def test_lambda_body_call_to_dace_program_becomes_call_scope():

    @dace.program
    def callee(A: dace.float64[4], B: dace.float64[4]):
        return A + B

    f = lambda a, b: callee(a, b)

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):
        return f(A, B)

    stree = prog.to_schedule_tree()

    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert calls[0].call.callee_name == 'callee'


def test_lambda_argument_to_nested_program_devirtualizes():

    @dace.program
    def inner(A: dace.float64[4], B: dace.float64[4], f):
        return f(A, B)

    @dace.program
    def outer(A: dace.float64[4], B: dace.float64[4]):
        f = lambda a, b: a + b
        g = f
        return inner(A, B, g)

    stree = outer.to_schedule_tree()

    assert 'f' in stree.containers
    assert 'g' in stree.containers
    assert isinstance(stree.containers['f'].dtype, dace.dtypes.callback)
    assert isinstance(stree.containers['g'].dtype, dace.dtypes.callback)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert any(isinstance(node, tn.MapScope) for node in calls[0].preorder_traversal())


def test_external_lambda_argument_to_nested_program_stays_callback_typed():
    external = eval('lambda a, b: a + b')

    @dace.program
    def inner(A: dace.float64, B: dace.float64, f):
        return f(A, B)

    @dace.program
    def outer(A: dace.float64, B: dace.float64):
        f = external
        return inner(A, B, f)

    stree = outer.to_schedule_tree()

    assert 'f' in stree.containers
    assert isinstance(stree.containers['f'], dace.data.Scalar)
    assert isinstance(stree.containers['f'].dtype, dace.dtypes.callback)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert 'tasklet(f[0], A[0], B[0])' in calls[0].as_string()
    assert 'assign __stree_retval = __stree_tmp' in calls[0].as_string()


def test_runtime_callback_argument_passes_through_nested_program():
    external = eval('lambda a, b: a + b')

    @dace.program
    def inner(A: dace.float64, B: dace.float64, f):
        return f(A, B)

    @dace.program
    def outer(A: dace.float64, B: dace.float64, f):
        return inner(A, B, f)

    stree = outer.to_schedule_tree(1.0, 2.0, external)

    assert 'f' in stree.containers
    assert isinstance(stree.containers['f'], dace.data.Scalar)
    assert isinstance(stree.containers['f'].dtype, dace.dtypes.callback)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert 'tasklet(f[0], A[0], B[0])' in calls[0].as_string()
    assert 'assign __stree_retval = __stree_tmp' in calls[0].as_string()
