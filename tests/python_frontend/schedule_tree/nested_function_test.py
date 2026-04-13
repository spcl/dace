# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_simple_nested_function_becomes_call_scope():

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):

        def helper(x, y):
            return x + y

        return helper(A, B)

    stree = prog.to_schedule_tree()

    assert 'helper' in stree.containers
    assert isinstance(stree.containers['helper'].dtype, dace.dtypes.callback)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert calls[0].call.callee_name == 'helper'
    assert any(isinstance(node, tn.MapScope) for node in calls[0].preorder_traversal())


def test_nested_function_capture_becomes_call_scope():
    offset = 2.0

    @dace.program
    def prog(A: dace.float64[4]):

        def helper(x):
            return x + offset

        return helper(A)

    stree = prog.to_schedule_tree()

    assert 'helper' in stree.containers
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert calls[0].call.callee_name == 'helper'
    assert any(isinstance(node, tn.MapScope) for node in calls[0].preorder_traversal())


def test_nested_function_body_call_to_dace_program_becomes_call_scope():

    @dace.program
    def callee(A: dace.float64[4], B: dace.float64[4]):
        return A + B

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):

        def helper(x, y):
            return callee(x, y)

        return helper(A, B)

    stree = prog.to_schedule_tree()

    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 2
    assert [call.call.callee_name for call in calls] == ['helper', 'callee']


def test_nested_function_argument_to_nested_program_becomes_call_regions():

    @dace.program
    def inner(A: dace.float64[4], B: dace.float64[4], f):
        return f(A, B)

    @dace.program
    def outer(A: dace.float64[4], B: dace.float64[4]):

        def helper(x, y):
            return x + y

        alias = helper
        return inner(A, B, alias)

    stree = outer.to_schedule_tree()

    assert 'helper' in stree.containers
    assert 'alias' in stree.containers
    assert isinstance(stree.containers['helper'].dtype, dace.dtypes.callback)
    assert isinstance(stree.containers['alias'].dtype, dace.dtypes.callback)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 2
    assert [call.call.callee_name for call in calls] == ['inner', 'helper']
    assert any(isinstance(node, tn.MapScope) for node in calls[1].preorder_traversal())


def test_multistatement_nested_function_becomes_call_scope():

    @dace.program
    def prog(A: dace.float64[4]):

        def helper(x):
            y = x + 1
            return y

        return helper(A)

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert calls[0].call.callee_name == 'helper'


def test_decorated_nested_function_stays_callback():

    def passthrough(fn):
        return fn

    @dace.program
    def prog(A: dace.float64[4]):

        @passthrough
        def helper(x):
            return x + 1

        return helper(A)

    stree = prog.to_schedule_tree()

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert len(callbacks) == 1
    assert callbacks[0].reason == 'nested function'
