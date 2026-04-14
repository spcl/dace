# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest
from dace.frontend.python.common import DaceSyntaxError
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


def test_nested_function_nonlocal_rebinds_outer_reference():

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):
        tmp = A

        def helper():
            nonlocal tmp
            tmp = B

        helper()
        return tmp

    stree = prog.to_schedule_tree()

    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    refsets = [node for node in calls[0].preorder_traversal() if isinstance(node, tn.RefSetNode)]
    assert len(refsets) == 1
    assert refsets[0].target == 'tmp'


def test_nested_function_nonlocal_external_capture_is_added_to_closure():

    def make_prog():
        captured = np.ones(4, dtype=np.float64)

        @dace.program
        def prog(A: dace.float64[4]):

            def helper():
                nonlocal captured
                return captured + A

            return helper()

        return prog

    stree = make_prog().to_schedule_tree()

    assert 'captured' in stree.containers
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert any(isinstance(node, tn.MapScope) for node in calls[0].preorder_traversal())


def test_nested_function_global_capture_is_added_to_closure():
    globals()['__schedule_tree_nested_global_capture'] = np.ones(4, dtype=np.float64)

    try:

        @dace.program
        def prog(A: dace.float64[4]):

            def helper():
                global __schedule_tree_nested_global_capture
                return __schedule_tree_nested_global_capture + A

            return helper()

        stree = prog.to_schedule_tree()

    finally:
        del globals()['__schedule_tree_nested_global_capture']

    assert '__schedule_tree_nested_global_capture' in stree.containers
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.FunctionCallScope)]
    assert len(calls) == 1
    assert any(isinstance(node, tn.MapScope) for node in calls[0].preorder_traversal())


def test_nested_function_with_nonlocal_callback_fallback_is_rejected():

    def passthrough(fn):
        return fn

    @dace.program
    def prog(A: dace.float64[4], B: dace.float64[4]):
        tmp = A

        @passthrough
        def helper():
            nonlocal tmp
            tmp = B

        helper()
        return tmp

    with pytest.raises(DaceSyntaxError, match='nonlocal'):
        prog.to_schedule_tree()


def test_nested_function_with_global_callback_fallback_is_allowed():
    globals()['__schedule_tree_callback_global'] = 2.0

    def passthrough(fn):
        return fn

    try:

        @dace.program
        def prog(A: dace.float64[4]):

            @passthrough
            def helper(x):
                global __schedule_tree_callback_global
                return x + __schedule_tree_callback_global

            return helper(A)

        stree = prog.to_schedule_tree()

    finally:
        del globals()['__schedule_tree_callback_global']

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert len(callbacks) == 1
    assert callbacks[0].reason == 'nested function'


def test_nested_function_with_global_callback_fallback_is_rejected_if_enclosing_program_uses_global():
    globals()['__schedule_tree_callback_global'] = 5

    def passthrough(fn):
        return fn

    try:

        @dace.program
        def prog(A: dace.float64[4]):

            @passthrough
            def helper(x):
                global __schedule_tree_callback_global
                __schedule_tree_callback_global = 6

            helper(A)
            return __schedule_tree_callback_global - 1

        with pytest.raises(DaceSyntaxError, match='used in the enclosing program'):
            prog.to_schedule_tree()

    finally:
        del globals()['__schedule_tree_callback_global']


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
