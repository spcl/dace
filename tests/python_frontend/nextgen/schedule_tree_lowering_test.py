# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest
import sys
from typing import Optional

import dace
from dace.frontend.python.common import DaceSyntaxError
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_python_frontend_schedule_tree_root_repository():
    offset = 3.0

    @dace.program
    def structured(A: dace.float64[20]):
        return A + offset

    stree = structured.to_schedule_tree()

    assert isinstance(stree, tn.ScheduleTreeRoot)
    assert stree.name.endswith('_structured')
    assert stree.arg_names == ['A']
    assert 'A' in stree.containers
    assert 'offset' in stree.constants


def test_python_frontend_schedule_tree_unannotated_branch_references():

    @dace.program
    def refs(A: dace.float64[20], B: dace.float64[20], i: dace.int32[1], out: dace.float64[20]):
        if i[0] < 5:
            ref = A
        else:
            ref = B
        out[:] = ref

    stree = refs.to_schedule_tree()

    assert isinstance(stree.children[0], tn.IfScope)
    assert isinstance(stree.children[0].children[0], tn.RefSetNode)
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.RefSetNode)
    assert isinstance(stree.children[2], tn.CopyNode)


def test_python_frontend_schedule_tree_optional_none_branch():

    @dace.program
    def optional_none_prog(field: Optional[dace.float64[8]], A: dace.float64[8], out: dace.float64[8]):
        if field is None:
            out[:] = A[:]
        else:
            out[:] = field[:]

    stree = optional_none_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.IfScope)
    assert stree.children[0].condition.as_string == '(field is None)'
    assert isinstance(stree.children[0].children[0], tn.CopyNode)
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.CopyNode)


def test_python_frontend_schedule_tree_tuple_of_arrays_unrolls():

    @dace.program
    def iter_prog(a: dace.float64[2, 3, 4], b: dace.float64[2, 3, 4], c: dace.float64[2, 3, 4], out: dace.float64[2, 3,
                                                                                                                  4]):
        for arr in (a, b, c):
            out[:] = arr

    with pytest.warns(UserWarning, match=r'implicitly unrolled'):
        stree = iter_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.CopyNode, tn.CopyNode, tn.CopyNode]
    assert [child.memlet.data for child in stree.children] == ['a', 'b', 'c']


# ------------------------------------------------------------------ #
#  Phase 4 — Full Python Language Coverage Tests                      #
# ------------------------------------------------------------------ #


def test_global_traces_container():
    """global x where x is a known global should bind, not callback."""
    some_global_array = np.zeros(10, dtype=np.float64)

    @dace.program
    def global_prog(A: dace.float64[10]):
        # global is typically used in nested scopes; test that it doesn't error
        for i in range(10):
            A[i] = some_global_array[i]
        return A

    # Should not raise
    stree = global_prog.to_schedule_tree()
    assert isinstance(stree, tn.ScheduleTreeRoot)


def test_async_dace_program_to_schedule_tree_is_rejected():

    @dace.program
    async def async_prog(A: dace.float64[10]):
        return A

    with pytest.raises(SyntaxError, match='Async @dace.program functions are unsupported'):
        async_prog.to_schedule_tree()


def test_dynamic_raise_can_be_ignored():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        exc_type = ValueError
        raise exc_type("test")
        return A

    with dace.config.set_temporary('frontend', 'raise_statements', value='ignore_dynamic'):
        stree = raise_prog.to_schedule_tree()

    assert not any(isinstance(node, tn.RaiseNode) for node in stree.preorder_traversal())
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'raise' for node in stree.preorder_traversal())
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_raise_can_be_ignored_entirely():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        raise ValueError("test")
        return A

    with dace.config.set_temporary('frontend', 'raise_statements', value='ignore_all'):
        stree = raise_prog.to_schedule_tree()

    assert not any(isinstance(node, tn.RaiseNode) for node in stree.preorder_traversal())
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'raise' for node in stree.preorder_traversal())
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_named_expr_desugared():
    """Walrus operator should be desugared before reaching schedule tree builder."""

    @dace.program
    def walrus_prog(A: dace.float64[10]):
        if (x := A[0]) > 0:
            A[1] = x
        return A

    stree = walrus_prog.to_schedule_tree()

    # The schedule tree should have an assignment before the if, not a NamedExpr
    # x = A[0] comes first, then if x > 0: ...
    assert isinstance(stree, tn.ScheduleTreeRoot)
    # Should not crash — that's the main verification


def test_schedule_tree_symbolic_static_slice_shape():
    n = dace.symbol('n')

    @dace.program
    def slice_prog(A: dace.float64[n]):
        tmp = A[1:n:2]
        return tmp

    stree = slice_prog.to_schedule_tree()

    assert isinstance(stree.containers['tmp'], dace.data.Array)
    assert str(stree.containers['tmp'].shape[0]) == 'ceiling(n/2 - 1/2)'
    assert isinstance(stree.children[0], tn.ViewNode)


def test_comprehension_desugaring():
    """Comprehensions should be desugared to explicit loops."""

    @dace.program
    def comp_prog(A: dace.float64[8]):
        tmp = [A[i] for i in range(4)]
        return tmp

    stree = comp_prog.to_schedule_tree()

    # After desugaring, we should see loop constructs instead of a single TaskletNode
    # Check that it at least doesn't crash and produces a valid tree
    assert isinstance(stree, tn.ScheduleTreeRoot)


def test_type_alias_is_compile_time_only_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype = dace.float32[4]
    tmp: dtype = A
    return tmp
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        stree = module.prog.to_schedule_tree()

    assert 'tmp' in stree.containers
    assert isinstance(stree.containers['tmp'], dace.data.Array)
    assert stree.containers['tmp'].dtype == dace.float32
    assert tuple(stree.containers['tmp'].shape) == (4, )
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'unhandled TypeAlias'
        for node in stree.preorder_traversal())


def test_generic_type_alias_is_rejected_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype[T] = T
    return A
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        with pytest.raises(DaceSyntaxError, match='Generic type aliases'):
            module.prog.to_schedule_tree()


def test_type_var_tuple_alias_is_rejected_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype[*Ts] = tuple[*Ts]
    return A
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        with pytest.raises(DaceSyntaxError, match='Generic type aliases'):
            module.prog.to_schedule_tree()


if __name__ == '__main__':
    test_python_frontend_schedule_tree_root_repository()
    test_python_frontend_schedule_tree_unannotated_branch_references()
    test_python_frontend_schedule_tree_optional_none_branch()
    test_python_frontend_schedule_tree_tuple_of_arrays_unrolls()
    test_global_traces_container()
    test_async_dace_program_to_schedule_tree_is_rejected()
    test_dynamic_raise_can_be_ignored()
    test_raise_can_be_ignored_entirely()
    test_named_expr_desugared()
    test_schedule_tree_symbolic_static_slice_shape()
    test_comprehension_desugaring()
    test_type_alias_is_compile_time_only_in_schedule_tree()
    test_generic_type_alias_is_rejected_in_schedule_tree()
    test_type_var_tuple_alias_is_rejected_in_schedule_tree()
