# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the general frontend view path (``dispatch._lower_view_call``): a
registry call whose RESULT is a view of one of its own data arguments binds a
``ViewNode``, because a view binding is frontend state a deferred
``ReplacementCallNode`` cannot represent.

The path is not keyed to any call name -- it trial-runs the replacement and
takes what it produced -- so the tests cover both the dtype-reinterpreting
``A.view(dtype)`` and a view-returning method registered here at runtime,
which must lower with no frontend change at all.

These check EXECUTION and write-through: a view that lowers to a fresh
container instead of an alias is callback-free and silently wrong, which is
exactly what the tests under ``tests/numpy`` (they compile the CLASSIC
frontend's SDFG) cannot see.
"""
import numpy as np
import pytest

import dace
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_reinterpret_smaller_writes_through():
    """``A.view(dace.int16)`` on an int32 array: twice the elements, aliasing
    the same storage, so writes reach the source."""

    @dace.program
    def reint(A: dace.int32[10]):
        C = A.view(dace.int16)
        C[:] += 1

    tree = nextgen.parse_program(reint, np.zeros(10, dtype=np.int32))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)
    views = _nodes_of_type(tree, tn.ViewNode)
    assert len(views) == 1 and tuple(views[0].view_desc.shape) == (20, )

    A = np.random.randint(0, 262144, size=[10], dtype=np.int32)
    expected = np.copy(A)
    expected.view(np.int16)[:] += 1
    tree.as_sdfg().compile()(A=A)
    assert np.array_equal(A, expected)


def test_reinterpret_larger_writes_through():
    """The other direction, ``A.view(dace.int32)`` on an int16 array: half the
    elements."""

    @dace.program
    def reint(A: dace.int16[10]):
        C = A.view(dace.int32)
        C[:] += 1

    tree = nextgen.parse_program(reint, np.zeros(10, dtype=np.int16))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    views = _nodes_of_type(tree, tn.ViewNode)
    assert len(views) == 1 and tuple(views[0].view_desc.shape) == (5, )

    A = np.random.randint(0, 32767, size=[10], dtype=np.int16)
    expected = np.copy(A)
    expected.view(np.int32)[:] += 1
    tree.as_sdfg().compile()(A=A)
    assert np.array_equal(A, expected)


def test_symbolic_reinterpretation_divides_with_int_floor():
    """The emitted descriptor is the one the replacement computed, not one
    re-derived by inference: ``view()`` divides with ``symbolic.int_floor``,
    and a plain ``//`` would build a sympy ``floor`` whose argument ``sym2cpp``
    prints without the floor (see ``tests/numpy/reshape_test.py``
    ``::test_reinterpret_symbolic_stride_uses_int_floor``)."""

    @dace.program
    def reint(A: dace.data.Array(dace.int16, [N, 4], strides=[2 * N + 1, 1], total_size=N * (2 * N + 1))):
        C = A.view(dace.int32)
        C[:] += 1

    tree = nextgen.parse_program(reint)
    views = _nodes_of_type(tree, tn.ViewNode)
    assert len(views) == 1
    descriptor = views[0].view_desc
    expressions = [str(e) for e in (*descriptor.shape, *descriptor.strides, descriptor.total_size)]
    assert any('int_floor' in e for e in expressions), expressions
    assert not any('floor' in e.replace('int_floor', '') for e in expressions), expressions


def test_invalid_reinterpretation_falls_back():
    """A reinterpretation NumPy rejects (float32 storage is not divisible into
    float64 elements) is not a view this path can build: the trial raises, the
    call falls through, and the statement degrades to a callback rather than
    silently binding a view that reads out of bounds."""

    @dace.program
    def reint(A: dace.float32[5]):
        C = A.view(dace.float64)
        C[:] += 1

    tree = nextgen.parse_program(reint, np.zeros(5, dtype=np.float32))
    assert not _nodes_of_type(tree, tn.ViewNode)
    assert _nodes_of_type(tree, tn.PythonCallbackNode)


@pytest.mark.parametrize('shape', [(2, 4), (2, 2)])
def test_view_disagreeing_with_its_window_falls_back(shape):
    """A view that provably does not hold the bytes of the subset it aliases
    is refused even when the replacement builds one: ``numpy.reshape`` does not
    check the element count, so an 8- or 4-element view of a 6-element array
    would read out of bounds / leave part of the window unreachable. NumPy
    raises for both, and the callback preserves that."""

    @dace.program
    def mismatched_reshape(A: dace.float64[6]):
        b = np.reshape(A, shape)
        return b + 1.0

    tree = nextgen.parse_program(mismatched_reshape, np.zeros(6))
    assert not _nodes_of_type(tree, tn.ViewNode)
    assert _nodes_of_type(tree, tn.PythonCallbackNode)


def test_newly_registered_view_replacement_needs_no_frontend_change():
    """Nothing in the path is keyed to a call name: a method registered here,
    returning a view of its receiver, lowers as a view binding on the strength
    of what the trial run produced."""

    def _firsthalf(pv, sdfg, state, arr: str) -> str:
        descriptor = sdfg.arrays[arr]
        half = descriptor.shape[0] // 2
        name, view_descriptor = sdfg.add_view(arr, [half], descriptor.dtype, find_new_name=True)
        pv.views[name] = (arr, dace.Memlet(data=arr, subset=f'0:{half}'))
        return name

    @dace.program
    def half_view(A: dace.float64[8]):
        h = A.firsthalf()  # noqa: F821 -- registered below, resolved through the registry
        h[:] = 1.0

    # Registered for the duration of this test only: the registry is global, and
    # a leftover entry shows up as an unpaired registration in
    # ``schedule_tree/registry_parity_test.py``.
    oprepo.replaces_method('Array', 'firsthalf')(_firsthalf)
    try:
        tree = nextgen.parse_program(half_view, np.zeros(8))
    finally:
        del oprepo.Replacements._method_rep[('Array', 'firsthalf')]
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    views = _nodes_of_type(tree, tn.ViewNode)
    assert len(views) == 1 and tuple(views[0].view_desc.shape) == (4, )

    A = np.zeros(8)
    tree.as_sdfg().compile()(A=A)
    assert np.array_equal(A, np.array([1.0] * 4 + [0.0] * 4))


if __name__ == '__main__':
    test_reinterpret_smaller_writes_through()
    test_reinterpret_larger_writes_through()
    test_symbolic_reinterpretation_divides_with_int_floor()
    test_invalid_reinterpretation_falls_back()
    test_view_disagreeing_with_its_window_falls_back((2, 4))
    test_view_disagreeing_with_its_window_falls_back((2, 2))
    test_newly_registered_view_replacement_needs_no_frontend_change()
