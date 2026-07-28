# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for NumPy basic-indexing result shapes in the next-generation Python
frontend: slice-formed dimensions survive in the bound view's shape even at
size 1 (``a[0:20, 1:2]`` is (20, 1)), while integer-indexed dimensions are
dropped (``a[0:20, 1]`` is (20,)).
"""
import warnings

import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_view_shapes_numpy_semantics():

    @dace.program
    def slices(a: dace.float64[20, 20]):
        c = a[0:20, 1:2]
        s = a[0:20, 1]
        c[0, 0] = 1.0
        s[1] = 2.0

    tree = nextgen.parse_program(slices)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    views = {node.target: node for node in _nodes_of_type(tree, tn.ViewNode)}
    assert set(views) == {'c', 's'}
    # Slice-formed size-1 dimension survives; integer index is dropped.
    assert tuple(tree.containers['c'].shape) == (20, 1)
    assert tuple(tree.containers['s'].shape) == (20, )


def test_single_element_slice_is_array():

    @dace.program
    def one_elem(a: dace.float64[10]):
        c = a[3:4]
        c[0] = 5.0

    tree = nextgen.parse_program(one_elem)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # a[3:4] is a shape-(1,) array in NumPy, not a scalar.
    assert tuple(tree.containers['c'].shape) == (1, )


def test_view_shapes_execution():

    @dace.program
    def write_through(a: dace.float64[20, 20]):
        c = a[0:20, 1:2]
        s = a[0:20, 2]
        c[5, 0] = 42.0
        s[7] = 43.0

    tree = nextgen.parse_program(write_through)
    func = tree.as_sdfg().compile()

    a = np.zeros((20, 20))
    func(a=a)
    reference = np.zeros((20, 20))
    reference[5, 1] = 42.0
    reference[7, 2] = 43.0
    assert np.allclose(a, reference)


def test_view_update_writes_through():
    """
    A whole-name update of a view-bound name writes through the view.

    This used to allocate a fresh container and rebind the name to it BEFORE
    the right-hand side was lowered, so the update read the new, uninitialized
    container and the viewed array was never touched -- silently, with a
    correct-looking schedule tree.
    """

    @dace.program
    def update(a: dace.float64[20]):
        y = a[0:10]
        y += 1

    tree = nextgen.parse_program(update)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # No versioned rebinding: the update targets the view itself.
    assert not [name for name in tree.containers if name.startswith('y_')]

    # Writing per-iteration through a view is not a write conflict: the
    # conversion must not report one for it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sdfg = tree.as_sdfg()
    assert not [w for w in caught if 'write conflict' in str(w.message)]

    func = sdfg.compile()
    a = np.ones(20)
    reference = a.copy()
    reference[0:10] += 1
    func(a=a)
    assert np.allclose(a, reference)


def test_view_computation_writes_through():
    """The same, for a computed value rather than an accumulation."""

    @dace.program
    def scale(a: dace.float64[4, 10], b: dace.float64[2, 10]):
        y = a[0:2, :]
        y = b * 2.0

    tree = nextgen.parse_program(scale)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    func = tree.as_sdfg().compile()
    a = np.zeros((4, 10))
    b = np.arange(20, dtype=np.float64).reshape(2, 10)
    reference = a.copy()
    reference[0:2, :] = b * 2.0
    func(a=a, b=b)
    assert np.allclose(a, reference)


def test_integer_element_read_stays_scalar():

    @dace.program
    def elem(a: dace.float64[10, 10], out: dace.float64[1]):
        v = a[3, 4]
        out[0] = v

    tree = nextgen.parse_program(elem)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # Fully integer-indexed access is a scalar element read, not a view.
    assert not _nodes_of_type(tree, tn.ViewNode)


if __name__ == '__main__':
    test_view_shapes_numpy_semantics()
    test_single_element_slice_is_array()
    test_view_shapes_execution()
    test_view_update_writes_through()
    test_view_computation_writes_through()
    test_integer_element_read_stays_scalar()
