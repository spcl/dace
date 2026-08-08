# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for CHAINED assignment targets (``A[:, 1:5][:, 0:2] += 5``,
``L[i + 1][i] = 1.0``), which canonicalization now reduces to the single-level
target grammar by naming the base.

The base of such a target must ALIAS what it names, or the write lands in a
disposable copy and is silently discarded. Canonicalization cannot tell which
bases can (that depends on their type), so it records the requirement and
lowering enforces it -- the two halves are tested here: the aliasing forms
execute and write through, and the copying forms (a computed ``A.T``, an
advanced-index gather) stay in ONE interpreter callback covering both the
hoisted base and the write, which is what preserves their semantics.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_chained_slice_target_writes_through():
    """``A[:, 1:5][:, 0:2] += 5``: the base binds a view, so the accumulation
    reaches the source array."""

    @dace.program
    def chained(A: dace.float64[5, 5]):
        A[:, 1:5][:, 0:2] += 5

    tree = nextgen.parse_program(chained, np.zeros((5, 5)))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert _nodes_of_type(tree, tn.ViewNode)

    A = np.random.rand(5, 5)
    expected = np.copy(A)
    expected[:, 1:5][:, 0:2] += 5
    tree.as_sdfg().compile()(A=A)
    assert np.allclose(A, expected)


def test_chained_integer_index_target_writes_through():
    """``L[i + 1][i] = 1.0`` under a loop (the shape from
    ``tests/codegen/allocation_lifetime_test.py::test_persistent_loop_bound``):
    an integer-indexed base drops a dimension, and the row view still aliases."""

    @dace.program
    def rows(L: dace.float64[10, 10], count: dace.int64):
        for i in range(0, count):
            L[i + 1][i] = 1.0

    tree = nextgen.parse_program(rows, np.zeros((10, 10)), np.int64(4))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    L = np.zeros((10, 10))
    expected = np.copy(L)
    for i in range(4):
        expected[i + 1][i] = 1.0
    tree.as_sdfg().compile()(L=L, count=np.int64(4))
    assert np.allclose(L, expected)


def test_chained_flat_target_writes_through():
    """``A.flat[10:15][0:2] += 5``: a contiguous flatiter aliases its source, so
    both the attribute materialization and the chained base stay views."""

    @dace.program
    def chained_flat(A: dace.float64[5, 5, 5]):
        A.flat[10:15][0:2] += 5

    tree = nextgen.parse_program(chained_flat, np.zeros((5, 5, 5)))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    A = np.random.rand(5, 5, 5)
    expected = np.copy(A)
    expected.reshape(-1)[10:12] += 5
    tree.as_sdfg().compile()(A=A)
    assert np.allclose(A, expected)


def test_computed_attribute_base_stays_in_one_callback():
    """``A.T[0:2][0] = 5.0``: the transpose is COMPUTED (a fresh array), so the
    base cannot be named without discarding the write. Both statements must end
    up in the same callback, where the interpreter's own ``A.T`` view writes
    through as NumPy does."""

    @dace.program
    def transposed(A: dace.float64[4, 4]):
        A.T[0:2][0] = 5.0

    tree = nextgen.parse_program(transposed, np.zeros((4, 4)))
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    code = callbacks[0].code.as_string
    assert 'A.T[0:2]' in code and '= 5.0' in code


def test_gather_base_target_stays_in_one_callback():
    """``A[ind][0] = 5.0``: an advanced-index base is a gathered COPY in NumPy
    too, so the write does not reach ``A`` either way -- but it has to be the
    interpreter that decides that, not a lowering that writes into a container
    of our own."""

    @dace.program
    def gathered(A: dace.float64[8], ind: dace.int32[3]):
        A[ind][0] = 5.0

    tree = nextgen.parse_program(gathered, np.zeros(8), np.zeros(3, dtype=np.int32))
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert 'assign-target' in callbacks[0].reason
    code = callbacks[0].code.as_string
    assert 'A[ind]' in code and '= 5.0' in code


if __name__ == '__main__':
    test_chained_slice_target_writes_through()
    test_chained_integer_index_target_writes_through()
    test_chained_flat_target_writes_through()
    test_computed_attribute_base_stays_in_one_callback()
    test_gather_base_target_stays_in_one_callback()
