# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for descriptor-property reads (``A.shape[0]``, ``A.ndim``) and the
parallel-range loop aliases (``prange``/``parrange``).

A shape read is a compile-time value, but every consumer downstream of
inference resolves ATTRIBUTES as data — ``A.shape`` names ``A`` — so the value
has to be substituted before anything tries to lower it. That makes a shape
read usable anywhere an ordinary number is: a loop bound, a map range, or one
operand of a larger expression.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_shape_in_range_loop_bounds():

    @dace.program
    def doublefor(A: dace.float64[N, N]):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] *= 5

    a = np.random.rand(20, 20)
    reference = a * 5
    tree = nextgen.parse_program(doublefor, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(A=a, N=20)
    assert np.allclose(a, reference)


def test_shape_inside_an_expression():
    """The read is one operand of a bigger expression, not the whole value."""

    @dace.program
    def scaled(A: dace.float64[N, N]):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] *= A.shape[0]

    a = np.random.rand(20, 20)
    reference = a * 20
    tree = nextgen.parse_program(scaled, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(A=a, N=20)
    assert np.allclose(a, reference)


def test_shape_and_ndim_in_map_ranges():

    @dace.program
    def mapshape(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[0:A.shape[0], 0:A.shape[1]]:
            B[i, j] = A[i, j] * A.ndim

    a, b = np.random.rand(8, 8), np.zeros((8, 8))
    tree = nextgen.parse_program(mapshape, a, b)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The map range is the symbol itself, not a dynamic-range read of A.
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1 and str(maps[0].node.map.range) == '0:N, 0:N'
    tree.as_sdfg().compile()(A=a, B=b, N=8)
    assert np.allclose(b, a * 2)


def test_concrete_shape_folds_to_a_constant():
    """A JIT program with a concrete argument has numeric shapes."""

    @dace.program
    def doublefor(A):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] *= 5

    a = np.random.rand(20, 20)
    reference = a * 5
    tree = nextgen.parse_program(doublefor, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    tree.as_sdfg().compile()(A=a)
    assert np.allclose(a, reference)


def test_parallel_range_is_a_map():
    """``parrange``/``prange`` are shorthand for a one-dimensional map."""

    @dace.program
    def prog(A: dace.float64[10], B: dace.float64[10]):
        for i in parrange(10):  # noqa: F821 -- resolved by the frontend
            B[i] = A[i] * 2

    a, b = np.random.rand(10), np.zeros(10)
    tree = nextgen.parse_program(prog, a, b)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1 and str(maps[0].node.map.range) == '0:10'
    tree.as_sdfg().compile()(A=a, B=b)
    assert np.allclose(b, a * 2)


if __name__ == '__main__':
    test_shape_in_range_loop_bounds()
    test_shape_inside_an_expression()
    test_shape_and_ndim_in_map_ranges()
    test_concrete_shape_folds_to_a_constant()
    test_parallel_range_is_a_map()
