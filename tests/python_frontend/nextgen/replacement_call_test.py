# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for deferred replacement expansion: calls with both a registered
replacement and a descriptor inference entry emit a
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`,
which ``tree_to_sdfg`` expands through the classic replacement implementation
(reusing it instead of reimplementing each call as tree emission).
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_sum_tuple_axis_structure():

    @dace.program
    def prog(A: dace.float64[4, 5, 6]):
        return np.sum(A, axis=(0, 2))

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1
    assert calls[0].qualname == 'numpy.sum'
    assert calls[0].keyword_arguments == {'axis': (0, 2)}
    assert calls[0].data_arguments == {'A'}


def test_sum_tuple_axis_execution():

    @dace.program
    def prog(A: dace.float64[4, 5, 6]):
        return np.sum(A, axis=(0, 2))

    tree = nextgen.parse_program(prog)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5, 6)
    assert np.allclose(func(A=A), A.sum(axis=(0, 2)))


def test_max_tuple_axis_execution():

    @dace.program
    def prog(A: dace.float64[4, 5, 6]):
        return np.max(A, axis=(1, 2))

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5, 6)
    assert np.allclose(func(A=A), A.max(axis=(1, 2)))


def test_mean_execution():
    """numpy.mean chains two replacements through NestedCall states."""

    @dace.program
    def prog(A: dace.float64[4, 5, 6]):
        return np.mean(A, axis=(0, 1))

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5, 6)
    assert np.allclose(func(A=A), A.mean(axis=(0, 1)))


def test_transpose_execution():

    @dace.program
    def prog(A: dace.float64[4, 5, 6]):
        return np.transpose(A, (2, 0, 1))

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5, 6)
    assert np.allclose(func(A=A), np.transpose(A, (2, 0, 1)))


def test_scalar_axis_keeps_wcr_mechanism():
    """Single constant axes stay on the dedicated WCR-map mechanism (no
    ReplacementCallNode)."""

    @dace.program
    def prog(A: dace.float64[4, 5]):
        return np.sum(A, axis=1)

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)


def test_concatenate_sequence_of_containers_structure():
    """A static sequence whose elements are data containers (e.g. the
    ``(A, B)`` in ``numpy.concatenate((A, B))``) passes through as a list of
    container names, rather than being rejected as an un-representable
    by-value argument."""

    @dace.program
    def prog(A: dace.float64[4, 5], B: dace.float64[4, 5]):
        return np.concatenate((A, B))

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1
    assert calls[0].qualname == 'numpy.concatenate'
    assert calls[0].arguments == [('A', 'B')]
    assert calls[0].data_arguments == {'A', 'B'}


def test_concatenate_sequence_of_containers_execution():

    @dace.program
    def prog(A: dace.float64[4, 5], B: dace.float64[4, 5]):
        return np.concatenate((A, B))

    tree = nextgen.parse_program(prog)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5)
    B = np.random.rand(4, 5)
    assert np.allclose(func(A=A, B=B), np.concatenate((A, B)))


def test_hstack_list_of_containers_execution():

    @dace.program
    def prog(A: dace.float64[4, 5], B: dace.float64[4, 5]):
        return np.hstack([A, B])

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    func = tree.as_sdfg().compile()
    A = np.random.rand(4, 5)
    B = np.random.rand(4, 5)
    assert np.allclose(func(A=A, B=B), np.hstack([A, B]))


def test_nonviable_replacement_falls_back():
    """numpy.reshape has both registrations but records view bindings, which
    deferred expansion cannot honor: the build-time viability trial rejects it
    and the call stays a callback (never an expansion-time crash)."""

    @dace.program
    def prog(A: dace.float64[6]):
        b = np.reshape(A, (2, 3))
        return b + 1.0

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)
    assert _nodes_of_type(tree, tn.PythonCallbackNode)


if __name__ == '__main__':
    test_sum_tuple_axis_structure()
    test_sum_tuple_axis_execution()
    test_max_tuple_axis_execution()
    test_mean_execution()
    test_transpose_execution()
    test_scalar_axis_keeps_wcr_mechanism()
    test_concatenate_sequence_of_containers_structure()
    test_concatenate_sequence_of_containers_execution()
    test_hstack_list_of_containers_execution()
    test_nonviable_replacement_falls_back()
