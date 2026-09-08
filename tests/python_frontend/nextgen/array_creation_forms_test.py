# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for array-creation call forms whose DESCRIPTOR is decided by the
replacement registry's inference: ``numpy.array`` of an existing container,
``strides=`` on ``dace.ndarray``, a keyword ``fill_value``, and nested
compile-time sequence literals.

These share one failure mode: the creation mechanism allocates from the
inferred descriptor and then only initializes contents, so anything the
inference does not model is silently dropped — which is why the lowering path
refuses the call rather than guessing, and why the fixes belong in inference.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_array_of_a_container():
    """``numpy.array(A)`` types from A's descriptor, not from the string "A"."""

    @dace.program
    def array_of_container(A: dace.float64[4, 5]):
        return np.array(A)

    a = np.random.rand(4, 5)
    tree = nextgen.parse_program(array_of_container, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(A=a)
    assert np.allclose(np.asarray(result).reshape(a.shape), a)


def test_array_of_a_constant_with_dtype():

    constant = np.random.rand(4, 4).astype(np.float32)

    @dace.program
    def array_of_constant():
        return np.array(constant, dtype=np.float32)

    tree = nextgen.parse_program(array_of_constant)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The closure constant is the call's data argument, and the result is
    # typed from it rather than from the string that names it.
    assert tuple(tree.containers['__return'].shape) == constant.shape
    assert tree.containers['__return'].dtype == dace.float32


def test_strides_reach_the_descriptor():
    """``strides=`` is descriptor state: inference has to carry it, or the
    allocated container would silently have the default layout."""

    @dace.program
    def strided_local():
        A = dace.ndarray((2, 2), dtype=dace.int32, strides=(1, 2))
        for i, j in dace.map[0:2, 0:2]:
            A[i, j] = i * 2 + j
        return A

    tree = nextgen.parse_program(strided_local)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # ``A`` is returned, so it carries the return container's name
    assert tuple(tree.containers['__return'].strides) == (1, 2)
    # ``return A`` hands back A's own layout, so the return container keeps it.
    assert tuple(tree.containers['__return'].strides) == (1, 2)
    result = tree.as_sdfg().compile()()
    assert result.strides == (4, 8)
    assert result.tolist() == [[0, 1], [2, 3]]


def test_full_like_with_keyword_fill_value():
    """NumPy names the parameter ``fill_value``, so it may arrive as one."""

    @dace.program
    def full_like_keyword(A: dace.complex64[3, 4]):
        return np.full_like(A, fill_value=5)

    a = np.zeros((3, 4), dtype=np.complex64)
    tree = nextgen.parse_program(full_like_keyword, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(A=a)
    assert np.allclose(np.asarray(result).reshape(a.shape), np.full_like(a, fill_value=5))


def test_nested_sequence_literal():
    """A sequence literal whose elements are themselves sequences is still
    compile-time, one dimension down."""

    @dace.program
    def nested_literal():
        return np.array([1, 2, 3]) * ((4, 5, 6), [1, 2, 3])

    tree = nextgen.parse_program(nested_literal)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()()
    expected = np.array([1, 2, 3]) * ((4, 5, 6), [1, 2, 3])
    assert np.allclose(np.asarray(result).reshape(expected.shape), expected)


def test_literal_from_runtime_elements():
    """
    ``numpy.array([A[0], B[i]])`` — a literal whose ELEMENTS are runtime
    values. The literal cannot be evaluated to get a shape and dtype, so both
    come from the element descriptors; the registry implementation then fills
    the container one element at a time.
    """

    @dace.program
    def dynamic_literal(A: dace.float64[1], B: dace.float64[4], i: dace.int32):
        return np.array([A[0], B[i]], dtype=np.float64)

    A, B, i = np.random.rand(1), np.random.rand(4), np.int32(2)
    tree = nextgen.parse_program(dynamic_literal, A, B, i)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(A=A, B=B, i=i)
    assert np.allclose(result, np.array([A[0], B[i]]))


def test_literal_mixing_runtime_and_constant_elements():
    """Constants and runtime reads in one literal."""

    @dace.program
    def mixed(A: dace.float64[4]):
        return np.array([A[0], 3.0, A[2]], dtype=np.float64)

    A = np.random.rand(4)
    tree = nextgen.parse_program(mixed, A)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(A=A)
    assert np.allclose(result, np.array([A[0], 3.0, A[2]]))


if __name__ == '__main__':
    test_array_of_a_container()
    test_array_of_a_constant_with_dtype()
    test_strides_reach_the_descriptor()
    test_full_like_with_keyword_fill_value()
    test_nested_sequence_literal()
    test_literal_from_runtime_elements()
    test_literal_mixing_runtime_and_constant_elements()
