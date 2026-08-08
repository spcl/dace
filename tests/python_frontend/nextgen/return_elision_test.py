# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the elision of the copy that materializes a return value.

A returned program-local array takes the ``__return`` container's name instead
of being copied into it, so the program allocates one buffer rather than two.
The elision has to decline whenever the rename would change what the caller
observes -- a returned argument, a returned view, a value still written after
the return container was filled -- and every test here pins one of those.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N', dtype=dace.int64)


def _copies_into_return(tree: tn.ScheduleTreeRoot) -> bool:
    """Whether the tree still materializes a return value through a copy."""
    return any(
        isinstance(node, tn.CopyNode) and node.target.startswith('__return') for node in tree.preorder_traversal())


def test_returned_local_array_takes_the_return_name():

    @dace.program
    def build(A: dace.float64[20]):
        b = np.zeros([20], dtype=np.float64)
        for i in dace.map[0:20]:
            b[i] = A[i] * 2
        return b

    tree = nextgen.parse_program(build)
    assert not _copies_into_return(tree)
    assert 'b' not in tree.containers
    assert tuple(tree.containers['__return'].shape) == (20, )
    assert not tree.containers['__return'].transient

    A = np.random.rand(20)
    assert np.allclose(build(A.copy()), A * 2)


def test_each_element_of_a_returned_tuple_is_elided():

    @dace.program
    def two(A: dace.float64[10]):
        b = np.zeros([10], dtype=np.float64)
        c = np.zeros([10], dtype=np.float64)
        for i in dace.map[0:10]:
            b[i] = A[i] + 1
            c[i] = A[i] - 1
        return b, c

    tree = nextgen.parse_program(two)
    assert not _copies_into_return(tree)
    assert 'b' not in tree.containers and 'c' not in tree.containers

    A = np.random.rand(10)
    first, second = two(A.copy())
    assert np.allclose(first, A + 1)
    assert np.allclose(second, A - 1)


def test_a_returned_argument_is_still_copied():
    """The caller's array and the returned one are separate objects, so the
    copy has to stay: renaming would hand back the argument's own memory."""

    @dace.program
    def identity(A: dace.float64[20]):
        return A

    tree = nextgen.parse_program(identity)
    assert _copies_into_return(tree)

    A = np.random.rand(20)
    returned = identity(A.copy())
    assert np.allclose(returned, A)
    returned[0] = 12345.0
    assert A[0] != 12345.0


def test_a_returned_slice_is_still_copied():
    """A subscript returns part of a container, not the container."""

    @dace.program
    def half(A: dace.float64[20]):
        return A[0:10]

    tree = nextgen.parse_program(half)
    assert _copies_into_return(tree)

    A = np.random.rand(20)
    assert np.allclose(half(A.copy()), A[0:10])


def test_a_returned_scalar_is_still_copied():
    """Only whole arrays are interchangeable with a return container."""

    @dace.program
    def total(A: dace.float64[N]):
        s = np.sum(A)
        return s

    tree = nextgen.parse_program(total)
    assert _copies_into_return(tree)

    A = np.random.rand(16)
    assert np.allclose(total(A.copy(), N=16), np.sum(A))


def test_a_conditional_return_is_not_elided():
    """Two containers reach one ``__return``; renaming one would leave the
    other writing a container that no longer exists."""

    @dace.program
    def branch(A: dace.float64[20], flag: dace.int32):
        if flag > 0:
            b = A + 1
            return b
        c = A - 1
        return c

    tree = nextgen.parse_program(branch)
    assert _copies_into_return(tree)

    A = np.random.rand(20)
    assert np.allclose(branch(A.copy(), 1), A + 1)
    assert np.allclose(branch(A.copy(), 0), A - 1)


def test_the_elided_container_keeps_its_layout():
    """``strides=`` is descriptor state the rename must not drop."""

    @dace.program
    def strided():
        A = dace.ndarray((2, 2), dtype=dace.int32, strides=(1, 2))
        for i, j in dace.map[0:2, 0:2]:
            A[i, j] = i * 2 + j
        return A

    tree = nextgen.parse_program(strided)
    assert not _copies_into_return(tree)
    assert tuple(tree.containers['__return'].strides) == (1, 2)


if __name__ == '__main__':
    test_returned_local_array_takes_the_return_name()
    test_each_element_of_a_returned_tuple_is_elided()
    test_a_returned_argument_is_still_copied()
    test_a_returned_slice_is_still_copied()
    test_a_returned_scalar_is_still_copied()
    test_a_conditional_return_is_not_elided()
    test_the_elided_container_keeps_its_layout()
