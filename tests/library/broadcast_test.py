# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`Broadcast` library node -- Fortran ``SPREAD`` and the
right-aligned NumPy rule behind ``np.broadcast_to``."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import Broadcast


def _build(src_shape, dst_shape, dim, dtype):
    sdfg = dace.SDFG(f"broadcast_{dim}")
    sdfg.add_array("src", list(src_shape), dtype)
    sdfg.add_array("dst", list(dst_shape), dtype)
    state = sdfg.add_state()
    node = Broadcast("broadcast", dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("src"), None, node, '_src', dace.Memlet.from_array("src", sdfg.arrays["src"]))
    state.add_edge(node, '_dst', state.add_write("dst"), None, dace.Memlet.from_array("dst", sdfg.arrays["dst"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_broadcast_1d_to_2d_dim1():
    """SPREAD([1,2,3], DIM=1, NCOPIES=2) -> [[1,2,3], [1,2,3]]."""
    src = np.array([1.0, 2.0, 3.0])
    dst = np.zeros((2, 3))
    sdfg = _build(src.shape, dst.shape, 1, dace.float64)
    sdfg(src=src, dst=dst)
    np.testing.assert_array_equal(dst, np.broadcast_to(src, (2, 3)))


def test_broadcast_1d_to_2d_dim2():
    """SPREAD([1,2,3], DIM=2, NCOPIES=4) -> each entry replicated columnwise."""
    src = np.array([1.0, 2.0, 3.0])
    dst = np.zeros((3, 4))
    sdfg = _build(src.shape, dst.shape, 2, dace.float64)
    sdfg(src=src, dst=dst)
    expected = np.broadcast_to(src.reshape(3, 1), (3, 4))
    np.testing.assert_array_equal(dst, expected)


@pytest.mark.parametrize('src_shape,dst_shape', [
    ((3, ), (2, 3)),
    ((3, 1), (3, 4)),
    ((1, 4), (3, 4)),
    ((1, ), (2, 5)),
    ((2, 3), (4, 2, 3)),
])
def test_broadcast_numpy_rule(src_shape, dst_shape):
    """``dim=None`` must agree with NumPy on every shape NumPy accepts -- including the two
    SPREAD cannot say: stretching an existing extent-1 axis, and adding more than one axis."""
    src = np.arange(np.prod(src_shape), dtype=np.float64).reshape(src_shape).copy()
    dst = np.zeros(dst_shape)
    sdfg = _build(src_shape, dst_shape, None, dace.float64)
    sdfg(src=src, dst=dst)
    np.testing.assert_array_equal(dst, np.broadcast_to(src, dst_shape))


def test_broadcast_numpy_rule_rejects_a_mismatch():
    """A source axis that is neither 1 nor equal to the destination's cannot broadcast; catching
    it in validate() is what keeps the expansion from emitting an out-of-bounds read."""
    with pytest.raises(ValueError, match='neither is 1'):
        _build((3, ), (2, 5), None, dace.float64)


def test_broadcast_to_frontend():
    """``np.broadcast_to`` in a dace.program must reach the library node and match NumPy."""

    @dace.program
    def bcast(a: dace.float64[3, 1], out: dace.float64[3, 4]):
        out[:] = np.broadcast_to(a, (3, 4))

    sdfg = bcast.to_sdfg(simplify=False)
    assert len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Broadcast)]) == 1

    a = np.random.randn(3, 1)
    out = np.zeros((3, 4))
    bcast(a=a, out=out)
    np.testing.assert_array_equal(out, np.broadcast_to(a, (3, 4)))


if __name__ == '__main__':
    test_broadcast_1d_to_2d_dim1()
    test_broadcast_1d_to_2d_dim2()
    for shapes in [((3, ), (2, 3)), ((3, 1), (3, 4)), ((1, 4), (3, 4)), ((1, ), (2, 5)), ((2, 3), (4, 2, 3))]:
        test_broadcast_numpy_rule(*shapes)
    test_broadcast_numpy_rule_rejects_a_mismatch()
    test_broadcast_to_frontend()
    print('Broadcast tests PASS')
