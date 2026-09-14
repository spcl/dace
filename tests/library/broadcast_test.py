# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`Broadcast` library node (Fortran ``SPREAD``)."""
import numpy as np

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


if __name__ == '__main__':
    test_broadcast_1d_to_2d_dim1()
    test_broadcast_1d_to_2d_dim2()
    print('Broadcast tests PASS')
