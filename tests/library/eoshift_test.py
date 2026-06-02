# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`EOShift` library node (Fortran ``EOSHIFT``)."""
import numpy as np

import dace
from dace.libraries.standard.nodes import EOShift


def _build(in_shape, dtype, dim=1):
    sdfg = dace.SDFG(f"eoshift_dim{dim}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("out", list(in_shape), dtype)
    sdfg.add_symbol("__shift", dace.int64)
    state = sdfg.add_state()
    node = EOShift("eoshift", dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_out', state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_eoshift_1d_positive_shift_fills_zero():
    """EOSHIFT([1,2,3,4,5], shift=2) -> [3,4,5,0,0]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sdfg = _build(x.shape, dace.float64)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=2)
    np.testing.assert_array_equal(out, np.array([3, 4, 5, 0, 0], dtype=np.float64))


def test_eoshift_1d_negative_shift_fills_zero():
    """EOSHIFT([1,2,3,4,5], shift=-1) -> [0,1,2,3,4]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sdfg = _build(x.shape, dace.float64)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=-1)
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 4], dtype=np.float64))


if __name__ == '__main__':
    test_eoshift_1d_positive_shift_fills_zero()
    test_eoshift_1d_negative_shift_fills_zero()
    print('EOShift tests PASS')
