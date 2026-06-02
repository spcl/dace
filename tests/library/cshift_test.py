# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`CShift` library node pure expansion."""
import numpy as np

import dace
from dace.libraries.standard.nodes import CShift


def _build(in_shape, dtype, dim=1):
    sdfg = dace.SDFG(f"cshift_dim{dim}_{'_'.join(map(str, in_shape))}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("out", list(in_shape), dtype)
    sdfg.add_symbol("__shift", dace.int64)
    state = sdfg.add_state()
    node = CShift("cshift", dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_out', state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_cshift_1d_positive_shift():
    """CSHIFT(x, 1) rotates [a,b,c,d,e] -> [b,c,d,e,a]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sdfg = _build(x.shape, dace.float64, dim=1)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=1)
    np.testing.assert_array_equal(out, np.roll(x, -1))


def test_cshift_1d_negative_shift():
    """CSHIFT(x, -2) rotates [a,b,c,d,e] -> [d,e,a,b,c]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sdfg = _build(x.shape, dace.float64, dim=1)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=-2)
    np.testing.assert_array_equal(out, np.roll(x, 2))


def test_cshift_2d_dim1():
    """2-D CSHIFT along Fortran dim=1 rotates each column independently."""
    x = np.arange(12, dtype=np.float64).reshape(3, 4).copy()
    sdfg = _build(x.shape, dace.float64, dim=1)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=1)
    # Fortran dim=1 = rows axis = numpy axis 0; rotate each column up by 1
    np.testing.assert_array_equal(out, np.roll(x, -1, axis=0))


def test_cshift_2d_dim2():
    x = np.arange(12, dtype=np.float64).reshape(3, 4).copy()
    sdfg = _build(x.shape, dace.float64, dim=2)
    out = np.zeros_like(x)
    sdfg(v=x, out=out, __shift=2)
    np.testing.assert_array_equal(out, np.roll(x, -2, axis=1))


if __name__ == '__main__':
    test_cshift_1d_positive_shift()
    test_cshift_1d_negative_shift()
    test_cshift_2d_dim1()
    test_cshift_2d_dim2()
    print('CShift pure expansion tests PASS')
