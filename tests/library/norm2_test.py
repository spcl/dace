# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`Norm2` library node pure expansion."""
import numpy as np

import dace
from dace.libraries.standard.nodes import Norm2
from dace.libraries.standard.nodes.reduce import Reduce


def _build(in_shape, dtype, dim=None):
    sdfg = dace.SDFG(f"norm2_dim{dim}")
    sdfg.add_array("v", list(in_shape), dtype)
    if dim is None:
        sdfg.add_array("r", [1], dtype)
    else:
        out_shape = [s for d, s in enumerate(in_shape) if d != (dim - 1)] or [1]
        sdfg.add_array("r", out_shape, dtype)
    state = sdfg.add_state()
    node = Norm2("norm2", dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_out', state.add_write("r"), None, dace.Memlet.from_array("r", sdfg.arrays["r"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_norm2_whole_array():
    x = np.array([3.0, 4.0])
    sdfg = _build(x.shape, dace.float64)
    r = np.zeros(1)
    sdfg(v=x, r=r)
    np.testing.assert_allclose(r[0], 5.0)


def test_norm2_random_1d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    sdfg = _build(x.shape, dace.float64)
    r = np.zeros(1)
    sdfg(v=x, r=r)
    np.testing.assert_allclose(r[0], float(np.linalg.norm(x)), rtol=1e-12)


def test_norm2_2d_dim1():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6))
    sdfg = _build(x.shape, dace.float64, dim=1)
    r = np.zeros(6)
    sdfg(v=x, r=r)
    np.testing.assert_allclose(r, np.linalg.norm(x, axis=0), rtol=1e-12)


if __name__ == '__main__':
    test_norm2_whole_array()
    test_norm2_random_1d()
    test_norm2_2d_dim1()
    print('Norm2 tests PASS')


def test_the_sum_of_squares_goes_through_a_reduce_node_not_a_wcr_map():
    """A WCR into one accumulator is one global atomic per element on a GPU, all contending for the
    same address. A ``Reduce`` node instead owns its per-target lowering and reaches CUB's
    ``DeviceReduce`` on CUDA, which is why ``count_node`` and ``allany`` reduce through one too."""
    sdfg = dace.SDFG('norm2_lowering')
    sdfg.add_array('v', [256], dace.float64)
    sdfg.add_array('r', [1], dace.float64)
    state = sdfg.add_state()
    node = Norm2('norm2', dim=None)
    state.add_node(node)
    state.add_edge(state.add_read('v'), None, node, '_x', dace.Memlet.from_array('v', sdfg.arrays['v']))
    state.add_edge(node, '_out', state.add_write('r'), None, dace.Memlet.from_array('r', sdfg.arrays['r']))
    # One level only: the Reduce node is what this asserts about, so it must not be expanded away.
    sdfg.expand_library_nodes(recursive=False)

    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 1, f'the sum of squares is not a Reduce node: {reduces}'
    wcr = [e.data for e, _ in sdfg.all_edges_recursive() if isinstance(e.data, dace.Memlet) and e.data.wcr is not None]
    assert not wcr, f'the expansion still accumulates through a WCR: {wcr}'
