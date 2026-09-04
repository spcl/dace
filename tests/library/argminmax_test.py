# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for :class:`ArgMin` / :class:`ArgMax` library nodes.

Covers:

* whole-array reduction (Fortran ``MINLOC``/``MAXLOC`` scalar form)
* per-axis reduction via the ``dim`` property
* tie-break direction toggle via ``back``
* 0-based vs 1-based output via ``one_based``
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ArgMin, ArgMax
from dace.libraries.standard.nodes.reduce import Reduce


def _build_whole(node_cls, in_shape, dtype, *, one_based=True, back=False):
    """Build the whole-array (``dim=None``) form.

    Fortran ``MINLOC`` / ``MAXLOC`` without ``DIM`` return a rank-1
    integer array of length ``rank(input)`` with the multi-dim subscript
    of the located element, NOT a flat scalar -- so the output buffer
    here has shape ``[rank]``.
    """
    rank = len(in_shape)
    sdfg = dace.SDFG(f"{node_cls.__name__}_whole_{'_'.join(map(str, in_shape))}_{int(back)}_{int(one_based)}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", [rank], dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=None)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_idx', state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def _build_dim(node_cls, in_shape, dim, dtype, *, one_based=True, back=False):
    out_shape = [s for d, s in enumerate(in_shape) if d != (dim - 1)] or [1]
    sdfg = dace.SDFG(f"{node_cls.__name__}_dim{dim}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", out_shape, dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, '_x', dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, '_idx', state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_argmin_whole_array_one_based():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    sdfg = _build_whole(ArgMin, x.shape, dace.float64)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmin(x)) + 1


def test_argmax_whole_array_one_based():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32)
    sdfg = _build_whole(ArgMax, x.shape, dace.float64)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmax(x)) + 1


def test_argmin_whole_array_zero_based():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(16)
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == int(np.argmin(x))


def test_argmin_first_occurrence_default():
    """Default (``back=False``) selects the first minimum position."""
    x = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == 1  # the first 0.0


def test_argmin_last_occurrence_back():
    """``back=True`` selects the last minimum position."""
    x = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=False, back=True)
    idx = np.zeros(1, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert idx[0] == 3  # the last 0.0


def test_argmin_2d_whole_array_returns_subscript():
    """``MINLOC`` without ``DIM`` returns the multi-dim subscript as rank-1."""
    x = np.array([[1.0, 2.0, 3.0], [0.5, 4.0, 5.0]])
    sdfg = _build_whole(ArgMin, x.shape, dace.float64, one_based=True)
    idx = np.zeros(2, dtype=np.int32)
    sdfg(v=x, idx=idx)
    # 0.5 lives at (1, 0) -- Fortran 1-based -> [2, 1].
    assert tuple(idx) == (2, 1), idx


def test_argmax_3d_whole_array_returns_subscript():
    x = np.zeros((3, 4, 2))
    x[2, 1, 0] = 99.0  # unique max
    sdfg = _build_whole(ArgMax, x.shape, dace.float64, one_based=True)
    idx = np.zeros(3, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert tuple(idx) == (3, 2, 1), idx


def test_argmax_2d_dim1():
    """Reduce along Fortran dim=1 (rows): result has shape (#cols,)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6))
    sdfg = _build_dim(ArgMax, x.shape, 1, dace.float64)
    idx = np.zeros((6, ), dtype=np.int32)
    sdfg(v=x, idx=idx)
    np.testing.assert_array_equal(idx, np.argmax(x, axis=0) + 1)


def test_argmin_2d_dim2():
    """Reduce along Fortran dim=2 (cols): result has shape (#rows,)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 7))
    sdfg = _build_dim(ArgMin, x.shape, 2, dace.float64)
    idx = np.zeros((5, ), dtype=np.int32)
    sdfg(v=x, idx=idx)
    np.testing.assert_array_equal(idx, np.argmin(x, axis=1) + 1)


if __name__ == '__main__':
    test_argmin_whole_array_one_based()
    test_argmax_whole_array_one_based()
    test_argmin_whole_array_zero_based()
    test_argmin_first_occurrence_default()
    test_argmin_last_occurrence_back()
    test_argmax_2d_dim1()
    test_argmin_2d_dim2()
    print('ArgMin / ArgMax library node tests PASS')


@pytest.mark.parametrize('cls', [ArgMin, ArgMax])
def test_both_reduction_passes_go_through_reduce_nodes(cls):
    """Neither pass may accumulate through a WCR.

    A WCR into one accumulator is a global atomic per element on a GPU, all of them contending for
    the same address -- and this node does it TWICE, once for the winning value and once for its
    index. ``Reduce`` nodes own their per-target lowering instead and reach CUB's ``DeviceReduce``.
    """
    sdfg = dace.SDFG(f'{cls.__name__}_lowering')
    sdfg.add_array('x', [256], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = cls('argminmax')
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_x', dace.Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(node, '_idx', state.add_write('idx'), None, dace.Memlet.from_array('idx', sdfg.arrays['idx']))
    # One level only: the Reduce nodes are what this asserts about.
    sdfg.expand_library_nodes(recursive=False)

    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 2, f'expected a Reduce for the value and one for the index, got {reduces}'
    wcr = [e.data for e, _ in sdfg.all_edges_recursive() if isinstance(e.data, dace.Memlet) and e.data.wcr is not None]
    assert not wcr, f'the expansion still accumulates through a WCR: {wcr}'
