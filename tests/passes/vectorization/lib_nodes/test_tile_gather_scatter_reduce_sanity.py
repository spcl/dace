# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sanity tests for :class:`TileGather`, :class:`TileScatter`,
:class:`TileReduce`.

Per user directive 2026-05-20: per-slice unit tests stay light
(constructor refusals + one end-to-end smoke each); the real
validation comes via the cross-suite parametrization in Slice H.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileGather, TileReduce, TileScatter


def test_tile_gather_constructor_refuses_invalid_widths():
    """Widths outside ``{1, 2, 3}`` are rejected."""
    with pytest.raises(ValueError, match="length in"):
        TileGather(name="G", widths=())
    with pytest.raises(ValueError, match="length in"):
        TileGather(name="G", widths=(2, 2, 2, 2))


def test_tile_gather_constructor_refuses_zero_source_ndim():
    """``source_ndim`` must be >= 1."""
    with pytest.raises(ValueError, match="source_ndim"):
        TileGather(name="G", widths=(8, ), source_ndim=0)


def test_tile_gather_declares_source_ndim_index_connectors():
    """``source_ndim=2`` declares ``_idx_0`` and ``_idx_1`` inputs."""
    node = TileGather(name="G", widths=(4, 8), source_ndim=2, has_mask=True)
    assert "_src" in node.in_connectors
    assert "_idx_0" in node.in_connectors
    assert "_idx_1" in node.in_connectors
    assert "_mask" in node.in_connectors
    assert "_dst" in node.out_connectors


def test_tile_scatter_constructor_refuses_invalid_widths():
    """Widths outside ``{1, 2, 3}`` are rejected."""
    with pytest.raises(ValueError, match="length in"):
        TileScatter(name="S", widths=())


def test_tile_scatter_declares_dest_ndim_index_connectors():
    """``dest_ndim=2`` declares ``_idx_0`` and ``_idx_1`` inputs."""
    node = TileScatter(name="S", widths=(4, 8), dest_ndim=2, has_mask=True)
    assert "_idx_0" in node.in_connectors
    assert "_idx_1" in node.in_connectors


def test_tile_reduce_constructor_refuses_unknown_op():
    """Constructor refuses ops outside ``{+, *, min, max}``."""
    with pytest.raises(ValueError, match="unknown op"):
        TileReduce(name="R", widths=(8, ), op="xor")


def test_tile_reduce_constructor_refuses_axis_out_of_range():
    """``axis`` must be in ``[0, K)``."""
    with pytest.raises(ValueError, match="axis 2 out of range"):
        TileReduce(name="R", widths=(4, 8), op="+", axis=2)


def test_tile_gather_pure_smoke_1d():
    """End-to-end smoke: 1D gather ``dst[k] = src[idx[k]]`` matches numpy."""
    W = 8
    sdfg = dace.SDFG("tile_gather_pure_1d")
    sdfg.add_array("SRC", (W * 2, ), dace.float64, transient=False)
    sdfg.add_array("IDX", (W, ), dace.int32, transient=False)
    sdfg.add_array("DST", (W, ), dace.float64, transient=False)
    state = sdfg.add_state("main")
    src = state.add_access("SRC")
    idx = state.add_access("IDX")
    dst = state.add_access("DST")
    node = TileGather(name="G", widths=(W, ), source_ndim=1)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", dace.Memlet(f"SRC[0:{W * 2}]"))
    state.add_edge(idx, None, node, "_idx_0", dace.Memlet(f"IDX[0:{W}]"))
    state.add_edge(node, "_dst", dst, None, dace.Memlet(f"DST[0:{W}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    rng = np.random.default_rng(seed=42)
    SRC = rng.random(W * 2)
    IDX = rng.integers(0, W * 2, size=W).astype(np.int32)
    DST = np.zeros(W)
    sdfg(SRC=SRC, IDX=IDX, DST=DST)
    np.testing.assert_allclose(DST, SRC[IDX], rtol=0, atol=0)


def test_tile_scatter_pure_smoke_1d():
    """End-to-end smoke: 1D scatter ``dst[idx[k]] = src[k]`` matches numpy."""
    W = 8
    sdfg = dace.SDFG("tile_scatter_pure_1d")
    sdfg.add_array("SRC", (W, ), dace.float64, transient=False)
    sdfg.add_array("IDX", (W, ), dace.int32, transient=False)
    sdfg.add_array("DST", (W * 2, ), dace.float64, transient=False)
    state = sdfg.add_state("main")
    src = state.add_access("SRC")
    idx = state.add_access("IDX")
    dst = state.add_access("DST")
    node = TileScatter(name="S", widths=(W, ), dest_ndim=1)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", dace.Memlet(f"SRC[0:{W}]"))
    state.add_edge(idx, None, node, "_idx_0", dace.Memlet(f"IDX[0:{W}]"))
    state.add_edge(node, "_dst", dst, None, dace.Memlet(f"DST[0:{W * 2}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    rng = np.random.default_rng(seed=43)
    SRC = rng.random(W)
    IDX = np.arange(W).astype(np.int32) * 2  # disjoint targets, no collisions
    DST = np.zeros(W * 2)
    sdfg(SRC=SRC, IDX=IDX, DST=DST)
    ref = np.zeros(W * 2)
    ref[IDX] = SRC
    np.testing.assert_allclose(DST, ref, rtol=0, atol=0)


def test_tile_reduce_pure_smoke_2d_axis_1():
    """End-to-end smoke: K=2 reduce along axis 1 matches numpy."""
    W0, W1 = 4, 8
    sdfg = dace.SDFG("tile_reduce_pure_2d_axis1")
    sdfg.add_array("SRC", (W0, W1), dace.float64, transient=False)
    sdfg.add_array("DST", (W0, ), dace.float64, transient=False)
    state = sdfg.add_state("main")
    src = state.add_access("SRC")
    dst = state.add_access("DST")
    node = TileReduce(name="R", widths=(W0, W1), op="+", axis=1)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", dace.Memlet(f"SRC[0:{W0}, 0:{W1}]"))
    state.add_edge(node, "_dst", dst, None, dace.Memlet(f"DST[0:{W0}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    rng = np.random.default_rng(seed=45)
    SRC = rng.random((W0, W1))
    DST = np.zeros(W0)
    sdfg(SRC=SRC, DST=DST)
    np.testing.assert_allclose(DST, SRC.sum(axis=1), rtol=1e-12, atol=1e-12)


def test_tile_gather_scalar_idx_no_subscript():
    """A ``Scalar`` ``_idx_<k>`` source (loop-invariant index — the
    canonical form after ``ConvertLengthOneArraysToScalars`` runs early
    in the K-dim orchestrator) must produce ``_idx_<k>`` without any
    subscript: DaCe passes Scalars by value, so ``_idx_<k>[0]`` would
    be a compile error (can't subscript a non-array). Reading the
    scalar directly broadcasts it across every lane."""
    import dace
    import numpy as np
    from dace.libraries.tileops import TileGather

    sdfg = dace.SDFG("gather_scalar_idx_by_value")
    sdfg.add_array("SRC", (10, ), dace.float64)
    sdfg.add_array("DST", (8, ), dace.float64)
    sdfg.add_scalar("IDX", dace.int64)
    state = sdfg.add_state("main")
    s, d, ix = state.add_access("SRC"), state.add_access("DST"), state.add_access("IDX")
    g = TileGather(name="g", widths=(8, ), source_ndim=1)
    state.add_node(g)
    state.add_edge(s, None, g, "_src", dace.Memlet("SRC[0:10]"))
    state.add_edge(ix, None, g, "_idx_0", dace.Memlet("IDX[0]"))
    state.add_edge(g, "_dst", d, None, dace.Memlet("DST[0:8]"))
    sdfg.expand_library_nodes()
    tasklet_bodies = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert not any("_idx_0[" in b for b in tasklet_bodies), tasklet_bodies

    # End-to-end: every lane reads SRC[IDX].
    SRC = np.arange(10, dtype=np.float64)
    DST = np.zeros(8)
    sdfg(SRC=SRC, DST=DST, IDX=np.int64(3))
    np.testing.assert_array_equal(DST, np.full(8, SRC[3]))
