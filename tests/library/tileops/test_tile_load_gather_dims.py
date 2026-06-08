# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``TileLoad`` / ``TileStore`` ``gather_dims`` + variable-shape ``_idx_<d>``.

Covers the lane-dependency patterns from
TILIFICATION_TRANSFORMATION_DESIGN.md section 9.2:

* ``deps = ()``                       -> ``_idx_<d>`` shape ``(1,)``       (scalar gather index)
* ``deps = (p,)``                     -> ``_idx_<d>`` shape ``(W_p,)``     (1-D)
* ``deps = (p, q)``                   -> ``_idx_<d>`` shape ``(W_p, W_q)`` (2-D)
* ``deps = (0, ..., K-1)``            -> ``_idx_<d>`` shape full           (N-D)
* mixed: two gather dims, distinct shapes
* refusal: shape not a Cartesian product of widths; unsorted gather_dims; out-of-range gather_dims; wrong dtype.
"""
import pytest

import dace
from dace.libraries.tileops.nodes.tile_load import TileLoad
from dace.libraries.tileops.nodes.tile_store import TileStore
from dace.memlet import Memlet


def _build_load(widths, gather_dims, idx_shapes, idx_dtype=dace.int64):
    """Build a minimal SDFG containing a TileLoad with the given gather_dims.

    Each ``idx_shapes[i]`` is the shape to use for the corresponding ``_idx_<d>``
    connector descriptor.
    """
    sdfg = dace.SDFG(f"tl_K{len(widths)}_g{''.join(str(d) for d in gather_dims)}")
    sdfg.add_array("Src", (64, 64, 64)[:max(2, len(widths))], dace.float64, transient=False)
    sdfg.add_array("Dst", widths, dace.float64, transient=True)
    for d, shape in zip(gather_dims, idx_shapes):
        sdfg.add_array(f"Idx{d}", shape, idx_dtype, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileLoad("tl", widths=widths, gather_dims=gather_dims)
    state.add_node(node)
    src_subset = ", ".join(f"0:{s}" for s in sdfg.arrays["Src"].shape)
    state.add_edge(src, None, node, "_src", Memlet(f"Src[{src_subset}]"))
    state.add_edge(node, "_dst", dst, None, Memlet(f"Dst[{', '.join(f'0:{w}' for w in widths)}]"))
    for d, shape in zip(gather_dims, idx_shapes):
        idx = state.add_access(f"Idx{d}")
        idx_subset = ", ".join(f"0:{s}" for s in shape)
        state.add_edge(idx, None, node, f"_idx_{d}", Memlet(f"Idx{d}[{idx_subset}]"))
    return sdfg, state, node


def test_no_gather_validates():
    """Structured load -- no _idx_<d> connectors."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(), idx_shapes=())
    node.validate(sdfg, state)
    assert tuple(node.gather_dims) == ()


def test_1d_gather_on_dim0():
    """deps = (0,) -> _idx_0 shape (W_0,)."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(4, )])
    node.validate(sdfg, state)
    assert tuple(node.gather_dims) == (0, )
    assert "_idx_0" in node.in_connectors


def test_2d_gather_on_dim0_with_full_shape():
    """deps = (0, 1) on dim 0's index -> _idx_0 shape (W_0, W_1).

    Covers B[idx[i, j], j] vectorising (i, j).
    """
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(4, 8)])
    node.validate(sdfg, state)


def test_scalar_gather_index():
    """deps = () -> _idx_<d> shape (1,) (scalar / loop-invariant index)."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(1, )])
    node.validate(sdfg, state)


def test_mixed_gather_dims_distinct_shapes():
    """Two gather dims, each with a different lane-dependency shape.

    Covers the ICON pattern B[idx[i, k], j, idb[i, k]] vectorising (i, j):
    both _idx_0 and _idx_2 have shape (W_i,).
    """
    # K=3 widths; gather on dims 0 and 2; each index shape (W_0,) = (4,).
    sdfg, state, node = _build_load(widths=(4, 8, 16), gather_dims=(0, 2), idx_shapes=[(4, ), (4, )])
    node.validate(sdfg, state)


def test_full_nd_gather():
    """deps = (0, 1, 2) on one gather dim -> _idx_<d> shape (W_0, W_1, W_2). Was _idx_full."""
    sdfg, state, node = _build_load(widths=(2, 4, 8), gather_dims=(0, ), idx_shapes=[(2, 4, 8)])
    node.validate(sdfg, state)


def test_refuse_index_shape_not_cartesian():
    """A shape that isn't any Cartesian product of widths -> ValueError."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(3, )])
    with pytest.raises(ValueError, match="not a Cartesian product"):
        node.validate(sdfg, state)


def test_refuse_unsorted_gather_dims():
    """Constructor refuses unsorted gather_dims."""
    with pytest.raises(ValueError, match="sorted"):
        TileLoad("bad", widths=(4, 8, 16), gather_dims=(2, 0))


def test_refuse_out_of_range_gather_dims():
    """Constructor refuses gather_dims outside range(K)."""
    with pytest.raises(ValueError, match="range"):
        TileLoad("bad", widths=(4, 8), gather_dims=(2, ))


def test_refuse_wrong_index_dtype():
    """Index dtype must be int32 or int64 (design section 10.4)."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(4, )], idx_dtype=dace.float64)
    with pytest.raises(ValueError, match="dtype.*not in"):
        node.validate(sdfg, state)


def test_tilestore_gather_dims_symmetric():
    """TileStore mirrors TileLoad's gather_dims surface."""
    sdfg = dace.SDFG("ts_g")
    sdfg.add_array("Src", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Dst", (64, 64), dace.float64, transient=False)
    sdfg.add_array("Idx0", (4, ), dace.int64, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    idx = state.add_access("Idx0")
    node = TileStore("ts", widths=(4, 8), gather_dims=(0, ))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:4, 0:8]"))
    state.add_edge(idx, None, node, "_idx_0", Memlet("Idx0[0:4]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:64, 0:64]"))
    node.validate(sdfg, state)
    assert tuple(node.gather_dims) == (0, )
    assert "_idx_0" in node.in_connectors


# ---------------------------------------------------------------------
# End-to-end correctness: the pure expansion compiles + runs correctly
# for each lane-dependency pattern.
# ---------------------------------------------------------------------


def _run_gather_load(src_np, idx_np_per_d, widths, gather_dims, src_dims=None):
    """Build, expand, compile, and run a TileLoad with the given gather setup;
    return the materialised destination tile as a numpy array."""
    import numpy as np
    sdfg = dace.SDFG(f"e2e_K{len(widths)}_g{''.join(str(d) for d in gather_dims)}")
    src_shape = src_np.shape
    sdfg.add_array("Src", src_shape, dace.float64, transient=False)
    sdfg.add_array("Dst", widths, dace.float64, transient=False)
    for d in gather_dims:
        sdfg.add_array(f"Idx{d}", idx_np_per_d[d].shape, dace.int64, transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileLoad("tl", widths=widths, gather_dims=gather_dims, src_dims=src_dims)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet(f"Src[{', '.join(f'0:{s}' for s in src_shape)}]"))
    state.add_edge(node, "_dst", dst, None, Memlet(f"Dst[{', '.join(f'0:{w}' for w in widths)}]"))
    for d in gather_dims:
        idx = state.add_access(f"Idx{d}")
        idx_subset = ", ".join(f"0:{s}" for s in idx_np_per_d[d].shape)
        state.add_edge(idx, None, node, f"_idx_{d}", Memlet(f"Idx{d}[{idx_subset}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    dst_np = np.zeros(widths, dtype=np.float64)
    kwargs = {"Src": src_np.astype(np.float64), "Dst": dst_np}
    for d in gather_dims:
        kwargs[f"Idx{d}"] = idx_np_per_d[d].astype(np.int64)
    sdfg(**kwargs)
    return dst_np


def test_e2e_1d_gather_K1():
    """1-D gather: dst[l] = src[idx[l]]."""
    import numpy as np
    W = 8
    src = np.arange(32, dtype=np.float64) * 10
    idx = np.array([3, 7, 1, 5, 2, 6, 0, 4], dtype=np.int64)
    out = _run_gather_load(src, {0: idx}, widths=(W, ), gather_dims=(0, ))
    expected = src[idx]
    np.testing.assert_array_equal(out, expected)


def test_e2e_partial_gather_K2():
    """2-D access with gather on dim 0 only: dst[l_0, l_1] = src[idx_0[l_0], l_1]."""
    import numpy as np
    W0, W1 = 4, 8
    src = np.arange(16 * W1, dtype=np.float64).reshape(16, W1)
    idx0 = np.array([7, 2, 11, 5], dtype=np.int64)
    out = _run_gather_load(src, {0: idx0}, widths=(W0, W1), gather_dims=(0, ))
    expected = np.zeros((W0, W1), dtype=np.float64)
    for l0 in range(W0):
        for l1 in range(W1):
            expected[l0, l1] = src[idx0[l0], l1]
    np.testing.assert_array_equal(out, expected)


def _run_scatter_store(src_tile, idx_np_per_d, dst_shape, widths, gather_dims, dst_dims=None, initial_dst=None):
    """Build, expand, compile, and run a TileStore with scatter; return the dst array."""
    import numpy as np
    sdfg = dace.SDFG(f"e2e_store_K{len(widths)}_g{''.join(str(d) for d in gather_dims)}")
    sdfg.add_array("Src", widths, dace.float64, transient=False)
    sdfg.add_array("Dst", dst_shape, dace.float64, transient=False)
    for d in gather_dims:
        sdfg.add_array(f"Idx{d}", idx_np_per_d[d].shape, dace.int64, transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileStore("ts", widths=widths, gather_dims=gather_dims, dst_dims=dst_dims)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet(f"Src[{', '.join(f'0:{w}' for w in widths)}]"))
    state.add_edge(node, "_dst", dst, None, Memlet(f"Dst[{', '.join(f'0:{s}' for s in dst_shape)}]"))
    for d in gather_dims:
        idx = state.add_access(f"Idx{d}")
        idx_subset = ", ".join(f"0:{s}" for s in idx_np_per_d[d].shape)
        state.add_edge(idx, None, node, f"_idx_{d}", Memlet(f"Idx{d}[{idx_subset}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    dst_np = np.zeros(dst_shape, dtype=np.float64) if initial_dst is None else initial_dst.copy()
    kwargs = {"Src": src_tile.astype(np.float64), "Dst": dst_np}
    for d in gather_dims:
        kwargs[f"Idx{d}"] = idx_np_per_d[d].astype(np.int64)
    sdfg(**kwargs)
    return dst_np


def test_e2e_1d_scatter_K1():
    """K=1 1-D scatter: dst[idx[l]] = src[l]."""
    import numpy as np
    W = 8
    src_tile = np.arange(W, dtype=np.float64) * 100
    idx = np.array([3, 7, 1, 5, 2, 6, 0, 4], dtype=np.int64)
    out = _run_scatter_store(src_tile, {0: idx}, dst_shape=(32, ), widths=(W, ), gather_dims=(0, ))
    expected = np.zeros(32, dtype=np.float64)
    for l in range(W):
        expected[idx[l]] = src_tile[l]
    np.testing.assert_array_equal(out, expected)


def test_e2e_partial_scatter_K2():
    """K=2 partial scatter on dim 0: dst[idx_0[l_0], l_1] = src[l_0, l_1]."""
    import numpy as np
    W0, W1 = 4, 8
    src_tile = np.arange(W0 * W1, dtype=np.float64).reshape(W0, W1)
    idx0 = np.array([7, 2, 11, 5], dtype=np.int64)
    out = _run_scatter_store(src_tile, {0: idx0}, dst_shape=(16, W1), widths=(W0, W1), gather_dims=(0, ))
    expected = np.zeros((16, W1), dtype=np.float64)
    for l0 in range(W0):
        for l1 in range(W1):
            expected[idx0[l0], l1] = src_tile[l0, l1]
    np.testing.assert_array_equal(out, expected)
