# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``TileLoad`` / ``TileStore`` ``gather_dims`` + full-K-dim ``_idx_<d>``.

Index-tile shape convention (positional, full-K-dim; see
:func:`dace.libraries.tileops._pure_codegen.resolve_gather_deps`). An
``_idx_<d>`` connector is ALWAYS rank ``K`` (the lib node's tile rank); tile dim
``p`` is a dependency iff ``idx_shape[p]`` equals ``widths[p]``, and a
non-dependency dim carries the ``dace.symbolic.ONE`` broadcast marker:

* ``deps = ()``        -> ``_idx_<d>`` shape ``(1,)``                  (scalar index)
* ``deps = (p,)``      -> ``_idx_<d>`` shape ``(.., W_p, ..)`` w/ ``ONE`` elsewhere
* ``deps = (p, q)``    -> ``W`` on dims ``p, q``; ``ONE`` elsewhere
* ``deps = (0..K-1)``  -> full ``(W_0, .., W_{K-1})``
* refusal: a non-marker extent that isn't ``widths[p]``; a rank other than ``K``
  (the legacy 1-D ``(W_p,)`` form is no longer accepted -- ``ONE`` markers are
  never collapsed); unsorted / out-of-range gather_dims; wrong dtype.
"""
import pytest

import dace
from dace.libraries.tileops.nodes.tile_load import TileLoad
from dace.libraries.tileops.nodes.tile_store import TileStore
from dace.memlet import Memlet
from dace.symbolic import ONE


def _add_one_constant(sdfg):
    """Register ``ONE`` as the integer constant ``1`` so codegen can size the
    length-1 broadcast dims of full-K-dim index tiles (mirrors
    ``InsertTileLoadStore.stage_tile_load``)."""
    if "ONE" not in sdfg.constants_prop:
        sdfg.add_constant("ONE", 1, dace.data.Scalar(dace.int32))


def _build_load(widths, gather_dims, idx_shapes, idx_dtype=dace.int64):
    """Build a minimal SDFG containing a TileLoad with the given gather_dims.

    Each ``idx_shapes[i]`` is the shape to use for the corresponding ``_idx_<d>``
    connector descriptor -- a full-K-dim ``(W_p or ONE)`` shape (or ``(1,)`` for
    a scalar index).
    """
    sdfg = dace.SDFG(f"tl_K{len(widths)}_g{''.join(str(d) for d in gather_dims)}")
    _add_one_constant(sdfg)
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
    """deps = (0,) -> _idx_0 full-K-dim shape (W_0, ONE): varies over tile dim 0,
    broadcast (ONE) over tile dim 1."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(4, ONE)])
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
    both _idx_0 and _idx_2 vary over tile dim 0 (i) only.
    """
    # K=3 widths; gather on dims 0 and 2; each index full-K-dim (W_0, ONE, ONE).
    sdfg, state, node = _build_load(widths=(4, 8, 16), gather_dims=(0, 2), idx_shapes=[(4, ONE, ONE), (4, ONE, ONE)])
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


def test_refuse_negative_gather_dims():
    """Constructor refuses negative gather_dims at construction time."""
    with pytest.raises(ValueError, match="non-negative"):
        TileLoad("bad", widths=(4, 8), gather_dims=(-1, ))


def test_validate_refuses_gather_dim_exceeding_src_ndim():
    """``validate()`` refuses ``gather_dims`` entries >= ``src_ndim`` (source-dim indexing).

    ``_build_load`` mints ``Src`` with ndim = ``max(2, len(widths))``; widths=(4, 8) -> src_ndim=2.
    gather_dims=(3,) exceeds src_ndim=2.
    """
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(3, ), idx_shapes=[(4, )])
    # _build_load already wires Idx3 -> _idx_3 so the missing-conn check passes and the
    # src-ndim upper bound is the failure cause.
    with pytest.raises(ValueError, match=r"gather_dims.*>= source ndim"):
        node.validate(sdfg, state)


def test_icon_pattern_K2_vec_K3_src_gather_dims_0_and_2():
    """ICON-shape: K_tile=2 vec(i, j), source 3D, ``gather_dims=(0, 2)`` (source dims 0+2 gather).

    Validates the source-dim-indexed contract. ``_idx_0`` and ``_idx_2`` each carry a
    full-K-dim ``(W_i, ONE)`` index (only i is the lane-dep var; j is structured ->
    broadcast ONE; k is outer-const).
    """
    sdfg = dace.SDFG("icon_pattern_node")
    _add_one_constant(sdfg)
    sdfg.add_array("Src", (32, 32, 64), dace.float64, transient=False)
    sdfg.add_array("Dst", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Idx0", (4, ONE), dace.int64, transient=True)
    sdfg.add_array("Idx2", (4, ONE), dace.int64, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    idx0 = state.add_access("Idx0")
    idx2 = state.add_access("Idx2")
    node = TileLoad("tl_icon", widths=(4, 8), gather_dims=(0, 2))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:32, 0:32, 0:64]"))
    state.add_edge(idx0, None, node, "_idx_0", Memlet("Idx0[0:4, 0:ONE]"))
    state.add_edge(idx2, None, node, "_idx_2", Memlet("Idx2[0:4, 0:ONE]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:4, 0:8]"))
    node.validate(sdfg, state)
    assert tuple(node.gather_dims) == (0, 2)
    assert "_idx_0" in node.in_connectors
    assert "_idx_2" in node.in_connectors
    assert "_idx_1" not in node.in_connectors


def test_refuse_wrong_index_dtype():
    """Index dtype must be int32 or int64 (design section 10.4). Uses a VALID
    full-K-dim shape so the dtype check (not the shape check) is the failure."""
    sdfg, state, node = _build_load(widths=(4, 8), gather_dims=(0, ), idx_shapes=[(4, ONE)], idx_dtype=dace.float64)
    with pytest.raises(ValueError, match="dtype.*not in"):
        node.validate(sdfg, state)


def test_tilestore_gather_dims_symmetric():
    """TileStore mirrors TileLoad's gather_dims surface."""
    sdfg = dace.SDFG("ts_g")
    _add_one_constant(sdfg)
    sdfg.add_array("Src", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Dst", (64, 64), dace.float64, transient=False)
    sdfg.add_array("Idx0", (4, ONE), dace.int64, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    idx = state.add_access("Idx0")
    node = TileStore("ts", widths=(4, 8), gather_dims=(0, ))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:4, 0:8]"))
    state.add_edge(idx, None, node, "_idx_0", Memlet("Idx0[0:4, 0:ONE]"))
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
    _add_one_constant(sdfg)
    src_shape = src_np.shape
    sdfg.add_array("Src", src_shape, dace.float64, transient=False)
    sdfg.add_array("Dst", widths, dace.float64, transient=False)
    # Full-K-dim index descriptor: a length-1 array dim is a broadcast tile dim,
    # declared with the ``ONE`` marker (a literal 1 would read as a width-1 *dep*
    # and fail resolution against the real tile width).
    idx_desc = {d: tuple(ONE if s == 1 else s for s in idx_np_per_d[d].shape) for d in gather_dims}
    for d in gather_dims:
        sdfg.add_array(f"Idx{d}", idx_desc[d], dace.int64, transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileLoad("tl", widths=widths, gather_dims=gather_dims, src_dims=src_dims)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet(f"Src[{', '.join(f'0:{s}' for s in src_shape)}]"))
    state.add_edge(node, "_dst", dst, None, Memlet(f"Dst[{', '.join(f'0:{w}' for w in widths)}]"))
    for d in gather_dims:
        idx = state.add_access(f"Idx{d}")
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc[d])
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
    # Full-K-dim index: (W0, ONE) descriptor -> (W0, 1) runtime array.
    out = _run_gather_load(src, {0: idx0.reshape(W0, 1)}, widths=(W0, W1), gather_dims=(0, ))
    expected = np.zeros((W0, W1), dtype=np.float64)
    for l0 in range(W0):
        for l1 in range(W1):
            expected[l0, l1] = src[idx0[l0], l1]
    np.testing.assert_array_equal(out, expected)


def _run_scatter_store(src_tile, idx_np_per_d, dst_shape, widths, gather_dims, dst_dims=None, initial_dst=None):
    """Build, expand, compile, and run a TileStore with scatter; return the dst array."""
    import numpy as np
    sdfg = dace.SDFG(f"e2e_store_K{len(widths)}_g{''.join(str(d) for d in gather_dims)}")
    _add_one_constant(sdfg)
    sdfg.add_array("Src", widths, dace.float64, transient=False)
    sdfg.add_array("Dst", dst_shape, dace.float64, transient=False)
    # Full-K-dim index descriptor (see _run_gather_load): length-1 dims -> ONE.
    idx_desc = {d: tuple(ONE if s == 1 else s for s in idx_np_per_d[d].shape) for d in gather_dims}
    for d in gather_dims:
        sdfg.add_array(f"Idx{d}", idx_desc[d], dace.int64, transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileStore("ts", widths=widths, gather_dims=gather_dims, dst_dims=dst_dims)
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet(f"Src[{', '.join(f'0:{w}' for w in widths)}]"))
    state.add_edge(node, "_dst", dst, None, Memlet(f"Dst[{', '.join(f'0:{s}' for s in dst_shape)}]"))
    for d in gather_dims:
        idx = state.add_access(f"Idx{d}")
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc[d])
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
    # Full-K-dim index: (W0, ONE) descriptor -> (W0, 1) runtime array.
    out = _run_scatter_store(src_tile, {0: idx0.reshape(W0, 1)}, dst_shape=(16, W1), widths=(W0, W1), gather_dims=(0, ))
    expected = np.zeros((16, W1), dtype=np.float64)
    for l0 in range(W0):
        for l1 in range(W1):
            expected[idx0[l0], l1] = src_tile[l0, l1]
    np.testing.assert_array_equal(out, expected)
