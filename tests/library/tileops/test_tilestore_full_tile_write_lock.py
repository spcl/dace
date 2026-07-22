# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the design section 6.7 full-tile-write lock on ``TileStore``.

Per user direction 2026-06-09: a structured (non-scatter) ``TileStore`` must
write a full ``widths``-shape tile -- the dest memlet's per-dim subset extents
must equal ``widths`` under the ``dst_dims`` permutation. Anything else
(scalar write to global, single-element write, partial-tile write) raises
``NotImplementedError`` until the reduction (scalar transient ->
single-element global write) and single-element tile-load paths are
designed.

Scatter mode (``gather_dims`` non-empty) is exempt: the dest memlet covers
the full dest range and per-lane addressing comes from ``_idx_<k>``.
"""
import pytest

import dace
from dace.libraries.tileops import TileStore
from dace.memlet import Memlet
from dace.symbolic import ONE


def _build_store(src_shape, dst_shape, dst_subset, widths, gather_dims=None, idx_shapes=None):
    sdfg = dace.SDFG("ts_fixture")
    if "ONE" not in sdfg.constants_prop:
        sdfg.add_constant("ONE", 1, dace.data.Scalar(dace.int32))
    sdfg.add_array("Src", src_shape, dace.float64, transient=True)
    sdfg.add_array("Dst", dst_shape, dace.float64, transient=False)
    state = sdfg.add_state("s")
    src_an = state.add_access("Src")
    dst_an = state.add_access("Dst")
    node = TileStore("ts", widths=widths, gather_dims=gather_dims or ())
    state.add_node(node)
    state.add_edge(src_an, None, node, "_src", Memlet(f"Src[{', '.join(f'0:{w}' for w in widths)}]"))
    state.add_edge(node, "_dst", dst_an, None, Memlet(f"Dst[{dst_subset}]"))
    if gather_dims:
        for d, shape in zip(gather_dims, idx_shapes or ()):
            sdfg.add_array(f"Idx{d}", shape, dace.int64, transient=True)
            idx_an = state.add_access(f"Idx{d}")
            state.add_edge(idx_an, None, node, f"_idx_{d}", Memlet(f"Idx{d}[{', '.join(f'0:{s}' for s in shape)}]"))
    return sdfg, state, node


def test_accepts_full_tile_window_K2():
    """``Dst[0:4, 0:8]`` with widths ``(4, 8)`` -- full-tile write, passes."""
    sdfg, state, node = _build_store(src_shape=(4, 8), dst_shape=(16, 32), dst_subset="0:4, 0:8", widths=(4, 8))
    node.validate(sdfg, state)


def test_accepts_full_tile_window_at_offset_K2():
    """``Dst[i:i+4, j:j+8]`` with widths ``(4, 8)`` -- full-tile write at offset, passes."""
    sdfg, state, node = _build_store(src_shape=(4, 8), dst_shape=(16, 32), dst_subset="i:i+4, j:j+8", widths=(4, 8))
    node.validate(sdfg, state)


def test_refuses_whole_array_write_with_smaller_tile():
    """``Dst[0:16, 0:32]`` (whole dest) with widths ``(4, 8)`` -- non-full-tile, refused."""
    sdfg, state, node = _build_store(src_shape=(4, 8), dst_shape=(16, 32), dst_subset="0:16, 0:32", widths=(4, 8))
    with pytest.raises(NotImplementedError, match=r"non-full-tile structured store"):
        node.validate(sdfg, state)


def test_refuses_single_element_write():
    """``Dst[0, 0]`` (single element) with widths ``(4, 8)`` -- refused (scalar-write path deferred)."""
    sdfg, state, node = _build_store(src_shape=(4, 8), dst_shape=(16, 32), dst_subset="0:1, 0:1", widths=(4, 8))
    with pytest.raises(NotImplementedError, match=r"non-full-tile structured store"):
        node.validate(sdfg, state)


def test_refuses_partial_inner_dim():
    """``Dst[0:4, 0:4]`` with widths ``(4, 8)`` -- inner dim only half-tile, refused."""
    sdfg, state, node = _build_store(src_shape=(4, 8), dst_shape=(16, 32), dst_subset="0:4, 0:4", widths=(4, 8))
    with pytest.raises(NotImplementedError, match=r"non-full-tile structured store"):
        node.validate(sdfg, state)


def test_scatter_mode_skips_full_tile_check():
    """When ``gather_dims`` is non-empty, the dest memlet covers the full dest range and the
    full-tile-shape check is skipped (per-lane addressing via ``_idx_<k>``)."""
    sdfg, state, node = _build_store(src_shape=(4, 8),
                                     dst_shape=(16, 32),
                                     dst_subset="0:16, 0:32",
                                     widths=(4, 8),
                                     gather_dims=(0, ),
                                     idx_shapes=[(4, ONE)])
    node.validate(sdfg, state)


def test_accepts_K1_full_tile_window():
    """K=1 ``Dst[0:8]`` with widths ``(8,)`` -- full-tile, passes."""
    sdfg, state, node = _build_store(src_shape=(8, ), dst_shape=(32, ), dst_subset="0:8", widths=(8, ))
    node.validate(sdfg, state)


def test_refuses_K1_single_element_write():
    """K=1 ``Dst[0:1]`` with widths ``(8,)`` -- single element, refused."""
    sdfg, state, node = _build_store(src_shape=(8, ), dst_shape=(32, ), dst_subset="0:1", widths=(8, ))
    with pytest.raises(NotImplementedError, match=r"non-full-tile structured store"):
        node.validate(sdfg, state)
