# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`rewrite_array_scalar_copy_to_tile_op` and the
:class:`RewriteArrayScalarToTileOp` pass.

Each test builds a minimal SDFG by hand exhibiting one shape (load /
store / refuse) and asserts the rewrite produces the expected
:class:`TileLoad` / :class:`TileStore` structure without touching
other edges.
"""
import pytest

import dace
from dace.libraries.tileops import TileLoad, TileStore
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.rewrite_array_scalar_to_tile_op import (
    rewrite_array_scalar_copy_to_tile_op, )


def _count_lib_nodes(state, cls):
    """Count instances of ``cls`` in ``state``."""
    return sum(1 for n in state.nodes() if isinstance(n, cls))


def _build_load_fixture(widths=(4, 8)):
    """Build a 2D SDFG containing one ``AN(global Array) -> AN(tile transient)`` copy.

    The array-side subset references the tile iter-vars (``G[i:i+W_0,
    j:j+W_1]``) so the classifier sees a CONTIGUOUS box, matching the
    post-``MarkTileDims`` shape this rewriter consumes.
    """
    sdfg = dace.SDFG("rewriter_load_fixture")
    sdfg.add_array("G", (16, 32), dace.float64, transient=False)
    sdfg.add_array("T", widths, dace.float64, transient=True)
    state = sdfg.add_state("s")
    g = state.add_access("G")
    t = state.add_access("T")
    state.add_edge(g, None, t, None, Memlet(f"G[i:i+{widths[0]}, j:j+{widths[1]}]"))
    return sdfg, state, g, t


def _build_store_fixture(widths=(4, 8)):
    """Build a 2D SDFG containing one ``AN(tile transient) -> AN(global Array)`` copy."""
    sdfg = dace.SDFG("rewriter_store_fixture")
    sdfg.add_array("G", (16, 32), dace.float64, transient=False)
    sdfg.add_array("T", widths, dace.float64, transient=True)
    state = sdfg.add_state("s")
    t = state.add_access("T")
    g = state.add_access("G")
    state.add_edge(t, None, g, None, Memlet(f"G[i:i+{widths[0]}, j:j+{widths[1]}]"))
    return sdfg, state, t, g


def test_load_global_to_tile_emits_tileload():
    """``AN(global) -> AN(tile)`` becomes ``AN(global) -> TileLoad -> AN(tile)``."""
    widths = (4, 8)
    sdfg, state, g, t = _build_load_fixture(widths)
    edge = state.edges()[0]
    assert _count_lib_nodes(state, TileLoad) == 0
    ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=widths)
    assert ok
    assert _count_lib_nodes(state, TileLoad) == 1
    assert _count_lib_nodes(state, TileStore) == 0
    # The direct AN -> AN edge is gone (the load now sits between them).
    direct_edges = [
        e for e in state.edges() if isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
    ]
    assert direct_edges == []
    # TileLoad's widths property matches what we passed.
    (load, ) = [n for n in state.nodes() if isinstance(n, TileLoad)]
    assert tuple(load.widths) == widths


def test_store_tile_to_global_emits_tilestore():
    """``AN(tile) -> AN(global)`` becomes ``AN(tile) -> TileStore -> AN(global)``."""
    widths = (4, 8)
    sdfg, state, t, g = _build_store_fixture(widths)
    edge = state.edges()[0]
    assert _count_lib_nodes(state, TileStore) == 0
    ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=widths)
    assert ok
    assert _count_lib_nodes(state, TileStore) == 1
    assert _count_lib_nodes(state, TileLoad) == 0
    direct_edges = [
        e for e in state.edges() if isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
    ]
    assert direct_edges == []


def test_refuses_when_neither_side_is_tile_shaped():
    """A ``AN(global) -> AN(global)`` copy (no tile transient) is left untouched."""
    sdfg = dace.SDFG("rewriter_no_tile_fixture")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("B", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    state.add_edge(a, None, b, None, Memlet("A[0:16, 0:32]"))
    edge = state.edges()[0]
    ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=(4, 8))
    assert not ok
    # State left untouched: edge still present, no lib nodes added.
    assert _count_lib_nodes(state, TileLoad) == 0
    assert _count_lib_nodes(state, TileStore) == 0
    assert len(state.edges()) == 1


def test_refuses_when_transient_shape_doesnt_match_widths():
    """A transient whose shape != widths isn't a tile box -- the rewrite refuses."""
    sdfg = dace.SDFG("rewriter_shape_mismatch_fixture")
    sdfg.add_array("G", (16, 32), dace.float64, transient=False)
    sdfg.add_array("T", (2, 4), dace.float64, transient=True)  # shape doesn't match widths=(4, 8)
    state = sdfg.add_state("s")
    g = state.add_access("G")
    t = state.add_access("T")
    state.add_edge(g, None, t, None, Memlet("G[i:i+2, j:j+4]"))
    edge = state.edges()[0]
    ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=(4, 8))
    assert not ok
    assert _count_lib_nodes(state, TileLoad) == 0


def test_refuses_when_endpoints_arent_both_access_nodes():
    """An edge whose endpoint is a tasklet (not an access node) is refused."""
    sdfg = dace.SDFG("rewriter_tasklet_endpoint_fixture")
    sdfg.add_array("G", (16, 32), dace.float64, transient=False)
    sdfg.add_array("T", (4, 8), dace.float64, transient=True)
    state = sdfg.add_state("s")
    g = state.add_access("G")
    t = state.add_access("T")
    tlet = state.add_tasklet("noop", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(g, None, tlet, "_in", Memlet("G[0:4, 0:8]"))
    state.add_edge(tlet, "_out", t, None, Memlet("T[0:4, 0:8]"))
    # First edge has a tasklet as dst -- not AN -> AN.
    for edge in list(state.edges()):
        ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=(4, 8))
        assert not ok


def test_load_with_mask_wires_mask_connector():
    """When ``mask_node`` is provided, the TileLoad has ``has_mask=True`` and a wired
    ``_mask`` connector."""
    widths = (4, 8)
    sdfg, state, g, t = _build_load_fixture(widths)
    sdfg.add_array("M", widths, dace.bool_, transient=True)
    m = state.add_access("M")
    edge = next(e for e in state.edges() if e.src is g and e.dst is t)
    ok = rewrite_array_scalar_copy_to_tile_op(state, edge, iter_vars=("i", "j"), widths=widths, mask_node=m)
    assert ok
    (load, ) = [n for n in state.nodes() if isinstance(n, TileLoad)]
    assert bool(load.has_mask) is True
    # ``_mask`` connector is wired from the mask access node.
    mask_in_edges = [e for e in state.in_edges(load) if e.dst_conn == "_mask"]
    assert len(mask_in_edges) == 1
    assert mask_in_edges[0].src is m
