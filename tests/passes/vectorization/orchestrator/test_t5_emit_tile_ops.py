# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Smoke tests for :class:`EmitTileOps` operating in the parent scope.

The emitter runs after :class:`GenerateTileIterationMask` and
:class:`StrideMapByTileWidths`. T5 MVP handles single-binop tasklet
bodies; tests verify the rewrite shape (TileLoad x 2 + TileBinop +
TileStore + mask wiring) and that the original tasklet is gone.
"""
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileMaskGen, TileStore
from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask, )
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths, )


def _build_k2_axpy_sdfg():
    """K=2 axpy: ``C[i, j] = A[i, j] + B[i, j]``."""
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG("k2_axpy_for_emit")
    sdfg.add_array("A", (M, N), dace.float64)
    sdfg.add_array("B", (M, N), dace.float64)
    sdfg.add_array("C", (M, N), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {
            "i": "0:M",
            "j": "0:N"
        },
        {
            "_a": dace.Memlet("A[i, j]"),
            "_b": dace.Memlet("B[i, j]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def _all_nodes_of(sdfg, cls):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls)]


def test_emit_tile_ops_rewrites_k2_axpy_body():
    """K=2 axpy body becomes TileLoad x 2 + TileBinop + TileStore."""
    sdfg = _build_k2_axpy_sdfg()
    GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {})
    rewritten = EmitTileOps(widths=(4, 8)).apply_pass(sdfg, {})
    assert rewritten == 1

    loads = _all_nodes_of(sdfg, TileLoad)
    stores = _all_nodes_of(sdfg, TileStore)
    binops = _all_nodes_of(sdfg, TileBinop)
    masks = _all_nodes_of(sdfg, TileMaskGen)
    assert len(loads) == 2
    assert len(stores) == 1
    assert len(binops) == 1
    assert len(masks) == 1

    for lib in loads + stores + binops:
        assert lib.has_mask is True
        assert tuple(lib.widths) == (4, 8)
    assert binops[0].op == "+"
    assert binops[0].kind_a == "Tile"
    assert binops[0].kind_b == "Tile"

    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert tasklets == []


def test_emit_tile_ops_refuses_without_mask():
    """Without ``GenerateTileIterationMask`` the emitter refuses loudly."""
    sdfg = _build_k2_axpy_sdfg()
    with pytest.raises(NotImplementedError, match="GenerateTileIterationMask"):
        EmitTileOps(widths=(4, 8)).apply_pass(sdfg, {})


def test_emit_tile_ops_refuses_invalid_K():
    """Constructor refuses widths outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        EmitTileOps(widths=())
