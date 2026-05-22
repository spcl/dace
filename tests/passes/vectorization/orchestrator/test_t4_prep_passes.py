# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the T4 prep passes:
:class:`StrideMapByTileWidths` and :class:`GenerateTileIterationMask`.

Both passes operate directly in the parent state (no body-NSDFG
nesting): the mask transient + the :class:`TileMaskGen` producer sit in
the outer map scope alongside the original tasklets so downstream
:class:`EmitTileOps` can wire the lib node chain without crossing a
NestedSDFG boundary.
"""
import pytest

import dace
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask,
)
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths,
)
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme


def _build_k2_axpy_sdfg():
    """K=2 axpy: ``C[i, j] = A[i, j] + B[i, j]`` over ``M x N``."""
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG("k2_axpy_for_prep")
    sdfg.add_array("A", (M, N), dace.float64)
    sdfg.add_array("B", (M, N), dace.float64)
    sdfg.add_array("C", (M, N), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "body",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def _find_innermost_map(sdfg):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState):
            return n, g
    raise AssertionError("no inner map found")


def test_stride_map_by_tile_widths_sets_k2_steps():
    """K=2 inner map gets ``step=(W_0, W_1)`` on its innermost two dims."""
    sdfg = _build_k2_axpy_sdfg()
    me, _ = _find_innermost_map(sdfg)
    assert all(str(r[2]) == "1" for r in me.map.range.ranges)
    rewritten = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {})
    assert rewritten == 1
    steps = tuple(str(r[2]) for r in me.map.range.ranges)
    assert steps == ("4", "8")


def test_stride_map_by_tile_widths_is_idempotent():
    """Running the pass twice yields no second rewrite."""
    sdfg = _build_k2_axpy_sdfg()
    StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {})
    second = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {})
    assert second is None


def test_stride_map_respects_mark_tile_dims_selection():
    """When MarkTileDims results are supplied, only those maps are rewritten."""
    sdfg = _build_k2_axpy_sdfg()
    me, _ = _find_innermost_map(sdfg)
    rewritten = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {"MarkTileDims": {}})
    assert rewritten is None
    rewritten = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {"MarkTileDims": {me: None}})
    assert rewritten == 1


def test_generate_tile_iteration_mask_allocates_in_outer_scope():
    """``GenerateTileIterationMask`` adds the transient + producer in the outer SDFG."""
    sdfg = _build_k2_axpy_sdfg()
    attached = GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    assert attached == 1
    assert TileNameScheme.ITER_MASK in sdfg.arrays
    arr = sdfg.arrays[TileNameScheme.ITER_MASK]
    assert tuple(int(s) for s in arr.shape) == (4, 8)
    assert arr.dtype == dace.bool_
    masks = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileMaskGen)]
    assert len(masks) == 1
    assert tuple(masks[0].widths) == (4, 8)


def test_generate_tile_iteration_mask_is_idempotent():
    """A second run does not add a duplicate mask."""
    sdfg = _build_k2_axpy_sdfg()
    GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    second = GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    assert second is None


def test_stride_map_rejects_invalid_K():
    """Constructor refuses widths outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        StrideMapByTileWidths(widths=())


def test_generate_mask_rejects_invalid_K():
    """Constructor refuses widths outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        GenerateTileIterationMask(widths=())
