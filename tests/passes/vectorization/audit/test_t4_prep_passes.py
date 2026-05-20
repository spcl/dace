# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the T4 prep passes:
:class:`StrideMapByTileWidths` and :class:`GenerateTileIterationMask`.
"""
import pytest

import dace
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask,
)
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG,
)
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
    pipeline_input = {}
    StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, pipeline_input)
    second = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, pipeline_input)
    assert second is None


def test_stride_map_respects_mark_tile_dims_selection():
    """When a MarkTileDims result is supplied, only those maps are rewritten."""
    sdfg = _build_k2_axpy_sdfg()
    me, _ = _find_innermost_map(sdfg)
    rewritten = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {"MarkTileDims": {}})
    assert rewritten is None
    rewritten = StrideMapByTileWidths(widths=(4, 8)).apply_pass(sdfg, {"MarkTileDims": {me: None}})
    assert rewritten == 1


def test_generate_tile_iteration_mask_allocates_mask_and_lib_node():
    """``GenerateTileIterationMask`` adds the transient + a TileMaskGen producer."""
    sdfg = _build_k2_axpy_sdfg()
    NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    attached = GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    assert attached == 1

    found_mask_in_inner = False
    found_lib_node = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            if TileNameScheme.ITER_MASK in n.sdfg.arrays:
                found_mask_in_inner = True
                arr = n.sdfg.arrays[TileNameScheme.ITER_MASK]
                assert tuple(int(s) for s in arr.shape) == (4, 8)
                assert arr.dtype == dace.bool_
            for inner_n, _ in n.sdfg.all_nodes_recursive():
                if isinstance(inner_n, TileMaskGen):
                    found_lib_node = True
                    assert tuple(inner_n.widths) == (4, 8)
    assert found_mask_in_inner
    assert found_lib_node


def test_generate_tile_iteration_mask_is_idempotent():
    """A second run does not add a duplicate mask."""
    sdfg = _build_k2_axpy_sdfg()
    NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    second = GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    assert second is None


def test_generate_tile_iteration_mask_threads_symbols():
    """Outer iter-vars and free symbols inside ``global_ubs`` are threaded
    into the body NSDFG's symbol table + symbol_mapping."""
    sdfg = _build_k2_axpy_sdfg()
    NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            assert "M" in n.symbol_mapping or "M" in n.sdfg.symbols
            assert "N" in n.symbol_mapping or "N" in n.sdfg.symbols


def test_generate_tile_iteration_mask_refuses_bare_tasklet_body():
    """Bodies that are NOT nested in a single NSDFG raise NotImplementedError."""
    sdfg = _build_k2_axpy_sdfg()
    with pytest.raises(NotImplementedError, match="single NestedSDFG"):
        GenerateTileIterationMask(widths=(4, 8)).apply_pass(sdfg, {})


def test_stride_map_rejects_invalid_K():
    """Constructor refuses widths outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        StrideMapByTileWidths(widths=())


def test_generate_mask_rejects_invalid_K():
    """Constructor refuses widths outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        GenerateTileIterationMask(widths=())
