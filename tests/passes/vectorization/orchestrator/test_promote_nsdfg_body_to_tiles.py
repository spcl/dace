# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NSDFG-body descent (Slice E.0): flat body-NSDFG tiled in place.

A ``vbor``-style per-iteration scalar chain is lowered by the frontend
into a multi-state body NestedSDFG (``loop_body``) inside the tile map's
scope — invisible to flat :class:`EmitTileOps`. :class:`PromoteNSDFGBodyToTiles`
promotes that body to tile ops in place (reshape ``(1,)`` transients to
``(W,)``, connector reads -> masked :class:`TileLoad`, split binops ->
:class:`TileBinop`, connector writes -> masked :class:`TileStore`), so the
kernel vectorizes instead of clean-skipping. The carried-dependency
LoopRegion bodies (s231) are NOT yet handled and must still clean-skip
(``EmitTileOps`` raises -> the orchestrator surfaces ``NotImplementedError``).
"""
import copy

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileStore
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

L = dace.symbol("LEN_2D")


@dace.program
def _vbor(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L],
          d: dace.float64[L], e: dace.float64[L], x: dace.float64[L]):
    for i in range(L):
        a1 = a[i]; b1 = b[i]; c1 = c[i]; d1 = d[i]; e1 = e[i]; f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 +
              a1 * c1 * f1 + a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@dace.program
def _s231(aa: dace.float64[L, L], bb: dace.float64[L, L]):
    for i in range(L):
        for j in range(1, L):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def _build(prog, name):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.name = name
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    return sdfg


@pytest.mark.parametrize("n", [16, 17])
def test_vbor_nsdfg_body_descent_matches_reference(n):
    """vbor's flat body-NSDFG is tiled in place and matches the
    unvectorized reference (n=17 forces the masked i-tail)."""
    rng = np.random.default_rng(seed=n)
    arrays = {k: rng.random(n) for k in "abcde"}
    arrays["x"] = np.zeros(n)
    ref = _build(_vbor, f"vbor_ref{n}")
    vec = _build(_vbor, f"vbor_vec{n}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

    rf = {k: v.copy() for k, v in arrays.items()}
    vf = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**rf, LEN_2D=n)
    vec.compile()(**vf, LEN_2D=n)
    np.testing.assert_allclose(vf["x"], rf["x"], rtol=1e-12, atol=1e-12)


def test_vbor_emits_tile_ops():
    """The descent leaves TileLoad / TileBinop / TileStore lib nodes
    inside the body NSDFG (before ``expand_library_nodes``)."""
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
    from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
    from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import PromoteNSDFGBodyToTiles
    from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths

    sdfg = _build(_vbor, "vbor_struct")
    W = (8, )
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    res = MarkTileDims(widths=W).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    StrideMapByTileWidths(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    handled = PromoteNSDFGBodyToTiles(widths=W).apply_pass(sdfg, {"MarkTileDims": res})

    assert handled, "expected the vbor map to be handled by the descent"
    loads = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad)]
    binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
    stores = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileStore)]
    assert loads, "expected TileLoad nodes for the connector reads"
    assert binops, "expected TileBinop nodes for the scalar chain"
    assert len(stores) == 1, "expected one TileStore for x[i]"


def test_s231_loopregion_body_still_clean_skips():
    """The carried-dep LoopRegion body (s231) is NOT a flat body, so the
    descent leaves it; ``EmitTileOps`` raises ``NotImplementedError``
    (the kernel stays a clean skip until its own slice lands)."""
    sdfg = _build(_s231, "s231_skip")
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(sdfg, {})
