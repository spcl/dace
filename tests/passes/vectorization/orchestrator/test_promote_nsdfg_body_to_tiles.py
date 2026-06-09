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

import pytest
pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileStore, TileUnop
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

L = dace.symbol("LEN_2D")


@dace.program
def _vbor(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L], e: dace.float64[L],
          x: dace.float64[L]):
    for i in range(L):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@dace.program
def _unop_chain(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L], e: dace.float64[L],
                x: dace.float64[L]):
    # vbor verbatim with two unary-minus unops injected on reused scalars: same
    # boundary-connector structure as _vbor (known to tile e2e through the
    # descent), so the e2e isolates the descent's TileUnop path.
    for i in range(L):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        a1 = -a1  # unary-minus unop on a reused scalar
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        c1 = -c1  # unary-minus unop
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@dace.program
def _const_store(a: dace.float64[L]):
    for i in range(L):
        a[i] = 3.0


@dace.program
def _s231(aa: dace.float64[L, L], bb: dace.float64[L, L]):
    for i in range(L):
        for j in range(1, L):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def _build_nested(prog, name):
    """Build + force every innermost body into a NestedSDFG (P1) so the descent
    owns it — the shape the orchestrator gets once nest-everything is on."""
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    from dace.transformation.passes.split_tasklets import SplitTasklets
    from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import normalize_loop_nests
    sdfg = _build(prog, name)
    normalize_loop_nests(sdfg)
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    SplitTasklets().apply_pass(sdfg, {})
    NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    return sdfg


def _tile_via_descent(sdfg, W):
    """Run the tile prep + descent (no EmitTileOps) + lib-node expansion."""
    from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
    from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
    from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import PromoteNSDFGBodyToTiles
    from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths
    res = MarkTileDims(widths=W).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    StrideMapByTileWidths(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    handled = PromoteNSDFGBodyToTiles(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    sdfg.expand_library_nodes()
    return handled


@pytest.mark.parametrize("n", [16, 17])
def test_const_store_to_output_descent_matches_reference(n):
    """``a[i] = 3.0`` into an output connector, nested into an NSDFG body, tiles
    through the descent (const-fill tile + masked TileStore) and matches the
    reference (n=17 forces the masked tail — the mask must keep the const out of
    the OOB lanes of the output array)."""
    ref = _build(_const_store, f"const_ref{n}")
    vec = _build_nested(_const_store, f"const_vec{n}")
    handled = _tile_via_descent(vec, (8, ))
    assert handled, "expected the const-store map to be handled by the descent"
    ra = np.full(n, -1.0)
    va = np.full(n, -1.0)
    ref.compile()(a=ra, LEN_2D=n)
    vec.compile()(a=va, LEN_2D=n)
    np.testing.assert_allclose(va, ra, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize("n", [16, 17])
def test_unop_chain_descent_matches_reference(n):
    """A reused-scalar chain with unary-minus unops tiles through the descent
    (NSDFG body) and matches the unvectorized reference (n=17 forces the
    masked i-tail). Proves the descent's TileUnop path is numerically correct."""
    rng = np.random.default_rng(seed=n)
    arrays = {k: rng.random(n) for k in "abcde"}
    arrays["x"] = np.zeros(n)
    ref = _build(_unop_chain, f"unop_ref{n}")
    vec = _build(_unop_chain, f"unop_vec{n}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

    rf = {k: v.copy() for k, v in arrays.items()}
    vf = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**rf, LEN_2D=n)
    vec.compile()(**vf, LEN_2D=n)
    np.testing.assert_allclose(vf["x"], rf["x"], rtol=1e-12, atol=1e-12)


def test_unop_chain_emits_tile_unop():
    """The descent emits a TileUnop for the unary-minus tasklets in a reused-
    scalar (NSDFG-body) chain — the capability EmitTileOps had but the descent
    previously lacked. Because the body is an NSDFG the descent owns it (and
    EmitTileOps is threaded to skip it), so a TileUnop here proves the descent
    produced it."""
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
    from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
    from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import PromoteNSDFGBodyToTiles
    from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths

    sdfg = _build(_unop_chain, "unop_struct")
    W = (8, )
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    res = MarkTileDims(widths=W).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    StrideMapByTileWidths(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    handled = PromoteNSDFGBodyToTiles(widths=W).apply_pass(sdfg, {"MarkTileDims": res})

    assert handled, "expected the unop-chain map to be handled by the descent"
    unops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileUnop)]
    assert unops, "expected TileUnop nodes for the unary-minus tasklets"


def test_s231_loopregion_body_still_clean_skips():
    """The carried-dep LoopRegion body (s231) is NOT a flat body, so the
    descent leaves it; ``EmitTileOps`` raises ``NotImplementedError``
    (the kernel stays a clean skip until its own slice lands)."""
    sdfg = _build(_s231, "s231_skip")
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(sdfg, {})
