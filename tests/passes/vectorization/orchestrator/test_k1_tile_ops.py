# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=1 tile-op generation through :class:`VectorizeCPUMultiDim`.

Production routing sends K=1 kernels to the existing tasklet emission
(``VectorizeCPU``) and reserves the tile libnodes for genuine K>=2 tiles
(see the harness ``tile_nodes`` arm). But ``VectorizeCPUMultiDim`` must
still be *able* to emit correct 1D tiles — every 1D kernel is a degenerate
1D tile. These tests pin that capability directly: a handful of simple
elementwise kernels (the shapes ``VectorizeCPU`` covers) plus the
NSDFG-body ``vbor`` chain are tiled with ``widths=(8,)`` and checked
numerically against the unvectorized reference, with a structural check
that tile lib nodes were actually emitted (not a scalar fallback).
"""

import pytest
pytest.importorskip("dace.transformation.passes.vectorization.emit_tile_ops", reason="legacy descent module deleted -- this test is frozen")
import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileStore
from dace.transformation.interstate import LoopToMap
from tests.corpus import tsvc
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern, )
# from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps  (frozen -- module deleted)
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import PromoteNSDFGBodyToTiles
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

from tests.passes.vectorization.passes.test_nest_innermost_map_body import bare_tasklet_body as scale

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """``c = a + 2 * b`` -- inlined fixture (previously imported from the
    deleted ``test_tile_map_by_num_cores.py``)."""
    for i in dace.map[0:N]:
        c[i] = a[i] + 2.0 * b[i]


@dace.program
def triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = b[i] + c[i] * d[i]


@dace.program
def vbor(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N],
         x: dace.float64[N]):
    for i in range(N):
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


_KERNELS = {
    "axpy": (axpy, ("a", "b", "c"), "c"),
    "scale": (scale, ("a", "b"), "b"),
    "triad": (triad, ("a", "b", "c", "d"), "a"),
    "vbor": (vbor, ("a", "b", "c", "d", "e", "x"), "x"),
}


def _build(prog, name):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.name = name
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    return sdfg


@pytest.mark.parametrize("kernel", list(_KERNELS))
@pytest.mark.parametrize("n", [16, 17])
def test_k1_tile_op_matches_reference(kernel, n):
    """``VectorizeCPUMultiDim(widths=(8,))`` produces a numerically
    correct 1D tile lowering (n=17 exercises the masked tail)."""
    prog, names, out = _KERNELS[kernel]
    rng = np.random.default_rng(seed=tsvc.stable_seed((kernel, n)))
    arrays = {nm: rng.random(n) for nm in names}
    arrays[out] = np.zeros(n)
    ref = _build(prog, f"k1_{kernel}_ref{n}")
    vec = _build(prog, f"k1_{kernel}_vec{n}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

    rf = {k: v.copy() for k, v in arrays.items()}
    vf = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**rf, N=n)
    vec.compile()(**vf, N=n)
    np.testing.assert_allclose(vf[out], rf[out], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kernel", list(_KERNELS))
def test_k1_emits_tile_ops(kernel):
    """The K=1 pipeline leaves tile lib nodes (a real tile lowering, not a
    scalar fallback) before ``expand_library_nodes``."""
    prog, _names, _out = _KERNELS[kernel]
    sdfg = _build(prog, f"k1_{kernel}_struct")
    W = (8, )
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    res = MarkTileDims(widths=W).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    StrideMapByTileWidths(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    # Thread the descent's handled-map set into EmitTileOps exactly as the
    # orchestrator pipeline does, so EmitTileOps skips an already-tiled
    # NSDFG body (vbor) instead of wrongly raising "no binop".
    handled = PromoteNSDFGBodyToTiles(widths=W).apply_pass(sdfg, {"MarkTileDims": res})
    EmitTileOps(widths=W).apply_pass(sdfg, {"MarkTileDims": res, "PromoteNSDFGBodyToTiles": handled})

    binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
    loads = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad)]
    stores = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileStore)]
    assert binops, f"{kernel}: expected TileBinop nodes"
    assert loads, f"{kernel}: expected TileLoad nodes"
    assert stores, f"{kernel}: expected TileStore nodes"
