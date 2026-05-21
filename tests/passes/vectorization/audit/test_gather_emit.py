# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end gather/scatter emission for non-perfect-box tile accesses.

A diagonal access ``a[i, i]`` (the tile var ``i`` indexes two array dims) is
NOT a perfect box, so ``classify_tile_access`` reports GATHER and
``EmitTileOps`` lowers it to a :class:`TileGather` (read) / :class:`TileScatter`
(write) over an affine per-dim index map (``_idx_k[lane] = i + lane``), rather
than a strided load. This pins the diagonal numerically against the
unvectorized reference and asserts the gather/scatter lib nodes are emitted.
"""
import copy

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileGather, TileScatter
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")


@dace.program
def _diag_kernel(aa: dace.float64[N, N], bb: dace.float64[N, N], cc: dace.float64[N, N]):
    for i in range(N):
        aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


def _prepped(tag=""):
    sdfg = copy.deepcopy(_diag_kernel.to_sdfg(simplify=False))
    if tag:
        sdfg.name = sdfg.name + f"_{tag}"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    return sdfg


def test_diagonal_emits_gather_and_scatter():
    """The diagonal read/write lower to TileGather + TileScatter lib nodes."""
    sdfg = _prepped()
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    MarkTileDims(widths=(8,)).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=(8,)).apply_pass(sdfg, {})
    StrideMapByTileWidths(widths=(8,)).apply_pass(sdfg, {})
    EmitTileOps(widths=(8,)).apply_pass(sdfg, {})
    gathers = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileGather)]
    scatters = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileScatter)]
    assert len(gathers) >= 1, "expected a TileGather for the diagonal read"
    assert len(scatters) == 1, "expected a TileScatter for the diagonal write"


@pytest.mark.parametrize("n", [16, 17])
def test_diagonal_gather_numerically_matches_reference(n):
    """Diagonal gather/scatter output matches the unvectorized SDFG."""
    rng = np.random.default_rng(seed=n)
    aa = rng.random((n, n)); bb = rng.random((n, n)); cc = rng.random((n, n))
    ref_aa, vec_aa = aa.copy(), aa.copy()

    ref = _prepped(f"ref{n}")
    vec = _prepped(f"vec{n}")
    VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(vec, {})

    ref.compile()(aa=ref_aa, bb=bb.copy(), cc=cc.copy(), N=n)
    vec.compile()(aa=vec_aa, bb=bb.copy(), cc=cc.copy(), N=n)
    # Tolerance allows a 1-ULP FMA-reordering difference in ``a + b*c``.
    np.testing.assert_allclose(vec_aa, ref_aa, rtol=1e-12, atol=1e-12)
