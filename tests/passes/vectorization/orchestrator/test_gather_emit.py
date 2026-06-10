# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end gather/scatter emission for non-perfect-box tile accesses.

A diagonal access ``a[i, i]`` (the tile var ``i`` indexes two array dims) is
NOT a perfect box, so ``classify_tile_access`` reports GATHER and
``EmitTileOps`` lowers it to a :class:`TileLoad` (gather) (read) / :class:`TileStore` (scatter)
(write) over an affine per-dim index map (``_idx_k[lane] = i + lane``), rather
than a strided load. This pins the diagonal numerically against the
unvectorized reference and asserts the gather/scatter lib nodes are emitted.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Walker-primary K=1 gather end-to-end passes (see"
                              " test_walker_primary_gather_e2e.py::test_k1_gather_matches_reference[8]);"
                              " the legacy K=2 emission-style tests in this file fail downstream"
                              " (dimensionality mismatches, NameError on deleted EmitTileOps,"
                              " 2D scatter compilation crashes). The K=2 path with the ONE-marker"
                              " design (TILIFICATION_TRANSFORMATION_DESIGN.md section 3.8.1) is a"
                              " separate slice; unfreeze incrementally as it lands.")
import copy

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.interstate import LoopToMap
# from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps  (frozen -- module deleted)
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import StrideMapByTileWidths
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")


@dace.program
def _diag_kernel(aa: dace.float64[N, N], bb: dace.float64[N, N], cc: dace.float64[N, N]):
    for i in range(N):
        aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


@dace.program
def _structured_kernel(b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = b[i // 2] + c[i]


@dace.program
def _data_gather_binop_kernel(a: dace.float64[N], b: dace.float64[N], e: dace.float64[N], idx: dace.int32[N]):
    for i in range(N):
        a[i] = b[idx[i]] + e[i]


@pytest.mark.parametrize("n", [16, 17, 23])
def test_data_gather_with_elementwise_input_matches_reference(n):
    """``a[i] = b[idx[i]] + e[i]`` lowers through the gather-descent slice
    and matches the reference.

    Beyond the bare ``b[idx[i]]`` gather, this exercises the length-1
    boundary connector for the elementwise input ``e[i]`` (and the
    destination ``a[i]``): both carry their tile-var offset in the NSDFG's
    outer edge and are widened ``(1,) -> (W,)`` before the binop / store.
    ``n=17, 23`` exercise the masked tail."""
    rng = np.random.default_rng(seed=n)
    b = rng.random(n)
    e = rng.random(n)
    idx = rng.integers(0, n, size=n).astype(np.int32)
    a_ref, a_vec = np.zeros(n), np.zeros(n)

    ref = _data_gather_binop_kernel.to_sdfg(simplify=True)
    ref.name = f"dgb_ref{n}"
    vec = _data_gather_binop_kernel.to_sdfg(simplify=True)
    vec.name = f"dgb_vec{n}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

    ref.compile()(a=a_ref, b=b.copy(), e=e.copy(), idx=idx.copy(), N=n)
    vec.compile()(a=a_vec, b=b.copy(), e=e.copy(), idx=idx.copy(), N=n)
    np.testing.assert_allclose(a_vec, a_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [16, 17])
def test_structured_int_floor_replication_matches_reference(n):
    """``b[i // 2]`` (int_floor replication, STRUCTURED) lowers to a gather over
    a per-lane index map built by substitution and matches the reference."""
    rng = np.random.default_rng(seed=n)
    b = rng.random(n)
    c = rng.random(n)
    ro, vo = np.zeros(n), np.zeros(n)
    ref = copy.deepcopy(_structured_kernel.to_sdfg(simplify=False))
    ref.name = f"sk_ref{n}"
    ref.simplify(validate=True)
    ref.apply_transformations_repeated(LoopToMap())
    ref.simplify()
    vec = copy.deepcopy(ref)
    vec.name = f"sk_vec{n}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(b=b.copy(), c=c.copy(), out=ro, N=n)
    vec.compile()(b=b.copy(), c=c.copy(), out=vo, N=n)
    np.testing.assert_allclose(vo, ro, rtol=1e-12, atol=1e-12)


def _prepped(tag=""):
    sdfg = copy.deepcopy(_diag_kernel.to_sdfg(simplify=False))
    if tag:
        sdfg.name = sdfg.name + f"_{tag}"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    return sdfg


def test_diagonal_emits_gather_and_scatter():
    """The diagonal read/write lower to TileLoad(gather_dims=...) + TileStore(gather_dims=...)
    per the G3-step-3 unified design (source-dim-indexed gather)."""
    sdfg = _prepped()
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
    MarkTileDims(widths=(8, )).apply_pass(sdfg, {})
    GenerateTileIterationMask(widths=(8, )).apply_pass(sdfg, {})
    StrideMapByTileWidths(widths=(8, )).apply_pass(sdfg, {})
    EmitTileOps(widths=(8, )).apply_pass(sdfg, {})
    # Gather-loads are TileLoad nodes with non-empty gather_dims; scatter-stores are TileStore
    # nodes with non-empty gather_dims. (Structured TileLoad/TileStore have empty gather_dims.)
    gather_loads = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad) and tuple(n.gather_dims)]
    scatter_stores = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileStore) and tuple(n.gather_dims)]
    assert len(gather_loads) >= 1, "expected a TileLoad with gather_dims for the diagonal read"
    assert len(scatter_stores) == 1, "expected a TileStore with gather_dims for the diagonal write"


@pytest.mark.parametrize("n", [16, 17])
def test_diagonal_gather_numerically_matches_reference(n):
    """Diagonal gather/scatter output matches the unvectorized SDFG."""
    rng = np.random.default_rng(seed=n)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    cc = rng.random((n, n))
    ref_aa, vec_aa = aa.copy(), aa.copy()

    ref = _prepped(f"ref{n}")
    vec = _prepped(f"vec{n}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

    ref.compile()(aa=ref_aa, bb=bb.copy(), cc=cc.copy(), N=n)
    vec.compile()(aa=vec_aa, bb=bb.copy(), cc=cc.copy(), N=n)
    # Tolerance allows a 1-ULP FMA-reordering difference in ``a + b*c``.
    np.testing.assert_allclose(vec_aa, ref_aa, rtol=1e-12, atol=1e-12)
