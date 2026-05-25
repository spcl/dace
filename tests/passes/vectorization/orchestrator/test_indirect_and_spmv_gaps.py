# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Gap tests pinning the v2 MVP's reject behaviour on kernels that need
``TileGather`` / ``TileScatter`` / ``TileReduce``.

Each kernel family below is checked rigorously. The **1D data gather**
``a[i] = b[idx[i]] + ...`` now lands through
:class:`PromoteNSDFGBodyToTiles` (the gather-descent slice: fan the
per-lane index into a ``(W,)`` index tile, collapse the ``b[__sym]``
reads into a :class:`TileGather`), so its test is an end-to-end
numerical equivalence assertion. The **2D / separable / SPMV (gather +
reduction)** families are still refused with a loud
:class:`NotImplementedError` (their descent + ``TileReduce`` slices are
pending); those tests stay ``pytest.raises``.

This file is the executable contract: when each remaining post-MVP
slice lands, its ``pytest.raises(NotImplementedError)`` block is
replaced by an end-to-end numerical equivalence assertion.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileGather
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    classify_tile_access,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim,
)


_N = dace.symbol("N")
_M = dace.symbol("M")
_NNZ = dace.symbol("NNZ")


@dace.program
def _k1_indirect_kernel(a: dace.float64[_N], b: dace.float64[_N], idx: dace.int32[_N]):
    """1D indirect stencil: ``a[i] = b[idx[i]] + 1.0``."""
    for i in dace.map[0:_N]:
        a[i] = b[idx[i]] + 1.0


@dace.program
def _k2_indirect_kernel(a: dace.float64[_M, _N], c: dace.float64[_M, _N], idx: dace.int32[_M, _N]):
    """2D indirect stencil: ``c[i, j] = a[idx[i, j], j] + 1.0``."""
    for i, j in dace.map[0:_M, 0:_N]:
        c[i, j] = a[idx[i, j], j] + 1.0


@dace.program
def _spmv_kernel(y: dace.float64[_N], A: dace.float64[_N, _NNZ],
                 x: dace.float64[_N], col: dace.int32[_NNZ]):
    """SpMV-style: ``y[i] = sum_k A[i, k] * x[col[k]]``."""
    for i, k in dace.map[0:_N, 0:_NNZ]:
        y[i] += A[i, k] * x[col[k]]


def _build_1d_indirect_stencil():
    """1D indirect (gather)."""
    return _k1_indirect_kernel.to_sdfg(simplify=True)


def _build_2d_indirect_stencil():
    """2D indirect on the leading dim (gather)."""
    return _k2_indirect_kernel.to_sdfg(simplify=True)


def _build_spmv():
    """SpMV (gather + reduction)."""
    return _spmv_kernel.to_sdfg(simplify=True)


def test_classify_tile_access_indirect_returns_gather():
    """The unified analysis classifies an indirect subset (``idx[i]``,
    a data-dependent / non-affine index) as :attr:`GATHER`: it is a
    non-box access — the index dim is neither an affine bijection of a
    tile iter-var nor a structured ``int_floor`` of one. (Emission of
    indirect gather is still refused by ``EmitTileOps`` — see the
    orchestrator-refusal tests below; only the *classification* lands
    here.)"""
    from dace import subsets
    from dace.symbolic import pystr_to_symbolic
    indirect = subsets.Range([(pystr_to_symbolic("idx[i]"), pystr_to_symbolic("idx[i]"), 1)])
    cls = classify_tile_access(indirect, array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.GATHER


@pytest.mark.parametrize("n", [16, 17, 23])
def test_vectorize_cpu_multi_dim_1d_indirect_stencil_matches_reference(n):
    """1D indirect stencil ``a[i] = b[idx[i]] + 1.0`` lowers via the
    gather-descent slice and matches the unvectorized reference.

    The compute lives in a body NSDFG; ``PromoteNSDFGBodyToTiles`` fans
    the per-lane index ``idx[i]`` into a ``(W,)`` index tile and collapses
    the ``b[idx[i]]`` reads into a :class:`TileGather`. The ``n=17, 23``
    cases exercise the masked tail (trip not a multiple of ``W=8``)."""
    rng = np.random.default_rng(seed=n)
    b = rng.random(n)
    idx = rng.integers(0, n, size=n).astype(np.int32)
    a_ref = np.zeros(n)
    a_vec = np.zeros(n)

    ref = _build_1d_indirect_stencil()
    ref.name = f"ind1d_ref{n}"
    vec = _build_1d_indirect_stencil()
    vec.name = f"ind1d_vec{n}"
    VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(vec, {})

    ref.compile()(a=a_ref, b=b.copy(), idx=idx.copy(), N=n)
    vec.compile()(a=a_vec, b=b.copy(), idx=idx.copy(), N=n)
    np.testing.assert_allclose(a_vec, a_ref, rtol=1e-12, atol=1e-12)


def test_1d_indirect_stencil_emits_tilegather():
    """The 1D data gather lowers to a :class:`TileGather` lib node (checked
    before ``expand_library_nodes`` collapses it to its ``pure`` form)."""
    from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
        CleanAccessNodeToScalarSliceToTaskletPattern, )
    from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
        GenerateTileIterationMask, )
    from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
    from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import (
        PromoteNSDFGBodyToTiles, )
    from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
        StrideMapByTileWidths, )
    sdfg = _build_1d_indirect_stencil()
    for p in (CleanAccessNodeToScalarSliceToTaskletPattern(), MarkTileDims(widths=(8,)),
              GenerateTileIterationMask(widths=(8,)), StrideMapByTileWidths(widths=(8,)),
              PromoteNSDFGBodyToTiles(widths=(8,))):
        p.apply_pass(sdfg, {})
    assert any(isinstance(node, TileGather) for node, _ in sdfg.all_nodes_recursive()), \
        "expected a TileGather for the 1D data gather"


def test_vectorize_cpu_multi_dim_refuses_2d_indirect_stencil():
    """2D indirect stencil raises ``NotImplementedError`` for the same
    reason — gather on the leading dim is not yet supported."""
    sdfg = _build_2d_indirect_stencil()
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(sdfg, {})


def test_vectorize_cpu_multi_dim_refuses_spmv():
    """SpMV (gather + reduction accumulator via wcr) raises
    ``NotImplementedError`` — covered by future TileGather + TileReduce
    slices."""
    sdfg = _build_spmv()
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(sdfg, {})


@pytest.mark.parametrize("widths", [(8,), (4, 8)])
def test_reduction_with_wcr_is_not_yet_supported(widths):
    """Element-wise sum reduction ``s += a[i]`` uses WCR; the v2 MVP
    has no ``TileReduce`` lib node and refuses the kernel. The R-series
    slices land this."""
    N = dace.symbol("N")
    sdfg = dace.SDFG(f"reduce_{'x'.join(str(w) for w in widths)}")
    sdfg.add_array("a", (N,) if len(widths) == 1 else (N, N), dace.float64)
    sdfg.add_array("s", (1,), dace.float64)
    state = sdfg.add_state("main")
    if len(widths) == 1:
        state.add_mapped_tasklet(
            "sum",
            {"i": "0:N"},
            {"_a": dace.Memlet("a[i]")},
            "_s = _a",
            {"_s": dace.Memlet("s[0]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
    else:
        state.add_mapped_tasklet(
            "sum2",
            {"i": "0:N", "j": "0:N"},
            {"_a": dace.Memlet("a[i, j]")},
            "_s = _a",
            {"_s": dace.Memlet("s[0]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(widths=widths, target_isa="SCALAR").apply_pass(sdfg, {})
