# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Gap tests pinning the v2 MVP's reject behaviour on kernels that need
``TileGather`` / ``TileScatter`` / ``TileReduce``.

Three kernel families are checked rigorously here. Each one is
expected to fail through :class:`VectorizeCPUMultiDim` today with a
loud :class:`NotImplementedError` from :class:`EmitTileOps` (the
operand classification returns ``UNRECOGNIZED`` for indirect / reduction
patterns, which the v2 MVP refuses). The future ``TileGather`` /
``TileScatter`` / ``TileReduce`` lib-node slices flip these tests
green.

This file is the executable contract: when the post-MVP slices land,
each ``pytest.raises(NotImplementedError)`` block must be replaced by
an end-to-end numerical equivalence assertion.
"""
import numpy as np
import pytest

import dace
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


def test_vectorize_cpu_multi_dim_refuses_1d_indirect_stencil():
    """1D indirect stencil ``a[i] = b[idx[i]] + 1.0`` raises
    ``NotImplementedError`` (T-post ``TileGather`` + new operand kind
    ``Gather`` will flip this green).

    The compute lives in a body NSDFG, so ``PromoteNSDFGBodyToTiles``
    (the flat-body descent) reaches the indirect ``b[idx[i]]`` access
    first and refuses the non-box load; either that pass or ``EmitTileOps``
    surfaces the refusal."""
    sdfg = _build_1d_indirect_stencil()
    with pytest.raises(NotImplementedError) as ei:
        VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(sdfg, {})
    msg = str(ei.value)
    assert ("EmitTileOps" in msg or "PromoteNSDFGBodyToTiles" in msg or "Unrecognized" in msg
            or "perfect-box" in msg or "Gather" in msg or "input" in msg)


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
