# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Kernels that need ``TileLoad`` (gather) / ``TileStore`` (scatter) / ``TileReduce``.

The **1D** and **2D data gathers** (``a[i] = b[idx[i]] + ...``) land through the walker --
``WidenAccesses`` materialises the per-lane index tile, ``InsertTileLoadStore`` collapses the
``b[__sym]`` reads into a :class:`TileLoad` carrying ``gather_dims`` -- so those tests assert
end-to-end numerical equivalence against the unvectorized reference, plus the lowered shape for
the 1D case (equal numbers alone cannot tell a real gather from a scalar fallback).

The **SpMV** and **WCR-reduction** families assert only that the orchestrator runs to completion:
it lowers what it can and declines the reduction fold with a warning, leaving the kernel correct
and un-tiled. When their ``TileReduce`` slice lands, those tests gain equivalence assertions too.
"""

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileLoad, TileReduce
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    classify_tile_access,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )

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
def _spmv_kernel(y: dace.float64[_N], A: dace.float64[_N, _NNZ], x: dace.float64[_N], col: dace.int32[_NNZ]):
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
    cls = classify_tile_access(indirect, array_strides=(1, ), tile_iter_vars=("i", ))
    assert cls.kind == TileAccessKind.GATHER


@pytest.mark.parametrize("n", [16, 17, 23])
def test_vectorize_cpu_multi_dim_1d_indirect_stencil_matches_reference(n):
    """1D indirect stencil ``a[i] = b[idx[i]] + 1.0`` lowers via the
    gather-descent slice and matches the unvectorized reference.

    The compute lives in a body NSDFG; ``PromoteNSDFGBodyToTiles`` fans
    the per-lane index ``idx[i]`` into a ``(W,)`` index tile and collapses
    the ``b[idx[i]]`` reads into a :class:`TileLoad` (gather). The ``n=17, 23``
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
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(vec, {})

    ref.compile()(a=a_ref, b=b.copy(), idx=idx.copy(), N=n)
    vec.compile()(a=a_vec, b=b.copy(), idx=idx.copy(), N=n)
    np.testing.assert_allclose(a_vec, a_ref, rtol=1e-12, atol=1e-12)


def test_1d_indirect_stencil_emits_tilegather():
    """The 1D data gather lowers to a :class:`TileLoad` (gather) lib node naming the gathered dim,
    fed by a second :class:`TileLoad` that materialises the per-lane index tile. Checked on the
    orchestrator's output, before ``expand_library_nodes`` collapses both to their ``pure`` form --
    the numerical test above cannot see the difference between a real gather and a scalar fallback
    that happens to compute the same values."""
    sdfg = _build_1d_indirect_stencil()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    loads = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, TileLoad)]
    gathers = [node for node in loads if tuple(node.gather_dims)]
    assert gathers, f"expected a TileLoad (gather) for the 1D data gather, got {[n.label for n in loads]}"
    assert all(tuple(node.gather_dims) == (0, ) for node in gathers), \
        f"the only gathered dim is the single data dim: {[tuple(n.gather_dims) for n in gathers]}"
    assert len(loads) > len(gathers), "the gather reads its lane indices through a TileLoad of its own"


@pytest.mark.parametrize("m,n", [(16, 16), (8, 24), (12, 17)])
def test_vectorize_cpu_multi_dim_2d_indirect_stencil_matches_reference(m, n):
    """2D indirect stencil ``c[i, j] = a[idx[i, j], j] + 1.0`` lowers via
    the K-aware gather-descent slice (multi-dim index source: ``idx``
    indexed by both tile vars) and matches the unvectorized reference.

    The descent's K-shape index fan-out widens the ``idx`` boundary
    connector to a ``(W_0, W_1)`` strided view of the source array, then
    the ``multidim_gather_dims`` path subscript-substitutes each
    tile iter-var-bound inner-array dim with its ``__l<p>`` so lane
    ``(l0, l1)`` reads ``idx[i + l0, j + l1]``. Non-W-divisible trips
    exercise the masked tail."""
    rng = np.random.default_rng(seed=m * 100 + n)
    a = rng.random((m, n))
    idx = rng.integers(0, m, size=(m, n)).astype(np.int32)
    c_ref = np.zeros((m, n))
    c_vec = np.zeros((m, n))

    ref = _build_2d_indirect_stencil()
    ref.name = f"ind2d_ref{m}_{n}"
    vec = _build_2d_indirect_stencil()
    vec.name = f"ind2d_vec{m}_{n}"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=ISA.SCALAR)).apply_pass(vec, {})

    ref.compile()(a=a.copy(), c=c_ref, idx=idx.copy(), M=m, N=n)
    vec.compile()(a=a.copy(), c=c_vec, idx=idx.copy(), M=m, N=n)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)


def test_vectorize_cpu_multi_dim_accepts_spmv():
    """SpMV (gather + reduction): the orchestrator must run to completion rather than raise. It
    declines the reduction fold -- the addend never widened to a tile, so no tile-op reduction
    shape matches -- and leaves the kernel correct and un-tiled. The prior contract was a hard
    ``NotImplementedError``; declining quietly and correctly is what this pins."""
    sdfg = _build_spmv()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})

    rng = np.random.default_rng(seed=20260824)
    n, nnz = 12, 21  # neither extent divides either width: the decline must hold on the tail too
    A = rng.random((n, nnz))
    x = rng.random(n)
    col = rng.integers(0, n, size=nnz).astype(np.int32)
    y = np.zeros(n)
    sdfg(y=y, A=A.copy(), x=x.copy(), col=col.copy(), N=n, NNZ=nnz)
    np.testing.assert_allclose(y, A @ x[col], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("widths", [(8, ), (4, 8)])
def test_reduction_with_wcr_lowers_to_tile_reduce(widths):
    """Element-wise sum reduction ``s += a[i]`` uses WCR; the orchestrator's
    ``NormalizeWCRSource`` pre-pass + ``EmitTileOps`` reduction emission now
    lower this to a ``TileReduce`` writing a private scalar that flows out
    via the surviving WCR edge to MapExit (OpenMP reduction). Asserts the
    pipeline accepts the kernel (no ``NotImplementedError``) — prior
    contract was the inverse refusal."""
    N = dace.symbol("N")
    sdfg = dace.SDFG(f"reduce_{'x'.join(str(w) for w in widths)}")
    sdfg.add_array("a", (N, ) if len(widths) == 1 else (N, N), dace.float64)
    sdfg.add_array("s", (1, ), dace.float64)
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
            {
                "i": "0:N",
                "j": "0:N"
            },
            {"_a": dace.Memlet("a[i, j]")},
            "_s = _a",
            {"_s": dace.Memlet("s[0]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=ISA.SCALAR)).apply_pass(sdfg, {})

    assert any(isinstance(node, TileReduce) for node, _ in sdfg.all_nodes_recursive()), \
        "the WCR fold must lower to a TileReduce, not stay a scalar accumulation"
    rng = np.random.default_rng(seed=len(widths))
    n = 37  # divides no width: the masked tail carries part of the sum
    a = rng.random(n if len(widths) == 1 else (n, n))
    total = np.zeros(1)
    sdfg(a=a.copy(), s=total, N=n)
    np.testing.assert_allclose(total[0], a.sum(), rtol=1e-12, atol=1e-12)
