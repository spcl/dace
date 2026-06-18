# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end + structural tests for cond-mask broadcasting.

Per design 7.5 + user direction 2026-06-10: when an ITE / mask-generating condition
depends only on a SUBSET of tile dims (e.g. ``A[i] > 0.0`` with K=2 widths
``(W_0, W_1)``), the condition's natural shape is ``(W_0,)``. The lib-node operand
contract requires the full ``(W_0, W_1)`` shape, so the condition must be
**broadcast** along the unused dim.

The transients-are-either-full-tile-or-scalar invariant means the broadcast happens
at the SOURCE side -- the walker classifies the access with REPLICATE along the
unused dim and TileLoad emits a per-lane replicated load. After the comparison /
ITE, every downstream lib node sees the canonical full-tile shape.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)


def _build_k2_cond_subset_of_dims(M, N):
    """K=2 ``B[i, j] = C[i, j] if A[i] > 0 else 0`` -- condition depends only on dim 0.

    The ITE's cond input is the comparison ``A[i] > 0``; the natural shape is ``(W_0,)``
    but TileITE wants full ``(W_0, W_1)``. The pipeline must broadcast.
    """
    sdfg = dace.SDFG("k2_cond_subset")
    sdfg.add_array("A", (M, ), dace.float64, transient=False)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    c = state.add_access("C")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a", "_c"}, {"_b"}, "_b = _c if (_a > 0.0) else 0.0")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    return sdfg


def _build_k2_cond_invariant_symbol(M, N):
    """K=2 ``B[i, j] = C[i, j] if FLAG > 0 else 0`` -- condition is a runtime symbol.

    Pure Symbol comparison: no array load, just ``FLAG > 0``. The Symbol broadcast
    path makes the comparison scalar; the ITE should still broadcast to full tile.
    """
    sdfg = dace.SDFG("k2_cond_invariant_sym")
    sdfg.add_symbol("FLAG", dace.float64)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    c = state.add_access("C")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_c"}, {"_b"}, "_b = _c if (FLAG > 0.0) else 0.0")
    state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    return sdfg


@pytest.mark.parametrize("M,N", [(8, 8), (16, 16), (16, 8)])
def test_k2_cond_subset_of_dims_matches_reference(M, N):
    """K=2 ITE where cond ``A[i] > 0`` depends only on dim 0; bit-equal to ref."""
    rng = np.random.default_rng(seed=M * 100 + N)
    a = rng.standard_normal(M)
    c = rng.random((M, N))
    b_ref = np.zeros((M, N))
    b_vec = np.zeros((M, N))
    ref = _build_k2_cond_subset_of_dims(M, N)
    ref.name = f"cond_subset_ref_{M}x{N}"
    vec = _build_k2_cond_subset_of_dims(M, N)
    vec.name = f"cond_subset_vec_{M}x{N}"
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=a.copy(), C=c.copy(), B=b_ref)
    vec.compile()(A=a.copy(), C=c.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("M,N", [(8, 8), (16, 16)])
def test_k2_cond_invariant_symbol_matches_reference(M, N):
    """K=2 ITE where cond ``FLAG > 0`` is a runtime symbol; bit-equal to ref."""
    rng = np.random.default_rng(seed=M * 1000 + N)
    c = rng.random((M, N))
    b_ref = np.zeros((M, N))
    b_vec = np.zeros((M, N))
    FLAG = 1.5  # positive, so all elements get copied
    ref = _build_k2_cond_invariant_symbol(M, N)
    ref.name = f"cond_invsym_ref_{M}x{N}"
    vec = _build_k2_cond_invariant_symbol(M, N)
    vec.name = f"cond_invsym_vec_{M}x{N}"
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(C=c.copy(), B=b_ref, FLAG=FLAG)
    vec.compile()(C=c.copy(), B=b_vec, FLAG=FLAG)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


# ---- structural tests -----------------------------------------------------


def test_k2_cond_subset_loads_with_replicate_factor():
    """For ``A[i] > 0`` in a K=2 body, ``A`` must load with REPLICATE along dim 1
    (W_1 copies). Equivalently: the loaded ``A_tile`` is full-tile shape ``(W_0, W_1)``
    with the same value along ``W_1`` per ``W_0``."""
    sdfg = _build_k2_cond_subset_of_dims(16, 16)
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR", expand_tile_nodes=False).apply_pass(sdfg, {})
    body_nsdfgs = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    inner = body_nsdfgs[0].sdfg
    # Find the TileLoad that reads A.
    body_state = list(inner.states())[0]
    a_loads = [
        n for n in body_state.nodes()
        if isinstance(n, TileLoad) and any((e.data.data == "A") for e in body_state.in_edges(n))
    ]
    assert len(a_loads) == 1, f"expected one TileLoad for A, found {len(a_loads)}"
    a_load = a_loads[0]
    # Walk the bridge: TileLoad._dst -> A_tile AN -> downstream.
    dst_edges = [e for e in body_state.out_edges(a_load) if e.src_conn == "_dst"]
    assert len(dst_edges) == 1
    bridge_an = dst_edges[0].dst
    bridge_desc = inner.arrays[bridge_an.data]
    # Per design 4: the bridge transient must be either Scalar or full-tile shape.
    # For an access that's REPLICATE along one dim, the bridge IS full-tile shape
    # (it must materialise the broadcast at the load step).
    bridge_shape = tuple(int(s) for s in bridge_desc.shape)
    assert bridge_shape == (8, 8), \
        f"A's bridge transient must be full-tile (8, 8) per design (broadcast at load); got {bridge_shape}"


def test_k2_cond_invariant_symbol_is_scalar_broadcast():
    """A pure-symbol condition (no array load) compares two Symbol operands and
    produces a Scalar bool output -- the TileITE then broadcasts at expansion time
    via kind_cond=Scalar / Symbol."""
    sdfg = _build_k2_cond_invariant_symbol(16, 16)
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR", expand_tile_nodes=False).apply_pass(sdfg, {})
    body_nsdfgs = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    inner = body_nsdfgs[0].sdfg
    body_state = list(inner.states())[0]
    # Find any TileBinop in the body whose op is comparison-related; verify it has
    # at least one Symbol kind (the FLAG > 0 comparison).
    binops = [n for n in body_state.nodes() if isinstance(n, TileBinop)]
    sym_binops = [b for b in binops if "Symbol" in (b.kind_a, b.kind_b)]
    assert sym_binops, "expected at least one TileBinop with a Symbol operand (the FLAG > 0 compare)"
