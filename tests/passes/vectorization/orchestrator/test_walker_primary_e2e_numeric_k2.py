# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=2 (and a few K=3) end-to-end numerical-equivalence smoke for the orchestrator.

K=2 matrix bodies (copy / axpy / unop / mixed) are the load-bearing case beyond
K=1: per-dim widths, two iter vars, and inner-tile semantics that didn't exist
for the 1-D path. K=3 is a few-test sanity (not a full corpus) per user
direction "limited number of K=3 (not many)".

Each test:
1. Build the unvectorized SDFG, compile, run -> reference.
2. Build the same SDFG, run ``VectorizeCPUMultiDim(widths=(K1, K0))`` over it,
   compile, run on the same input -> vectorized output.
3. Assert ``rtol=1e-12`` numerical equivalence.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)

# ----- K=2 kernels -------------------------------------------------------


def _build_k2_copy_sdfg(M, N):
    """``B[i, j] = A[i, j]``."""
    sdfg = dace.SDFG("copy_k2_numeric")
    sdfg.add_array("A", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    return sdfg


def _build_k2_axpy_sdfg(M, N):
    """``C[i, j] = A[i, j] + B[i, j]``."""
    sdfg = dace.SDFG("axpy_k2_numeric")
    sdfg.add_array("A", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    return sdfg


def _build_k2_unop_sdfg(M, N):
    """``C[i, j] = abs(A[i, j])``."""
    sdfg = dace.SDFG("unop_k2_numeric")
    sdfg.add_array("A", (M, N), dace.float64, transient=False)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a"}, {"_c"}, "_c = abs(_a)")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    return sdfg


def _build_k2_mixed_sdfg(M, N):
    """``C[i, j] = (A[i, j] + B[i, j]) * A[i, j]`` -- simple chain (no math intrinsic).

    A two-step expression that exercises the multi-op tasklet path without
    requiring SplitTasklets to break complex math-call chains.
    """
    sdfg = dace.SDFG("mixed_k2_numeric")
    sdfg.add_array("A", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_b", "_a2"}, {"_c"}, "_c = (_a + _b) * _a2")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(a, me, t, dst_conn="_a2", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    return sdfg


def _run(sdfg, **kwargs):
    return sdfg.compile()(**kwargs)


@pytest.mark.parametrize("M,N", [(8, 8), (16, 8), (8, 16), (16, 16)])
def test_k2_copy_matches_reference(M, N):
    rng = np.random.default_rng(seed=M * 100 + N)
    a = rng.random((M, N))
    b_ref = np.zeros((M, N))
    b_vec = np.zeros((M, N))
    ref = _build_k2_copy_sdfg(M, N)
    ref.name = f"copy_k2_ref_{M}_{N}"
    vec = _build_k2_copy_sdfg(M, N)
    vec.name = f"copy_k2_vec_{M}_{N}"
    VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b_ref)
    _run(vec, A=a.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("M,N", [(8, 8), (16, 16)])
def test_k2_axpy_matches_reference(M, N):
    rng = np.random.default_rng(seed=M * 100 + N + 1)
    a = rng.random((M, N))
    b = rng.random((M, N))
    c_ref = np.zeros((M, N))
    c_vec = np.zeros((M, N))
    ref = _build_k2_axpy_sdfg(M, N)
    ref.name = f"axpy_k2_ref_{M}_{N}"
    vec = _build_k2_axpy_sdfg(M, N)
    vec.name = f"axpy_k2_vec_{M}_{N}"
    VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b.copy(), C=c_ref)
    _run(vec, A=a.copy(), B=b.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("M,N", [(8, 8), (16, 16)])
def test_k2_unop_matches_reference(M, N):
    rng = np.random.default_rng(seed=M * 100 + N + 2)
    a = rng.random((M, N)) - 0.5
    c_ref = np.zeros((M, N))
    c_vec = np.zeros((M, N))
    ref = _build_k2_unop_sdfg(M, N)
    ref.name = f"unop_k2_ref_{M}_{N}"
    vec = _build_k2_unop_sdfg(M, N)
    vec.name = f"unop_k2_vec_{M}_{N}"
    VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), C=c_ref)
    _run(vec, A=a.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("M,N", [(8, 8), (16, 16)])
def test_k2_mixed_chain_matches_reference(M, N):
    rng = np.random.default_rng(seed=M * 100 + N + 3)
    a = rng.random((M, N))
    b = rng.random((M, N))
    c_ref = np.zeros((M, N))
    c_vec = np.zeros((M, N))
    ref = _build_k2_mixed_sdfg(M, N)
    ref.name = f"mixed_k2_ref_{M}_{N}"
    vec = _build_k2_mixed_sdfg(M, N)
    vec.name = f"mixed_k2_vec_{M}_{N}"
    VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b.copy(), C=c_ref)
    _run(vec, A=a.copy(), B=b.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)


# ----- K=3 (limited) -----------------------------------------------------


def _build_k3_copy_sdfg(M, N, P):
    """``B[i, j, k] = A[i, j, k]``."""
    sdfg = dace.SDFG("copy_k3_numeric")
    sdfg.add_array("A", (M, N, P), dace.float64, transient=False)
    sdfg.add_array("B", (M, N, P), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}", "kk": f"0:{P}"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj, kk]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj, kk]"))
    return sdfg


def _build_k3_axpy_sdfg(M, N, P):
    """``C[i, j, k] = A[i, j, k] + B[i, j, k]``."""
    sdfg = dace.SDFG("axpy_k3_numeric")
    sdfg.add_array("A", (M, N, P), dace.float64, transient=False)
    sdfg.add_array("B", (M, N, P), dace.float64, transient=False)
    sdfg.add_array("C", (M, N, P), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}", "kk": f"0:{P}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj, kk]"))
    state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet("B[ii, jj, kk]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii, jj, kk]"))
    return sdfg


def test_k3_copy_matches_reference():
    """One K=3 copy sanity (limited K=3 coverage per user direction)."""
    M, N, P = 4, 4, 8
    rng = np.random.default_rng(seed=42)
    a = rng.random((M, N, P))
    b_ref = np.zeros((M, N, P))
    b_vec = np.zeros((M, N, P))
    ref = _build_k3_copy_sdfg(M, N, P)
    ref.name = "copy_k3_ref"
    vec = _build_k3_copy_sdfg(M, N, P)
    vec.name = "copy_k3_vec"
    VectorizeCPUMultiDim(widths=(2, 4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b_ref)
    _run(vec, A=a.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


def test_k3_axpy_matches_reference():
    """One K=3 axpy sanity."""
    M, N, P = 4, 4, 8
    rng = np.random.default_rng(seed=43)
    a = rng.random((M, N, P))
    b = rng.random((M, N, P))
    c_ref = np.zeros((M, N, P))
    c_vec = np.zeros((M, N, P))
    ref = _build_k3_axpy_sdfg(M, N, P)
    ref.name = "axpy_k3_ref"
    vec = _build_k3_axpy_sdfg(M, N, P)
    vec.name = "axpy_k3_vec"
    VectorizeCPUMultiDim(widths=(2, 4, 8), target_isa="SCALAR").apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b.copy(), C=c_ref)
    _run(vec, A=a.copy(), B=b.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)
