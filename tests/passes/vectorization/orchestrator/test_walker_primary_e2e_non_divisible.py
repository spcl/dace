# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical tests where the kernel size is NOT divisible by the tile widths.

All previous e2e tests used sizes that divide the widths cleanly (mask all-True). This
file exercises the masked-tail path: TileLoad / TileStore / Tile{Binop, Unop, ITE,
Reduce} must honour the iter_mask to skip out-of-range lanes.

Per the walker pipeline's ``GenerateTileIterationMask`` placement (inside the body
NSDFG, design 7.4): the mask is computed per outer-tile-iteration as
``mask[l] = (ii + l < N)`` and consumed by every masked lib node.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)


def _build_k1_axpy(N):
    """``B[i] = A[i] * 2.0 + C[i]`` -- K=1, varying N."""
    sdfg = dace.SDFG("k1_axpy_nd")
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    sdfg.add_array("C", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_c"}, {"_b"}, "_b = _a * 2.0 + _c")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet("C[ii]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


def _build_k2_axpy(M, N):
    """``B[i, j] = A[i, j] * 2.0 + C[i, j]`` -- K=2 axpy, varying (M, N)."""
    sdfg = dace.SDFG("k2_axpy_nd")
    sdfg.add_array("A", (M, N), dace.float64, transient=False)
    sdfg.add_array("B", (M, N), dace.float64, transient=False)
    sdfg.add_array("C", (M, N), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{M}", "jj": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_c"}, {"_b"}, "_b = _a * 2.0 + _c")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet("C[ii, jj]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    return sdfg


@pytest.mark.parametrize("N", [9, 17, 23, 31])
def test_k1_axpy_non_divisible_matches_reference(N):
    """K=1 axpy with N not divisible by W=8: masked-tail must produce correct output
    for the last partial tile."""
    rng = np.random.default_rng(seed=N)
    a = rng.random(N)
    c = rng.random(N)
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)
    ref = _build_k1_axpy(N)
    ref.name = f"k1_axpy_nd_ref_{N}"
    vec = _build_k1_axpy(N)
    vec.name = f"k1_axpy_nd_vec_{N}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=a.copy(), C=c.copy(), B=b_ref)
    vec.compile()(A=a.copy(), C=c.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("M,N", [(9, 9), (8, 9), (9, 8), (10, 17), (17, 23)])
def test_k2_axpy_non_divisible_matches_reference(M, N):
    """K=2 axpy with (M, N) not divisible by widths=(8, 8): both dims' masked-tail
    cases must be handled. Includes:
    * (9, 9): both dims have a 1-element tail per tile.
    * (8, 9), (9, 8): only one dim has a tail.
    * (10, 17), (17, 23): larger non-divisible cases (multiple tiles + tail)."""
    rng = np.random.default_rng(seed=M * 100 + N)
    a = rng.random((M, N))
    c = rng.random((M, N))
    b_ref = np.zeros((M, N))
    b_vec = np.zeros((M, N))
    ref = _build_k2_axpy(M, N)
    ref.name = f"k2_axpy_nd_ref_{M}x{N}"
    vec = _build_k2_axpy(M, N)
    vec.name = f"k2_axpy_nd_vec_{M}x{N}"
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=a.copy(), C=c.copy(), B=b_ref)
    vec.compile()(A=a.copy(), C=c.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)
