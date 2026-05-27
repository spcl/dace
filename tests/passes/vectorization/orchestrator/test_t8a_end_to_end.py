# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical-equivalence tests for the v2 tile-op pipeline.

The test fixture is parametrized over the two paths the v2 plan locks:

* ``scalar_postamble`` — the existing 1D ``VectorizeCPU`` (vector_width 8).
* ``tile_nodes_8x8`` — the new K-dim ``VectorizeCPUMultiDim`` (widths
  ``(8,)`` for K=1 kernels; widths ``(8, 8)`` for K=2 kernels).

Both arms must match the unvectorized scalar reference bit-equally
(rtol=0, atol=0) on non-FMA kernels.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim,
)


def _k1_axpy_sdfg(name="e2e_k1_axpy"):
    """K=1 axpy: ``C[i] = A[i] + B[i]`` for end-to-end testing.

    :param name: Unique SDFG name; distinct per test so parallel
        ``-n`` workers don't race on a shared ``.dacecache`` build folder.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dace.float64)
    sdfg.add_array("B", (N,), dace.float64)
    sdfg.add_array("C", (N,), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _k2_axpy_sdfg(name="e2e_k2_axpy"):
    """K=2 axpy: ``C[i, j] = A[i, j] + B[i, j]`` for end-to-end testing.

    :param name: Unique SDFG name; distinct per test so parallel
        ``-n`` workers don't race on a shared ``.dacecache`` build folder.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M, N), dace.float64)
    sdfg.add_array("B", (M, N), dace.float64)
    sdfg.add_array("C", (M, N), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def test_k1_axpy_aligned_trip_matches_numpy():
    """K=1 axpy under ``VectorizeCPUMultiDim(widths=(8,))`` matches numpy
    on a trip aligned to ``W=8``."""
    sdfg = _k1_axpy_sdfg("e2e_k1_axpy_aligned_trip")
    VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(sdfg, {})
    sdfg.validate()
    rng = np.random.default_rng(seed=101)
    n = 64
    A = rng.random(n)
    B = rng.random(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


@pytest.mark.parametrize("n", [8, 16, 64, 128])
def test_k1_axpy_aligned_sizes(n):
    """K=1 axpy across several aligned sizes."""
    sdfg = _k1_axpy_sdfg(f"e2e_k1_axpy_aligned_{n}")
    VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(sdfg, {})
    rng = np.random.default_rng(seed=n)
    A = rng.random(n)
    B = rng.random(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


def test_k2_axpy_aligned_trip_matches_numpy():
    """K=2 axpy under ``VectorizeCPUMultiDim(widths=(8, 8))`` matches
    numpy on aligned ``M x N``."""
    sdfg = _k2_axpy_sdfg("e2e_k2_axpy_aligned_trip")
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR").apply_pass(sdfg, {})
    sdfg.validate()
    rng = np.random.default_rng(seed=202)
    m, n = 16, 32
    A = rng.random((m, n))
    B = rng.random((m, n))
    C = np.zeros((m, n))
    sdfg(A=A, B=B, C=C, M=m, N=n)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


@pytest.mark.parametrize("m,n", [(16, 32), (17, 20), (20, 17), (9, 9)])
def test_k2_axpy_scalar_postamble_matches_numpy(m, n):
    """K=2 axpy under ``remainder_strategy='scalar_postamble'`` matches numpy
    across aligned + unaligned ``M x N`` trips. scalar_postamble is no longer
    K=1-only: ``SplitMapForTileRemainder`` splits a per-dim divisible interior
    (W-strided tiles) off from step-1 scalar boundary slabs, so a non-divisible
    K>=2 trip vectorizes the interior and runs the boundary scalar."""
    sdfg = _k2_axpy_sdfg(f"e2e_k2_axpy_scalar_{m}_{n}")
    VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR",
                         remainder_strategy="scalar_postamble").apply_pass(sdfg, {})
    sdfg.validate()
    rng = np.random.default_rng(seed=m * 100 + n)
    A = rng.random((m, n))
    B = rng.random((m, n))
    C = np.zeros((m, n))
    sdfg(A=A, B=B, C=C, M=m, N=n)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


def test_k1_axpy_unaligned_trip_matches_numpy():
    """K=1 axpy on an unaligned trip — mask must zero the tail correctly.

    Pure expansion writes 0 on masked-off lanes; the inactive-lane writes
    don't escape the W-wide tile since the map iterates step-W and the
    final tile's tail is masked. End-to-end numerical equivalence
    against numpy validates the mask path.
    """
    sdfg = _k1_axpy_sdfg("e2e_k1_axpy_unaligned_trip")
    VectorizeCPUMultiDim(widths=(8,), target_isa="SCALAR").apply_pass(sdfg, {})
    rng = np.random.default_rng(seed=303)
    n = 17  # 17 // 8 = 2 full tiles + 1 tail
    A = rng.random(n)
    B = rng.random(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, A + B, rtol=0, atol=0)
