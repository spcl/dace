# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile offset-slice load/store element offsets (Bug 16).

The cuTile ``TileGather`` / ``TileScatter`` expansions build their per-lane element
index from the block id only (``ct.arange(W) + __pid*W`` / ``ct.load(index=pid)``),
which reconstructs the block-aligned start but drops the constant offset carried
by the memlet begin. A write/read through an offset slice (``B[1:-1]``, ``A[2:]``)
therefore landed one (or more) elements too low -- silently wrong, no crash. The
fix (`cutile_tile_dim_offsets`) recovers the offset and adds it to the per-element
gather/scatter index, forcing the gather/scatter path when the offset is non-zero.

Every test runs end-to-end on the GPU (``@dace.program`` -> ``VectorizeCuTile`` ->
compile -> run) and compares against NumPy, on divisible AND non-divisible sizes
and with the offset on leading and trailing dims.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization import VectorizeCuTile

pytestmark = pytest.mark.gpu

N = dace.symbol("N")
M = dace.symbol("M")


def _lower(prog, widths, canon=False):
    """Lower a ``@dace.program`` through the cuTile pipeline to a runnable SDFG."""
    sdfg = prog.to_sdfg(simplify=True)
    if canon:
        canonicalize(sdfg)
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg.compile()


# ---------------------------------------------------------------------------
# 1-D offset store / load
# ---------------------------------------------------------------------------


@dace.program
def _store_offset_1d(A: dace.float64[N], B: dace.float64[N]):
    B[1:-1] = 2.0 * A[1:-1]


@dace.program
def _load_offset_1d(A: dace.float64[N], B: dace.float64[N]):
    B[:-2] = 2.0 * A[2:]


@pytest.mark.parametrize("n", [40, 43, 64, 71])
def test_store_offset_1d(n):
    """``B[1:-1] = 2*A[1:-1]`` must not shift the interior write."""
    csdfg = _lower(_store_offset_1d, widths=(8,))
    rng = np.random.default_rng(n)
    A = rng.random(n)
    B = np.full(n, -7.0)
    csdfg(A=A, B=B, N=n)
    expected = np.full(n, -7.0)
    expected[1:-1] = 2.0 * A[1:-1]
    np.testing.assert_allclose(B, expected, rtol=1e-14)


@pytest.mark.parametrize("n", [40, 43, 64, 71])
def test_load_offset_1d(n):
    """``B[:-2] = 2*A[2:]`` must read A from the +2 offset, not from base 0."""
    csdfg = _lower(_load_offset_1d, widths=(8,))
    rng = np.random.default_rng(n + 1)
    A = rng.random(n)
    B = np.zeros(n)
    csdfg(A=A, B=B, N=n)
    expected = np.zeros(n)
    expected[:-2] = 2.0 * A[2:]
    np.testing.assert_allclose(B, expected, rtol=1e-14)


def test_store_offset_1d_matches_and_differs_from_unshifted():
    """Before/after guard: the offset write differs from the base-0 write.

    A base-0 write (``B[:-2]``) and an offset write (``B[1:-1]``) of the same
    extent produce *different* arrays; the bug made them identical (both landed
    at base 0). This asserts the offset result equals its own NumPy reference
    and is NOT equal to the base-0 reference.
    """
    n = 48
    csdfg = _lower(_store_offset_1d, widths=(8,))
    rng = np.random.default_rng(7)
    A = rng.random(n)
    B = np.zeros(n)
    csdfg(A=A, B=B, N=n)
    offset_ref = np.zeros(n)
    offset_ref[1:-1] = 2.0 * A[1:-1]
    base0_ref = np.zeros(n)
    base0_ref[:-2] = 2.0 * A[:-2]
    np.testing.assert_allclose(B, offset_ref, rtol=1e-14)
    assert not np.allclose(B, base0_ref)


# ---------------------------------------------------------------------------
# 2-D offset store: offset on the trailing (inner) and leading (outer) dim
# ---------------------------------------------------------------------------


@dace.program
def _store_offset_2d_trailing(A: dace.float64[M, N], B: dace.float64[M, N]):
    B[:, 1:-1] = 2.0 * A[:, 1:-1]


@dace.program
def _store_offset_2d_leading(A: dace.float64[M, N], B: dace.float64[M, N]):
    B[1:-1, :] = 2.0 * A[1:-1, :]


@pytest.mark.parametrize("mm,nn", [(16, 40), (13, 43)])
def test_store_offset_2d_trailing(mm, nn):
    """Offset on the inner dim: ``B[:, 1:-1]``."""
    csdfg = _lower(_store_offset_2d_trailing, widths=(8, 8))
    rng = np.random.default_rng(mm * 100 + nn)
    A = rng.random((mm, nn))
    B = np.full((mm, nn), -3.0)
    csdfg(A=A, B=B, M=mm, N=nn)
    expected = np.full((mm, nn), -3.0)
    expected[:, 1:-1] = 2.0 * A[:, 1:-1]
    np.testing.assert_allclose(B, expected, rtol=1e-14)


@pytest.mark.parametrize("mm,nn", [(16, 40), (13, 43)])
def test_store_offset_2d_leading(mm, nn):
    """Offset on the outer dim: ``B[1:-1, :]``."""
    csdfg = _lower(_store_offset_2d_leading, widths=(8, 8))
    rng = np.random.default_rng(mm * 200 + nn)
    A = rng.random((mm, nn))
    B = np.full((mm, nn), 5.0)
    csdfg(A=A, B=B, M=mm, N=nn)
    expected = np.full((mm, nn), 5.0)
    expected[1:-1, :] = 2.0 * A[1:-1, :]
    np.testing.assert_allclose(B, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# jacobi_1d stencil end-to-end (offset loads A[:-2], A[1:-1], A[2:] + offset store)
# ---------------------------------------------------------------------------


@dace.program
def _jacobi_1d(TSTEPS: dace.int64, A: dace.float64[N], B: dace.float64[N]):
    for _ in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@pytest.mark.parametrize("n", [32, 34])
def test_jacobi_1d_end_to_end(n):
    """Full 3-point stencil: the three offset loads and the offset store must
    each land at their correct global position.

    Runs WITHOUT canonicalize: canonicalize miscompiles this in-place stencil
    on some (hash-seed/allocation-order dependent) trajectories, which made
    this test flaky when run after the other tests in this file. See
    ``dace/transformation/passes/canonicalize/SOUNDNESS_BUG_INPLACE_STENCIL.md``
    and the skipped ``_canonicalized`` variant below.
    """
    tsteps = 4
    csdfg = _lower(_jacobi_1d, widths=(8,), canon=False)
    rng = np.random.default_rng(n)
    A = rng.random(n)
    B = rng.random(n)
    A_ref = A.copy()
    B_ref = B.copy()
    for _ in range(1, tsteps):
        B_ref[1:-1] = 0.33333 * (A_ref[:-2] + A_ref[1:-1] + A_ref[2:])
        A_ref[1:-1] = 0.33333 * (B_ref[:-2] + B_ref[1:-1] + B_ref[2:])
    csdfg(TSTEPS=tsteps, A=A, B=B, N=n)
    np.testing.assert_allclose(A, A_ref, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(B, B_ref, rtol=1e-11, atol=1e-12)


@pytest.mark.skip(reason="canonicalize() miscompiles this in-place stencil on some process "
                  "trajectories (order/hash-seed dependent): a duplicated B-statement is "
                  "scheduled after the A-update and rewrites B from stale operands. The "
                  "post-canonicalize SDFG is already wrong on the plain CPU backend. See "
                  "dace/transformation/passes/canonicalize/SOUNDNESS_BUG_INPLACE_STENCIL.md")
@pytest.mark.parametrize("n", [32, 34])
def test_jacobi_1d_end_to_end_canonicalized(n):
    """Same stencil through canonicalize(); kept as the repro for the
    canonicalize soundness bug. Un-skip once canonicalize is fixed."""
    tsteps = 4
    csdfg = _lower(_jacobi_1d, widths=(8,), canon=True)
    rng = np.random.default_rng(n)
    A = rng.random(n)
    B = rng.random(n)
    A_ref = A.copy()
    B_ref = B.copy()
    for _ in range(1, tsteps):
        B_ref[1:-1] = 0.33333 * (A_ref[:-2] + A_ref[1:-1] + A_ref[2:])
        A_ref[1:-1] = 0.33333 * (B_ref[:-2] + B_ref[1:-1] + B_ref[2:])
    csdfg(TSTEPS=tsteps, A=A, B=B, N=n)
    np.testing.assert_allclose(A, A_ref, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(B, B_ref, rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    test_store_offset_1d(43)
    test_load_offset_1d(43)
    test_store_offset_1d_matches_and_differs_from_unshifted()
    test_store_offset_2d_trailing(13, 43)
    test_store_offset_2d_leading(13, 43)
    test_jacobi_1d_end_to_end(34)
    print("all offset regression tests passed")
