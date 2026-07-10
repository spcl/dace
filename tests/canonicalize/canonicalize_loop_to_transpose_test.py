# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`LoopToTranspose` -- lifting a pure tensor-permutation loop
nest to a ``Transpose`` (2-D) / ``TensorTranspose`` (N-D) library node.

A match requires a perfectly nested loop whose innermost body is a PURE copy
``out[perm(idx)] = in[idx]`` between two DISTINCT arrays, with each memlet a
point subset affine in exactly one loop variable and the loop-var<->axis map a
bijection. Anything else -- arithmetic, in-place, identity permutation, a mixed
subscript, a single loop -- must be refused.

Value-preservation is checked against a plain-Python execution of the same loop
body (not ``np.transpose``, to avoid coupling the test to that convention).
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose

N = dace.symbol('N')
M = dace.symbol('M')


def _apply(sdfg) -> int:
    res = Pipeline([LoopToTranspose()]).apply_pass(sdfg, {})
    return (res or {}).get('LoopToTranspose', 0) or 0


def _libnodes(sdfg):
    return [type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]


def _num_loops(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))


# --------------------------------------------------------------------------- #
# Positive: a pure permutation nest lifts and is value-preserving.
# --------------------------------------------------------------------------- #


@dace.program
def _transpose2d(A: dace.float64[N, M], B: dace.float64[M, N]):
    for i in range(M):
        for j in range(N):
            B[i, j] = A[j, i]


@dace.program
def _transpose3d(A: dace.float64[4, 5, 6], B: dace.float64[6, 4, 5]):
    for i in range(6):
        for j in range(4):
            for k in range(5):
                B[i, j, k] = A[j, k, i]


@dace.program
def _sub_region_transpose(A: dace.float64[8, 8], B: dace.float64[8, 8]):
    for i in range(1, 5):
        for j in range(2, 6):
            B[i, j] = A[j, i]


@dace.program
def _strided_transpose(A: dace.float64[8, 8], B: dace.float64[8, 8]):
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            B[i, j] = A[j, i]


def test_transpose_2d_lifts_to_transpose_node():
    sdfg = _transpose2d.to_sdfg(simplify=True)
    assert _apply(sdfg) == 1
    assert 'Transpose' in _libnodes(sdfg)
    assert _num_loops(sdfg) == 0
    sdfg.validate()

    n, m = 7, 5
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, m))
    b = np.zeros((m, n))
    ref = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ref[i, j] = a[j, i]
    sdfg(A=a.copy(), B=b, N=n, M=m)
    assert np.allclose(b, ref)


def test_transpose_3d_lifts_to_tensortranspose_node():
    sdfg = _transpose3d.to_sdfg(simplify=True)
    assert _apply(sdfg) == 1
    assert 'TensorTranspose' in _libnodes(sdfg)
    assert _num_loops(sdfg) == 0
    sdfg.validate()

    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 5, 6))
    b = np.zeros((6, 4, 5))
    ref = np.zeros((6, 4, 5))
    for i in range(6):
        for j in range(4):
            for k in range(5):
                ref[i, j, k] = a[j, k, i]
    sdfg(A=a.copy(), B=b)
    assert np.allclose(b, ref)


def test_sub_region_transpose_lifts_via_view():
    """Offset sub-block (``i in 1:5``, ``j in 2:6``): the operands are routed
    through strided Views; only the accessed window is transposed."""
    sdfg = _sub_region_transpose.to_sdfg(simplify=True)
    assert _apply(sdfg) == 1
    assert 'Transpose' in _libnodes(sdfg)
    sdfg.validate()

    rng = np.random.default_rng(2)
    a = rng.standard_normal((8, 8))
    b = np.full((8, 8), -1.0)
    ref = b.copy()
    for i in range(1, 5):
        for j in range(2, 6):
            ref[i, j] = a[j, i]
    sdfg(A=a.copy(), B=b)
    assert np.allclose(b, ref)


def test_strided_transpose_lifts_via_view():
    """Strided loops (``range(0, 8, 2)``): the View strides encode the step."""
    sdfg = _strided_transpose.to_sdfg(simplify=True)
    assert _apply(sdfg) == 1
    sdfg.validate()

    rng = np.random.default_rng(3)
    a = rng.standard_normal((8, 8))
    b = np.full((8, 8), -1.0)
    ref = b.copy()
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            ref[i, j] = a[j, i]
    sdfg(A=a.copy(), B=b)
    assert np.allclose(b, ref)


# --------------------------------------------------------------------------- #
# Negative: every non-permutation-copy shape must be refused.
# --------------------------------------------------------------------------- #


@dace.program
def _identity_copy(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            B[i, j] = A[i, j]


@dace.program
def _inplace_transpose(A: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            A[i, j] = A[j, i]


@dace.program
def _arith_body(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[j, i] + 1.0


@dace.program
def _mixed_subscript(A: dace.float64[2 * N, N], B: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[i + j, i]


@dace.program
def _single_loop_diag(A: dace.float64[N, N], B: dace.float64[N]):
    for i in range(N):
        B[i] = A[i, i]


def test_refuses_identity_copy():
    """``B[i,j] = A[i,j]`` is a plain copy, not a transpose."""
    sdfg = _identity_copy.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert not _libnodes(sdfg)


def test_refuses_inplace():
    """``A[i,j] = A[j,i]`` (same array) is LoopToSymmetrize's domain, not ours."""
    sdfg = _inplace_transpose.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert not _libnodes(sdfg)


def test_refuses_arithmetic_body():
    """``B[i,j] = A[j,i] + 1`` is not a pure copy."""
    sdfg = _arith_body.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert not _libnodes(sdfg)


def test_refuses_mixed_subscript():
    """``A[i+j, i]`` is affine in two loop variables on one axis -- not a
    clean bijection."""
    sdfg = _mixed_subscript.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert not _libnodes(sdfg)


def test_refuses_single_loop():
    """A single loop (``B[i] = A[i,i]``) is a gather, not a >=2-D permutation."""
    sdfg = _single_loop_diag.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert not _libnodes(sdfg)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
