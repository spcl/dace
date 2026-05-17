# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that ``LoopToMap`` parallelizes a loop only when its writes are
    provably non-overlapping across iterations.

    Each iteration of a loop must write disjoint locations for the loop to be a
    valid map. Two affine write subscripts ``a1*i + b1`` and ``a2*i + b2`` into
    the same container collide on some pair of iterations if and only if
    ``gcd(a1, a2)`` divides ``b2 - b1``; otherwise they are provably disjoint
    for any iteration range.
"""
import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')


@dace.program
def overlapping_writes(A: dace.int64[5 * N]):
    for i in range(N):
        A[5 * i] = 1
        A[3 * i] = 2


@dace.program
def injective_write(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = i


@dace.program
def disjoint_stride_writes(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = 1
        A[2 * i + 1] = 2


@dace.program
def shifted_writes(A: dace.int64[N + 1]):
    for i in range(N):
        A[i] = 1
        A[i + 1] = 2


@dace.program
def disjoint_outer_dim(B: dace.int64[2 * N, 4]):
    for i in range(N):
        B[2 * i, :] = 1
        B[2 * i + 1, :] = 2


def _applies(program) -> int:
    sdfg = program.to_sdfg(simplify=False)
    return sdfg.apply_transformations_repeated(LoopToMap)


def test_rejects_overlapping_writes():
    """ ``A[5*i]`` and ``A[3*i]`` collide at ``A[15]`` (i=3 and i=5). """
    sdfg = overlapping_writes.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0

    n = 64
    a = np.full(5 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(5 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[5 * i] = 1
        ref[3 * i] = 2
    assert np.array_equal(a, ref)


def test_accepts_injective_write():
    """ A single ``a*i + b`` write is injective in ``i`` and parallelizable. """
    sdfg = injective_write.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = i
    assert np.array_equal(a, ref)


def test_accepts_disjoint_strides():
    """ ``A[2*i]`` (even) and ``A[2*i+1]`` (odd) never collide, for any range. """
    sdfg = disjoint_stride_writes.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = 1
        ref[2 * i + 1] = 2
    assert np.array_equal(a, ref)


def test_rejects_shifted_writes():
    """ ``A[i]`` and ``A[i+1]`` collide between consecutive iterations. """
    assert _applies(shifted_writes) == 0


def test_accepts_disjoint_outer_dimension():
    """ A provably disjoint leading dimension makes the whole access disjoint. """
    sdfg = disjoint_outer_dim.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 32
    b = np.full((2 * n, 4), -1, dtype=np.int64)
    sdfg(B=b, N=n)
    ref = np.full((2 * n, 4), -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i, :] = 1
        ref[2 * i + 1, :] = 2
    assert np.array_equal(b, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
