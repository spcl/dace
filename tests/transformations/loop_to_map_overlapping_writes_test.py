# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap must not parallelize a loop with overlapping write subscripts.

Minimal reproducer of the soundness bug found via the Polly SCoP
canonicalization test: a loop whose body writes ``A[5*i]`` and ``A[3*i]``. Each
subscript is individually injective in ``i``, but their images overlap across
iterations (e.g. ``i=3`` writes ``A[15]`` via ``5*i`` and ``i=5`` writes
``A[15]`` via ``3*i``). Parallelizing reorders the colliding writes, so
``LoopToMap`` must refuse this loop while still accepting a genuinely
independent one.
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
def independent_writes(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = i


@dace.program
def disjoint_stride_writes(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = 1
        A[2 * i + 1] = 2


def test_loop_to_map_rejects_overlapping_writes():
    sdfg = overlapping_writes.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied == 0, "LoopToMap unsoundly parallelized overlapping writes"

    n = 64
    a = np.full(5 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)

    ref = np.full(5 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[5 * i] = 1
        ref[3 * i] = 2
    assert np.array_equal(a, ref)


def test_loop_to_map_accepts_independent_writes():
    sdfg = independent_writes.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied >= 1, "LoopToMap failed to parallelize an independent loop"

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)

    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = i
    assert np.array_equal(a, ref)


def test_loop_to_map_accepts_disjoint_strides():
    # ``A[2*i]`` and ``A[2*i+1]`` are provably disjoint (even vs. odd) for any
    # range, so the loop is parallelizable and must be accepted.
    sdfg = disjoint_stride_writes.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied >= 1, "LoopToMap rejected a provably disjoint-stride loop"

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)

    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = 1
        ref[2 * i + 1] = 2
    assert np.array_equal(a, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
