# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A write that starts at ``a*i+b`` is unique per iteration only if its slab fits in the stride.

householder_qr applies its reflectors with ``Q[k:M, :] = ...`` under the ``k`` loop: every iteration
rewrites the rows of every later one, so the loop is carried although each lower bound moves with ``k``.
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")
S = dace.symbol("S")


@dace.program
def overlapping_slabs(Q: dace.int64[M, N]):
    for k in range(M):
        Q[k:M, :] = k


@dace.program
def strided_point_writes(A: dace.int64[N]):
    for i in range(0, N, S):
        A[i] = i


@dace.program
def clamped_tiles(A: dace.int64[N, K]):
    for t in range(0, N, S):
        A[t : min(N, t + S), :] = t


@dace.program
def interleaved_slabs(A: dace.int64[N]):
    for i in range(S):
        A[i:N:S] = i


@dace.program
def stride_wide_slabs(A: dace.int64[N * K]):
    for i in range(N):
        A[i * K : i * K + K] = i


def loops_left(sdfg: dace.SDFG) -> int:
    sdfg.apply_transformations_repeated(LoopToMap)
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def test_a_slab_wider_than_the_stride_stays_a_loop():
    sdfg = overlapping_slabs.to_sdfg(simplify=False)
    assert loops_left(sdfg) == 1
    q = np.full((6, 3), -1, dtype=np.int64)
    sdfg(Q=q, M=6, N=3)
    assert np.array_equal(q, np.repeat(np.arange(6), 3).reshape(6, 3))


def test_a_point_write_under_a_symbolic_stride_becomes_a_map():
    sdfg = strided_point_writes.to_sdfg(simplify=False)
    assert loops_left(sdfg) == 0
    a = np.full(20, -1, dtype=np.int64)
    sdfg(A=a, N=20, S=3)
    ref = np.full(20, -1, dtype=np.int64)
    ref[::3] = np.arange(0, 20, 3)
    assert np.array_equal(a, ref)


def test_a_slab_as_wide_as_its_symbolic_stride_becomes_a_map():
    sdfg = stride_wide_slabs.to_sdfg(simplify=False)
    assert loops_left(sdfg) == 0
    a = np.full(12, -1, dtype=np.int64)
    sdfg(A=a, N=4, K=3)
    assert np.array_equal(a, np.repeat(np.arange(4), 3))


def test_a_clamped_tile_as_wide_as_its_symbolic_stride_becomes_a_map():
    # Unsimplified, the clamp is a per-iteration symbol whose value no analysis bounds.
    sdfg = clamped_tiles.to_sdfg(simplify=True)
    assert loops_left(sdfg) == 0
    a = np.full((10, 2), -1, dtype=np.int64)
    sdfg(A=a, N=10, K=2, S=4)
    assert np.array_equal(a, np.repeat([0, 0, 0, 0, 4, 4, 4, 4, 8, 8], 2).reshape(10, 2))


def test_interleaved_slabs_within_one_range_step_become_a_map():
    sdfg = interleaved_slabs.to_sdfg(simplify=False)
    assert loops_left(sdfg) == 0
    a = np.full(10, -1, dtype=np.int64)
    sdfg(A=a, N=10, S=3)
    assert np.array_equal(a, np.arange(10) % 3)


if __name__ == "__main__":
    test_a_slab_wider_than_the_stride_stays_a_loop()
    test_a_point_write_under_a_symbolic_stride_becomes_a_map()
    test_a_slab_as_wide_as_its_symbolic_stride_becomes_a_map()
    test_a_clamped_tile_as_wide_as_its_symbolic_stride_becomes_a_map()
    test_interleaved_slabs_within_one_range_step_become_a_map()
