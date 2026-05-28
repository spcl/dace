# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the single-shot ``parallelize`` pipeline (reduction-aware
loop-to-map). SDFGs are produced through the DaCe Python frontend."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes import (ParallelizePipeline, parallelize, BestEffortLoopPeeling, ShortLoopUnroll)

M, N = (dace.symbol(s) for s in ('M', 'N'))


def _num_maps(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _num_loops(sdfg: dace.SDFG) -> int:
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def test_parallelize_independent_loop_maps_value_preserving():
    """An embarrassingly parallel loop is mapped and stays value-preserving."""

    @dace.program
    def scale(A: dace.float64[M, N], B: dace.float64[M, N]):
        for i in range(M):
            for j in range(N):
                B[i, j] = A[i, j] * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    parallelize(sdfg, validate=True)
    assert _num_maps(sdfg) > 0
    assert _num_loops(sdfg) < loops_before

    A = np.random.default_rng(0).random((6, 5))
    B = np.zeros((6, 5))
    sdfg(A=A, B=B, M=6, N=5)
    assert np.allclose(B, A * 2.0)


def test_parallelize_rowsum_reduction_value_preserving():
    """A per-row accumulator reduction parallelizes (the outer row loop becomes a
    map; the accumulator is privatized/WCR'd) and matches the numpy oracle."""

    @dace.program
    def rowsum(A: dace.float64[M, N], out: dace.float64[M]):
        for i in range(M):
            acc = 0.0
            for j in range(N):
                acc += A[i, j]
            out[i] = acc

    sdfg = rowsum.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _num_maps(sdfg) > 0

    A = np.random.default_rng(1).random((7, 4))
    out = np.zeros(7)
    sdfg(A=A, out=out, M=7, N=4)
    assert np.allclose(out, A.sum(axis=1))


def test_parallelize_runs_once_idempotent():
    """The pipeline is single-shot: a second run does not change a fixed SDFG's
    results, and reports the same number of stages."""

    @dace.program
    def add(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        for i in range(M):
            C[i] = A[i] + B[i]

    sdfg = add.to_sdfg(simplify=True)
    assert ParallelizePipeline().apply_pass(sdfg, {}) == 12  # composed stages
    # Re-running is harmless (nothing left to parallelize).
    assert ParallelizePipeline().apply_pass(sdfg, {}) == 12
    sdfg.validate()

    A = np.random.default_rng(2).random(8)
    B = np.random.default_rng(3).random(8)
    C = np.zeros(8)
    sdfg(A=A, B=B, C=C, M=8)
    assert np.allclose(C, A + B)


def test_parallelize_unrolls_short_constant_loop():
    """A short constant-trip loop is fully unrolled (no loop, no atomics) below the
    limit, and left alone above it; the result is value-preserving either way."""

    @dace.program
    def short_reduce(A: dace.float64[5], B: dace.float64[1]):
        acc = 0.0
        for j in range(5):
            acc += A[j]
        B[0] = acc

    # The unroll gate fires only at/below the limit (probe the stage directly,
    # since downstream LoopToReduce would also eliminate the loop).
    below = short_reduce.to_sdfg(simplify=True)
    assert _num_loops(below) == 1
    ShortLoopUnroll(unroll_limit=8).apply_pass(below, {})
    assert _num_loops(below) == 0  # trip 5 <= 8 -> unrolled

    above = short_reduce.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=4).apply_pass(above, {})
    assert _num_loops(above) == 1  # trip 5 > 4 -> left alone

    off = short_reduce.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=0).apply_pass(off, {})
    assert _num_loops(off) == 1  # disabled

    # Full pipeline (default limit 8) is value-preserving end to end.
    sdfg = short_reduce.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _num_loops(sdfg) == 0
    A = np.random.default_rng(4).random(5)
    B = np.zeros(1)
    sdfg(A=A, B=B)
    assert np.allclose(B[0], A.sum())


def test_parallelize_peel_mechanism_value_preserving():
    """``BestEffortLoopPeeling._peel_loops`` peels boundary iterations off a loop
    (front / back / both) and stays value-preserving, even for symbolic bounds."""

    @dace.program
    def scale(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            B[i] = A[i] * 3.0

    for direction in ('front', 'back', 'both'):
        sdfg = scale.to_sdfg(simplify=True)
        peeled = BestEffortLoopPeeling()._peel_loops(sdfg, count=2, direction=direction)
        assert peeled == 1, direction
        sdfg.validate()
        A = np.arange(10, dtype=np.float64)
        B = np.zeros(10)
        sdfg(A=A, B=B, N=10)
        assert np.allclose(B, A * 3.0), direction


def test_parallelize_peeling_reverts_on_recurrence():
    """A genuine loop-carried recurrence cannot be peeled into a map, so the
    best-effort search reverts (no spurious map) and the result is correct."""

    @dace.program
    def prefix_sum(A: dace.float64[N], B: dace.float64[N]):
        B[0] = A[0]
        for i in range(1, N):
            B[i] = B[i - 1] + A[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)  # peeling cannot help -> reverts
    assert _num_loops(sdfg) >= 1  # recurrence stays a sequential loop

    A = np.arange(1, 9, dtype=np.float64)
    B = np.zeros(8)
    sdfg(A=A, B=B, N=8)
    assert np.allclose(B, np.cumsum(A))


def test_parallelize_peel_limit_zero_disables():
    """``peel_limit=0`` skips the peeling search entirely."""

    @dace.program
    def prefix_sum(A: dace.float64[N], B: dace.float64[N]):
        B[0] = A[0]
        for i in range(1, N):
            B[i] = B[i - 1] + A[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    before = _num_loops(sdfg)
    BestEffortLoopPeeling(peel_limit=0).apply_pass(sdfg, {})
    assert _num_loops(sdfg) == before  # untouched


if __name__ == '__main__':
    test_parallelize_independent_loop_maps_value_preserving()
    test_parallelize_rowsum_reduction_value_preserving()
    test_parallelize_runs_once_idempotent()
    test_parallelize_unrolls_short_constant_loop()
    test_parallelize_peel_mechanism_value_preserving()
    test_parallelize_peeling_reverts_on_recurrence()
    test_parallelize_peel_limit_zero_disables()
