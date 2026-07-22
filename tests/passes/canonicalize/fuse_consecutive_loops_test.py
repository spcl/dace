# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`FuseConsecutiveLoops`.

The pass re-joins two directly-consecutive, identical-bodied, unit-stride loops
over adjacent index ranges (``[A, B)`` then ``[B, C)``) into one loop over
``[A, C)`` -- undoing the main-body / remainder split of a hand-tiled loop so a
tiled reduction lifts to a single ``Reduce`` (exact) instead of two un-chained
``Reduce`` nodes writing the same accumulator (half-sum).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended

N = dace.symbol('N')


@dace.program
def _two_adjacent_sums(a: dace.float64[N], out: dace.float64[1]):
    s = 0.0
    for i in range(0, 10):
        s = s + a[i]
    for i in range(10, N):
        s = s + a[i]
    out[0] = s


@dace.program
def _two_diff_accum(a: dace.float64[N], out: dace.float64[2]):
    s = 0.0
    t = 0.0
    for i in range(0, 10):
        s = s + a[i]
    for i in range(10, N):
        t = t + a[i]
    out[0] = s
    out[1] = t


@dace.program
def _non_adjacent_sums(a: dace.float64[N], out: dace.float64[1]):
    s = 0.0
    for i in range(0, 5):
        s = s + a[i]
    for i in range(10, N):  # gap [5, 10) -- ranges are not adjacent
        s = s + a[i]
    out[0] = s


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def _nreduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, nodes.LibraryNode) and 'Reduce' in type(n).__name__)


def _prep(prog):
    """Lower a program to single-state-bodied LoopRegions (simplify + state
    fusion) so the pass -- which matches single-state bodies -- can be exercised
    in isolation on the frontend shape."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.simplify()
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    return sdfg


def test_fuses_two_adjacent_reduction_loops():
    """Two adjacent identical reduction loops fuse into one; value exact."""
    sdfg = _prep(_two_adjacent_sums)
    assert _nloops(sdfg) == 2
    fused = FuseConsecutiveLoops().apply_pass(sdfg, {})
    assert fused == 1, "the two adjacent identical loops must fuse into one"
    assert _nloops(sdfg) == 1
    sdfg.validate()

    n = 25
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=n)
    assert np.isclose(out[0], np.sum(a)), f'got {out[0]} expected {np.sum(a)}'


def test_end_to_end_single_reduce_and_exact():
    """Full canonicalize collapses the two adjacent reduction loops to one loop
    that lifts to a SINGLE Reduce -- not two un-chained reduces (the half-sum)."""
    sdfg = _two_adjacent_sums.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nloops(sdfg) == 0, "the fused reduction must fully lift"
    assert _nreduces(sdfg) == 1, "exactly one Reduce over the whole range"

    n = 25
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=n)
    assert np.isclose(out[0], np.sum(a)), f'got {out[0]} expected {np.sum(a)}'


def test_different_accumulators_not_fused():
    """Adjacent loops that reduce into DIFFERENT accumulators are not fused
    (their bodies differ) and stay value-correct."""
    sdfg = _prep(_two_diff_accum)
    assert _nloops(sdfg) == 2
    assert FuseConsecutiveLoops().apply_pass(sdfg, {}) is None, "different bodies must not fuse"
    assert _nloops(sdfg) == 2
    sdfg.validate()

    n = 25
    rng = np.random.default_rng(2)
    a = rng.standard_normal(n)
    out = np.zeros(2)
    sdfg(a=a.copy(), out=out, N=n)
    assert np.isclose(out[0], np.sum(a[:10]))
    assert np.isclose(out[1], np.sum(a[10:]))


def test_non_adjacent_ranges_not_fused():
    """Adjacent-in-CFG loops whose index ranges leave a gap are not fused."""
    sdfg = _prep(_non_adjacent_sums)
    assert _nloops(sdfg) == 2
    assert FuseConsecutiveLoops().apply_pass(sdfg, {}) is None, "non-adjacent ranges must not fuse"
    assert _nloops(sdfg) == 2
    sdfg.validate()

    n = 25
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=n)
    assert np.isclose(out[0], np.sum(a[:5]) + np.sum(a[10:]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
