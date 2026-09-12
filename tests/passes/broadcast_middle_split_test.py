# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the broadcast-read index-set split in ``BestEffortLoopPeeling``.

A loop that reads a loop-invariant broadcast element ``A[c]`` while writing
``A[f(i)]`` has a single conflicting iteration ``x`` (where ``f(x) == c``): it is
the sole producer of the element every other iteration reads. Splitting the loop
at ``x`` into ``[start, x-1] + {x} + [x+1, end]`` -- which preserves the exact
sequential order (the range before ``x`` reads the original ``A[c]``, ``{x}``
overwrites it, the range after reads the new value) -- turns the two range
segments into conflict-free parallel maps. This is the TSVC ``s1113`` shape.

Each scenario is a dace-Python (numpy-style) challenge program run to a numpy
reference before and after canonicalization to confirm value-preservation.
"""

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('N')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _split_evaluates_to(points, n_val, expected):
    """A split point equals ``expected`` at ``N == n_val`` -- robust to
    floor-representation differences (``N//2`` parses to ``floor(N/2)`` while the
    subset carries dace's ``int_floor(N, 2)``; both evaluate identically)."""
    for p in points:
        try:
            if int(symbolic.evaluate(p, {N: n_val})) == expected:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
#  TSVC s1113: a[i] = a[N//2] + b[i]. The broadcast read a[N//2] collides with
#  the write a[i] only at i == N//2.
# ---------------------------------------------------------------------------
@dace.program
def s1113(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        a[i] = a[N // 2] + b[i]


def _ref_s1113(a, b):
    n = a.shape[0]
    for i in range(n):
        a[i] = a[n // 2] + b[i]
    return a


def test_detector_finds_broadcast_conflict_point():
    sdfg = s1113.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    points = BestEffortLoopPeeling(peel_limit=4)._broadcast_conflict_split_points(loop)
    assert _split_evaluates_to(points, 64, 32), f"expected split at N//2 (=32 for N=64), got {points}"


def test_s1113_value_preserving_and_parallelizes():
    sdfg = s1113.to_sdfg(simplify=True)
    rng = np.random.default_rng(0)
    a0 = rng.random(64)
    b = rng.random(64)
    ref = _ref_s1113(a0.copy(), b.copy())

    canonicalize(sdfg, validate=True, peel_limit=4)

    got = a0.copy()
    sdfg(a=got, b=b.copy(), N=64)
    assert np.allclose(got, ref), "broadcast middle-split must preserve the sequential result"
    assert _nmaps(sdfg) >= 1, "the range segments around the conflict must parallelize"


# ---------------------------------------------------------------------------
#  Boundary conflict: a[i] = a[0] + b[i]. The conflict is at i == 0 (a boundary),
#  so the split drops the empty [start, x-1] side -> {0} + [1, N-1].
# ---------------------------------------------------------------------------
@dace.program
def broadcast_at_zero(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        a[i] = a[0] + b[i]


def test_boundary_broadcast_value_preserving():
    sdfg = broadcast_at_zero.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    points = BestEffortLoopPeeling(peel_limit=4)._broadcast_conflict_split_points(loop)
    assert _split_evaluates_to(points, 48, 0), f"expected split at 0, got {points}"

    rng = np.random.default_rng(1)
    a0 = rng.random(48)
    b = rng.random(48)
    ref = a0.copy()
    for i in range(48):
        ref[i] = ref[0] + b[i]

    canonicalize(sdfg, validate=True, peel_limit=4)
    got = a0.copy()
    sdfg(a=got, b=b.copy(), N=48)
    assert np.allclose(got, ref)


# ---------------------------------------------------------------------------
#  No conflict: the broadcast read is on a DIFFERENT array (c[N//2]), so writing
#  a[i] never collides -> no split point, and the loop maps directly.
# ---------------------------------------------------------------------------
@dace.program
def broadcast_other_array(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        a[i] = c[N // 2] + b[i]


def test_no_conflict_when_broadcast_is_other_array():
    sdfg = broadcast_other_array.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    points = BestEffortLoopPeeling(peel_limit=4)._broadcast_conflict_split_points(loop)
    assert points == [], f"a read of a different array is not a self-conflict, got {points}"

    rng = np.random.default_rng(2)
    a0 = rng.random(32)
    b = rng.random(32)
    c = rng.random(32)
    ref = c[16] + b  # a[i] = c[N//2] + b[i], fully parallel
    canonicalize(sdfg, validate=True, peel_limit=4)
    got = a0.copy()
    sdfg(a=got, b=b.copy(), c=c.copy(), N=32)
    assert np.allclose(got, ref)
    assert _nmaps(sdfg) >= 1, "an elementwise loop with a loop-invariant read maps directly"


def test_broadcast_split_applied_directly_by_peeling():
    """Applying ``BestEffortLoopPeeling`` DIRECTLY (not via full canonicalize)
    carves the s1113 broadcast loop at the conflict iteration into range segments,
    value-preservingly."""
    sdfg = s1113.to_sdfg(simplify=True)
    before = len(_loops(sdfg))
    rng = np.random.default_rng(3)
    a0 = rng.random(48)
    b = rng.random(48)
    ref = _ref_s1113(a0.copy(), b.copy())

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()
    after = len(_loops(sdfg))
    assert after > before, f"the broadcast split should carve the loop into segments (was {before}, now {after})"

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=48)
    assert np.allclose(got, ref), "the direct peel-split must preserve the value"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
