# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize's optional parallelization knobs, exercised end-to-end through
the pipeline: ``break_anti_dependence`` (snapshot-rename a read-ahead WAR) and
``peel_limit`` (best-effort loop peeling). Both are off by default and target
loops that ``LoopToMap`` would otherwise refuse; each test checks that the knob
flips the loop from sequential to a parallel Map AND stays value-preserving."""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


@dace.program
def _s121(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = a[i + 1] + b[i]


def test_break_anti_dependence_knob_parallelizes():
    """``a[i] = a[i+1] + b[i]`` is a pure read-ahead WAR (TSVC s121): off by
    default it stays a sequential loop (LoopToMap refuses the read-write
    conflict); with ``break_anti_dependence=True`` the array is snapshot-renamed
    so the loop becomes a Map, value-preserving."""
    off = _s121.to_sdfg(simplify=True)
    canonicalize(off, validate=True)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'WAR loop must stay sequential without the knob'

    on = _s121.to_sdfg(simplify=True)
    canonicalize(on, validate=True, break_anti_dependence=True)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'break_anti_dependence must parallelize the WAR loop'

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(7):
        ref[i] = a[i + 1] + b[i]  # reads the ORIGINAL a (read-ahead)
    got = a.copy()
    on(a=got, b=b.copy(), N=8)
    assert np.allclose(got, ref)


@dace.program
def _front_conflict(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == 0:
            A[N - 1] = A[N - 1] + 1.0


def test_loop_peeling_front_conflict_knob_parallelizes():
    """A first-iteration guard writes a conflicting extra location
    (``if i==0: A[N-1]+=1``): off by default the write-write conflict keeps the
    loop sequential; with ``peel_limit>0`` the front iteration is peeled off and
    the now-dead guard pruned, leaving a disjoint-write remainder that maps,
    value-preserving."""
    off = _front_conflict.to_sdfg(simplify=True)
    canonicalize(off, validate=True)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'boundary-conflict loop must stay sequential without the knob'

    on = _front_conflict.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling must unblock the boundary-conflict loop'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _front_conflict.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _fixed_read(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        a[i] = a[0] + b[i]


def test_loop_peeling_fixed_read_first_iter_knob_parallelizes():
    """``a[i] = a[1] + b[i]`` (textbook): every iteration reads the fixed ``a[1]``,
    which iteration 1 itself writes -- a loop-carried flow dependence (0-indexed:
    iteration 0 writes ``a[0]``, the rest read it). Off by default it stays a
    sequential loop; with ``peel_limit>0`` iteration 0 is peeled off and the
    remainder reads a now-fixed ``a[0]`` (disjoint from the ``a[1:N]`` writes), so
    it maps and runs, value-preserving. Exercises the LoopToMap conflict-analysis
    fix (a loop-invariant read disjoint from the ranged write is not a conflict)
    plus the LICM map-scope fix (the read of a map-written array is not hoisted)."""
    off = _fixed_read.to_sdfg(simplify=True)
    canonicalize(off, validate=True)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'the carried fixed-read loop must stay sequential without the knob'

    on = _fixed_read.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling iteration 0 must unblock the fixed-read remainder'

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) + 0.5
    ref_a = a.copy()
    _fixed_read.to_sdfg(simplify=True)(a=ref_a, b=b.copy(), N=8)
    got = a.copy()
    on(a=got, b=b.copy(), N=8)
    assert np.allclose(got, ref_a)


if __name__ == '__main__':
    test_break_anti_dependence_knob_parallelizes()
    test_loop_peeling_front_conflict_knob_parallelizes()
    test_loop_peeling_fixed_read_first_iter_knob_parallelizes()
