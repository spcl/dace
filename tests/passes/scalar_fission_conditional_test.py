# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar fission of thread-local scalars written inside conditional branches.

A scalar reused as a per-iteration temporary across several loop bodies appears
outside any single loop, so ``LoopToMap`` treats it as a cross-iteration value
and refuses (its ``[0]`` write does not depend on the loop iterator). Privatizing
it -- giving each loop its own container (``PrivatizeScalars`` / ``ScalarFission``)
-- makes it loop-local so the loop maps. This works when a write dominates the
read, but a scalar written in *every branch* of a conditional and read *after the
merge* has no single dominating write (a ``None`` write-scope), and is currently
left un-privatized -- which is the cloudsc ``zcor``/``zfac``/``zqe`` pattern.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import PrivatizeScalars
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')


@dace.program
def _shared_in_if(a: dace.float64[N], c: dace.float64[N], o1: dace.float64[N], o2: dace.float64[N]):
    # z is written AND read inside the if (a write dominates the read), reused
    # across two loops.
    for i in range(N):
        o1[i] = a[i]
        if c[i] > 0.5:
            z = a[i] * 2.0
            o1[i] = z + 1.0
    for i in range(N):
        o2[i] = a[i]
        if c[i] > 0.3:
            z = a[i] * 3.0
            o2[i] = z + 2.0


@dace.program
def _shared_read_after_conditional(a: dace.float64[N], c: dace.float64[N], o1: dace.float64[N], o2: dace.float64[N]):
    # z is written in BOTH branches and read AFTER the merge (no single dominating
    # write -> None write-scope), reused across two loops. This is the cloudsc
    # zcor pattern.
    for i in range(N):
        if c[i] > 0.5:
            z = a[i] * 2.0
        else:
            z = a[i] + 1.0
        o1[i] = z * 3.0
    for i in range(N):
        if c[i] > 0.3:
            z = a[i] * 4.0
        else:
            z = a[i] + 5.0
        o2[i] = z * 6.0


@dace.program
def _shared_carried_no_else(a: dace.float64[N], c: dace.float64[N], o1: dace.float64[N], o2: dace.float64[N]):
    # NON-exhaustive: z written only in the if (no else), read after. If c is
    # false, the read sees the previous iteration's z -> genuinely loop-carried.
    # Privatizing this would be a miscompilation; the guard must NOT privatize it.
    z = 0.0
    for i in range(N):
        if c[i] > 0.5:
            z = a[i] * 2.0
        o1[i] = z * 3.0
    z = 0.0
    for i in range(N):
        if c[i] > 0.3:
            z = a[i] * 4.0
        o2[i] = z * 6.0


def _remaining_loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _privatize_then_map(prog):
    """Build, privatize thread-local scalars, then LoopToMap. Returns the number
    of LoopRegions left sequential (0 == every loop became a map)."""
    sdfg = prog.to_sdfg(simplify=True)
    PrivatizeScalars().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    return sdfg, len(_remaining_loops(sdfg))


def _run(sdfg, n=64, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    c = rng.random(n)
    o1 = np.zeros(n)
    o2 = np.zeros(n)
    sdfg(a=a, c=c, o1=o1, o2=o2, N=n)
    return a, c, o1, o2


def test_shared_scalar_in_if_privatized_enables_loop_to_map():
    """A thread-local scalar written+read inside an if, reused across loops, is
    privatized so both loops map, and the result is unchanged."""
    sdfg, left = _privatize_then_map(_shared_in_if)
    assert left == 0, f"expected both loops mapped after privatization, {left} left sequential"
    a, c, o1, o2 = _run(sdfg)
    assert np.allclose(o1, np.where(c > 0.5, a * 2.0 + 1.0, a))
    assert np.allclose(o2, np.where(c > 0.3, a * 3.0 + 2.0, a))


def test_shared_scalar_read_after_conditional_privatized_enables_loop_to_map():
    """A thread-local scalar written in both branches and read after the merge,
    reused across loops, is privatized (no upward-exposed use) so both loops map.
    This is the cloudsc ``zcor``/``zfac``/``zqe`` pattern."""
    sdfg, left = _privatize_then_map(_shared_read_after_conditional)
    assert left == 0, f"expected both loops mapped after privatization, {left} left sequential"
    a, c, o1, o2 = _run(sdfg)
    assert np.allclose(o1, np.where(c > 0.5, a * 2.0, a + 1.0) * 3.0)
    assert np.allclose(o2, np.where(c > 0.3, a * 4.0, a + 5.0) * 6.0)


def test_carried_scalar_no_else_not_miscompiled():
    """A scalar written only in the if (no else) and read after is upward-exposed
    (loop-carried). Privatizing it would be unsound, so the result after
    PrivatizeScalars must equal the un-transformed reference."""
    ref = _shared_carried_no_else.to_sdfg(simplify=True)
    cand = _shared_carried_no_else.to_sdfg(simplify=True)
    PrivatizeScalars().apply_pass(cand, {})
    cand.apply_transformations_repeated(LoopToMap)
    cand.validate()

    n, seed = 48, 1
    ra = _run(ref, n, seed)
    ca = _run(cand, n, seed)
    # outputs (o1, o2) must match the un-transformed reference exactly
    assert np.array_equal(ra[2], ca[2]) and np.array_equal(ra[3], ca[3])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
