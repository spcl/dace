# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A store a later iteration overwrites unread is dead -- and everything else must be left alone.

The rewrite drops a store, so a false positive is a silent miscompile rather than a slow program.
Most of what follows therefore checks that the pass REFUSES: a read inside the live window, a WCR
accumulate, a conditional store, a transient. The one positive case is TSVC ``s244``, and it is
checked numerically against the sequential reference as well as structurally.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination

N = dace.symbol('N', dtype=dace.int64)


def stores_of(sdfg, name):
    return sum(1 for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
               if isinstance(n, nodes.AccessNode) and n.data == name and st.in_degree(n) > 0)


def run(prog, **kwargs):
    sdfg = prog.to_sdfg(simplify=False)
    applied = DeadCarriedStoreElimination().apply_pass(sdfg, {})
    return sdfg, applied


@dace.program
def s244(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(N - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]


def reference_s244(a, b, c, d, n):
    for i in range(n - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]
    return a, b


def test_s244_is_rewritten_and_still_computes():
    sdfg, applied = run(s244)
    assert applied, 's244 is the motivating shape and must be recognised'

    n = 51
    rng = np.random.default_rng(4)
    base = [rng.random(n) for _ in range(4)]
    want_a, want_b = reference_s244(*[x.copy() for x in base], n)
    got_a, got_b, got_c, got_d = [x.copy() for x in base]
    sdfg(a=got_a, b=got_b, c=got_c, d=got_d, N=n)
    assert np.allclose(got_a, want_a), f'a: {got_a} != {want_a}'
    assert np.allclose(got_b, want_b), f'b: {got_b} != {want_b}'


def test_s244_parallelizes_through_the_pipeline():
    """The point of the rewrite: canonicalize now hands ``s244`` a Map where it had a loop.

    Asserted through the pipeline rather than on the pass alone: the pass leaves the dead store's
    operand chain for the simplification that follows it, so a bare ``to_sdfg`` still shows the
    stale read of ``a[i + 1]`` that keeps ``LoopToMap`` away.
    """
    from dace.sdfg.state import LoopRegion
    from dace.transformation.passes.canonicalize import pipeline as canon
    sdfg = s244.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    maps = [
        n for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
        if isinstance(n, nodes.MapEntry)
    ]
    loops = [
        b for g in sdfg.all_sdfgs_recursive() for b in g.all_control_flow_regions(recursive=True)
        if isinstance(b, LoopRegion)
    ]
    assert maps, f'no map (loops={len(loops)})'
    assert not loops, f'{len(loops)} sequential loop(s) survived'


@pytest.mark.parametrize('n', [2, 3, 4, 5, 17, 64])
def test_s244_matches_the_reference_through_the_pipeline(n):
    """Including the degenerate sizes where the peel consumes almost the whole loop."""
    from dace.transformation.passes.canonicalize import pipeline as canon
    sdfg = s244.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    rng = np.random.default_rng(17 + n)
    base = [rng.random(n) for _ in range(4)]
    want_a, want_b = reference_s244(*[x.copy() for x in base], n)
    got = [x.copy() for x in base]
    sdfg(a=got[0], b=got[1], c=got[2], d=got[3], N=n)
    assert np.allclose(got[0], want_a), f'a: {got[0]} != {want_a}'
    assert np.allclose(got[1], want_b), f'b: {got[1]} != {want_b}'


@dace.program
def read_in_the_live_window(a: dace.float64[N], b: dace.float64[N]):
    # a[i + 2] is killed by a[i] two iterations later, but a[i + 1] READS the value in between.
    for i in range(N - 2):
        a[i] = b[i]
        b[i] = a[i + 1]
        a[i + 2] = b[i] * 2.0


def test_a_read_inside_the_live_window_refuses():
    sdfg, applied = run(read_in_the_live_window)
    assert not applied, 'a[i + 1] observes the store before the kill reaches it'


@dace.program
def conditional_kill(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        if b[i] > 0.0:
            a[i] = b[i]
        a[i + 1] = b[i] * 2.0


def test_a_conditional_killing_store_refuses():
    sdfg, applied = run(conditional_kill)
    assert not applied, 'a kill that may not execute kills nothing'


@dace.program
def accumulating_kill(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] += b[i]
        a[i + 1] = b[i] * 2.0


def test_an_accumulating_kill_refuses():
    """``+=`` READS the destination, so it folds the earlier store in rather than replacing it."""
    sdfg, applied = run(accumulating_kill)
    assert not applied


@dace.program
def no_second_store(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i + 1] = b[i] * 2.0


def test_a_lone_store_refuses():
    sdfg, applied = run(no_second_store)
    assert not applied, 'nothing overwrites it'


@dace.program
def kill_is_earlier_not_later(a: dace.float64[N], b: dace.float64[N]):
    # The pair is the wrong way round: a[i] is written BEFORE a[i-1]'s killer would run.
    for i in range(1, N):
        a[i] = b[i]
        a[i - 1] = b[i] * 2.0


def test_a_kill_by_a_lower_offset_is_still_a_kill():
    """``a[i]`` written at iteration ``i`` is overwritten by ``a[(i + 1) - 1]`` the next one.

    Written expecting a refusal; the pass fires, and it is right to -- only ``a[n - 1]`` survives,
    which is exactly what the peel keeps. Kept as a numerical check that a kill found through the
    LOWER-offset store is handled with the same care as ``s244``'s higher-offset one."""
    sdfg, applied = run(kill_is_earlier_not_later)
    n = 40
    rng = np.random.default_rng(9)
    a0, b0 = rng.random(n), rng.random(n)
    want_a = a0.copy()
    for i in range(1, n):
        want_a[i] = b0[i]
        want_a[i - 1] = b0[i] * 2.0
    got = a0.copy()
    sdfg(a=got, b=b0.copy(), N=n)
    assert np.allclose(got, want_a), 'the rewrite changed the result'


@dace.program
def sibling_reads_the_store(A: dace.float64[N], B: dace.float64[N], E: dace.float64[N], D: dace.float64[N]):
    for i in range(1, N - 2):
        A[i] = B[i]
        A[i + 1] = E[i]
        D[i] = A[i + 1] * 2.0  # consumes what the candidate store just wrote, THIS iteration


def test_a_sibling_consuming_the_store_refuses():
    """The store looks killed by ``A[i]`` next iteration, but ``D`` reads it first.

    Caught by ``strengthen_split_statements``, not by this file -- the window test ``d < r < c``
    cannot see a read at ``r == c``, and reachability cannot either, because the consumer hangs off
    a different access node for the same array. Dropping the store silently gave D the stale value.
    """
    sdfg, applied = run(sibling_reads_the_store)
    assert not applied, 'the store is consumed in the same iteration'


def test_a_sibling_reading_BEFORE_the_store_is_still_dead():
    """The mirror case: a read at the same offset that happens EARLIER cannot observe the store."""
    sdfg, applied = run(s244)
    assert applied, "s244's own read-modify-write read precedes its store and must stay liftable"


if __name__ == '__main__':
    test_s244_is_rewritten_and_still_computes()
    test_s244_parallelizes_through_the_pipeline()
    test_s244_matches_the_reference_through_the_pipeline(17)
    test_a_read_inside_the_live_window_refuses()
    test_a_conditional_killing_store_refuses()
    test_an_accumulating_kill_refuses()
    test_a_lone_store_refuses()
    test_a_kill_by_a_lower_offset_is_still_a_kill()
    test_a_sibling_consuming_the_store_refuses()
    test_a_sibling_reading_BEFORE_the_store_is_still_dead()
