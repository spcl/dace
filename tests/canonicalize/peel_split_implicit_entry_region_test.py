# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Index-set splitting a loop that sits in a region with an IMPLICIT entry block (CloudSC ``peel``).

Most regions never pin an entry: a control-flow region with a single source block simply *is* entered
there, and ``ControlFlowRegion.start_block`` derives it. :meth:`BestEffortLoopPeeling._split_loop_at`
replaces a loop by a chain of range segments, and each segment joins the parent graph disconnected --
so from the first clone on, the parent has SEVERAL source blocks and the derivation no longer works.
Asking which block is the entry at that point raised ``ValueError: Ambiguous or undefined starting
block``, which is how the CloudSC pipeline's ``peel`` phase died once the symbol-identity fix let the
pass act on CloudSC at all.

The loop here is the SECOND block of its enclosing loop body (a ``c[j] = 0.0`` state runs first), so
the body region's entry is that state, implicit and unpinned -- the shape a Fortran-frontend nest hits
constantly. Splitting must neither ask an unanswerable question nor disturb the entry.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling

M = dace.symbol('M', nonnegative=True)
N = dace.symbol('N', nonnegative=True)


@dace.program
def guarded_inner(a: dace.float64[M, N], b: dace.float64[M, N], c: dace.float64[M]):
    for j in range(M):
        c[j] = 0.0
        for i in range(N):
            a[j, i] = b[j, i] * 2.0
            if i == N - 1:
                a[j, N - 2] = a[j, N - 2] + 1.0


def _reference(a, b, c, m, n):
    for j in range(m):
        c[j] = 0.0
        for i in range(n):
            a[j, i] = b[j, i] * 2.0
            if i == n - 1:
                a[j, n - 2] = a[j, n - 2] + 1.0
    return a, c


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _inner_loop(sdfg) -> LoopRegion:
    inner = [loop for loop in _loops(sdfg) if loop.loop_variable == 'i']
    assert len(inner) == 1
    return inner[0]


def test_the_split_target_sits_in_a_region_whose_entry_is_implicit():
    """Precondition of the regression: the enclosing region never pinned an entry block, so the
    entry is only derivable while the region has exactly one source block."""
    sdfg = guarded_inner.to_sdfg(simplify=True)
    parent = _inner_loop(sdfg).parent_graph
    # Read directly: ``start_block`` answers from the single source block either way, so the
    # unpinned state -- the thing that makes the entry underivable mid-surgery -- is only visible here.
    assert parent._start_block is None, 'expected an unpinned (implicit) region entry'
    assert parent.start_block is not _inner_loop(sdfg), 'the split target must not be the region entry'


def test_split_at_keeps_the_region_entry_and_does_not_ask_an_ambiguous_question():
    """The split itself: it must not raise, and the entry block must come out unchanged."""
    sdfg = guarded_inner.to_sdfg(simplify=True)
    loop = _inner_loop(sdfg)
    parent = loop.parent_graph
    entry_before = parent.start_block
    peel = BestEffortLoopPeeling(peel_limit=4)
    found = peel._best_split_for(loop, sdfg)
    assert found is not None, 'expected an index-set split point for the guarded inner loop'
    x, middle_singleton, _guarded = found
    assert peel._split_loop_at(sdfg, loop, x, middle_singleton=middle_singleton)
    assert parent.start_block is entry_before, 'the split must not move the region entry'
    sdfg.validate()


def test_apply_pass_splits_the_nested_loop():
    """End to end: the pass runs, the SDFG stays valid, and the inner loop became range segments."""
    sdfg = guarded_inner.to_sdfg(simplify=True)
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()
    inner = [loop for loop in _loops(sdfg) if loop.loop_variable == 'i']
    assert len(inner) > 1, f'the inner loop must be split into segments, got {[l.label for l in inner]}'


def test_split_form_is_bit_exact_against_the_sequential_reference():
    sdfg = guarded_inner.to_sdfg(simplify=True)
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    m, n = 5, 16
    rng = np.random.default_rng(1234)
    b = rng.random((m, n))
    a0 = rng.random((m, n))
    ref_a, ref_c = _reference(a0.copy(), b, np.zeros(m), m, n)

    got_a, got_c = a0.copy(), np.zeros(m)
    sdfg.compile()(a=got_a, b=b.copy(), c=got_c, M=m, N=n)
    assert np.array_equal(got_a, ref_a), 'split form must match the sequential meaning'
    assert np.array_equal(got_c, ref_c), 'split form must match the sequential meaning'


if __name__ == '__main__':
    test_the_split_target_sits_in_a_region_whose_entry_is_implicit()
    test_split_at_keeps_the_region_entry_and_does_not_ask_an_ambiguous_question()
    test_apply_pass_splits_the_nested_loop()
    test_split_form_is_bit_exact_against_the_sequential_reference()
    print('OK')
