# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``ParallelizeLoops`` pass.

What needs guarding is not "does it lift a loop" -- ``LoopToMap`` owns that -- but the assumptions
that let the pass reuse analysis across probes:

* a lift never changes a surviving block's ``free_symbols``, which is what lets the memo live for
  the whole pass instead of being rebuilt per lift. A stale entry makes the "symbol used after the
  loop" check wrongly ACCEPT, which is a miscompile, so it is asserted rather than argued.
* reusing analysis never costs a map against the plain matcher. An earlier version cached refusals
  too and lost one here, because a lift can make a loop liftable that is nowhere near it.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import (LiftContext, LoopToMap, block_free_symbols, build_lift_context,
                                                        build_lift_invariants)
from dace.transformation.passes.parallelize_loops import ParallelizeLoops, candidate_loops, loop_order_key
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = 24


@dace.program
def three_independent_sweeps(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N]):
    for i in range(N):
        for j in range(N):
            a[i, j] = b[i, j] * 2.0 + 1.0
    for i in range(N):
        c[i] = a[i, i]
    for i in range(N):
        for j in range(N):
            b[i, j] = a[i, j] - c[i]


def map_count(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def loop_count(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def all_blocks(sdfg: dace.SDFG):
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for cfg in sd.all_control_flow_regions(recursive=True):
            out.append(cfg)
            out.extend(cfg.nodes())
    return out


def test_lift_never_changes_a_surviving_blocks_free_symbols():
    """The invariant the pass-lifetime ``free_symbols`` memo rests on.

    A lift swaps a LoopRegion for a state holding a Map plus a NestedSDFG. The map range reuses the
    loop's own bound expressions, and the iterate is defined by the loop before and by the map scope
    after -- so no block that survives the lift gains or loses a free symbol.
    """
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    real_apply = LoopToMap.apply
    mismatches = []
    comparisons = 0

    def checking_apply(self, graph, inner_sdfg):
        root = inner_sdfg
        while root.parent_sdfg is not None:
            root = root.parent_sdfg
        before = {b: set(b.free_symbols) for b in all_blocks(root)}
        out = real_apply(self, graph, inner_sdfg)
        nonlocal comparisons
        for block, was in before.items():
            if block.parent_graph is None and not isinstance(block, dace.SDFG):
                continue  # detached by the lift; the memo can never be asked about it again
            comparisons += 1
            now = set(block.free_symbols)
            if now != was:
                mismatches.append((block, sorted(was - now), sorted(now - was)))
        return out

    LoopToMap.apply = checking_apply
    try:
        ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    finally:
        LoopToMap.apply = real_apply

    assert comparisons > 0, 'no lift happened, so the invariant was never exercised'
    assert not mismatches, ('a lift changed a surviving block\'s free symbols, so the pass-lifetime '
                            f'memo is unsound: {mismatches}')


def test_reused_analysis_costs_no_maps_against_the_matcher():
    """Sharing analysis must not lose parallelism. Caching REFUSALS did, which is why it is gone."""
    base = three_independent_sweeps.to_sdfg(simplify=True)

    reference = copy.deepcopy(base)
    PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(reference, {})

    candidate = copy.deepcopy(base)
    ParallelizeLoops().apply_pass(candidate, {})

    assert map_count(candidate) >= map_count(reference)
    assert loop_count(candidate) <= loop_count(reference)
    candidate.validate()


def test_lifts_outermost_first():
    """Order is correctness-relevant, not a preference: lifting the inner loop first puts the body
    behind a NestedSDFG whose propagated memlet fails the enclosing loop's ``a*i+b`` write check."""
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    depths = [loop_order_key(loop) for loop in sorted(candidate_loops(sdfg), key=loop_order_key)]
    assert depths == sorted(depths), 'candidates must be visited outermost-first'


def test_invariants_survive_the_per_lift_context_rebuild():
    """The memo has to outlive the context, or it saves nothing: contexts are dropped every lift."""
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    invariants = build_lift_invariants(sdfg)
    assert invariants.block_free_symbols == {}

    block = all_blocks(sdfg)[0]
    ctx = build_lift_context(sdfg, invariants)
    assert isinstance(ctx, LiftContext) and ctx.invariants is invariants
    first = block_free_symbols(block, ctx)

    rebuilt = build_lift_context(sdfg, invariants)
    assert block in rebuilt.invariants.block_free_symbols, 'the memo did not survive the rebuild'
    assert block_free_symbols(block, rebuilt) is first


def test_numerics_survive_the_pass():
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    sdfg.validate()

    a = np.zeros((N, N))
    b = np.random.rand(N, N)
    c = np.zeros(N)
    a_ref, b_ref, c_ref = a.copy(), b.copy(), c.copy()
    three_independent_sweeps.f(a_ref, b_ref, c_ref)

    sdfg(a=a, b=b, c=c)
    assert np.allclose(a, a_ref)
    assert np.allclose(b, b_ref)
    assert np.allclose(c, c_ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
