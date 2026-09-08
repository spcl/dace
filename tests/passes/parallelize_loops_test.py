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


def test_a_lift_never_moves_the_sdfgs_own_free_symbols():
    """The invariant the whole-graph walk gate rests on.

    ``LoopToMap.apply`` used to ask for ``sdfg.free_symbols`` twice per lift -- once mid-rewrite and
    once at the end -- purely to spot symbols the lift turned free. It cannot turn any symbol free
    that the loop did not itself define: the iterate is defined by the loop before and the map after,
    and the body's interstate assignments move inside the nested SDFG together with every use of
    them. So the set is the same on both sides of a lift, and a loop declaring none of those symbols
    skips both walks. Measured on CloudSC: unchanged across all 314 lifts, 310 of which declare
    nothing. Asserted rather than argued -- a lift that did move the set would leave the next
    context holding a stale ``sdfg_free_symbols`` and mis-report what the FOLLOWING lift freed.
    """
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    real_apply = LoopToMap.apply
    moved = []
    lifts = 0

    def checking_apply(self, graph, inner_sdfg):
        before = set(inner_sdfg.free_symbols)
        out = real_apply(self, graph, inner_sdfg)
        nonlocal lifts
        lifts += 1
        after = set(inner_sdfg.free_symbols)
        if after != before:
            moved.append((inner_sdfg.label, sorted(after - before), sorted(before - after)))
        return out

    LoopToMap.apply = checking_apply
    try:
        ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    finally:
        LoopToMap.apply = real_apply

    assert lifts > 0, 'nothing was lifted, so the invariant was never exercised'
    assert not moved, f'a lift moved the SDFG free symbols: {moved}'
    sdfg.validate()


def test_a_loop_that_declares_a_body_symbol_still_gets_it_deregistered():
    """The gate must NOT fire for a loop whose body assigns a declared symbol.

    This is the case the two whole-graph walks exist for: ``k`` is declared on the SDFG and assigned
    on an interstate edge INSIDE the loop body, so the lift moves its definition into the nested
    SDFG. The lift has to deregister it and let the nested node map what it still needs; skipping
    the walk here would leave ``sdfg.symbols`` holding a symbol nothing defines any more.
    """
    sdfg = dace.SDFG('declared_body_symbol')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('n', dace.int64)
    loop = LoopRegion('sweep', 'i < n', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state('assign_k', is_start_block=True)
    second = loop.add_state('use_k')
    # ``k`` is DECLARED on the SDFG and defined only here, inside the body.
    loop.add_edge(first, second, dace.InterstateEdge(assignments={'k': 'i'}))
    tasklet = second.add_tasklet('w', {}, {'o'}, 'o = k')
    second.add_edge(tasklet, 'o', second.add_write('a'), None, dace.Memlet('a[i]'))
    sdfg.validate()

    assert ParallelizeLoops(propagate=False).apply_pass(sdfg, {}), 'the loop was not lifted'
    sdfg.validate()
    assert 'k' not in sdfg.symbols, 'a symbol defined only inside the lifted body stayed declared'

    out = np.zeros(N)
    sdfg(a=out, n=N)
    assert np.allclose(out, np.arange(N)), f'wrong values after the lift: {out}'


@dace.program
def sequential_outer_parallel_inner(a: dace.float64[N, N], b: dace.float64[N, N]):
    """A carried outer sweep over parallel inner ones, so a lift always has an ENCLOSING region."""
    for t in range(1, N):
        for j in range(N):
            b[t, j] = a[t - 1, j] * 2.0
        for j in range(N):
            a[t, j] = b[t, j] + 1.0


def test_a_lift_only_ever_removes_from_an_enclosing_regions_read_write_sets():
    """The invariant that lets an ancestor's cached read/write sets be PATCHED, not rebuilt.

    A lift adds no access: it re-homes the ones it finds behind a nested SDFG that re-exposes them
    through its connectors. So nothing can enter an enclosing region's read or write set, and the
    only names that leave are the body-local containers the lift internalized. Measured over every
    CloudSC lift -- 334 enclosing observations, 0 additions -- and asserted here, because an
    addition would leave an ancestor's patched set MISSING a container and the write analysis that
    reads it would then accept a loop it must refuse.
    """
    sdfg = sequential_outer_parallel_inner.to_sdfg(simplify=True)
    real_apply = LoopToMap.apply
    additions = []
    unexplained = []
    observations = 0

    def checking_apply(self, graph, inner_sdfg):
        ancestors = []
        region = graph
        while region is not None and not isinstance(region, dace.SDFG):
            read_set, write_set = region.read_and_write_sets()
            ancestors.append((region, set(read_set), set(write_set)))
            region = region.parent_graph
        out = real_apply(self, graph, inner_sdfg)
        ctx = vars(self).get('lift_context')
        gone = set() if ctx is None or ctx.internalized_data is None else ctx.internalized_data
        nonlocal observations
        for region, was_read, was_written in ancestors:
            if region.parent_graph is None:
                continue  # detached by the lift; nothing will ask about it again
            observations += 1
            now_read, now_written = region.read_and_write_sets()
            added = (set(now_read) - was_read) | (set(now_written) - was_written)
            removed = (was_read - set(now_read)) | (was_written - set(now_written))
            if added:
                additions.append((region.label, sorted(added)))
            if removed - gone:
                unexplained.append((region.label, sorted(removed - gone)))
        return out

    LoopToMap.apply = checking_apply
    try:
        ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    finally:
        LoopToMap.apply = real_apply

    assert observations > 0, 'no enclosing region was observed, so the invariant was never exercised'
    assert not additions, f'a lift ADDED to an enclosing region read/write set: {additions}'
    assert not unexplained, f'a name left an enclosing set without being internalized: {unexplained}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_a_mapping_the_callee_stopped_needing_is_pruned():
    """A nested SDFG node must not map a symbol its own SDFG no longer reads.

    A lift moves the loop's interstate assignments INTO the new body, so a symbol the enclosing
    SDFG used to receive from outside can end up produced inside it. The wrapping node still maps
    it, and a mapping's VALUE is what keeps a name alive in the CALLER -- so the entry holds a
    symbol one level up that nothing defines any more. Harmless where it sits, and a hard failure
    the moment that scope is itself nested: azimint_naive died on ``Missing symbols on nested SDFG:
    ['__map_fusion___tmp0', '__map_fusion___tmp1']``, one lift making the entry stale and a later
    one tripping over it.
    """
    from dace.transformation.passes.parallelize_loops import prune_stale_symbol_mappings

    inner = dace.SDFG('callee_binds_it')
    inner.add_array('a', (4, ), dace.float64)
    inner.add_symbol('k', dace.int64)
    first = inner.add_state('first', is_start_block=True)
    second = inner.add_state('second')
    # ``k`` is ASSIGNED here, so the callee does not need it from outside -- it is not free.
    inner.add_edge(first, second, dace.InterstateEdge(assignments={'k': '1'}))
    second.add_edge(second.add_tasklet('w', {}, {'o'}, 'o = 2.0'), 'o', second.add_write('a'), None,
                    dace.Memlet('a[k]'))

    sdfg = dace.SDFG('caller')
    sdfg.add_array('a', (4, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    node = state.add_nested_sdfg(inner, {}, {'a'})
    state.add_edge(node, 'a', state.add_write('a'), None, dace.Memlet('a[0:4]'))
    node.symbol_mapping['k'] = dace.symbolic.pystr_to_symbolic('k')

    assert 'k' not in {str(s) for s in inner.free_symbols}, 'fixture broken: the callee still needs k'
    assert 'k' in {str(s) for s in sdfg.free_symbols}, 'the stale entry should keep k alive in the caller'

    assert prune_stale_symbol_mappings(sdfg) == 1
    assert 'k' not in node.symbol_mapping
    assert 'k' not in {str(s) for s in sdfg.free_symbols}, 'pruning did not release the caller'
    sdfg.validate()
