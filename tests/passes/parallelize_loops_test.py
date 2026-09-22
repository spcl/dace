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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import sympy

import dace
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_to_map import (LiftContext, LoopToMap, block_free_symbols, build_lift_context,
                                                        build_lift_invariants)
from dace.transformation.passes.analysis import smt_dependence
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


@dace.program
def prefix_max(x: dace.float64[N], out: dace.float64[N]):
    best = -1.0
    for i in range(N):
        v = abs(x[i])
        if v > best:
            best = v + 0
        out[i] = best


def test_a_running_max_with_a_conditional_update_stays_a_loop():
    """``best`` becomes a symbol read by the ``if`` and assigned in its branch; hiding that read made
    the loop look parallel and the map computed a per-element max, not a prefix max (cegterg)."""
    sdfg = prefix_max.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    sdfg.validate()

    assert any(isinstance(region, LoopRegion) for region in sdfg.all_control_flow_regions(recursive=True))
    x = np.random.default_rng(7).standard_normal(N) * 4.0
    out = np.zeros(N)
    sdfg(x=x, out=out)
    assert np.array_equal(out, np.maximum.accumulate(np.maximum(np.abs(x), -1.0)))


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


@dace.program
def one_sweep(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        b[i] = a[i] * 2.0 + 1.0


@dace.program
def carried_sweep(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i]


@dace.program
def permuted_scatter(a: dace.float64[N], b: dace.float64[N], idx: dace.int64[N]):
    for i in range(N):
        a[idx[i]] = b[i]


def test_lifting_one_loop_leaves_the_sdfg_the_sweep_leaves():
    """A pass that lifts one given loop must get the sweep's post-lift steps, not a bare ``LoopToMap.apply``."""
    swept = one_sweep.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(swept, {})

    single = one_sweep.to_sdfg(simplify=True)
    (loop, ) = candidate_loops(single)
    assert ParallelizeLoops().parallelize_loop(single, loop)

    single.validate()
    assert single.hash_sdfg() == swept.hash_sdfg()


def test_a_loop_the_probe_refuses_is_left_untouched():
    sdfg = carried_sweep.to_sdfg(simplify=True)
    before = sdfg.hash_sdfg()
    (loop, ) = candidate_loops(sdfg)

    assert not ParallelizeLoops().parallelize_loop(sdfg, loop)
    assert sdfg.hash_sdfg() == before


def test_a_proven_lift_is_taken_where_the_probe_refuses():
    """A caller's own proof (here: ``idx`` is a permutation) replaces the probe, which cannot see it."""
    sdfg = permuted_scatter.to_sdfg(simplify=True)
    (loop, ) = candidate_loops(sdfg)
    assert not LoopToMap.can_be_applied_to(sdfg, loop=loop), 'fixture broken: the probe accepts the scatter'

    assert ParallelizeLoops().parallelize_loop(sdfg, loop, proven=True)
    assert loop_count(sdfg) == 0
    sdfg.validate()

    rng = np.random.default_rng(0)
    a = np.zeros(N)
    b = rng.random(N)
    idx = rng.permutation(N).astype(np.int64)
    a_ref = a.copy()
    a_ref[idx] = b
    sdfg(a=a, b=b, idx=idx)
    np.testing.assert_array_equal(a, a_ref)


def context_facts(sd: dace.SDFG) -> Tuple[Dict[str, List[dace.SDFGState]], List[Any], OrderedSet[str]]:
    ctx = build_lift_context(sd, build_lift_invariants(sd))
    access = {name: list(states) for name, states in ctx.access_states.items()}
    return access, list(ctx.block_order), OrderedSet(sd.free_symbols)


def test_a_lift_leaves_every_other_sdfgs_context_exact() -> None:
    """The invariant that lets a lift drop only its OWN SDFG's context.

    A lift rewrites its own SDFG, and outside it touches only the mapping of the node nesting it,
    which only the parent's free symbols read. So every other SDFG's access-node index, block order
    and free symbols are unchanged. A kept context that went stale would hand the next probe a wrong
    index or a wrong "used after the loop" answer, which is a miscompile, so it is asserted.
    """
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    real_apply = LoopToMap.apply
    stale: List[str] = []
    comparisons = 0
    nested_lifts = 0

    def checking_apply(self: LoopToMap, graph: ControlFlowRegion, inner_sdfg: dace.SDFG) -> Any:
        root = inner_sdfg
        while root.parent_sdfg is not None:
            root = root.parent_sdfg
        pnode = inner_sdfg.parent_nsdfg_node
        keys = None if pnode is None else tuple(pnode.symbol_mapping.keys())
        before = {sd: context_facts(sd) for sd in root.all_sdfgs_recursive() if sd is not inner_sdfg}
        out = real_apply(self, graph, inner_sdfg)
        nonlocal comparisons, nested_lifts
        nested_lifts += pnode is not None
        mapping_moved = pnode is not None and keys != tuple(pnode.symbol_mapping.keys())
        for sd, was in before.items():
            if mapping_moved and sd is inner_sdfg.parent_sdfg:
                continue  # the one context the pass drops besides the lifted SDFG's own
            comparisons += 1
            if context_facts(sd) != was:
                stale.append(sd.label)
        return out

    LoopToMap.apply = checking_apply
    try:
        ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    finally:
        LoopToMap.apply = real_apply

    assert nested_lifts > 0 and comparisons > 0, 'no lift ran beside another SDFG, so nothing was exercised'
    assert not stale, f'a lift changed the context of an SDFG it did not lift in: {stale}'


@dace.program
def colliding_scatter_then_independent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        a[min(i, N - 1 - i)] = b[i]
    for i in range(N):
        c[i] = b[i] + 1.0


def test_a_refused_smt_write_reaches_z3_once_however_often_its_loop_is_reprobed() -> None:
    """Every lift restarts the sweep, so a loop the oracle refused is probed again after each one.
    The question it asks is the same text every time; z3 must answer it once per pass."""
    sdfg = colliding_scatter_then_independent.to_sdfg(simplify=True)
    real_prove = smt_dependence.prove_injective_write
    real_can = LoopToMap.can_be_applied
    questions: List[Tuple[str, str, str, str]] = []
    probes: List[LoopRegion] = []

    def spy_prove(write_expr: sympy.Basic, itervar: str, start: Any, end: Any, *args: Any,
                  **kwargs: Any) -> Optional[bool]:
        questions.append((str(write_expr), itervar, str(start), str(end)))
        return real_prove(write_expr, itervar, start, end, *args, **kwargs)

    def spy_can(self: LoopToMap,
                graph: ControlFlowRegion,
                expr_index: int,
                inner_sdfg: dace.SDFG,
                permissive: bool = False) -> bool:
        probes.append(self.loop)
        return real_can(self, graph, expr_index, inner_sdfg, permissive)

    smt_dependence.prove_injective_write = spy_prove
    LoopToMap.can_be_applied = spy_can
    try:
        ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    finally:
        smt_dependence.prove_injective_write = real_prove
        LoopToMap.can_be_applied = real_can

    refused = candidate_loops(sdfg)
    assert len(refused) == 1 and map_count(sdfg) == 1, 'the colliding scatter must stay a loop, the other lift'
    assert probes.count(refused[0]) >= 3, 'the refused loop was not re-probed, so nothing was exercised'
    assert questions == [(f'Min(i, {N - 1} - i)', 'i', '0', str(N - 1))], f'z3 was asked again: {questions}'


def test_a_sweep_rebuilds_the_cfg_list_once(monkeypatch) -> None:
    """Every lift rebuilt the CFG list of the whole tree, and the sweep never reads it: 9% of the
    parallelize stage on warpx_field_gather (3300 lifts in one SDFG)."""
    sdfg = three_independent_sweeps.to_sdfg(simplify=True)
    lists = [sdfg.cfg_list]
    original = dace.sdfg.state.AbstractControlFlowRegion.reset_cfg_list

    def recorded(self):
        result = original(self)
        if sdfg.cfg_list is not lists[-1]:
            lists.append(sdfg.cfg_list)
        return result

    monkeypatch.setattr(dace.sdfg.state.AbstractControlFlowRegion, 'reset_cfg_list', recorded)
    lifted = ParallelizeLoops(propagate=False).apply_pass(sdfg, {})
    assert lifted and lifted > 1, lifted
    assert len(lists) == 2, len(lists)
    assert sdfg.cfg_list == list(sdfg.all_control_flow_regions(recursive=True))
    sdfg.validate()
