# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The core graph operations keep the CFG list and every parent pointer equal to a fresh reset, in place."""
import copy

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from tests.cfg_tree import (assert_tree_consistent, assert_tree_matches_a_reset, conditional, inner_sdfg, loop,
                            spy_on_resets)


def two_level_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('outer')
    sdfg.add_symbol('n', dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    entry.add_nested_sdfg(inner_sdfg('a', 1), {}, {}, symbol_mapping={'n': 'n'})
    guard = conditional('top_if')
    sdfg.add_node(guard)
    sdfg.add_edge(entry, guard, dace.InterstateEdge())
    body = loop('top_loop')
    sdfg.add_node(body)
    sdfg.add_edge(guard, body, dace.InterstateEdge())
    body.nodes()[0].add_nested_sdfg(inner_sdfg('b', 1), {}, {}, symbol_mapping={'n': 'n'})
    return sdfg


def find(sdfg: dace.SDFG, label: str):
    return next(r for r in sdfg.all_control_flow_regions(recursive=True) if r.label == label)


def add_region_first_in_a_nested_branch(sdfg):
    branch = find(sdfg, 'a_if_b0')
    branch.add_node(loop('late'))


def add_state_holding_a_nested_sdfg(sdfg):
    state = dace.SDFGState('carrier')
    state.add_nested_sdfg(inner_sdfg('c', 0), {}, {}, symbol_mapping={'n': 'n'})
    find(sdfg, 'top_loop').add_node(state)


def add_nested_sdfg_into_an_early_state(sdfg):
    sdfg.nodes()[0].add_nested_sdfg(inner_sdfg('d', 1), {}, {}, symbol_mapping={'n': 'n'})


def remove_a_region_with_nested_sdfgs(sdfg):
    sdfg.remove_node(find(sdfg, 'top_loop'))


def remove_a_nested_sdfg_node(sdfg):
    state = sdfg.nodes()[0]
    state.remove_node(next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)))


def add_and_remove_branches(sdfg):
    block = find(sdfg, 'top_if')
    extra = ControlFlowRegion('top_if_extra')
    extra.add_state('top_if_extra_s', is_start_block=True)
    extra.add_node(loop('extra_loop'))
    block.add_branch(CodeBlock('n > 5'), extra)
    block.remove_branch(find(sdfg, 'top_if_b0'))


def reorder_a_region(sdfg):
    sdfg.reorder_nodes(list(reversed(sdfg.nodes())))


def move_a_block_between_regions(sdfg):
    moved = find(sdfg, 'a_loop')
    source = moved.parent_graph
    source.remove_node(moved)
    find(sdfg, 'top_if_b1').add_node(moved)


def re_add_a_deep_copy(sdfg):
    find(sdfg, 'top_if_b1').add_node(copy.deepcopy(find(sdfg, 'a_if')))


def add_state_before_a_region(sdfg):
    sdfg.add_state_before(find(sdfg, 'top_if'), 'before_if')


def add_blocks_to_the_caller_before_dropping_the_callee(sdfg):
    """What inlining does: the callee's blocks join the caller while the callee still lists them."""
    state = sdfg.nodes()[0]
    node = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    callee = node.sdfg
    for block in list(callee.nodes()):
        sdfg.add_node(block)
    state.remove_node(node)


def move_blocks_into_a_new_nested_sdfg_before_dropping_their_region(sdfg):
    """What a lift does: a loop's blocks join a fresh body SDFG, which is nested, then the loop leaves."""
    moved_loop = find(sdfg, 'top_loop')
    body = sdfg.add_state_before(moved_loop, 'lifted_body')
    fresh = dace.SDFG('lifted', parent=body)
    fresh.add_symbol('n', dace.int64)
    for k, block in enumerate(list(moved_loop.nodes())):
        fresh.add_node(block, is_start_block=k == 0)
    body.add_nested_sdfg(fresh, {}, {}, symbol_mapping={'n': 'n'})
    sdfg.remove_node(moved_loop)


def move_a_nested_sdfg_node_to_another_state_before_dropping_it(sdfg):
    """What state splitting does: the node joins the new state while the old one still holds it."""
    state = sdfg.nodes()[0]
    node = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    after = sdfg.add_state_after(state, 'split_after')
    after.add_node(node)
    state.remove_node(node)


@pytest.mark.parametrize('operation', [
    add_region_first_in_a_nested_branch,
    add_state_holding_a_nested_sdfg,
    add_nested_sdfg_into_an_early_state,
    remove_a_region_with_nested_sdfgs,
    remove_a_nested_sdfg_node,
    add_and_remove_branches,
    reorder_a_region,
    move_a_block_between_regions,
    re_add_a_deep_copy,
    add_state_before_a_region,
    add_blocks_to_the_caller_before_dropping_the_callee,
    move_blocks_into_a_new_nested_sdfg_before_dropping_their_region,
    move_a_nested_sdfg_node_to_another_state_before_dropping_it,
])
def test_a_core_operation_keeps_the_tree_as_a_reset_builds_it(operation, monkeypatch):
    """``cfg_id`` is a position in the list, which serialization and code generation read; a pass keeps it
    exact only if every operation does, without a whole-tree reset per change."""
    sdfg = two_level_sdfg()
    assert_tree_consistent(sdfg)
    resets = []
    original = dace.sdfg.state.AbstractControlFlowRegion.reset_cfg_list
    monkeypatch.setattr(dace.sdfg.state.AbstractControlFlowRegion, 'reset_cfg_list',
                        lambda self: resets.append(self) or original(self))
    operation(sdfg)
    assert resets == [], [r.label for r in resets]
    assert_tree_consistent(sdfg)


N = dace.symbol('N')


@dace.program
def maps_loops_and_guards(a: dace.float64[N], b: dace.float64[N, N], c: dace.int32[1]):
    for i in dace.map[0:N]:
        a[i] = a[i] + 1.0
    if c[0] > 0:
        for i in range(N):
            for j in dace.map[0:N]:
                b[i, j] = a[j] * 2.0
    if c[0] > 0:
        for i in range(N):
            a[i] = b[i, i] - 1.0
    for i in range(N):
        for j in range(N):
            b[i, j] = b[i, j] + a[i]


def lower_maps(sdfg):
    from dace.transformation.dataflow.map_for_loop import MapToForLoop
    from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere
    PatternApplyOnceEverywhere([MapToForLoop()], validate=False, state_local=True).apply_pass(sdfg, {})


def nest_a_map(sdfg):
    from dace.transformation.helpers import nest_state_subgraph
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and state.entry_node(node) is None:
                nest_state_subgraph(sdfg, state, state.scope_subgraph(node))
                return


def fuse_conditions(sdfg):
    from dace.transformation.passes.canonicalize.fuse_conditions import FuseConditions
    FuseConditions(matcher_order=True).apply_pass(sdfg, {})


def move_ifs_into_loops(sdfg):
    from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
    MoveIfIntoLoop().apply_pass(sdfg, {})


def parallelize(sdfg):
    from dace.transformation.passes.parallelize_loops import ParallelizeLoops
    # Without the closing memlet propagation: its reachability analysis resets the list itself.
    ParallelizeLoops(propagate=False).apply_pass(sdfg, {})


@pytest.mark.parametrize('rewrite', [lower_maps, nest_a_map, fuse_conditions, move_ifs_into_loops, parallelize])
def test_a_rewrite_keeps_the_tree_as_a_reset_builds_it_and_computes_the_same(rewrite, monkeypatch):
    sdfg = maps_loops_and_guards.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    resets = []
    original = dace.sdfg.state.AbstractControlFlowRegion.reset_cfg_list
    monkeypatch.setattr(dace.sdfg.state.AbstractControlFlowRegion, 'reset_cfg_list',
                        lambda self: resets.append(self) or original(self))
    rewrite(sdfg)
    monkeypatch.undo()
    assert resets == [], [r.label for r in resets]
    assert_tree_consistent(sdfg)
    sdfg.validate()
    n = 7
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random((n, n))
    for cval in (0, 1):
        a_ref, b_ref, a_out, b_out = a.copy(), b.copy(), a.copy(), b.copy()
        c = np.array([cval], np.int32)
        reference(a=a_ref, b=b_ref, c=c, N=n)
        sdfg(a=a_out, b=b_out, c=c, N=n)
        np.testing.assert_allclose(a_out, a_ref)
        np.testing.assert_allclose(b_out, b_ref)


def test_a_region_pass_may_remove_the_region_it_visits():
    """PruneEmptyConditionalBranches replaces a ConditionalBlock whose branches are all empty, and the
    region-pass loop then read the removed block's ``cfg_id``: the block is no longer in the CFG list,
    so canonicalize raised ``list.index(x): x not in list``."""
    from dace.transformation.passes.simplification.prune_empty_conditional_branches import \
        PruneEmptyConditionalBranches
    sdfg = dace.SDFG('prune_the_visited_region')
    sdfg.add_symbol('n', dace.int64)
    first = sdfg.add_state('first', is_start_block=True)
    guard = conditional('guard')
    sdfg.add_node(guard)
    sdfg.add_edge(first, guard, dace.InterstateEdge())
    sdfg.add_edge(guard, sdfg.add_state('last'), dace.InterstateEdge())
    visited_id = guard.cfg_id

    result = PruneEmptyConditionalBranches().apply_pass(sdfg, {})
    assert result is not None and visited_id in result, result
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.nodes()), 'the empty ConditionalBlock is gone'
    assert_tree_consistent(sdfg)
    sdfg.validate()


def test_set_nested_sdfg_parent_references_rebuilds_the_list_once_and_repairs_every_level(monkeypatch):
    """It reset the whole tree once per nested SDFG, quadratic in their number (wfg NormalizeMapBody)."""
    sdfg = two_level_sdfg()
    nested = [n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]
    assert len(nested) == 4
    for sd in nested:
        sd._parent_sdfg = None
    resets = spy_on_resets(monkeypatch)
    dace.sdfg.utils.set_nested_sdfg_parent_references(sdfg)
    monkeypatch.undo()
    assert resets == [sdfg]
    assert_tree_matches_a_reset(sdfg)
