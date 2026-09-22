# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reach sets of :class:`ControlFlowBlockReachability` and :class:`StateReachability`.

Codegen places allocations by these sets and iterates some of them, so their iteration order is part of
the contract, and a set must answer ``in`` and ``len`` exactly as its items do.
"""
import copy
from typing import Callable, Dict, List

import pytest

import dace
from dace.sdfg.state import BreakBlock, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.analysis import ControlFlowBlockReachability, StateReachability

Labels = Dict[int, Dict[str, List[str]]]


def cyclic_sdfg() -> dace.SDFG:
    """A cycle ``a <-> c`` entered from two sides, then a single-state loop, then a sink.

    Every region holds at most one block, so no iteration order here depends on object addresses.
    """
    sdfg = dace.SDFG('reach_order')
    s0 = sdfg.add_state('s0', is_start_block=True)
    a, b, c, end = (sdfg.add_state(label) for label in ('a', 'b', 'c', 'end'))
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    loop.add_state('body', is_start_block=True)
    for src, dst in ((s0, b), (s0, a), (a, c), (b, c), (c, a), (c, loop), (loop, end)):
        sdfg.add_edge(src, dst, dace.InterstateEdge())
    return sdfg


def nested_sdfg() -> dace.SDFG:
    """Every block kind the analysis expands: nested loops, a two-branch conditional, a break, a back edge, and
    a nested SDFG with a loop of its own."""
    sdfg = dace.SDFG('reach_nested')
    start = sdfg.add_state('start', is_start_block=True)
    outer = LoopRegion('outer', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(outer)
    head = outer.add_state('head', is_start_block=True)
    branch = ConditionalBlock('branch')
    outer.add_node(branch)
    taken = ControlFlowRegion('taken')
    branch.add_branch('i > 1', taken)
    taken_state = taken.add_state('taken_state', is_start_block=True)
    stop = BreakBlock('stop')
    taken.add_node(stop)
    taken.add_edge(taken_state, stop, dace.InterstateEdge())
    other = ControlFlowRegion('other')
    branch.add_branch(None, other)
    other.add_state('other_state', is_start_block=True)
    inner = LoopRegion('inner', 'j < 3', 'j', 'j = 0', 'j = j + 1')
    outer.add_node(inner)
    inner.add_state('inner_body', is_start_block=True)
    outer.add_edge(head, branch, dace.InterstateEdge())
    outer.add_edge(branch, inner, dace.InterstateEdge())
    retry = sdfg.add_state('retry')
    end = sdfg.add_state('end')
    sdfg.add_edge(start, outer, dace.InterstateEdge())
    sdfg.add_edge(outer, retry, dace.InterstateEdge())
    sdfg.add_edge(retry, start, dace.InterstateEdge(condition='i < 2'))
    sdfg.add_edge(retry, end, dace.InterstateEdge(condition='i >= 2'))

    callee = dace.SDFG('callee')
    callee_start = callee.add_state('callee_start', is_start_block=True)
    callee_loop = LoopRegion('callee_loop', 'k < 2', 'k', 'k = 0', 'k = k + 1')
    callee.add_node(callee_loop)
    callee_loop.add_state('callee_body', is_start_block=True)
    callee.add_edge(callee_start, callee_loop, dace.InterstateEdge())
    head.add_nested_sdfg(callee, {}, {})
    return sdfg


def labels(result: Dict) -> Labels:
    return {
        cfg_id: {
            block.label: [r.label for r in reach]
            for block, reach in sets.items()
        }
        for cfg_id, sets in result.items()
    }


def block_reach(sdfg: dace.SDFG) -> Dict:
    return ControlFlowBlockReachability().apply_pass(sdfg, {})


def single_level_reach(sdfg: dace.SDFG) -> Dict:
    return ControlFlowBlockReachability(contain_to_single_level=True).apply_pass(sdfg, {})


def state_reach(sdfg: dace.SDFG) -> Dict:
    return StateReachability().apply_pass(sdfg, {})


@pytest.mark.parametrize(('analysis', 'expected'), [
    (block_reach, {
        0: {
            's0': ['b', 'a', 'c', 'body', 'loop', 'end'],
            'a': ['c', 'a', 'body', 'loop', 'end'],
            'b': ['c', 'a', 'body', 'loop', 'end'],
            'c': ['a', 'body', 'loop', 'c', 'end'],
            'loop': ['end'],
        },
        1: {
            'body': ['body', 'loop', 'end']
        },
    }),
    (single_level_reach, {
        0: {
            's0': ['b', 'a', 'c', 'loop', 'body', 'end'],
            'a': ['c', 'a', 'loop', 'body', 'end'],
            'b': ['c', 'a', 'loop', 'body', 'end'],
            'c': ['a', 'loop', 'body', 'c', 'end'],
            'end': [],
            'loop': ['end'],
        },
        1: {
            'body': ['body']
        },
    }),
    (state_reach, {
        0: {
            's0': ['b', 'a', 'c', 'body', 'end'],
            'a': ['c', 'a', 'body', 'end'],
            'b': ['c', 'a', 'body', 'end'],
            'c': ['a', 'body', 'c', 'end'],
            'body': ['body', 'end'],
        },
    }),
],
                         ids=['block', 'single_level', 'state'])
def test_reach_sets_list_blocks_in_breadth_first_order_with_each_region_before_what_follows_it(
        analysis: Callable[[dace.SDFG], Dict], expected: Labels):
    """Keys follow graph order, and a block reached only through a cycle lists itself where the cycle closes."""
    got = labels(analysis(cyclic_sdfg()))
    assert got == expected, got


@pytest.mark.parametrize('analysis', [block_reach, single_level_reach, state_reach],
                         ids=['block', 'single_level', 'state'])
def test_membership_and_size_of_a_reach_set_match_its_items(analysis: Callable[[dace.SDFG], Dict]):
    """``in`` and ``len`` are answered without listing the items, so they must not drift from the listing."""
    sdfg = nested_sdfg()
    universe = list(sdfg.all_control_flow_blocks(recursive=True))
    result = analysis(sdfg)
    answers = {
        (cfg_id, block): (len(reach), [b for b in universe if b in reach])
        for cfg_id, sets in result.items()
        for block, reach in sets.items()
    }
    assert any(size > 0 for size, _ in answers.values())
    for (cfg_id, block), (size, members) in answers.items():
        items = list(result[cfg_id][block])
        assert size == len(items), (block.label, size, [b.label for b in items])
        assert set(members) == set(items), (block.label, [b.label for b in members], [b.label for b in items])


@pytest.mark.parametrize('analysis', [block_reach, single_level_reach, state_reach],
                         ids=['block', 'single_level', 'state'])
def test_a_reach_set_describes_the_graph_as_it_was_when_the_analysis_ran(analysis: Callable[[dace.SDFG], Dict]):
    """A consumer holding a result while it adds blocks (``UniqueLoopIterators`` does) must keep reading the
    reachability it computed, whether it lists a set before or after the edit."""
    sdfg = nested_sdfg()
    listed_before = labels(analysis(sdfg))
    result = analysis(sdfg)
    blocks = {block.label: block for block in sdfg.all_control_flow_blocks(recursive=True)}
    late = sdfg.add_state('late')
    sdfg.add_edge(blocks['end'], late, dace.InterstateEdge())
    sdfg.add_edge(blocks['start'], late, dace.InterstateEdge())
    sdfg.remove_edge(sdfg.edges_between(blocks['retry'], blocks['start'])[0])
    blocks['outer'].add_state('appended')
    assert all(late not in reach for sets in result.values() for reach in sets.values())
    assert labels(result) == listed_before


def test_a_copy_of_an_unlisted_reach_set_has_the_same_items_in_the_same_order():
    sdfg = cyclic_sdfg()
    listed = [block.label for block in block_reach(sdfg)[0][sdfg.start_block]]
    reach = block_reach(sdfg)[0][sdfg.start_block]
    copies = (copy.copy(reach), reach.copy())
    assert [[block.label for block in duplicate] for duplicate in copies] == [listed, listed]


@pytest.mark.parametrize('analysis', [block_reach, single_level_reach, state_reach],
                         ids=['block', 'single_level', 'state'])
def test_unordered_members_of_a_reach_set_are_its_items_without_listing_them(analysis: Callable[[dace.SDFG], Dict]):
    """``unordered`` reads the members off the bitset: the item set, and the set stays unlisted."""
    for sdfg in (cyclic_sdfg(), nested_sdfg()):
        result = analysis(sdfg)
        for sets in result.values():
            for reach in sets.values():
                deferred = reach.replay is not None
                members = list(reach.unordered())
                assert (reach.replay is not None) == deferred
                assert len(members) == len(set(members)) == len(reach)
                assert set(members) == set(reach)
                # Once listed, the same members come from the listing.
                assert set(reach.unordered()) == set(members)


def test_dead_dataflow_elimination_does_not_list_the_reach_sets():
    """Its read-set union has no order, so the reach sets it reads stay deferred."""
    from dace.transformation import pass_pipeline as ppl
    from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination

    sdfg = nested_sdfg()
    results = {}
    ppl.Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, results)
    reach = results[ControlFlowBlockReachability.__name__]
    deferred = [r for sets in reach.values() for r in sets.values() if len(r) > 0]
    assert deferred and all(r.replay is not None for r in deferred)
