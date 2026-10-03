# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the chain fast path in ``blockorder_topological_sort``.

The sort runs the whole dominator machinery -- immediate dominators, the dominator closure, a
branch-merge scan and a parent tree -- for every region it visits, and it recurses, so every nested
region pays separately. A region whose blocks form a straight chain has exactly ONE topological
order, the one its edges already spell out, so that machinery can only rediscover it. The fast path
walks the edges instead; what needs guarding is that it agrees with the general path wherever it
fires, and that it declines everywhere the order is a real question.
"""
from collections import defaultdict

import dace
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ControlFlowRegion


def general_order(cfg):
    """The order the pre-fast-path implementation produced, straight from the dominator machinery."""
    loopexits = defaultdict(lambda: None)
    idom = cfg_analysis.block_immediate_dominators(cfg)
    alldoms = cfg_analysis.all_dominators(cfg, idom)
    merges = cfg_analysis.branch_merges(cfg, idom, alldoms)
    ptree = cfg_analysis.block_parent_tree(cfg, loopexits, idom=idom, merges=merges, alldoms=alldoms)
    return list(cfg_analysis._blockorder_topological_sort(cfg, cfg.start_block, ptree, merges, loopexits=loopexits))


def test_a_chain_takes_the_fast_path_and_agrees_with_the_dominator_walk():
    sdfg = dace.SDFG('chain')
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    third = sdfg.add_state('third')
    sdfg.add_edge(first, second, dace.InterstateEdge())
    sdfg.add_edge(second, third, dace.InterstateEdge())

    fast = cfg_analysis._chain_order(sdfg)
    assert fast is not None, 'a straight chain did not take the fast path'
    assert fast == [first, second, third]
    assert fast == general_order(sdfg), 'the fast path disagreed with the dominator walk'


def test_a_branch_declines_the_fast_path():
    """Two successors: which one comes first is a real question, so the general path must answer it."""
    sdfg = dace.SDFG('branch')
    sdfg.add_symbol('c', dace.int32)
    head = sdfg.add_state('head', is_start_block=True)
    left = sdfg.add_state('left')
    right = sdfg.add_state('right')
    tail = sdfg.add_state('tail')
    sdfg.add_edge(head, left, dace.InterstateEdge(condition='c > 0'))
    sdfg.add_edge(head, right, dace.InterstateEdge(condition='not (c > 0)'))
    sdfg.add_edge(left, tail, dace.InterstateEdge())
    sdfg.add_edge(right, tail, dace.InterstateEdge())

    assert cfg_analysis._chain_order(sdfg) is None, 'a branching region must fall through to the general path'
    assert set(cfg_analysis.blockorder_topological_sort(sdfg)) == {head, left, right, tail}


def test_a_back_edge_declines_the_fast_path():
    """A cycle has no chain order; the walk must notice the revisit rather than spin."""
    region = ControlFlowRegion('cyclic')
    sdfg = dace.SDFG('with_cycle')
    sdfg.add_node(region, is_start_block=True)
    one = region.add_state('one', is_start_block=True)
    two = region.add_state('two')
    region.add_edge(one, two, dace.InterstateEdge())
    region.add_edge(two, one, dace.InterstateEdge())

    assert cfg_analysis._chain_order(region) is None, 'a back edge must fall through to the general path'


def test_an_unreachable_block_declines_the_fast_path():
    """The walk reaches only part of the region, so it must not report a partial order as complete."""
    sdfg = dace.SDFG('unreachable')
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge())
    sdfg.add_state('orphan')  # no edge into it

    assert cfg_analysis._chain_order(sdfg) is None, 'an unreached block must fall through to the general path'


def test_an_empty_region_orders_to_nothing():
    assert cfg_analysis._chain_order(ControlFlowRegion('empty')) == []


if __name__ == '__main__':
    test_a_chain_takes_the_fast_path_and_agrees_with_the_dominator_walk()
    test_a_branch_declines_the_fast_path()
    test_a_back_edge_declines_the_fast_path()
    test_an_unreachable_block_declines_the_fast_path()
    test_an_empty_region_orders_to_nothing()
