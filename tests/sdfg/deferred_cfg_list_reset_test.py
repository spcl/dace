# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``deferred_cfg_list_reset`` batches the whole-tree CFG-list resets of a rewrite into one."""
import copy

import pytest

import dace
from dace.sdfg.state import LoopRegion


def sdfg_with_loops(count: int) -> dace.SDFG:
    sdfg = dace.SDFG('deferred')
    last = sdfg.add_state('entry', is_start_block=True)
    for k in range(count):
        loop = LoopRegion(f'loop{k}', 'i < 4', 'i', 'i = 0', 'i = i + 1')
        loop.add_state(f'body{k}', is_start_block=True)
        sdfg.add_node(loop)
        sdfg.add_edge(last, loop, dace.InterstateEdge())
        last = loop
    return sdfg


def fresh(sdfg: dace.SDFG) -> list:
    return list(sdfg.all_control_flow_regions(recursive=True))


def test_regions_added_in_the_block_are_listed_once_it_ends():
    sdfg = sdfg_with_loops(1)
    with sdfg.deferred_cfg_list_reset():
        loop = LoopRegion('late', 'j < 2', 'j', 'j = 0', 'j = j + 1')
        loop.add_state('late_body', is_start_block=True)
        sdfg.add_node(loop)
        assert loop not in sdfg.cfg_list
    assert sdfg.cfg_list == fresh(sdfg)
    assert all(region.cfg_list is sdfg.cfg_list for region in fresh(sdfg))


def test_a_block_that_asked_for_no_reset_leaves_the_list_object_alone():
    """A pass that applies nothing must not renumber a list an earlier pass left behind."""
    sdfg = sdfg_with_loops(2)
    before = sdfg.cfg_list
    with sdfg.deferred_cfg_list_reset():
        pass
    assert sdfg.cfg_list is before


def test_an_inner_block_leaves_the_reset_to_the_outer_one():
    sdfg = sdfg_with_loops(1)
    with sdfg.deferred_cfg_list_reset():
        with sdfg.nodes()[1].deferred_cfg_list_reset():
            sdfg.add_node(LoopRegion('inner_added', 'k < 2', 'k', 'k = 0', 'k = k + 1'))
        assert len(sdfg.cfg_list) == 2
    assert sdfg.cfg_list == fresh(sdfg)


def test_the_reset_still_happens_when_the_block_raises():
    sdfg = sdfg_with_loops(1)
    with pytest.raises(RuntimeError):
        with sdfg.deferred_cfg_list_reset():
            sdfg.add_node(LoopRegion('before_raise', 'k < 2', 'k', 'k = 0', 'k = k + 1'))
            raise RuntimeError('rewrite failed')
    assert sdfg.cfg_list == fresh(sdfg)
    assert sdfg.cfg_list_reset_pending is None


def test_a_copy_taken_inside_the_block_resets_normally():
    """The deferral lives on the tree being rewritten; a deep copy made meanwhile is its own tree."""
    sdfg = sdfg_with_loops(1)
    with sdfg.deferred_cfg_list_reset():
        clone = copy.deepcopy(sdfg)
    clone.add_node(LoopRegion('clone_added', 'k < 2', 'k', 'k = 0', 'k = k + 1'))
    assert clone.cfg_list == fresh(clone)
