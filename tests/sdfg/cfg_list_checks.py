# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Checks that a rewrite kept the CFG list exact in place, without a whole-tree reset."""
import copy
import sys
from typing import Callable, List, Tuple

import dace
from dace.sdfg.state import AbstractControlFlowRegion
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent, conditional, inner_sdfg


def cfg_rows(sdfg: dace.SDFG) -> List[Tuple[str, str, int]]:
    return [(type(region).__name__, region.label, region.cfg_id) for region in sdfg.cfg_list]


def assert_cfg_list_as_after_a_reset(sdfg: dace.SDFG) -> None:
    """The kept list, every ``cfg_id`` and every parent pointer against the pre-order walk, and against a deep
    copy whose list ``reset_cfg_list`` rebuilt from scratch."""
    assert_tree_consistent(sdfg)
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    assert cfg_rows(sdfg) == cfg_rows(fresh)


def record_tree_resets(monkeypatch, root: Callable[[dace.SDFG], bool]) -> List[str]:
    """From now on, the function that asked for each reset of a whole tree whose root ``root`` accepts."""
    callers: List[str] = []
    original = AbstractControlFlowRegion.reset_cfg_list
    depth = [0]

    def recorded(self):
        # A reset climbs to the root by calling itself on the parent: only the outermost call is a request.
        caller = sys._getframe(1).f_code.co_name
        depth[0] += 1
        try:
            result = original(self)
        finally:
            depth[0] -= 1
        if depth[0] == 0 and result and root(result[0]):
            callers.append(caller)
        return result

    monkeypatch.setattr(AbstractControlFlowRegion, 'reset_cfg_list', recorded)
    return callers


def loop_over_nested_sdfg(multi_state: bool) -> dace.SDFG:
    """pre -> loop{ body[nested SDFG holding a loop, a branch and a further nested SDFG] (-> if) } -> post."""
    sdfg = dace.SDFG(f'loop_over_nested_{int(multi_state)}')
    sdfg.add_symbol('n', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    pre = sdfg.add_state('pre', is_start_block=True)
    loop = dace.sdfg.state.LoopRegion('outer', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    body = loop.add_state('body', is_start_block=True)
    body.add_nested_sdfg(inner_sdfg('a', 1), {}, {}, symbol_mapping={'n': 'n'})
    if multi_state:
        guard = conditional('guard')
        loop.add_node(guard)
        loop.add_edge(body, guard, dace.InterstateEdge())
    post = sdfg.add_state('post')
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    return sdfg
