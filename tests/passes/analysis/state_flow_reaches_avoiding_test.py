# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StateFlow.reaches_avoiding`` answers from one search per (source, kill) pair: the same answers
as a search per query, over loops, branches and a kill on the only path."""
import itertools
from collections import deque
from unittest import mock

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.analysis.analysis import StateFlow


def branchy_loop_sdfg() -> dace.SDFG:
    """start -> loop{ head -> if{ a | b } -> tail } -> end."""
    sdfg = dace.SDFG('reaches_avoiding')
    start = sdfg.add_state('start', is_start_block=True)
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    head = loop.add_state('head', is_start_block=True)
    cond = ConditionalBlock('cond')
    loop.add_node(cond)
    for label, condition in (('a', CodeBlock('i < 2')), ('b', None)):
        branch = ControlFlowRegion(f'branch_{label}', sdfg=sdfg)
        branch.add_state(label, is_start_block=True)
        cond.add_branch(condition, branch)
    tail = loop.add_state('tail')
    loop.add_edge(head, cond, dace.InterstateEdge())
    loop.add_edge(cond, tail, dace.InterstateEdge())
    end = sdfg.add_state('end')
    sdfg.add_edge(start, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    return sdfg


def search_per_query(flow: StateFlow, src, dst, kill) -> bool:
    """The early-exit breadth-first search, one per query."""
    target, blocked = ('in', id(dst)), ('in', id(kill))
    seen = {('out', id(src))}
    work = deque(seen)
    while work:
        node = work.popleft()
        if node == target:
            return True
        if node == blocked:
            continue
        for nxt in flow.succ.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                work.append(nxt)
    return False


def test_answers_match_a_search_per_query():
    sdfg = branchy_loop_sdfg()
    flow = StateFlow(sdfg)
    states = list(sdfg.all_states())
    answers = {
        (s.label, d.label, k.label): flow.reaches_avoiding(s, d, k)
        for s, d, k in itertools.product(states, repeat=3)
    }
    assert answers == {
        (s.label, d.label, k.label): search_per_query(flow, s, d, k)
        for s, d, k in itertools.product(states, repeat=3)
    }
    # The kill really cuts: 'head' is the only way from 'start' into the loop body.
    assert not answers[('start', 'a', 'head')] and answers[('start', 'a', 'b')]
    assert answers[('tail', 'head', 'a')] and answers[('start', 'end', 'end')]


def test_consecutive_queries_for_one_pair_search_once():
    sdfg = branchy_loop_sdfg()
    flow = StateFlow(sdfg)
    states = {s.label: s for s in sdfg.all_states()}
    with mock.patch.object(StateFlow, 'reached_avoiding', autospec=True, side_effect=StateFlow.reached_avoiding) as spy:
        for dst in states.values():
            flow.reaches_avoiding(states['head'], dst, states['b'])
        flow.reaches_avoiding(states['start'], states['end'], states['b'])
        flow.reaches_avoiding(states['head'], states['end'], states['b'])
    assert spy.call_count == 3


if __name__ == '__main__':
    test_answers_match_a_search_per_query()
    test_consecutive_queries_for_one_pair_search_once()
