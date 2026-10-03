# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StateFlow.reaches_avoiding`` answers from one dominator tree per source: the same answers as a
search per query, over loops, branches, breaks and a kill on the only path."""
import itertools
from collections import deque
from unittest import mock

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion
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


def loops_with_jumps_sdfg() -> dace.SDFG:
    """start -> outer{ pre -> inner{ x -> if{ break | continue } -> y } -> post } -> mid -> loop{ z } -> end."""
    sdfg = dace.SDFG('reaches_avoiding_jumps')
    start = sdfg.add_state('start', is_start_block=True)
    outer = LoopRegion('outer', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(outer)
    pre = outer.add_state('pre', is_start_block=True)
    inner = LoopRegion('inner', 'j < 4', 'j', 'j = 0', 'j = j + 1')
    outer.add_node(inner)
    x = inner.add_state('x', is_start_block=True)
    cond = ConditionalBlock('jump')
    inner.add_node(cond)
    for label, condition, jump in (('br', CodeBlock('j == 2'), BreakBlock), ('co', None, ContinueBlock)):
        branch = ControlFlowRegion(f'branch_{label}', sdfg=sdfg)
        branch.add_node(jump(f'{label}_jump'), is_start_block=True)
        cond.add_branch(condition, branch)
    y = inner.add_state('y')
    inner.add_edge(x, cond, dace.InterstateEdge())
    inner.add_edge(cond, y, dace.InterstateEdge())
    post = outer.add_state('post')
    outer.add_edge(pre, inner, dace.InterstateEdge())
    outer.add_edge(inner, post, dace.InterstateEdge())
    mid = sdfg.add_state('mid')
    second = LoopRegion('second', 'k < 4', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(second)
    second.add_state('z', is_start_block=True)
    end = sdfg.add_state('end')
    sdfg.add_edge(start, outer, dace.InterstateEdge())
    sdfg.add_edge(outer, mid, dace.InterstateEdge())
    sdfg.add_edge(mid, second, dace.InterstateEdge())
    sdfg.add_edge(second, end, dace.InterstateEdge())
    return sdfg


def test_answers_match_a_search_per_query_across_breaks_and_sibling_loops():
    sdfg = loops_with_jumps_sdfg()
    flow = StateFlow(sdfg)
    states = list(sdfg.all_states())
    for s, d, k in itertools.product(states, repeat=3):
        assert flow.reaches_avoiding(s, d, k) == search_per_query(flow, s, d, k), (s.label, d.label, k.label)
    labels = {s.label: s for s in states}
    # The break skips 'y', and 'post' runs after it; 'mid' is the only way into the second loop.
    assert flow.reaches_avoiding(labels['x'], labels['post'], labels['y'])
    assert not flow.reaches_avoiding(labels['pre'], labels['z'], labels['mid'])


def test_a_foreign_state_reaches_nothing():
    sdfg = branchy_loop_sdfg()
    flow = StateFlow(sdfg)
    other = dace.SDFG('other').add_state('alone', is_start_block=True)
    states = list(sdfg.all_states())
    assert not flow.reaches_avoiding(other, states[0], states[1])
    assert not flow.reaches_avoiding(states[0], other, states[1])
    assert flow.reaches_avoiding(states[0], states[-1], other)


def test_every_kill_for_one_source_shares_one_dominator_tree():
    sdfg = branchy_loop_sdfg()
    flow = StateFlow(sdfg)
    states = {s.label: s for s in sdfg.all_states()}
    with mock.patch.object(StateFlow, 'dominator_tree', autospec=True, side_effect=StateFlow.dominator_tree) as spy:
        for dst, kill in itertools.product(states.values(), repeat=2):
            flow.reaches_avoiding(states['head'], dst, kill)
        flow.reaches_avoiding(states['start'], states['end'], states['b'])
        flow.reaches_avoiding(states['head'], states['end'], states['a'])
    built = {call.args[1] for call in spy.call_args_list}
    assert len(built) == 2 and len(flow.dominance) == 2


if __name__ == '__main__':
    test_answers_match_a_search_per_query()
    test_answers_match_a_search_per_query_across_breaks_and_sibling_loops()
    test_a_foreign_state_reaches_nothing()
    test_every_kill_for_one_source_shares_one_dominator_tree()
