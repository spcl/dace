# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import sympy

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import utils as sdutils


def test_loop_inlining_regular_for():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1', inverted=False)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_loop_blocks(sdfg)

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    assert len(states) == 8
    assert state0 in states
    assert state1 in states
    assert state2 in states
    assert state3 in states


def test_loop_inlining_regular_while():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10')
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_loop_blocks(sdfg)

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    guard = None
    for state in states:
        if state.label == 'loop1_guard':
            guard = state
            break
    assert guard is not None
    cond_edges = sdfg.out_edges(guard)
    assert len(cond_edges) == 2
    assert cond_edges[0].data.condition_sympy() == sympy.Not(cond_edges[1].data.condition_sympy())
    assert len(states) == 8
    assign_edges = sdfg.in_edges(guard)
    assert len(assign_edges) == 2
    assert not any(e.data.assignments for e in assign_edges)
    assert state0 in states
    assert state1 in states
    assert state2 in states
    assert state3 in states


def test_loop_inlining_do_while():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', inverted=True)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_loop_blocks(sdfg)

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    guard = None
    init_state = None
    for state in states:
        if state.label == 'loop1_guard':
            guard = state
        elif state.label == 'loop1_init':
            init_state = state
    assert guard is not None
    cond_edges = sdfg.out_edges(guard)
    assert len(cond_edges) == 2
    assert cond_edges[0].data.condition_sympy() == sympy.Not(cond_edges[1].data.condition_sympy())
    assert len(states) == 8
    assign_edges = sdfg.in_edges(guard)
    assert len(assign_edges) == 1
    assert not assign_edges[0].data.assignments
    init_edges = sdfg.out_edges(init_state)
    assert len(init_edges) == 1
    assert not init_edges[0].data.assignments
    assert state0 in states
    assert state1 in states
    assert state2 in states
    assert state3 in states


def test_loop_inlining_do_for():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1', inverted=True)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_loop_blocks(sdfg)

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    guard = None
    init_state = None
    for state in states:
        if state.label == 'loop1_guard':
            guard = state
        elif state.label == 'loop1_init':
            init_state = state
    assert guard is not None
    cond_edges = sdfg.out_edges(guard)
    assert len(cond_edges) == 2
    assert cond_edges[0].data.condition_sympy() == sympy.Not(cond_edges[1].data.condition_sympy())
    assert len(states) == 8
    assign_edges = sdfg.in_edges(guard)
    assert len(assign_edges) == 1
    assert assign_edges[0].data.assignments == {'i': '(i + 1)'}
    init_edges = sdfg.out_edges(init_state)
    assert len(init_edges) == 1
    assert init_edges[0].data.assignments == {'i': '0'}
    assert state0 in states
    assert state1 in states
    assert state2 in states
    assert state3 in states


if __name__ == '__main__':
    test_loop_inlining_regular_for()
    test_loop_inlining_regular_while()
    test_loop_inlining_do_while()
    test_loop_inlining_do_for()
