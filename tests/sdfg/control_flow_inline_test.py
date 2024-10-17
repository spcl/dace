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

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

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

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

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

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

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

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

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


def test_inline_triple_nested_for():
    sdfg = dace.SDFG('gemm')
    N = dace.symbol('N')
    M = dace.symbol('M')
    K = dace.symbol('K')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [N, K], dace.float32)
    sdfg.add_array('B', [K, M], dace.float32)
    sdfg.add_array('C', [N, M], dace.float32)
    sdfg.add_array('tmp', [N, M, K], dace.float32, transient=True)
    i_loop = LoopRegion('outer', 'i < N', 'i', 'i = 0', 'i = i + 1')
    j_loop = LoopRegion('middle', 'j < M', 'j', 'j = 0', 'j = j + 1')
    k_loop = LoopRegion('inner', 'k < K', 'k', 'k = 0', 'k = k + 1')
    reduce_state = sdfg.add_state('reduce')
    sdfg.add_node(i_loop, is_start_block=True)
    sdfg.add_edge(i_loop, reduce_state, dace.InterstateEdge())
    i_loop.add_node(j_loop, is_start_block=True)
    j_loop.add_node(k_loop, is_start_block=True)
    comp_state = k_loop.add_state('comp', is_start_block=True)
    anode = comp_state.add_access('A')
    bnode = comp_state.add_access('B')
    tmpnode = comp_state.add_access('tmp')
    tasklet = comp_state.add_tasklet('comp', {'a', 'b'}, {'t'}, 't = a * b')
    comp_state.add_memlet_path(anode, tasklet, dst_conn='a', memlet=dace.Memlet.simple('A', 'i, k'))
    comp_state.add_memlet_path(bnode, tasklet, dst_conn='b', memlet=dace.Memlet.simple('B', 'k, j'))
    comp_state.add_memlet_path(tasklet, tmpnode, src_conn='t', memlet=dace.Memlet.simple('tmp', 'i, j, k'))

    tmpnode2 = reduce_state.add_access('tmp')
    cnode = reduce_state.add_access('C')
    red = reduce_state.add_reduce('lambda a, b: a + b', (2,), 0)
    reduce_state.add_edge(tmpnode2, None, red, None, dace.Memlet.simple('tmp', '0:N, 0:M, 0:K'))
    reduce_state.add_edge(red, None, cnode, None, dace.Memlet.simple('C', '0:N, 0:M'))

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

    assert len(sdfg.nodes()) == 14
    assert not any(isinstance(s, LoopRegion) for s in sdfg.nodes())
    assert sdfg.is_valid()


def test_loop_inlining_for_continue_break():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1', inverted=False)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_continue('state2')
    state3 = loop1.add_state('state3')
    state4 = loop1.add_break('state4')
    state5 = loop1.add_state('state5')
    state6 = loop1.add_state('state6')
    loop1.add_edge(state1, state2, dace.InterstateEdge(condition='i < 5'))
    loop1.add_edge(state1, state3, dace.InterstateEdge(condition='i >= 5'))
    loop1.add_edge(state3, state4, dace.InterstateEdge(condition='i < 6'))
    loop1.add_edge(state3, state5, dace.InterstateEdge(condition='i >= 6'))
    loop1.add_edge(state5, state6, dace.InterstateEdge())
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    state7 = sdfg.add_state('state7')
    sdfg.add_edge(loop1, state7, dace.InterstateEdge())

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    assert len(states) == 12
    assert not any(isinstance(s, LoopRegion) for s in states)
    end_state = None
    latch_state = None
    break_state = None
    continue_state = None
    for state in states:
        if state.label == 'loop1_end':
            end_state = state
        elif state.label == 'loop1_latch':
            latch_state = state
        elif state.label == 'loop1_state2':
            continue_state = state
        elif state.label == 'loop1_state4':
            break_state = state
    assert end_state is not None
    assert len(sdfg.edges_between(break_state, end_state)) == 1
    assert len(sdfg.edges_between(continue_state, latch_state)) == 1


def test_loop_inlining_multi_assignments():
    sdfg = dace.SDFG('inlining')
    sdfg.add_symbol('j', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0; j = 10 + 200 - 1',
                       update_expr='i = i + 1; j = j + i', inverted=False)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

    states = sdfg.nodes() # Get top-level states only, not all (.states()), in case something went wrong
    assert len(states) == 8
    assert state0 in states
    assert state1 in states
    assert state2 in states
    assert state3 in states

    guard_state = None
    init_state = None
    latch_state = None
    for state in sdfg.states():
        if state.label == 'loop1_guard':
            guard_state = state
        elif state.label == 'loop1_init':
            init_state = state
        elif state.label == 'loop1_latch':
            latch_state = state
    init_edge = sdfg.edges_between(init_state, guard_state)[0]
    assert 'i' in init_edge.data.assignments
    assert 'j' in init_edge.data.assignments
    update_edge = sdfg.edges_between(latch_state, guard_state)[0]
    assert 'i' in update_edge.data.assignments
    assert 'j' in update_edge.data.assignments


def test_loop_inlining_invalid_update_statement():
    # Inlining should not be applied here.
    sdfg = dace.SDFG('inlining')
    sdfg.add_symbol('j', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1; j < i', inverted=False)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_control_flow_regions(sdfg, [LoopRegion])

    nodes = sdfg.nodes()
    assert len(nodes) == 3


if __name__ == '__main__':
    test_loop_inlining_regular_for()
    test_loop_inlining_regular_while()
    test_loop_inlining_do_while()
    test_loop_inlining_do_for()
    test_inline_triple_nested_for()
    test_loop_inlining_for_continue_break()
    test_loop_inlining_multi_assignments()
    test_loop_inlining_invalid_update_statement()
