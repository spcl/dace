# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.state import LoopRegion


def test_loop_regular_for():
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

    assert sdfg.is_valid()


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

    assert sdfg.is_valid()


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

    assert sdfg.is_valid()


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

    assert sdfg.is_valid()


def test_tripple_nested_for():
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
    tasklet = comp_state.add_tasklet('comp', {'a', 'b'}, {'tmp'}, 'tmp = a * b')
    comp_state.add_memlet_path(anode, tasklet, dst_conn='a', memlet=dace.Memlet.simple('A', 'i, k'))
    comp_state.add_memlet_path(bnode, tasklet, dst_conn='b', memlet=dace.Memlet.simple('B', 'k, j'))
    comp_state.add_memlet_path(tasklet, tmpnode, src_conn='tmp', memlet=dace.Memlet.simple('tmp', 'i, j, k'))

    tmpnode2 = reduce_state.add_access('tmp')
    cnode = reduce_state.add_access('C')
    red = reduce_state.add_reduce('lambda a, b: a + b', axes=[2])
    reduce_state.add_edge(tmpnode2, None, red, None, dace.Memlet.simple('tmp', 'i, j, k'))
    reduce_state.add_edge(red, None, cnode, None, dace.Memlet.simple('C', 'i, j'))

    assert sdfg.is_valid()


if __name__ == '__main__':
    test_loop_regular_for()
    test_loop_inlining_regular_while()
    test_loop_inlining_do_while()
    test_loop_inlining_do_for()
    test_tripple_nested_for()
