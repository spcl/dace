# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.state import LoopRegion


def test_loop_regular_for():
    sdfg = dace.SDFG('regular_for')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1', inverted=False)
    sdfg.add_node(loop1)
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [10], dace.float32)
    state1 = loop1.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = i')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[i]'))
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_regular_while():
    sdfg = dace.SDFG('regular_while')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10')
    sdfg.add_array('A', [10], dace.float32)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = i')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[i]'))
    sdfg.add_symbol('i', dace.int32)
    loop1.add_edge(state1, state2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge(assignments={'i': '0'}))
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_do_while():
    sdfg = dace.SDFG('do_while')
    sdfg.add_symbol('i', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', inverted=True)
    sdfg.add_node(loop1)
    sdfg.add_array('A', [10], dace.float32)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = i')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[i]'))
    loop1.add_edge(state1, state2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge(assignments={'i': '10'}))
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.zeros([11], dtype=np.float32)
    a_test = np.zeros([11], dtype=np.float32)
    a_validation[10] = 10
    sdfg(A=a_test)
    assert np.allclose(a_validation, a_test)


def test_loop_do_for():
    sdfg = dace.SDFG('do_for')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [10], dace.float32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0',
                       update_expr='i = i + 1', inverted=True)
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = i')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[i]'))
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_triple_nested_for():
    sdfg = dace.SDFG('gemm')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)
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

    assert sdfg.is_valid()

    N = 5
    M = 10
    K = 8
    A = np.random.rand(N, K).astype(np.float32)
    B = np.random.rand(K, M).astype(np.float32)
    C_test = np.random.rand(N, M).astype(np.float32)
    C_validation = np.random.rand(N, M).astype(np.float32)

    C_validation = A @ B

    sdfg(A=A, B=B, C=C_test, N=N, M=M, K=K)

    assert np.allclose(C_validation, C_test)


if __name__ == '__main__':
    test_loop_regular_for()
    test_loop_regular_while()
    test_loop_do_while()
    test_loop_do_for()
    test_triple_nested_for()
