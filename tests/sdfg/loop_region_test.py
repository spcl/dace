# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree import sdfg_to_tree as s2t, treenodes as tn


def _make_regular_for_loop() -> SDFG:
    sdfg = dace.SDFG('regular_for')
    sdfg.using_explicit_control_flow = True
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1',
                       condition_expr='i < 10',
                       loop_var='i',
                       initialize_expr='i = 0',
                       update_expr='i = i + 1',
                       inverted=False)
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
    return sdfg


def _make_regular_while_loop() -> SDFG:
    sdfg = dace.SDFG('regular_while')
    sdfg.using_explicit_control_flow = True
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
    return sdfg


def _make_do_while_loop() -> SDFG:
    sdfg = dace.SDFG('do_while')
    sdfg.using_explicit_control_flow = True
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
    return sdfg


def _make_do_for_loop() -> SDFG:
    sdfg = dace.SDFG('do_for')
    sdfg.using_explicit_control_flow = True
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [10], dace.float32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1',
                       condition_expr='i < 10',
                       loop_var='i',
                       initialize_expr='i = 0',
                       update_expr='i = i + 1',
                       inverted=True)
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
    return sdfg


def _make_do_for_inverted_cond_loop() -> SDFG:
    sdfg = dace.SDFG('do_for_inverted_cond')
    sdfg.using_explicit_control_flow = True
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [10], dace.float32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion(label='loop1',
                       condition_expr='i < 8',
                       loop_var='i',
                       initialize_expr='i = 0',
                       update_expr='i = i + 1',
                       inverted=True,
                       update_before_condition=False)
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
    return sdfg


def _make_triple_nested_for_loop() -> SDFG:
    sdfg = dace.SDFG('gemm')
    sdfg.using_explicit_control_flow = True
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
    red = reduce_state.add_reduce('lambda a, b: a + b', (2, ), 0)
    reduce_state.add_edge(tmpnode2, None, red, None, dace.Memlet.simple('tmp', '0:N, 0:M, 0:K'))
    reduce_state.add_edge(red, None, cnode, None, dace.Memlet.simple('C', '0:N, 0:M'))
    return sdfg


def test_loop_regular_for():
    sdfg = _make_regular_for_loop()

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_regular_while():
    sdfg = _make_regular_while_loop()

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_do_while():
    sdfg = _make_do_while_loop()

    assert sdfg.is_valid()

    a_validation = np.zeros([11], dtype=np.float32)
    a_test = np.zeros([11], dtype=np.float32)
    a_validation[10] = 10
    sdfg(A=a_test)
    assert np.allclose(a_validation, a_test)
    assert 'do {' in sdfg.generate_code()[0].code


def test_loop_do_for():
    sdfg = _make_do_for_loop()

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(10):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_do_for_inverted_condition():
    sdfg = _make_do_for_inverted_cond_loop()

    assert sdfg.is_valid()

    a_validation = np.zeros([10], dtype=np.float32)
    a_test = np.zeros([10], dtype=np.float32)
    sdfg(A=a_test)
    for i in range(9):
        a_validation[i] = i
    assert np.allclose(a_validation, a_test)


def test_loop_triple_nested_for():
    sdfg = _make_triple_nested_for_loop()

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


def test_loop_to_stree_regular_for():
    sdfg = _make_regular_for_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    assert stree.as_string() == (f'{tn.INDENTATION}for i = 0; (i < 10); i = (i + 1):\n' +
                                 f'{2 * tn.INDENTATION}A[i] = tasklet()')


def test_loop_to_stree_regular_while():
    sdfg = _make_regular_while_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    assert stree.as_string() == (f'{tn.INDENTATION}assign i = 0\n' + f'{tn.INDENTATION}while (i < 10):\n' +
                                 f'{2 * tn.INDENTATION}A[i] = tasklet()\n' + f'{2 * tn.INDENTATION}assign i = (i + 1)')


def test_loop_to_stree_do_while():
    sdfg = _make_do_while_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    assert stree.as_string() == (f'{tn.INDENTATION}assign i = 10\n' + f'{tn.INDENTATION}do:\n' +
                                 f'{2 * tn.INDENTATION}A[i] = tasklet()\n' +
                                 f'{2 * tn.INDENTATION}assign i = (i + 1)\n' + f'{tn.INDENTATION}while (i < 10)')


def test_loop_to_stree_do_for():
    sdfg = _make_do_for_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    assert stree.as_string() == (f'{tn.INDENTATION}i = 0\n' + f'{tn.INDENTATION}do:\n' +
                                 f'{2 * tn.INDENTATION}A[i] = tasklet()\n' + f'{2 * tn.INDENTATION}i = (i + 1)\n' +
                                 f'{tn.INDENTATION}while (i < 10)')


def test_loop_to_stree_do_for_inverted_cond():
    sdfg = _make_do_for_inverted_cond_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    assert stree.as_string() == (f'{tn.INDENTATION}i = 0\n' + f'{tn.INDENTATION}while True:\n' +
                                 f'{2 * tn.INDENTATION}A[i] = tasklet()\n' +
                                 f'{2 * tn.INDENTATION}if (not (i < 8)):\n' + f'{3 * tn.INDENTATION}break\n' +
                                 f'{2 * tn.INDENTATION}i = (i + 1)\n')


def test_loop_to_stree_triple_nested_for():
    sdfg = _make_triple_nested_for_loop()

    assert sdfg.is_valid()

    stree = s2t.as_schedule_tree(sdfg)

    po_nodes = list(stree.preorder_traversal())[1:]
    assert [type(n) for n in po_nodes
            ] == [tn.LoopScope, tn.LoopScope, tn.LoopScope, tn.TaskletNode, tn.LibraryCall]


if __name__ == '__main__':
    test_loop_regular_for()
    test_loop_regular_while()
    test_loop_do_while()
    test_loop_do_for()
    test_loop_do_for_inverted_condition()
    test_loop_triple_nested_for()
    test_loop_to_stree_regular_for()
    test_loop_to_stree_regular_while()
    test_loop_to_stree_do_while()
    test_loop_to_stree_do_for()
    test_loop_to_stree_do_for_inverted_cond()
    test_loop_to_stree_triple_nested_for()
