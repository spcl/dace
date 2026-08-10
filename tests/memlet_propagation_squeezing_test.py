# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import propagation
import numpy as np


def make_sdfg(squeeze, name):
    N, M = dace.symbol('N'), dace.symbol('M')
    sdfg = dace.SDFG('memlet_propagation_%s' % name)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [N + 1, M], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map('map', dict(j='1:M'))
    w = state.add_write('A')

    # Create nested SDFG
    nsdfg = dace.SDFG('nested')
    if squeeze:
        nsdfg.add_array('a1', [N + 1], dace.int64, strides=[M])
        nsdfg.add_array('a2', [N - 1], dace.int64, strides=[M])
    else:
        nsdfg.add_array('a', [N + 1, M], dace.int64)

    nstate = nsdfg.add_state()
    a1 = nstate.add_write('a1' if squeeze else 'a')
    a2 = nstate.add_write('a2' if squeeze else 'a')
    t1 = nstate.add_tasklet('add99', {}, {'out'}, 'out = i + 99')
    t2 = nstate.add_tasklet('add101', {}, {'out'}, 'out = i + 101')
    nstate.add_edge(t1, 'out', a1, None, dace.Memlet('a1[i]' if squeeze else 'a[i, 1]'))
    nstate.add_edge(t2, 'out', a2, None, dace.Memlet('a2[i]' if squeeze else 'a[i+2, 0]'))
    nsdfg.add_loop(None, nstate, None, 'i', '0', 'i < N - 2', 'i + 1')

    # Connect nested SDFG to toplevel one
    nsdfg_node = state.add_nested_sdfg(nsdfg, {}, {'a1', 'a2'} if squeeze else {'a'},
                                       symbol_mapping=dict(j='j', N='N', M='M'))
    state.add_nedge(me, nsdfg_node, dace.Memlet())
    # Add outer memlet that is overapproximated
    if squeeze:
        # This is expected to propagate to A[0:N - 2, j].
        state.add_memlet_path(nsdfg_node, mx, w, src_conn='a1', memlet=dace.Memlet('A[0:N+1, j]'))
        # This is expected to propagate to A[2:N, j - 1].
        state.add_memlet_path(nsdfg_node, mx, w, src_conn='a2', memlet=dace.Memlet('A[2:N+1, j-1]'))
    else:
        # This memlet is expected to propagate to A[0:N, j - 1:j + 1].
        state.add_memlet_path(nsdfg_node, mx, w, src_conn='a', memlet=dace.Memlet('A[0:N+1, j-1:j+1]'))

    propagation.propagate_memlets_sdfg(sdfg)

    return sdfg


def make_conditional_sdfg():
    M = dace.symbol('M')
    sdfg = dace.SDFG('memlet_propagation_conditional')
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('cond', dace.bool_)
    sdfg.add_array('A', [4, M], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map('map', dict(j='1:M'))
    w = state.add_write('A')

    nsdfg = dace.SDFG('nested_conditional')
    nsdfg.add_symbol('M', dace.int64)
    nsdfg.add_symbol('cond', dace.bool_)
    nsdfg.add_array('a', [4, M], dace.int64)

    cond_region = dace.sdfg.state.ConditionalBlock('if_region', sdfg=nsdfg)

    then_body = dace.sdfg.state.ControlFlowRegion('then_body', sdfg=nsdfg, parent=cond_region)
    then_state = then_body.add_state('then_state', is_start_block=True)
    then_write = then_state.add_write('a')
    then_tasklet = then_state.add_tasklet('write_one', {}, {'out'}, 'out = 1')
    then_state.add_edge(then_tasklet, 'out', then_write, None, dace.Memlet('a[0, 1]'))

    else_body = dace.sdfg.state.ControlFlowRegion('else_body', sdfg=nsdfg, parent=cond_region)
    else_state = else_body.add_state('else_state', is_start_block=True)
    else_write = else_state.add_write('a')
    else_tasklet_1 = else_state.add_tasklet('write_two', {}, {'out'}, 'out = 2')
    else_tasklet_2 = else_state.add_tasklet('write_three', {}, {'out'}, 'out = 3')
    else_state.add_edge(else_tasklet_1, 'out', else_write, None, dace.Memlet('a[2, 0]'))
    else_state.add_edge(else_tasklet_2, 'out', else_write, None, dace.Memlet('a[3, 0]'))

    cond_region.add_branch(dace.sdfg.state.CodeBlock('cond'), then_body)
    cond_region.add_branch(dace.sdfg.state.CodeBlock('not cond'), else_body)
    nsdfg.add_node(cond_region, is_start_block=True)

    nsdfg_node = state.add_nested_sdfg(nsdfg, {}, {'a'}, symbol_mapping=dict(j='j', M='M', cond='cond'))
    state.add_nedge(me, nsdfg_node, dace.Memlet())
    state.add_memlet_path(nsdfg_node, mx, w, src_conn='a', memlet=dace.Memlet('A[0:4, j-1:j+1]'))

    propagation.propagate_memlets_sdfg(sdfg)

    return sdfg


def make_inverted_loop_sdfg():
    M = dace.symbol('M')
    sdfg = dace.SDFG('memlet_propagation_inverted_loop')
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [5, M], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map('map', dict(j='1:M'))
    w = state.add_write('A')

    nsdfg = dace.SDFG('nested_inverted_loop')
    nsdfg.using_explicit_control_flow = True
    nsdfg.add_symbol('M', dace.int64)
    nsdfg.add_symbol('i', dace.int64)
    nsdfg.add_array('a', [5, M], dace.int64)

    loop_region = dace.sdfg.state.LoopRegion('loop_region',
                                             condition_expr='i < 3',
                                             loop_var='i',
                                             initialize_expr='i = 0',
                                             update_expr='i = i + 1',
                                             inverted=True,
                                             sdfg=nsdfg,
                                             update_before_condition=False)
    loop_state = loop_region.add_state('loop_state', is_start_block=True)
    loop_write = loop_state.add_write('a')
    loop_tasklet = loop_state.add_tasklet('write_value', {}, {'out'}, 'out = i + 7')
    loop_state.add_edge(loop_tasklet, 'out', loop_write, None, dace.Memlet('a[i, 1]'))
    loop_latch = loop_region.add_state('loop_latch')
    loop_region.add_edge(loop_state, loop_latch, dace.InterstateEdge())
    nsdfg.add_node(loop_region, is_start_block=True)

    nsdfg_node = state.add_nested_sdfg(nsdfg, {}, {'a'}, symbol_mapping=dict(j='j', M='M'))
    state.add_nedge(me, nsdfg_node, dace.Memlet())
    state.add_memlet_path(nsdfg_node, mx, w, src_conn='a', memlet=dace.Memlet('A[0:5, j-1]'))

    propagation.propagate_memlets_sdfg(sdfg)

    return sdfg


def test_memlets_no_squeeze():
    sdfg = make_sdfg(False, 'nonsqueezed')

    N = 20
    M = 10
    A = np.random.randint(0, 100, size=[N + 1, M]).astype(np.int64)
    expected = np.copy(A)
    expected[0:N - 2, 1:M] = np.reshape(np.arange(N - 2) + 99, (N - 2, 1))
    expected[2:N, 0:M - 1] = np.reshape(np.arange(N - 2) + 101, (N - 2, 1))

    sdfg(A=A, N=N, M=M)
    assert np.allclose(A, expected)

    # Check the propagated memlet out of the nested SDFG.
    N = dace.symbolic.symbol('N')
    j = dace.symbolic.symbol('j')
    main_state = sdfg.nodes()[0]
    out_memlet = main_state.edges()[1].data
    assert out_memlet.volume == 2 * N - 4
    assert out_memlet.dynamic == False
    assert out_memlet.subset[0] == (0, N - 1, 1)
    assert out_memlet.subset[1] == (j - 1, j, 1)


def test_memlets_squeeze():
    sdfg = make_sdfg(True, 'squeezed')

    N = 20
    M = 10
    A = np.random.randint(0, 100, size=[N + 1, M]).astype(np.int64)
    expected = np.copy(A)
    expected[0:N - 2, 1:M] = np.reshape(np.arange(N - 2) + 99, (N - 2, 1))
    expected[2:N, 0:M - 1] = np.reshape(np.arange(N - 2) + 101, (N - 2, 1))

    sdfg(A=A, N=N, M=M)
    assert np.allclose(A, expected)

    # Check the propagated memlets out of the nested SDFG.
    N = dace.symbolic.symbol('N')
    j = dace.symbolic.symbol('j')
    main_state = sdfg.nodes()[0]
    out_memlet_1 = main_state.edges()[1].data
    assert out_memlet_1.volume == N - 2
    assert out_memlet_1.dynamic == False
    assert out_memlet_1.subset[0] == (0, N - 3, 1)
    assert out_memlet_1.subset[1] == (j, j, 1)
    out_memlet_2 = main_state.edges()[3].data
    assert out_memlet_2.volume == N - 2
    assert out_memlet_2.dynamic == False
    assert out_memlet_2.subset[0] == (2, N - 1, 1)
    assert out_memlet_2.subset[1] == (j - 1, j - 1, 1)


def test_memlets_conditional_upper_bound():
    sdfg = make_conditional_sdfg()

    M = 6
    A = np.zeros((4, M), dtype=np.int64)
    expected_then = np.copy(A)
    expected_then[0, 1:M] = 1

    sdfg(A=A, M=M, cond=True)
    assert np.allclose(A, expected_then)

    A = np.zeros((4, M), dtype=np.int64)
    expected_else = np.copy(A)
    expected_else[2, 0:M - 1] = 2
    expected_else[3, 0:M - 1] = 3

    sdfg(A=A, M=M, cond=False)
    assert np.allclose(A, expected_else)

    j = dace.symbolic.symbol('j')
    main_state = sdfg.nodes()[0]
    out_memlet = main_state.edges()[1].data
    assert out_memlet.volume == 2
    assert out_memlet.dynamic == True
    assert out_memlet.subset[0] == (0, 3, 1)
    assert out_memlet.subset[1] == (j - 1, j, 1)


def test_memlets_inverted_loop():
    sdfg = make_inverted_loop_sdfg()

    M = 6
    A = np.zeros((5, M), dtype=np.int64)
    expected = np.copy(A)
    expected[0:4, 1:M] = np.reshape(np.arange(4) + 7, (4, 1))

    sdfg(A=A, M=M)
    assert np.allclose(A, expected)

    j = dace.symbolic.symbol('j')
    main_state = sdfg.nodes()[0]
    out_memlet = main_state.edges()[1].data
    assert out_memlet.volume == 4
    assert out_memlet.dynamic == False
    assert out_memlet.subset[0] == (0, 3, 1)
    assert out_memlet.subset[1] == (j - 1, j, 1)


if __name__ == '__main__':
    test_memlets_no_squeeze()
    test_memlets_squeeze()
    test_memlets_conditional_upper_bound()
    test_memlets_inverted_loop()
