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
    nsdfg_node = state.add_nested_sdfg(nsdfg,
                                       None, {}, {'a1', 'a2'} if squeeze else {'a'},
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


if __name__ == '__main__':
    test_memlets_no_squeeze()
    test_memlets_squeeze()
