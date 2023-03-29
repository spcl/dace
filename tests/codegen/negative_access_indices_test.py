# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests (relative) negative access indices. """
import dace
import numpy as np


def test_negative_access():

    M, N = dace.symbol('M'), dace.symbol('N')
    K, L = dace.symbol('K'), dace.symbol('L')

    sdfg = dace.SDFG('negative_access')
    for s in ('M', 'N', 'K', 'L'):
        sdfg.add_symbol(s, dace.int32)
    sdfg.add_array('A', [M, N], dace.int32, strides=[1, M])
    sdfg.add_array('B', [M, N], dace.int32, strides=[1, M])
    state = sdfg.add_state('state')

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_array('A', [L-K, L-K], dace.int32, strides=[1, M])
    nsdfg.add_array('B', [L-K, L-K], dace.int32, strides=[1, M])
    nstate_0 = nsdfg.add_state('nested_state_0')
    nstate_1 = nsdfg.add_state('nested_state_1')
    nstate_2 = nsdfg.add_state('nested_state_2')
    nsdfg.add_edge(nstate_0, nstate_1, dace.InterstateEdge(condition='A[-1,-2] == 10'))
    nsdfg.add_edge(nstate_0, nstate_2, dace.InterstateEdge(condition='A[-1,-2] != 10'))

    nb1 = nstate_1.add_write('B')
    t1 = nstate_1.add_tasklet('t1', {}, {'b'}, 'b = 1')
    nstate_1.add_edge(t1, 'b', nb1, None, dace.Memlet(data='B', subset='-2, -1'))

    nb2 = nstate_2.add_write('B')
    t2 = nstate_2.add_tasklet('t2', {}, {'b'}, 'b = 2')
    nstate_2.add_edge(t2, 'b', nb2, None, dace.Memlet(data='B', subset='-2, -1'))

    t = state.add_nested_sdfg(nsdfg, sdfg, {'A'}, {'B'})
    a = state.add_read('A')
    b = state.add_write('B')
    state.add_edge(a, None, t, 'A', dace.Memlet('A[K:L, K:L]'))
    state.add_edge(t, 'B', b, None, dace.Memlet('B[K:L, K:L]'))

    A = np.asfortranarray(np.arange(100, dtype=np.int32).reshape(10, 10))
    B = np.asfortranarray(np.zeros((10, 10), dtype=np.int32))
    func = sdfg.compile(validate=False)
    func(A=A, B=B, M=10, N=10, K=2, L=7)
    K, L = 2, 7
    assert B[K-2, K-1] == 1


if __name__ == '__main__':
    test_negative_access()
