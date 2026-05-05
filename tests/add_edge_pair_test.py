# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


def test():
    sdfg = dace.SDFG('addedgepair')
    state = sdfg.add_state()
    sdfg.add_array('A', [31], dace.float64)
    sdfg.add_array('B', [1], dace.float64)

    # Add nodes
    t = state.add_tasklet('do', {'a'}, {'b'}, 'b = 2*a')
    a = state.add_read('A')
    b = state.add_write('B')
    me, mx = state.add_map('m', dict(i='0:31'))

    # Add edges
    state.add_edge_pair(me, t, a, dace.Memlet.simple(a, 'i'), internal_connector='a')
    state.add_edge_pair(mx,
                        t,
                        b,
                        dace.Memlet.simple(b, '0', wcr_str='lambda a,b: a+b'),
                        internal_connector='b',
                        scope_connector='o')

    A = np.random.rand(31).astype(np.float64)
    B = np.array([0.], dtype=np.float64)
    sdfg(A=A, B=B)

    diff = np.linalg.norm(B[0] - np.sum(2 * A))
    print('Difference:', diff)
    assert diff <= 1e-5


@pytest.mark.parametrize('data_is_dst', (False, True))
def test_src_dst_memlet(data_is_dst):
    sdfg = dace.SDFG('aep_test')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [3, 20], dace.float64)
    sdfg.add_transient('a', [3], dace.float64)

    if data_is_dst:
        memlet = dace.Memlet(data='a', subset='0:3', other_subset='1:4, i')
    else:
        memlet = dace.Memlet(data='A', subset='1:4, i', other_subset='0:3')

    state = sdfg.add_state()
    me, mx = state.add_map('doit', dict(i='0:20'))

    r = state.add_read('A')
    w = state.add_write('B')
    a = state.add_access('a')

    state.add_edge_pair(me, a, r, memlet)
    state.add_memlet_path(a, mx, w, memlet=dace.Memlet('B[0:3, i]'))

    sdfg.validate()

    a = np.random.rand(20, 20)
    b = np.random.rand(3, 20)
    sdfg(A=a, B=b)
    assert np.allclose(a[1:4, :], b)


if __name__ == '__main__':
    test()
    test_src_dst_memlet(False)
    test_src_dst_memlet(True)
