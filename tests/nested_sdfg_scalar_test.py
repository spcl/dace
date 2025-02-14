# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def _construct_sdfg():
    sdfg = dace.SDFG('nsstest')
    sdfg.add_array('A', [2], dace.float64)
    state = sdfg.add_state()

    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [1], dace.float64)
    nsdfg.add_array('b', [1], dace.float64)
    nstate = nsdfg.add_state()
    nstate.add_mapped_tasklet('m',
                              dict(i='0'),
                              dict(inp=dace.Memlet.simple('a', 'i')),
                              'out = inp * 5.0',
                              dict(out=dace.Memlet.simple('b', 'i')),
                              external_edges=True)

    r = state.add_read('A')
    n = state.add_nested_sdfg(nsdfg, None, {'a'}, {'b'})
    w = state.add_write('A')
    state.add_edge(r, None, n, 'a', dace.Memlet.simple('A', '1'))
    state.add_edge(n, 'b', w, None, dace.Memlet.simple('A', '0'))

    return sdfg


def test_nss():
    sdfg = _construct_sdfg()
    A = np.random.rand(2)
    sdfg(A=A)
    assert A[0] == A[1] * 5


if __name__ == '__main__':
    test_nss()
