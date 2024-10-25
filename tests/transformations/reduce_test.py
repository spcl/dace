# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.transformation.dataflow import RedundantArray

import dace
from dace import SDFG, Memlet

C_in, C_out, H, K, N, W = (dace.symbol(s, dace.int64) for s in ('C_in', 'C_out', 'H', 'K', 'N', 'W'))


def make_sdfg():
    g = SDFG('prog')
    g.add_array('A', (N, 1, 1, C_in, C_out), dace.float32,
                strides=(C_in * C_out, C_in * C_out, C_in * C_out, C_out, 1))
    g.add_array('C', (N, H, W, C_out), dace.float32,
                strides=(C_out * H * W, C_out * W, C_out, 1))

    st0 = g.add_state('st0', is_start_block=True)
    st = st0

    A = st.add_access('A')
    C = st.add_access('C')
    R = st.add_reduce('lambda x, y: x + y', [1, 2, 3], 0)
    st.add_memlet_path(A, R, memlet=Memlet(expr='A[0:N, 0, 0, 0:C_in, 0:C_out]'))
    st.add_memlet_path(R, C, memlet=Memlet(expr='C[0:N, 5, 5, 0:C_out]'))

    return g, R


def test_library_node_expand_reduce_pure():
    n, cin, cout = 7, 7, 7
    h, k, w = 25, 35, 45
    A = np.ones((n, 1, 1, cin, cout), np.float32)

    g, R = make_sdfg()
    R.implementation = 'pure-seq'
    g.validate()
    g.compile()

    wantC = np.ones((n, h, w, cout), np.float32) * 42
    g(A=A, C=wantC, N=n, C_in=cin, C_out=cout, H=h, K=k, W=w)

    g, R = make_sdfg()
    R.implementation = 'pure'
    g.validate()
    g.compile()

    gotC = np.ones((n, h, w, cout), np.float32) * 42
    g(A=A, C=gotC, N=n, C_in=cin, C_out=cout, H=h, K=k, W=w)
    assert np.allclose(wantC, gotC)


if __name__ == '__main__':
    test_library_node_expand_reduce_pure()
