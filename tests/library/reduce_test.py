# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
import dace.libraries.standard as std
from dace import SDFG, Memlet

C_in, C_out, H, K, N, W = (dace.symbol(s, dace.int64) for s in ('C_in', 'C_out', 'H', 'K', 'N', 'W'))


def make_sdfg():
    g = SDFG('prog')
    g.add_array('A', (N, 1, 1, C_in, C_out), dace.float32, strides=(C_in * C_out, C_in * C_out, C_in * C_out, C_out, 1))
    g.add_array('C', (N, H, W, C_out), dace.float32, strides=(C_out * H * W, C_out * W, C_out, 1))

    st0 = g.add_state('st0', is_start_block=True)
    st = st0

    A = st.add_access('A')
    C = st.add_access('C')
    R = st.add_reduce('lambda x, y: x + y', [1, 2, 3], 0)
    st.add_nedge(A, R, Memlet(expr='A[0:N, 0, 0, 0:C_in, 0:C_out]'))
    st.add_nedge(R, C, Memlet(expr='C[0:N, 5, 5, 0:C_out]'))

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


_params = ['pure', 'CUDA (device)', 'pure-seq', 'GPUAuto']


@pytest.mark.gpu
@pytest.mark.parametrize('impl', _params)
def test_multidim_gpu(impl):
    test_cases = [([1, 64, 60, 60], (0, 2, 3), [64], np.float32), ([8, 512, 4096], (0, 1), [4096], np.float32),
                  ([8, 512, 4096], (0, 1), [4096], np.float64), ([1024, 8], (0), [8], np.float32),
                  ([111, 111, 111], (0, 1), [111], np.float64), ([111, 111, 111], (1, 2), [111], np.float64),
                  ([1000000], (0), [1], np.float64), ([1111111], (0), [1], np.float64),
                  ([123, 21, 26, 8], (1, 2), [123, 8], np.float32), ([2, 512, 2], (0, 2), [512], np.float32),
                  ([512, 555, 257], (0, 2), [555], np.float64)]

    for in_shape, ax, out_shape, dtype in test_cases:
        print(in_shape, ax, out_shape, dtype)
        axes = ax

        @dace.program
        def multidimred(a, b):
            b[:] = np.sum(a, axis=axes)

        a = np.random.rand(*in_shape).astype(dtype)
        b = np.random.rand(*out_shape).astype(dtype)
        sdfg = multidimred.to_sdfg(a, b)
        sdfg.apply_gpu_transformations()
        rednode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
        rednode.implementation = impl

        sdfg(a, b)

        assert np.allclose(b, np.sum(a, axis=axes))


if __name__ == '__main__':
    for p in _params:
        test_multidim_gpu(p)
    test_library_node_expand_reduce_pure()
