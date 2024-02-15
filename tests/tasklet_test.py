# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math as mt
import numpy as np


@dace.program
def myprint(input, N, M):

    @dace.tasklet
    def myprint():
        a << input
        for i in range(0, N):
            for j in range(0, M):
                mt.sin(a[i, j])


def test():
    input = dace.ndarray([10, 10], dtype=dace.float32)
    input[:] = np.random.rand(10, 10).astype(dace.float32.type)

    myprint(input, 10, 10)


def test_duplicate_connector_name():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_array('b', [20], dace.float64)
    sdfg.add_array('out', [20], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('add', {'a', 'b'}, {'out'}, 'out = a + 2*b')
    state.add_edge(state.add_read('a'), None, t, 'b', dace.Memlet('a[0]'))
    state.add_edge(state.add_read('b'), None, t, 'a', dace.Memlet('b[0]'))
    state.add_edge(t, 'out', state.add_write('b'), None, dace.Memlet('b[0]'))

    a = np.random.rand(20)
    b = np.random.rand(20)
    out = np.random.rand(20)
    ref = b + 2 * a
    sdfg(a=a, b=b, out=out)
    assert np.allclose(b[0], ref[0])


if __name__ == "__main__":
    test()
    test_duplicate_connector_name()
