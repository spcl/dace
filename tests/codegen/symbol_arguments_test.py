# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

N = dace.symbol('N')


def test_global_sizes():

    @dace.program
    def tester(A: dace.float64[N]):
        for i in dace.map[0:10]:
            A[i] = 2

    sdfg = tester.to_sdfg()
    # Since N is not used anywhere, it should not be listed in the arguments
    assert 'N' not in sdfg.arglist()

    a = np.random.rand(20)
    sdfg(a, N=20)
    assert np.allclose(a[:10], 2)


def test_global_sizes_used():

    @dace.program
    def tester(A: dace.float64[N]):
        for i in dace.map[0:10]:
            with dace.tasklet:
                a >> A[i]
                a = N

    sdfg = tester.to_sdfg()
    # N is used in a tasklet
    assert 'N' in sdfg.arglist()


def test_global_sizes_multidim():

    @dace.program
    def tester(A: dace.float64[N, N]):
        for i, j in dace.map[0:10, 0:10]:
            A[i, j] = 2

    sdfg = tester.to_sdfg()
    # Here N is implicitly used in the index expression, so it should be in the arguments
    assert 'N' in sdfg.arglist()


if __name__ == '__main__':
    test_global_sizes()
    test_global_sizes_used()
    test_global_sizes_multidim()
