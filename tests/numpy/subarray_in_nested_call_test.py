# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import math
import numpy as np
import dace

M = dace.symbol('M')
K = dace.symbol('K')


@dace.program
def sdfg_transpose(A: dace.float32[M, K], B: dace.float32[K, M]):
    for i, j in dace.map[0:M, 0:K]:
        B[j, i] = A[i, j]


@dace.program
def transpose_test(C: dace.float32[20, 20], D: dace.float32[20, 20]):
    sdfg_transpose(C[:], D[:])


def test():
    c = np.arange(400, dtype=np.float32).reshape((20, 20)).copy()
    d = np.zeros((20, 20), dtype=np.float32)

    transpose_test(c, d, K=20, M=20)

    assert np.allclose(c.transpose(), d)


@dace.program
def pb(a, i):
    a[i] = a[20 - i - 1]


@dace.program
def pa(a):
    for i in dace.map[0:5]:
        pb(a, i)


def test_inout_connector():
    a = np.arange(20, dtype=np.float64)
    ref = a.copy()
    pa(a)
    pa.f(ref)
    assert np.allclose(a, ref)


def test_indirect_symbolic_access():

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[3], c: dace.float64[19]):
        for i in dace.map[0:10]:
            xtmp: dace.float64 = 0

            local_offset_i = (i + 1) % 2

            if local_offset_i < 3 % 2:
                for kx in dace.unroll(range(math.ceil(3 / 2))):
                    ind_i = (i + 1 - kx * 2) // 2
                    kind_i = local_offset_i + kx * 2
                    if ind_i >= 0 and ind_i < 20:
                        xtmp += a[ind_i] * b[kind_i]

            c[i] = xtmp

    a = np.random.rand(20)
    b = np.random.rand(15)
    c = np.random.rand(10)
    refc = np.copy(c)
    tester.f(a, b, refc)
    tester(a, b, c)
    assert np.allclose(c, refc)


if __name__ == '__main__':
    test()
    test_inout_connector()
    test_indirect_symbolic_access()
