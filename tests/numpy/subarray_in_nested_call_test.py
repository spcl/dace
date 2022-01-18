# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    c = np.random.rand(20, 20).astype(np.float32)
    d = np.zeros((20, 20), dtype=np.float32)

    transpose_test(c, d, K=20, M=20)

    assert np.allclose(c.transpose(), d)


@dace.program
def pb(a, i):
    a[i] = a[20 - i]


@dace.program
def pa(a):
    for i in dace.map[0:5]:
        pb(a, i)


def test_inout_connector():
    a = np.random.rand(20)
    ref = a.copy()
    pa(a)
    pa.f(ref)
    assert (np.allclose(a, ref))


if __name__ == '__main__':
    test()
    test_inout_connector()
