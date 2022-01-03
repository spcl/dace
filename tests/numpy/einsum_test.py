# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')


def test_general_einsum():
    @dace.program
    def einsumtest(A: dace.float64[M, N], B: dace.float64[N, M], C: dace.float64[M]):
        return np.einsum('ij,ji,i->', A, B, C)

    A = np.random.rand(10, 20)
    B = np.random.rand(20, 10)
    C = np.random.rand(10)
    out = einsumtest(A, B, C)
    assert np.allclose(out, np.einsum('ij,ji,i->', A, B, C))


def test_matmul():
    @dace.program
    def einsumtest(A: dace.float64[M, N], B: dace.float64[N, M]):
        return np.einsum('ik,kj', A, B)

    A = np.random.rand(10, 20)
    B = np.random.rand(20, 10)
    assert np.allclose(einsumtest(A, B), A @ B)


def test_batch_matmul():
    @dace.program
    def einsumtest(A: dace.float64[4, M, N], B: dace.float64[4, N, M]):
        return np.einsum('bik,bkj->bij', A, B)

    A = np.random.rand(4, 10, 20)
    B = np.random.rand(4, 20, 10)
    assert np.allclose(einsumtest(A, B), A @ B)


def test_opteinsum_sym():
    @dace.program
    def einsumtest(A: dace.float64[N, N, N, N], B: dace.float64[N, N, N, N], C: dace.float64[N, N, N, N],
                   D: dace.float64[N, N, N, N], E: dace.float64[N, N, N, N]):
        return np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E, optimize=True)

    A, B, C, D, E = tuple(np.random.rand(10, 10, 10, 10) for _ in range(5))
    try:
        einsumtest(A, B, C, D, E)
        raise AssertionError('Exception should have been raised')
    except ValueError:
        print('Exception successfully caught')


def test_opteinsum():
    N = 10

    @dace.program
    def einsumtest(A: dace.float64[N, N, N, N], B: dace.float64[N, N, N, N], C: dace.float64[N, N, N, N],
                   D: dace.float64[N, N, N, N], E: dace.float64[N, N, N, N]):
        return np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E, optimize=True)

    A, B, C, D, E = tuple(np.random.rand(10, 10, 10, 10) for _ in range(5))

    assert np.allclose(einsumtest(A, B, C, D, E), np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E))


if __name__ == '__main__':
    test_general_einsum()
    test_matmul()
    test_batch_matmul()
    test_opteinsum_sym()
    test_opteinsum()
