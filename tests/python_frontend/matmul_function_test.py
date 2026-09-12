# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.matmul`` is the function spelling of ``@`` and must lower the same way.

Only the operator was registered, so ``np.matmul(a, b)`` raised "Function numpy.matmul is not
registered with an SDFG implementation" for code DaCe could already lower -- a refusal an author
cannot see coming from the ``@`` form working. PEP 465 defines the two as the same operation.
"""
import numpy as np

import dace

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
B = dace.symbol('B')


@dace.program
def matmul_2d(a: dace.float64[M, K], b: dace.float64[K, N], res: dace.float64[M, N]):
    res[:] = np.matmul(a, b)


@dace.program
def matmul_batched(a: dace.float64[B, M, K], b: dace.float64[B, K, N], res: dace.float64[B, M, N]):
    res[:] = np.matmul(a, b)


@dace.program
def matmul_matrix_vector(a: dace.float64[M, K], b: dace.float64[K], res: dace.float64[M]):
    res[:] = np.matmul(a, b)


def test_matmul_2d():
    a = np.random.rand(4, 5)
    b = np.random.rand(5, 3)
    res = np.zeros((4, 3))
    matmul_2d(a=a, b=b, res=res, M=4, K=5, N=3)
    assert np.allclose(res, a @ b)


def test_matmul_batched():
    """The batched case matters on its own: the operator picks a different output shape for it."""
    a = np.random.rand(2, 4, 5)
    b = np.random.rand(2, 5, 3)
    res = np.zeros((2, 4, 3))
    matmul_batched(a=a, b=b, res=res, B=2, M=4, K=5, N=3)
    assert np.allclose(res, a @ b)


def test_matmul_matrix_vector():
    a = np.random.rand(4, 5)
    b = np.random.rand(5)
    res = np.zeros((4, ))
    matmul_matrix_vector(a=a, b=b, res=res, M=4, K=5)
    assert np.allclose(res, a @ b)


def test_matmul_agrees_with_the_operator_it_delegates_to():
    """Two spellings, one implementation -- the property that keeps them from drifting apart."""

    @dace.program
    def with_operator(a: dace.float64[M, K], b: dace.float64[K, N], res: dace.float64[M, N]):
        res[:] = a @ b

    a, b = np.random.rand(4, 5), np.random.rand(5, 3)
    from_function, from_operator = np.zeros((4, 3)), np.zeros((4, 3))
    matmul_2d(a=a, b=b, res=from_function, M=4, K=5, N=3)
    with_operator(a=a, b=b, res=from_operator, M=4, K=5, N=3)
    assert np.allclose(from_function, from_operator)


if __name__ == '__main__':
    test_matmul_2d()
    test_matmul_batched()
    test_matmul_matrix_vector()
    test_matmul_agrees_with_the_operator_it_delegates_to()
