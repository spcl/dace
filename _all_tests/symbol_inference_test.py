# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def symbol_inference(A: dace.float64[N, N], B: dace.float64[M + 1, M * 2]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a >> A[i, j]
            a = N

    for i, j in dace.map[0:M + 1, 0:M * 2]:
        with dace.tasklet:
            b >> B[i, j]
            b = M


@dace.program
def symbol_inference_joint(A: dace.float64[N + M], B: dace.float64[N + 2 * M]):
    for i in dace.map[0:N + M]:
        with dace.tasklet:
            a >> A[i]
            a = N

    for i in dace.map[0:N + 2 * M]:
        with dace.tasklet:
            b >> B[i]
            b = M


def test_symbol_inference():
    real_N = 5
    real_M = 7
    A = np.random.rand(real_N, real_N)
    B = np.random.rand(real_M + 1, real_M * 2)
    symbol_inference(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


def test_symbol_inference_joint():
    real_N = 3
    real_M = 2
    A = np.random.rand(real_N + real_M)
    B = np.random.rand(real_N + real_M * 2)
    symbol_inference_joint(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


if __name__ == '__main__':
    test_symbol_inference()
    test_symbol_inference_joint()
