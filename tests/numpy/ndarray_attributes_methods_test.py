# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


M = 10
N = 20


@compare_numpy_output()
def test_T(A: dace.float32[M, N]):
    return A.T


@compare_numpy_output()
def test_real(A: dace.complex64[M, N]):
    return A.real


@compare_numpy_output()
def test_imag(A: dace.complex64[M, N]):
    return A.imag


@compare_numpy_output()
def test_reshape(A: dace.float32[N, N]):
    return A.reshape([1, N*N])


if __name__ == "__main__":
    test_T()
    test_real()
    test_imag()
    test_reshape()