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
def test_copy(A: dace.float32[M, N]):
    return A.copy()


@compare_numpy_output()
def test_astype(A: dace.int32[M, N]):
    return A.astype(np.float32)


@compare_numpy_output()
def test_fill(A: dace.int32[M, N]):
    A.fill(5)
    return A  # return A.fill(5) doesn't work because A is not copied


@compare_numpy_output()
def test_reshape(A: dace.float32[N, N]):
    return A.reshape([1, N*N])


@compare_numpy_output()
def test_transpose1(A: dace.float32[M, N]):
    return A.transpose()


@compare_numpy_output()
def test_transpose2(A: dace.float32[M, N, N, M]):
    return A.transpose((3, 0, 2, 1))


@compare_numpy_output()
def test_transpose3(A: dace.float32[M, N, N, M]):
    return A.transpose(3, 0, 2, 1)


if __name__ == "__main__":
    test_T()
    test_real()
    test_imag()
    test_copy()
    test_astype()
    test_fill()
    test_reshape()
    test_transpose1()
    test_transpose2()
    test_transpose3()
