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
    return A.reshape([1, N * N])


@compare_numpy_output()
def test_transpose1(A: dace.float32[M, N]):
    return A.transpose()


@compare_numpy_output()
def test_transpose2(A: dace.float32[M, N, N, M]):
    return A.transpose((3, 0, 2, 1))


@compare_numpy_output()
def test_transpose3(A: dace.float32[M, N, N, M]):
    return A.transpose(3, 0, 2, 1)


@compare_numpy_output()
def test_flatten(A: dace.float32[M, N, N, M]):
    return A.flatten()


@compare_numpy_output()
def test_ravel(A: dace.float32[M, N, N, M]):
    return A.ravel()


@compare_numpy_output()
def test_max(A: dace.float32[M, N, N, M]):
    return A.max()


# TODO: Need to debug `_argminmax`
# @compare_numpy_output()
# def test_argmax(A: dace.float32[M, N, N, M]):
#     return A.argmax()


@compare_numpy_output()
def test_min(A: dace.float32[M, N, N, M]):
    return A.min()


@compare_numpy_output()
def test_conj(A: dace.complex64[M, N, N, M]):
    return A.conj()


@compare_numpy_output()
def test_sum(A: dace.float32[M, N, N, M]):
    return A.sum()


@compare_numpy_output()
def test_mean(A: dace.float32[M, N, N, M]):
    return A.mean()


@compare_numpy_output()
def test_prod(A: dace.float32[M, N, N, M]):
    return A.prod()


@compare_numpy_output()
def test_all():
    A = np.full([5], True, dtype=np.bool_)
    return A.all()


@compare_numpy_output()
def test_any():
    A = np.full([5], False, dtype=np.bool_)
    A[4] = True
    return A.any()


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
    test_flatten()
    test_ravel()
    test_max()
    # test_argmax()
    test_min()
    test_conj()
    test_sum()
    test_mean()
    test_prod()
    test_all()
    test_any()
