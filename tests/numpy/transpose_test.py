# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import pytest
from common import compare_numpy_output

M, N = 24, 24


@dace.program
def transpose_test(A: dace.float32[M, N], B: dace.float32[M, N]):
    B[:] = np.transpose(A)


@compare_numpy_output()
def test_transpose_axes0(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 2, 0])


@compare_numpy_output()
def test_transpose_axes1(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 0, 2])


@compare_numpy_output()
def test_transpose_axes2(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 0, 2])

@compare_numpy_output()
def test_transpose_none(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A)

@compare_numpy_output()
def test_transpose_no(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[0, 1, 2, 3])


def test_transpose():
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros([M, N], dtype=np.float32)
    transpose_test(A, B)

    realB = np.transpose(A)
    rel_error = np.linalg.norm(B - realB) / np.linalg.norm(realB)
    print('Relative_error:', rel_error)
    assert rel_error <= 1e-5


@pytest.mark.hptt
def test_hptt():
    with dace.config.set_temporary('library', 'ttranspose', 'default_implementation', value='HPTT'):
        test_transpose_axes0()
        test_transpose_axes1()
        test_transpose_axes2()
        test_transpose_none()


if __name__ == '__main__':
    test_transpose_axes0()
    test_transpose_axes1()
    test_transpose_axes2()
    test_transpose()
    test_transpose_none()
    test_transpose_no()
    test_hptt()
