# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


def generate_invertible_matrix(size, dtype):
    if dtype == np.float32:
        tol = 1e-6
    elif dtype == np.float64:
        tol = 1e-12
    else:
        raise NotImplementedError
    while True:
        A = np.random.randn(size, size).astype(dtype)
        B = A @ A.T
        err = np.absolute(B @ np.linalg.inv(B) - np.eye(size))
        if np.all(err < tol):
            break
    return A


def generate_positive_semidefinite_matrix(size, dtype):
    A = np.random.randn(size, size).astype(dtype)
    return (0.5 * A @ A.T).copy()


def relative_error(value, ref):
    return np.linalg.norm(value - ref) / np.linalg.norm(ref)


@dace.program
def linalg_inv(A: dace.float64[100, 100]):
    return np.linalg.inv(A)


def test_linalg_inv():
    A = generate_invertible_matrix(100, np.float64)
    ref = np.linalg.inv(A)
    val = linalg_inv(A)
    assert relative_error(val, ref) < 1e-10


@dace.program
def linalg_solve(A: dace.float64[100, 100], B: dace.float64[100, 10]):
    return np.linalg.solve(A, B)


def test_linalg_solve():
    A = generate_invertible_matrix(100, np.float64)
    B = np.random.randn(100, 10)
    ref = np.linalg.solve(A, B)
    val = linalg_solve(A, B)
    assert relative_error(val, ref) < 1e-10


@dace.program
def linalg_cholesky(A: dace.float64[100, 100]):
    return np.linalg.cholesky(A)


def test_linalg_cholesky():
    A = generate_positive_semidefinite_matrix(100, np.float64)
    ref = np.linalg.cholesky(A)
    val = linalg_cholesky(A)
    assert relative_error(val, ref) < 1e-10


def test_tensordot_0():

    @dace.program
    def tensordot_0(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B)
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='pure'):
        assert(np.allclose(tensordot_0(A.copy(), B.copy()), tensordot_0.f(A, B)))


def test_tensordot_01():

    @dace.program
    def tensordot_0(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B)
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='TTGT'):
        assert(np.allclose(tensordot_0(A.copy(), B.copy()), tensordot_0.f(A, B)))

@pytest.mark.gpu
def test_tensordot_02():

    @dace.program(device=dace.dtypes.DeviceType.GPU)
    def tensordot_0(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B)
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='cuTENSOR'):
        assert(np.allclose(tensordot_0(A.copy(), B.copy()), tensordot_0.f(A, B)))


def test_tensordot_1():

    @dace.program
    def tensordot_1(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]))
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='pure'):
        assert(np.allclose(tensordot_1(A.copy(), B.copy()), tensordot_1.f(A, B)))


def test_tensordot_11():

    @dace.program
    def tensordot_1(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]))
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='TTGT'):
        assert(np.allclose(tensordot_1(A.copy(), B.copy()), tensordot_1.f(A, B)))


@pytest.mark.gpu
def test_tensordot_12():

    @dace.program(device=dace.dtypes.DeviceType.GPU)
    def tensordot_1(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]))
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='cuTENSOR'):
        assert(np.allclose(tensordot_1(A.copy(), B.copy()), tensordot_1.f(A, B)))


def test_tensordot_2():

    @dace.program
    def tensordot_2a(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[7, 6, 5, 4, 3, 2, 1, 0])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[7, 6, 5, 4, 3, 2, 1, 0])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='pure'):
        assert(np.allclose(tensordot_2a(A.copy(), B.copy()), ref))
    
    @dace.program
    def tensordot_2b(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[0, 7, 1, 6, 2, 5, 3, 4])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[0, 7, 1, 6, 2, 5, 3, 4])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='pure'):
        assert(np.allclose(tensordot_2b(A.copy(), B.copy()), ref))


def test_tensordot_21():

    @dace.program
    def tensordot_2a(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[7, 6, 5, 4, 3, 2, 1, 0])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[7, 6, 5, 4, 3, 2, 1, 0])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='TTGT'):
        assert(np.allclose(tensordot_2a(A.copy(), B.copy()), ref))
    
    @dace.program
    def tensordot_2b(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[0, 7, 1, 6, 2, 5, 3, 4])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[0, 7, 1, 6, 2, 5, 3, 4])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='TTGT'):
        assert(np.allclose(tensordot_2b(A.copy(), B.copy()), ref))


def test_tensordot_22():

    @dace.program(device=dace.dtypes.DeviceType.GPU)
    def tensordot_2a(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[7, 6, 5, 4, 3, 2, 1, 0])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[7, 6, 5, 4, 3, 2, 1, 0])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='cuTENSOR'):
        assert(np.allclose(tensordot_2a(A.copy(), B.copy()), ref))
    
    @dace.program(device=dace.dtypes.DeviceType.GPU)
    def tensordot_2b(A: dace.float32[3, 3, 3, 3, 3, 3], B: dace.float32[3, 3, 3, 3, 3, 3]):
        return np.tensordot(A, B, axes=([0, 3], [4, 2]), out_axes=[0, 7, 1, 6, 2, 5, 3, 4])
    
    A = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    B = np.arange(3**6, dtype=np.float32).reshape(3, 3, 3, 3, 3, 3)
    ref = np.transpose(np.tensordot(A, B, axes=([0, 3], [4, 2])), axes=[0, 7, 1, 6, 2, 5, 3, 4])
    with dace.config.set_temporary('library', 'linalg', 'default_implementation', value='cuTENSOR'):
        assert(np.allclose(tensordot_2b(A.copy(), B.copy()), ref))


if __name__ == "__main__":
    # test_linalg_inv()
    # test_linalg_solve()
    # test_linalg_cholesky()
    # test_tensordot_0()
    # test_tensordot_1()
    # test_tensordot_01()
    # test_tensordot_11()
    # test_tensordot_2()
    # test_tensordot_21()
    test_tensordot_02()
    test_tensordot_12()
    test_tensordot_22()
