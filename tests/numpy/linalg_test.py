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
    assert relative_error(val, ref) < 1e-12


@dace.program
def linalg_solve(A: dace.float64[100, 100], B: dace.float64[100, 10]):
    return np.linalg.solve(A, B)


def test_linalg_solve():
    A = generate_invertible_matrix(100, np.float64)
    B = np.random.randn(100, 10)
    ref = np.linalg.solve(A, B)
    val = linalg_solve(A, B)
    assert relative_error(val, ref) < 1e-12


@dace.program
def linalg_cholesky(A: dace.float64[100, 100]):
    return np.linalg.cholesky(A)


def test_linalg_cholesky():
    A = generate_positive_semidefinite_matrix(100, np.float64)
    ref = np.linalg.cholesky(A)
    val = linalg_cholesky(A)
    assert relative_error(val, ref) < 1e-12


if __name__ == "__main__":
    test_linalg_inv()
    test_linalg_solve()
    test_linalg_cholesky()
