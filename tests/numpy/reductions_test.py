import dace
import numpy as np
from copy import deepcopy as dc


def test_sum_all():
    @dace.program
    def dace_sum(A: dace.float64[10, 5, 3]):
        return np.sum(A)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_sum(A.copy()) - np.sum(A.copy()))
    assert diff < 1e-5


def test_sum_1():
    @dace.program
    def dace_sum(A: dace.float64[10, 5, 3]):
        return np.sum(A, axis=1)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_sum(A.copy()) - np.sum(A.copy(), axis=1))
    assert diff < 1e-5


def test_max_all():
    @dace.program
    def dace_max(A: dace.float64[10, 5, 3]):
        return np.max(A)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_max(A.copy()) - np.max(A.copy()))
    assert diff < 1e-5


def test_max_1():
    @dace.program
    def dace_max(A: dace.float64[10, 5, 3]):
        return np.max(A, axis=1)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_max(A.copy()) - np.max(A.copy(), axis=1))
    assert diff < 1e-5

def test_argmax_1():
    @dace.program
    def dace_argmax(A: dace.float64[10, 5, 3]):
        return np.argmax(A, axis=1)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_argmax(A.copy()) - np.argmax(A.copy(), axis=1))
    assert diff < 1e-5

def test_argmin_1():
    @dace.program
    def dace_argmin(A: dace.float64[10, 5, 3]):
        return np.argmin(A, axis=1)

    A = np.random.rand(10, 5, 3)
    diff = np.linalg.norm(dace_argmin(A.copy()) - np.argmin(A.copy(), axis=1))
    assert diff < 1e-5


if __name__ == '__main__':
    test_sum_all()
    test_sum_1()
    test_max_all()
    test_max_1()
    test_argmax_1()
    test_argmin_1()
