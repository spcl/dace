import dace
import numpy as np
from copy import deepcopy as dc
from common import compare_numpy_output


@compare_numpy_output
def test_sum(A: dace.float64[10, 5, 3]):
    return np.sum(A)


@compare_numpy_output
def test_sum_1(A: dace.float64[10, 5, 3]):
    return np.sum(A, axis=1)


@compare_numpy_output
def test_min(A: dace.float64[10, 5, 3]):
    return np.min(A)


@compare_numpy_output
def test_max(A: dace.float64[10, 5, 3]):
    return np.max(A)


@compare_numpy_output
def test_min_1(A: dace.float64[10, 5, 3]):
    return np.min(A, axis=1)


@compare_numpy_output
def test_max_1(A: dace.float64[10, 5, 3]):
    return np.max(A, axis=1)


@compare_numpy_output
def test_argmax_1(A: dace.float64[10, 5, 3]):
    return np.argmax(A, axis=1)


@compare_numpy_output
def test_argmin_1(A: dace.float64[10, 5, 3]):
    return np.argmin(A, axis=1)


def test_argmin_result_type():
    @dace.program
    def test_argmin_result(A: dace.float64[10, 5, 3]):
        return np.argmin(A, axis=1, result_type=dace.int64)

    res = test_argmin_result(np.random.rand(10, 5, 3))
    assert res.dtype == np.int64


if __name__ == '__main__':
    test_sum()
    test_sum_1()
    test_max()
    test_max_1()
    test_min()
    test_min_1()
    test_argmax_1()
    test_argmin_1()
    test_argmin_result_type()

    # Test supported reduction with OpenMP library node implementation
    from dace.libraries.standard import Reduce
    Reduce.default_implementation = 'OpenMP'
    test_sum()
    test_sum_1()
    test_max()
    test_max_1()
    test_min()
    test_min_1()
