# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from copy import deepcopy as dc
from common import compare_numpy_output


@compare_numpy_output()
def test_sum(A: dace.float64[10, 5, 3]):
    return np.sum(A)


@compare_numpy_output()
def test_sum_1(A: dace.float64[10, 5, 3]):
    return np.sum(A, axis=1)


@compare_numpy_output()
def test_min(A: dace.float64[10, 5, 3]):
    return np.min(A)


@compare_numpy_output()
def test_max(A: dace.float64[10, 5, 3]):
    return np.max(A)


@compare_numpy_output()
def test_min_1(A: dace.float64[10, 5, 3]):
    return np.min(A, axis=1)


@compare_numpy_output()
def test_min_int32(A: dace.int32[10, 5, 3]):
    return np.min(A, axis=1)


@compare_numpy_output()
def test_min_int64(A: dace.int64[10, 5, 3]):
    return np.min(A, axis=1)


@compare_numpy_output()
def test_max_int32(A: dace.int32[10, 5, 3]):
    return np.max(A, axis=1)


@compare_numpy_output()
def test_max_int64(A: dace.int64[10, 5, 3]):
    return np.max(A, axis=1)


@compare_numpy_output()
def test_max_1(A: dace.float64[10, 5, 3]):
    return np.max(A, axis=1)


@compare_numpy_output()
def test_argmax_1(A: dace.float64[10, 5, 3]):
    return np.argmax(A, axis=1)


@compare_numpy_output()
def test_argmin_1(A: dace.float64[10, 5, 3]):
    return np.argmin(A, axis=1)


@compare_numpy_output()
def test_argmin_1_int32(A: dace.int32[10, 5, 3]):
    return np.argmin(A, axis=1)


@compare_numpy_output()
def test_argmin_1_int64(A: dace.int64[10, 5, 3]):
    return np.argmin(A, axis=1)


@compare_numpy_output()
def test_argmax_1_int32(A: dace.int32[10, 5, 3]):
    return np.argmax(A, axis=1)


@compare_numpy_output()
def test_argmax_1_int64(A: dace.int64[10, 5, 3]):
    return np.argmax(A, axis=1)


def test_return_both():
    from dace.frontend.python.replacements import _argminmax

    sdfg = dace.SDFG("test_return_both")
    state = sdfg.add_state()

    sdfg.add_array("IN", [10, 5, 3], dace.float64)

    _, (outval, outidx) = _argminmax(None, sdfg, state, "IN", 1, "min", return_both=True)

    IN = np.random.rand(10, 5, 3)
    OUT_IDX = np.zeros((10, 3), dtype=np.int32)
    OUT_VAL = np.zeros((10, 3), dtype=np.float64)

    sdfg.arrays[outval].transient = False
    sdfg.arrays[outidx].transient = False

    sdfg(**{"IN": IN.copy(), outval: OUT_VAL, outidx: OUT_IDX})

    np.allclose(OUT_IDX, np.argmin(IN.copy(), axis=1))
    np.allclose(OUT_VAL, np.min(IN.copy(), axis=1))


def test_argmin_result_type():
    @dace.program
    def test_argmin_result(A: dace.float64[10, 5, 3]):
        return np.argmin(A, axis=1, result_type=dace.int64)

    res = test_argmin_result(np.random.rand(10, 5, 3))
    assert res.dtype == np.int64

    @dace.program
    def test_argmin_result(A: dace.float64[10, 5, 3]):
        return np.argmin(A, axis=1)

    res = test_argmin_result(np.random.rand(10, 5, 3))
    assert res.dtype == np.int32


@compare_numpy_output()
def test_sum_negative_axis(A: dace.float64[10, 5, 3]):
    return np.sum(A, axis=-1)


@compare_numpy_output()
def test_sum_multiple_axes(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=(-1, 0))


@compare_numpy_output()
def test_mean(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=2)


@compare_numpy_output()
def test_mean_negative(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=-2)


@compare_numpy_output()
def test_mean_multiple_axes(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=(-2, 0))


def test_mean_reduce_symbolic_shape():
    N = dace.symbol('N')

    @dace.program
    def mean_reduce_symbolic_shape(A: dace.float64[10, N, 3]):
        return np.mean(A, axis=(-2, 0))

    X = np.random.normal(scale=10, size=(10, 12, 3)).astype(np.float64)

    dace_result = mean_reduce_symbolic_shape(A=X)
    numpy_result = np.mean(X, axis=(-2, 0))

    assert np.allclose(dace_result, numpy_result)


@compare_numpy_output()
def test_reduce_all_axes(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=(0, -2, 2))


# test accessing a global variable
my_none = None


@compare_numpy_output()
def test_reduce_global_None(A: dace.float64[10, 5, 3]):
    return np.mean(A, axis=my_none)


if __name__ == '__main__':

    # generated with cat tests/numpy/reductions_test.py | grep -oP '(?<=^def ).*(?=\()' | awk '{print $0 "()"}'
    test_sum()
    test_sum_1()
    test_min()
    test_max()
    test_min_1()
    test_min_int32()
    test_min_int64()
    test_max_int32()
    test_max_int64()
    test_max_1()
    test_argmax_1()
    test_argmin_1()
    test_argmin_1_int32()
    test_argmin_1_int64()
    test_argmax_1_int32()
    test_argmax_1_int64()
    test_return_both()
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

    test_sum_negative_axis()
    test_sum_multiple_axes()

    test_mean()
    test_mean_negative()
    test_mean_multiple_axes()

    test_mean_reduce_symbolic_shape()
