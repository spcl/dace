# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


@compare_numpy_output(check_dtype=True)
def test_ufunc_add_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.add(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_subtract_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.subtract(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_subtract_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.subtract(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_multiply_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.multiply(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divide_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divide_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.divide(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logaddexp_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logaddexp(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logaddexp2_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logaddexp2(A, B)


if __name__ == "__main__":
    # test_ufunc_add_ff()
    # test_ufunc_subtract_ff()
    # test_ufunc_subtract_uu()
    # test_ufunc_multiply_ff()
    # test_ufunc_divide_ff()
    # test_ufunc_divide_uu()
    # test_ufunc_logaddexp_ff()
    test_ufunc_logaddexp2_ff()
