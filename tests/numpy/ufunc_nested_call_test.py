# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from common import compare_numpy_output


@compare_numpy_output(check_dtype=True)
def test_ufunc_nested_call(Z: dace.complex64[10]):
    return np.less(np.absolute(Z), 2.0)


@compare_numpy_output(check_dtype=True)
def test_ufunc_reduce_nested_call(Z: dace.complex64[10]):
    return np.add.reduce(np.absolute(Z))


@compare_numpy_output(check_dtype=True)
def test_ufunc_accumulate_nested_call(Z: dace.complex64[10, 10]):
    return np.add.accumulate(np.absolute(Z))


@compare_numpy_output(check_dtype=True)
def test_ufunc_outer_nested_call(A: dace.float32[10, 10]):
    return np.add.outer(np.divmod(A))


if __name__ == "__main__":
    test_ufunc_nested_call()
    test_ufunc_reduce_nested_call()
    test_ufunc_accumulate_nested_call()
    test_ufunc_outer_nested_call()
