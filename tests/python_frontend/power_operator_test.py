# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import numpy as np


@dace.program
def pow_num_literals(a: dace.int64[1]):
    a[0] = 2**3


def test_pow_num_literals():
    res = np.zeros((1, ), dtype=np.int64)
    pow_num_literals(a=res)
    assert (res[0] == 8)


@dace.program
def pow_op_preced(a: dace.int64[1]):
    a[0] = -1**2


def test_pow_op_preced():
    res = np.zeros((1, ), dtype=np.int64)
    pow_op_preced(a=res)
    assert (res[0] == -1)


@dace.program
def pow_neg_exp(a: dace.float64[1]):
    a[0] = 10**-2


def test_pow_neg_exp():
    res = np.zeros((1, ), dtype=np.float64)
    pow_neg_exp(a=res)
    assert (res[0] == 0.01)


in_types = [dace.float32, dace.float64, dace.int8, dace.int16, dace.int32, dace.int64]


@pytest.mark.parametrize("a_type", in_types)
@pytest.mark.parametrize("b_type", in_types)
def test_pow_types(a_type, b_type):

    @dace.program
    def pow_types(A: a_type[1], B: b_type[1], R: dace.float64[1]):
        with dace.tasklet(dace.Language.Python):
            scalar_a << A[0]
            scalar_b << B[0]
            scalar_r >> R[0]
            scalar_r = scalar_a**scalar_b

    # a ** b needs to fit into the smallest type (int8)
    a = np.random.rand(1) * 4
    b = np.random.rand(1) * 4
    r = np.random.rand(1).astype(np.float64)

    a = a.astype(a_type.as_numpy_dtype())
    b = b.astype(b_type.as_numpy_dtype())

    pow_types(A=a, B=b, R=r)
    assert np.allclose(r, a**b)


if __name__ == "__main__":
    test_pow_num_literals()
    test_pow_op_preced()
    test_pow_neg_exp()

    for a_type in in_types:
        for b_type in in_types:
            test_pow_types(a_type, b_type)
