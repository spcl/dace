# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


if __name__ == "__main__":
    test_pow_num_literals()
    test_pow_op_preced()
    test_pow_neg_exp()
