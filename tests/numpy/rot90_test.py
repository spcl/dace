# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from sympy.core.numbers import comp
import dace
from common import compare_numpy_output


@compare_numpy_output()
def test_rot90_2d_k0(A: dace.int32[10, 10]):
    return np.rot90(A, k=0)


@compare_numpy_output()
def test_rot90_2d_k1(A: dace.int32[10, 10]):
    return np.rot90(A)


@compare_numpy_output()
def test_rot90_2d_k2(A: dace.int32[10, 10]):
    return np.rot90(A, k=2)


@compare_numpy_output()
def test_rot90_2d_k3(A: dace.int32[10, 10]):
    return np.rot90(A, k=3)


if __name__ == '__main__':
    test_rot90_2d_k0()
    test_rot90_2d_k1()
    test_rot90_2d_k2()
    test_rot90_2d_k3()
