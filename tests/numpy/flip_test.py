# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from sympy.core.numbers import comp
import dace
from common import compare_numpy_output


@compare_numpy_output()
def test_flip_1d(A: dace.int32[10]):
    return np.flip(A)


@compare_numpy_output()
def test_flip_2d(A: dace.int32[10, 5]):
    return np.flip(A)


@compare_numpy_output()
def test_flip_2d_axis0(A: dace.int32[10, 5]):
    return np.flip(A, axis=(0, ))


@compare_numpy_output()
def test_flip_2d_axis0n(A: dace.int32[10, 5]):
    return np.flip(A, axis=(-2, ))


@compare_numpy_output()
def test_flip_2d_axis1(A: dace.int32[10, 5]):
    return np.flip(A, axis=(1, ))


@compare_numpy_output()
def test_flip_2d_axis1n(A: dace.int32[10, 5]):
    return np.flip(A, axis=(-1, ))


@compare_numpy_output()
def test_flip_3d(A: dace.int32[10, 5, 7]):
    return np.flip(A)


@compare_numpy_output()
def test_flip_3d_axis01(A: dace.int32[10, 5, 7]):
    return np.flip(A, axis=(0, 1))


@compare_numpy_output()
def test_flip_3d_axis02(A: dace.int32[10, 5, 7]):
    return np.flip(A, axis=(0, 2))


@compare_numpy_output()
def test_flip_3d_axis12(A: dace.int32[10, 5, 7]):
    return np.flip(A, axis=(1, 2))


if __name__ == '__main__':
    test_flip_1d()
    test_flip_2d()
    test_flip_2d_axis0()
    test_flip_2d_axis0n()
    test_flip_2d_axis1()
    test_flip_2d_axis1n()
    test_flip_3d()
    test_flip_3d_axis01()
    test_flip_3d_axis02()
    test_flip_3d_axis12()
