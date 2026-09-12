# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the numpy.triu and numpy.tril replacements (upper/lower triangular masks)."""
import numpy as np
import dace
from common import compare_numpy_output


@compare_numpy_output()
def test_triu_square_k0(A: dace.float64[8, 8]):
    return np.triu(A)


@compare_numpy_output()
def test_triu_square_kpos(A: dace.float64[8, 8]):
    return np.triu(A, k=2)


@compare_numpy_output()
def test_triu_square_kneg(A: dace.float64[8, 8]):
    return np.triu(A, k=-3)


@compare_numpy_output()
def test_triu_rectangular(A: dace.float64[5, 9]):
    return np.triu(A, k=1)


@compare_numpy_output()
def test_tril_square_k0(A: dace.float64[8, 8]):
    return np.tril(A)


@compare_numpy_output()
def test_tril_square_kpos(A: dace.float64[8, 8]):
    return np.tril(A, k=2)


@compare_numpy_output()
def test_tril_square_kneg(A: dace.float64[8, 8]):
    return np.tril(A, k=-3)


@compare_numpy_output()
def test_tril_rectangular(A: dace.float64[9, 5]):
    return np.tril(A, k=-1)


@compare_numpy_output()
def test_triu_int(A: dace.int32[6, 6]):
    return np.triu(A)


@compare_numpy_output()
def test_tril_3d(A: dace.float64[3, 5, 5]):
    # numpy applies the mask to the final two axes for ndim > 2.
    return np.tril(A, k=1)


if __name__ == '__main__':
    test_triu_square_k0()
    test_triu_square_kpos()
    test_triu_square_kneg()
    test_triu_rectangular()
    test_tril_square_k0()
    test_tril_square_kpos()
    test_tril_square_kneg()
    test_tril_rectangular()
    test_triu_int()
    test_tril_3d()
