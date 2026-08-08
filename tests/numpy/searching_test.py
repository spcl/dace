# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

N = 100


def test_numpy_where():

    @dace.program
    def numpy_where(A: dace.float64[N]):
        return np.where(A > 0.5, A, 0.0)

    for _ in range(10):
        A = np.random.randn(N)
        assert (np.allclose(numpy_where(A), np.where(A > 0.5, A, 0.0)))


def test_numpy_select():

    @dace.program
    def numpy_where(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        return np.select([A > 0.5, B > 0.5, C > 0.5], [A, B, C], 0.0)

    for _ in range(10):
        A = np.random.randn(N)
        B = np.random.randn(N)
        C = np.random.randn(N)
        assert (np.allclose(numpy_where(A, B, C), np.select([A > 0.5, B > 0.5, C > 0.5], [A, B, C], 0.0)))


def test_numpy_where_scalar_operands():
    """ Both x and y are constants: the result is broadcast to the shape of the condition. """

    @dace.program
    def numpy_where_scalars(A: dace.float64[N]):
        return np.where(A > 0.5, 0.0, 1.0)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 0.0, 1.0)
    res = numpy_where_scalars(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_int():
    """ Two integer constants must not promote the result to floating point. """

    @dace.program
    def numpy_where_scalars_int(A: dace.float64[N]):
        return np.where(A > 0.5, 1, 2)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 1, 2)
    res = numpy_where_scalars_int(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_mixed():
    """ Mixed integer/floating-point constants follow NumPy's type promotion. """

    @dace.program
    def numpy_where_scalars_mixed(A: dace.float64[N]):
        return np.where(A > 0.5, 0.0, 1)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 0.0, 1)
    res = numpy_where_scalars_mixed(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_2d():

    @dace.program
    def numpy_where_scalars_2d(A: dace.float64[N, 5]):
        return np.where(A > 0.5, -1.0, 2.5)

    A = np.random.randn(N, 5)
    ref = np.where(A > 0.5, -1.0, 2.5)
    res = numpy_where_scalars_2d(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_scalar_condition():
    """ Scalar condition and scalar operands would give a 0-dimensional result, which DaCe cannot represent. """

    @dace.program
    def numpy_where_scalar_cond(a: dace.float64):
        return np.where(a > 0.5, 0.0, 1.0)

    with pytest.raises(ValueError, match='0-dimensional'):
        numpy_where_scalar_cond(1.0)


if __name__ == "__main__":
    test_numpy_where()
    test_numpy_select()
    test_numpy_where_scalar_operands()
    test_numpy_where_scalar_operands_int()
    test_numpy_where_scalar_operands_mixed()
    test_numpy_where_scalar_operands_2d()
    test_numpy_where_scalar_operands_scalar_condition()
