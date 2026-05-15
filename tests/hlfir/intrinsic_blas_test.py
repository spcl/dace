"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_blas_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_dot(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: arg2
  double precision, dimension(2) :: res1
  res1(1) = dot_product(arg1, arg2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5

    sdfg(arg1=arg1, arg2=arg2, res1=res1)

    assert res1[0] == np.dot(arg1, arg2)


def test_fortran_frontend_dot_range(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: arg2
  double precision, dimension(2) :: res1
  res1(1) = dot_product(arg1(1:3), arg2(1:3))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5

    sdfg(arg1=arg1, arg2=arg2, res1=res1)
    assert res1[0] == np.dot(arg1[:3], arg2[:3])


def test_fortran_frontend_transpose(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5,4) :: arg1
  double precision, dimension(4,5) :: res1
  res1 = transpose(arg1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size_x = 5
    size_y = 4
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    res1 = np.full([size_y, size_x], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + 1

    sdfg(arg1=arg1, res1=res1)

    assert np.all(np.transpose(res1) == arg1)


def test_fortran_frontend_transpose_hoist_out(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5,4) :: arg1
  double precision, dimension(4,5) :: res1
  res1 = 1.0 - transpose(arg1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size_x = 5
    size_y = 4
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    res1 = np.full([size_y, size_x], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + 1

    sdfg(arg1=arg1, res1=res1)

    assert np.all((1.0 - np.transpose(res1)) == arg1)


def test_fortran_frontend_transpose_struct(tmp_path):
    src = """

MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        double precision, dimension(5,4) :: arg1
    END TYPE array_container
END MODULE

MODULE test_transpose

    contains

    subroutine test_function(arg1, res1)
        USE test_types
        IMPLICIT NONE
        TYPE(array_container) :: container
        double precision, dimension(5,4) :: arg1
        double precision, dimension(4,5) :: res1

        container%arg1 = arg1

        res1 = transpose(container%arg1)
    end subroutine test_function

END MODULE
"""
    sdfg = build_sdfg(src, tmp_path, name='test_function', entry='_QMtest_transposePtest_function').build()

    size_x = 5
    size_y = 4
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    res1 = np.full([size_y, size_x], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + 1

    sdfg(arg1=arg1, res1=res1)

    assert np.all(np.transpose(res1) == arg1)


def test_fortran_frontend_matmul(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5,3) :: arg1
  double precision, dimension(3,7) :: arg2
  double precision, dimension(5,7) :: res1
  res1 = matmul(arg1, arg2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size_x = 5
    size_y = 3
    size_z = 7
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    arg2 = np.full([size_y, size_z], 42, order="F", dtype=np.float64)
    res1 = np.full([size_x, size_z], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + j + 1
    for i in range(size_y):
        for j in range(size_z):
            arg2[i, j] = i + j + 7

    sdfg(arg1=arg1, arg2=arg2, res1=res1)

    assert np.all(np.matmul(arg1, arg2) == res1)


def test_fortran_frontend_matmul_hoist_out(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5,3) :: arg1
  double precision, dimension(3,7) :: arg2
  double precision, dimension(5,7) :: res1
  res1 = 2.0 - matmul(arg1, arg2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size_x = 5
    size_y = 3
    size_z = 7
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    arg2 = np.full([size_y, size_z], 42, order="F", dtype=np.float64)
    res1 = np.full([size_x, size_z], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + j + 1
    for i in range(size_y):
        for j in range(size_z):
            arg2[i, j] = i + j + 7

    sdfg(arg1=arg1, arg2=arg2, res1=res1)

    x = np.matmul(arg1, arg2)
    assert np.all([2.0 - val for val in x] == res1)
