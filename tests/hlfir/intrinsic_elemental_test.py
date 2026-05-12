"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_elemental_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_elemental_exp(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1 = exp(arg1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    py_res = np.exp(arg1)
    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_pardecl(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1 = exp(arg1(:))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    py_res = np.exp(arg1)
    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_subset(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = exp(arg1(2:4))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    assert res[0] == 0
    assert res[4] == 0
    py_res = np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_struct(tmp_path):
    src = """

MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        double precision, DIMENSION(5) :: data
    END TYPE array_container
END MODULE

MODULE test_elemental
    USE test_types
    IMPLICIT NONE

    CONTAINS

    subroutine test_func(arg1, res1)
    TYPE(array_container) :: container
    double precision, dimension(5) :: arg1
    double precision, dimension(5) :: res1

    container%data = arg1

    res1(2:4) = exp(container%data(2:4))

    end subroutine test_func

END MODULE

"""
    sdfg = build_sdfg(src, tmp_path, name='test_func', entry='_QMtest_elementalPtest_func').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    assert res[0] == 0
    assert res[4] == 0
    py_res = np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_subset_hoist(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = 1.0 - exp(arg1(2:4))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    assert res[0] == 0
    assert res[4] == 0
    py_res = 1.0 - np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_complex(tmp_path):
    src = """
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = arg1(2:4) - exp(arg1(2:4))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    assert res[0] == 0
    assert res[4] == 0
    py_res = arg1[1:4] - np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_ecrad_bug(tmp_path):
    src = """
subroutine main(ng_var_114, od_var_115, trans_dir_dir_var_119)
  INTEGER, INTENT(IN) :: ng_var_114
  REAL(KIND = 8), INTENT(IN), DIMENSION(ng_var_114) :: od_var_115
  REAL(KIND = 8), INTENT(OUT), DIMENSION(ng_var_114) :: trans_dir_dir_var_119
  REAL(KIND = 8) :: mu0

  mu0 = 3.14

  trans_dir_dir_var_119 = MAX(- MAX(od_var_115 * (1.0D0 / mu0), 0.0D0), - 1000.0D0)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 5

    arg_in = np.full([size], 42, order="F", dtype=np.float64)
    arg_out = np.full([size], 0, order="F", dtype=np.float64)

    mu0 = 3.14

    for i in range(size):
        arg_in[i] = i + 1

    sdfg(ng_var_114=size, od_var_115=arg_in, trans_dir_dir_var_119=arg_out)

    assert np.allclose(arg_out, np.maximum(-np.maximum(arg_in * 1.0 / mu0, 0.0), -1000.0))
