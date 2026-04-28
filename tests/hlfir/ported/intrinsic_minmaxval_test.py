"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_minmaxval_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_minval_double(tmp_path):
    src = """
SUBROUTINE minval_test_function(d, res)
double precision, dimension(7) :: d
double precision, dimension(0) :: dt
double precision, dimension(4) :: res

res(1) = MINVAL(d)
res(2) = MINVAL(d(:))
res(3) = MINVAL(d(3:6))
res(4) = MINVAL(dt)

END SUBROUTINE minval_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='minval_test_function').build()
    size = 7

    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    assert res[3] == np.finfo(np.float64).max

    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    assert res[3] == np.finfo(np.float64).max


def test_fortran_frontend_minval_int(tmp_path):
    src = """
SUBROUTINE minval_test_function(d, res)
integer, dimension(7) :: d
integer, dimension(0) :: dt
integer, dimension(4) :: res

res(1) = MINVAL(d)
res(2) = MINVAL(d(:))
res(3) = MINVAL(d(3:6))
res(4) = MINVAL(dt)

END SUBROUTINE minval_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='minval_test_function').build()
    size = 7

    d = np.full([size], 0, order="F", dtype=np.int32)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    assert res[3] == np.iinfo(np.int32).max

    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    assert res[3] == np.iinfo(np.int32).max

    d = np.full([size], 0, order="F", dtype=np.int32)
    d[:] = [-5, 10, -6, 4, 32, 42, -1]
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[2]
    assert res[1] == d[2]
    assert res[2] == d[2]
    assert res[3] == np.iinfo(np.int32).max


def test_fortran_frontend_maxval_double(tmp_path):
    src = """
SUBROUTINE minval_test_function(d, res)
double precision, dimension(7) :: d
double precision, dimension(0) :: dt
double precision, dimension(4) :: res

res(1) = MAXVAL(d)
res(2) = MAXVAL(d(:))
res(3) = MAXVAL(d(3:6))
res(4) = MAXVAL(dt)

END SUBROUTINE minval_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='minval_test_function').build()
    size = 7

    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    assert res[3] == np.finfo(np.float64).min

    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    assert res[3] == np.finfo(np.float64).min


def test_fortran_frontend_maxval_int(tmp_path):
    src = """
SUBROUTINE minval_test_function(d, res)
integer, dimension(7) :: d
integer, dimension(0) :: dt
integer, dimension(4) :: res

res(1) = MAXVAL(d)
res(2) = MAXVAL(d(:))
res(3) = MAXVAL(d(3:6))
res(4) = MAXVAL(dt)

END SUBROUTINE minval_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='minval_test_function').build()
    size = 7

    d = np.full([size], 0, order="F", dtype=np.int32)
    for i in range(size):
        d[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[-1]
    assert res[1] == d[-1]
    assert res[2] == d[5]
    assert res[3] == np.iinfo(np.int32).min

    for i in range(size):
        d[i] = 10 - i
    sdfg(d=d, res=res)
    assert res[0] == d[0]
    assert res[1] == d[0]
    assert res[2] == d[2]
    assert res[3] == np.iinfo(np.int32).min

    d = np.full([size], 0, order="F", dtype=np.int32)
    d[:] = [41, 10, 42, -5, 32, 41, 40]
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == d[2]
    assert res[1] == d[2]
    assert res[2] == d[2]
    assert res[3] == np.iinfo(np.int32).min


def test_fortran_frontend_minval_struct(tmp_path):
    src = """
MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        INTEGER, DIMENSION(7) :: data
    END TYPE array_container
END MODULE

MODULE test_minval
    USE test_types
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE minval_test_func(inp, res)
        TYPE(array_container) :: container
        INTEGER, DIMENSION(7) :: inp
        INTEGER, DIMENSION(4) :: res

        container%data = inp

        CALL minval_test_func_internal(container, res)
    END SUBROUTINE

    SUBROUTINE minval_test_func_internal(container, res)
        TYPE(array_container), INTENT(IN) :: container
        INTEGER, DIMENSION(4) :: res

        res(1) = MAXVAL(container%data)
        res(2) = MAXVAL(container%data(:))
        res(3) = MAXVAL(container%data(3:6))
        res(4) = MAXVAL(container%data(2:5))
    END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path, name='minval_test_func', entry='_QMtest_minvalPminval_test_func').build()

    size = 7
    inp = np.full([size], 0, order="F", dtype=np.int32)
    for i in range(size):
        inp[i] = i + 1
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(inp=inp, res=res)

    assert res[0] == inp[6]
    assert res[1] == inp[6]
    assert res[2] == inp[5]
    assert res[3] == inp[4]
