"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_merge_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_merge_1d(tmp_path):
    # Original f2dace test had ``mask`` declared INTEGER -- invalid
    # Fortran (MERGE's mask must be LOGICAL).  Declare it LOGICAL; the
    # caller still binds an int32 numpy buffer (LOGICAL(4) shares ABI
    # with INTEGER(4)).
    src = """
SUBROUTINE merge_test_function(input1, input2, mask, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
logical, dimension(7) :: mask
double precision, dimension(7) :: res

res = MERGE(input1, input2, mask)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], False, order="F", dtype=np.bool_)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)
    for val in res:
        assert val == 42

    for i in range(int(size / 2)):
        mask[i] = True
    sdfg(input1=first, input2=second, mask=mask, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 13
    for i in range(int(size / 2), size):
        assert res[i] == 42

    mask[:] = False
    for i in range(size):
        if i % 2 == 1:
            mask[i] = True
    sdfg(input1=first, input2=second, mask=mask, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 13
        else:
            assert res[i] == 42


def test_fortran_frontend_merge_comparison_scalar(tmp_path):
    src = """
SUBROUTINE merge_test_function(input1, input2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(7) :: res

res = MERGE(input1, input2, input1 .eq. 3)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, res=res)
    for val in res:
        assert val == 42

    for i in range(int(size / 2)):
        first[i] = 3
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 3
    for i in range(int(size / 2), size):
        assert res[i] == 42

    first[:] = 13
    for i in range(size):
        if i % 2 == 1:
            first[i] = 3
    sdfg(input1=first, input2=second, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 3
        else:
            assert res[i] == 42


def test_fortran_frontend_merge_comparison_arrays(tmp_path):
    src = """
SUBROUTINE merge_test_function(input1, input2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(7) :: res

res = MERGE(input1, input2, input1 .lt. input2)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, res=res)
    for val in res:
        assert val == 13

    for i in range(int(size / 2)):
        first[i] = 45
    sdfg(input1=first, input2=second, res=res)
    for i in range(int(size / 2)):
        assert res[i] == 42
    for i in range(int(size / 2), size):
        assert res[i] == 13

    first[:] = 13
    for i in range(size):
        if i % 2 == 1:
            first[i] = 45
    sdfg(input1=first, input2=second, res=res)
    for i in range(size):
        if i % 2 == 1:
            assert res[i] == 42
        else:
            assert res[i] == 13


def test_fortran_frontend_merge_comparison_arrays_offset(tmp_path):
    src = """
SUBROUTINE merge_test_function(input1, input2, mask1, mask2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(14) :: mask1
double precision, dimension(14) :: mask2
double precision, dimension(7) :: res

res = MERGE(input1, input2, mask1(3:9) .lt. mask2(5:11))

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask1 = np.full([size * 2], 30, order="F", dtype=np.float64)
    mask2 = np.full([size * 2], 0, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    mask1[2:9] = 3
    mask2[4:11] = 4
    sdfg(input1=first, input2=second, mask1=mask1, mask2=mask2, res=res)
    for val in res:
        assert val == 13


def test_fortran_frontend_merge_array_shift(tmp_path):
    src = """
SUBROUTINE merge_test_function(input1, input2, mask1, mask2, res)
double precision, dimension(7) :: input1
double precision, dimension(21) :: input2
double precision, dimension(14) :: mask1
double precision, dimension(14) :: mask2
double precision, dimension(7) :: res

res = MERGE(input1, input2(13:19), mask1(3:9) .gt. mask2(5:11))

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size * 3], 42, order="F", dtype=np.float64)
    mask1 = np.full([size * 2], 30, order="F", dtype=np.float64)
    mask2 = np.full([size * 2], 0, order="F", dtype=np.float64)
    res = np.full([size], 40, order="F", dtype=np.float64)

    second[12:19] = 100
    mask1[2:9] = 3
    mask2[4:11] = 4
    sdfg(input1=first, input2=second, mask1=mask1, mask2=mask2, res=res)
    for val in res:
        assert val == 100


def test_fortran_frontend_merge_nonarray(tmp_path):
    src = """
SUBROUTINE merge_test_function(val, res)
logical :: val(2)
double precision :: res(2)
double precision :: input1
double precision :: input2

input1 = 1
input2 = 5

res(1) = MERGE(input1, input2, val(1))

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()

    val = np.full([2], 1, order="F", dtype=np.int32)
    res = np.full([2], 40, order="F", dtype=np.float64)

    sdfg(val=val, res=res)
    assert res[0] == 1

    val[0] = 0
    sdfg(val=val, res=res)
    assert res[0] == 5


def test_fortran_frontend_merge_recursive(tmp_path):
    # Original f2dace test had ``mask1`` / ``mask2`` declared INTEGER --
    # invalid (MERGE's mask must be LOGICAL).  Declare LOGICAL.
    src = """
SUBROUTINE merge_test_function(input1, input2, input3, mask1, mask2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(7) :: input3
logical, dimension(7) :: mask1
logical, dimension(7) :: mask2
double precision, dimension(7) :: res

res = MERGE(MERGE(input1, input2, mask1), input3, mask2)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    third = np.full([size], 43, order="F", dtype=np.float64)
    mask1 = np.full([size], False, order="F", dtype=np.bool_)
    mask2 = np.full([size], True, order="F", dtype=np.bool_)
    res = np.full([size], 40, order="F", dtype=np.float64)

    for i in range(int(size / 2)):
        mask1[i] = True

    mask2[-1] = False

    sdfg(input1=first, input2=second, input3=third, mask1=mask1, mask2=mask2, res=res)

    assert np.allclose(res, [13, 13, 13, 42, 42, 42, 43])


def test_fortran_frontend_merge_scalar(tmp_path):
    # Original f2dace had ``mask`` INTEGER -- invalid; LOGICAL required.
    src = """
SUBROUTINE merge_test_function(input1, input2, mask, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
logical, dimension(7) :: mask
double precision, dimension(7) :: res

res(1) = MERGE(input1(1), input2(1), mask(1))

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)

    assert res[0] == 42
    for val in res[1:]:
        assert val == 40

    mask[0] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 13
    for val in res[1:]:
        assert val == 40


def test_fortran_frontend_merge_scalar2(tmp_path):
    # Original f2dace had ``mask`` INTEGER (must be LOGICAL) and the
    # false-source literal ``0.0`` defaulted to REAL(4) while the
    # true-source ``input1`` is REAL(8) -- MERGE requires both sources
    # to share kind.  Use ``0.0D0`` for the false source.
    src = """
SUBROUTINE merge_test_function(input1, input2, mask, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
logical, dimension(7) :: mask
double precision, dimension(7) :: res

res(1) = MERGE(input1(1), 0.0D0, mask(1))

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 0

    mask[:] = 1
    sdfg(input1=first, input2=second, mask=mask, res=res)
    assert res[0] == 13


def test_fortran_frontend_merge_scalar3(tmp_path):
    # The original f2dace test had ``mask`` / ``mask2`` INTEGER and the
    # false-source literal ``0.0`` was REAL(4) while the true source is
    # REAL(8) -- both invalid Fortran.  ``mask`` / ``mask2`` are kept
    # INTEGER because the comparison ``mask(1) > mask2(1)`` is the
    # actual mask (LOGICAL); the literal is now ``0.0D0``.
    src = """
SUBROUTINE merge_test_function(input1, input2, mask, mask2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
integer, dimension(7) :: mask
integer, dimension(7) :: mask2
double precision, dimension(7) :: res

res(1) = MERGE(input1(1), 0.0D0, mask(1) > mask2(1) .AND. mask2(2) == 0)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()
    size = 7

    first = np.full([size], 13, order="F", dtype=np.float64)
    second = np.full([size], 42, order="F", dtype=np.float64)
    mask = np.full([size], 0, order="F", dtype=np.int32)
    mask2 = np.full([size], 0, order="F", dtype=np.int32)
    res = np.full([size], 40, order="F", dtype=np.float64)

    sdfg(input1=first, input2=second, mask=mask, mask2=mask2, res=res)
    assert res[0] == 0

    mask[:] = 1
    sdfg(input1=first, input2=second, mask=mask, mask2=mask2, res=res)
    assert res[0] == 13


def test_fortran_frontend_merge_literal(tmp_path):
    src = """
SUBROUTINE merge_test_function(input1, input2, res)
double precision :: input1
double precision :: input2
double precision, dimension(1) :: res

res = MERGE(1.0D0, input2, input2 .lt. 3)

END SUBROUTINE merge_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test_function').build()

    res = np.full([1], 40, order="F", dtype=np.float64)

    sdfg(input1=np.array([13.0], dtype=np.float64), input2=np.array([42.0], dtype=np.float64), res=res)
    assert res[0] == 42

    sdfg(input1=np.array([13.0], dtype=np.float64), input2=np.array([10.0], dtype=np.float64), res=res)
    assert res[0] == 10

    sdfg(input1=np.array([13.0], dtype=np.float64), input2=np.array([2.0], dtype=np.float64), res=res)
    assert res[0] == 1


def test_fortran_frontend_merge_dataref(tmp_path):
    src = """
                    module lib
                      implicit none
                      type test_type
                          double precision, dimension(3) :: input_data
                          double precision, dimension(3) :: input_data_second
                      end type

                      type test_type2
                          type(test_type) :: var
                      end type
                    end module lib

                    MODULE test_merge

                        contains

                        SUBROUTINE merge_test(input1, input2, res)
                        use lib, only: test_type2
                        implicit none

                        type(test_type2) :: data
                        double precision, dimension(3) :: input1
                        double precision, dimension(3) :: input2
                        double precision, dimension(3) :: res

                        data%var%input_data = input1
                        data%var%input_data_second = input2

                        CALL merge_test_function(data, res)
                        end SUBROUTINE merge_test

                        SUBROUTINE merge_test_function(data, res)
                        use lib, only: test_type2
                        implicit none

                        double precision, dimension(3) :: res
                        type(test_type2) :: data

                        res = MERGE(data%var%input_data, data%var%input_data_second, data%var%input_data .lt. 3)
                        !res = MERGE(data%var%input_data, data%var%input_data_second, 1 .lt. 3)

                        END SUBROUTINE merge_test_function

                    END MODULE
"""
    sdfg = build_sdfg(src, tmp_path, name='merge_test', entry='_QMtest_mergePmerge_test').build()

    data1 = np.full([3], 42, order="F", dtype=np.float64)
    data2 = np.full([3], 40, order="F", dtype=np.float64)
    res = np.full([3], 0, order="F", dtype=np.float64)

    sdfg(input1=data1, input2=data2, res=res)
    assert res[0] == 40
