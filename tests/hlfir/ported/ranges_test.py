"""Verbatim port of f2dace/dev:tests/fortran/ranges_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from ported._helpers import xfail

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_multiple_ranges_all(tmp_path):
    src = """
SUBROUTINE multiple_ranges_function(input1, input2, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(7) :: res

res(:) = input1(:) - input2(:)

END SUBROUTINE multiple_ranges_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_function').build()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    input2 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 1
        input2[i] = i
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, input2=input2, res=res)
    for val in res:
        assert val == 1.0


def test_fortran_frontend_multiple_ranges_selection(tmp_path):
    src = """
SUBROUTINE multiple_ranges_selection_function(input1, res)
double precision, dimension(7,2) :: input1
double precision, dimension(7) :: res

res(:) = input1(:, 1) - input1(:, 2)

END SUBROUTINE multiple_ranges_selection_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_selection_function').build()

    size = 7
    size2 = 2
    input1 = np.full([size, size2], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i][0] = i + 1
        input1[i][1] = 0
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res)
    for idx, val in enumerate(res):
        assert val == idx + 1.0


def test_fortran_frontend_multiple_ranges_selection_var(tmp_path):
    src = """
SUBROUTINE multiple_ranges_selection_function(input1, res, pos1, pos2)
double precision, dimension(7,2) :: input1
double precision, dimension(7) :: res
integer :: pos1
integer :: pos2

res(:) = input1(:, pos1) - input1(:, pos2)

END SUBROUTINE multiple_ranges_selection_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_selection_function').build()

    size = 7
    size2 = 2
    input1 = np.full([size, size2], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i][1] = i + 1
        input1[i][0] = 0
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res, pos1=2, pos2=1)
    for idx, val in enumerate(res):
        assert val == idx + 1.0

    sdfg(input1=input1, res=res, pos1=1, pos2=2)
    for idx, val in enumerate(res):
        assert -val == idx + 1.0


def test_fortran_frontend_multiple_ranges_subset(tmp_path):
    src = """
SUBROUTINE multiple_ranges_subset_function(input1, res)
double precision, dimension(7) :: input1
double precision, dimension(3) :: res

res(:) = input1(1:3) - input1(4:6)

END SUBROUTINE multiple_ranges_subset_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_subset_function').build()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res)
    for idx, val in enumerate(res):
        assert val == -3.0


def test_fortran_frontend_multiple_ranges_subset_var(tmp_path):
    src = """
SUBROUTINE multiple_ranges_subset_var_function(input1, res, pos)
double precision, dimension(9) :: input1
double precision, dimension(3) :: res
integer, dimension(4) :: pos

res(:) = input1(pos(1):pos(2)) - input1(pos(3):pos(4))

END SUBROUTINE multiple_ranges_subset_var_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_subset_var_function').build()

    size = 9
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = 2**i

    pos = np.full([4], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 4
    pos[2] = 6
    pos[3] = 8

    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res)

    for i in range(len(res)):
        assert res[i] == input1[pos[0] - 1 + i] - input1[pos[2] - 1 + i]


def test_fortran_frontend_multiple_ranges_ecrad_pattern(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_function(input1, res, pos)
double precision, dimension(7, 7) :: input1
double precision, dimension(7, 7) :: res
integer, dimension(2) :: pos

res(:, pos(1):pos(2)) = input1(:, pos(1):pos(2))

END SUBROUTINE multiple_ranges_ecrad_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_function').build()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2**j

    pos = np.full([2], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res)

    for i in range(size):
        for j in range(pos[0], pos[1] + 1):
            assert res[i - 1, j - 1] == input1[i - 1, j - 1]


@xfail('res(:, pos(1):pos(2)) = a(:, pos(3):pos(4)) + a(:, pos(5):pos(6)) — invalid expression')
def test_fortran_frontend_multiple_ranges_ecrad_pattern_complex(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_function(input1, res, pos)
double precision, dimension(7, 7) :: input1
double precision, dimension(7, 7) :: res
integer, dimension(6) :: pos

res(:, pos(1):pos(2)) = input1(:, pos(3):pos(4)) + input1(:, pos(5):pos(6))

END SUBROUTINE multiple_ranges_ecrad_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_function').build()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2**j

    pos = np.full([6], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5
    pos[2] = 1
    pos[3] = 4
    pos[4] = 4
    pos[5] = 7

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res)

    iter_1 = pos[0]
    iter_2 = pos[2]
    iter_3 = pos[4]
    length = pos[1] - pos[0] + 1

    for i in range(size):
        for j in range(length):
            assert res[i - 1, iter_1 + j - 1] == input1[i - 1, iter_2 + j - 1] + input1[i - 1, iter_3 + j - 1]


@xfail("dimension(7,21:27) offset declarations not yet honoured by FaCe")
def test_fortran_frontend_multiple_ranges_ecrad_pattern_complex_offsets(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_offset_function(input1, res, pos)
double precision, dimension(7, 21:27) :: input1
double precision, dimension(7, 31:37) :: res
integer, dimension(6) :: pos

res(:, pos(1):pos(2)) = input1(:, pos(3):pos(4)) + input1(:, pos(5):pos(6))

END SUBROUTINE multiple_ranges_ecrad_offset_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_offset_function').build()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2**j

    pos = np.full([6], 0, order="F", dtype=np.int32)
    pos[0] = 2 + 30
    pos[1] = 5 + 30
    pos[2] = 1 + 20
    pos[3] = 4 + 20
    pos[4] = 4 + 20
    pos[5] = 7 + 20

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res)

    iter_1 = pos[0] - 30
    iter_2 = pos[2] - 20
    iter_3 = pos[4] - 20
    length = pos[1] - pos[0] + 1

    for i in range(size):
        for j in range(length):
            assert res[i - 1, iter_1 + j - 1] == input1[i - 1, iter_2 + j - 1] + input1[i - 1, iter_3 + j - 1]


@xfail('res(:, pos(1)+k) = … broadcast assignments — Memlet subset mismatch')
def test_fortran_frontend_array_assignment(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_function(input1, input2, res, pos)
double precision, dimension(7) :: input1
double precision, dimension(7) :: input2
double precision, dimension(7, 7) :: res
integer, dimension(2) :: pos
integer :: nlev

nlev = input1(1)

! write 5 to column 2
res(:, pos(1)) = nlev

! write input1 values to column 3
res(:, pos(1) + 1) = input1

res(:, pos(1) + 2) = input1 + input2

res(:, pos(1) + 3) = input1 + input2(:)

END SUBROUTINE multiple_ranges_ecrad_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_function').build()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    input2 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 5
        input2[i] = i + 6

    pos = np.full([2], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, input2=input2, pos=pos, res=res, nlev=0)

    for i in range(size):
        assert res[i, 1] == input1[0]
        assert res[i, 2] == input1[i]
        assert res[i, 3] == input1[i] + input2[i]
        assert res[i, 4] == input1[i] + input2[i]


@xfail('res(nval, pos(1):pos(2)) = a(nval, pos(3):pos(4)) — wrong result')
def test_fortran_frontend_multiple_ranges_ecrad_bug(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_bug_function(input1, res, pos)
double precision, dimension(7, 7) :: input1
double precision, dimension(7, 7) :: res
integer, dimension(4) :: pos
integer :: nval

nval = pos(1)

res(nval, pos(1):pos(2)) = input1(nval, pos(3):pos(4))

END SUBROUTINE multiple_ranges_ecrad_bug_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_bug_function').build()

    size = 7
    input1 = np.full([size, size], 0, order="F", dtype=np.float64)
    for i in range(size):
        for j in range(size):
            input1[i, j] = i + 2**j

    pos = np.full([4], 0, order="F", dtype=np.int32)
    pos[0] = 2
    pos[1] = 5
    pos[2] = 1
    pos[3] = 4

    res = np.full([size, size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, pos=pos, res=res, nval=0)

    iter_1 = pos[0]
    iter_2 = pos[2]
    length = pos[1] - pos[0] + 1

    i = pos[0] - 1
    for j in range(length):
        assert res[i, iter_1 - 1] == input1[i, iter_2 - 1]
        iter_1 += 1
        iter_2 += 1


def test_fortran_frontend_ranges_array_bug(tmp_path):
    src = """
SUBROUTINE multiple_ranges_ecrad_bug_function(input1, res)
double precision, dimension(7) :: input1
double precision, dimension(7) :: res

res(:) = input1(2) * input1(:)

END SUBROUTINE multiple_ranges_ecrad_bug_function
"""
    sdfg = build_sdfg(src, tmp_path, name='multiple_ranges_ecrad_bug_function').build()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 2

    res = np.full([size], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res)

    assert np.all(res == input1 * input1[1])


@xfail('res = literal — Memlet subset mismatch')
def test_fortran_frontend_ranges_noarray(tmp_path):
    src = """
SUBROUTINE ranges_noarray_function(res)
double precision, dimension(7,4) :: res

res = 3

END SUBROUTINE ranges_noarray_function
"""
    sdfg = build_sdfg(src, tmp_path, name='ranges_noarray_function').build()

    res = np.full([7, 4], 42, order="F", dtype=np.float64)
    sdfg(res=res)

    assert np.all(res == 3)


def test_fortran_frontend_ranges_noarray2(tmp_path):
    src = """
SUBROUTINE ranges_noarray_function(inp, res)
double precision, dimension(7,4) :: inp
double precision, dimension(7,4) :: res

res = inp

END SUBROUTINE ranges_noarray_function
"""
    sdfg = build_sdfg(src, tmp_path, name='ranges_noarray_function').build()

    size_x = 7
    size_y = 4
    inp = np.full([size_x, size_y], 0, order="F", dtype=np.float64)
    for i in range(size_x):
        for j in range(size_y):
            inp[i, j] = i + 2**j
    res = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    sdfg(inp=inp, res=res)

    assert np.all(res == inp)


def test_fortran_frontend_ranges_noarray3(tmp_path):
    src = """
SUBROUTINE ranges_noarray_function(inp, res)
double precision, dimension(7,4) :: inp
double precision, dimension(7,4) :: res

res = inp(:,:)

END SUBROUTINE ranges_noarray_function
"""
    sdfg = build_sdfg(src, tmp_path, name='ranges_noarray_function').build()

    size_x = 7
    size_y = 4
    inp = np.full([size_x, size_y], 0, order="F", dtype=np.float64)
    for i in range(size_x):
        for j in range(size_y):
            inp[i, j] = i + 2**j
    res = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    sdfg(inp=inp, res=res)

    assert np.all(res == inp)


def test_fortran_frontend_ranges_scalar(tmp_path):
    src = """
subroutine main(input1, input2, res)
  ! NOTE: `input2`'s declaration is intentially missing, and it still is a valid program.
  double precision, dimension(7) :: input1
  double precision, dimension(7) :: res
  res = 1.0 - input1
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 7
    input1 = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        input1[i] = i + 1
    res = np.full([7], 42, order="F", dtype=np.float64)
    sdfg(input1=input1, res=res)
    assert np.allclose(res, [1.0 - x for x in input1])


@xfail("module-level derived type with array member percent-access not yet lowered")
def test_fortran_frontend_ranges_struct(tmp_path):
    src = """

MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        double precision, dimension(5,4) :: arg1
    END TYPE array_container
END MODULE

MODULE test_range

    contains

    subroutine test_function(arg1, res1)
        USE test_types
        IMPLICIT NONE
        TYPE(array_container) :: container
        double precision, dimension(5,4) :: arg1
        double precision, dimension(5,4) :: res1

        container%arg1(:, :) = arg1

        container%arg1(:, :) = container%arg1 + 1

        res1 = container%arg1
    end subroutine test_function

END MODULE
"""
    sdfg = build_sdfg(src, tmp_path, name='test_function', entry='_QMtest_rangePtest_function').build()

    size_x = 5
    size_y = 4
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    res1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + 1

    sdfg(arg1=arg1, res1=res1)

    assert np.all(res1 == (arg1 + 1))


@xfail("module-level derived type with array member percent-access not yet lowered")
def test_fortran_frontend_ranges_struct_implicit(tmp_path):
    src = """

MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        double precision, dimension(5,4) :: data
    END TYPE array_container
END MODULE

MODULE test_transpose

    contains

    subroutine test_function(arg1, res1)
        USE test_types
        IMPLICIT NONE
        TYPE(array_container) :: container
        double precision, dimension(5,4) :: arg1
        double precision, dimension(5,4) :: res1

        container%data = arg1

        container%data = container%data + 1

        res1 = container%data
    end subroutine test_function

END MODULE
"""
    sdfg = build_sdfg(src, tmp_path, name='test_function', entry='_QMtest_transposePtest_function').build()

    size_x = 5
    size_y = 4
    arg1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    res1 = np.full([size_x, size_y], 42, order="F", dtype=np.float64)

    for i in range(size_x):
        for j in range(size_y):
            arg1[i, j] = i + 1

    sdfg(arg1=arg1, res1=res1)

    assert np.all(res1 == (arg1 + 1))
