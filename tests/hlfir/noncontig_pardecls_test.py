"""Verbatim port of f2dace/dev:tests/fortran/noncontig-pardecls_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_noncontiguous_slices(tmp_path):
    src = """
subroutine main(d, d2)
  double precision, dimension(5) :: d
  double precision, dimension(3) :: d2
  integer, dimension(3) :: cols

  cols(1) = 1
  cols(2) = 3
  cols(3) = 5

  call fun( d(cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(3) :: d
  double precision, dimension(3) :: d2
  integer :: i
  do i = 1, 3
    d2(i) = d(i)*2.0
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size = 5
    d = np.full([size], 42, order="F", dtype=np.float64)
    d2 = np.full([3], 42, order="F", dtype=np.float64)
    for i in range(0, size):
        d[i] = i + 1

    sdfg(d=d, d2=d2)

    assert np.all(d == [1, 2, 3, 4, 5])
    assert np.all(d2 == [2, 6, 10])


def test_fortran_frontend_noncontiguous_slices_2d(tmp_path):
    src = """
subroutine main(d, d2)
  double precision, dimension(4, 5) :: d
  double precision, dimension(3) :: d2
  integer, dimension(3) :: cols

  cols(1) = 1
  cols(2) = 3
  cols(3) = 5

  call fun( d(2, cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(5) :: d
  double precision, dimension(3) :: d2
  integer :: i
  do i = 1, 3
    d2(i) = d(i)*2.0
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[1, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_slices_2d_double_copy(tmp_path):
    src = """
subroutine main(d, d2)
  double precision, dimension(4, 5) :: d
  double precision, dimension(2, 3) :: d2
  integer, dimension(3) :: cols
  integer, dimension(2) :: cols2

  cols(1) = 1
  cols(2) = 3
  cols(3) = 5

  cols2(1) = 2
  cols2(2) = 4

  call fun( d(cols2, cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(2, 5) :: d
  double precision, dimension(2, 3) :: d2
  integer :: i, j

  do j = 1, 2
    do i = 1, 3
        d2(j, i) = d(j, i)*2.0
    end do
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([2, 3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[[1, 3]][:, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_slices_2d_pardecl(tmp_path):
    src = """
subroutine main(d, d2)
  double precision, dimension(4, 5) :: d
  double precision, dimension(3, 3) :: d2
  integer, dimension(3) :: cols

  cols(1) = 1
  cols(2) = 3
  cols(3) = 5

  call fun( d(2:4, cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(2, 5) :: d
  double precision, dimension(3, 3) :: d2
  integer :: i, j

  do j = 1, 3
    do i = 1, 3
        d2(j, i) = d(j, i)*2.0
    end do
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([3, 3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[1:4][:, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_slices_2d_pardecl2(tmp_path):
    src = """
subroutine main(d, d2)
  double precision, dimension(4, 5) :: d
  double precision, dimension(3, 3) :: d2
  integer, dimension(3) :: cols

  cols(1) = 1
  cols(2) = 3
  cols(3) = 5

  call fun( d(:, cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(4, 3) :: d
  double precision, dimension(3, 3) :: d2
  integer :: i, j

  do j = 1, 3
    do i = 1, 3
        d2(j, i) = d(j + 1, i)*2.0
    end do
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([3, 3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[1:4][:, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_slices_2d_data_refs(tmp_path):
    src = """

module lib
    implicit none
    type test_type
        double precision, dimension(4,5) :: input_data
        integer, dimension(3) :: cols
    end type

end module lib

subroutine main(d, d2)
  use lib, only: test_type
  implicit none

  double precision, dimension(4, 5) :: d
  double precision, dimension(3, 3) :: d2

  type(test_type) :: data
  integer :: startcol, endcol

  data%cols(1) = 1
  data%cols(2) = 3
  data%cols(3) = 5

  data%input_data = d

  startcol = 2
  endcol = 4

  call fun( data%input_data( startcol : endcol , data%cols), d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(3, 3) :: d
  double precision, dimension(3, 3) :: d2
  integer :: i, j

  do j = 1, 3
    do i = 1, 3
        d2(j, i) = d(j, i)*2.0
    end do
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([3, 3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[1:4][:, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_nested(tmp_path):
    src = """

subroutine main(d, d2)
  double precision, dimension(4, 5) :: d
  double precision, dimension(3) :: d2
  double precision, dimension(3) :: d3
  integer, dimension(6) :: cols
  integer, dimension(3) :: cols2

  cols(1) = 6
  cols(2) = 5
  cols(3) = 4
  cols(4) = 3
  cols(5) = 2
  cols(6) = 1

  cols2(1) = 6
  cols2(2) = 4
  cols2(3) = 2

  ! We should in the end select columns 1, 3, 5

  call fun( d(2, cols(cols2)) , d2)
end subroutine main

subroutine fun(d, d2)
  double precision, dimension(5) :: d
  double precision, dimension(3) :: d2
  integer :: i
  do i = 1, 3
    d2(i) = d(i)*2.0
  end do
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size_x, size_y = 4, 5
    d = np.full([size_x, size_y], 42, order="F", dtype=np.float64)
    d2 = np.full([3], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        for j in range(0, size_y):
            d[i, j] = i + 20 * j

    sdfg(d=d, d2=d2)

    assert np.all(d[1, [0, 2, 4]] * 2 == d2)


def test_fortran_frontend_noncontiguous_nested_ecrad(tmp_path):
    src = """

subroutine main(d, sizex, firstcols, secondcols, res)

  integer :: sizex
  double precision, dimension(sizex, 5) :: d

  integer, dimension(16) :: firstcols
  integer, dimension(140) :: secondcols

  double precision, dimension(5, 140) :: res

  res = 1.0 - transpose( d(firstcols(secondcols), :))
end subroutine main

"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size_x = 5
    d = np.full([1, size_x], 42, order="F", dtype=np.float64)
    for i in range(0, size_x):
        d[0, i] = 0 + (i + 1) / 10

    firstcols = np.full([16], 1, order="F", dtype=np.int32)

    secondcols = np.full([140], 1, order="F", dtype=np.int32)
    for i in range(140):
        secondcols[i] = (i % 16) + 1

    res = np.full([size_x, 140], 42, order="F", dtype=np.float64)

    sdfg(d=d, sizex=1, res=res, firstcols=firstcols, secondcols=secondcols)

    firstcols -= 1
    secondcols -= 1
    d_new = d[firstcols[secondcols], :]
    new_res = 1.0 - np.transpose(d_new)

    assert (new_res == res).all()
