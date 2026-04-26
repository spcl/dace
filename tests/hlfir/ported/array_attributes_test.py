"""Verbatim port of f2dace/dev:tests/fortran/array_attributes_test.py."""
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


def test_fortran_frontend_array_attribute_no_offset(tmp_path):
    src = """
subroutine main(d)
  integer :: i
  double precision, dimension(5) :: d
  do i = 1, 5
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0)
    for i in range(1, 5):
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_no_offset_symbol(tmp_path):
    src = """
subroutine main(d, arrsize)
  integer :: arrsize,i
  double precision, dimension(arrsize) :: d

  do i = 1, arrsize
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 10
    a = np.full([size], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=size, i=0)
    for i in range(1, size):
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_offset(tmp_path):
    src = """
subroutine main(d)
  integer :: i
  double precision, dimension(50:54) :: d
  do i = 50, 54
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0)
    for i in range(1, 5):
        assert a[i - 1] == (i - 1 + 50) * 2


def test_fortran_frontend_array_attribute_offset_symbol(tmp_path):
    src = """
subroutine main(d, arrsize)
  integer :: arrsize,i
  double precision, dimension(arrsize:arrsize + 4) :: d
  do i = arrsize, arrsize + 4
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    arrsize = 50
    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, i=0)
    arrsize = 1
    for i in range(arrsize, arrsize + 4):
        assert a[i - 1] == (i - 1 + 50) * 2


def test_fortran_frontend_array_attribute_offset_symbol2(tmp_path):
    src = """
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision, dimension(arrsize:arrsize2) :: d
  do i = arrsize, arrsize2
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    arrsize = 50
    arrsize2 = 54
    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2, i=0)
    for i in range(1, 5):
        assert a[i - 1] == (i - 1 + 50) * 2


def test_fortran_frontend_array_offset(tmp_path):
    src = """
subroutine main(d)
  double precision d(50:54)
  integer :: i
  do i = 50, 54
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0)
    for i in range(1, 5):
        assert a[i - 1] == (50 + i - 1) * 2


def test_fortran_frontend_array_offset_symbol(tmp_path):
    src = """
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision :: d(arrsize:arrsize2)
  do i = arrsize, arrsize2
    d(i) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    arrsize = 50
    arrsize2 = 54
    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2, i=0)
    for i in range(1, 5):
        assert a[i - 1] == (i + 50 - 1) * 2


@xfail("assumed-shape arrays (d(:,:)) need allocatable descriptor lowering")
def test_fortran_frontend_array_arbitrary(tmp_path):
    src = """
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision :: d(:, :)
  do i = 1, arrsize
    d(i, 1) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2, i=0)
    for i in range(arrsize):
        assert a[i, 0] == (i + 1) * 2


@xfail("assumed-shape arrays (dimension(:,:)) need allocatable descriptor lowering")
def test_fortran_frontend_array_arbitrary_attribute(tmp_path):
    src = """
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision, dimension(:, :) :: d
  do i = 1, arrsize
    d(i, 1) = i*2.0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2, i=0)
    for i in range(arrsize):
        assert a[i, 0] == (i + 1) * 2


@xfail("assumed-shape array + module-contained subroutine call not yet lowered")
def test_fortran_frontend_array_arbitrary_attribute2(tmp_path):
    src = """
module lib
contains
  subroutine main(d, d2)
    double precision, dimension(:, :) :: d, d2
    call other(d, d2)
  end subroutine main

  subroutine other(d, d2)
    double precision, dimension(:, :) :: d, d2
    d(1, 1) = size(d, 1)
    d(1, 2) = size(d, 2)
    d(1, 3) = size(d2, 1)
    d(1, 4) = size(d2, 2)
  end subroutine other
end module lib
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QMlibPmain').build()

    arrsize = 5
    arrsize2 = 10
    arrsize3 = 3
    arrsize4 = 7
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    b = np.full([arrsize3, arrsize4], 42, order="F", dtype=np.float64)
    sdfg(d=a, d2=b)
    assert a[0, 0] == arrsize
    assert a[0, 1] == arrsize2
    assert a[0, 2] == arrsize3
    assert a[0, 3] == arrsize4
