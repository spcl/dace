"""Verbatim port of f2dace/dev:tests/fortran/type_array_test.py."""
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


@xfail("module-contained derived-type array slice (conf%fraction(1,:)) not lowered")
def test_fortran_frontend_type_array_slice(tmp_path):
    src = """
module lib
  implicit none
  type conf_type
    double precision :: fraction(5,5)
  end type conf_type
contains
  subroutine f2(array)
    implicit none
    double precision :: array(5)
    call deepest(array)
  end subroutine f2

  subroutine deepest(my_arr)
    double precision :: my_arr(5)
    my_arr(:) = 1
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(conf_type) :: conf
  call f2(conf%fraction(1,:))
  d(1, 1) = conf%fraction(1,2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert d[0][0] == 1


def test_fortran_frontend_type_array(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog(10)
  end type simple_type2
contains
  subroutine f2(stuff)
    implicit none
    type(simple_type) :: stuff
    call deepest(stuff%w)
  end subroutine f2

  subroutine deepest(my_arr)
    real :: my_arr(:, :)
    my_arr(1, 1) = 47
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  call f2(p_prog%pprog(1))
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=d)
    assert d[0][0] == 47


@xfail("nested derived type with whole-struct copy + recall not lowered")
def test_fortran_frontend_type_arrayv2(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog(10)
  end type simple_type2
contains
  subroutine f2(stuff)
    implicit none
    type(simple_type) :: stuff
    call deepest(stuff%w)
  end subroutine f2

  subroutine deepest(my_arr)
    real :: my_arr(:, :)
    my_arr(1, 1) = 42
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  type(simple_type) :: t0
  t0=p_prog%pprog(1)
  call f2(t0)
  d(1, 1) = t0%w(1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type2_array(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real, allocatable :: w(:, :)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog
  end type simple_type2
contains
  subroutine f2(d, stuff)
    type(simple_type2) :: stuff
    real :: d(5, 5)
    call deepest(stuff, d)
  end subroutine f2

  subroutine deepest(my_arr, d)
    real :: d(5, 5)
    type(simple_type2), target :: my_arr
    real, dimension(:, :), pointer, contiguous :: my_arr2
    my_arr2 => my_arr%pprog%w
    d(1, 1) = my_arr2(1, 1)
  end subroutine deepest
end module lib

subroutine main(d, p_prog)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  call f2(d, p_prog)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()


@xfail("nested derived types + allocatable + scalar branches not lowered")
def test_fortran_frontend_type3_array(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real, allocatable :: w(:, :)
  end type simple_type
  type bla_type
    real, allocatable :: a
  end type bla_type
  type metrics_type
    real, allocatable :: b
  end type metrics_type
  type simple_type2
    type(simple_type) :: pprog
    type(bla_type) :: diag
    type(metrics_type):: metrics
  end type simple_type2
contains
  subroutine f2(d, stuff, diag, metrics, istep)
    type(simple_type) :: stuff
    type(bla_type) :: diag
    type(metrics_type) :: metrics
    integer :: istep
    real :: d(5, 5)
    diag%a = 1
    metrics%b = 2
    d(1, 1) = stuff%w(1, 1) + diag%a + metrics%b
    if (istep == 1) then
      call deepest(stuff, d)
    end if
  end subroutine f2
  subroutine deepest(my_arr, d)
    real :: d(5, 5)
    type(simple_type), target :: my_arr
    real, dimension(:, :), pointer, contiguous :: my_arr2
    my_arr2 => my_arr%w
    d(1, 1) = my_arr2(1, 1)
  end subroutine deepest
end module lib

subroutine main(d, p_prog)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  integer :: istep
  istep = 1
  do istep = 1, 2
    if (istep == 1) then
      call f2(d, p_prog%pprog, p_prog%diag, p_prog%metrics, istep)
    else
      call f2(d, p_prog%pprog, p_prog%diag, p_prog%metrics, istep)
    end if
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
