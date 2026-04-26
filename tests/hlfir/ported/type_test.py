"""Verbatim port of f2dace/dev:tests/fortran/type_test.py."""
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


@xfail("module-level derived type with array members not lowered")
def test_fortran_frontend_basic_type(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: s
  s%w(1, 1, 1) = 5.5
  d(2, 1) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("nested module-level derived types not lowered")
def test_fortran_frontend_basic_type2(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: w(5, 5, 5), z(5)
    integer:: a
  end type simple_type
  type comlex_type
    type(simple_type):: s
    real:: b
  end type comlex_type
  type meta_type
    type(comlex_type):: cc
    real:: omega
  end type meta_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: s(3)
  type(comlex_type) :: c
  type(meta_type) :: m
  c%b = 1.0
  c%s%w(1, 1, 1) = 5.5
  m%cc%s%a = 17
  s(1)%w(1, 1, 1) = 5.5 + c%b
  d(2, 1) = c%s%w(1, 1, 1) + s(1)%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("derived type with array dim from struct field (real bob(st%a)) not lowered")
def test_fortran_frontend_type_symbol(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5)
    integer:: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d, st)
end subroutine main

subroutine internal_function(d, st)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: st
  real bob(st%a)
  bob(1) = 5.5
  d(2, 1) = 2*bob(1)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("derived type with parametric array dimension (real bob2(st%a)) not lowered")
def test_fortran_frontend_type_pardecl(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5, 5, 5)
    integer:: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d, st)
end subroutine main

subroutine internal_function(d, st)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: st

  integer, parameter :: n = 5
  real bob(n)
  real bob2(st%a)
  bob(1) = 5.5
  bob2(:) = 0
  bob2(1) = 5.5
  d(:, 1) = bob(1) + bob2
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 11)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 5.5)
    assert (a[1, 1] == 42)


@xfail("derived type passed by reference between subroutines not lowered")
def test_fortran_frontend_type_struct(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5, 5, 5)
    integer:: a
    !real, allocatable :: unknown(:)
    !INTEGER :: unkown_size
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d,st)
end subroutine main

subroutine internal_function(d,st)
  use lib
  implicit none
  !! WHAT DOES THIS MEAN?
  ! st.a.shape = [st.a_size]
  real d(5, 5)
  type(simple_type) :: st
  real bob(st%a)
  integer, parameter :: n = 5
  real BOB2(n)
  bob(1) = 5.5
  bob2(1) = 5.5
  st%z(1, :, 2:3) = bob(1)
  d(2, 1) = bob(1) + bob2(1)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("circular type definitions (a_t referencing b_t pointer) not lowered")
def test_fortran_frontend_circular_type(tmp_path):
    src = """
module lib
  implicit none
  type a_t
    real :: w(5, 5, 5)
    type(b_t), pointer :: b
  end type a_t
  type b_t
    type(a_t) :: a
    integer :: x
  end type b_t
  type c_t
    type(d_t), pointer :: ab
    integer :: xz
  end type c_t
  type d_t
    type(c_t) :: ac
    integer :: xy
  end type d_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(a_t) :: s
  type(b_t) :: b(3)
  s%w(1, 1, 1) = 5.5
  ! s%b=>b(1)
  ! s%b%a=>s
  b(1)%x = 1
  d(2, 1) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_in_call(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type), target :: s
  real, pointer :: tmp(:, :, :)
  tmp => s%w
  tmp(1, 1, 1) = 11.0
  d(2, 1) = max(1.0, tmp(1, 1, 1))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_array(tmp_path):
    src = """
module lib
  implicit none

  type simple_type3
    integer :: a
  end type simple_type3

  type simple_type2
    type(simple_type3) :: w(7:12, 8:13)
  end type simple_type2

  type simple_type
    type(simple_type2) :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type) :: s
  call f2(s)
  d(1, 1) = s%name%w(8, 10)%a
end subroutine main

subroutine f2(s)
  use lib
  implicit none
  type(simple_type) :: s
  s%name%w(8, 10)%a = 42
end subroutine f2
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_array2(tmp_path):
    src = """
module lib
  implicit none

  type simple_type3
    integer :: a
  end type simple_type3

  type simple_type2
    type(simple_type3) :: w(7:12, 8:13)
    integer :: wx(7:12, 8:13)
  end type simple_type2

  type simple_type
    type(simple_type2) :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  integer :: x(3, 3, 3)
  type(simple_type) :: s
  call f2(s, x)
  !d(1,1) = s%name%w(8, x(3,3,3))%a
  d(1, 2) = s%name%wx(8, x(3, 3, 3))
end subroutine main

subroutine f2(s, x)
  use lib
  implicit none
  type(simple_type) :: s
  integer :: x(3, 3, 3)
  x(3, 3, 3) = 10
  !s%name%w(8,x(3,3,3))%a = 42
  s%name%wx(8, x(3, 3, 3)) = 43
end subroutine f2
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_pointer(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type), target :: s
  real, dimension(:, :, :), pointer :: tmp
  tmp => s%w
  tmp(1, 1, 1) = 11.0
  d(2, 1) = max(1.0, tmp(1, 1, 1))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("nested derived types with allocatable arrays not lowered")
def test_fortran_frontend_type_arg(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real, pointer, contiguous :: w(:, :)
  end type simple_type
  type simple_type2
    type(simple_type), allocatable :: pprog(:)
  end type simple_type2
contains
  subroutine f2(stuff)
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
  call f2(p_prog%pprog(1))
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


@xfail("nested derived types passed via percent-percent path not lowered")
def test_fortran_frontend_type_arg2(tmp_path):
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
  subroutine deepest(my_arr, d)
    real :: my_arr(:, :)
    real :: d(5, 5)
    my_arr(1, 1) = 5.5
    d(1, 1) = my_arr(1, 1)
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  integer :: i
  i = 1

  !p_prog%pprog(1)%w(1,1) = 5.5
  call deepest(p_prog%pprog(i)%w, d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


@xfail("derived type field passed as assumed-shape arg not lowered")
def test_fortran_frontend_type_view(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: z(3, 3)
    integer :: a
  end type simple_type
contains
  subroutine internal_function(d, sta)
    real d(5, 5)
    real sta(:, :)
    d(2, 1) = 2*sta(1, 1)
  end subroutine internal_function
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%z(1, 1) = 5.5
  call internal_function(d, st%z)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("module-contained function returning real ** 2.0 not yet lowered")
def test_fortran_frontend_func_type_prefix(tmp_path):
    src = """
module lib
  implicit none
contains
  real function custom_sum(d)
    real :: d(5, 5)
    integer :: i, j
    do i = 1, 5
      do j = 1, 5
        custom_sum = custom_sum + d(i, j)
      end do
    end do
  end function custom_sum
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: norm
  real :: d(5, 5)
  d(1, 1) = custom_sum(d) ** 2.0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)


@xfail("module-contained function with declared body return type not yet lowered")
def test_fortran_frontend_func_type_body(tmp_path):
    src = """
module lib
  implicit none
contains
  function custom_sum(d)
    real :: custom_sum
    real :: d(5, 5)
    integer :: i, j
    do i = 1, 5
      do j = 1, 5
        custom_sum = custom_sum + d(i, j)
      end do
    end do
  end function custom_sum
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: norm
  real :: d(5, 5)
  d(1, 1) = custom_sum(d) ** 2.0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)
