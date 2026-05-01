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
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 12)
    assert (a[2, 0] == 42)


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


def test_fortran_frontend_type_pardecl(tmp_path):
    """Parametric struct-dim local (``real bob2(st%a)``) plus a
    PARAMETER-sized companion (``real bob(n)``).

    Originally xfailed claiming "parametric array dimension not lowered",
    but the actual blocker was test-side bugs: the test passed
    ``np.full([4, 5])`` for a ``d(5, 5)`` dummy (shape mismatch) AND
    wrote ``d(:, 1) = bob(1) + bob2`` — illegal Fortran since LHS is
    rank-1 length 5 and RHS broadcasts to length 10 (the size of
    ``bob2``).  Flang lowered the illegal assign by writing 10
    elements column-major, spilling into column 2 of ``d`` — undefined
    behaviour that made the assertion ``a[1, 1] == 42`` fail (the
    value got overwritten to 5.5).

    Fixes (preserve original intent):
      * Truncate ``bob2`` to length 5 via the slice ``bob2(1:5)`` so the
        assignment is valid Fortran.  ``bob2(st%a=10)`` retains its
        full declared extent so the parametric-dim feature is still
        exercised.
      * ``np.full([5, 5])`` matches the dummy shape.

    Parametric struct dim works today (Phase 5a + 6); this test is the
    cross-subroutine variant of ``derived_type_test.py::
    test_parametric_dim_via_inlined_subprogram``.
    """
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
  d(:, 1) = bob(1) + bob2(1:5)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 11)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 5.5)
    assert (a[1, 1] == 42)


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
    # Should NOT need to bind ``sta_d0`` / ``sta_d1`` — ``st_z`` is
    # concretely (3, 3) and ``sta`` is just an inlined alias.  The
    # SDFG signature surfaces these synth symbols today only because
    # ``asAssumedShapeAlias`` doesn't trace through a flattened-field
    # designate; once that's fixed they should disappear.
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


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
