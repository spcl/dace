# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder
import pytest


def test_fortran_frontend_basic_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@pytest.mark.skip("Fails due to not correctly matching arguments on the InterstateEdge")
def test_fortran_frontend_type_symbol():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@pytest.mark.skip(reason="Crashed pytest in codegen")
def test_fortran_frontend_type_pardecl():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 11)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 5.5)
    assert (a[1, 1] == 42)


@pytest.mark.skip("Fails due to not correctly matching arguments on the InterstateEdge")
def test_fortran_frontend_type_struct():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_circular_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_in_call():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_pointer():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_view():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_func_type_prefix():
    """
    Tests that the Fortran frontend can infer the type of a function in a mathematical expression.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)


def test_fortran_frontend_func_type_body():
    """
    Tests that the Fortran frontend can infer the type of a function in a mathematical expression.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)


if __name__ == "__main__":
    test_fortran_frontend_basic_type()
    test_fortran_frontend_type_symbol()
    test_fortran_frontend_type_pardecl()
    test_fortran_frontend_type_struct()
    test_fortran_frontend_circular_type()
    test_fortran_frontend_type_in_call()
    test_fortran_frontend_type_pointer()
    test_fortran_frontend_type_view()
    test_fortran_frontend_func_type_prefix()
    test_fortran_frontend_func_type_body()
