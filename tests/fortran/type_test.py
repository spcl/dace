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
""").add_file("""
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

@pytest.mark.skip(reason="Nested types with arrays to be revisited after merge of struct flattening")
def test_fortran_frontend_basic_type2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


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
""").add_file("""
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

@pytest.mark.skip(reason="This test is segfaulting deterministically in pytest, works fine in debug")
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
""").add_file("""
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
    assert (a[1,1] == 42)


@pytest.mark.skip(reason="Revisit after merge of struct flattening")
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
""").add_file("""
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


@pytest.mark.skip(reason="Circular type removal needs revisiting after merge of struct flattening")
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
""").add_file("""
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
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
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
""").add_file("""
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

@pytest.mark.skip(reason="Revisit after merge of struct flattening")
def test_fortran_frontend_type_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    sdfg.save('test.sdfg')
    sdfg.compile()

    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)


def test_fortran_frontend_type_array2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.save("before.sdfg")
    sdfg.simplify(verbose=True)
    sdfg.save("after.sdfg")
    sdfg.compile()

    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)


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
""").add_file("""
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

@pytest.mark.skip(reason="Nested types with arrays to be revisited after merge of struct flattening")
def test_fortran_frontend_type_arg():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  call f2(p_prog%pprog(1))
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.view()
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)

@pytest.mark.skip(reason="Nested types with arrays to be revisited after merge of struct flattening")
def test_fortran_frontend_type_arg2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.save("before.sdfg")
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)


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
""").add_file("""
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


if __name__ == "__main__":
    test_fortran_frontend_basic_type()
    test_fortran_frontend_basic_type2()
    test_fortran_frontend_type_symbol()
    test_fortran_frontend_type_pardecl()
    test_fortran_frontend_type_struct()
    test_fortran_frontend_circular_type()
    test_fortran_frontend_type_in_call()
    test_fortran_frontend_type_array()
    test_fortran_frontend_type_array2()
    test_fortran_frontend_type_pointer()
    test_fortran_frontend_type_arg()
    test_fortran_frontend_type_view()
    test_fortran_frontend_type_arg2()