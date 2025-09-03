# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_type_array_slice():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(conf_type) :: conf
  call f2(conf%fraction(1,:))
  d(1, 1) = conf%fraction(1,2)
end subroutine main
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(sources, entry_point='main')
    g.simplify(verbose=True)
    d = np.full([5, 5], 42, order="F", dtype=np.float32)
    g(d=d)
    assert d[0][0] == 1


def test_fortran_frontend_type_array():
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
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(sources, entry_point='main')
    g.simplify(verbose=True)
    g.compile()
    d = np.full([5, 5], 42, order="F", dtype=np.float32)
    g(d=d)
    assert d[0][0] == 47


def test_fortran_frontend_type_arrayv2():
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
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)


def test_fortran_frontend_type2_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    # TODO: Just don't use globals :( Otherwise, this line is to keep the name suffixes stable.
    fortran_parser.global_struct_instance_counter = 0

    g = create_singular_sdfg_from_string(sources, entry_point='main')
    g.simplify(verbose=True)

    w = np.full([1, 1], 47, order="F", dtype=np.float32)
    st1_T = g.arrays['v_my_arr_pprog'].dtype.base_type.as_ctypes()
    st1_P = g.arrays['v_my_arr_pprog'].dtype.as_ctypes()
    pprog = st1_T(w=w.ctypes.data,
                  __f2dace_SA_w_d_0_s_2=1, __f2dace_SA_w_d_1_s_3=1,
                  __f2dace_SOA_w_d_0_s_2=1, __f2dace_SOA_w_d_1_s_3=1)
    st2_T = g.arrays['p_prog'].dtype.base_type.as_ctypes()
    p_prog = st2_T(pprog=st1_P(pprog))
    d = np.full([5, 5], 42, order="F", dtype=np.float32)

    g(d=d, p_prog=p_prog,
      __f2dace_SA_w_d_0_s_2_pprog_p_prog_1=1, __f2dace_SA_w_d_1_s_3_pprog_p_prog_1=1,
      __f2dace_SOA_w_d_0_s_2_pprog_p_prog_1=1, __f2dace_SOA_w_d_1_s_3_pprog_p_prog_1=1)
    assert d[0][0] == 47


def test_fortran_frontend_type3_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    # TODO: Just don't use globals :( Otherwise, this line is to keep the name suffixes stable.
    fortran_parser.global_struct_instance_counter = 0

    g = create_singular_sdfg_from_string(sources, entry_point='main')
    g.simplify(verbose=True)
    g.compile()

    w = np.full([1, 1], 47, order="F", dtype=np.float32)
    st1_T = g.arrays['p_prog_pprog_1'].dtype.base_type.as_ctypes()
    st1_P = g.arrays['p_prog_pprog_1'].dtype.as_ctypes()
    pprog = st1_T(w=w.ctypes.data,
                  __f2dace_SA_w_d_0_s_2=1, __f2dace_SA_w_d_1_s_3=1,
                  __f2dace_SOA_w_d_0_s_2=1, __f2dace_SOA_w_d_1_s_3=1)
    bla_T = g.arrays['p_prog_diag_2'].dtype.base_type.as_ctypes()
    bla_P = g.arrays['p_prog_diag_2'].dtype.as_ctypes()
    diag = bla_T(a=1042.0)
    metrics_T = g.arrays['p_prog_metrics_3'].dtype.base_type.as_ctypes()
    metrics_P = g.arrays['p_prog_metrics_3'].dtype.as_ctypes()
    metrics = metrics_T(b=1042.0)
    st2_T = g.arrays['p_prog'].dtype.base_type.as_ctypes()
    p_prog = st2_T(pprog=st1_P(pprog), diag=bla_P(diag), metrics=metrics_P(metrics))
    d = np.full([5, 5], 42, order="F", dtype=np.float32)
    g(d=d, p_prog=p_prog,
      __f2dace_SA_w_d_0_s_2_pprog_p_prog_1=1, __f2dace_SA_w_d_1_s_3_pprog_p_prog_1=1,
      __f2dace_SOA_w_d_0_s_2_pprog_p_prog_1=1, __f2dace_SOA_w_d_1_s_3_pprog_p_prog_1=1)
    assert d[0][0] == 50


if __name__ == "__main__":
    test_fortran_frontend_type_array_slice()
    test_fortran_frontend_type_array()
    test_fortran_frontend_type2_array()
    test_fortran_frontend_type3_array()
    test_fortran_frontend_type_arrayv2()
