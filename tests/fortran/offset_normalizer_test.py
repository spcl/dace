# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.fortran_parser import create_internal_ast, ParseConfig, create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_offset_normalizer_1d():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision, dimension(50:54) :: d
  integer :: i
  do i = 50, 54
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.all(a == [(50 + i) * 2 for i in range(0, 5)])


def test_fortran_frontend_offset_normalizer_1d_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2
  double precision :: d(arrsize:arrsize2)
  integer :: i
  do i = arrsize, arrsize2
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    from dace.symbolic import evaluate
    arrsize = 50
    arrsize2 = 54
    assert len(sdfg.data('d').shape) == 1
    assert evaluate(sdfg.data('d').shape[0], {'sym_arrsize': arrsize, 'sym_arrsize2': arrsize2}) == 5

    arrsize = 50
    arrsize2 = 54
    a = np.full([arrsize2 - arrsize + 1], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(0, arrsize2 - arrsize + 1):
        assert a[i] == (50 + i) * 2


def test_fortran_frontend_offset_normalizer_2d():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision, dimension(50:54, 7:9) :: d
  integer :: i,j
  do i = 50, 54
    do j = 7, 9
      d(i, j) = i*2.0 + 3*j
    end do
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == (50 + i) * 2 + 3 * (7 + j)


def test_fortran_frontend_offset_normalizer_2d_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2, arrsize3, arrsize4)
  integer :: arrsize
  integer :: arrsize2
  integer :: arrsize3
  integer :: arrsize4
  integer :: i,j
  double precision, dimension(arrsize:arrsize2, arrsize3:arrsize4) :: d
  do i = arrsize, arrsize2
    do j = arrsize3, arrsize4
      d(i, j) = i*2.0 + 3*j
    end do
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    from dace.symbolic import evaluate
    values = {'sym_arrsize': 50, 'sym_arrsize2': 54, 'sym_arrsize3': 7, 'sym_arrsize4': 9}
    assert len(sdfg.data('d').shape) == 2
    assert evaluate(sdfg.data('d').shape[0], values) == 5
    assert evaluate(sdfg.data('d').shape[1], values) == 3
    values = {
        'sym_arrsize': 50,
        'sym_arrsize2': 54,
        'sym_arrsize3': 7,
        'sym_arrsize4': 9,
        'arrsize': 50,
        'arrsize2': 54,
        'arrsize3': 7,
        'arrsize4': 9
    }
    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a, **values)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == (50 + i) * 2 + 3 * (7 + j)


def test_fortran_frontend_offset_normalizer_2d_arr2loop():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision, dimension(50:54, 7:9) :: d
  integer :: i
  do i = 50, 54
    d(i, :) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == (50 + i) * 2


def test_fortran_frontend_offset_normalizer_2d_arr2loop_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2, arrsize3, arrsize4)
  integer :: arrsize
  integer :: arrsize2
  integer :: arrsize3
  integer :: arrsize4
  double precision, dimension(arrsize:arrsize2, arrsize3:arrsize4) :: d
  integer :: i
  do i = arrsize, arrsize2
    d(i, :) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    from dace.symbolic import evaluate
    values = {'sym_arrsize': 50, 'sym_arrsize2': 54, 'sym_arrsize3': 7, 'sym_arrsize4': 9}
    assert len(sdfg.data('d').shape) == 2
    assert evaluate(sdfg.data('d').shape[0], values) == 5
    assert evaluate(sdfg.data('d').shape[1], values) == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    values = {
        'sym_arrsize': 50,
        'sym_arrsize2': 54,
        'sym_arrsize3': 7,
        'sym_arrsize4': 9,
        'arrsize': 50,
        'arrsize2': 54,
        'arrsize3': 7,
        'arrsize4': 9
    }
    sdfg(d=a, **values)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == (50 + i) * 2


def test_fortran_frontend_offset_normalizer_struct():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type simple_type
    double precision :: d(5, 3)
    integer :: arrsize
    integer :: arrsize2
    integer :: arrsize3
    integer :: arrsize4
  end type simple_type
end module lib

subroutine main(d, arrsize, arrsize2, arrsize3, arrsize4)
  use lib
  implicit none
  integer :: arrsize
  integer :: arrsize2
  integer :: arrsize3
  integer :: arrsize4,i,j
  double precision, dimension(arrsize:arrsize2, arrsize3:arrsize4) :: d
  type(simple_type) :: struct_data

  struct_data%arrsize = arrsize
  struct_data%arrsize2 = arrsize2
  struct_data%arrsize3 = arrsize3
  struct_data%arrsize4 = arrsize4

  do i=struct_data%arrsize,struct_data%arrsize2
    do j=struct_data%arrsize3,struct_data%arrsize4
      struct_data%d(i, j) = i * 2.0 +j
      d(i, j) = struct_data%d(i, j)
    end do
  end do



end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    from dace.symbolic import evaluate
    values = {'sym_arrsize': 50, 'sym_arrsize2': 54, 'sym_arrsize3': 7, 'sym_arrsize4': 9}
    assert len(sdfg.data('d').shape) == 2
    assert evaluate(sdfg.data('d').shape[0], values) == 5
    assert evaluate(sdfg.data('d').shape[1], values) == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    values = {
        'sym_arrsize': 50,
        'sym_arrsize2': 54,
        'sym_arrsize3': 7,
        'sym_arrsize4': 9,
        'arrsize': 50,
        'arrsize2': 54,
        'arrsize3': 7,
        'arrsize4': 9
    }
    sdfg(d=a, **values)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == (i + 50) * 2 + 7 + j


def test_fortran_frontend_offset_pardecl():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision, dimension(50:54) :: d
  call fun(d(51:53))
end subroutine main

subroutine fun(d)
  double precision, dimension(3) :: d
  integer :: i
  do i = 1, 3
    d(i) = i*2.0
  end do
end subroutine fun
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 3):
        assert a[i] == (i) * 2


def test_fortran_frontend_offset_pardecl_struct():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type simple_type
    double precision :: d(50:54)
  end type simple_type
end module lib

subroutine main(in1)
  use lib
  implicit none
  double precision, dimension(50:54) :: in1
  type(simple_type) :: struct_data

  struct_data%d = in1

  call fun(struct_data%d(51:53))

  in1(51:53) = struct_data%d(51:53)
end subroutine main

subroutine fun(d)
  double precision, dimension(3) :: d
  integer :: i
  do i = 1, 3
    d(i) = i*2.0
  end do
end subroutine fun
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main')
    sdfg.simplify()
    sdfg.compile()

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(in1=a)
    for i in range(1, 3):
        assert a[i] == (i) * 2


if __name__ == "__main__":
    test_fortran_frontend_offset_normalizer_1d()
    test_fortran_frontend_offset_normalizer_2d()
    test_fortran_frontend_offset_normalizer_2d_arr2loop()
    test_fortran_frontend_offset_normalizer_1d_symbol()
    test_fortran_frontend_offset_normalizer_2d_symbol()
    test_fortran_frontend_offset_normalizer_2d_arr2loop_symbol()
    test_fortran_frontend_offset_normalizer_struct()
    test_fortran_frontend_offset_pardecl()
    test_fortran_frontend_offset_pardecl_struct()
