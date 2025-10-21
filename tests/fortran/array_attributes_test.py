# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder, deduce_f2dace_variables_for_array


def test_fortran_frontend_array_attribute_no_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  integer :: i
  double precision, dimension(5) :: d
  do i = 1, 5
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_no_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize)
  integer :: arrsize,i
  double precision, dimension(arrsize) :: d

  do i = 1, arrsize
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    assert len(sdfg.data('d').shape) == 1
    from dace.symbolic import symbol
    assert isinstance(sdfg.data('d').shape[0], symbol)
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    size = 10
    a = np.full([size], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=size)
    for i in range(1, size):
        # offset -1 is already added
        assert a[i - 1] == i * 2


def test_fortran_frontend_array_attribute_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  integer :: i
  double precision, dimension(50:54) :: d
  do i = 50, 54
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == (i - 1 + 50) * 2


def test_fortran_frontend_array_attribute_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize)
  integer :: arrsize,i
  double precision, dimension(arrsize:arrsize + 4) :: d
  do i = arrsize, arrsize + 4
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    arrsize = 50
    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize)
    arrsize = 1
    for i in range(arrsize, arrsize + 4):
        # offset -1 is already added
        assert a[i - 1] == (i - 1 + 50) * 2


def test_fortran_frontend_array_attribute_offset_symbol2():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    Compared to the previous one, this one should prevent simplification from removing symbols
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision, dimension(arrsize:arrsize2) :: d
  do i = arrsize, arrsize2
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    from dace.symbolic import evaluate

    arrsize = 50
    arrsize2 = 54
    assert len(sdfg.data('d').shape) == 1
    assert evaluate(sdfg.data('d').shape[0], {'sym_arrsize': arrsize, 'sym_arrsize2': arrsize2}) == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == (i + 50 - 1) * 2


def test_fortran_frontend_array_offset():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(50:54)
  integer :: i
  do i = 50, 54
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([60], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == (50 + i - 1) * 2


def test_fortran_frontend_array_offset_symbol():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    Compared to the previous one, this one should prevent simplification from removing symbols
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision :: d(arrsize:arrsize2)
  do i = arrsize, arrsize2
    d(i) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', False)
    sdfg.simplify()

    from dace.symbolic import evaluate

    arrsize = 50
    arrsize2 = 54
    assert len(sdfg.data('d').shape) == 1
    assert evaluate(sdfg.data('d').shape[0], {'sym_arrsize': arrsize, 'sym_arrsize2': arrsize2}) == 5
    assert len(sdfg.data('d').offset) == 1
    assert sdfg.data('d').offset[0] == -1

    a = np.full([arrsize + 10], 42, order="F", dtype=np.float64)
    sdfg(d=a, arrsize=arrsize, arrsize2=arrsize2)
    for i in range(1, 5):
        # offset -1 is already added
        assert a[i - 1] == (i + 50 - 1) * 2


def test_fortran_frontend_array_arbitrary():
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision :: d(:, :)
  do i = 1, arrsize
    d(i, 1) = i*2.0
  end do
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=False)
    sdfg.simplify()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, **deduce_f2dace_variables_for_array(a, 'd', 0), arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize):
        # offset -1 is already added
        assert a[i, 0] == (i + 1) * 2


def test_fortran_frontend_array_arbitrary_attribute():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, arrsize, arrsize2)
  integer :: arrsize
  integer :: arrsize2,i
  double precision, dimension(:, :) :: d
  do i = 1, arrsize
    d(i, 1) = i*2.0
  end do
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=False)
    sdfg.simplify()

    arrsize = 5
    arrsize2 = 10
    a = np.full([arrsize, arrsize2], 42, order="F", dtype=np.float64)
    sdfg(d=a, **deduce_f2dace_variables_for_array(a, 'd', 0), arrsize=arrsize, arrsize2=arrsize2)
    for i in range(arrsize):
        # offset -1 is already added
        assert a[i, 0] == (i + 1) * 2


if __name__ == "__main__":
    test_fortran_frontend_array_offset()
    test_fortran_frontend_array_attribute_no_offset()
    test_fortran_frontend_array_attribute_offset()
    test_fortran_frontend_array_attribute_no_offset_symbol()
    test_fortran_frontend_array_attribute_offset_symbol()
    test_fortran_frontend_array_attribute_offset_symbol2()
    test_fortran_frontend_array_offset_symbol()
    test_fortran_frontend_array_arbitrary()
    test_fortran_frontend_array_arbitrary_attribute()
