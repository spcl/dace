# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder, deduce_f2dace_variables_for_array


def test_fortran_frontend_nested_array_access():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(4)
  integer test(3, 3, 3)
  integer indices(3, 3, 3)
  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_fortran_frontend_nested_array_access_pointer_args_1():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - But the function actually does not read from the input pointers, but immediately repoint them to internal arrays.
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  integer, target :: internal_test(3, 4, 5)
  integer, target :: internal_indices(3, 4, 5)
  indices => internal_test
  indices => internal_indices
  internal_indices(1, 1, 1) = 2
  internal_indices(1, 1, 2) = 3
  internal_indices(1, 1, 3) = 1
  internal_test(internal_indices(1, 1, 1), internal_indices(1, 1, 2), internal_indices(1, 1, 3)) = 2
  d(internal_test(2, 3, 1)) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([0, 0, 0], 42, order="F", dtype=np.int32)
    indices = np.full([0, 0, 0], 42, order="F", dtype=np.int32)
    sdfg(d=d,
         test=test, **deduce_f2dace_variables_for_array(test, 'test', 0),
         indices=indices, **deduce_f2dace_variables_for_array(indices, 'indices', 3))
    assert np.allclose(d, [42, 5.5, 42, 42])


def test_fortran_frontend_nested_array_access_pointer_args_2():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - This time, the function will actually write and read from those arrays (although igoring their initial values).
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    indices = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    sdfg(d=d,
         test=test, **deduce_f2dace_variables_for_array(test, 'test', 0),
         indices=indices, **deduce_f2dace_variables_for_array(indices, 'indices', 3))
    assert np.allclose(d, [42, 5.5, 42, 42])


if __name__ == "__main__":
    test_fortran_frontend_nested_array_access()
    test_fortran_frontend_nested_array_access_pointer_args_1()
    test_fortran_frontend_nested_array_access_pointer_args_2()
