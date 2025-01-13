# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


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
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_nested_array_access2():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, test1, indices1)
  implicit none
  integer, pointer :: test1(:, :, :)
  integer, pointer :: indices1(:, :, :)
  double precision d(4)
  integer, pointer :: test(:, :, :)
  integer, pointer :: indices(:, :, :)
  test1 => test
  indices1 => indices
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
    sdfg(d=a,__f2dace_A_test_d_0_s_6=3,__f2dace_A_test_d_1_s_7=3,__f2dace_A_test_d_2_s_8=3,
         __f2dace_OA_test_d_0_s_6=1,__f2dace_OA_test_d_1_s_7=1,__f2dace_OA_test_d_2_s_8=1,
         __f2dace_A_indices_d_0_s_9=3,__f2dace_A_indices_d_1_s_10=3,__f2dace_A_indices_d_2_s_11=3,
         __f2dace_OA_indices_d_0_s_9=1,__f2dace_OA_indices_d_1_s_10=1,__f2dace_OA_indices_d_2_s_11=1)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


if __name__ == "__main__":

    #test_fortran_frontend_nested_array_access()
    test_fortran_frontend_nested_array_access2()
