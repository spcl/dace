# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_long_tasklet():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type test_type
    integer :: indices(5)
    integer :: start
    integer :: end
  end type
end module lib

subroutine main(d)
  use lib
  implicit none
  double precision d(5)
  double precision, dimension(50:54) :: arr4
  double precision, dimension(5) :: arr
  type(test_type) :: ind
  arr(:) = 2.0
  ind%indices(:) = 1
  d(2) = 5.5
  d(1) = arr(1)*arr(ind%indices(1))!+arr(2,2,2)*arr(ind%indices(2,2,2),2,2)!+arr(3,3,3)*arr(ind%indices(3,3,3),3,3)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[1] == 5.5)
    assert (a[0] == 4)


if __name__ == "__main__":
    test_fortran_frontend_long_tasklet()
