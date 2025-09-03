# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_cond_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: id
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: ptr_patch
  logical :: bla = .true.
  ptr_patch%w(1, 1, 1) = 5.5
  ptr_patch%id = 6
  if (ptr_patch%id .gt. 5) then
    d(2, 1) = 5.5 + ptr_patch%w(1, 1, 1)
  else
    d(2, 1) = 12
  end if
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_cond_type()
