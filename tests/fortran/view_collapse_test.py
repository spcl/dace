# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder
import dace


def test_fortran_frontend_view_collapse():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib1
  implicit none
  real :: outside_init = 1
end module lib1

module lib2
contains
  subroutine view_collapse_test_function(dd)
    use lib1, only: outside_init
    double precision dd
    real:: bob = 1

    if (dd>3) then
      dd = 5.5 + bob
    else
      dd = 336.5

    end if
  end subroutine view_collapse_test_function
end module lib2

subroutine main(d)
  use lib2, only: view_collapse_test_function
  implicit none
  integer :: i
  integer :: j
  double precision d(4,4)

  i=2
  j=1
  call view_collapse_test_function(d(i,j))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')

    sdfg.simplify()

    a = np.full([4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a, outside_init=0)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 6.5)
    assert (a[2, 0] == 42)


if __name__ == "__main__":

    test_fortran_frontend_view_collapse()
