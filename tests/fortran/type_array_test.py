# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

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


if __name__ == "__main__":
    test_fortran_frontend_type_array_slice()
