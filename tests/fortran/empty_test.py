# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_empty():
    """ 
    Test that empty subroutines and functions are correctly parsed.
    """
    sources, main = SourceCodeBuilder().add_file("""
module module_mpi
  integer, parameter :: process_mpi_all_size = 0
contains
  logical function fun_with_no_arguments()
    fun_with_no_arguments = (process_mpi_all_size <= 1)
  end function fun_with_no_arguments
end module module_mpi

subroutine main(d)
  use module_mpi, only: fun_with_no_arguments
  double precision d(2, 3)
  logical :: bla = .false.

  bla = fun_with_no_arguments()
  if (bla) then
    d(1, 1) = 0
    d(1, 2) = 5
    d(2, 3) = 0
  else
    d(1, 2) = 1
  end if
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


if __name__ == "__main__":
    test_fortran_frontend_empty()
