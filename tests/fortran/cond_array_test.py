# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_cond_array_test():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""


subroutine main(d)
  
  implicit none
  real d(5, 5)
  real s1,s2
  s1=d(2,1)+1.0                                               
  call fun(s1)
  s2=5.5
  if (s1+s2 .gt. 5) then
    d(2, 1) = 11
  else
    d(2, 1) = 12
  end if
end subroutine main
                                                 
subroutine fun(s)
  implicit none
  real, intent(inout) :: s
  if (s .lt. 10.0) then
    s = s + 1.0
  else
    s = s - 1.0
  end if
end subroutine fun                                                 
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_cond_array_test()
