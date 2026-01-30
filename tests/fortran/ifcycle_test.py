# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_if_cycle():
    sources, main = SourceCodeBuilder().add_file(
        """
subroutine main(d)
  double precision d(4)
  integer :: i
  do i = 1, 4
    if (i .eq. 2) cycle
    d(i) = 5.5
  end do
  if (d(2) .eq. 42) d(2) = 6.5
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 5.5)
    assert (a[1] == 6.5)
    assert (a[2] == 5.5)


if __name__ == "__main__":
    test_fortran_frontend_if_cycle()
