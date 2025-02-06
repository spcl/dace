# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_if_cycle():
    sources, main = SourceCodeBuilder().add_file("""
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
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 5.5)
    assert (a[1] == 6.5)
    assert (a[2] == 5.5)


def test_fortran_frontend_if_nested_cycle():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(4, 4)
  double precision :: tmp
  integer :: i, j, stop, start, count
  stop = 4
  start = 1
  do i = start, stop
    count = 0
    do j = start, stop
      if (j .eq. 2) count = count + 2
    end do
    if (count .eq. 2) cycle
    if (count .eq. 3) cycle
    do j = start, stop
      d(i, j) = d(i, j) + 1.5
    end do
    d(i, 1) = 5.5
  end do
  if (d(2, 1) .eq. 42.0) d(2, 1) = 6.5
end subroutine main
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4,4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0,0] == 42)
    assert (a[1,0] == 6.5)
    assert (a[2,0] == 42)    


if __name__ == "__main__":
    test_fortran_frontend_if_cycle()
    test_fortran_frontend_if_nested_cycle()
