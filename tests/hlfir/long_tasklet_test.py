"""Verbatim port of f2dace/dev:tests/fortran/long_tasklet_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_long_tasklet(tmp_path):
    src = """
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
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[1] == 5.5)
    assert (a[0] == 4)
