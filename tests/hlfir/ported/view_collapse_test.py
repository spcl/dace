"""Verbatim port of f2dace/dev:tests/fortran/view_collapse_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_view_collapse(tmp_path):
    src = """
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
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    a = np.full([4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0, j=0)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 6.5)
    assert (a[2, 0] == 42)
