"""Verbatim port of f2dace/dev:tests/fortran/cond_type_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cond_type(tmp_path):
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    src = """
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
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)
