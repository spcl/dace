"""Verbatim port of f2dace/dev:tests/fortran/non-interactive/pointers_test.py."""
from __future__ import annotations

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_pointer_test(tmp_path):
    src = """
subroutine main(lon, lout)
  real, intent(in) :: lon(10)
  real, intent(out) :: lout(10)
  type simple_type
    real:: w(5, 5, 5), z(5)
    integer:: a
  end type simple_type
  type(simple_type), target :: s
  real :: area
  real, pointer, contiguous :: p_area(:, :, :)
  integer :: i, j
  s%w(1, 1, 1) = 5.5
  lout(:) = 0.0
  p_area => s%w
  lout(1) = p_area(1, 1, 1) + lon(1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
