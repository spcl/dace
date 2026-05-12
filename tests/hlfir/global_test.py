"""Verbatim port of f2dace/dev:tests/fortran/global_test.py."""
from __future__ import annotations

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_global(tmp_path):
    src = """
module global_test_module
  implicit none
  type simple_type
    double precision, pointer :: w(:, :, :)
    integer a
  end type simple_type
  integer :: outside_init = 1
end module global_test_module

module nested_two
  implicit none
contains
  subroutine nestedtwo(i)
    use global_test_module, only: outside_init
    integer :: i
    i = outside_init + 1
  end subroutine nestedtwo
end module nested_two

module nested_one
  implicit none
contains
  subroutine nested(i, a)
    use nested_two, only: nestedtwo
    integer :: i
    double precision :: a(:, :, :)
    i = 0
    call nestedtwo(i)
    a(i + 1, i + 1, i + 1) = 5.5
  end subroutine nested
end module nested_one

subroutine main(d)
  use global_test_module, only: outside_init, simple_type
  use nested_one, only: nested
  double precision :: d(4)
  double precision :: a(4, 4, 4)
  integer :: i
  type(simple_type) :: ptr_patch
  ptr_patch%w(:, :, :) = 5.5
  i = outside_init
  call nested(i, ptr_patch%w)
  d(i + 1) = 5.5 + ptr_patch%w(3, 3, 3)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
