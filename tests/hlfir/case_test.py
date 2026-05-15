"""Verbatim port of f2dace/dev:tests/fortran/case_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_case_const(tmp_path):
    """Tests that the cases statement can use parameters."""
    src = """
module lib
  implicit none
  integer, parameter :: a = 1

contains
  subroutine foo(v)
    integer, intent(inout) :: v

    select case(v)
    case(a)
      v = 5
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(5, 5)
  call foo(d(1, 2))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.array([[i + j for i in range(5)] for j in range(5)], order="F", dtype=np.int32)
    sdfg(d=a)
    assert (a[0, 1] == 5)


def test_fortran_frontend_case_const_range(tmp_path):
    """Tests that the cases statement can use parameters."""
    src = """
module lib
  implicit none
  integer, parameter :: a = 1
  integer, parameter :: b = 2
  integer, parameter :: c = 3

contains
  subroutine foo(v)
    integer, intent(inout) :: v

    select case(v)
    case(b:c)
      v = 6
    end select
  end subroutine foo
end module lib

subroutine main(d)
  use lib
  implicit none
  integer :: d(5, 5)
  call foo(d(1, 3))
  call foo(d(1, 5))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.array([[i + j for i in range(5)] for j in range(5)], order="F", dtype=np.int32)
    sdfg(d=a)
    assert (a[0, 2] == 6)
    assert (a[0, 4] == 4)
