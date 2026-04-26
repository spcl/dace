"""Verbatim port of f2dace/dev:tests/fortran/struct_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from ported._helpers import xfail

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@xfail("module-level derived type passed across subroutines not lowered")
def test_fortran_struct(tmp_path):
    src = """
module lib
  implicit none
  type test_type
    integer :: start
    integer :: end
  end type
end module lib

subroutine main(res, startidx, endidx)
  use lib
  implicit none
  integer, dimension(6) :: res
  integer :: startidx
  integer :: endidx
  type(test_type) :: indices
  indices%start = startidx
  indices%end = endidx
  call fun(res, indices)
end subroutine main

subroutine fun(res, idx)
  use lib
  implicit none
  integer, dimension(6) :: res
  type(test_type) :: idx
  res(idx%start:idx%end) = 42
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size = 6
    res = np.full([size], 42, order="F", dtype=np.int32)
    res[:] = 0
    sdfg(res=res, startidx=2, endidx=5)


@xfail("nested derived types passed across subroutines not lowered")
def test_fortran_struct_lhs(tmp_path):
    src = """
module lib
  implicit none
  type test_type
    integer, dimension(6) :: res
    integer :: start
    integer :: end
  end type
  type test_type2
    type(test_type) :: var
  end type
end module lib

subroutine main(res, start, end)
  use lib
  implicit none
  integer, dimension(6) :: res
  integer :: start
  integer :: end
  type(test_type) :: indices
  type(test_type2) :: val
  indices%res=res
  indices%start = start
  indices%end = end
  val%var= indices
  call fun(val)
end subroutine main

subroutine fun(idx)
  use lib
  implicit none
  type(test_type2) :: idx
  idx%var%res(idx%var%start:idx%var%end) = 42
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    size = 6
    res = np.full([size], 42, order="F", dtype=np.int32)
    res[:] = 0
    sdfg(res=res, start=2, end=5)
