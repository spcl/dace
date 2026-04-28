"""Verbatim port of f2dace/dev:tests/fortran/nested_array_test.py."""
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


@xfail('indices(...) used as array index — LazyFunction subtraction')
def test_fortran_frontend_nested_array_access(tmp_path):
    src = """
subroutine main(d)
  double precision d(4)
  integer test(3, 3, 3)
  integer indices(3, 3, 3)
  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_fortran_frontend_nested_array_access_pointer_args_1(tmp_path):
    src = """
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - But the function actually does not read from the input pointers, but immediately repoint them to internal arrays.
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  integer, target :: internal_test(3, 4, 5)
  integer, target :: internal_indices(3, 4, 5)
  indices => internal_test
  indices => internal_indices
  internal_indices(1, 1, 1) = 2
  internal_indices(1, 1, 2) = 3
  internal_indices(1, 1, 3) = 1
  internal_test(internal_indices(1, 1, 1), internal_indices(1, 1, 2), internal_indices(1, 1, 3)) = 2
  d(internal_test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    indices = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    sdfg(d=d, test=test, indices=indices)
    assert np.allclose(d, [42, 5.5, 42, 42])


@xfail("POINTER not lowered")
def test_fortran_frontend_nested_array_access_pointer_args_2(tmp_path):
    src = """
subroutine main(d, test, indices)
  implicit none
  double precision, intent(inout) :: d(4)

  ! WARNING: There are some caveats about these arguments.
  ! - SDFG will treat these pointer args as **allocatable array args**, and will generate the extra symbols for them.
  ! - This time, the function will actually write and read from those arrays (although igoring their initial values).
  integer, pointer, intent(inout) :: test(:, :, :)
  integer, pointer, intent(inout) :: indices(:, :, :)

  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([4], 42, order="F", dtype=np.float64)
    test = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    indices = np.full([3, 4, 5], 42, order="F", dtype=np.int32)
    sdfg(d=d, test=test, indices=indices)
    assert np.allclose(d, [42, 5.5, 42, 42])
