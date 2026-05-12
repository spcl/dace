"""Verbatim port of f2dace/dev:tests/fortran/ifcycle_test.py."""
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


def test_fortran_frontend_if_cycle(tmp_path):
    src = """
subroutine main(d)
  double precision d(4)
  integer :: i
  do i = 1, 4
    if (i .eq. 2) cycle
    d(i) = 5.5
  end do
  if (d(2) .eq. 42) d(2) = 6.5
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0)
    assert (a[0] == 5.5)
    assert (a[1] == 6.5)
    assert (a[2] == 5.5)


def test_fortran_frontend_if_nested_cycle(tmp_path):
    src = """
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
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a, i=0, j=0, stop=0, start=0, count=0)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 6.5)
    assert (a[2, 0] == 42)
