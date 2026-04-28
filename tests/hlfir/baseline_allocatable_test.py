"""Baseline HLFIR coverage — Fortran ``ALLOCATABLE`` standalone arrays.

Pinned coverage (already supported by the bridge):
  * Local allocatable, ``ALLOCATE`` + element write/read.
  * Allocatable round-trip (``allocate; x = src; out = x; deallocate``).

What this file does NOT yet exercise (deferred):
  * Allocatable / pointer struct MEMBERS (``type t :: real, allocatable :: w(:)``)
    — covered by ``derived_type_test.py``, currently xfailed pending the
    "allocatable + pointer struct member support" feature.
  * Array-of-struct + allocatable member with per-instance shapes
    (the padding-to-max story).  Saved memory pins the design:
    ``A_csr_rowptr[N, X1]`` with X1 = max over instances; bindings
    layer pads-to-max at the SDFG boundary.

The baseline assumes the program correctly tracks allocation state —
no runtime ``ALLOCATED`` checks are inserted by the bridge.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_allocate_then_element_writes(tmp_path: Path):
    """Local ``integer, allocatable :: data(:)``, allocate then per-element
    write through a DO loop, then read back."""
    src = """
subroutine main(n, out)
  implicit none
  integer, intent(in)  :: n
  integer, intent(out) :: out(3)
  integer, allocatable :: data(:)
  integer :: i
  allocate(data(n))
  do i = 1, n
    data(i) = i * 10
  end do
  out(1) = data(1)
  out(2) = data(n / 2 + 1)
  out(3) = data(n)
  deallocate(data)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    out = np.zeros(3, order="F", dtype=np.int32)
    sdfg(n=6, out=out)
    np.testing.assert_array_equal(out, [10, 40, 60])


def test_allocate_whole_array_copy_roundtrip(tmp_path: Path):
    """``allocate(x(n)); x = src; out = x; deallocate(x)`` — verifies the
    allocatable round-trip end to end."""
    src = """
subroutine main(n, src, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: src(n)
  double precision, intent(out) :: out(n)
  double precision, allocatable :: x(:)
  allocate(x(n))
  x = src
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 8
    src_arr = np.arange(1.0, n + 1.0, dtype=np.float64)
    src_arr = np.asfortranarray(src_arr)
    out = np.zeros(n, order='F', dtype=np.float64)
    sdfg(n=n, src=src_arr, out=out)
    np.testing.assert_array_equal(out, src_arr)
