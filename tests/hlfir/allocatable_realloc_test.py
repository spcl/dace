"""End-to-end tests for ``ALLOCATABLE`` re-allocation.

Each ``ALLOCATE`` lands as a fresh SDFG transient  --  the first one
keeps the variable's original Fortran name (``x``), every subsequent
one gets a synthetic alias (``x_alloc1``, ``x_alloc2``, ...).  The
intervening ``DEALLOCATE`` is a no-op at the SDFG level.

We don't model pointer aliasing, so this only covers straight-line
``deallocate(x); allocate(x(...))`` re-allocation; branched ALLOCATE
sites (one ``allocate`` per arm of an ``if``) are out of scope.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_realloc_size_change(tmp_path: Path):
    """Allocate ``x(n1)``, deallocate, re-allocate ``x(n2)``, copy
    ``out = x``.  Verifies the second allocation lives under its own
    transient (``x_alloc1``) so ``out`` reflects the second
    allocation's contents."""
    src = """
subroutine main(n1, n2, src1, src2, out)
  integer, intent(in) :: n1, n2
  double precision, intent(in)  :: src1(n1)
  double precision, intent(in)  :: src2(n2)
  double precision, intent(out) :: out(n2)
  double precision, allocatable :: x(:)
  allocate(x(n1))
  x = src1
  deallocate(x)
  allocate(x(n2))
  x = src2
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    # ``x_alloc1`` should appear as a transient  --  the second allocation.
    assert 'x_alloc1' in sdfg.arrays, f"expected x_alloc1 transient, got {list(sdfg.arrays)}"
    assert sdfg.arrays['x_alloc1'].transient
    n1, n2 = 4, 6
    src1 = np.empty(n1, dtype=np.float64, order='F')
    src1[:] = np.arange(1, n1 + 1)
    src2 = np.empty(n2, dtype=np.float64, order='F')
    src2[:] = np.arange(100, 100 + n2)
    out = np.zeros(n2, dtype=np.float64, order='F')
    sdfg(n1=n1, n2=n2, src1=src1, src2=src2, out=out)
    np.testing.assert_array_equal(out, src2)


def test_realloc_same_size(tmp_path: Path):
    """Two ALLOCATEs with the same shape  --  the second still gets its
    own transient.  Without the per-site rebind, leftover writes from
    the first allocation would leak through."""
    src = """
subroutine main(n, src1, src2, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: src1(n)
  double precision, intent(in)  :: src2(n)
  double precision, intent(out) :: out(n)
  double precision, allocatable :: x(:)
  allocate(x(n))
  x = src1
  deallocate(x)
  allocate(x(n))
  x = src2
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    assert 'x_alloc1' in sdfg.arrays
    n = 5
    src1 = np.empty(n, dtype=np.float64, order='F')
    src1[:] = -1.0
    src2 = np.empty(n, dtype=np.float64, order='F')
    src2[:] = np.arange(1, n + 1)
    out = np.zeros(n, dtype=np.float64, order='F')
    sdfg(n=n, src1=src1, src2=src2, out=out)
    np.testing.assert_array_equal(out, src2)
