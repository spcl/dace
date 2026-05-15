"""Pinned coverage for the bridge's LOGICAL type mapping.

Contract: Fortran ``LOGICAL(KIND=N)`` (any kind) surfaces on the SDFG
signature as ``np.bool_`` (= C++ ``bool``, 1 byte).  Callers hand a
NumPy bool array; element-wise boolean ops in tasklets render as
``bool`` operations directly  --  no ``(x != 0)`` truthiness coercion
needed.  The caller-side bindings wrapper translates between the
original Fortran ``LOGICAL(KIND=N)`` image (e.g. 4-byte ``int32``
with ``-1``/``0`` encoding) and the SDFG's bool layout at the
Fortran boundary.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_logical_array_copy_in_copy_out_roundtrip(tmp_path: Path):
    """``b = a`` over LOGICAL arrays.  Input handed as ``np.bool_``;
    output read back as ``np.bool_``; bit-exact equality required."""
    src = """
subroutine roundtrip(a, b, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: a(n)
  logical, intent(out) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = a(i)
  end do
end subroutine roundtrip
"""
    sdfg = build_sdfg(src, tmp_path, name='roundtrip', entry='_QProundtrip').build()

    n = 5
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, a)


def test_logical_invert_per_element(tmp_path: Path):
    """``b(i) = .not. a(i)``  --  per-element logical NOT over a LOGICAL
    array.  Verifies the bridge lowers boolean unary ops through the
    bool-typed SDFG path."""
    src = """
subroutine not_kernel(a, b, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: a(n)
  logical, intent(out) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = .not. a(i)
  end do
end subroutine not_kernel
"""
    sdfg = build_sdfg(src, tmp_path, name='not_kernel', entry='_QPnot_kernel').build()

    n = 5
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, np.logical_not(a))


def test_logical_array_inplace_invert_roundtrip(tmp_path: Path):
    """In-place ``mask = .not. mask`` over a LOGICAL array.  The dummy
    is ``intent(inout)`` so the caller's buffer is read AND written;
    the SDFG signature exposes it as a non-transient ``np.bool_`` Array.
    Round-trip pins both directions of the data flow: caller's input
    pattern is read by the kernel, inverted in place, and the inverted
    pattern observed in the caller's buffer."""
    src = """
subroutine invert_in_place(mask, n)
  implicit none
  integer, intent(in)    :: n
  logical, intent(inout) :: mask(n)
  integer :: i
  do i = 1, n
    mask(i) = .not. mask(i)
  end do
end subroutine invert_in_place
"""
    sdfg = build_sdfg(src, tmp_path, name='invert_in_place', entry='_QPinvert_in_place').build()

    n = 6
    original = np.array([True, False, True, True, False, True], dtype=np.bool_)
    mask = original.copy()
    sdfg(mask=mask, n=n)
    # SDFG read the original pattern, wrote back the inverted one.
    np.testing.assert_array_equal(mask, np.logical_not(original))
    # Symmetry: invoking again restores the original.
    sdfg(mask=mask, n=n)
    np.testing.assert_array_equal(mask, original)
