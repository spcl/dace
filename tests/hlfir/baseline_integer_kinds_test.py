"""Pinned coverage for Fortran ``INTEGER(KIND=N)`` mappings.

Contract:
  * ``INTEGER(1)`` (1-byte signed) -> ``np.int8``
  * ``INTEGER(2)`` (2-byte signed) -> ``np.int16``
  * ``INTEGER(4)`` (default, 4-byte) -> ``np.int32``  (already covered)
  * ``INTEGER(8)`` (8-byte) -> ``np.int64``  (already covered)

The smaller kinds are exercised here because the wider kinds are
implicit in nearly every other test.  ``INTEGER(1)`` is the byte type
ICON / CloudSC / QE rarely use directly but pops up in mask packing
and bit-stream interchange.

Distinguishes ``INTEGER(1)`` (signed 8-bit ``int8_t``) from MLIR's
``i1`` (1-bit boolean predicate, surfaced as ``bool`` after the
LOGICAL->bool migration lands; today as ``uint8``).  The two are
unrelated despite both involving "1".
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_integer_kind_1_array_copy(tmp_path: Path):
    """Round-trip an ``INTEGER(1)`` array through a copy kernel."""
    src = """
subroutine copy_int1(a, b, n)
  implicit none
  integer, intent(in) :: n
  integer(kind=1), intent(in)    :: a(n)
  integer(kind=1), intent(out)   :: b(n)
  integer :: i
  do i = 1, n
    b(i) = a(i)
  end do
end subroutine copy_int1
"""
    sdfg = build_sdfg(src, tmp_path, name='copy_int1', entry='_QPcopy_int1').build()
    n = 5
    a = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    b = np.zeros(n, dtype=np.int8)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, a)


def test_integer_kind_2_array_copy(tmp_path: Path):
    """Round-trip an ``INTEGER(2)`` array through a copy kernel."""
    src = """
subroutine copy_int2(a, b, n)
  implicit none
  integer, intent(in) :: n
  integer(kind=2), intent(in)    :: a(n)
  integer(kind=2), intent(out)   :: b(n)
  integer :: i
  do i = 1, n
    b(i) = a(i)
  end do
end subroutine copy_int2
"""
    sdfg = build_sdfg(src, tmp_path, name='copy_int2', entry='_QPcopy_int2').build()
    n = 4
    a = np.array([-32768, -1, 0, 32767], dtype=np.int16)
    b = np.zeros(n, dtype=np.int16)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, a)


def test_integer_kind_1_same_kind_addition(tmp_path: Path):
    """``c(i) = a(i) + b(i)`` over ``INTEGER(1)`` arrays  --  same-kind
    addition with no widening literals.  Verifies tasklet arithmetic
    on ``int8`` operands stays in ``int8`` end-to-end."""
    src = """
subroutine add_int1(a, b, c, n)
  implicit none
  integer, intent(in) :: n
  integer(kind=1), intent(in)    :: a(n), b(n)
  integer(kind=1), intent(out)   :: c(n)
  integer :: i
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine add_int1
"""
    sdfg = build_sdfg(src, tmp_path, name='add_int1', entry='_QPadd_int1').build()
    n = 4
    a = np.array([-1, 0, 1, 5], dtype=np.int8)
    b = np.array([10, 20, -3, -1], dtype=np.int8)
    c = np.zeros(n, dtype=np.int8)
    sdfg(a=a, b=b, c=c, n=n)
    np.testing.assert_array_equal(c, (a + b).astype(np.int8))


def test_integer_kind_2_same_kind_multiply(tmp_path: Path):
    """``c(i) = a(i) * b(i)`` over ``INTEGER(2)`` arrays  --  same-kind
    multiplication.  Verifies ``int16`` arithmetic stays in ``int16``."""
    src = """
subroutine mul_int2(a, b, c, n)
  implicit none
  integer, intent(in) :: n
  integer(kind=2), intent(in)    :: a(n), b(n)
  integer(kind=2), intent(out)   :: c(n)
  integer :: i
  do i = 1, n
    c(i) = a(i) * b(i)
  end do
end subroutine mul_int2
"""
    sdfg = build_sdfg(src, tmp_path, name='mul_int2', entry='_QPmul_int2').build()
    n = 4
    a = np.array([2, 100, -50, 7], dtype=np.int16)
    b = np.array([3, 200, -10, 11], dtype=np.int16)
    c = np.zeros(n, dtype=np.int16)
    sdfg(a=a, b=b, c=c, n=n)
    np.testing.assert_array_equal(c, (a * b).astype(np.int16))
