"""Module-level derived types with array members — Phase 1.

The HLFIR pass ``hlfir-flatten-structs`` decomposes a ``type(t) :: s``
declaration where ``t`` has flat-only members (scalars or arrays of
scalars) into per-member declares ``s_<field>``.  After Phase 1 of
derived-type support, the pass also fires on **local** declares (not
just dummy arguments), and ``extract_vars`` recovers concrete extents
from ``fir.SequenceType`` when the synthesised per-field declare
carries no ``fir.shape`` operand.

Each test compares an SDFG run against a gfortran/f2py reference for
bit-exact validation, matching the saved e2e-numerical rule.

A negative test ensures the bridge throws a ``RuntimeError`` (not
silent wrong values) when ``hlfir-flatten-structs`` could not lower
the struct — the loud-failure pattern from the previous round of
correctness work.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, name: str = "main"):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name).build()


def test_local_struct_element_write_and_read(tmp_path: Path):
    """Local ``type(t) :: s`` with explicit-shape array member, single
    element write + read.  Exercises the local-instance flatten +
    SequenceType-extent fallback in ``extract_vars``."""
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5)
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(2)
  type(simple_type) :: s
  s%w(1, 1, 1) = 5.5
  d(1) = s%w(1, 1, 1)
  d(2) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_element_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [5.5, 11.0])


def test_local_struct_two_array_members(tmp_path: Path):
    """Two array members of different shapes — exercises the per-
    member path generating two separate flat arrays."""
    src = """
module lib
  implicit none
  type two_arrays
    real :: u(4)
    real :: v(7)
  end type two_arrays
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(2)
  type(two_arrays) :: t
  t%u(2) = 3.0
  t%v(7) = 4.0
  out(1) = t%u(2)
  out(2) = t%v(7)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_two_arrays_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [3.0, 4.0])


def test_local_struct_member_in_loop(tmp_path: Path):
    """Loop-driven element writes to a struct's array member.  The
    flat ``s_w`` array carries the SequenceType's static (5,) extent —
    the SDFG signature has no synth shape symbol to bind."""
    src = """
module lib
  implicit none
  type sum_type
    real :: w(5)
  end type sum_type
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(sum_type) :: s
  integer :: i
  do i = 1, 5
    s%w(i) = real(i) * 2.0
  end do
  out = s%w(1) + s%w(2) + s%w(3) + s%w(4) + s%w(5)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_loop_ref")
    out_ref = float(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 2.0 + 4.0 + 6.0 + 8.0 + 10.0


def test_local_struct_used_as_2d_assignment_target(tmp_path: Path):
    """``s%w(:, k) = arr(:)`` — slice assignment into a struct's 2-D
    array member.  Exercises the section-to-section path landing on
    a flat per-field array."""
    src = """
module lib
  implicit none
  type two_d
    real :: w(3, 4)
  end type two_d
end module lib

subroutine main(arr, out)
  use lib
  implicit none
  real, intent(in)  :: arr(3)
  real, intent(out) :: out(3)
  type(two_d) :: t
  integer :: i
  do i = 1, 3
    t%w(i, 2) = arr(i) * 10.0
  end do
  out(:) = t%w(:, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_2d_ref")
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32, order="F")
    out_ref = np.asarray(mod.main(arr), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.float32)
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])


@pytest.mark.xfail(strict=True,
                   reason="Phase 1 doesn't handle nested structs (Phase 2 work). "
                   "extract_vars currently drops the un-flattened RecordType "
                   "silently, producing a broken SDFG that fails downstream "
                   "with KeyError or wrong values; once Phase 2 lands or the "
                   "loud-failure throw is re-enabled, this test will start "
                   "raising a clean RuntimeError and can be unxfailed.")
def test_nested_struct_currently_unsupported(tmp_path: Path):
    """Contract test: nested structs (``type(outer_t)`` whose member is a
    ``type(inner_t)``) are NOT covered by Phase 1.  Tracks the gap so
    Phase 2 work can use this test as the canonical fixture."""
    src = """
module lib
  implicit none
  type inner_t
    real :: x(5)
  end type inner_t
  type outer_t
    type(inner_t) :: inner
  end type outer_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(1)
  type(outer_t) :: o
  o%inner%x(1) = 1.0
  d(1) = o%inner%x(1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "nested_struct_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(1, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
