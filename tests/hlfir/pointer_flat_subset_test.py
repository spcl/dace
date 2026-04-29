"""Pointer-to-flat-subset tests — POINTER rebound to a slice of an
array that came out of ``hlfir-flatten-structs``.

The rewriter (``RewritePointerAssigns.cpp``) collapses
``ptr => target`` rebinds under the strict-no-aliasing assumption.
After ``hlfir-flatten-structs``, struct-member targets like ``s%w``
get rewritten to top-level flat companion declares (``s_w``), so a
rebind chain ``ptr => s%w(2:5)`` reaches the rewriter as
``ptr => s_w(2:5)`` — a section over a flat-companion declare.  The
slice-target arm that handles ``ptr => store(1:n)`` for a top-level
TARGET should handle this case the same way (trace the rebox/embox
chain back to the parent ``hlfir.designate``, forward every read of
the pointer to a designate of the flat companion's slice).

These tests pin the path from a few angles.  Each compares the
SDFG result against an f2py reference so a numeric regression
surfaces instead of an SDFG-builds-but-wrong-result silent failure.

Note on Fortran ``TARGET`` placement: gfortran's f2py rejects the
``target`` attribute on derived-type components (Fortran 2003+ feature
that f2py's harness doesn't accept here), so the tests apply ``target``
to the variable as a whole — Fortran promotes the attribute to every
component automatically.
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


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_pointer_to_full_struct_member(tmp_path: Path):
    """Rank-1 struct member (no slice) — pointer to the whole flat
    companion.  Baseline that the embox(declare) shape works after
    flatten: ``p => s%w`` rebinds onto the flat ``s_w`` declare and
    the rewriter forwards every ``p(i)`` to ``s_w(i)``.
    """
    src = """
module lib
  implicit none
  type t
    real :: w(10)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t), target :: s
  real, pointer :: p(:)
  integer :: i
  do i = 1, 10
    s%w(i) = real(i)
  end do
  p => s%w
  out = p(3) + p(7)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_full_member_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # p(3) = s%w(3) = 3.0; p(7) = s%w(7) = 7.0; sum = 10.0.
    assert out[0] == 10.0


def test_pointer_to_struct_member_slice(tmp_path: Path):
    """Rank-1 struct member, pointer to a triplet section of the
    flat companion.  ``p => s%w(3:7)`` lowers (after flatten) to a
    slice over ``s_w``.  Tests that the slice-target arm threads the
    bounds correctly across the flatten boundary.
    """
    src = """
module lib
  implicit none
  type t
    real :: w(10)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t), target :: s
  real, pointer :: p(:)
  integer :: i
  do i = 1, 10
    s%w(i) = real(i)
  end do
  p => s%w(3:7)
  out = p(1) + p(5)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_member_slice_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # p(1) = s%w(3) = 3.0; p(5) = s%w(7) = 7.0; sum = 10.0.
    assert out[0] == 10.0


def test_pointer_to_2d_member_column(tmp_path: Path):
    """Rank-2 struct member, pointer to a column slice.
    ``p => s%w(:, 3)`` after flatten rebinds onto ``s_w(:, 3)``.
    """
    src = """
module lib
  implicit none
  type t
    real :: w(4, 5)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(4)
  type(t), target :: s
  real, pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      s%w(i, j) = real(10 * j + i)
    end do
  end do
  p => s%w(:, 3)
  do i = 1, 4
    out(i) = p(i)
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_2d_column_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(4, order="F", dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # Column j=3: 31, 32, 33, 34.
    np.testing.assert_array_equal(out, [31.0, 32.0, 33.0, 34.0])


def test_pointer_write_through_to_member_slice(tmp_path: Path):
    """Write through the pointer, read back through the host struct
    member.  Pins that the rebind preserves write-back semantics
    after the flatten rewrite (the flat companion's storage is the
    one ultimately mutated; the struct-member view sees the writes).
    """
    src = """
module lib
  implicit none
  type t
    integer :: w(8)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, intent(out) :: out(8)
  type(t), target :: s
  integer, pointer :: p(:)
  integer :: i
  do i = 1, 8
    s%w(i) = 0
  end do
  p => s%w(3:5)
  do i = 1, 3
    p(i) = 100 + i
  end do
  do i = 1, 8
    out(i) = s%w(i)
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_write_member_slice_ref")
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(8, order="F", dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # s%w = [0, 0, 101, 102, 103, 0, 0, 0] after the writes through p.
    np.testing.assert_array_equal(out, [0, 0, 101, 102, 103, 0, 0, 0])
