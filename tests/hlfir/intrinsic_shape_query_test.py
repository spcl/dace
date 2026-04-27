"""SIZE / LBOUND / UBOUND / SHAPE intrinsics — Flang lowers these to
``fir.box_dims`` on the array's box, and the bridge maps each
``fir.box_dims`` result number to the right shape symbol:

* ``#0`` (lower bound) → declared lb (``fir.shape_shift`` 1st operand)
   or Fortran-default ``1`` for plain ``fir.shape``.
* ``#1`` (extent)      → declare's extent operand or the assumed-shape
   synthesised symbol ``<arr>_d<dim>``.
* ``#2`` (stride)      → ``1`` (contiguous).

Each test compares an SDFG run against an f2py / gfortran reference so
the lowering matches Fortran semantics bit-for-bit.
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


def test_size_assumed_shape_2d(tmp_path: Path):
    src = """
subroutine main(arr, out)
  integer, intent(in)  :: arr(:, :)
  integer, intent(out) :: out(3)
  out(1) = SIZE(arr)
  out(2) = SIZE(arr, 1)
  out(3) = SIZE(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "size_assumed_shape_ref")
    n, m = 4, 7
    arr = np.asfortranarray(np.arange(n * m, dtype=np.int32).reshape((n, m)))
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.int32)
    sdfg(arr=arr, out=out, arr_d0=n, arr_d1=m)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [n * m, n, m])


def test_size_explicit_shape(tmp_path: Path):
    src = """
subroutine main(arr, out, n, m)
  integer, intent(in)  :: n, m
  integer, intent(in)  :: arr(n, m)
  integer, intent(out) :: out(3)
  out(1) = SIZE(arr)
  out(2) = SIZE(arr, 1)
  out(3) = SIZE(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "size_explicit_ref")
    n, m = 4, 7
    arr = np.asfortranarray(np.arange(n * m, dtype=np.int32).reshape((n, m)))
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.int32)
    sdfg(arr=arr, out=out, n=n, m=m)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [n * m, n, m])


def test_lbound_assumed_shape(tmp_path: Path):
    """Assumed-shape arrays default to lower bound 1 on every dim — the
    callee receives a plain box without lb metadata.  ``LBOUND`` returns
    1 regardless of what the caller's array looked like."""
    src = """
subroutine main(arr, out)
  integer, intent(in)  :: arr(:, :)
  integer, intent(out) :: out(2)
  out(1) = LBOUND(arr, 1)
  out(2) = LBOUND(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "lbound_assumed_ref")
    n, m = 5, 3
    arr = np.zeros((n, m), order="F", dtype=np.int32)
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.int32)
    sdfg(arr=arr, out=out, arr_d0=n, arr_d1=m)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [1, 1])


def test_lbound_explicit_offset(tmp_path: Path):
    """``dimension(L:U)`` syntax — Flang lowers via ``fir.shape_shift``
    so ``LBOUND(arr, K)`` returns ``L`` (not the default 1)."""
    src = """
subroutine main(arr, out)
  integer, intent(in)  :: arr(20:24, 4)
  integer, intent(out) :: out(2)
  out(1) = LBOUND(arr, 1)
  out(2) = LBOUND(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "lbound_offset_ref")
    arr = np.zeros((5, 4), order="F", dtype=np.int32)
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.int32)
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [20, 1])


def test_ubound_default_lb(tmp_path: Path):
    """Default lower bound 1 → ``UBOUND(arr, K) == SIZE(arr, K)``."""
    src = """
subroutine main(arr, out, n, m)
  integer, intent(in)  :: n, m
  integer, intent(in)  :: arr(n, m)
  integer, intent(out) :: out(2)
  out(1) = UBOUND(arr, 1)
  out(2) = UBOUND(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ubound_default_ref")
    n, m = 6, 9
    arr = np.zeros((n, m), order="F", dtype=np.int32)
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.int32)
    sdfg(arr=arr, out=out, n=n, m=m)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [n, m])


def test_ubound_explicit_offset(tmp_path: Path):
    """``dimension(L:U)`` → ``UBOUND(arr, K) == U`` (lb + size - 1)."""
    src = """
subroutine main(arr, out)
  integer, intent(in)  :: arr(20:24, 7:10)
  integer, intent(out) :: out(2)
  out(1) = UBOUND(arr, 1)
  out(2) = UBOUND(arr, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ubound_offset_ref")
    arr = np.zeros((5, 4), order="F", dtype=np.int32)
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.int32)
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [24, 10])


def test_size_assumed_shape_3d(tmp_path: Path):
    """3-D assumed-shape — exercises the per-dim symbol path with rank>2."""
    src = """
subroutine main(arr, out)
  integer, intent(in)  :: arr(:, :, :)
  integer, intent(out) :: out(4)
  out(1) = SIZE(arr)
  out(2) = SIZE(arr, 1)
  out(3) = SIZE(arr, 2)
  out(4) = SIZE(arr, 3)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "size_3d_ref")
    n, m, p = 3, 4, 5
    arr = np.asfortranarray(np.arange(n * m * p, dtype=np.int32).reshape((n, m, p)))
    out_ref = np.asarray(mod.main(arr), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(4, dtype=np.int32)
    sdfg(arr=arr, out=out, arr_d0=n, arr_d1=m, arr_d2=p)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [n * m * p, n, m, p])
