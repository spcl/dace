"""ALLOCATED intrinsic  --  Flang lowers ``ALLOCATED(arr)`` to
``box_addr(load arr_box) != 0`` (a null-pointer check on the
allocatable's heap descriptor).  The bridge:

* registers a companion ``<arr>_allocated`` int32 SDFG symbol per
  allocatable (writes go on interstate edges so DaCe enforces ordering
  across reads);
* emits ``<arr>_allocated = 1`` at every detected ALLOCATE site
  (``fir.store(fir.embox(fir.allocmem))``);
* emits ``<arr>_allocated = 0`` at every standalone ``fir.freemem``
  (DEALLOCATE) and at module-walk start (initial-zero).

Each test compares against an f2py / gfortran reference so the
state-tracking matches Fortran semantics exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, name: str = "main"):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name).build()


def test_allocated_initially_false(tmp_path: Path):
    """An allocatable that's never been ALLOCATEd reads as 0."""
    src = """
subroutine main(out)
  integer, allocatable :: data(:)
  integer, intent(out) :: out
  out = MERGE(1, 0, ALLOCATED(data))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "alloc_initial_ref")
    out_ref = int(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert out[0] == out_ref == 0


def test_allocated_after_allocate(tmp_path: Path):
    """``ALLOCATE(arr(N))`` flips ``ALLOCATED(arr)`` to 1."""
    src = """
subroutine main(out)
  integer, allocatable :: data(:)
  integer, intent(out) :: out
  ALLOCATE(data(8))
  out = MERGE(1, 0, ALLOCATED(data))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "alloc_after_ref")
    out_ref = int(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert out[0] == out_ref == 1


def test_allocated_after_deallocate(tmp_path: Path):
    """``DEALLOCATE`` flips ``ALLOCATED`` back to 0."""
    src = """
subroutine main(out)
  integer, allocatable :: data(:)
  integer, intent(out) :: out
  ALLOCATE(data(4))
  DEALLOCATE(data)
  out = MERGE(1, 0, ALLOCATED(data))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "alloc_dealloc_ref")
    out_ref = int(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert out[0] == out_ref == 0


def test_allocated_state_sequence(tmp_path: Path):
    """Read ALLOCATED before, between, and after ALLOCATE/DEALLOCATE
    in the same subroutine  --  exercises the per-state ordering
    guarantees that motivate using a SYMBOL (not a transient scalar)
    for ``<arr>_allocated``."""
    src = """
subroutine main(out)
  integer, allocatable :: data(:)
  integer, intent(out) :: out(3)
  out(1) = MERGE(1, 0, ALLOCATED(data))
  ALLOCATE(data(6))
  out(2) = MERGE(1, 0, ALLOCATED(data))
  DEALLOCATE(data)
  out(3) = MERGE(1, 0, ALLOCATED(data))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "alloc_seq_ref")
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [0, 1, 0])


def test_allocated_two_arrays_independent(tmp_path: Path):
    """Two allocatables track state independently  --  the ``a_allocated``
    and ``b_allocated`` symbols mustn't collide."""
    src = """
subroutine main(out)
  integer, allocatable :: a(:), b(:)
  integer, intent(out) :: out(2)
  ALLOCATE(a(3))
  out(1) = MERGE(1, 0, ALLOCATED(a))
  out(2) = MERGE(1, 0, ALLOCATED(b))
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "alloc_two_ref")
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [1, 0])
