"""Baseline HLFIR coverage — Fortran ``POINTER`` rebinding under the
strict-no-aliasing assumption.

The bridge collapses ``ptr => target`` rebinds in
``hlfir-rewrite-pointer-assigns``: every read or write of the pointer
becomes an access to the rebind target's storage.  The pass emits a
warning per firing so callers see the no-alias assumption — Fortran
allows aliased pointer access, this collapse is unsafe if the program
relies on it.

Pinned coverage:
  * Pointer to a scalar struct field (``tmp => s%a``).
  * Pointer to a scalar local (``tmp => x``).
  * Both reads and writes through the pointer (``tmp = 13``,
    ``r = func(tmp)``).
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

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_pointer_to_scalar_local(tmp_path: Path):
    """``tmp => x; tmp = 13; res = tmp + 1`` — pointer to a scalar local."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x
  integer, pointer :: tmp
  x = 0
  tmp => x
  tmp = 13
  out = tmp + 1
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_scalar_local")
    out_ref = mod.main()
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert int(out[0]) == int(out_ref) == 14


def test_pointer_to_struct_scalar_field(tmp_path: Path):
    """``tmp => s%a; tmp = 13`` — pointer rebound onto a scalar struct field.

    flatten-structs runs first and replaces ``s%a`` with a flat ``s_a``
    declare; the rewrite-pointer-assigns pass then traces the rebind's
    target through the box+embox chain to ``s_a`` and replaces every
    pointer use with the flat declare.
    """
    src = """
module lib
  implicit none
  type simple_type
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(inout) :: d(2)
  type(simple_type), target :: s
  integer, pointer :: tmp
  s%a = 0
  tmp => s%a
  tmp = 13
  d(1) = real(s%a)
  d(2) = real(tmp)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_struct_field")
    d_ref = np.zeros(2, order="F", dtype=np.float32)
    mod.main(d_ref)
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [13.0, 13.0])
