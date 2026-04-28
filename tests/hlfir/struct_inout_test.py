"""Pinned coverage for the interstate-edge ``[0]`` subscript path.

When a Fortran scalar dummy carries ``intent(inout)`` (or ``intent(out)``),
the Scalar I/O convention surfaces it on the SDFG signature as a
length-1 ``Array`` (so the caller has a writable slot).  The bridge's
symbol-staging path emits an interstate-edge assignment from that
dummy into a struct-field-promoted symbol — but the C ABI binds a
length-1 Array as ``T*``, so the bare-name RHS would render as
``indices_end = endidx`` (``int = int*``) and fail to compile.

The fix in ``builder/emit_cfg.py::emit_assign`` checks the SDFG
descriptor's type+shape: when it's an ``Array`` with ``shape=(1,)``,
the RHS is rewritten to ``<name>[0]`` so the codegen sees a scalar
value.  This file pins that behaviour with a paired test against the
``intent(in)`` companion in ``ported/struct_test.py`` (where the
descriptor is a true ``Scalar`` and the bare name is correct).
"""
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


def test_struct_with_inout_scalar_dummies(tmp_path):
    """Same struct + array-section pattern as
    ``ported/struct_test.py::test_fortran_struct`` but the scalar
    dummies carry ``intent(inout)`` so they surface as length-1
    Arrays on the SDFG signature.  The interstate-edge assignment
    that promotes ``indices%start`` and ``indices%end`` to symbols
    must subscript the RHS as ``startidx[0]`` / ``endidx[0]``."""
    src = """
module lib
  implicit none
  type test_type
    integer :: start
    integer :: end
  end type
end module lib

subroutine main(res, startidx, endidx)
  use lib
  implicit none
  integer, dimension(6)     :: res
  integer, intent(inout)    :: startidx
  integer, intent(inout)    :: endidx
  type(test_type) :: indices
  indices%start = startidx
  indices%end = endidx
  call fun(res, indices)
end subroutine main

subroutine fun(res, idx)
  use lib
  implicit none
  integer, dimension(6) :: res
  type(test_type) :: idx
  res(idx%start:idx%end) = 42
end subroutine fun
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()

    res = np.zeros(6, order="F", dtype=np.int32)
    # ``intent(inout)`` scalar dummies → length-1 Arrays on the
    # signature; pass them boxed.
    startidx = np.array([2], dtype=np.int32)
    endidx = np.array([5], dtype=np.int32)
    sdfg(res=res, startidx=startidx, endidx=endidx)

    # Section ``res(2:5) = 42`` writes positions 2..5 (1-based) → 1..4 in
    # 0-based numpy.  Positions 0 and 5 stay zero.
    assert res[0] == 0
    assert all(res[1:5] == 42)
    assert res[5] == 0
