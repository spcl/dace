"""Verbatim port of f2dace/dev:tests/fortran/allocate_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from ported._helpers import xfail

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@xfail("ALLOCATE — DaCe deferred-allocation not supported")
def test_fortran_frontend_basic_allocate(tmp_path):
    src = """
subroutine main(d)
  double precision, allocatable, intent(out) :: d(:, :)
  allocate (d(4, 5))
  d(2, 1) = 5.5
end subroutine main"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 42)
