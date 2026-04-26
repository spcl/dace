"""Verbatim port of f2dace/dev:tests/fortran/while_test.py.

Note: f2dace's while_test wraps the test in a PROGRAM + CALL idiom; the
HLFIR frontend runs on the subroutine directly (cross-subroutine PROGRAM
lowering is not yet implemented), so the verbatim port targets the
``while_test_function`` subroutine."""
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


def test_fortran_frontend_while(tmp_path):
    src = """
SUBROUTINE while_test_function(d, res)
real, dimension(2) :: d
real, dimension(2) :: res

integer :: i
i = 0
res(1) = d(1) * 2
do while (i < 10)
    res(1) = res(1) + 1
    i = i + 1
end do

END SUBROUTINE while_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='while_test_function').build()

    inp = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=inp, res=res, i=0)
    assert np.allclose(res, [94, 42])
