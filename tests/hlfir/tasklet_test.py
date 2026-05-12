"""Verbatim port of f2dace/dev:tests/fortran/tasklet_test.py.

The PROGRAM wrapper is stripped — FaCe runs on the SUBROUTINE directly.
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


def test_fortran_frontend_tasklet(tmp_path):
    src = """
SUBROUTINE tasklet_test_function(d,res)
real, dimension(2) :: d
real, dimension(2) :: res
real :: temp


integer :: i
i=1
temp = 88
d(1)=d(1)*2
temp = MIN(d(i), temp)
res(1) = temp + 10

END SUBROUTINE tasklet_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='tasklet_test_function').build()

    inp = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=inp, res=res, i=0)
    assert np.allclose(res, [94, 42])
