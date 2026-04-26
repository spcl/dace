"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_product_test.py."""
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


def test_fortran_frontend_product_array(tmp_path):
    src = """
SUBROUTINE intrinsic_product_array_function(d, res)
double precision, dimension(7) :: d
double precision, dimension(3) :: res

res(1) = PRODUCT(d)
res(2) = PRODUCT(d(:))
res(3) = PRODUCT(d(2:5))

END SUBROUTINE intrinsic_product_array_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_product_array_function').build()

    size = 7
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == np.prod(d)
    assert res[1] == np.prod(d)
    assert res[2] == np.prod(d[1:5])


@xfail("PRODUCT(d, dim) — DIM= argument not yet lowered")
def test_fortran_frontend_product_array_dim(tmp_path):
    src = """
SUBROUTINE intrinsic_product_array_dim_function(d, res)
logical, dimension(5) :: d
logical, dimension(2) :: res

res(1) = PRODUCT(d, 1)

END SUBROUTINE intrinsic_product_array_dim_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_product_array_dim_function').build()


def test_fortran_frontend_product_2d(tmp_path):
    src = """
SUBROUTINE intrinsic_product_2d_test_function(d, res)
double precision, dimension(5,3) :: d
double precision, dimension(4) :: res

res(1) = PRODUCT(d)
res(2) = PRODUCT(d(:,:))
res(3) = PRODUCT(d(2:4, 2))
res(4) = PRODUCT(d(2:4, 2:3))

END SUBROUTINE intrinsic_product_2d_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_product_2d_test_function').build()

    sizes = [5, 3]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 1
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == np.prod(d)
    assert res[1] == np.prod(d)
    assert res[2] == np.prod(d[1:4, 1])
    assert res[3] == np.prod(d[1:4, 1:3])
