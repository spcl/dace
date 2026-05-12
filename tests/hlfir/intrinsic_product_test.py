"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_product_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

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


def test_fortran_frontend_product_array_dim(tmp_path):
    """``PRODUCT(d, dim)`` with the explicit DIM= argument.

    The original f2dace port used ``logical, dimension(5)`` for ``d``,
    but Fortran 2018 restricts ``PRODUCT`` to numeric types
    (``INTEGER`` / ``REAL`` / ``COMPLEX``); ``flang-new-21`` correctly
    rejects ``PRODUCT(LOGICAL_array, 1)`` with "bad type LOGICAL(4)".
    Switching ``d`` to ``integer`` keeps the spirit of the test (the
    ``dim`` argument is what we want to exercise) while satisfying
    the standard."""
    src = """
SUBROUTINE intrinsic_product_array_dim_function(d, res)
integer, dimension(5) :: d
integer, dimension(2) :: res

res(1) = PRODUCT(d, 1)

END SUBROUTINE intrinsic_product_array_dim_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_product_array_dim_function').build()

    d = np.array([1, 2, 3, 4, 5], dtype=np.int32, order='F')
    res = np.zeros(2, dtype=np.int32, order='F')
    sdfg(d=d, res=res)
    assert int(res[0]) == 120


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
