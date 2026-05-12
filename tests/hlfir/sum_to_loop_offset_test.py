"""Ported from f2dace/dev:tests/fortran/sum_to_loop_offset_test.py.

Exercises the old frontend's offset-normalisation path for SUM
reductions over array slices.  Under the HLFIR bridge, the same
expressions are lowered as SUM intrinsic calls with the access
chain's offsets handled by the standard memlet machinery.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_sum2loop_1d_without_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res
                    CALL index_test_function(d, res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(7) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d(:))
                    res(2) = SUM(d)
                    res(3) = SUM(d(2:6))

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    size = 7
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == (1 + size) * size / 2
    assert res[1] == (1 + size) * size / 2
    assert res[2] == (2 + size - 1) * (size - 2) / 2


def test_fortran_frontend_sum2loop_1d_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6) :: d
                    double precision, dimension(3) :: res
                    CALL index_test_function(d,res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(2:6) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:))
                    res(3) = SUM(d(3:5))

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    size = 5
    d = np.full([size], 0, order="F", dtype=np.float64)
    for i in range(size):
        d[i] = i + 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == (1 + size) * size / 2
    assert res[1] == (1 + size) * size / 2
    assert res[2] == (2 + size - 1) * (size - 2) / 2


def test_fortran_frontend_arr2loop_2d(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res
                    CALL index_test_function(d,res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(5,3) :: d
                    double precision, dimension(4) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:,:))
                    res(3) = SUM(d(2:4, 2))
                    res(4) = SUM(d(2:4, 2:3))

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    sizes = [5, 3]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 0
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == 105
    assert res[1] == 105
    assert res[2] == 21
    assert res[3] == 45


def test_fortran_frontend_arr2loop_2d_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6,7:10) :: d
                    double precision, dimension(3) :: res
                    CALL index_test_function(d,res)
                    end

                    SUBROUTINE index_test_function(d, res)
                    double precision, dimension(2:6,7:10) :: d
                    double precision, dimension(3) :: res

                    res(1) = SUM(d)
                    res(2) = SUM(d(:,:))
                    res(3) = SUM(d(3:5, 8:9))

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    sizes = [5, 4]
    d = np.full(sizes, 42, order="F", dtype=np.float64)
    cnt = 0
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            d[i, j] = cnt
            cnt += 1
    res = np.full([3], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    assert res[0] == 190
    assert res[1] == 190
    assert res[2] == 57
