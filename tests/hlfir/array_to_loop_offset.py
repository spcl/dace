"""Ported from f2dace/dev:tests/fortran/array_to_loop_offset.py.

The old Fortran frontend had a `normalize_offsets` toggle on
``create_sdfg_from_string``; under HLFIR the bridge handles
offset normalisation uniformly through the
``offset_<arr>_d<i>`` symbol convention (declared in
``builder/descriptors.py`` and folded by ``sdfg.specialize``).
The two paths from the old test (with vs without normalisation)
collapse to one path here.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_arr2loop_without_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,3) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,3) :: d
                    integer :: i

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    assert len(sdfg.arrays['d'].shape) == 2
    assert sdfg.arrays['d'].shape[0] == 5
    assert sdfg.arrays['d'].shape[1] == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 6):
        for j in range(1, 4):
            assert a[i - 1, j - 1] == i * 2


def test_fortran_frontend_arr2loop_1d_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(2:6) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(2:6) :: d

                    d(:) = 5

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    assert len(sdfg.arrays['d'].shape) == 1
    assert sdfg.arrays['d'].shape[0] == 5

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for v in a:
        assert v == 5


def test_fortran_frontend_arr2loop_2d_offset(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,7:9) :: d
                    integer :: i

                    do i=1,5
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    assert len(sdfg.arrays['d'].shape) == 2
    assert sdfg.arrays['d'].shape[0] == 5
    assert sdfg.arrays['d'].shape[1] == 3

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 6):
        for j in range(0, 3):
            assert a[i - 1, j] == i * 2


def test_fortran_frontend_arr2loop_2d_offset2(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,7:9) :: d

                    d(:,:) = 43

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0, 5):
        for j in range(0, 3):
            assert a[i, j] == 43


def test_fortran_frontend_arr2loop_2d_offset3(tmp_path):
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(5,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(5,7:9) :: d

                    d(2:4, 7:8) = 43

                    END SUBROUTINE index_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='index_offset_test', entry='_QPindex_test_function').build()

    a = np.full([5, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(1, 4):
        for j in range(0, 2):
            assert a[i, j] == 43
        for j in range(2, 3):
            assert a[i, j] == 42
    for i in [0, 4]:
        for j in range(0, 3):
            assert a[i, j] == 42
