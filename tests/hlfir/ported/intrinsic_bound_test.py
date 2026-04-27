"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_bound_test.py."""
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


def test_fortran_frontend_bound(tmp_path):
    src = """
SUBROUTINE intrinsic_bound_test_function(res)
integer, dimension(4,7) :: input
integer, dimension(4) :: res

res(1) = LBOUND(input, 1)
res(2) = LBOUND(input, 2)
res(3) = UBOUND(input, 1)
res(4) = UBOUND(input, 2)

END SUBROUTINE intrinsic_bound_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [1, 1, 4, 7])


def test_fortran_frontend_bound_offsets(tmp_path):
    src = """
SUBROUTINE intrinsic_bound_test_function(res)
integer, dimension(3:8, 9:12) :: input
integer, dimension(4) :: res

res(1) = LBOUND(input, 1)
res(2) = LBOUND(input, 2)
res(3) = UBOUND(input, 1)
res(4) = UBOUND(input, 2)

END SUBROUTINE intrinsic_bound_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [3, 9, 8, 12])


def test_fortran_frontend_bound_assumed(tmp_path):
    src = """
MODULE intrinsic_bound_interfaces
    INTERFACE
        SUBROUTINE intrinsic_bound_test_function2(input, res)
            integer, dimension(:,:) :: input
            integer, dimension(4) :: res
        END SUBROUTINE intrinsic_bound_test_function2
    END INTERFACE
END MODULE

SUBROUTINE intrinsic_bound_test_function(res)
USE intrinsic_bound_interfaces
implicit none
integer, dimension(4,7) :: input
integer, dimension(4) :: res

CALL intrinsic_bound_test_function2(input, res)

END SUBROUTINE intrinsic_bound_test_function

SUBROUTINE intrinsic_bound_test_function2(input, res)
integer, dimension(:,:) :: input
integer, dimension(4) :: res

res(1) = LBOUND(input, 1)
res(2) = LBOUND(input, 2)
res(3) = UBOUND(input, 1)
res(4) = UBOUND(input, 2)

END SUBROUTINE intrinsic_bound_test_function2
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_bound_test_function',
                      entry='_QPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    # ``input`` is declared 4x7 in the outer subroutine and passed to
    # an interface that takes ``dimension(:,:)`` — the bridge surfaces
    # the assumed-shape extents on the SDFG signature so the caller
    # binds them to the actual dims.
    sdfg(res=res, input_d0=4, input_d1=7)

    assert np.allclose(res, [1, 1, 4, 7])


def test_fortran_frontend_bound_assumed_offsets(tmp_path):
    src = """
MODULE intrinsic_bound_interfaces
    INTERFACE
        SUBROUTINE intrinsic_bound_test_function2(input, res)
            integer, dimension(:,:) :: input
            integer, dimension(4) :: res
        END SUBROUTINE intrinsic_bound_test_function2
    END INTERFACE
END MODULE

SUBROUTINE intrinsic_bound_test_function(res)
USE intrinsic_bound_interfaces
implicit none
integer, dimension(42:45,13:19) :: input
integer, dimension(4) :: res

CALL intrinsic_bound_test_function2(input, res)

END SUBROUTINE intrinsic_bound_test_function

SUBROUTINE intrinsic_bound_test_function2(input, res)
integer, dimension(:,:) :: input
integer, dimension(4) :: res

res(1) = LBOUND(input, 1)
res(2) = LBOUND(input, 2)
res(3) = UBOUND(input, 1)
res(4) = UBOUND(input, 2)

END SUBROUTINE intrinsic_bound_test_function2
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_bound_test_function',
                      entry='_QPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    # Outer ``input`` is declared ``dimension(42:45, 13:19)`` (4 elements
    # in dim0, 7 in dim1, with non-default lower bounds 42 / 13).  When
    # passed to an INTERFACE that takes ``dimension(:,:)`` the assumed-
    # shape callee re-bases everything to lb=1 — Fortran spec — so the
    # callee's ``LBOUND`` returns 1 and ``UBOUND`` returns the extent.
    sdfg(res=res, input_d0=4, input_d1=7)

    assert np.allclose(res, [1, 1, 4, 7])


@xfail("ALLOCATABLE + INTERFACE not yet lowered")
def test_fortran_frontend_bound_allocatable_offsets(tmp_path):
    src = """
MODULE intrinsic_bound_interfaces
    INTERFACE
        SUBROUTINE intrinsic_bound_test_function3(input, res)
            integer, allocatable, dimension(:,:) :: input
            integer, dimension(4) :: res
        END SUBROUTINE intrinsic_bound_test_function3
    END INTERFACE
END MODULE

SUBROUTINE intrinsic_bound_test_function(res)
USE intrinsic_bound_interfaces
implicit none
integer, allocatable, dimension(:,:) :: input
integer, dimension(4) :: res

allocate(input(42:45, 13:19))
CALL intrinsic_bound_test_function3(input, res)
deallocate(input)

END SUBROUTINE intrinsic_bound_test_function

SUBROUTINE intrinsic_bound_test_function3(input, res)
integer, allocatable, dimension(:,:) :: input
integer, dimension(4) :: res

res(1) = LBOUND(input, 1)
res(2) = LBOUND(input, 2)
res(3) = UBOUND(input, 1)
res(4) = UBOUND(input, 2)

END SUBROUTINE intrinsic_bound_test_function3
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_bound_test_function',
                      entry='_QPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)

    sdfg(res=res)

    assert np.allclose(res, [42, 13, 45, 19])


def test_fortran_frontend_bound_structure(tmp_path):
    src = """
MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        INTEGER, DIMENSION(2:5, 3:9) :: data
    END TYPE array_container
END MODULE

MODULE test_bounds
    USE test_types
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE intrinsic_bound_test_function( res)
        TYPE(array_container) :: container
        INTEGER, DIMENSION(4) :: res

        CALL intrinsic_bound_test_function_impl(container, res)
    END SUBROUTINE

    SUBROUTINE intrinsic_bound_test_function_impl(container, res)
        TYPE(array_container), INTENT(IN) :: container
        INTEGER, DIMENSION(4) :: res

        res(1) = LBOUND(container%data, 1)  ! Should return 2
        res(2) = LBOUND(container%data, 2)  ! Should return 3
        res(3) = UBOUND(container%data, 1)  ! Should return 5
        res(4) = UBOUND(container%data, 2)  ! Should return 9
    END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src,
                      tmp_path,
                      name='intrinsic_bound_test_function',
                      entry='_QMtest_boundsPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [2, 3, 5, 9])


def test_fortran_frontend_bound_structure_override(tmp_path):
    src = """
MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        INTEGER, DIMENSION(2:5, 3:9) :: data
    END TYPE array_container
END MODULE

MODULE test_bounds
    USE test_types
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE intrinsic_bound_test_function( res)
        TYPE(array_container) :: container
        INTEGER, DIMENSION(4) :: res

        CALL intrinsic_bound_test_function_impl(container, res)
    END SUBROUTINE

    SUBROUTINE intrinsic_bound_test_function_impl(container, res)
        TYPE(array_container), INTENT(IN) :: container
        INTEGER, DIMENSION(4) :: res
        ! if we handle the refs correctly, this override won't fool us
        integer, dimension(3, 10) :: data

        res(1) = LBOUND(container%data, 1)  ! Should return 2
        res(2) = LBOUND(container%data, 2)  ! Should return 3
        res(3) = UBOUND(container%data, 1)  ! Should return 5
        res(4) = UBOUND(container%data, 2)  ! Should return 9
    END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src,
                      tmp_path,
                      name='intrinsic_bound_test_function',
                      entry='_QMtest_boundsPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [2, 3, 5, 9])


def test_fortran_frontend_bound_structure_recursive(tmp_path):
    src = """
MODULE test_types
    IMPLICIT NONE

    TYPE inner_container
        INTEGER, DIMENSION(-1:2, 0:3) :: inner_data
    END TYPE

    TYPE middle_container
        INTEGER, DIMENSION(2:5, 3:9) :: middle_data
        TYPE(inner_container) :: inner
    END TYPE

    TYPE array_container
        INTEGER, DIMENSION(0:3, -2:4) :: outer_data
        TYPE(middle_container) :: middle
    END TYPE
END MODULE

MODULE test_bounds
    USE test_types
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE intrinsic_bound_test_function( res)
        TYPE(array_container) :: container
        INTEGER, DIMENSION(4) :: res

        CALL intrinsic_bound_test_function_impl(container, res)
    END SUBROUTINE

    SUBROUTINE intrinsic_bound_test_function_impl(container, res)
        TYPE(array_container), INTENT(IN) :: container
        INTEGER, DIMENSION(4) :: res

        res(1) = LBOUND(container%middle%inner%inner_data, 1)   ! Should return -1
        res(2) = LBOUND(container%middle%inner%inner_data, 2)  ! Should return 0
        res(3) = UBOUND(container%middle%inner%inner_data, 1)  ! Should return 2
        res(4) = UBOUND(container%middle%inner%inner_data, 2)  ! Should return 3
    END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src,
                      tmp_path,
                      name='intrinsic_bound_test_function',
                      entry='_QMtest_boundsPintrinsic_bound_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [-1, 0, 2, 3])
