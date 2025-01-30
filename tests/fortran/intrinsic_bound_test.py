# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder

"""
    Test the implementation of LBOUND/UBOUND functions.
    * Standard-sized arrays.
    * Standard-sized arrays with offsets.
    * Arrays with assumed shape.
    * Arrays with assumed shape - passed externally.
    * Arrays with assumed shape with offsets.
    * Arrays inside structures.
    * Arrays inside structures with local override.
    * Arrays inside structures with multiple layers of indirection.
    * Arrays inside structures with multiple layers of indirection + assumed size.
"""


def test_fortran_frontend_bound():
    test_string = """
                    PROGRAM intrinsic_bound_test
                    implicit none
                    integer, dimension(4,7) :: input
                    integer, dimension(4) :: res
                    CALL intrinsic_bound_test_function(res)
                    end

                    SUBROUTINE intrinsic_bound_test_function(res)
                    integer, dimension(4,7) :: input
                    integer, dimension(4) :: res

                    res(1) = LBOUND(input, 1)
                    res(2) = LBOUND(input, 2)
                    res(3) = UBOUND(input, 1)
                    res(4) = UBOUND(input, 2)

                    END SUBROUTINE intrinsic_bound_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_bound_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [1, 1, 4, 7])


def test_fortran_frontend_bound_offsets():
    test_string = """
                    PROGRAM intrinsic_bound_test
                    implicit none
                    integer, dimension(3:8, 9:12) :: input
                    integer, dimension(4) :: res
                    CALL intrinsic_bound_test_function(res)
                    end

                    SUBROUTINE intrinsic_bound_test_function(res)
                    integer, dimension(3:8, 9:12) :: input
                    integer, dimension(4) :: res

                    res(1) = LBOUND(input, 1)
                    res(2) = LBOUND(input, 2)
                    res(3) = UBOUND(input, 1)
                    res(4) = UBOUND(input, 2)

                    END SUBROUTINE intrinsic_bound_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_bound_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [3, 9, 8, 12])


def test_fortran_frontend_bound_assumed():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'intrinsic_bound_test_function', normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [1, 1, 4, 7])


def test_fortran_frontend_bound_assumed_offsets():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'intrinsic_bound_test_function', normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [1, 1, 4, 7])


def test_fortran_frontend_bound_allocatable_offsets():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'intrinsic_bound_test_function', normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)

    # TODO: This really should not work like this! Reasons follow:
    #  - `input` is not a function argument. It's an entirely local variable that the function allcoates and discards.
    #  - Even if our `allocate()` is does not really allocate, we should set the f2dace variables correctly. Currently
    #    that happens only for the "ALLCOATED" variable.
    sdfg(res=res,
         __f2dace_A_input_var_0_d_0_s_0=4,
         __f2dace_A_input_var_0_d_1_s_1=7,
         __f2dace_OA_input_var_0_d_0_s_0=42,
         __f2dace_OA_input_var_0_d_1_s_1=13)

    assert np.allclose(res, [42, 13, 45, 19])


def test_fortran_frontend_bound_structure():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'test_bounds.intrinsic_bound_test_function',
                                            normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [2, 3, 5, 9])


def test_fortran_frontend_bound_structure_override():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'test_bounds.intrinsic_bound_test_function',
                                            normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [2, 3, 5, 9])


def test_fortran_frontend_bound_structure_recursive():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'test_bounds.intrinsic_bound_test_function',
                                            normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [-1, 0, 2, 3])


@pytest.mark.skip(reason="Needs suport for allocatable + datarefs")
def test_fortran_frontend_bound_structure_recursive_allocatable():
    sources, main = SourceCodeBuilder().add_file("""
MODULE test_types
    IMPLICIT NONE

    TYPE inner_container
        INTEGER, ALLOCATABLE, DIMENSION(:, :) :: inner_data
    END TYPE

    TYPE middle_container
        INTEGER, ALLOCATABLE, DIMENSION(:, :) :: middle_data
        TYPE(inner_container) :: inner
    END TYPE

    TYPE array_container
        INTEGER, ALLOCATABLE, DIMENSION(:, :) :: outer_data
        TYPE(middle_container) :: middle
    END TYPE
END MODULE

MODULE test_bounds
    USE test_types
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE intrinsic_bound_test_function(res)
        IMPLICIT NONE
        TYPE(array_container) :: container
        INTEGER, DIMENSION(4) :: res

        ALLOCATE(container%middle%inner%inner_data(-1:2, 0:3))
        CALL intrinsic_bound_test_function_impl(container, res)
        DEALLOCATE(container%middle%inner%inner_data)

    END SUBROUTINE

    SUBROUTINE intrinsic_bound_test_function_impl(container, res)
        IMPLICIT NONE
        TYPE(array_container), INTENT(IN) :: container
        INTEGER, DIMENSION(4) :: res

        res(1) = LBOUND(container%middle%inner%inner_data, 1)  ! Should return -1
        res(2) = LBOUND(container%middle%inner%inner_data, 2)  ! Should return 0
        res(3) = UBOUND(container%middle%inner%inner_data, 1)  ! Should return 2
        res(4) = UBOUND(container%middle%inner%inner_data, 2)  ! Should return 3
    END SUBROUTINE
END MODULE
""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'test_bounds.intrinsic_bound_test_function',
                                            normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [-1, 0, 2, 3])


if __name__ == "__main__":
    test_fortran_frontend_bound()
    test_fortran_frontend_bound_offsets()
    test_fortran_frontend_bound_assumed()
    test_fortran_frontend_bound_assumed_offsets()
    test_fortran_frontend_bound_allocatable_offsets()
    test_fortran_frontend_bound_structure()
    test_fortran_frontend_bound_structure_override()
    test_fortran_frontend_bound_structure_recursive()
    # FIXME: ALLOCATBLE does not support data refs
    # test_fortran_frontend_bound_structure_recursive_allocatable()
