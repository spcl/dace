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
    # FIXME: ALLOCATBLE does not support data refs
    # test_fortran_frontend_bound_structure_recursive_allocatable()
