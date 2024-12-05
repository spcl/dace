# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser
from tests.fortran.fortran_test_helper import SourceCodeBuilder, create_singular_sdfg_from_string

"""
    Test the implementation of LBOUND/UBOUND functions.
    * Standard-sized arrays.
    * Standard-sized arrays with offsets.
    * Arrays with assumed shape.
    * Arrays with assumed shape - passed externally.
    * Arrays with assumed shape with offsets.
    * Arrays inside structures (TODO).
    * Arrays inside structures with local override (TODO).
    * Arrays inside structures with multiple layers of indirection (TODO).
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

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_bound_test", False)
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

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_bound_test", False)
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
    print(res)

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
    sdfg(
        res=res,
        __f2dace_A_input_d_0_s_0=4,
        __f2dace_A_input_d_1_s_1=7,
        __f2dace_OA_input_d_0_s_0=42,
        __f2dace_OA_input_d_1_s_1=13
    )

    assert np.allclose(res, [42, 13, 45, 19])

if __name__ == "__main__":

    test_fortran_frontend_bound()
    test_fortran_frontend_bound_offsets()
    test_fortran_frontend_bound_assumed()
    test_fortran_frontend_bound_assumed_offsets()
    test_fortran_frontend_bound_allocatable_offsets()
