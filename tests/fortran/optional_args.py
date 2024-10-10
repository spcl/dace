# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_optional():
    test_string = """
                    PROGRAM intrinsic_optional_test_function
                    implicit none
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a
                    CALL intrinsic_optional_test_function(res, res2, a)
                    end

                    SUBROUTINE intrinsic_optional_test_function(res, res2, a)
                    integer, dimension(4) :: res
                    integer, dimension(4) :: res2
                    integer :: a

                    CALL intrinsic_optional_test_function2(res, a)
                    CALL intrinsic_optional_test_function2(res2)

                    END SUBROUTINE intrinsic_optional_test_function

                    SUBROUTINE intrinsic_optional_test_function2(res, a)
                    integer, dimension(2) :: res
                    integer, optional :: a

                    res(1) = a

                    END SUBROUTINE intrinsic_optional_test_function2
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_optional_test_function", False)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5)

    assert res[0] == 5
    assert res2[0] == 0

def test_fortran_frontend_optional_complex():
    test_string = """
                    PROGRAM intrinsic_optional_test_function
                    implicit none
                    integer, dimension(5) :: res
                    integer, dimension(5) :: res2
                    integer :: a
                    double precision :: b
                    logical :: c
                    CALL intrinsic_optional_test_function(res, res2, a, b, c)
                    end

                    SUBROUTINE intrinsic_optional_test_function(res, res2, a, b, c)
                    integer, dimension(5) :: res
                    integer, dimension(5) :: res2
                    integer :: a
                    double precision :: b
                    logical :: c

                    CALL intrinsic_optional_test_function2(res, a, b)
                    CALL intrinsic_optional_test_function2(res2)

                    END SUBROUTINE intrinsic_optional_test_function

                    SUBROUTINE intrinsic_optional_test_function2(res, a, b, c)
                    integer, dimension(5) :: res
                    integer, optional :: a
                    double precision, optional :: b
                    logical, optional :: c

                    res(1) = a
                    res(2) = b
                    res(3) = c

                    END SUBROUTINE intrinsic_optional_test_function2
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_optional_test_function", False)
    #sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5, b=7, c=1)

    assert res[0] == 5
    assert res[1] == 7
    assert res[2] == 0

    assert res2[0] == 0
    assert res2[1] == 0
    assert res2[2] == 0


if __name__ == "__main__":

    test_fortran_frontend_optional()
    test_fortran_frontend_optional_complex()
