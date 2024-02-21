# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_dot():
    test_string = """
                    PROGRAM intrinsic_math_test_min_max
                    implicit none
                    double precision, dimension(5) :: arg1
                    double precision, dimension(5) :: arg2
                    double precision, dimension(2) :: res1
                    CALL intrinsic_math_test_function(arg1, arg2, res1)
                    end

                    SUBROUTINE intrinsic_math_test_function(arg1, arg2, res1)
                    double precision, dimension(5) :: arg1
                    double precision, dimension(5) :: arg2
                    double precision, dimension(2) :: res1

                    res1(1) = DOT_PRODUCT(arg1, arg2)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_min_max", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5 

    sdfg(arg1=arg1, arg2=arg2, res1=res1)

    assert res1[0] == np.dot(arg1, arg2)

def test_fortran_frontend_dot_range():
    test_string = """
                    PROGRAM intrinsic_math_test_min_max
                    implicit none
                    double precision, dimension(5) :: arg1
                    double precision, dimension(5) :: arg2
                    double precision, dimension(2) :: res1
                    CALL intrinsic_math_test_function(arg1, arg2, res1)
                    end

                    SUBROUTINE intrinsic_math_test_function(arg1, arg2, res1)
                    double precision, dimension(5) :: arg1
                    double precision, dimension(5) :: arg2
                    double precision, dimension(2) :: res1

                    res1(1) = DOT_PRODUCT(arg1(1:3), arg2(1:3))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_min_max", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5 

    sdfg(arg1=arg1, arg2=arg2, res1=res1)
    print(arg1)
    print(arg2)
    print(res1)

    assert res1[0] == np.dot(arg1, arg2)

if __name__ == "__main__":

    test_fortran_frontend_dot()
    #test_fortran_frontend_dot_range()
