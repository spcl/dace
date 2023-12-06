# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_bit_size():
    test_string = """
                    PROGRAM intrinsic_math_test_bit_size
                    implicit none
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(res)
                    end

                    SUBROUTINE intrinsic_math_test_function(res)
                    integer, dimension(4) :: res
                    logical :: a = .TRUE.
                    integer :: b = 1
                    real :: c = 1
                    double precision :: d = 1

                    res(1) = BIT_SIZE(a)
                    res(2) = BIT_SIZE(b)
                    res(3) = BIT_SIZE(c)
                    res(4) = BIT_SIZE(d)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_bit_size", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    res1 = np.full([2], 42, order="F", dtype=np.int32)
    sdfg(arg1=arg1, arg2=arg2, res1=res1, res2=res2)
    print(res)

def test_fortran_frontend_bit_size_symbolic():
    test_string = """
                    PROGRAM intrinsic_math_test_bit_size
                    implicit none
                    integer :: arrsize = 2
                    integer :: res(arrsize)
                    CALL intrinsic_math_test_function(arrsize, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(arrsize, res)
                    implicit none
                    integer :: arrsize
                    integer :: res(arrsize)

                    res(1) = SIZE(res)
                    res(2) = SIZE(res)*2

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_bit_size", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 24
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, arrsize=size)
    print(res)


if __name__ == "__main__":

    #test_fortran_frontend_bit_size()
    test_fortran_frontend_bit_size_symbolic()
