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
    res = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res)

    assert np.allclose(res, [32, 32, 32, 64])

def test_fortran_frontend_bit_size_symbolic():
    test_string = """
                    PROGRAM intrinsic_math_test_bit_size
                    implicit none
                    integer, parameter :: arrsize = 2
                    integer, parameter :: arrsize2 = 3
                    integer, parameter :: arrsize3 = 4
                    integer :: res(arrsize)
                    integer :: res2(arrsize, arrsize2, arrsize3)
                    integer :: res3(arrsize+arrsize2, arrsize2 * 5, arrsize3 + arrsize2*arrsize)
                    CALL intrinsic_math_test_function(arrsize, arrsize2, arrsize3, res, res2, res3)
                    end

                    SUBROUTINE intrinsic_math_test_function(arrsize, arrsize2, arrsize3, res, res2, res3)
                    implicit none
                    integer :: arrsize
                    integer :: arrsize2
                    integer :: arrsize3
                    integer :: res(arrsize)
                    integer :: res2(arrsize, arrsize2, arrsize3)
                    integer :: res3(arrsize+arrsize2, arrsize2 * 5, arrsize3 + arrsize2*arrsize)

                    res(1) = SIZE(res)
                    res(2) = SIZE(res2)
                    res(3) = SIZE(res3)
                    res(4) = SIZE(res)*2
                    res(5) = SIZE(res)*SIZE(res2)*SIZE(res3)
                    res(6) = SIZE(res2, 1) + SIZE(res2, 2) + SIZE(res2, 3)
                    res(7) = SIZE(res3, 1) + SIZE(res3, 2) + SIZE(res3, 3)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_bit_size", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 24
    size2 = 5
    size3 = 7
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size, size2, size3], 42, order="F", dtype=np.int32)
    res3 = np.full([size+size2, size2*5, size3 + size*size2], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, res3=res3, arrsize=size, arrsize2=size2, arrsize3=size3)
    print(res)

    assert res[0] == size
    assert res[1] == size*size2*size3
    assert res[2] == (size + size2) * (size2 * 5) * (size3 + size2*size)
    assert res[3] == size * 2 
    assert res[4] == res[0] * res[1] * res[2]
    assert res[5] == size + size2 + size3
    assert res[6] == size + size2 + size2*5 + size3 + size*size2


if __name__ == "__main__":
    test_fortran_frontend_bit_size()
    test_fortran_frontend_bit_size_symbolic()
