# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_min_max():
    test_string = """
                    PROGRAM intrinsic_math_test_min_max
                    implicit none
                    double precision, dimension(2) :: arg1
                    double precision, dimension(2) :: arg2
                    double precision, dimension(2) :: res1
                    double precision, dimension(2) :: res2
                    CALL intrinsic_math_test_function(arg1, arg2, res1, res2)
                    end

                    SUBROUTINE intrinsic_math_test_function(arg1, arg2, res1, res2)
                    double precision, dimension(2) :: arg1
                    double precision, dimension(2) :: arg2
                    double precision, dimension(2) :: res1
                    double precision, dimension(2) :: res2

                    res1(1) = MIN(arg1(1), arg2(1))
                    res1(2) = MIN(arg1(2), arg2(2))

                    res2(1) = MAX(arg1(1), arg2(1))
                    res2(2) = MAX(arg1(2), arg2(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_min_max", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)

    arg1[0] = 20
    arg1[1] = 25
    arg2[0] = 30
    arg2[1] = 18

    res1 = np.full([2], 42, order="F", dtype=np.float64)
    res2 = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(arg1=arg1, arg2=arg2, res1=res1, res2=res2)

    assert res1[0] == 20
    assert res1[1] == 18
    assert res2[0] == 30
    assert res2[1] == 25


def test_fortran_frontend_sqrt():
    test_string = """
                    PROGRAM intrinsic_math_test_sqrt
                    implicit none
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res

                    res(1) = SQRT(d(1))
                    res(2) = SQRT(d(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_sqrt", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2
    d[1] = 5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.sqrt(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9

def test_fortran_frontend_abs():
    test_string = """
                    PROGRAM intrinsic_math_test_abs
                    implicit none
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res

                    res(1) = ABS(d(1))
                    res(2) = ABS(d(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_abs", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = -30
    d[1] = 40
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == 30
    assert res[1] == 40

def test_fortran_frontend_exp():
    test_string = """
                    PROGRAM intrinsic_math_test_exp
                    implicit none
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res

                    res(1) = EXP(d(1))
                    res(2) = EXP(d(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_exp", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.exp(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


if __name__ == "__main__":

    test_fortran_frontend_min_max()
    test_fortran_frontend_sqrt()
    test_fortran_frontend_abs()
    test_fortran_frontend_exp()
