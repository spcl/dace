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

def test_fortran_frontend_log():
    test_string = """
                    PROGRAM intrinsic_math_test_log
                    implicit none
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res

                    res(1) = LOG(d(1))
                    res(2) = LOG(d(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_exp", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2.71
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.log(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9

def test_fortran_frontend_log():
    test_string = """
                    PROGRAM intrinsic_math_test_log
                    implicit none
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(2) :: d
                    double precision, dimension(2) :: res

                    res(1) = LOG(d(1))
                    res(2) = LOG(d(2))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_exp", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2.71
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.log(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9

def test_fortran_frontend_mod_float():
    test_string = """
                    PROGRAM intrinsic_math_test_mod
                    implicit none
                    double precision, dimension(12) :: d
                    double precision, dimension(6) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(12) :: d
                    double precision, dimension(6) :: res

                    res(1) = MOD(d(1), d(2))
                    res(2) = MOD(d(3), d(4))
                    res(3) = MOD(d(5), d(6))
                    res(4) = MOD(d(7), d(8))
                    res(5) = MOD(d(9), d(10))
                    res(6) = MOD(d(11), d(12))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_mod", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 12
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 17.
    d[1] = 3.
    d[2] = -17.
    d[3] = 3.
    d[4] = 17.
    d[5] = -3.
    d[6] = -17.
    d[7] = -3.
    d[8] = 17.5
    d[9] = 5.5
    d[10] = -17.5
    d[11] = 5.5
    res = np.full([6], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == 2.0
    assert res[1] == -2.0
    assert res[2] == 2.0
    assert res[3] == -2.0
    assert res[4] == 1
    assert res[5] == -1

def test_fortran_frontend_mod_integer():
    test_string = """
                    PROGRAM intrinsic_math_test_mod
                    implicit none
                    integer, dimension(8) :: d
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    integer, dimension(8) :: d
                    integer, dimension(4) :: res

                    res(1) = MOD(d(1), d(2))
                    res(2) = MOD(d(3), d(4))
                    res(3) = MOD(d(5), d(6))
                    res(4) = MOD(d(7), d(8))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 12
    d = np.full([size], 42, order="F", dtype=np.int32)
    d[0] = 17
    d[1] = 3
    d[2] = -17
    d[3] = 3
    d[4] = 17
    d[5] = -3
    d[6] = -17
    d[7] = -3
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)
    assert res[0] == 2
    assert res[1] == -2
    assert res[2] == 2
    assert res[3] == -2

def test_fortran_frontend_modulo_float():
    test_string = """
                    PROGRAM intrinsic_math_test_modulo
                    implicit none
                    double precision, dimension(12) :: d
                    double precision, dimension(6) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    double precision, dimension(12) :: d
                    double precision, dimension(6) :: res

                    res(1) = MODULO(d(1), d(2))
                    res(2) = MODULO(d(3), d(4))
                    res(3) = MODULO(d(5), d(6))
                    res(4) = MODULO(d(7), d(8))
                    res(5) = MODULO(d(9), d(10))
                    res(6) = MODULO(d(11), d(12))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 12
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 17.
    d[1] = 3.
    d[2] = -17.
    d[3] = 3.
    d[4] = 17.
    d[5] = -3.
    d[6] = -17.
    d[7] = -3.
    d[8] = 17.5
    d[9] = 5.5
    d[10] = -17.5
    d[11] = 5.5
    res = np.full([6], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == 2.0
    assert res[1] == 1.0
    assert res[2] == -1.0
    assert res[3] == -2.0
    assert res[4] == 1.0
    assert res[5] == 4.5

def test_fortran_frontend_modulo_integer():
    test_string = """
                    PROGRAM intrinsic_math_test_modulo
                    implicit none
                    integer, dimension(8) :: d
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    integer, dimension(8) :: d
                    integer, dimension(4) :: res

                    res(1) = MODULO(d(1), d(2))
                    res(2) = MODULO(d(3), d(4))
                    res(3) = MODULO(d(5), d(6))
                    res(4) = MODULO(d(7), d(8))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 12
    d = np.full([size], 42, order="F", dtype=np.int32)
    d[0] = 17
    d[1] = 3
    d[2] = -17
    d[3] = 3
    d[4] = 17
    d[5] = -3
    d[6] = -17
    d[7] = -3
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == 2
    assert res[1] == 1
    assert res[2] == -1
    assert res[3] == -2

def test_fortran_frontend_floor():
    test_string = """
                    PROGRAM intrinsic_math_test_floor
                    implicit none
                    real, dimension(4) :: d
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    real, dimension(4) :: d
                    integer, dimension(4) :: res

                    res(1) = FLOOR(d(1))
                    res(2) = FLOOR(d(2))
                    res(3) = FLOOR(d(3))
                    res(4) = FLOOR(d(4))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 3.5
    d[1] = 63.000001
    d[2] = -3.5
    d[3] = -63.00001
    res = np.full([4], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == 3
    assert res[1] == 63
    assert res[2] == -4
    assert res[3] == -64

def test_fortran_frontend_scale():
    test_string = """
                    PROGRAM intrinsic_math_test_scale
                    implicit none
                    real, dimension(4) :: d
                    integer, dimension(4) :: d2
                    real, dimension(5) :: res
                    CALL intrinsic_math_test_function(d, d2, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, d2, res)
                    real, dimension(4) :: d
                    integer, dimension(4) :: d2
                    real, dimension(5) :: res

                    res(1) = SCALE(d(1), d2(1))
                    res(2) = SCALE(d(2), d2(2))
                    res(3) = SCALE(d(3), d2(3))
                    ! Verifies that we properly replace call even inside a complex expression
                    res(4) = (SCALE(d(4), d2(4))) + (SCALE(d(4), d2(4))*2)
                    res(5) = (SCALE(SCALE(d(4), d2(4)), d2(4)))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 178.1387e-4
    d[1] = 5.5
    d[2] = 5.5
    d[3] = 42.5
    d2 = np.full([size], 42, order="F", dtype=np.int32)
    d2[0] = 5
    d2[1] = 5
    d2[2] = 7
    d2[3] = 9
    res = np.full([5], 42, order="F", dtype=np.float32)
    sdfg(d=d, d2=d2, res=res)

    assert abs(res[0] - 0.570043862) < 10**-7
    assert res[1] == 176.
    assert res[2] == 704.
    assert res[3] == 65280.
    assert res[4] == 11141120.

def test_fortran_frontend_exponent():
    test_string = """
                    PROGRAM intrinsic_math_test_exponent
                    implicit none
                    real, dimension(4) :: d
                    integer, dimension(4) :: res
                    CALL intrinsic_math_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res)
                    real, dimension(4) :: d
                    integer, dimension(4) :: res

                    res(1) = EXPONENT(d(1))
                    res(2) = EXPONENT(d(2))
                    res(3) = EXPONENT(d(3))
                    res(4) = EXPONENT(d(4))

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 0.0
    d[1] = 1.0
    d[2] = 13.0
    d[3] = 390.0
    res = np.full([5], 42, order="F", dtype=np.int32)
    sdfg(d=d, res=res)

    assert res[0] == 0
    assert res[1] == 1
    assert res[2] == 4
    assert res[3] == 9

def test_fortran_frontend_int():
    test_string = """
                    PROGRAM intrinsic_math_test_int
                    implicit none
                    real, dimension(4) :: d
                    integer, dimension(4) :: res
                    real, dimension(4) :: res2
                    CALL intrinsic_math_test_function(d, res, res2)
                    end

                    SUBROUTINE intrinsic_math_test_function(d, res, res2)
                    real, dimension(4) :: d
                    integer, dimension(4) :: res
                    real, dimension(4) :: res2

                    res(1) = INT(d(1))
                    res(2) = INT(d(2))
                    res(3) = INT(d(3))
                    res(4) = INT(d(4))

                    res2(1) = AINT(d(1))
                    res2(2) = AINT(d(2))
                    res2(3) = AINT(d(3))
                    ! KIND parameter is ignored
                    res2(4) = AINT(d(4), 4)

                    END SUBROUTINE intrinsic_math_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_math_test_modulo", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 4
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 1.0
    d[1] = 1.5
    d[2] = 42.5
    d[3] = -42.5
    res = np.full([5], 42, order="F", dtype=np.int32)
    res2 = np.full([5], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res, res2=res2)

    assert res[0] == 1
    assert res[1] == 1
    assert res[2] == 42
    assert res[3] == -42

    assert res2[0] == 1.
    assert res2[1] == 1.
    assert res2[2] == 42.
    assert res2[3] == -42.

if __name__ == "__main__":

    #test_fortran_frontend_min_max()
    #test_fortran_frontend_sqrt()
    #test_fortran_frontend_abs()
    #test_fortran_frontend_exp()
    #test_fortran_frontend_log()
    #test_fortran_frontend_mod_float()
    #test_fortran_frontend_mod_integer()
    #test_fortran_frontend_modulo_float()
    #test_fortran_frontend_modulo_integer()
    #test_fortran_frontend_floor()
    #test_fortran_frontend_scale()
    #test_fortran_frontend_exponent()
    test_fortran_frontend_int()
