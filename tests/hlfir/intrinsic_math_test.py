"""Verbatim port of f2dace/dev:tests/fortran/intrinsic_math_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_min_max(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_min_max_function(arg1, arg2, res1, res2)
double precision, dimension(2) :: arg1
double precision, dimension(2) :: arg2
double precision, dimension(2) :: res1
double precision, dimension(2) :: res2

res1(1) = MIN(arg1(1), arg2(1))
res1(2) = MIN(arg1(2), arg2(2))

res2(1) = MAX(arg1(1), arg2(1))
res2(2) = MAX(arg1(2), arg2(2))

END SUBROUTINE intrinsic_math_test_min_max_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_min_max_function').build()

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


def test_fortran_frontend_sqrt(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_sqrt_function(d, res)
double precision, dimension(2) :: d
double precision, dimension(2) :: res

res(1) = SQRT(d(1))
res(2) = SQRT(d(2))

END SUBROUTINE intrinsic_math_test_sqrt_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_sqrt_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2
    d[1] = 5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.sqrt(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_abs(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_abs_function(d, res)
double precision, dimension(2) :: d
double precision, dimension(2) :: res

res(1) = ABS(d(1))
res(2) = ABS(d(2))

END SUBROUTINE intrinsic_math_test_abs_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_abs_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = -30
    d[1] = 40
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)

    assert res[0] == 30
    assert res[1] == 40


def test_fortran_frontend_exp(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_exp_function(d, res)
double precision, dimension(2) :: d
double precision, dimension(2) :: res

res(1) = EXP(d(1))
res(2) = EXP(d(2))

END SUBROUTINE intrinsic_math_test_exp_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_exp_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.exp(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_log(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_exp_function(d, res)
double precision, dimension(2) :: d
double precision, dimension(2) :: res

res(1) = LOG(d(1))
res(2) = LOG(d(2))

END SUBROUTINE intrinsic_math_test_exp_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_exp_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2.71
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res)
    py_res = np.log(d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_mod_float(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_mod_function(d, res)
double precision, dimension(12) :: d
double precision, dimension(6) :: res

res(1) = MOD(d(1), d(2))
res(2) = MOD(d(3), d(4))
res(3) = MOD(d(5), d(6))
res(4) = MOD(d(7), d(8))
res(5) = MOD(d(9), d(10))
res(6) = MOD(d(11), d(12))

END SUBROUTINE intrinsic_math_test_mod_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_mod_function').build()

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


def test_fortran_frontend_mod_integer(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_modulo_function(d, res)
integer, dimension(8) :: d
integer, dimension(4) :: res

res(1) = MOD(d(1), d(2))
res(2) = MOD(d(3), d(4))
res(3) = MOD(d(5), d(6))
res(4) = MOD(d(7), d(8))

END SUBROUTINE intrinsic_math_test_modulo_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_modulo_function').build()

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


def test_fortran_frontend_modulo_float(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_modulo_function(d, res)
double precision, dimension(12) :: d
double precision, dimension(6) :: res

res(1) = MODULO(d(1), d(2))
res(2) = MODULO(d(3), d(4))
res(3) = MODULO(d(5), d(6))
res(4) = MODULO(d(7), d(8))
res(5) = MODULO(d(9), d(10))
res(6) = MODULO(d(11), d(12))

END SUBROUTINE intrinsic_math_test_modulo_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_modulo_function').build()

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


def test_fortran_frontend_modulo_integer(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_modulo_function(d, res)
integer, dimension(8) :: d
integer, dimension(4) :: res

res(1) = MODULO(d(1), d(2))
res(2) = MODULO(d(3), d(4))
res(3) = MODULO(d(5), d(6))
res(4) = MODULO(d(7), d(8))

END SUBROUTINE intrinsic_math_test_modulo_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_modulo_function').build()

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


def test_fortran_frontend_floor(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_modulo_function(d, res)
real, dimension(4) :: d
integer, dimension(4) :: res

res(1) = FLOOR(d(1))
res(2) = FLOOR(d(2))
res(3) = FLOOR(d(3))
res(4) = FLOOR(d(4))

END SUBROUTINE intrinsic_math_test_modulo_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_modulo_function').build()

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


def test_fortran_frontend_scale(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_scale_function(d, d2, res)
real, dimension(4) :: d
integer, dimension(4) :: d2
real, dimension(5) :: res

res(1) = SCALE(d(1), d2(1))
res(2) = SCALE(d(2), d2(2))
res(3) = SCALE(d(3), d2(3))
! Verifies that we properly replace call even inside a complex expression
res(4) = (SCALE(d(4), d2(4))) + (SCALE(d(4), d2(4))*2)
res(5) = (SCALE(SCALE(d(4), d2(4)), d2(4)))

END SUBROUTINE intrinsic_math_test_scale_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_scale_function').build()

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


def test_fortran_frontend_exponent(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_exponent_function(d, res)
real, dimension(4) :: d
integer, dimension(4) :: res

res(1) = EXPONENT(d(1))
res(2) = EXPONENT(d(2))
res(3) = EXPONENT(d(3))
res(4) = EXPONENT(d(4))

END SUBROUTINE intrinsic_math_test_exponent_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_exponent_function').build()

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


def test_fortran_frontend_int(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_int_function(d, d2, res, res2, res3, res4)
integer :: n
real, dimension(4) :: d
real, dimension(8) :: d2
integer, dimension(4) :: res
real, dimension(4) :: res2
integer, dimension(8) :: res3
real, dimension(8) :: res4

res(1) = INT(d(1))
res(2) = INT(d(2))
res(3) = INT(d(3))
res(4) = INT(d(4))

res2(1) = AINT(d(1))
res2(2) = AINT(d(2))
res2(3) = AINT(d(3))
! KIND parameter is ignored
res2(4) = AINT(d(4), 4)

DO n=1,8
    ! KIND parameter is ignored
    res3(n) = NINT(d2(n), 4)
END DO

DO n=1,8
    ! KIND parameter is ignored
    res4(n) = ANINT(d2(n), 4)
END DO

END SUBROUTINE intrinsic_math_test_int_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_int_function').build()

    size = 4
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 1.0
    d[1] = 1.5
    d[2] = 42.5
    d[3] = -42.5
    d2 = np.full([size * 2], 42, order="F", dtype=np.float32)
    d2[0] = 3.49
    d2[1] = 3.5
    d2[2] = 3.51
    d2[3] = 4
    d2[4] = -3.49
    d2[5] = -3.5
    d2[6] = -3.51
    d2[7] = -4
    res = np.full([4], 42, order="F", dtype=np.int32)
    res2 = np.full([4], 42, order="F", dtype=np.float32)
    res3 = np.full([8], 42, order="F", dtype=np.int32)
    res4 = np.full([8], 42, order="F", dtype=np.float32)
    sdfg(d=d, d2=d2, res=res, res2=res2, res3=res3, res4=res4, n=0)

    assert np.array_equal(res, [1, 1, 42, -42])
    assert np.array_equal(res2, [1., 1., 42., -42.])
    assert np.array_equal(res3, [3, 4, 4, 4, -3, -4, -4, -4])
    assert np.array_equal(res4, [3., 4., 4., 4., -3., -4., -4., -4.])


def test_fortran_frontend_real(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_real_function(d, d2, d3, res, res2)
integer :: n
double precision, dimension(2) :: d
real, dimension(2) :: d2
integer, dimension(2) :: d3
double precision, dimension(6) :: res
real, dimension(6) :: res2

res(1) = DBLE(d(1))
res(2) = DBLE(d(2))
res(3) = DBLE(d2(1))
res(4) = DBLE(d2(2))
res(5) = DBLE(d3(1))
res(6) = DBLE(d3(2))

res2(1) = REAL(d(1))
res2(2) = REAL(d(2))
res2(3) = REAL(d2(1))
res2(4) = REAL(d2(2))
res2(5) = REAL(d3(1))
res2(6) = REAL(d3(2))

END SUBROUTINE intrinsic_math_test_real_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_real_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 7.0
    d[1] = 13.11
    d2 = np.full([size], 42, order="F", dtype=np.float32)
    d2[0] = 7.0
    d2[1] = 13.11
    d3 = np.full([size], 42, order="F", dtype=np.int32)
    d3[0] = 7
    d3[1] = 13

    res = np.full([size * 3], 42, order="F", dtype=np.float64)
    res2 = np.full([size * 3], 42, order="F", dtype=np.float32)
    sdfg(d=d, d2=d2, d3=d3, res=res, res2=res2, n=0)

    assert np.allclose(res, [7.0, 13.11, 7.0, 13.11, 7., 13.])
    assert np.allclose(res2, [7.0, 13.11, 7.0, 13.11, 7., 13.])


def test_fortran_frontend_real_kind(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_real_function(d, d2, d3, res)
integer :: n
double precision, dimension(2) :: d
real, dimension(2) :: d2
integer, dimension(2) :: d3
double precision, dimension(6) :: res

res(1) = REAL(d(1), 8)
res(2) = REAL(d(2), 8)
res(3) = REAL(d2(1), 8)
res(4) = REAL(d2(2), 8)
res(5) = REAL(d3(1), 8)
res(6) = REAL(d3(2), 8)

END SUBROUTINE intrinsic_math_test_real_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_real_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 7.0
    d[1] = 13.11
    d2 = np.full([size], 42, order="F", dtype=np.float32)
    d2[0] = 7.0
    d2[1] = 13.11
    d3 = np.full([size], 42, order="F", dtype=np.int32)
    d3[0] = 7
    d3[1] = 13

    res = np.full([size * 3], 42, order="F", dtype=np.float64)
    sdfg(d=d, d2=d2, d3=d3, res=res, n=0)

    assert np.allclose(res, [7.0, 13.11, 7.0, 13.11, 7.0, 13.0])


def test_fortran_frontend_trig(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_trig_function(d, res)
integer :: n
real, dimension(3) :: d
real, dimension(6) :: res

DO n=1,3
    res(n) = SIN(d(n))
END DO

DO n=1,3
    res(n+3) = COS(d(n))
END DO

END SUBROUTINE intrinsic_math_test_trig_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_trig_function').build()

    size = 3
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 0
    d[1] = 3.14 / 2
    d[2] = 3.14

    res = np.full([size * 2], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res, n=0)

    assert np.allclose(res, [0.0, 0.999999702, 1.59254798E-03, 1.0, 7.96274282E-04, -0.999998748])


def test_fortran_frontend_hyperbolic(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_hyperbolic_function(d, res)
integer :: n
real, dimension(3) :: d
real, dimension(9) :: res

DO n=1,3
    res(n) = SINH(d(n))
END DO

DO n=1,3
    res(n+3) = COSH(d(n))
END DO

DO n=1,3
    res(n+6) = TANH(d(n))
END DO

END SUBROUTINE intrinsic_math_test_hyperbolic_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_hyperbolic_function').build()

    size = 3
    d = np.full([size], 42, order="F", dtype=np.float32)
    d[0] = 0
    d[1] = 1
    d[2] = 3.14

    res = np.full([size * 3], 42, order="F", dtype=np.float32)
    sdfg(d=d, res=res, n=0)

    assert np.allclose(
        res,
        [0.00000000, 1.17520118, 11.5302935, 1.00000000, 1.54308057, 11.5735760, 0.00000000, 0.761594176, 0.996260226])


def test_fortran_frontend_trig_inverse(tmp_path):
    src = """
subroutine main(sincos_args, tan_args, tan2_args, res)
  integer :: n
  real, dimension(3) :: sincos_args
  real, dimension(3) :: tan_args
  real, dimension(6) :: tan2_args
  real, dimension(12) :: res
  do n = 1, 3
    res(n) = asin(sincos_args(n))
  end do
  do n = 1, 3
    res(n + 3) = acos(sincos_args(n))
  end do
  do n = 1, 3
    res(n + 6) = atan(tan_args(n))
  end do
  do n = 1, 3
    res(n + 9) = atan2(tan2_args(2*n - 1), tan2_args(2*n))
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    size = 3
    sincos_args = np.full([size], 42, order="F", dtype=np.float32)
    sincos_args[0] = -0.5
    sincos_args[1] = 0.0
    sincos_args[2] = 1.0

    atan_args = np.full([size], 42, order="F", dtype=np.float32)
    atan_args[0] = 0.0
    atan_args[1] = 1.0
    atan_args[2] = 3.14

    atan2_args = np.full([size * 2], 42, order="F", dtype=np.float32)
    atan2_args[0] = 0.0
    atan2_args[1] = 1.0
    atan2_args[2] = 1.0
    atan2_args[3] = 1.0
    atan2_args[4] = 1.0
    atan2_args[5] = 0.0

    res = np.full([size * 4], 42, order="F", dtype=np.float32)
    sdfg(sincos_args=sincos_args, tan_args=atan_args, tan2_args=atan2_args, res=res, n=0)

    assert np.allclose(res, [
        -0.523598790, 0.00000000, 1.57079637, 2.09439516, 1.57079637, 0.00000000, 0.00000000, 0.785398185, 1.26248074,
        0.00000000, 0.785398185, 1.57079637
    ])


def test_fortran_frontend_exp2(tmp_path):
    src = """
SUBROUTINE intrinsic_math_test_exp2_function(d, res)
double precision, dimension(2) :: d
double precision, dimension(2) :: res
integer :: n

do n=1,2
    res(n) = EXP(- 1.66D0 * d(n))
end do

END SUBROUTINE intrinsic_math_test_exp2_function
"""
    sdfg = build_sdfg(src, tmp_path, name='intrinsic_math_test_exp2_function').build()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = 2
    d[1] = 4.5
    res = np.full([2], 42, order="F", dtype=np.float64)
    sdfg(d=d, res=res, n=0)
    py_res = np.exp(-1.66 * d)

    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9
