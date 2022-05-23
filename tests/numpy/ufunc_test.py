# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math
import numpy as np
import pytest
from common import compare_numpy_output


@compare_numpy_output(check_dtype=True)
def test_ufunc_add_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.add(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_subtract_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.subtract(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_subtract_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.subtract(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_multiply_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.multiply(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divide_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divide_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.divide(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logaddexp_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logaddexp(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logaddexp2_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logaddexp2(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_true_divide_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.true_divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_true_divide_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.true_divide(A, B)


@pytest.mark.skip
@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_floor_divide_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.floor_divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_floor_divide_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.floor_divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_floor_divide_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.floor_divide(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_floor_divide_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.floor_divide(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_negative_f(A: dace.float32[10]):
    return np.negative(A)


@compare_numpy_output(validation_func=lambda a: -a)
def test_ufunc_negative_u(A: dace.uint32[10]):
    return np.negative(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_positive_f(A: dace.float32[10]):
    return np.positive(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_power_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.power(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_power_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.power(A, B)


@compare_numpy_output(non_zero=True, validation_func=lambda a, b: a**b)
def test_ufunc_power_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.power(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_float_power_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.float_power(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_float_power_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.float_power(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_float_power_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.float_power(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_remainder_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.remainder(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_remainder_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.remainder(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_remainder_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.remainder(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_fmod_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.fmod(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_fmod_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.fmod(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_fmod_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.fmod(A, B)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divmod_ff(A: dace.float32[10], B: dace.float32[10]):
    Q, R = np.divmod(A, B)
    return Q, R


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divmod_uu(A: dace.uint32[10], B: dace.uint32[10]):
    Q, R = np.divmod(A, B)
    return Q, R


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_divmod_ss(A: dace.int32[10], B: dace.int32[10]):
    Q, R = np.divmod(A, B)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_absolute_c(A: dace.complex64[10]):
    return np.absolute(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_absolute_f(A: dace.float32[10]):
    return np.absolute(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_absolute_u(A: dace.uint32[10]):
    return np.absolute(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fabs_c(A: dace.complex64[10]):
    return np.fabs(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fabs_f(A: dace.float32[10]):
    return np.fabs(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fabs_u(A: dace.uint32[10]):
    return np.fabs(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_rint_c(A: dace.complex64[10]):
    return np.rint(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_rint_f(A: dace.float32[10]):
    return np.rint(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_rint_u(A: dace.uint32[10]):
    return np.rint(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sign_c(A: dace.complex64[10]):
    return np.sign(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sign_f(A: dace.float32[10]):
    return np.sign(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sign_u(A: dace.uint32[10]):
    return np.sign(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_heaviside_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.heaviside(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_heaviside_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.heaviside(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_heaviside_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.heaviside(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conj_c(A: dace.complex64[10]):
    return np.conj(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conj_f(A: dace.float32[10]):
    return np.conj(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conj_u(A: dace.uint32[10]):
    return np.conj(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conjugate_c(A: dace.complex64[10]):
    return np.conjugate(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conjugate_f(A: dace.float32[10]):
    return np.conjugate(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_conjugate_u(A: dace.uint32[10]):
    return np.conjugate(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp_c(A: dace.complex64[10]):
    return np.exp(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp_f(A: dace.float32[10]):
    return np.exp(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp_u(A: dace.uint32[10]):
    return np.exp(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp2_c(A: dace.complex64[10]):
    return np.exp2(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp2_f(A: dace.float32[10]):
    return np.exp2(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_exp2_u(A: dace.uint32[10]):
    return np.exp2(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log_c(A: dace.complex64[10]):
    return np.log(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log_f(A: dace.float32[10]):
    return np.log(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log_u(A: dace.uint32[10]):
    return np.log(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log2_c(A: dace.complex64[10]):
    return np.log2(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log2_f(A: dace.float32[10]):
    return np.log2(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log2_u(A: dace.uint32[10]):
    return np.log2(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log10_c(A: dace.complex64[10]):
    return np.log10(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log10_f(A: dace.float32[10]):
    return np.log10(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log10_u(A: dace.uint32[10]):
    return np.log10(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_expm1_c(A: dace.complex64[10]):
    return np.expm1(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_expm1_f(A: dace.float32[10]):
    return np.expm1(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_expm1_u(A: dace.uint32[10]):
    return np.expm1(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log1p_c(A: dace.complex64[10]):
    return np.log1p(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log1p_f(A: dace.float32[10]):
    return np.log1p(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_ufunc_log1p_u(A: dace.uint32[10]):
    return np.log1p(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sqrt_c(A: dace.complex64[10]):
    return np.sqrt(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sqrt_f(A: dace.float32[10]):
    return np.sqrt(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sqrt_u(A: dace.uint32[10]):
    return np.sqrt(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_square_c(A: dace.complex64[10]):
    return np.square(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_square_f(A: dace.float32[10]):
    return np.square(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_square_u(A: dace.uint32[10]):
    return np.square(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_cbrt_c(A: dace.complex64[10]):
    return np.cbrt(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_cbrt_f(A: dace.float32[10]):
    return np.cbrt(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_cbrt_u(A: dace.uint32[10]):
    return np.cbrt(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_reciprocal_c(A: dace.complex64[10]):
    return np.reciprocal(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_reciprocal_f(A: dace.float32[10]):
    return np.reciprocal(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_reciprocal_u(A: dace.uint32[10]):
    return np.reciprocal(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_gcd_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.gcd(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_gcd_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.gcd(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_gcd_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.gcd(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_lcm_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.lcm(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_lcm_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.lcm(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_lcm_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.lcm(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sin_c(A: dace.complex64[10]):
    return np.sin(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_sin_f(A: dace.float32[10]):
    return np.sin(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_sin_u(A: dace.uint32[10]):
    return np.sin(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_cos_c(A: dace.complex64[10]):
    return np.cos(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_cos_f(A: dace.float32[10]):
    return np.cos(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_cos_u(A: dace.uint32[10]):
    return np.cos(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_tan_c(A: dace.complex64[10]):
    return np.tan(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_tan_f(A: dace.float32[10]):
    return np.tan(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_tan_u(A: dace.uint32[10]):
    return np.tan(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arcsin_c(A: dace.complex64[10]):
    return np.arcsin(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arcsin_f(A: dace.float32[10]):
    return np.arcsin(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arcsin_u(A: dace.uint32[10]):
    return np.arcsin(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arccos_c(A: dace.complex64[10]):
    return np.arccos(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arccos_f(A: dace.float32[10]):
    return np.arccos(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arccos_u(A: dace.uint32[10]):
    return np.arccos(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arctan_c(A: dace.complex64[10]):
    return np.arctan(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arctan_f(A: dace.float32[10]):
    return np.arctan(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arctan_u(A: dace.uint32[10]):
    return np.arctan(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_sinh_c(A: dace.complex64[10]):
    return np.sinh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_sinh_f(A: dace.float32[10]):
    return np.sinh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_sinh_u(A: dace.uint32[10]):
    return np.sinh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_cosh_c(A: dace.complex64[10]):
    return np.cosh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_cosh_f(A: dace.float32[10]):
    return np.cosh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_cosh_u(A: dace.uint32[10]):
    return np.cosh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_tanh_c(A: dace.complex64[10]):
    return np.tanh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_tanh_f(A: dace.float32[10]):
    return np.tanh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_tanh_u(A: dace.uint32[10]):
    return np.tanh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arcsinh_c(A: dace.complex64[10]):
    return np.arcsinh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arcsinh_f(A: dace.float32[10]):
    return np.arcsinh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arcsinh_u(A: dace.uint32[10]):
    return np.arcsinh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arccosh_c(A: dace.complex64[10]):
    return np.arccosh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arccosh_f(A: dace.float32[10]):
    return np.arccosh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arccosh_u(A: dace.uint32[10]):
    return np.arccosh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arctanh_c(A: dace.complex64[10]):
    return np.arctanh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arctanh_f(A: dace.float32[10]):
    return np.arctanh(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_arctanh_u(A: dace.uint32[10]):
    return np.arctanh(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arctan2_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.arctan2(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arctan2_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.arctan2(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_arctan2_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.arctan2(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_hypot_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.hypot(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_hypot_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.hypot(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_hypot_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.hypot(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_degrees_c(A: dace.complex64[10]):
    return np.degrees(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_degrees_f(A: dace.float32[10]):
    return np.degrees(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_degrees_u(A: dace.uint32[10]):
    return np.degrees(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_rad2deg_c(A: dace.complex64[10]):
    return np.rad2deg(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_rad2deg_f(A: dace.float32[10]):
    return np.rad2deg(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_rad2deg_u(A: dace.uint32[10]):
    return np.rad2deg(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_radians_c(A: dace.complex64[10]):
    return np.radians(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_radians_f(A: dace.float32[10]):
    return np.radians(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_radians_u(A: dace.uint32[10]):
    return np.radians(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_deg2rad_c(A: dace.complex64[10]):
    return np.deg2rad(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_deg2rad_f(A: dace.float32[10]):
    return np.deg2rad(A)


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_ufunc_deg2rad_u(A: dace.uint32[10]):
    return np.deg2rad(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_and_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.bitwise_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_and_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.bitwise_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_and_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.bitwise_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_and_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.bitwise_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_or_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.bitwise_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_or_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.bitwise_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_or_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.bitwise_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_or_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.bitwise_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_xor_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.bitwise_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_xor_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.bitwise_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_xor_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.bitwise_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_bitwise_xor_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.bitwise_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_invert_c(A: dace.complex64[10]):
    return np.invert(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_invert_f(A: dace.float32[10]):
    return np.invert(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_invert_u(A: dace.uint32[10]):
    return np.invert(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_invert_s(A: dace.int32[10]):
    return np.invert(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_left_shift_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.left_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_left_shift_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.left_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_left_shift_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.left_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_left_shift_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.left_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_right_shift_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.right_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_right_shift_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.right_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_right_shift_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.right_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_right_shift_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.right_shift(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_greater_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.greater(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_greater_equal_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.greater_equal(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_less_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.less(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_less_equal_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.less_equal(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_equal_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.equal(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_not_equal_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.not_equal(A, B)


@pytest.mark.skip
@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_and_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.logical_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_and_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logical_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_and_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.logical_and(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_and_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.logical_and(A, B)


@pytest.mark.skip
@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_or_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.logical_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_or_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logical_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_or_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.logical_or(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_or_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.logical_or(A, B)


@pytest.mark.skip
@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_xor_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.logical_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_xor_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.logical_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_xor_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.logical_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_logical_xor_su(A: dace.int32[10], B: dace.uint32[10]):
    return np.logical_xor(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_maximum_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.maximum(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_maximum_nan_ff(A: dace.float32[10], B: dace.float32[10]):
    C = np.true_divide(A, 0)
    return np.maximum(C, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fmax_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.fmax(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fmax_nan_ff(A: dace.float32[10], B: dace.float32[10]):
    C = np.true_divide(A, 0)
    return np.fmax(C, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_minimum_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.minimum(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_minimum_nan_ff(A: dace.float32[10], B: dace.float32[10]):
    C = np.true_divide(A, 0)
    return np.minimum(C, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fmin_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.fmin(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_fmin_nan_ff(A: dace.float32[10], B: dace.float32[10]):
    C = np.true_divide(A, 0)
    return np.fmin(C, B)


def test_ufunc_isfinite_c():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isfinite_c(A: dace.complex64[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isfinite(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isfinite_c()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


def test_ufunc_isfinite_f():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isfinite_f(A: dace.float32[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isfinite(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isfinite_f()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


# NumPy accepts integer arrays in np.isfinite.
# However, if any element of an integer array is inf, it will fail because it
# "<class 'OverflowError'>: cannot convert float infinity to integer"
@pytest.mark.skip
@compare_numpy_output(validation_func=lambda a: np.isfinite(a))
def test_ufunc_isfinite_u(A: dace.uint32[10]):
    A[0] = np.inf
    A[1] = np.NaN
    return np.isfinite(A)


def test_ufunc_isinf_c():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isinf_c(A: dace.complex64[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isinf(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isinf_c()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


def test_ufunc_isinf_f():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isinf_f(A: dace.float32[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isinf(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isinf_f()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


# NumPy accepts integer arrays in np.isinf.
# However, if any element of an integer array is inf, it will fail because it
# "<class 'OverflowError'>: cannot convert float infinity to integer"
@pytest.mark.skip
@compare_numpy_output(validation_func=lambda a: np.isinf(a))
def test_ufunc_isinf_u(A: dace.uint32[10]):
    A[0] = np.inf
    A[1] = np.NaN
    return np.isinf(A)


def test_ufunc_isnan_c():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isnan_c(A: dace.complex64[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isnan(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isnan_c()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


def test_ufunc_isnan_f():
    @compare_numpy_output(check_dtype=True)
    def ufunc_isnan_f(A: dace.float32[10]):
        A[0] = np.inf
        A[1] = np.NaN
        return np.isnan(A)

    args = dace.Config.get('compiler', 'cpu', 'args')
    print(args)
    if args.find('-ffast-math') >= 0:
        new_args = args.replace('-ffast-math', '-fno-finite-math-only')
        print(new_args)
        dace.Config.set('compiler', 'cpu', 'args', value=new_args)
        print(dace.Config.get('compiler', 'cpu', 'args'))
    ufunc_isnan_f()
    dace.Config.set('compiler', 'cpu', 'args', value=args)


# NumPy accepts integer arrays in np.isnan.
# However, if any element of an integer array is inf, it will fail because it
# "<class 'OverflowError'>: cannot convert float infinity to integer"
@pytest.mark.skip
@compare_numpy_output(validation_func=lambda a: np.isnan(a))
def test_ufunc_isnan_u(A: dace.uint32[10]):
    A[0] = np.inf
    A[1] = np.NaN
    return np.isnan(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_signbit_c(A: dace.complex64[10]):
    return np.signbit(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_signbit_f(A: dace.float32[10]):
    return np.signbit(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_signbit_u(A: dace.uint32[10]):
    return np.signbit(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_copysign_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.copysign(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_copysign_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.copysign(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_copysign_fd(A: dace.float32[10], B: dace.float64[10]):
    return np.copysign(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_copysign_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.copysign(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_nextafter_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.nextafter(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_nextafter_ff(A: dace.float32[10], B: dace.float32[10]):
    return np.nextafter(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_nextafter_fd(A: dace.float32[10], B: dace.float64[10]):
    return np.nextafter(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_nextafter_uu(A: dace.uint32[10], B: dace.uint32[10]):
    return np.nextafter(A, B)


@compare_numpy_output(validation_func=lambda a: np.nextafter(a, np.inf) - a)
def test_ufunc_spacing_c(A: dace.complex64[10]):
    return np.spacing(A)


@compare_numpy_output(validation_func=lambda a: np.nextafter(a, np.inf) - a)
def test_ufunc_spacing_f(A: dace.float32[10]):
    return np.spacing(A)


@compare_numpy_output(validation_func=lambda a: np.nextafter(a, np.inf) - a)
def test_ufunc_spacing_u(A: dace.uint32[10]):
    return np.spacing(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_modf_c(A: dace.complex64[10]):
    Q, R = np.modf(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_modf_f(A: dace.float32[10]):
    Q, R = np.modfd(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_modf_u(A: dace.uint32[10]):
    Q, R = np.modf(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_ldexp_cc(A: dace.complex64[10], B: dace.complex64[10]):
    return np.ldexp(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_ldexp_fs(A: dace.float32[10], B: dace.int32[10]):
    return np.ldexp(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_ldexp_ss(A: dace.int32[10], B: dace.int32[10]):
    return np.ldexp(A, B)


@compare_numpy_output(check_dtype=True)
def test_ufunc_frexp_c(A: dace.complex64[10]):
    Q, R = np.frexp(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_frexp_f(A: dace.float32[10]):
    Q, R = np.frexp(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_frexp_u(A: dace.uint32[10]):
    Q, R = np.frexp(A)
    return Q, R


@compare_numpy_output(check_dtype=True)
def test_ufunc_floor_c(A: dace.complex64[10]):
    return np.floor(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_floor_f(A: dace.float32[10]):
    return np.floor(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_floor_u(A: dace.uint32[10]):
    return np.floor(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_ceil_c(A: dace.complex64[10]):
    return np.ceil(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_ceil_f(A: dace.float32[10]):
    return np.ceil(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_ceil_u(A: dace.uint32[10]):
    return np.ceil(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_trunc_c(A: dace.complex64[10]):
    return np.trunc(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_trunc_f(A: dace.float32[10]):
    return np.trunc(A)


@compare_numpy_output(check_dtype=True)
def test_ufunc_trunc_u(A: dace.uint32[10]):
    return np.trunc(A)


if __name__ == "__main__":
    test_ufunc_add_ff()
    test_ufunc_subtract_ff()
    test_ufunc_subtract_uu()
    test_ufunc_multiply_ff()
    test_ufunc_divide_ff()
    test_ufunc_divide_uu()
    test_ufunc_logaddexp_ff()
    test_ufunc_logaddexp2_ff()
    test_ufunc_true_divide_ff()
    test_ufunc_true_divide_uu()
    # test_ufunc_floor_divide_cc()
    test_ufunc_floor_divide_ff()
    test_ufunc_floor_divide_uu()
    test_ufunc_floor_divide_ss()
    test_ufunc_negative_f()
    test_ufunc_negative_u()  # NumPy doesn't change unsigned to signed
    test_ufunc_positive_f()
    test_ufunc_power_ff()
    test_ufunc_power_uu()
    test_ufunc_power_ss()  # DaCe implementation behaves like Python
    test_ufunc_float_power_ff()
    test_ufunc_float_power_uu()
    test_ufunc_float_power_ss()
    test_ufunc_remainder_ff()
    test_ufunc_remainder_uu()
    test_ufunc_remainder_ss()
    test_ufunc_fmod_ff()
    test_ufunc_fmod_uu()
    test_ufunc_fmod_ss()
    test_ufunc_divmod_ff()
    test_ufunc_divmod_uu()
    test_ufunc_divmod_ss()
    test_ufunc_absolute_c()
    test_ufunc_absolute_f()
    test_ufunc_absolute_u()
    test_ufunc_fabs_c()
    test_ufunc_fabs_f()
    test_ufunc_fabs_u()
    test_ufunc_rint_c()
    test_ufunc_rint_f()
    test_ufunc_rint_u()
    test_ufunc_sign_c()
    test_ufunc_sign_f()
    test_ufunc_sign_u()
    test_ufunc_heaviside_cc()
    test_ufunc_heaviside_ff()
    test_ufunc_heaviside_uu()
    test_ufunc_conj_c()
    test_ufunc_conj_f()
    test_ufunc_conj_u()
    test_ufunc_conjugate_c()
    test_ufunc_conjugate_f()
    test_ufunc_conjugate_u()
    test_ufunc_exp_c()
    test_ufunc_exp_f()
    test_ufunc_exp_u()
    test_ufunc_exp2_c()
    test_ufunc_exp2_f()
    test_ufunc_exp2_u()
    test_ufunc_log_c()
    test_ufunc_log_f()
    test_ufunc_log_u()
    test_ufunc_log2_c()
    test_ufunc_log2_f()
    test_ufunc_log2_u()
    test_ufunc_log10_c()
    test_ufunc_log10_f()
    test_ufunc_log10_u()
    test_ufunc_expm1_c()
    test_ufunc_expm1_f()
    test_ufunc_expm1_u()
    test_ufunc_log1p_c()
    test_ufunc_log1p_f()
    test_ufunc_log1p_u()
    test_ufunc_sqrt_c()
    test_ufunc_sqrt_f()
    test_ufunc_sqrt_u()
    test_ufunc_square_c()
    test_ufunc_square_f()
    test_ufunc_square_u()
    test_ufunc_cbrt_c()
    test_ufunc_cbrt_f()
    test_ufunc_cbrt_u()
    test_ufunc_reciprocal_c()
    test_ufunc_reciprocal_f()
    test_ufunc_reciprocal_u()
    test_ufunc_gcd_cc()
    test_ufunc_gcd_ff()
    test_ufunc_gcd_uu()
    test_ufunc_lcm_cc()
    test_ufunc_lcm_ff()
    test_ufunc_lcm_uu()
    test_ufunc_sin_c()
    test_ufunc_sin_f()
    test_ufunc_sin_u()
    test_ufunc_cos_c()
    test_ufunc_cos_f()
    test_ufunc_cos_u()
    test_ufunc_tan_c()
    test_ufunc_tan_f()
    test_ufunc_tan_u()
    test_ufunc_arcsin_c()
    test_ufunc_arcsin_f()
    test_ufunc_arcsin_u()
    test_ufunc_arccos_c()
    test_ufunc_arccos_f()
    test_ufunc_arccos_u()
    test_ufunc_arctan_c()
    test_ufunc_arctan_f()
    test_ufunc_arctan_u()
    test_ufunc_sinh_c()
    test_ufunc_sinh_f()
    test_ufunc_sinh_u()
    test_ufunc_cosh_c()
    test_ufunc_cosh_f()
    test_ufunc_cosh_u()
    test_ufunc_tanh_c()
    test_ufunc_tanh_f()
    test_ufunc_tanh_u()
    test_ufunc_arcsinh_c()
    test_ufunc_arcsinh_f()
    test_ufunc_arcsinh_u()
    test_ufunc_arccosh_c()
    test_ufunc_arccosh_f()
    test_ufunc_arccosh_u()
    test_ufunc_arctanh_c()
    test_ufunc_arctanh_f()
    test_ufunc_arctanh_u()
    test_ufunc_arctan2_cc()
    test_ufunc_arctan2_ff()
    test_ufunc_arctan2_uu()
    test_ufunc_hypot_cc()
    test_ufunc_hypot_ff()
    test_ufunc_hypot_uu()
    test_ufunc_degrees_c()
    test_ufunc_degrees_f()
    test_ufunc_degrees_u()
    test_ufunc_rad2deg_c()
    test_ufunc_rad2deg_f()
    test_ufunc_rad2deg_u()
    test_ufunc_radians_c()
    test_ufunc_radians_f()
    test_ufunc_radians_u()
    test_ufunc_deg2rad_c()
    test_ufunc_deg2rad_f()
    test_ufunc_deg2rad_u()
    test_ufunc_bitwise_and_cc()
    test_ufunc_bitwise_and_ff()
    test_ufunc_bitwise_and_uu()
    test_ufunc_bitwise_and_su()
    test_ufunc_bitwise_or_cc()
    test_ufunc_bitwise_or_ff()
    test_ufunc_bitwise_or_uu()
    test_ufunc_bitwise_or_su()
    test_ufunc_bitwise_xor_cc()
    test_ufunc_bitwise_xor_ff()
    test_ufunc_bitwise_xor_uu()
    test_ufunc_bitwise_xor_su()
    test_ufunc_invert_c()
    test_ufunc_invert_f()
    test_ufunc_invert_u()
    test_ufunc_invert_s()
    test_ufunc_left_shift_cc()
    test_ufunc_left_shift_ff()
    test_ufunc_left_shift_uu()
    test_ufunc_left_shift_su()
    test_ufunc_right_shift_cc()
    test_ufunc_right_shift_ff()
    test_ufunc_right_shift_uu()
    test_ufunc_right_shift_su()
    test_ufunc_greater_ff()
    test_ufunc_greater_equal_ff()
    test_ufunc_less_ff()
    test_ufunc_less_equal_ff()
    test_ufunc_equal_ff()
    test_ufunc_not_equal_ff()
    # test_ufunc_logical_and_cc()  # TODO: How to convert to bool?
    test_ufunc_logical_and_ff()
    test_ufunc_logical_and_uu()
    test_ufunc_logical_and_su()
    # test_ufunc_logical_or_cc()   # TODO: How to convert to bool?
    test_ufunc_logical_or_ff()
    test_ufunc_logical_or_uu()
    test_ufunc_logical_or_su()
    # test_ufunc_logical_xor_cc()  # TODO: How to convert to bool?
    test_ufunc_logical_xor_ff()
    test_ufunc_logical_xor_uu()
    test_ufunc_logical_xor_su()
    test_ufunc_maximum_ff()
    test_ufunc_maximum_nan_ff()
    test_ufunc_fmax_ff()
    test_ufunc_fmax_nan_ff()
    test_ufunc_minimum_ff()
    test_ufunc_minimum_nan_ff()
    test_ufunc_fmin_ff()
    test_ufunc_fmin_nan_ff()
    test_ufunc_isfinite_c()
    test_ufunc_isfinite_f()
    # test_ufunc_isfinite_u()
    test_ufunc_isinf_c()
    test_ufunc_isinf_f()
    # test_ufunc_isinf_u())
    test_ufunc_isnan_c()
    test_ufunc_isnan_f()
    # test_ufunc_isnan_u()
    test_ufunc_signbit_c()
    test_ufunc_signbit_f()
    test_ufunc_signbit_u()
    test_ufunc_copysign_cc()
    test_ufunc_copysign_ff()
    test_ufunc_copysign_fd()
    test_ufunc_copysign_uu()
    test_ufunc_nextafter_cc()
    test_ufunc_nextafter_ff()
    test_ufunc_nextafter_fd()
    test_ufunc_nextafter_uu()
    test_ufunc_spacing_c()  # Unclear formula
    test_ufunc_spacing_f()  # Unclear formula
    test_ufunc_spacing_u()  # Unclear formula
    test_ufunc_modf_c()
    test_ufunc_modf_f()
    test_ufunc_modf_u()
    test_ufunc_ldexp_cc()
    test_ufunc_ldexp_fs()
    test_ufunc_ldexp_ss()
    test_ufunc_frexp_c()
    test_ufunc_frexp_f()
    test_ufunc_frexp_u()
    test_ufunc_floor_c()
    test_ufunc_floor_f()
    test_ufunc_floor_u()
    test_ufunc_ceil_c()
    test_ufunc_ceil_f()
    test_ufunc_ceil_u()
    test_ufunc_trunc_c()
    test_ufunc_trunc_f()
    test_ufunc_trunc_u()
