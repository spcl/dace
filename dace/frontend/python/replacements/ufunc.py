# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for NumPy ufuncs.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.nested_call import NestedCall
from dace.frontend.python.replacements.utils import (ProgramVisitor, Shape, UfuncInput, UfuncOutput, normalize_axes,
                                                     sym_type)
import dace.frontend.python.memlet_parser as mem_parser
from dace import InterstateEdge, Memlet, SDFG, SDFGState
from dace import dtypes, data, symbolic, nodes
from dace.sdfg import dealias

import ast
import copy
import functools
import itertools
from numbers import Integral, Number
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
import warnings

import numpy as np
import sympy as sp

numpy_version = np.lib.NumpyVersion(np.__version__)

# TODO: Add all ufuncs in subsequent PR's.
ufuncs = dict(
    add=dict(name="_numpy_add_",
             operator="Add",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = __in1 + __in2",
             reduce="lambda a, b: a + b",
             initial=np.add.identity),
    subtract=dict(name="_numpy_subtract_",
                  operator="Sub",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = __in1 - __in2",
                  reduce="lambda a, b: a - b",
                  initial=np.subtract.identity),
    multiply=dict(name="_numpy_multiply_",
                  operator="Mul",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = __in1 * __in2",
                  reduce="lambda a, b: a * b",
                  initial=np.multiply.identity),
    divide=dict(name="_numpy_divide_",
                operator="Div",
                inputs=["__in1", "__in2"],
                outputs=["__out"],
                code="__out = __in1 / __in2",
                reduce="lambda a, b: a / b",
                initial=np.divide.identity),
    logaddexp=dict(name="_numpy_logaddexp_",
                   operator=None,
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = log( exp(__in1) + exp(__in2) )",
                   reduce="lambda a, b: log( exp(a) + exp(b) )",
                   initial=np.logaddexp.identity),
    logaddexp2=dict(name="_numpy_logaddexp2_",
                    operator=None,
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = log2( exp2(__in1) + exp2(__in2) )",
                    reduce="lambda a, b: log( exp2(a) + exp2(b) )",
                    initial=np.logaddexp2.identity),
    true_divide=dict(name="_numpy_true_divide_",
                     operator="Div",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 / __in2",
                     reduce="lambda a, b: a / b",
                     initial=np.true_divide.identity),
    floor_divide=dict(name="_numpy_floor_divide_",
                      operator="FloorDiv",
                      inputs=["__in1", "__in2"],
                      outputs=["__out"],
                      code="__out = py_floor(__in1, __in2)",
                      reduce="lambda a, b: py_floor(a, b)",
                      initial=np.floor_divide.identity),
    negative=dict(name="_numpy_negative_",
                  operator="USub",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = - __in1",
                  reduce=None,
                  initial=np.negative.identity),
    positive=dict(name="_numpy_positive_",
                  operator="UAdd",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = + __in1",
                  reduce=None,
                  initial=np.positive.identity),
    power=dict(name="_numpy_power_",
               operator="Pow",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = __in1 ** __in2",
               reduce="lambda a, b: a ** b",
               initial=np.power.identity),
    float_power=dict(name="_numpy_float_power_",
                     operator="FloatPow",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = np_float_pow(__in1, __in2)",
                     reduce="lambda a, b: np_float_pow(a, b)",
                     initial=np.float_power.identity),
    remainder=dict(name="_numpy_remainder_",
                   operator="Mod",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = py_mod(__in1, __in2)",
                   reduce="lambda a, b: py_mod(a, b)",
                   initial=np.remainder.identity),
    mod=dict(name="_numpy_mod_",
             operator="Mod",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = py_mod(__in1, __in2)",
             reduce="lambda a, b: py_mod(a, b)",
             initial=np.mod.identity),
    fmod=dict(name="_numpy_fmod_",
              operator="Mod",
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = cpp_mod(__in1, __in2)",
              reduce="lambda a, b: cpp_mod(a, b)",
              initial=np.fmod.identity),
    divmod=dict(name="_numpy_divmod_",
                operator="Div",
                inputs=["__in1", "__in2"],
                outputs=["__out1", "__out2"],
                code="py_divmod(__in1, __in2, __out1, __out2)",
                reduce=None,
                initial=np.divmod.identity),
    absolute=dict(name="_numpy_absolute_",
                  operator="Abs",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = abs(__in1)",
                  reduce=None,
                  initial=np.absolute.identity),
    abs=dict(name="_numpy_abs_",
             operator="Abs",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = abs(__in1)",
             reduce=None,
             initial=np.abs.identity),
    fabs=dict(name="_numpy_fabs_",
              operator="Fabs",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = fabs(__in1)",
              reduce=None,
              initial=np.fabs.identity),
    rint=dict(name="_numpy_rint_",
              operator="Rint",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = round(__in1)",
              reduce=None,
              initial=np.rint.identity),
    sign=dict(name="_numpy_sign_",
              operator=None,
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sign_numpy_2(__in1)" if numpy_version >= '2.0.0' else "__out = sign(__in1)",
              reduce=None,
              initial=np.sign.identity),
    heaviside=dict(name="_numpy_heaviside_",
                   operator="Heaviside",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = heaviside(__in1, __in2)",
                   reduce="lambda a, b: heaviside(a, b)",
                   initial=np.heaviside.identity),
    conj=dict(name="_numpy_conj_",
              operator=None,
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = conj(__in1)",
              reduce=None,
              initial=np.conj.identity),
    conjugate=dict(name="_numpy_conjugate_",
                   operator=None,
                   inputs=["__in1"],
                   outputs=["__out"],
                   code="__out = conj(__in1)",
                   reduce=None,
                   initial=np.conjugate.identity),
    exp=dict(name="_numpy_exp_",
             operator="Exp",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = exp(__in1)",
             reduce=None,
             initial=np.exp.identity),
    exp2=dict(name="_numpy_exp2_",
              operator="Exp",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = exp2(__in1)",
              reduce=None,
              initial=np.exp2.identity),
    log=dict(name="_numpy_log_",
             operator="Log",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = log(__in1)",
             reduce=None,
             initial=np.log.identity),
    log2=dict(name="_numpy_log2_",
              operator="Log",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = log2(__in1)",
              reduce=None,
              initial=np.log2.identity),
    log10=dict(name="_numpy_log10_",
               operator="Log",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = log10(__in1)",
               reduce=None,
               initial=np.log10.identity),
    expm1=dict(name="_numpy_expm1_",
               operator="Exp",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = expm1(__in1)",
               reduce=None,
               initial=np.expm1.identity),
    log1p=dict(name="_numpy_log1p_",
               operator="Log",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = log1p(__in1)",
               reduce=None,
               initial=np.log1p.identity),
    clip=dict(name="_numpy_clip_",
              operator=None,
              inputs=["__in_a", "__in_amin", "__in_amax"],
              outputs=["__out"],
              code="__out = min(max(__in_a, __in_amin), __in_amax)",
              reduce=None,
              initial=np.inf),
    sqrt=dict(name="_numpy_sqrt_",
              operator="Sqrt",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sqrt(__in1)",
              reduce=None,
              initial=np.sqrt.identity),
    square=dict(name="_numpy_square_",
                operator=None,
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = __in1 * __in1",
                reduce=None,
                initial=np.square.identity),
    cbrt=dict(name="_numpy_cbrt_",
              operator="Cbrt",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = cbrt(__in1)",
              reduce=None,
              initial=np.cbrt.identity),
    reciprocal=dict(name="_numpy_reciprocal_",
                    operator="Div",
                    inputs=["__in1"],
                    outputs=["__out"],
                    code="__out = reciprocal(__in1)",
                    reduce=None,
                    initial=np.reciprocal.identity),
    gcd=dict(name="_numpy_gcd_",
             operator="Gcd",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = gcd(__in1, __in2)",
             reduce="lambda a, b: gcd(a, b)",
             initial=np.gcd.identity),
    lcm=dict(name="_numpy_lcm_",
             operator="Lcm",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = lcm(__in1, __in2)",
             reduce="lambda a, b: lcm(a, b)",
             initial=np.lcm.identity),
    sin=dict(name="_numpy_sin_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = sin(__in1)",
             reduce=None,
             initial=np.sin.identity),
    cos=dict(name="_numpy_cos_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = cos(__in1)",
             reduce=None,
             initial=np.cos.identity),
    tan=dict(name="_numpy_tan_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = tan(__in1)",
             reduce=None,
             initial=np.tan.identity),
    arcsin=dict(name="_numpy_arcsin_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = asin(__in1)",
                reduce=None,
                initial=np.arcsin.identity),
    arccos=dict(name="_numpy_arccos_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = acos(__in1)",
                reduce=None,
                initial=np.arccos.identity),
    arctan=dict(name="_numpy_arctan_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = atan(__in1)",
                reduce=None,
                initial=np.arctan.identity),
    sinh=dict(name="_numpy_sinh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sinh(__in1)",
              reduce=None,
              initial=np.sinh.identity),
    cosh=dict(name="_numpy_cosh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = cosh(__in1)",
              reduce=None,
              initial=np.cosh.identity),
    tanh=dict(name="_numpy_tanh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = tanh(__in1)",
              reduce=None,
              initial=np.tanh.identity),
    arcsinh=dict(name="_numpy_arcsinh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = asinh(__in1)",
                 reduce=None,
                 initial=np.arcsinh.identity),
    arccosh=dict(name="_numpy_arccosh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = acosh(__in1)",
                 reduce=None,
                 initial=np.arccos.identity),
    arctanh=dict(name="_numpy_arctanh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = atanh(__in1)",
                 reduce=None,
                 initial=np.arctanh.identity),
    arctan2=dict(name="_numpy_arctan2_",
                 operator="Arctan2",
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = atan2(__in1, __in2)",
                 reduce="lambda a, b: atan2(a, b)",
                 initial=np.arctan2.identity),
    hypot=dict(name="_numpy_hypot_",
               operator="Hypot",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = hypot(__in1, __in2)",
               reduce="lambda a, b: hypot(a, b)",
               initial=np.arctan2.identity),
    degrees=dict(name="_numpy_degrees_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = rad2deg(__in1)",
                 reduce=None,
                 initial=np.degrees.identity),
    rad2deg=dict(name="_numpy_rad2deg_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = rad2deg(__in1)",
                 reduce=None,
                 initial=np.rad2deg.identity),
    radians=dict(name="_numpy_radians_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = deg2rad(__in1)",
                 reduce=None,
                 initial=np.radians.identity),
    deg2rad=dict(name="_numpy_deg2rad_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = deg2rad(__in1)",
                 reduce=None,
                 initial=np.deg2rad.identity),
    bitwise_and=dict(name="_numpy_bitwise_and_",
                     operator="BitAnd",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 & __in2",
                     reduce="lambda a, b: a & b",
                     initial=np.bitwise_and.identity),
    bitwise_or=dict(name="_numpy_bitwise_or_",
                    operator="BitOr",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 | __in2",
                    reduce="lambda a, b: a | b",
                    initial=np.bitwise_or.identity),
    bitwise_xor=dict(name="_numpy_bitwise_xor_",
                     operator="BitXor",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 ^ __in2",
                     reduce="lambda a, b: a ^ b",
                     initial=np.bitwise_xor.identity),
    invert=dict(name="_numpy_invert_",
                operator="Invert",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = ~ __in1",
                reduce=None,
                initial=np.invert.identity),
    left_shift=dict(name="_numpy_left_shift_",
                    operator="LShift",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 << __in2",
                    reduce="lambda a, b: a << b",
                    initial=np.left_shift.identity),
    right_shift=dict(name="_numpy_right_shift_",
                     operator="RShift",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 >> __in2",
                     reduce="lambda a, b: a >> b",
                     initial=np.right_shift.identity),
    greater=dict(name="_numpy_greater_",
                 operator="Gt",
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = __in1 > __in2",
                 reduce="lambda a, b: a > b",
                 initial=np.greater.identity),
    greater_equal=dict(name="_numpy_greater_equal_",
                       operator="GtE",
                       inputs=["__in1", "__in2"],
                       outputs=["__out"],
                       code="__out = __in1 >= __in2",
                       reduce="lambda a, b: a >= b",
                       initial=np.greater_equal.identity),
    less=dict(name="_numpy_less_",
              operator="Lt",
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = __in1 < __in2",
              reduce="lambda a, b: a < b",
              initial=np.less.identity),
    less_equal=dict(name="_numpy_less_equal_",
                    operator="LtE",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 <= __in2",
                    reduce="lambda a, b: a <= b",
                    initial=np.less_equal.identity),
    equal=dict(name="_numpy_equal_",
               operator="Eq",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = __in1 == __in2",
               reduce="lambda a, b: a == b",
               initial=np.equal.identity),
    not_equal=dict(name="_numpy_not_equal_",
                   operator="NotEq",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = __in1 != __in2",
                   reduce="lambda a, b: a != b",
                   initial=np.not_equal.identity),
    logical_and=dict(name="_numpy_logical_and_",
                     operator="And",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 and __in2",
                     reduce="lambda a, b: a and b",
                     initial=np.logical_and.identity),
    logical_or=dict(name="_numpy_logical_or_",
                    operator="Or",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 or __in2",
                    reduce="lambda a, b: a or b",
                    initial=np.logical_or.identity),
    logical_xor=dict(name="_numpy_logical_xor_",
                     operator="Xor",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = (not __in1) != (not __in2)",
                     reduce="lambda a, b: (not a) != (not b)",
                     initial=np.logical_xor.identity),
    logical_not=dict(name="_numpy_logical_not_",
                     operator="Not",
                     inputs=["__in1"],
                     outputs=["__out"],
                     code="__out = not __in1",
                     reduce=None,
                     initial=np.logical_not.identity),
    maximum=dict(name="_numpy_maximum_",
                 operator=None,
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = max(__in1, __in2)",
                 reduce="lambda a, b: max(a, b)",
                 initial=-np.inf),  # np.maximum.identity is None
    fmax=dict(name="_numpy_fmax_",
              operator=None,
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = fmax(__in1, __in2)",
              reduce="lambda a, b: fmax(a, b)",
              initial=-np.inf),  # np.fmax.identity is None
    minimum=dict(name="_numpy_minimum_",
                 operator=None,
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = min(__in1, __in2)",
                 reduce="lambda a, b: min(a, b)",
                 initial=np.inf),  # np.minimum.identity is None
    fmin=dict(name="_numpy_fmin_",
              operator=None,
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = fmin(__in1, __in2)",
              reduce="lambda a, b: fmin(a, b)",
              initial=np.inf),  # np.fmin.identity is None
    isfinite=dict(name="_numpy_isfinite_",
                  operator="FpBoolean",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = isfinite(__in1)",
                  reduce=None,
                  initial=np.isfinite.identity),
    isinf=dict(name="_numpy_isinf_",
               operator="FpBoolean",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = isinf(__in1)",
               reduce=None,
               initial=np.isinf.identity),
    isnan=dict(name="_numpy_isnan_",
               operator="FpBoolean",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = isnan(__in1)",
               reduce=None,
               initial=np.isnan.identity),
    signbit=dict(name="_numpy_signbit_",
                 operator="SignBit",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = signbit(__in1)",
                 reduce=None,
                 initial=np.signbit.identity),
    copysign=dict(name="_numpy_copysign_",
                  operator="CopySign",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = copysign(__in1, __in2)",
                  reduce="lambda a, b: copysign(a, b)",
                  initial=np.copysign.identity),
    nextafter=dict(name="_numpy_nextafter_",
                   operator="NextAfter",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = nextafter(__in1, __in2)",
                   reduce="lambda a, b: nextafter(a, b)",
                   initial=np.nextafter.identity),
    spacing=dict(name="_numpy_spacing_",
                 operator="Spacing",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = nextafter(__in1, inf) - __in1",
                 reduce=None,
                 initial=np.spacing.identity),
    modf=dict(name="_numpy_modf_",
              operator="Modf",
              inputs=["__in1"],
              outputs=["__out1", "__out2"],
              code="np_modf(__in1, __out1, __out2)",
              reduce=None,
              initial=np.modf.identity),
    ldexp=dict(
        name="_numpy_ldexp_",
        operator="Ldexp",
        inputs=["__in1", "__in2"],
        outputs=["__out"],
        code="__out = ldexp(__in1, __in2)",
        # NumPy apparently has np.ldexp.reduce, but for any kind of input array
        # it returns "TypeError: No loop matching the specified signature and
        # casting was found for ufunc ldexp". Considering that the method
        # computes __in1 * 2 ** __in2, it is hard to define a reduction.
        reduce=None,
        initial=np.ldexp.identity),
    frexp=dict(name="_numpy_frexp_",
               operator="Frexp",
               inputs=["__in1"],
               outputs=["__out1", "__out2"],
               code="np_frexp(__in1, __out1, __out2)",
               reduce=None,
               initial=np.frexp.identity),
    floor=dict(name="_numpy_floor_",
               operator="Floor",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = floor(__in1)",
               reduce=None,
               initial=np.floor.identity),
    ceil=dict(name="_numpy_ceil_",
              operator="Ceil",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = ceil(__in1)",
              reduce=None,
              initial=np.ceil.identity),
    trunc=dict(name="_numpy_trunc_",
               operator="Trunc",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = trunc(__in1)",
               reduce=None,
               initial=np.trunc.identity),
)


def _get_ufunc_impl(visitor: ProgramVisitor, ast_node: ast.Call, ufunc_name: str) -> Dict[str, Any]:
    """ Retrieves the implementation details for a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc

        :raises DaCeSyntaxError: When the ufunc implementation is missing
    """

    try:
        return ufuncs[ufunc_name]
    except KeyError:
        raise mem_parser.DaceSyntaxError(visitor, ast_node,
                                         "Missing implementation for NumPy ufunc {f}.".format(f=ufunc_name))


def _validate_ufunc_num_arguments(visitor: ProgramVisitor, ast_node: ast.Call, ufunc_name: str, num_inputs: int,
                                  num_outputs: int, num_args: int):
    """ Validates the number of positional arguments in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param num_outputs: Number of ufunc outputs
        :param num_args: Number of positional arguments in the ufunc call

        :raises DaCeSyntaxError: When validation fails
    """

    if num_args > num_inputs + num_outputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected a maximum of {i} input(s) and {o} output(s), "
            "but a total of {a} arguments were given).".format(f=ufunc_name, i=num_inputs, o=num_outputs, a=num_args))


def _validate_ufunc_inputs(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, ufunc_name: str, num_inputs: int,
                           num_args: int, args: Sequence[UfuncInput]) -> List[UfuncInput]:
    """ Validates the number of type of inputs in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param args: Positional arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of input datanames and constants
    """

    # Validate number of inputs
    if num_args > num_inputs:
        # Assume that the first "num_inputs" arguments are inputs
        inputs = args[:num_inputs]
    elif num_args < num_inputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} inputs, but {a} were given).".format(f=ufunc_name, e=num_inputs, a=num_args))
    else:
        inputs = args
    if isinstance(inputs, (list, tuple)):
        inputs = list(inputs)
    else:
        inputs = [inputs]

    # Validate type of inputs
    for arg in inputs:
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            pass
        elif isinstance(arg, (Number, sp.Basic)):
            pass
        else:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Input arguments in call to numpy.{f} must be of dace.data.Data "
                "type or numerical/boolean constants (invalid argument {a})".format(f=ufunc_name, a=arg))

    return inputs


def _validate_ufunc_outputs(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, ufunc_name: str, num_inputs: int,
                            num_outputs: int, num_args: int, args: Sequence[UfuncInput],
                            kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Validates the number of type of outputs in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param num_outputs: Number of ufunc outputs
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of output datanames and None
    """

    # Validate number of outputs
    num_pos_outputs = num_args - num_inputs
    if num_pos_outputs == 0 and "out" not in kwargs.keys():
        outputs = [None] * num_outputs
    elif num_pos_outputs > 0 and "out" in kwargs.keys():
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "You cannot specify 'out' in call to numpy.{f} as both a positional"
            " and keyword argument (positional {p}, keyword {w}).".format(f=ufunc_name,
                                                                          p=args[num_outputs, :],
                                                                          k=kwargs['out']))
    elif num_pos_outputs > 0:
        outputs = list(args[num_inputs:])
        # TODO: Support the following undocumented NumPy behavior?
        # NumPy allows to specify less than `expected_num_outputs` as
        # positional arguments. For example, `np.divmod` has 2 outputs, the
        # quotient and the remainder. `np.divmod(A, B, C)` works, but
        # `np.divmod(A, B, out=C)` or `np.divmod(A, B, out=(C))` doesn't.
        # In the case of output as a positional argument, C will be set to
        # the quotient of the floor division, while a new array will be
        # generated for the remainder.
    else:
        outputs = kwargs["out"]
    if isinstance(outputs, (list, tuple)):
        outputs = list(outputs)
    else:
        outputs = [outputs]
    if len(outputs) != num_outputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} outputs, but {a} were given).".format(f=ufunc_name, e=num_outputs, a=len(outputs)))

    # Validate outputs
    for arg in outputs:
        if arg is None:
            pass
        elif isinstance(arg, str) and arg in sdfg.arrays.keys():
            pass
        else:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Return arguments in call to numpy.{f} must be of "
                "dace.data.Data type.".format(f=ufunc_name))

    return outputs


def _validate_where_kword(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, ufunc_name: str,
                          kwargs: Dict[str, Any]) -> Tuple[bool, Union[str, bool]]:
    """ Validates the 'where' keyword argument passed to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call

        :raises DaceSyntaxError: When validation fails

        :return: Tuple of a boolean value indicating whether the 'where'
                 keyword is defined, and the validated 'where' value
    """

    has_where = False
    where = None
    if 'where' in kwargs.keys():
        where = kwargs['where']
        if isinstance(where, str) and where in sdfg.arrays.keys():
            has_where = True
        elif isinstance(where, (bool, np.bool_)):
            has_where = True
        elif isinstance(where, (list, tuple)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Values for the 'where' keyword that are a sequence of boolean "
                " constants are unsupported. Please, pass these values to the "
                " {n} call through a DaCe boolean array.".format(n=ufunc_name))
        else:
            # NumPy defaults to "where=True" for invalid values for the keyword
            pass

    return has_where, where


def _validate_shapes(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, ufunc_name: str, inputs: List[UfuncInput],
                     outputs: List[UfuncOutput]) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Validates the data shapes of inputs and outputs to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: Tuple with the output shape, the map, output and input indices
    """

    shapes = []
    for arg in inputs + outputs:
        if isinstance(arg, str):
            array = sdfg.arrays[arg]
            shapes.append(array.shape)
        else:
            shapes.append([])
    try:
        result = _broadcast(shapes)
    except SyntaxError as e:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Shape validation in numpy.{f} call failed. The following error "
            "occured : {m}".format(f=ufunc_name, m=str(e)))
    return result


def _broadcast(shapes: Sequence[Shape]) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Applies the NumPy ufunc brodacsting rules in a sequence of data shapes
        (see https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting).

        :param shapes: Sequence (list, tuple) of data shapes

        :raises SyntaxError: When broadcasting fails

        :return: Tuple with the output shape, the map, output and input indices
    """

    map_lengths = dict()
    output_indices = []
    input_indices = [[] for _ in shapes]

    ndims = [len(shape) for shape in shapes]
    max_i = max(ndims)

    def get_idx(i):
        return "__i" + str(max_i - i - 1)

    def to_string(idx):
        return ", ".join(reversed(idx))

    reversed_shapes = [reversed(shape) for shape in shapes]
    for i, dims in enumerate(itertools.zip_longest(*reversed_shapes)):
        output_indices.append(get_idx(i))

        not_none_dims = [d for d in dims if d is not None]
        # Per NumPy broadcasting rules, we need to find the largest dimension.
        # However, `max_dim = max(not_none_dims)` does not work with symbols.
        # Therefore, we sequentially check every not-none dimension.
        # Symbols are assumed to be larger than constants.
        # This will not work properly otherwise.
        # If more than 1 (different) symbols are found, then this fails, because
        # we cannot know which will have the greater size.
        # NOTE: This is a compromise. NumPy broadcasting depends on knowing
        # the exact array sizes. However, symbolic sizes are not known at this
        # point.
        max_dim = 0
        for d in not_none_dims:
            if isinstance(max_dim, Number):
                if isinstance(d, Number):
                    max_dim = max(max_dim, d)
                elif symbolic.issymbolic(d):
                    max_dim = d
                else:
                    raise NotImplementedError
            elif symbolic.issymbolic(max_dim):
                if isinstance(d, Number):
                    pass
                elif symbolic.issymbolic(d):
                    if max_dim != d:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        map_lengths[get_idx(i)] = max_dim
        for j, d in enumerate(dims):
            if d is None:
                pass
            elif d == 1:
                input_indices[j].append('0')
            elif d == max_dim:
                input_indices[j].append(get_idx(i))
            else:
                raise SyntaxError("Operands could not be broadcast together with shapes {}.".format(','.join(
                    str(shapes))))

    out_shape = tuple(reversed([map_lengths[idx] for idx in output_indices]))
    map_indices = [(k, "0:" + str(map_lengths[k])) for k in reversed(output_indices)]
    output_indices = to_string(output_indices)
    input_indices = [to_string(idx) for idx in input_indices]

    if not out_shape:
        out_shape = (1, )
        output_indices = "0"

    return out_shape, map_indices, output_indices, input_indices


def _create_output(sdfg: SDFG,
                   inputs: List[UfuncInput],
                   outputs: List[UfuncOutput],
                   output_shape: Shape,
                   output_dtype: Union[dtypes.typeclass, List[dtypes.typeclass]],
                   storage: dtypes.StorageType = None,
                   force_scalar: bool = False,
                   name_hint: Optional[str] = None) -> List[UfuncOutput]:
    """ Creates output data for storing the result of a NumPy ufunc call.

        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call
        :param output_shape: Shape of the output data
        :param output_dtype: Datatype of the output data
        :param storage: Storage type of the output data
        :param force_scalar: If True and output shape is (1,) then output
                             becomes a ``dace.data.Scalar``, regardless of the data-type of the inputs
        :param name_hint: Optional name hint for the output data
        :return: New outputs of the ufunc call
    """

    # Check if the result is scalar
    is_output_scalar = True
    for arg in inputs:
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            datadesc = sdfg.arrays[arg]
            # If storage is not set, then choose the storage of the first
            # data input.
            if not storage:
                storage = datadesc.storage
            # TODO: What about streams?
            if not isinstance(datadesc, data.Scalar):
                is_output_scalar = False
                break

    # Set storage
    storage = storage or dtypes.StorageType.Default

    # Validate datatypes
    if isinstance(output_dtype, (list, tuple)):
        if len(output_dtype) == 1:
            datatypes = [output_dtype[0]] * len(outputs)
        elif len(output_dtype) == len(outputs):
            datatypes = output_dtype
        else:
            raise ValueError("Missing output datatypes")
    else:
        datatypes = [output_dtype] * len(outputs)

    # Create output data (if needed)
    for i, (arg, datatype) in enumerate(zip(outputs, datatypes)):
        if arg is None:
            output_name = name_hint or sdfg.temp_data_name()
            if (len(output_shape) == 1 and output_shape[0] == 1 and (is_output_scalar or force_scalar)):
                output_name, _ = sdfg.add_scalar(output_name,
                                                 output_dtype,
                                                 transient=True,
                                                 storage=storage,
                                                 find_new_name=True)
                outputs[i] = output_name
            else:
                outputs[i], _ = sdfg.add_transient(output_name, output_shape, datatype, find_new_name=True)

    return outputs


def _set_tasklet_params(ufunc_impl: Dict[str, Any],
                        inputs: List[UfuncInput],
                        casting: List[dtypes.typeclass] = None) -> Dict[str, Any]:
    """ Sets the tasklet parameters for a NumPy ufunc call.

        :param ufunc_impl: Information on how the ufunc must be implemented
        :param inputs: Inputs of the ufunc call

        :return: Dictionary with the (1) tasklet name, (2) input connectors,
                 (3) output connectors, and (4) tasklet code
    """

    # (Deep) copy default tasklet parameters from the ufunc_impl dictionary
    name = ufunc_impl['name']
    inp_connectors = copy.deepcopy(ufunc_impl['inputs'])
    out_connectors = copy.deepcopy(ufunc_impl['outputs'])
    code = ufunc_impl['code']

    # Remove input connectors related to constants
    # and fix constants/symbols in the tasklet code
    for i, arg in reversed(list(enumerate(inputs))):
        inp_conn = inp_connectors[i]
        if casting and casting[i]:
            repl = "{c}({o})".format(c=str(casting[i]).replace('::', '.'), o=inp_conn)
            code = code.replace(inp_conn, repl)
        if isinstance(arg, (Number, sp.Basic)):
            inp_conn = inp_connectors[i]
            code = code.replace(inp_conn, astutils.unparse(arg))
            inp_connectors.pop(i)

    return dict(name=name, inputs=inp_connectors, outputs=out_connectors, code=code)


def _create_subgraph(visitor: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     inputs: List[UfuncInput],
                     outputs: List[UfuncOutput],
                     map_indices: Tuple[str, str],
                     input_indices: List[str],
                     output_indices: str,
                     output_shape: Shape,
                     tasklet_params: Dict[str, Any],
                     has_where: bool = False,
                     where: Union[str, bool] = None):
    """ Creates the subgraph that implements a NumPy ufunc call.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call
        :param map_indices: Map (if needed) indices
        :param input_indices: Input indices for inner-most memlets
        :param output_indices: Output indices for inner-most memlets
        :param output_shape: Shape of the output
        :param tasklet_params: Dictionary with the tasklet parameters
        :param has_where: True if the 'where' keyword is set
        :param where: Keyword 'where' value
    """

    # Create subgraph
    if list(output_shape) == [1]:
        # No map needed
        if has_where:
            if isinstance(where, (bool, np.bool_)):
                if where == True:
                    pass
                elif where == False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                cond_state = state
                where_data = sdfg.arrays[where]
                if not isinstance(where_data, data.Scalar):
                    name = 'where_cond'
                    name, _ = sdfg.add_scalar(name, where_data.dtype, transient=True, find_new_name=True)
                    r = cond_state.add_read(where)
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, Memlet("{}[0]".format(r)))
                true_state = sdfg.add_state(label=cond_state.label + '_true')
                state = true_state
                visitor.last_block = state
                cond = name
                cond_else = 'not ({})'.format(cond)
                sdfg.add_edge(cond_state, true_state, InterstateEdge(cond))
        tasklet = state.add_tasklet(**tasklet_params)
        inp_conn_idx = 0
        for arg in inputs:
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                inp_node = state.add_read(arg)
                state.add_edge(inp_node, None, tasklet, tasklet_params['inputs'][inp_conn_idx],
                               Memlet.from_array(arg, sdfg.arrays[arg]))
                inp_conn_idx += 1
        for i, arg in enumerate(outputs):
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                out_node = state.add_write(arg)
                state.add_edge(tasklet, tasklet_params['outputs'][i], out_node, None,
                               Memlet.from_array(arg, sdfg.arrays[arg]))
        if has_where and isinstance(where, str) and where in sdfg.arrays.keys():
            visitor._add_state(label=cond_state.label + '_true')
            sdfg.add_edge(cond_state, visitor.last_block, InterstateEdge(cond_else))
    else:
        # Map needed
        if has_where:
            if isinstance(where, (bool, np.bool_)):
                if where == True:
                    pass
                elif where == False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                nested_sdfg = SDFG(state.label + "_where")
                nested_sdfg_inputs = dict()
                nested_sdfg_outputs = dict()

                for idx, arg in enumerate(inputs + [where]):
                    if not (isinstance(arg, str) and arg in sdfg.arrays.keys()):
                        continue
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg._find_new_name(arg)
                    nested_sdfg_inputs[arg] = (conn_name, input_indices[idx])
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError

                for arg in outputs:
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg._find_new_name(arg)
                    nested_sdfg_outputs[arg] = (conn_name, output_indices)
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError

                cond_state = nested_sdfg.add_state(label=state.label + "_where_cond", is_start_block=True)
                where_data = sdfg.arrays[where]
                if isinstance(where_data, data.Scalar):
                    name = nested_sdfg_inputs[where]
                elif isinstance(where_data, data.Array):
                    name = nested_sdfg._find_new_name(where)
                    nested_sdfg.add_scalar(name, where_data.dtype, transient=True)
                    r = cond_state.add_read(nested_sdfg_inputs[where][0])
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, Memlet("{}[0]".format(r)))

                true_state = nested_sdfg.add_state(label=cond_state.label + '_where_true')
                cond = name
                cond_else = 'not ({})'.format(cond)
                nested_sdfg.add_edge(cond_state, true_state, InterstateEdge(cond))

                tasklet = true_state.add_tasklet(**tasklet_params)
                idx = 0
                for arg in inputs:
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        inp_name, _ = nested_sdfg_inputs[arg]
                        inp_data = nested_sdfg.arrays[inp_name]
                        inp_node = true_state.add_read(inp_name)
                        true_state.add_edge(inp_node, None, tasklet, tasklet_params['inputs'][idx],
                                            Memlet.from_array(inp_name, inp_data))
                        idx += 1
                for i, arg in enumerate(outputs):
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        out_name, _ = nested_sdfg_outputs[arg]
                        out_data = nested_sdfg.arrays[out_name]
                        out_node = true_state.add_write(out_name)
                        true_state.add_edge(tasklet, tasklet_params['outputs'][i], out_node, None,
                                            Memlet.from_array(out_name, out_data))

                false_state = nested_sdfg.add_state(label=state.label + '_where_false')
                nested_sdfg.add_edge(cond_state, false_state, InterstateEdge(cond_else))
                nested_sdfg.add_edge(true_state, false_state, InterstateEdge())

                codenode = state.add_nested_sdfg(nested_sdfg, set([n for n, _ in nested_sdfg_inputs.values()]),
                                                 set([n for n, _ in nested_sdfg_outputs.values()]))
                me, mx = state.add_map(state.label + '_map', map_indices)
                for arg in inputs + [where]:
                    if not (isinstance(arg, str) and arg in sdfg.arrays.keys()):
                        continue
                    n = state.add_read(arg)
                    conn, idx = nested_sdfg_inputs[arg]
                    state.add_memlet_path(n, me, codenode, memlet=Memlet("{a}[{i}]".format(a=n, i=idx)), dst_conn=conn)
                for arg in outputs:
                    n = state.add_write(arg)
                    conn, idx = nested_sdfg_outputs[arg]
                    state.add_memlet_path(codenode, mx, n, memlet=Memlet("{a}[{i}]".format(a=n, i=idx)), src_conn=conn)

                dealias.integrate_nested_sdfg(nested_sdfg)
                return

        input_memlets = dict()
        inp_conn_idx = 0
        for arg, idx in zip(inputs, input_indices):
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                conn = tasklet_params['inputs'][inp_conn_idx]
                input_memlets[conn] = Memlet.simple(arg, idx)
                inp_conn_idx += 1
        output_memlets = {
            out_conn: Memlet.simple(arg, output_indices)
            for arg, out_conn in zip(outputs, tasklet_params['outputs'])
        }
        state.add_mapped_tasklet(tasklet_params['name'],
                                 map_indices,
                                 input_memlets,
                                 tasklet_params['code'],
                                 output_memlets,
                                 external_edges=True)


def _flatten_args(args: Sequence[UfuncInput]) -> Sequence[UfuncInput]:
    """ Flattens arguments of a NumPy ufunc. This is useful in cases where
        one of the arguments is the result of another operation or ufunc, which
        may be a list of Dace data.
    """
    flat_args = []
    for arg in args:
        if isinstance(arg, list):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)
    return flat_args


@oprepo.replaces_ufunc('ufunc')
def implement_ufunc(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                    args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of output datanames
    """
    from dace.frontend.python.replacements.operators import result_type

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg, ufunc_name, kwargs)

    # Validate data shapes and apply NumPy broadcasting rules
    inp_shapes = copy.deepcopy(inputs)
    if has_where:
        inp_shapes += [where]
    (out_shape, map_indices, out_indices, inp_indices) = _validate_shapes(visitor, ast_node, sdfg, ufunc_name,
                                                                          inp_shapes, outputs)

    # Infer result type
    result_type, casting = result_type(
        [sdfg.arrays[arg] if isinstance(arg, str) and arg in sdfg.arrays else arg for arg in inputs],
        ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.dtype_to_typeclass().keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type, name_hint=visitor.get_target_name())

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor,
                     sdfg,
                     state,
                     inputs,
                     outputs,
                     map_indices,
                     inp_indices,
                     out_indices,
                     out_shape,
                     tasklet_params,
                     has_where=has_where,
                     where=where)

    return outputs


def _validate_keepdims_kword(visitor: ProgramVisitor, ast_node: ast.Call, ufunc_name: str, kwargs: Dict[str,
                                                                                                        Any]) -> bool:
    """ Validates the 'keepdims' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: Boolean value of the 'keepdims' keyword argument
    """

    keepdims = False
    if 'keepdims' in kwargs.keys():
        keepdims = kwargs['keepdims']
        if not isinstance(keepdims, (Integral, bool, np.bool_)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Integer or boolean value expected for keyword argument "
                "'keepdims' in reduction operation {f} (got {v}).".format(f=ufunc_name, v=keepdims))
        if not isinstance(keepdims, (bool, np.bool_)):
            keepdims = bool(keepdims)

    return keepdims


def _validate_axis_kword(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, inputs: List[UfuncInput],
                         kwargs: Dict[str, Any], keepdims: bool) -> Tuple[Tuple[int, ...], Union[Shape, None], Shape]:
    """ Validates the 'axis' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call
        :param keepdims: Boolean value of the 'keepdims' keyword argument

        :raises DaCeSyntaxError: When validation fails

        :return: The value of the 'axis' keyword argument, the intermediate
                 data shape (if needed), and the expected output shape
    """

    # Validate 'axis' keyword
    axis = (0, )
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        inp_shape = sdfg.arrays[inputs[0]].shape
    else:
        inp_shape = [1]
    if 'axis' in kwargs.keys():
        # Set to (0, 1, 2, ...) if the keyword arg value is None
        if kwargs['axis'] is None:
            axis = tuple(range(len(inp_shape)))
        else:
            axis = kwargs['axis']
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
    if axis is not None:
        axis = tuple(symbolic.pystr_to_symbolic(a) for a in axis)
        axis = tuple(normalize_axes(axis, len(inp_shape)))
        if len(axis) > len(inp_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Axis {a} is out of bounds for data of dimension {d}".format(a=axis, d=inp_shape))
        for a in axis:
            if a >= len(inp_shape):
                raise mem_parser.DaceSyntaxError(
                    visitor, ast_node, "Axis {a} is out of bounds for data of dimension {d}".format(a=a, d=inp_shape))
        if keepdims:
            intermediate_shape = [d for i, d in enumerate(inp_shape) if i not in axis]
            expected_out_shape = [d if i not in axis else 1 for i, d in enumerate(inp_shape)]
        else:
            intermediate_shape = None
            expected_out_shape = [d for i, d in enumerate(inp_shape) if i not in axis]
        expected_out_shape = expected_out_shape or [1]
        intermediate_shape = intermediate_shape or [1]
    else:
        if keepdims:
            intermediate_shape = [1]
            expected_out_shape = [1] * len(inp_shape)
        else:
            intermediate_shape = None
            expected_out_shape = [1]

    return axis, intermediate_shape, expected_out_shape


@oprepo.replaces_ufunc('reduce')
def implement_ufunc_reduce(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                           args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'reduce' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of output datanames
    """
    from dace.frontend.python.replacements.reduction import reduce

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'keepdims' keyword
    keepdims = _validate_keepdims_kword(visitor, ast_node, ufunc_name, kwargs)

    # Validate 'axis' keyword
    axis, intermediate_shape, expected_out_shape = _validate_axis_kword(visitor, ast_node, sdfg, inputs, kwargs,
                                                                        keepdims)

    # Validate 'where' keyword
    # Throw a warning that it is currently unsupported.
    if 'where' in kwargs.keys():
        warnings.warn("Keyword argument 'where' in 'reduce' method of NumPy "
                      "ufunc calls is unsupported. It will be ignored.")

    # Validate data shapes and apply NumPy broadcasting rules
    # In the case of reduce we may only validate the broadcasting of the
    # single input with the 'where' value. Since 'where' is currently
    # unsupported, only validate output shape.
    # TODO: Maybe add special error when 'keepdims' is True
    if isinstance(outputs[0], str) and outputs[0] in sdfg.arrays.keys():
        out_shape = sdfg.arrays[outputs[0]].shape
        if len(out_shape) < len(expected_out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} does not have "
                "enough dimensions (output shape {o}, expected shape {e}).".format(f=ufunc_name,
                                                                                   o=out_shape,
                                                                                   e=expected_out_shape))
        if len(out_shape) > len(expected_out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} has too many "
                "dimensions (output shape {o}, expected shape {e}).".format(f=ufunc_name,
                                                                            o=out_shape,
                                                                            e=expected_out_shape))
        if (list(out_shape) != list(expected_out_shape)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} has non-reduction"
                " dimension not equal to the input one (output shape {o}, "
                "expected shape {e}).".format(f=ufunc_name, o=out_shape, e=expected_out_shape))
    else:
        out_shape = expected_out_shape

    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str):
        datadesc = sdfg.arrays[arg]
        result_type = datadesc.dtype
    elif isinstance(arg, (Number, np.bool_)):
        result_type = dtypes.dtype_to_typeclass(type(arg))
    elif isinstance(arg, sp.Basic):
        result_type = sym_type(arg)

    # Create output data (if needed)
    outputs = _create_output(sdfg,
                             inputs,
                             outputs,
                             out_shape,
                             result_type,
                             force_scalar=True,
                             name_hint=visitor.get_target_name())
    if keepdims:
        intermediate_name = visitor.get_target_name() + '_keepdims'
        if (len(intermediate_shape) == 1 and intermediate_shape[0] == 1):
            intermediate_name, _ = sdfg.add_scalar(intermediate_name, result_type, transient=True, find_new_name=True)
        else:
            intermediate_name, _ = sdfg.add_transient(intermediate_name,
                                                      intermediate_shape,
                                                      result_type,
                                                      find_new_name=True)
    else:
        intermediate_name = outputs[0]

    # Validate 'initial' keyword
    # This is set to be ufunc.identity, when it exists
    initial = ufunc_impl['initial']
    if 'initial' in kwargs.keys():
        # NumPy documentation says that when 'initial' is set to None,
        # then the first element of the reduction is used. However, it seems
        # that when 'initial' is None and the ufunc has 'identity', then
        # ufunc.identity is the default.
        initial = kwargs['initial'] or initial
        if initial is None:
            if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
                inpdata = sdfg.arrays[inputs[0]]
                # In the input data has more than 1 dimensions and 'initial'
                # is None, then NumPy uses a different 'initial' value for every
                # non-reduced dimension.
                if isinstance(inpdata, data.Array):
                    state.add_mapped_tasklet(name=state.label + "_reduce_initial",
                                             map_ranges={
                                                 "__i{i}".format(i=i): "0:{s}".format(s=s)
                                                 for i, s in enumerate(inpdata.shape) if i not in axis
                                             },
                                             inputs={
                                                 "__inp":
                                                 Memlet("{a}[{i}]".format(a=inputs[0],
                                                                          i=','.join([
                                                                              "0" if i in axis else "__i{i}".format(i=i)
                                                                              for i in range(len(inpdata.shape))
                                                                          ])))
                                             },
                                             outputs={
                                                 "__out":
                                                 Memlet("{a}[{i}]".format(a=intermediate_name,
                                                                          i=','.join([
                                                                              "__i{i}".format(i=i)
                                                                              for i in range(len(inpdata.shape))
                                                                              if i not in axis
                                                                          ])))
                                             },
                                             code="__out = __inp",
                                             external_edges=True)
                else:
                    r = state.add_read(inputs[0])
                    w = state.add_write(intermediate_name)
                    state.add.nedge(r, w, Memlet.from_array(inputs[0], inpdata))
                state = visitor._add_state(state.label + 'b')
            else:
                initial = intermediate_name

    # Special case for infinity
    if np.isinf(initial):
        if np.sign(initial) < 0:
            initial = dtypes.min_value(result_type)
        else:
            initial = dtypes.max_value(result_type)

    # Create subgraph
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        reduce(visitor, sdfg, state, ufunc_impl['reduce'], inputs[0], intermediate_name, axis=axis, identity=initial)
    else:
        tasklet = state.add_tasklet(state.label + "_tasklet", {}, {'__out'}, "__out = {}".format(inputs[0]))
        out_node = state.add_write(intermediate_name)
        datadesc = sdfg.arrays[intermediate_name]
        state.add_edge(tasklet, '__out', out_node, None, Memlet.from_array(intermediate_name, datadesc))

    if keepdims:
        intermediate_node = None
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == intermediate_name:
                intermediate_node = n
                break
        if not intermediate_node:
            raise ValueError("Keyword argument 'keepdims' is True, but "
                             "intermediate access node was not found.")
        out_node = state.add_write(outputs[0])
        state.add_nedge(intermediate_node, out_node, Memlet.from_array(outputs[0], sdfg.arrays[outputs[0]]))

    return outputs


@oprepo.replaces_ufunc('accumulate')
def implement_ufunc_accumulate(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, state: SDFGState,
                               ufunc_name: str, args: Sequence[UfuncInput], kwargs: Dict[str,
                                                                                         Any]) -> List[UfuncOutput]:
    """ Implements the 'accumulate' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of output datanames
    """

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str) and arg in sdfg.arrays.keys():
        datadesc = sdfg.arrays[arg]
        if not isinstance(datadesc, data.Array):
            raise mem_parser.DaceSyntaxError(visitor, ast_node,
                                             "Cannot accumulate on a dace.data.Scalar or dace.data.Stream.")
        out_shape = datadesc.shape
        result_type = datadesc.dtype
    else:
        raise mem_parser.DaceSyntaxError(visitor, ast_node, "Can accumulate only on a dace.data.Array.")

    # Validate 'axis' keyword argument
    axis = 0
    if 'axis' in kwargs.keys():
        axis = kwargs['axis'] or axis
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            axis = axis[0]
        if not isinstance(axis, Integral):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Value of keyword argument 'axis' in 'accumulate' method of {f}"
                " must be an integer (value {v}).".format(f=ufunc_name, v=axis))
        if axis >= len(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Axis {a} is out of bounds for dace.data.Array of dimension "
                "{l}".format(a=axis, l=len(out_shape)))
        # Normalize negative axis
        axis = normalize_axes([axis], len(out_shape))[0]

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type, name_hint=visitor.get_target_name())

    # Create subgraph
    shape = datadesc.shape
    map_range = {"__i{}".format(i): "0:{}".format(s) for i, s in enumerate(shape) if i != axis}
    input_idx = ','.join(["__i{}".format(i) if i != axis else "0:{}".format(shape[i]) for i in range(len(shape))])
    output_idx = ','.join(["__i{}".format(i) if i != axis else "0:{}".format(shape[i]) for i in range(len(shape))])

    nested_sdfg = SDFG(state.label + "_for_loop")
    inpconn = nested_sdfg._find_new_name(arg)
    outconn = nested_sdfg._find_new_name(outputs[0])
    shape = [datadesc.shape[axis]]
    strides = [datadesc.strides[axis]]
    nested_sdfg.add_array(inpconn, shape, result_type, strides=strides)
    nested_sdfg.add_array(outconn, shape, result_type, strides=strides)

    init_state = nested_sdfg.add_state(label="init")
    r = init_state.add_read(inpconn)
    w = init_state.add_write(outconn)
    init_state.add_nedge(r, w, Memlet("{a}[{i}] -> [{oi}]".format(a=inpconn, i='0', oi='0')))

    body_state = nested_sdfg.add_state(label="body")
    r1 = body_state.add_read(inpconn)
    r2 = body_state.add_read(outconn)
    w = body_state.add_write(outconn)
    t = body_state.add_tasklet(name=state.label + "_for_loop_tasklet",
                               inputs=ufunc_impl['inputs'],
                               outputs=ufunc_impl['outputs'],
                               code=ufunc_impl['code'])

    loop_idx = "__i{}".format(axis)
    loop_idx_m1 = "__i{} - 1".format(axis)
    body_state.add_edge(r1, None, t, '__in1', Memlet("{a}[{i}]".format(a=inpconn, i=loop_idx)))
    body_state.add_edge(r2, None, t, '__in2', Memlet("{a}[{i}]".format(a=outconn, i=loop_idx_m1)))
    body_state.add_edge(t, '__out', w, None, Memlet("{a}[{i}]".format(a=outconn, i=loop_idx)))

    init_expr = str(1)
    cond_expr = "__i{i} < {s}".format(i=axis, s=shape[0])
    incr_expr = "__i{} + 1".format(axis)
    nested_sdfg.add_loop(init_state, body_state, None, loop_idx, init_expr, cond_expr, incr_expr)

    r = state.add_read(inputs[0])
    w = state.add_write(outputs[0])
    codenode = state.add_nested_sdfg(nested_sdfg, {inpconn}, {outconn})
    me, mx = state.add_map(state.label + '_map', map_range)
    state.add_memlet_path(r, me, codenode, memlet=Memlet("{a}[{i}]".format(a=inputs[0], i=input_idx)), dst_conn=inpconn)
    state.add_memlet_path(codenode,
                          mx,
                          w,
                          memlet=Memlet("{a}[{i}]".format(a=outputs[0], i=output_idx)),
                          src_conn=outconn)

    dealias.integrate_nested_sdfg(nested_sdfg)
    return outputs


@oprepo.replaces_ufunc('outer')
def implement_ufunc_outer(visitor: ProgramVisitor, ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                          args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'outer' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :return: List of output datanames
    """
    from dace.frontend.python.replacements.operators import result_type

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg, ufunc_name, kwargs)

    # Validate data shapes
    out_shape = []
    map_vars = []
    map_range = dict()
    input_indices = []
    output_idx = None
    for i, arg in enumerate(inputs):
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            datadesc = sdfg.arrays[arg]
            if isinstance(datadesc, data.Scalar):
                input_idx = '0'
            elif isinstance(datadesc, data.Array):
                shape = datadesc.shape
                out_shape.extend(shape)
                map_vars.extend(["__i{i}_{j}".format(i=i, j=j) for j in range(len(shape))])
                map_range.update({"__i{i}_{j}".format(i=i, j=j): "0:{}".format(sz) for j, sz in enumerate(shape)})
                input_idx = ','.join(["__i{i}_{j}".format(i=i, j=j) for j in range(len(shape))])
                if output_idx:
                    output_idx = ','.join([output_idx, input_idx])
                else:
                    output_idx = input_idx
            else:
                raise mem_parser.DaceSyntaxError(
                    visitor, ast_node, "Unsuported data type {t} in 'outer' method of NumPy ufunc "
                    "{f}.".format(t=type(datadesc), f=ufunc_name))
        elif isinstance(arg, (Number, sp.Basic)):
            input_idx = None
        input_indices.append(input_idx)

    if has_where and not isinstance(where, (bool, np.bool_)):
        where_shape = sdfg.arrays[where].shape
        try:
            bcast_out_shape, _, _, bcast_inp_indices = _broadcast([out_shape, where_shape])
        except SyntaxError:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "'where' shape {w} could not be broadcast together with 'out' "
                "shape {o}.".format(w=where_shape, o=out_shape))
        if list(bcast_out_shape) != list(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Broadcasting 'where' shape {w} together with expected 'out' "
                "shape {o} resulted in a different output shape {no}. This is "
                "currently unsupported.".format(w=where_shape, o=out_shape, no=bcast_out_shape))
        where_idx = bcast_inp_indices[1]
        for i in range(len(out_shape)):
            where_idx = where_idx.replace("__i{}".format(i), map_vars[i])
        input_indices.append(where_idx)
    else:
        input_indices.append(None)

    # Infer result type
    result_type, casting = result_type(
        [sdfg.arrays[arg] if isinstance(arg, str) and arg in sdfg.arrays else arg for arg in inputs],
        ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.dtype_to_typeclass().keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type, name_hint=visitor.get_target_name())

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor,
                     sdfg,
                     state,
                     inputs,
                     outputs,
                     map_range,
                     input_indices,
                     output_idx,
                     out_shape,
                     tasklet_params,
                     has_where=has_where,
                     where=where)

    return outputs


@oprepo.replaces_method('Array', 'sum')
@oprepo.replaces_method('Scalar', 'sum')
@oprepo.replaces_method('View', 'sum')
def _ndarray_sum(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'add', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'mean')
@oprepo.replaces_method('Scalar', 'mean')
@oprepo.replaces_method('View', 'mean')
def _ndarray_mean(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    kwargs = kwargs or dict(axis=None)
    sumarr = implement_ufunc_reduce(pv, None, sdfg, nest.add_state(), 'add', [arr], kwargs)[0]
    desc = sdfg.arrays[arr]
    sz = functools.reduce(lambda x, y: x * y, desc.shape)
    return nest, elementwise(pv, sdfg, nest.add_state(), "lambda x: x / {}".format(sz), sumarr)


@oprepo.replaces_method('Array', 'prod')
@oprepo.replaces_method('Scalar', 'prod')
@oprepo.replaces_method('View', 'prod')
def _ndarray_prod(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'multiply', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'all')
@oprepo.replaces_method('Scalar', 'all')
@oprepo.replaces_method('View', 'all')
def _ndarray_all(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'logical_and', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'any')
@oprepo.replaces_method('Scalar', 'any')
@oprepo.replaces_method('View', 'any')
def _ndarray_any(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'logical_or', [arr], kwargs)[0]


@oprepo.replaces('numpy.clip')
def _clip(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a, a_min=None, a_max=None, **kwargs):
    if a_min is None and a_max is None:
        raise ValueError("clip() requires at least one of `a_min` or `a_max`")
    if a_min is None:
        return implement_ufunc(pv, None, sdfg, state, 'minimum', [a, a_max], kwargs)[0]
    if a_max is None:
        return implement_ufunc(pv, None, sdfg, state, 'maximum', [a, a_min], kwargs)[0]
    return implement_ufunc(pv, None, sdfg, state, 'clip', [a, a_min, a_max], kwargs)[0]
