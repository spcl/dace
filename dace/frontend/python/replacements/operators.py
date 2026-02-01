# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains operator replacements (e.g., NumPy Mathematical Functions) for supported objects.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import (ProgramVisitor, broadcast_together, cast_str, np_result_type,
                                                     representative_num, sym_type)
from dace import data, dtypes, subsets, symbolic, Memlet, SDFG, SDFGState

from numbers import Number
from typing import List, Sequence, Tuple, Union
import warnings

import numpy as np
import sympy as sp
import dace  # For evaluation of data types

numpy_version = np.lib.NumpyVersion(np.__version__)


def _unop(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, opcode: str, opname: str):
    """ Implements a general element-wise array unary operator. """
    arr1 = sdfg.arrays[op1]

    restype, cast = result_type([arr1], opname)
    tasklet_code = "__out = {} __in1".format(opcode)
    if cast:
        tasklet_code = tasklet_code.replace('__in1', "{}(__in1)".format(cast))

    # NOTE: This is a fix for np.bool_, which is a true boolean.
    # In this case, the invert operator must become a not operator.
    if opcode == '~' and arr1.dtype == dtypes.bool_:
        opcode = 'not'

    name, _ = pv.add_temp_transient(arr1.shape, restype, arr1.storage)
    state.add_mapped_tasklet("_%s_" % opname, {
        '__i%d' % i: '0:%s' % s
        for i, s in enumerate(arr1.shape)
    }, {'__in1': Memlet.simple(op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))},
                             '__out = %s __in1' % opcode,
                             {'__out': Memlet.simple(name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))},
                             external_edges=True)
    return name


# Defined as a function in order to include the op and the opcode in the closure
def _makeunop(op, opcode):

    @oprepo.replaces_operator('Array', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        return _unop(visitor, sdfg, state, op1, opcode, op)

    @oprepo.replaces_operator('View', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        return _unop(visitor, sdfg, state, op1, opcode, op)

    @oprepo.replaces_operator('Scalar', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        scalar1 = sdfg.arrays[op1]
        restype, _ = result_type([scalar1], op)
        op2 = visitor.get_target_name()
        op2, scalar2 = sdfg.add_scalar(op2, restype, transient=True, find_new_name=True)
        tasklet = state.add_tasklet("_%s_" % op, {'__in'}, {'__out'}, "__out = %s __in" % opcode)
        node1 = state.add_read(op1)
        node2 = state.add_write(op2)
        state.add_edge(node1, None, tasklet, '__in', Memlet.from_array(op1, scalar1))
        state.add_edge(tasklet, '__out', node2, None, Memlet.from_array(op2, scalar2))
        return op2

    @oprepo.replaces_operator('NumConstant', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: Number, op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)

    @oprepo.replaces_operator('BoolConstant', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: Number, op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)

    @oprepo.replaces_operator('symbol', op)
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: symbolic.symbol, op2=None):
        if opcode in _pyop2symtype.keys():
            try:
                return _pyop2symtype[opcode](op1)
            except TypeError:
                pass
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)


def _is_op_arithmetic(op: str):
    if op in {'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Pow', 'Mod', 'FloatPow', 'Heaviside', 'Arctan2', 'Hypot'}:
        return True
    return False


def _is_op_bitwise(op: str):
    if op in {'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'Invert'}:
        return True
    return False


def _is_op_boolean(op: str):
    if op in {
            'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'NotIs', 'Xor', 'FpBoolean', 'SignBit'
    }:
        return True
    return False


def result_type(arguments: Sequence[Union[str, Number, symbolic.symbol, sp.Basic]],
                operator: str = None) -> Tuple[Union[List[dtypes.typeclass], dtypes.typeclass, str], ...]:

    datatypes = []
    dtypes_for_result = []
    dtypes_for_result_np2 = []
    for arg in arguments:
        if isinstance(arg, (data.Array, data.Stream)):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(arg.dtype.type)
            dtypes_for_result_np2.append(arg.dtype.type)
        elif isinstance(arg, data.Scalar):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(representative_num(arg.dtype))
            dtypes_for_result_np2.append(arg.dtype.type)
        elif isinstance(arg, (Number, np.bool_)):
            datatypes.append(dtypes.dtype_to_typeclass(type(arg)))
            dtypes_for_result.append(arg)
            dtypes_for_result_np2.append(arg)
        elif symbolic.issymbolic(arg):
            datatypes.append(sym_type(arg))
            dtypes_for_result.append(representative_num(sym_type(arg)))
            dtypes_for_result_np2.append(sym_type(arg).type)
        elif isinstance(arg, dtypes.typeclass):
            datatypes.append(arg)
            dtypes_for_result.append(representative_num(arg))
            dtypes_for_result_np2.append(arg.type)
        else:
            raise TypeError("Type {t} of argument {a} is not supported".format(t=type(arg), a=arg))

    complex_types = {dtypes.complex64, dtypes.complex128, np.complex64, np.complex128}
    float_types = {dtypes.float16, dtypes.float32, dtypes.float64, np.float16, np.float32, np.float64}
    signed_types = {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, np.int8, np.int16, np.int32, np.int64}
    # unsigned_types = {np.uint8, np.uint16, np.uint32, np.uint64}

    coarse_types = []
    for dtype in datatypes:
        if dtype in complex_types:
            coarse_types.append(3)  # complex
        elif dtype in float_types:
            coarse_types.append(2)  # float
        elif dtype in signed_types:
            coarse_types.append(1)  # signed integer, bool
        else:
            coarse_types.append(0)  # unsigned integer

    casting = [None] * len(arguments)

    if len(arguments) == 1:  # Unary operators

        if not operator:
            restype = datatypes[0]
        elif operator == 'USub' and coarse_types[0] == 0:
            restype = eval('dtypes.int{}'.format(8 * datatypes[0].bytes))
        elif operator == 'Abs' and coarse_types[0] == 3:
            restype = eval('dtypes.float{}'.format(4 * datatypes[0].bytes))
        elif (operator in ('Fabs', 'Cbrt', 'Angles', 'SignBit', 'Spacing', 'Modf', 'Floor', 'Ceil', 'Trunc')
              and coarse_types[0] == 3):
            raise TypeError("ufunc '{}' not supported for complex input".format(operator))
        elif operator in ('Ceil', 'Floor', 'Trunc') and coarse_types[0] < 2 and numpy_version < '2.1.0':
            restype = dtypes.float64
            casting[0] = cast_str(restype)
        elif (operator in ('Fabs', 'Rint', 'Exp', 'Log', 'Sqrt', 'Cbrt', 'Trigonometric', 'Angles', 'FpBoolean',
                           'Spacing', 'Modf') and coarse_types[0] < 2):
            restype = dtypes.float64
            casting[0] = cast_str(restype)
        elif operator in ('Frexp'):
            if coarse_types[0] == 3:
                raise TypeError("ufunc '{}' not supported for complex "
                                "input".format(operator))
            restype = [None, dtypes.int32]
            if coarse_types[0] < 2:
                restype[0] = dtypes.float64
                casting[0] = cast_str(restype[0])
            else:
                restype[0] = datatypes[0]
        elif _is_op_bitwise(operator) and coarse_types[0] > 1:
            raise TypeError("unsupported operand type for {}: '{}'".format(operator, datatypes[0]))
        elif _is_op_boolean(operator):
            restype = dtypes.bool_
            if operator == 'SignBit' and coarse_types[0] < 2:
                casting[0] = cast_str(dtypes.float64)
        else:
            restype = datatypes[0]

    elif len(arguments) == 2:  # Binary operators

        type1 = coarse_types[0]
        type2 = coarse_types[1]
        dtype1 = datatypes[0]
        dtype2 = datatypes[1]
        max_bytes = max(dtype1.bytes, dtype2.bytes)
        left_cast = None
        right_cast = None

        if _is_op_arithmetic(operator):

            # Float/True division between integers
            if operator == 'Div' and max(type1, type2) < 2:
                # NOTE: Leaving this here in case we implement a C/C++ flag
                # if type1 == type2 and type1 == 0:  # Unsigned integers
                #     restype = eval('dtypes.uint{}'.format(8 * max_bytes))
                # else:
                #     restype = eval('dtypes.int{}'.format(8 * max_bytes))
                restype = dtypes.float64
            # Floor division with at least one complex argument
            # NOTE: NumPy allows this operation
            # elif operator == 'FloorDiv' and max(type1, type2) == 3:
            #     raise TypeError("can't take floor of complex number")
            # Floor division with at least one float argument
            elif operator == 'FloorDiv' and max(type1, type2) == 2:
                if type1 == type2:
                    restype = eval('dtypes.float{}'.format(8 * max_bytes))
                else:
                    restype = dtypes.float64
            # Floor division between integers
            elif operator == 'FloorDiv' and max(type1, type2) < 2:
                if type1 == type2 and type1 == 0:  # Unsigned integers
                    restype = eval('dtypes.uint{}'.format(8 * max_bytes))
                else:
                    restype = eval('dtypes.int{}'.format(8 * max_bytes))
            # Multiplication between integers
            elif operator == 'Mult' and max(type1, type2) < 2:
                if type1 == 0 or type2 == 0:  # Unsigned integers
                    restype = eval('dtypes.uint{}'.format(8 * max_bytes))
                else:
                    restype = eval('dtypes.int{}'.format(8 * max_bytes))
            # Power with base integer and exponent signed integer
            elif (operator == 'Pow' and max(type1, type2) < 2 and dtype2 in signed_types):
                restype = dtypes.float64
            elif operator == 'FloatPow':
                # Float power with integers or floats
                if max(type1, type2) < 3:
                    restype = dtypes.float64
                # Float power with complex numbers
                else:
                    restype = dtypes.complex128
            elif (operator in ('Heaviside', 'Arctan2', 'Hypot') and max(type1, type2) == 3):
                raise TypeError("ufunc '{}' not supported for complex input".format(operator))
            elif (operator in ('Heaviside', 'Arctan2', 'Hypot') and max(type1, type2) < 2):
                restype = dtypes.float64
            # All other arithmetic operators and cases of the above operators
            else:
                if numpy_version >= '2.0.0':
                    restype = np_result_type(dtypes_for_result_np2)
                else:
                    restype = np_result_type(dtypes_for_result)

            if dtype1 != restype:
                left_cast = cast_str(restype)
            if dtype2 != restype:
                right_cast = cast_str(restype)

        elif _is_op_bitwise(operator):

            type1 = coarse_types[0]
            type2 = coarse_types[1]
            dtype1 = datatypes[0]
            dtype2 = datatypes[1]

            # Only integers may be arguments of bitwise and shifting operations
            if max(type1, type2) > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            restype = np_result_type(dtypes_for_result)
            if dtype1 != restype:
                left_cast = cast_str(restype)
            if dtype2 != restype:
                right_cast = cast_str(restype)

        elif _is_op_boolean(operator):
            restype = dtypes.bool_

        elif operator in ('Gcd', 'Lcm'):
            if max(type1, type2) > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            restype = np_result_type(dtypes_for_result)
            if dtype1 != restype:
                left_cast = cast_str(restype)
            if dtype2 != restype:
                right_cast = cast_str(restype)

        elif operator and operator in ('CopySign', 'NextAfter'):
            if max(type1, type2) > 2:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            if max(type1, type2) < 2:
                restype = dtypes.float64
            else:
                restype = np_result_type(dtypes_for_result)
            if dtype1 != restype:
                left_cast = cast_str(restype)
            if dtype2 != restype:
                right_cast = cast_str(restype)

        elif operator and operator in ('Ldexp'):
            if max(type1, type2) > 2 or type2 > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            if type1 < 2:
                restype = dtypes.float64
                left_cast = cast_str(restype)
            else:
                restype = dtype1
            if dtype2 != dtypes.int32:
                right_cast = cast_str(dtypes.int32)
                if not np.can_cast(dtype2.type, np.int32):
                    warnings.warn("Second input to {} is of type {}, which "
                                  "cannot be safely cast to {}".format(operator, dtype2, dtypes.int32))

        else:  # Other binary operators
            restype = np_result_type(dtypes_for_result)
            if dtype1 != restype:
                left_cast = cast_str(restype)
            if dtype2 != restype:
                right_cast = cast_str(restype)

        casting = [left_cast, right_cast]

    else:  # Operators with 3 or more arguments
        restype = np_result_type(dtypes_for_result)
        coarse_result_type = None
        if result_type in complex_types:
            coarse_result_type = 3  # complex
        elif result_type in float_types:
            coarse_result_type = 2  # float
        elif result_type in signed_types:
            coarse_result_type = 1  # signed integer, bool
        else:
            coarse_result_type = 0  # unsigned integer
        for i, t in enumerate(coarse_types):
            if t != coarse_result_type:
                casting[i] = cast_str(restype)

    return restype, casting


def _array_array_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    """
    Both operands are Arrays (or Data in general)
    """

    left_arr = sdfg.arrays[left_operand]
    right_arr = sdfg.arrays[right_operand]

    left_type = left_arr.dtype
    right_type = right_arr.dtype

    # Implicit Python coversion implemented as casting
    arguments = [left_arr, right_arr]
    tasklet_args = ['__in1', '__in2']
    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))

    left_shape = left_arr.shape
    right_shape = right_arr.shape

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])

    out_operand, out_arr = visitor.add_temp_transient(out_shape, restype, left_arr.storage)

    if list(out_shape) == [1]:
        tasklet = state.add_tasklet('_%s_' % operator, {'__in1', '__in2'}, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n1 = state.add_read(left_operand)
        n2 = state.add_read(right_operand)
        n3 = state.add_write(out_operand)
        state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_arr))
        state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_arr))
    else:
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict, {
                                     '__in1': Memlet.simple(left_operand, left_idx),
                                     '__in2': Memlet.simple(right_operand, right_idx)
                                 },
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _array_const_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    """
    Operands are an Array and a Constant
    """

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = dtypes.dtype_to_typeclass(type(right_operand))
        right_shape = [1]
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', f'({str(right_operand)})']
    else:
        left_arr = None
        left_type = dtypes.dtype_to_typeclass(type(left_operand))
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [f'({str(left_operand)})', '__in2']

    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    out_operand, out_arr = visitor.add_temp_transient(out_shape, restype, storage)

    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict,
                                 inp_memlets,
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _array_sym_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                     operator: str, opcode: str):
    """
    Operands are an Array and a Symbol
    """

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = sym_type(right_operand)
        right_shape = [1]
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', f'({astutils.unparse(right_operand)})']
    else:
        left_arr = None
        left_type = sym_type(left_operand)
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [f'({astutils.unparse(left_operand)})', '__in2']

    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    out_operand, out_arr = visitor.add_temp_transient(out_shape, restype, storage)

    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict,
                                 inp_memlets,
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _scalar_scalar_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                         operator: str, opcode: str):
    """
    Both operands are Scalars
    """

    left_scal = sdfg.arrays[left_operand]
    right_scal = sdfg.arrays[right_operand]

    left_type = left_scal.dtype
    right_type = right_scal.dtype

    # Implicit Python coversion implemented as casting
    arguments = [left_scal, right_scal]
    tasklet_args = ['__in1', '__in2']
    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))

    out_operand = visitor.get_target_name()
    out_operand, out_scal = sdfg.add_scalar(out_operand,
                                            restype,
                                            transient=True,
                                            storage=left_scal.storage,
                                            find_new_name=True)

    tasklet = state.add_tasklet('_%s_' % operator, {'__in1', '__in2'}, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n1 = state.add_read(left_operand)
    n2 = state.add_read(right_operand)
    n3 = state.add_write(out_operand)
    state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_scal))
    state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_scal))

    return out_operand


def _scalar_const_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                        operator: str, opcode: str):
    """
    Operands are a Scalar and a Constant
    """

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        storage = left_scal.storage
        right_scal = None
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', f'({str(right_operand)})']
    else:
        left_scal = None
        right_scal = sdfg.arrays[right_operand]
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [f'({str(left_operand)})', '__in2']

    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    out_operand = visitor.get_target_name()
    out_operand, out_scal = sdfg.add_scalar(out_operand, restype, transient=True, storage=storage, find_new_name=True)

    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_scal))

    return out_operand


def _scalar_sym_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                      operator: str, opcode: str):
    """
    Operands are a Scalar and a Symbol
    """

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        left_type = left_scal.dtype
        storage = left_scal.storage
        right_scal = None
        right_type = sym_type(right_operand)
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', f'({astutils.unparse(right_operand)})']
    else:
        left_scal = None
        left_type = sym_type(left_operand)
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [f'({astutils.unparse(left_operand)})', '__in2']

    restype, casting = result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    out_operand = visitor.get_target_name()
    out_operand, out_scal = sdfg.add_scalar(out_operand, restype, transient=True, storage=storage, find_new_name=True)

    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_scal))

    return out_operand


_pyop2symtype = {
    # Boolean ops
    "and": sp.And,
    "or": sp.Or,
    "not": sp.Not,
    # Comparsion ops
    "==": sp.Equality,
    "!=": sp.Unequality,
    ">=": sp.GreaterThan,
    "<=": sp.LessThan,
    ">": sp.StrictGreaterThan,
    "<": sp.StrictLessThan,
    # Binary ops
    "//": symbolic.int_floor,
}


def _const_const_binop(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    """
    Both operands are Constants or Symbols
    """

    _, casting = result_type([left_operand, right_operand], operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if isinstance(left_operand, (Number, np.bool_)) and left_cast is not None:
        left = eval(left_cast)(left_operand)
    else:
        left = left_operand
    if isinstance(right_operand, (Number, np.bool_)) and right_cast is not None:
        right = eval(right_cast)(right_operand)
    else:
        right = right_operand

    # Support for SymPy expressions
    if isinstance(left, sp.Basic) or isinstance(right, sp.Basic):
        if opcode in _pyop2symtype.keys():
            try:
                return _pyop2symtype[opcode](left, right)
            except TypeError:
                # This may happen in cases such as `False or (N + 1)`.
                # (N + 1) is a symbolic expressions, but because it is not
                # boolean, SymPy returns TypeError when trying to create
                # `sympy.Or(False, N + 1)`. In such a case, we try again with
                # the normal Python operator.
                pass

    expr = 'l {o} r'.format(o=opcode)
    vars = {'l': left, 'r': right}
    return eval(expr, vars)


def _makebinop(op, opcode):

    @oprepo.replaces_operator('Array', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_scalar_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='Array')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='View')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='Scalar')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='NumConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='BoolConstant')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='symbol')
    def _op(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)


def _makeboolop(op: str, method: str):

    @oprepo.replaces_operator('StringLiteral', op, otherclass='StringLiteral')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: StringLiteral, op2: StringLiteral):
        return getattr(op1, method)(op2)


# Define all standard Python unary operators
for op, opcode in [('UAdd', '+'), ('USub', '-'), ('Not', 'not'), ('Invert', '~')]:
    _makeunop(op, opcode)

# Define all standard Python binary operators
# NOTE: ('MatMult', '@') is defined separately
for op, opcode in [('Add', '+'), ('Sub', '-'), ('Mult', '*'), ('Div', '/'), ('FloorDiv', '//'), ('Mod', '%'),
                   ('Pow', '**'), ('LShift', '<<'), ('RShift', '>>'), ('BitOr', '|'), ('BitXor', '^'), ('BitAnd', '&'),
                   ('And', 'and'), ('Or', 'or'), ('Eq', '=='), ('NotEq', '!='), ('Lt', '<'), ('LtE', '<='), ('Gt', '>'),
                   ('GtE', '>='), ('Is', 'is'), ('IsNot', 'is not')]:
    _makebinop(op, opcode)

# Define all boolean operators
_boolop_to_method = {
    'Eq': '__eq__',
    'NotEq': '__ne__',
    'Lt': '__lt__',
    'LtE': '__le__',
    'Gt': '__gt__',
    'GtE': '__ge__'
}
for op, method in _boolop_to_method.items():
    _makeboolop(op, method)
