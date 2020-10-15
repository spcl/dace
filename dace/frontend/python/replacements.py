# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import ast
import copy
import itertools
from functools import reduce
from numbers import Number, Integral
from typing import Any, Dict, Union, Callable, Tuple, List

import dace
from dace.config import Config
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import inverse_dict_lookup
from dace.frontend.python.memlet_parser import parse_memlet_subset
from dace.frontend.python import astutils
from dace.frontend.python.nested_call import NestedCall
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic

import numpy as np
import sympy as sp

Size = Union[int, dace.symbolic.symbol]
ShapeTuple = Tuple[Size]
ShapeList = List[Size]
Shape = Union[ShapeTuple, ShapeList]

##############################################################################
# Python function replacements ###############################################
##############################################################################


@oprepo.replaces('dace.define_local')
@oprepo.replaces('dace.ndarray')
def _define_local_ex(sdfg: SDFG,
                     state: SDFGState,
                     shape: Shape,
                     dtype: dace.typeclass,
                     storage: dtypes.StorageType = dtypes.StorageType.Default):
    """ Defines a local array in a DaCe program. """
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    name, _ = sdfg.add_temp_transient(shape, dtype, storage=storage)
    return name


@oprepo.replaces('numpy.ndarray')
def _define_local(sdfg: SDFG, state: SDFGState, shape: Shape,
                  dtype: dace.typeclass):
    """ Defines a local array in a DaCe program. """
    return _define_local_ex(sdfg, state, shape, dtype)


@oprepo.replaces('dace.define_local_scalar')
def _define_local_scalar(
    sdfg: SDFG,
    state: SDFGState,
    dtype: dace.typeclass,
    storage: dtypes.StorageType = dtypes.StorageType.Default):
    """ Defines a local scalar in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_scalar(name, dtype, transient=True, storage=storage)
    return name


@oprepo.replaces('dace.define_stream')
def _define_stream(sdfg: SDFG,
                   state: SDFGState,
                   dtype: dace.typeclass,
                   buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_stream(name, dtype, buffer_size=buffer_size, transient=True)
    return name


@oprepo.replaces('dace.define_streamarray')
@oprepo.replaces('dace.stream')
def _define_streamarray(sdfg: SDFG,
                        state: SDFGState,
                        shape: Shape,
                        dtype: dace.typeclass,
                        buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_stream(name,
                    dtype,
                    shape=shape,
                    buffer_size=buffer_size,
                    transient=True)
    return name


@oprepo.replaces('dace.reduce')
def _reduce(sdfg: SDFG,
            state: SDFGState,
            redfunction: Callable[[Any, Any], Any],
            in_array: str,
            out_array=None,
            axis=None,
            identity=None):
    if out_array is None:
        inarr = in_array
        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
        input_subset = parse_memlet_subset(sdfg.arrays[inarr],
                                           ast.parse(in_array).body[0].value,
                                           {})
        input_memlet = Memlet.simple(inarr, input_subset)
        output_shape = None
        if axis is None:
            output_shape = [1]
        else:
            output_subset = copy.deepcopy(input_subset)
            output_subset.pop(axis)
            output_shape = output_subset.size()
        outarr, arr = sdfg.add_temp_transient(output_shape,
                                              sdfg.arrays[inarr].dtype,
                                              sdfg.arrays[inarr].storage)
        output_memlet = Memlet.from_array(outarr, arr)
    else:
        inarr = in_array
        outarr = out_array

        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)

        # Compute memlets
        input_subset = parse_memlet_subset(sdfg.arrays[inarr],
                                           ast.parse(in_array).body[0].value,
                                           {})
        input_memlet = Memlet.simple(inarr, input_subset)
        output_subset = parse_memlet_subset(sdfg.arrays[outarr],
                                            ast.parse(out_array).body[0].value,
                                            {})
        output_memlet = Memlet.simple(outarr, output_subset)

    # Create reduce subgraph
    inpnode = state.add_read(inarr)
    rednode = state.add_reduce(redfunction, axis, identity)
    outnode = state.add_write(outarr)
    state.add_nedge(inpnode, rednode, input_memlet)
    state.add_nedge(rednode, outnode, output_memlet)

    if out_array is None:
        return outarr
    else:
        return []


@oprepo.replaces('numpy.eye')
def eye(sdfg: SDFG, state: SDFGState, N, M=None, k=0, dtype=dace.float64):
    M = M or N
    name, _ = sdfg.add_temp_transient([N, M], dtype)

    state.add_mapped_tasklet('eye',
                             dict(i='0:%s' % N, j='0:%s' % M), {},
                             'val = 1 if i == (j - %s) else 0' % k,
                             dict(val=dace.Memlet.simple(name, 'i, j')),
                             external_edges=True)

    return name


@oprepo.replaces('elementwise')
@oprepo.replaces('dace.elementwise')
def _elementwise(sdfg: SDFG,
                 state: SDFGState,
                 func: str,
                 in_array: str,
                 out_array=None):
    """Apply a lambda function to each element in the input"""

    inparr = sdfg.arrays[in_array]
    restype = sdfg.arrays[in_array].dtype

    if out_array is None:
        out_array, outarr = sdfg.add_temp_transient(inparr.shape, restype,
                                                    inparr.storage)
    else:
        outarr = sdfg.arrays[out_array]

    func_ast = ast.parse(func)
    try:
        lambda_ast = func_ast.body[0].value
        if len(lambda_ast.args.args) != 1:
            raise SyntaxError(
                "Expected lambda with one arg, but {} has {}".format(
                    func, len(lambda_ast.args.arrgs)))
        arg = lambda_ast.args.args[0].arg
        body = astutils.unparse(lambda_ast.body)
    except AttributeError:
        raise SyntaxError("Could not parse func {}".format(func))

    code = "__out = {}".format(body)

    num_elements = reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(in_array)
        out = state.add_write(out_array)
        tasklet = state.add_tasklet("_elementwise_", {arg}, {'__out'}, code)
        state.add_edge(inp, None, tasklet, arg,
                       Memlet.from_array(in_array, inparr))
        state.add_edge(tasklet, '__out', out, None,
                       Memlet.from_array(out_array, outarr))
    else:
        state.add_mapped_tasklet(
            name="_elementwise_",
            map_ranges={
                '__i%d' % i: '0:%s' % n
                for i, n in enumerate(inparr.shape)
            },
            inputs={
                arg:
                Memlet.simple(
                    in_array,
                    ','.join(['__i%d' % i for i in range(len(inparr.shape))]))
            },
            code=code,
            outputs={
                '__out':
                Memlet.simple(
                    out_array,
                    ','.join(['__i%d' % i for i in range(len(inparr.shape))]))
            },
            external_edges=True)

    return out_array


def _simple_call(sdfg: SDFG,
                 state: SDFGState,
                 inpname: str,
                 func: str,
                 restype: dace.typeclass = None):
    """ Implements a simple call of the form `out = func(inp)`. """
    if isinstance(inpname, (list, tuple)):  # TODO investigate this
        inpname = inpname[0]
    inparr = sdfg.arrays[inpname]
    if restype is None:
        restype = sdfg.arrays[inpname].dtype
    outname, outarr = sdfg.add_temp_transient(inparr.shape, restype,
                                              inparr.storage)
    num_elements = reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(inpname)
        out = state.add_write(outname)
        tasklet = state.add_tasklet(func, {'__inp'}, {'__out'},
                                    '__out = {f}(__inp)'.format(f=func))
        state.add_edge(inp, None, tasklet, '__inp',
                       Memlet.from_array(inpname, inparr))
        state.add_edge(tasklet, '__out', out, None,
                       Memlet.from_array(outname, outarr))
    else:
        state.add_mapped_tasklet(
            name=func,
            map_ranges={
                '__i%d' % i: '0:%s' % n
                for i, n in enumerate(inparr.shape)
            },
            inputs={
                '__inp':
                Memlet.simple(
                    inpname,
                    ','.join(['__i%d' % i for i in range(len(inparr.shape))]))
            },
            code='__out = {f}(__inp)'.format(f=func),
            outputs={
                '__out':
                Memlet.simple(
                    outname,
                    ','.join(['__i%d' % i for i in range(len(inparr.shape))]))
            },
            external_edges=True)

    return outname


def _complex_to_scalar(complex_type: dace.typeclass):
    if complex_type is dace.complex64:
        return dace.float32
    elif complex_type is dace.complex128:
        return dace.float64
    else:
        return complex_type


@oprepo.replaces('exp')
@oprepo.replaces('dace.exp')
@oprepo.replaces('numpy.exp')
@oprepo.replaces('math.exp')
def _exp(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'exp')


@oprepo.replaces('sin')
@oprepo.replaces('dace.sin')
@oprepo.replaces('numpy.sin')
@oprepo.replaces('math.sin')
def _sin(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sin')


@oprepo.replaces('cos')
@oprepo.replaces('dace.cos')
@oprepo.replaces('numpy.cos')
@oprepo.replaces('math.cos')
def _cos(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'cos')


@oprepo.replaces('sqrt')
@oprepo.replaces('dace.sqrt')
@oprepo.replaces('numpy.sqrt')
@oprepo.replaces('math.sqrt')
def _sqrt(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sqrt')


@oprepo.replaces('log')
@oprepo.replaces('dace.log')
@oprepo.replaces('numpy.log')
@oprepo.replaces('math.log')
def _log(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'log')


@oprepo.replaces('conj')
@oprepo.replaces('dace.conj')
@oprepo.replaces('numpy.conj')
def _conj(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'conj')


@oprepo.replaces('real')
@oprepo.replaces('dace.real')
@oprepo.replaces('numpy.real')
def _real(sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return _simple_call(sdfg, state, input, 'real', _complex_to_scalar(inptype))


@oprepo.replaces('imag')
@oprepo.replaces('dace.imag')
@oprepo.replaces('numpy.imag')
def _imag(sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return _simple_call(sdfg, state, input, 'imag', _complex_to_scalar(inptype))


@oprepo.replaces('transpose')
@oprepo.replaces('dace.transpose')
@oprepo.replaces('numpy.transpose')
def _transpose(sdfg: SDFG, state: SDFGState, inpname: str):

    arr1 = sdfg.arrays[inpname]
    restype = arr1.dtype
    outname, arr2 = sdfg.add_temp_transient((arr1.shape[1], arr1.shape[0]),
                                            restype, arr1.storage)

    acc1 = state.add_read(inpname)
    acc2 = state.add_write(outname)
    import dace.libraries.blas  # Avoid import loop
    tasklet = dace.libraries.blas.Transpose('_Transpose_', restype)
    state.add_node(tasklet)
    state.add_edge(acc1, None, tasklet, '_inp',
                   dace.Memlet.from_array(inpname, arr1))
    state.add_edge(tasklet, '_out', acc2, None,
                   dace.Memlet.from_array(outname, arr2))

    return outname


@oprepo.replaces('numpy.sum')
def _sum(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(sdfg, state, "lambda x, y: x + y", a, axis=axis, identity=0)


@oprepo.replaces('numpy.max')
@oprepo.replaces('numpy.amax')
def _max(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(sdfg,
                   state,
                   "lambda x, y: max(x, y)",
                   a,
                   axis=axis,
                   identity=dtypes.min_value(sdfg.arrays[a].dtype))


@oprepo.replaces('numpy.min')
@oprepo.replaces('numpy.amin')
def _min(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(sdfg,
                   state,
                   "lambda x, y: min(x, y)",
                   a,
                   axis=axis,
                   identity=dtypes.max_value(sdfg.arrays[a].dtype))


@oprepo.replaces('numpy.argmax')
def _argmax(sdfg: SDFG, state: SDFGState, a: str, axis, result_type=dace.int32):
    return _argminmax(sdfg, state, a, axis, func="max", result_type=result_type)


@oprepo.replaces('numpy.argmin')
def _argmin(sdfg: SDFG, state: SDFGState, a: str, axis, result_type=dace.int32):
    return _argminmax(sdfg, state, a, axis, func="min", result_type=result_type)


def _argminmax(sdfg: SDFG,
               state: SDFGState,
               a: str,
               axis,
               func,
               result_type=dace.int32,
               return_both=False):
    nest = NestedCall(sdfg, state)

    assert func in ['min', 'max']

    if axis is None or not isinstance(axis, Integral):
        raise SyntaxError('Axis must be an int')

    a_arr = sdfg.arrays[a]

    if not 0 <= axis < len(a_arr.shape):
        raise SyntaxError("Expected 0 <= axis < len({}.shape), got {}".format(
            a, axis))

    reduced_shape = list(copy.deepcopy(a_arr.shape))
    reduced_shape.pop(axis)

    val_and_idx = dace.struct('_val_and_idx', val=a_arr.dtype, idx=result_type)

    # HACK: since identity cannot be specified for structs, we have to init the output array
    reduced_structs, reduced_struct_arr = sdfg.add_temp_transient(
        reduced_shape, val_and_idx)

    code = "__init = _val_and_idx(val={}, idx=-1)".format(
        dtypes.min_value(a_arr.dtype) if func ==
        'max' else dtypes.max_value(a_arr.dtype))

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_convert_".format(func),
        map_ranges={
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(a_arr.shape) if i != axis
        },
        inputs={},
        code=code,
        outputs={
            '__init':
            Memlet.simple(
                reduced_structs, ','.join('__i%d' % i
                                          for i in range(len(a_arr.shape))
                                          if i != axis))
        },
        external_edges=True)

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_reduce_".format(func),
        map_ranges={'__i%d' % i: '0:%s' % n
                    for i, n in enumerate(a_arr.shape)},
        inputs={
            '__in':
            Memlet.simple(
                a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))
        },
        code="__out = _val_and_idx(idx={}, val=__in)".format("__i%d" % axis),
        outputs={
            '__out':
            Memlet.simple(
                reduced_structs,
                ','.join('__i%d' % i for i in range(len(a_arr.shape))
                         if i != axis),
                wcr_str=("lambda x, y:"
                         "_val_and_idx(val={}(x.val, y.val), "
                         "idx=(y.idx if x.val {} y.val else x.idx))").format(
                             func, '<' if func == 'max' else '>'))
        },
        external_edges=True)

    if return_both:
        outidx, outidxarr = sdfg.add_temp_transient(
            sdfg.arrays[reduced_structs].shape, result_type)
        outval, outvalarr = sdfg.add_temp_transient(
            sdfg.arrays[reduced_structs].shape, a_arr.dtype)

        nest.add_state().add_mapped_tasklet(
            name="_arg{}_extract_".format(func),
            map_ranges={
                '__i%d' % i: '0:%s' % n
                for i, n in enumerate(a_arr.shape) if i != axis
            },
            inputs={
                '__in':
                Memlet.simple(
                    reduced_structs, ','.join('__i%d' % i
                                              for i in range(len(a_arr.shape))
                                              if i != axis))
            },
            code="__out_val = __in.val\n__out_idx = __in.idx",
            outputs={
                '__out_val':
                Memlet.simple(
                    outval, ','.join('__i%d' % i
                                     for i in range(len(a_arr.shape))
                                     if i != axis)),
                '__out_idx':
                Memlet.simple(
                    outidx, ','.join('__i%d' % i
                                     for i in range(len(a_arr.shape))
                                     if i != axis))
            },
            external_edges=True)

        return nest, (outval, outidx)

    else:
        # map to result_type
        out, outarr = sdfg.add_temp_transient(
            sdfg.arrays[reduced_structs].shape, result_type)
        nest(_elementwise)("lambda x: x.idx", reduced_structs, out_array=out)
        return nest, out


##############################################################################
# Python operation replacements ##############################################
##############################################################################

def _unop(sdfg: SDFG, state: SDFGState, op1: str, opcode: str, opname: str):
    """ Implements a general element-wise array unary operator. """
    arr1 = sdfg.arrays[op1]

    name, _ = sdfg.add_temp_transient(arr1.shape, arr1.dtype, arr1.storage)
    state.add_mapped_tasklet(
        "_%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr1.shape)}, {
             '__in1':
             Memlet.simple(
                 op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
         },
        '__out = %s __in1' % opcode, {
            '__out':
            Memlet.simple(
                name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
        },
        external_edges=True)
    return name


def _broadcast_together(arr1_shape, arr2_shape):

    all_idx_dict, all_idx, a1_idx, a2_idx = {}, [], [], []

    max_i = max(len(arr1_shape), len(arr2_shape))

    def get_idx(i):
        return "__i" + str(max_i - i - 1)

    for i, (dim1, dim2) in enumerate(
            itertools.zip_longest(reversed(arr1_shape), reversed(arr2_shape))):
        all_idx.append(get_idx(i))

        if dim1 == dim2:
            a1_idx.append(get_idx(i))
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim1

        elif dim1 == 1:
            a1_idx.append("0")
            # dim2 != 1 must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 == 1:
            # dim1 != 1 must hold here
            a1_idx.append(get_idx(i))
            a2_idx.append("0")

            all_idx_dict[get_idx(i)] = dim1

        elif dim1 == None:
            # dim2 != None must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 == None:
            # dim1 != None must hold here
            a1_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim1
        else:
            raise SyntaxError(
                "operands could not be broadcast together with shapes {}, {}".
                format(arr1_shape, arr2_shape))

    def to_string(idx):
        return ", ".join(reversed(idx))

    out_shape = tuple(reversed([all_idx_dict[idx] for idx in all_idx]))

    all_idx_tup = [(k, "0:" + str(all_idx_dict[k])) for k in reversed(all_idx)]

    return out_shape, all_idx_tup, to_string(all_idx), to_string(
        a1_idx), to_string(a2_idx)


def _binop(sdfg: SDFG, state: SDFGState, op1: str, op2: str, opcode: str,
           opname: str, restype: dace.typeclass):
    """ Implements a general element-wise array binary operator. """
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    out_shape, all_idx_tup, all_idx, arr1_idx, arr2_idx = _broadcast_together(
        arr1.shape, arr2.shape)

    name, _ = sdfg.add_temp_transient(out_shape, restype, arr1.storage)
    state.add_mapped_tasklet("_%s_" % opname,
                             all_idx_tup, {
                                 '__in1': Memlet.simple(op1, arr1_idx),
                                 '__in2': Memlet.simple(op2, arr2_idx)
                             },
                             '__out = __in1 %s __in2' % opcode,
                             {'__out': Memlet.simple(name, all_idx)},
                             external_edges=True)
    return name


# Defined as a function in order to include the op and the opcode in the closure
def _makeunop(op, opcode):
    @oprepo.replaces_operator('Array', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: str,
            op2=None):
        return _unop(sdfg, state, op1, opcode, op)
    
    @oprepo.replaces_operator('Scalar', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: str,
            op2=None):
        scalar1 = sdfg.arrays[op1]
        restype = scalar1.dtype
        op2 = sdfg.temp_data_name()
        _, scalar2 = sdfg.add_scalar(op2, restype, transient=True)
        tasklet = state.add_tasklet("_%s_" % op, {'__in'}, {'__out'},
                                    "__out = %s __in" % opcode)
        node1 = state.add_read(op1)
        node2 = state.add_write(op2)
        state.add_edge(node1, None, tasklet, '__in',
                       dace.Memlet.from_array(op1, scalar1))
        state.add_edge(tasklet, '__out', node2, None,
                        dace.Memlet.from_array(op2, scalar2))
        return op2
    
    @oprepo.replaces_operator('NumConstant', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: Number,
            op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)

    @oprepo.replaces_operator('BoolConstant', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: Number,
            op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars) 
    
    @oprepo.replaces_operator('symbol', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: 'symbol',
            op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)  


# @oprepo.replaces_operator('int', 'USub', None)
# @oprepo.replaces_operator('float', 'USub', None)
# def _neg(visitor: 'ProgramVisitor',
#          sdfg: SDFG,
#          state: SDFGState,
#          op1: Union[int, float],
#          op2=None):
#     return -op1


# @oprepo.replaces_operator('symbol', 'Add', 'int')
# @oprepo.replaces_operator('symbol', 'Add', 'float')
# def _addsym(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
#             op1: symbolic.symbol, op2: Union[int, float]):
#     return op1 + op2


# @oprepo.replaces_operator('symbol', 'Gt', 'symbol')
# def _gtsym(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
#            op1: symbolic.symbol, op2: Union[int, float]):
#     return op1 > op2


# def _is_scalar(sdfg: SDFG, arrname: str):
#     """ Checks whether array is pseudo-scalar (shape=(1,)). """
#     shape = sdfg.arrays[arrname].shape
#     if len(shape) == 1 and shape[0] == 1:
#         return True
#     return False


def _is_op_arithmetic(op: str):
    if op in {'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Pow', 'Mod'}:
        return True
    return False


def _is_op_bitwise(op: str):
    if op in {'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd'}:
        return True
    return False


def _is_op_boolean(op: str):
    if op in {'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE'}:
        return True
    return False


def _sym_type(expr: Union[symbolic.symbol, sp.Expr]) -> dtypes.typeclass:
    if isinstance(expr, symbolic.symbol):
        return expr.dtype
    freesym = [s for s in expr.free_symbols]
    if len(freesym) == 1:
        return freesym[0].dtype
    typeclasses = [s.dtype.type for s in freesym]
    return dtypes.DTYPE_TO_TYPECLASS[np.result_type(*typeclasses).type]


def _convert_type(dtype1, dtype2, operator) -> Tuple[dace.dtypes.typeclass]:

    complex_types = {dace.complex64, dace.complex128,
                     np.complex64, np.complex128}
    float_types = {dace.float16, dace.float32, dace.float64,
                   np.float16, np.float32, np.float64}
    signed_types = {dace.int8, dace.int16, dace.int32, dace.int64,
                    np.int8, np.int16, np.int32, np.int64}
    # unsigned_types = {np.uint8, np.uint16, np.uint32, np.uint64}

    if dtype1 in complex_types:
        type1 = 3  # complex
    elif dtype1 in float_types:
        type1 = 2  # float
    elif dtype1 in signed_types:
        type1 = 1  # signed integer, bool
    else:
        type1 = 0  # unsigned integer
    
    if dtype2 in complex_types:
        type2 = 3  # complex
    elif dtype2 in float_types:
        type2 = 2  # float
    elif dtype2 in signed_types:
        type2 = 1  # signed integer, bool
    else:
        type2 = 0  # unsigned integer

    left_cast = None
    right_cast = None

    if _is_op_arithmetic(operator):

        # Float division between integers
        if operator == 'Div' and max(type1, type2) < 2:
            max_bytes = max(dtype1.bytes, dtype2.bytes)
            if type1 == type2 and type1 == 0:  # Unsigned integers
                result_type = eval('dace.uint{}'.format(4 * max_bytes))
            else:
                result_type = eval('dace.int{}'.format(4 * max_bytes))
        # Floor division with at least one complex argument
        elif operator == 'FloorDiv' and max(type1, type2) == 3:
            raise TypeError("can't take floor of complex number")
        # Floor division with at least one float argument
        elif operator == 'FloorDiv' and max(type1, type2) == 2:
            result_type = dace.int64
        # Floor division between integers
        elif operator == 'FloorDiv' and max(type1, type2) < 2:
            max_bytes = max(dtype1.bytes, dtype2.bytes)
            if type1 == type2 and type1 == 0:  # Unsigned integers
                result_type = eval('dace.uint{}'.format(4 * max_bytes))
            else:
                result_type = eval('dace.int{}'.format(4 * max_bytes))
            # TODO: Improve this performance-wise?
            right_cast = dace.float64
        # Power with base integer and exponent signed integer
        elif (operator == 'Pow' and max(type1, type2) < 2 and
                dtype2 in signed_types):
            result_type = dace.float64
            left_cast = dace.float64
            right_cast = dace.float64
        # All other arithmetic operators and cases of the above operators
        else:
            # TODO: Does this always make sense?
            result_type = dace.DTYPE_TO_TYPECLASS[
                np.result_type(dtype1.type, dtype2.type).type]
            if max(type1, type2) == 3:
                if type1 < 3:
                    left_cast = dtype2
                elif type2 < 3:
                    right_cast = dtype1
    
    elif _is_op_bitwise(operator):
        # Only integers may be arguments of bitwise and shifting operations
        if max(type1, type2) > 1:
            raise TypeError("unsupported operand type(s) for {}: "
                            "'{}' and '{}'".format(operator, dtype1, dtype2))
        max_bytes = max(dtype1.bytes, dtype2.bytes)
        result_type = eval('dace.int{}'.format(4 * max_bytes))
    
    elif _is_op_boolean(operator):
        result_type = dace.int8
                
    return result_type, left_cast, right_cast


def _array_array_binop(visitor: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       left_operand: str,
                       right_operand: str,
                       operator: str,
                       opcode: str):
    '''Both operands are Arrays (or Data in general)'''

    left_arr = sdfg.arrays[left_operand]
    right_arr = sdfg.arrays[right_operand]

    left_type = left_arr.dtype
    right_type = right_arr.dtype

    # Implicit Python coversion implemented as casting
    tasklet_args = ['__in1', '__in2']
    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))
    
    left_shape = left_arr.shape
    right_shape = right_arr.shape

    (out_shape, all_idx_dict, out_idx,
     left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type,
                                                   left_arr.storage)
    
    if list(out_shape) == [1]:
        tasklet = state.add_tasklet(
            '_%s_' % operator,
            {'__in1', '__in2'},
            {'__out'},
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1])
        )
        n1 = state.add_read(left_operand)
        n2 = state.add_read(right_operand)
        n3 = state.add_write(out_operand)
        state.add_edge(n1, None, tasklet, '__in1',
                       dace.Memlet.from_array(left_operand, left_arr))
        state.add_edge(n2, None, tasklet, '__in2',
                       dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None,
                       dace.Memlet.from_array(out_operand, out_arr))
    else:
        state.add_mapped_tasklet(
            "_%s_" % operator,
            all_idx_dict,
            {
                '__in1': Memlet.simple(left_operand, left_idx),
                '__in2': Memlet.simple(right_operand, right_idx)
            },
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
            {'__out': Memlet.simple(out_operand, out_idx)},
            external_edges=True
        )
    
    return out_operand


def _array_const_binop(visitor: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       left_operand: str,
                       right_operand: str,
                       operator: str,
                       opcode: str):
    '''Operands are an Array and a Constant'''

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = dtypes.DTYPE_TO_TYPECLASS[type(right_operand)]
        right_shape = [1]
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_arr = None
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        tasklet_args = [str(left_operand), '__in2']

    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'),
                                            o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'),
                                            o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx,
     left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type,
                                                   storage)
    
    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet(
            '_%s_' % operator,
            inp_conn,
            {'__out'},
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1])
        )
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1',
                           dace.Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2',
                           dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None,
                       dace.Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet(
            "_%s_" % operator,
            all_idx_dict,
            inp_memlets,
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
            {'__out': Memlet.simple(out_operand, out_idx)},
            external_edges=True
        )
    
    return out_operand


def _array_sym_binop(visitor: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     left_operand: str,
                     right_operand: str,
                     operator: str,
                     opcode: str):
    '''Operands are an Array and a Symbol'''

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = _sym_type(right_operand)
        right_shape = [1]
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_arr = None
        left_type = _sym_type(left_operand)
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        tasklet_args = [str(left_operand), '__in2']

    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'),
                                            o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'),
                                            o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx,
     left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type,
                                                   storage)
    
    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet(
            '_%s_' % operator,
            inp_conn,
            {'__out'},
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1])
        )
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1',
                           dace.Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2',
                           dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None,
                       dace.Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet(
            "_%s_" % operator,
            all_idx_dict,
            inp_memlets,
            '__out = {i1} {op} {i2}'.format(
                    i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
            {'__out': Memlet.simple(out_operand, out_idx)},
            external_edges=True
        )
    
    return out_operand


def _scalar_scalar_binop(visitor: 'ProgramVisitor',
                         sdfg: SDFG,
                         state: SDFGState,
                         left_operand: str,
                         right_operand: str,
                         operator: str,
                         opcode: str):
    '''Both operands are Scalars'''

    left_scal = sdfg.arrays[left_operand]
    right_scal = sdfg.arrays[right_operand]

    left_type = left_scal.dtype
    right_type = right_scal.dtype

    # Implicit Python coversion implemented as casting
    tasklet_args = ['__in1', '__in2']
    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))

    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True,
                                  storage=left_scal.storage)
    
    tasklet = state.add_tasklet(
        '_%s_' % operator,
        {'__in1', '__in2'},
        {'__out'},
        '__out = {i1} {op} {i2}'.format(
                i1=tasklet_args[0], op=opcode, i2=tasklet_args[1])
    )
    n1 = state.add_read(left_operand)
    n2 = state.add_read(right_operand)
    n3 = state.add_write(out_operand)
    state.add_edge(n1, None, tasklet, '__in1',
                   dace.Memlet.from_array(left_operand, left_scal))
    state.add_edge(n2, None, tasklet, '__in2',
                   dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None,
                   dace.Memlet.from_array(out_operand, out_scal))
    
    return out_operand


def _scalar_const_binop(visitor: 'ProgramVisitor',
                        sdfg: SDFG,
                        state: SDFGState,
                        left_operand: str,
                        right_operand: str,
                        operator: str,
                        opcode: str):
    '''Operands are a Scalar and a Constant'''

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        left_type = left_scal.dtype
        storage = left_scal.storage
        right_scal = None
        right_type = dtypes.DTYPE_TO_TYPECLASS[type(right_operand)]
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_scal = None
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        tasklet_args = [str(left_operand), '__in2']

    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'),
                                            o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'),
                                            o=tasklet_args[1])
    
    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True,
                                  storage=storage)
    
    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet(
        '_%s_' % operator,
        inp_conn,
        {'__out'},
        '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode,
                                        i2=tasklet_args[1])
    )
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1',
                       dace.Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2',
                       dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None,
                   dace.Memlet.from_array(out_operand, out_scal))
    
    return out_operand


def _scalar_sym_binop(visitor: 'ProgramVisitor',
                      sdfg: SDFG,
                      state: SDFGState,
                      left_operand: str,
                      right_operand: str,
                      operator: str,
                      opcode: str):
    '''Operands are a Scalar and a Symbol'''

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        left_type = left_scal.dtype
        storage = left_scal.storage
        right_scal = None
        right_type = _sym_type(right_operand)
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_scal = None
        left_type = _sym_type(left_operand)
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        tasklet_args = [str(left_operand), '__in2']

    result_type, left_cast, right_cast = _convert_type(left_type, right_type,
                                                       operator)
    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'),
                                            o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'),
                                            o=tasklet_args[1])
    
    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True,
                                  storage=storage)
    
    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet(
        '_%s_' % operator,
        inp_conn,
        {'__out'},
        '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode,
                                        i2=tasklet_args[1])
    )
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1',
                       dace.Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2',
                       dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None,
                   dace.Memlet.from_array(out_operand, out_scal))
    
    return out_operand


def _const_const_binop(visitor: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       left_operand: str,
                       right_operand: str,
                       operator: str,
                       opcode: str):
    '''Both operands are Constants or Symbols'''

    if isinstance(left_operand, Number):
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
    else:
        left_type = None
    if isinstance(right_operand, Number):
        right_type = dtypes.DTYPE_TO_TYPECLASS[type(right_operand)]
    else:
        right_type = None

    if left_type and right_type:
        _, left_cast, right_cast = _convert_type(left_type, right_type,
                                                 operator)
    else:
        left_cast = None
        right_cast = None

    if isinstance(left_operand, Number) and left_cast is not None:
        left = left_cast(left_operand)
    else:
        left = left_operand
    if isinstance(right_operand, Number) and right_cast is not None:
        right = right_cast(right_operand)
    else:
        right = right_operand

    expr = 'l {o} r'.format(o=opcode)
    vars = {'l': left, 'r': right}
    return eval(expr, vars)


def _makebinop(op, opcode):
    @oprepo.replaces_operator('Array', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Array', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Array', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Array', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_scalar_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Scalar', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Scalar', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('Scalar', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('NumConstant', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('NumConstant', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('NumConstant', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('NumConstant', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('BoolConstant', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('BoolConstant', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('BoolConstant', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('BoolConstant', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('BoolConstant', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('symbol', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('symbol', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('symbol', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('symbol', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)
    
    @oprepo.replaces_operator('symbol', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)


# Define all standard Python unary operators
for op, opcode in [('UAdd', '+'), ('USub', '-'), ('Not', 'not'),
                   ('Invert', '~')]:
    _makeunop(op, opcode)

# Define all standard Python binary operators
# NOTE: ('MatMult', '@') is defined separately
for op, opcode in [('Add', '+'), ('Sub', '-'), ('Mult', '*'), ('Div', '/'),
                   ('FloorDiv', '//'), ('Mod', '%'), ('Pow', '**'),
                   ('LShift', '<<'), ('RShift', '>>'), ('BitOr', '|'),
                   ('BitXor', '^'), ('BitAnd', '&'), ('And', 'and'),
                   ('Or', 'or'), ('Eq', '=='), ('NotEq', '!='), ('Lt', '<'),
                   ('LtE', '<='), ('Gt', '>'), ('GtE', '>=')]:
    _makebinop(op, opcode)


@oprepo.replaces_operator('Array', 'MatMult')
def _matmult(visitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):

    from dace.libraries.blas.nodes.matmul import MatMul  # Avoid import loop

    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    if len(arr1.shape) > 1 and len(arr2.shape) > 1:  # matrix * matrix

        if len(arr1.shape) > 3 or len(arr2.shape) > 3:
            raise SyntaxError(
                'Matrix multiplication of tensors of dimensions > 3 '
                'not supported')

        if arr1.shape[-1] != arr2.shape[-2]:
            raise SyntaxError('Matrix dimension mismatch %s != %s' %
                              (arr1.shape[-1], arr2.shape[-2]))

        from dace.libraries.blas.nodes.matmul import _get_batchmm_opts

        # Determine batched multiplication
        bopt = _get_batchmm_opts(arr1.shape, arr1.strides, arr2.shape,
                                 arr2.strides, None, None)
        if bopt:
            output_shape = (bopt['b'], arr1.shape[-2], arr2.shape[-1])
        else:
            output_shape = (arr1.shape[-2], arr2.shape[-1])

    elif len(arr1.shape) == 2 and len(arr2.shape) == 1:  # matrix * vector

        if arr1.shape[1] != arr2.shape[0]:
            raise SyntaxError("Number of matrix columns {} must match"
                              "size of vector {}.".format(
                                  arr1.shape[1], arr2.shape[0]))

        output_shape = (arr1.shape[0], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 1:  # vector * vector

        if arr1.shape[0] != arr2.shape[0]:
            raise SyntaxError("Vectors in vector product must have same size: "
                              "{} vs. {}".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (1, )

    else:  # Dunno what this is, bail

        raise SyntaxError(
            "Cannot multiply arrays with shapes: {} and {}".format(
                arr1.shape, arr2.shape))

    type1 = arr1.dtype.type
    type2 = arr2.dtype.type
    restype = dace.DTYPE_TO_TYPECLASS[np.result_type(type1, type2).type]

    op3, arr3 = sdfg.add_temp_transient(output_shape, restype, arr1.storage)

    acc1 = state.add_read(op1)
    acc2 = state.add_read(op2)
    acc3 = state.add_write(op3)

    tasklet = MatMul('_MatMult_', restype)
    state.add_node(tasklet)
    state.add_edge(acc1, None, tasklet, '_a', dace.Memlet.from_array(op1, arr1))
    state.add_edge(acc2, None, tasklet, '_b', dace.Memlet.from_array(op2, arr2))
    state.add_edge(tasklet, '_c', acc3, None, dace.Memlet.from_array(op3, arr3))

    return op3
