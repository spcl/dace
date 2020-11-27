# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import ast
import copy
import itertools
import warnings
from functools import reduce
from numbers import Number, Integral
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Union

import dace
from dace.codegen.tools import type_inference
from dace.config import Config
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.frontend.common import op_repository as oprepo
import dace.frontend.python.memlet_parser as mem_parser
from dace.frontend.python.memlet_parser import parse_memlet_subset
from dace.frontend.python import astutils
from dace.frontend.python.nested_call import NestedCall
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic

import numpy as np
import sympy as sp

Size = Union[int, dace.symbolic.symbol]
Shape = Sequence[Size]

def normalize_axes(axes: Tuple[int], max_dim: int) -> List[int]:
    """ Normalize a list of axes by converting negative dimensions to positive.

        :param dims: the list of dimensions, possibly containing negative ints.
        :param max_dim: the total amount of dimensions.
        :return: a list of dimensions containing only positive ints.
    """

    return [ax if ax >= 0 else max_dim + ax for ax in axes]


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
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        input_subset = parse_memlet_subset(sdfg.arrays[inarr],
                                           ast.parse(in_array).body[0].value,
                                           {})
        input_memlet = Memlet.simple(inarr, input_subset)
        output_shape = None

        # check if we are reducing along all axes
        if axis is not None and len(axis) == len(input_subset.size()):
            reduce_all = all(
                x == y for x, y in zip(axis, range(len(input_subset.size()))))
        else:
            reduce_all = False

        if axis is None or reduce_all:
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
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

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
def _transpose(sdfg: SDFG, state: SDFGState, inpname: str, axes=None):

    if axes is None:
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
    else:
        arr1 = sdfg.arrays[inpname]
        if len(axes) != len(arr1.shape) or sorted(axes) != list(
                range(len(arr1.shape))):
            raise ValueError("axes don't match array")

        new_shape = [arr1.shape[i] for i in axes]
        outname, arr2 = sdfg.add_temp_transient(new_shape, arr1.dtype,
                                                arr1.storage)

        state.add_mapped_tasklet(
            "_transpose_", {
                "_i{}".format(i): "0:{}".format(s)
                for i, s in enumerate(arr1.shape)
            },
            dict(_in=Memlet.simple(
                inpname, ", ".join("_i{}".format(i)
                                   for i, _ in enumerate(arr1.shape)))),
            "_out = _in",
            dict(_out=Memlet.simple(
                outname, ", ".join("_i{}".format(axes[i])
                                   for i, _ in enumerate(arr1.shape)))),
            external_edges=True)

    return outname


@oprepo.replaces('numpy.sum')
def _sum(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(sdfg, state, "lambda x, y: x + y", a, axis=axis, identity=0)


@oprepo.replaces('numpy.mean')
def _mean(sdfg: SDFG, state: SDFGState, a: str, axis=None):

    nest = NestedCall(sdfg, state)

    sum = nest(_sum)(a, axis=axis)

    if axis is None:
        div_amount = reduce(lambda x, y: x * y,
                            (d for d in sdfg.arrays[a].shape))
    elif isinstance(axis, (tuple, list)):
        axis = normalize_axes(axis, len(sdfg.arrays[a].shape))
        # each entry needs to be divided by the size of the reduction
        div_amount = reduce(
            lambda x, y: x * y,
            (d for i, d in enumerate(sdfg.arrays[a].shape) if i in axis))
    else:
        div_amount = sdfg.arrays[a].shape[axis]

    return nest, nest(_elementwise)("lambda x: x / ({})".format(div_amount),
                                    sum)


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

    restype, _ = _result_type([arr1], opname)

    name, _ = sdfg.add_temp_transient(arr1.shape, restype, arr1.storage)
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

        elif dim1 == 1 and dim2 is not None:
            a1_idx.append("0")
            # dim2 != 1 must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 == 1 and dim1 is not None:
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
        restype, _ = _result_type([scalar1], op)
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
        if opcode in _pyop2symtype.keys():
            try:
                return _pyop2symtype[opcode](op1)
            except TypeError:
                pass
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)


def _is_op_arithmetic(op: str):
    if op in {'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Pow', 'Mod'}:
        return True
    return False


def _is_op_bitwise(op: str):
    if op in {'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd'}:
        return True
    return False


def _is_op_boolean(op: str):
    if op in {'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE',
              'Is', 'NotIs'}:
        return True
    return False


def _representative_num(dtype: Union[dtypes.typeclass, Number]) -> Number:
    if isinstance(dtype, dtypes.typeclass):
        nptype = dtype.type
    else:
        nptype = dtype
    if issubclass(nptype, bool):
        return True
    elif issubclass(nptype, Integral):
        return nptype(np.iinfo(nptype).max)
    else:
        return nptype(np.finfo(nptype).resolution)


def _np_result_type(nptypes):
    # Fix for np.result_type returning platform-dependent types,
    # e.g. np.longlong
    restype = np.result_type(*nptypes)
    if restype.type not in dtypes.DTYPE_TO_TYPECLASS.keys():
        for k in dtypes.DTYPE_TO_TYPECLASS.keys():
            if k == restype.type:
                return dtypes.DTYPE_TO_TYPECLASS[k]
    return dtypes.DTYPE_TO_TYPECLASS[restype.type]


def _sym_type(expr: Union[symbolic.symbol, sp.Basic]) -> dtypes.typeclass:
    if isinstance(expr, symbolic.symbol):
        return expr.dtype
    representative_value = expr.subs([(s, _representative_num(s.dtype))
                                      for s in expr.free_symbols])
    pyval = eval(astutils.unparse(representative_value))
    # Overflow check
    if isinstance(pyval, int) and (pyval > np.iinfo(np.int64).max or
                                  pyval < np.iinfo(np.int64).min):
        nptype = np.int64
    else:
        nptype = np.result_type(pyval)
    return _np_result_type([nptype])


def _result_type(
    arguments: Sequence[Union[str, Number, symbolic.symbol, sp.Basic]],
    operator: str = None
) -> Tuple[dtypes.typeclass, ...]:

    datatypes = []
    dtypes_for_result = []
    for arg in arguments:
        if isinstance(arg, (data.Array, data.Stream)):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(arg.dtype.type)
        elif isinstance(arg, data.Scalar):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(_representative_num(arg.dtype))
        elif isinstance(arg, Number):
            datatypes.append(dtypes.DTYPE_TO_TYPECLASS[type(arg)])
            dtypes_for_result.append(arg)
        elif symbolic.issymbolic(arg):
            datatypes.append(_sym_type(arg))
            dtypes_for_result.append(_representative_num(_sym_type(arg)))
        else:
            raise TypeError("Type {t} of argument {a} is not supported".format(
                t=type(arg), a=arg))
    
    complex_types = {dace.complex64, dace.complex128,
                     np.complex64, np.complex128}
    float_types = {dace.float16, dace.float32, dace.float64,
                   np.float16, np.float32, np.float64}
    signed_types = {dace.int8, dace.int16, dace.int32, dace.int64,
                    np.int8, np.int16, np.int32, np.int64}
    # unsigned_types = {np.uint8, np.uint16, np.uint32, np.uint64}

    coarse_types = []
    for dtype in datatypes:
        if dtype in complex_types:
            coarse_types.append(3) # complex
        elif dtype in float_types:
            coarse_types.append(2)  # float
        elif dtype in signed_types:
            coarse_types.append(1) # signed integer, bool
        else:
            coarse_types.append(0)  # unsigned integer
    
    casting = [None] * len(arguments)

    if len(arguments) == 1:  # Unary operators

        if operator == 'USub' and coarse_types[0] == 0:
            result_type = eval('dace.int{}'.format(8 * datatypes[0].bytes))
        elif _is_op_boolean(operator):
            result_type = dace.bool_
        else:
            result_type = datatypes[0]

    elif len(arguments) == 2:  # Binary operators

        type1 = coarse_types[0]
        type2 = coarse_types[1]
        dtype1 = datatypes[0]
        dtype2 = datatypes[1]
        left_cast = None
        right_cast = None

        if _is_op_arithmetic(operator):

            # Float/True division between integers
            if operator == 'Div' and max(type1, type2) < 2:
                # TODO: Leaving this here in case we implement a C/C++ flag
                # max_bytes = max(dtype1.bytes, dtype2.bytes)
                # if type1 == type2 and type1 == 0:  # Unsigned integers
                #     result_type = eval('dace.uint{}'.format(8 * max_bytes))
                # else:
                #     result_type = eval('dace.int{}'.format(8 * max_bytes))
                result_type = dace.float64
                left_cast = dace.float64
                right_cast = dace.float64
            # Floor division with at least one complex argument
            elif operator == 'FloorDiv' and max(type1, type2) == 3:
                raise TypeError("can't take floor of complex number")
            # Floor division with at least one float argument
            elif operator == 'FloorDiv' and max(type1, type2) == 2:
                result_type = dace.float64
            # Floor division between integers
            elif operator == 'FloorDiv' and max(type1, type2) < 2:
                max_bytes = max(dtype1.bytes, dtype2.bytes)
                if type1 == type2 and type1 == 0:  # Unsigned integers
                    result_type = eval('dace.uint{}'.format(8 * max_bytes))
                else:
                    result_type = eval('dace.int{}'.format(8 * max_bytes))
                right_cast = dace.float64
            # Power with base integer and exponent signed integer
            elif (operator == 'Pow' and max(type1, type2) < 2 and
                    dtype2 in signed_types):
                result_type = dace.float64
                left_cast = dace.float64
                right_cast = dace.float64
            # All other arithmetic operators and cases of the above operators
            else:
                result_type = _np_result_type(dtypes_for_result)
                if max(type1, type2) == 3:
                    if type1 < 3:
                        left_cast = dtype2
                    elif type2 < 3:
                        right_cast = dtype1
                    else:  # type1 == type2
                        max_bytes = max(dtype1.bytes, dtype2.bytes)
                        cast = eval('dace.complex{}'.format(8 * max_bytes))
                        if dtype1 != cast:
                            left_cast = cast
                        if dtype2 != cast:
                            right_cast = cast
            
            casting = [left_cast, right_cast]

        elif _is_op_bitwise(operator):

            type1 = coarse_types[0]
            type2 = coarse_types[1]
            dtype1 = datatypes[0]
            dtype2 = datatypes[1]

            # Only integers may be arguments of bitwise and shifting operations
            if max(type1, type2) > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            max_bytes = max(dtype1.bytes, dtype2.bytes)
            result_type = eval('dace.int{}'.format(8 * max_bytes))

        elif _is_op_boolean(operator):
            result_type = dace.bool_
        
        else:  # Other binary operators
            result_type = _np_result_type(dtypes_for_result)
            if max(type1, type2) == 3:
                if type1 < 3:
                    left_cast = dtype2
                elif type2 < 3:
                    right_cast = dtype1
            
        casting = [left_cast, right_cast]

    else:  # Operators with 3 or more arguments
        result_type = _np_result_type(dtypes_for_result)
        for i, t in enumerate(coarse_types):
            if t != result_type:
                casting[i] = t

    return result_type, casting


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
    arguments = [left_arr, right_arr]
    tasklet_args = ['__in1', '__in2']
    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_arr = None
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [str(left_operand), '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', astutils.unparse(right_operand)]
    else:
        left_arr = None
        left_type = _sym_type(left_operand)
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [astutils.unparse(left_operand), '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
    arguments = [left_scal, right_scal]
    tasklet_args = ['__in1', '__in2']
    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', str(right_operand)]
    else:
        left_scal = None
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [str(left_operand), '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', astutils.unparse(right_operand)]
    else:
        left_scal = None
        left_type = _sym_type(left_operand)
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [astutils.unparse(left_operand), '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

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
    "<": sp.StrictLessThan
}


def _const_const_binop(visitor: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       left_operand: str,
                       right_operand: str,
                       operator: str,
                       opcode: str):
    '''Both operands are Constants or Symbols'''

    _, casting = _result_type([left_operand, right_operand], operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if isinstance(left_operand, Number) and left_cast is not None:
        left = left_cast(left_operand)
    else:
        left = left_operand
    if isinstance(right_operand, Number) and right_cast is not None:
        right = right_cast(right_operand)
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
                   ('LtE', '<='), ('Gt', '>'), ('GtE', '>='),
                   ('Is', 'is'), ('IsNot', 'is not')]:
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


# NumPy ufunc support #########################################################

UfuncInput = Union[str, Number, sp.Basic]
UfuncOutput = Union[str, None]

# TODO: Add all ufuncs in subsequent PR's.
ufuncs = dict(
    add = dict(
        name="_numpy_add_",
        operator="Add",
        inputs=["__in1", "__in2"],
        outputs=["__out"], code="__out = __in1 + __in2",
        reduce="lambda a, b: a + b", initial=np.add.identity),
    subtract = dict(
        name="_numpy_subtract_",
        operator="Sub",
        inputs=["__in1", "__in2"],
        outputs=["__out"], code="__out = __in1 - __in2",
        reduce="lambda a, b: a - b", initial=np.subtract.identity),
    multiply = dict(
        name="_numpy_multipy_",
        operator="Mul",
        inputs=["__in1", "__in2"],
        outputs=["__out"], code="__out = __in1 * __in2",
        reduce="lambda a, b: a * b", initial=np.multiply.identity),
    # TODO: Will be enabled when proper casting is implemented.
    # divide = dict(
    #     name="_numpy_divide_",
    #     inputs=["__in1", "__in2"],
    #     outputs=["__out"], code="__out = __in1 / __in2",
    #     reduce="lambda a, b: a / b", initial=np.divide.identity),
    minimum = dict(
        name="_numpy_min_",
        operator=None,
        inputs=["__in1", "__in2"],
        outputs=["__out"],
        code="__out = min(__in1, __in2)",
        reduce="lambda a, b: min(a, b)", initial=np.minimum.identity)
)


def _get_ufunc_impl(visitor: 'ProgramVisitor',
                    ast_node: ast.Call,
                    ufunc_name: str) -> Dict[str, Any]:
    """ Retrieves the implementation details for a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc

        :raises DaCeSyntaxError: When the ufunc implementation is missing
    """

    try:
        return ufuncs[ufunc_name]
    except KeyError:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node,
            "Missing implementation for NumPy ufunc {f}.".format(f=ufunc_name))


def _validate_ufunc_num_arguments(visitor: 'ProgramVisitor',
                                  ast_node: ast.Call,
                                  ufunc_name: str,
                                  num_inputs: int,
                                  num_outputs: int,
                                  num_args: int):
    """ Validates the number of positional arguments in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param num_outputs: Number of ufunc outputs
        :param num_args: Number of positional argumnents in the ufunc call

        :raises DaCeSyntaxError: When validation fails
    """

    if num_args > num_inputs + num_outputs:
        raise mem_parser.DaceSyntaxError(
            visitor,
            ast_node,
            "Invalid number of arguments in call to numpy.{f} "
            "(expected a maximum of {i} input(s) and {o} output(s), "
            "but a total of {a} arguments were given).".format(
                f=ufunc_name, i=num_inputs, o=num_outputs, a=num_args
            )
        )


def _validate_ufunc_inputs(visitor: 'ProgramVisitor',
                           ast_node: ast.Call,
                           sdfg: SDFG,
                           ufunc_name: str,
                           num_inputs: int,
                           num_args: int,
                           args: Sequence[UfuncInput]) -> List[UfuncInput]:
    """ Validates the number of type of inputs in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param args: Positional arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of input datanames and constants
    """

    # Validate number of inputs
    if num_args > num_inputs:
        # Assume that the first "num_inputs" arguments are inputs
        inputs = args[:num_inputs]
    elif num_args < num_inputs:
        raise mem_parser.DaceSyntaxError(
            visitor,
            ast_node,
            "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} inputs, but {a} were given).".format(
                f=ufunc_name, e=num_inputs, a=num_args
            )
        )
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
            visitor,
            ast_node,
            "Input arguments in call to numpy.{f} must be of dace.data.Data "
            "type or numerical/boolean constants (invalid argument {a})".format(
                f=ufunc_name, a=arg
            )
        )
    
    return inputs


def _validate_ufunc_outputs(visitor: 'ProgramVisitor',
                            ast_node: ast.Call,
                            sdfg: SDFG,
                            ufunc_name: str,
                            num_inputs: int,
                            num_outputs: int,
                            num_args: int,
                            args: Sequence[UfuncInput],
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

        :returns: List of output datanames and None
    """

     # Validate number of outputs
    num_pos_outputs = num_args - num_inputs
    if num_pos_outputs == 0 and "out" not in kwargs.keys():
        outputs = [None] * num_outputs
    elif num_pos_outputs > 0 and "out" in kwargs.keys():
        raise mem_parser.DaceSyntaxError(
            visitor,
            ast_node,
            "You cannot specify 'out' in call to numpy.{f} as both a positional"
            " and keyword argument (positional {p}, keyword {w}).".format(
                f=ufunc_name, p=args[num_outputs, :], k=kwargs['out']
            )
        )
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
            visitor,
            ast_node,
           "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} outputs, but {a} were given).".format(
                f=ufunc_name, e=num_outputs, a=len(outputs)
            )
        )
    
    # Validate outputs
    for arg in outputs:
        if arg is None:
            pass
        elif isinstance(arg, str) and arg in sdfg.arrays.keys():
            pass
        else:
            raise mem_parser.DaceSyntaxError(
                visitor,
                ast_node,
                "Return arguments in call to numpy.{f} must be of "
                "dace.data.Data type.".format(f=ufunc_name)
            )
    
    return outputs


def _validate_where_kword(
    visitor: 'ProgramVisitor',
    ast_node: ast.Call,
    sdfg: SDFG,
    ufunc_name: str,
    kwargs: Dict[str, Any]
) -> Tuple[bool, Union[str, bool]]:
    """ Validates the 'where' keyword argument passed to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call

        :raises DaceSyntaxError: When validation fails

        :returns: Tuple of a boolean value indicating whether the 'where'
        keyword is defined, and the validated 'where' value
    """

    has_where = False
    where = None
    if 'where' in kwargs.keys():
        where = kwargs['where']
        if isinstance(where, str) and where in sdfg.arrays.keys():
            has_where = True
        elif isinstance(where, bool):
            has_where = True
        elif isinstance(where, (list, tuple)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Values for the 'where' keyword that are a sequence of boolean "
                " constants are unsupported. Please, pass these values to the "
                " {n} call through a DaCe boolean array.".format(n=ufunc_name)
            )
        else:
            # NumPy defaults to "where=True" for invalid values for the keyword
            pass
    
    return has_where, where


def _validate_shapes(
    visitor: 'ProgramVisitor',
    ast_node: ast.Call,
    sdfg: SDFG,
    ufunc_name: str,
    inputs: List[UfuncInput],
    outputs: List[UfuncOutput]
) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Validates the data shapes of inputs and outputs to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: Tuple with the output shape, the map, output and input indices
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
            visitor,
            ast_node,
            "Shape validation in numpy.{f} call failed. The following error "
            "occured : {m}".format(f=ufunc_name, m=str(e))
        )
    return result


def _broadcast(
    shapes: Sequence[Shape]
) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Applies the NumPy ufunc brodacsting rules in a sequence of data shapes
        (see https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting).

        :param shapes: Sequence (list, tuple) of data shapes

        :raises SyntaxError: When broadcasting fails

        :returns: Tuple with the output shape, the map, output and input indices
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
        max_dim = max(not_none_dims)

        map_lengths[get_idx(i)] = max_dim
        for j, d in enumerate(dims):
            if d is None:
                pass
            elif d == 1:
                input_indices[j].append('0')
            elif d == max_dim:
                input_indices[j].append(get_idx(i))
            else:
                raise SyntaxError(
                    "Operands could not be broadcast together with shapes {}.".
                    format(','.join(str(shapes)))
                )

    out_shape = tuple(reversed([map_lengths[idx] for idx in output_indices]))
    map_indices = [(k, "0:" + str(map_lengths[k]))
                    for k in reversed(output_indices)]
    output_indices = to_string(output_indices)
    input_indices = [to_string(idx) for idx in input_indices]

    if not out_shape:
        out_shape = (1,)
        output_indices = "0"

    return out_shape, map_indices, output_indices, input_indices


def _create_output(sdfg: SDFG,
                   inputs: List[UfuncInput],
                   outputs: List[UfuncOutput],
                   output_shape: Shape,
                   output_dtype: dtypes.typeclass,
                   storage: dtypes.StorageType = None,
                   force_scalar: bool = False,) -> List[UfuncOutput]:
    """ Creates output data for storing the result of a NumPy ufunc call.

        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call
        :param output_shape: Shape of the output data
        :param output_dtype: Datatype of the output data
        :param storage: Storage type of the output data
        :param force_scalar: If True and output shape is (1,) then output
        becomes a dace.data.Scalar, regardless of the data-type of the inputs

        :returns: New outputs of the ufunc call
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

    # Create output data (if needed)
    for i, arg in enumerate(outputs):
        if arg is None:
            if (len(output_shape) == 1 and output_shape[0] == 1
                    and (is_output_scalar or force_scalar)):
                output_name = sdfg.temp_data_name()
                sdfg.add_scalar(output_name, output_dtype,
                                transient=True, storage=storage)
                outputs[i] = output_name
            else:
                outputs[i], _ = sdfg.add_temp_transient(output_shape,
                                                        output_dtype)

    return outputs


def _set_tasklet_params(
    ufunc_impl: Dict[str, Any],
    inputs: List[UfuncInput],
    casting: List[dtypes.typeclass] = None
) -> Dict[str, Any]:
    """ Sets the tasklet parameters for a NumPy ufunc call.

        :param ufunc_impl: Information on how the ufunc must be implemented
        :param inputs: Inputs of the ufunc call

        :returns: Dictionary with the (1) tasklet name, (2) input connectors,
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
            repl = "{c}({o})".format(c=str(casting[i]).replace('::', '.'),
                                     o=inp_conn)
            code = code.replace(inp_conn, repl)
        if isinstance(arg, (Number, sp.Basic)):
            inp_conn = inp_connectors[i]
            code = code.replace(inp_conn, astutils.unparse(arg))
            inp_connectors.pop(i)
    

    return dict(name=name, inputs=inp_connectors,
                outputs=out_connectors, code=code)


def _create_subgraph(visitor: 'ProgramVisitor',
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
            if isinstance(where, bool):
                if where is True:
                    pass
                elif where is False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                cond_state = state
                where_data = sdfg.arrays[where]
                if not isinstance(where_data, data.Scalar):
                    name = sdfg.temp_data_name()
                    sdfg.add_scalar(name, where_data.dtype, transient=True)
                    r = cond_state.add_read(where)
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, dace.Memlet("{}[0]".format(r)))
                true_state = sdfg.add_state(label=cond_state.label + '_true')
                state = true_state
                visitor.last_state = state
                cond = name
                cond_else = 'not ({})'.format(cond)
                sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(cond))
        tasklet = state.add_tasklet(**tasklet_params)
        inp_conn_idx = 0
        for arg in inputs:
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                inp_node = state.add_read(arg)
                state.add_edge(inp_node, None, tasklet,
                               tasklet_params['inputs'][inp_conn_idx],
                               dace.Memlet.from_array(arg, sdfg.arrays[arg]))
                inp_conn_idx += 1
        for i, arg in enumerate(outputs):
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                out_node = state.add_write(arg)
                state.add_edge(tasklet, tasklet_params['outputs'][i],
                               out_node, None,
                               dace.Memlet.from_array(arg, sdfg.arrays[arg]))
        if has_where and isinstance(where, str) and where in sdfg.arrays.keys():
            visitor._add_state(label=cond_state.label + '_true')
            sdfg.add_edge(cond_state, visitor.last_state,
                          dace.InterstateEdge(cond_else))
    else:
        # Map needed
        if has_where:
            if isinstance(where, bool):
                if where is True:
                    pass
                elif where is False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                nested_sdfg = dace.SDFG(state.label + "_where")
                nested_sdfg_inputs = dict()
                nested_sdfg_outputs = dict()
                nested_sdfg._temp_transients = sdfg._temp_transients

                idx = 0
                for arg in inputs + [where]:
                    if not (isinstance(arg, str) and arg in sdfg.arrays.keys()):
                        continue
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg.temp_data_name()
                    nested_sdfg_inputs[arg] = (conn_name, input_indices[idx])
                    idx += 1
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError
                
                for arg in outputs:
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg.temp_data_name()
                    nested_sdfg_outputs[arg] = (conn_name, output_indices)
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError
                
                cond_state = nested_sdfg.add_state(
                    label=state.label + "_where_cond", is_start_state = True)
                where_data = sdfg.arrays[where]
                if isinstance(where_data, data.Scalar):
                    name = nested_sdfg_inputs[where]
                elif isinstance(where_data, data.Array):
                    name = nested_sdfg.temp_data_name()
                    nested_sdfg.add_scalar(name, where_data.dtype,
                                           transient=True)
                    r = cond_state.add_read(nested_sdfg_inputs[where][0])
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, dace.Memlet("{}[0]".format(r)))

                sdfg._temp_transients = nested_sdfg._temp_transients

                true_state = nested_sdfg.add_state(
                    label=cond_state.label + '_where_true')
                cond = name
                cond_else = 'not ({})'.format(cond)
                nested_sdfg.add_edge(cond_state, true_state,
                                     dace.InterstateEdge(cond))

                tasklet = true_state.add_tasklet(**tasklet_params)
                idx = 0
                for arg in inputs:
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        inp_name, _ = nested_sdfg_inputs[arg]
                        inp_data = nested_sdfg.arrays[inp_name]
                        inp_node = true_state.add_read(inp_name)
                        true_state.add_edge(
                            inp_node, None, tasklet,
                            tasklet_params['inputs'][idx],
                            dace.Memlet.from_array(inp_name, inp_data))
                        idx += 1
                for i, arg in enumerate(outputs):
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        out_name, _ = nested_sdfg_outputs[arg]
                        out_data = nested_sdfg.arrays[out_name]
                        out_node = true_state.add_write(out_name)
                        true_state.add_edge(
                            tasklet, tasklet_params['outputs'][i],
                            out_node, None,
                            dace.Memlet.from_array(out_name, out_data))

                false_state = nested_sdfg.add_state(
                    label=state.label + '_where_false')
                nested_sdfg.add_edge(cond_state, false_state,
                                     dace.InterstateEdge(cond_else))
                nested_sdfg.add_edge(true_state, false_state,
                                     dace.InterstateEdge())


                codenode = state.add_nested_sdfg(
                    nested_sdfg, sdfg,
                    set([n for n, _ in nested_sdfg_inputs.values()]),
                    set([n for n, _ in nested_sdfg_outputs.values()]))
                me, mx = state.add_map(state.label + '_map', map_indices)
                for arg in inputs + [where]:
                    n = state.add_read(arg)
                    conn, idx = nested_sdfg_inputs[arg]
                    state.add_memlet_path(
                        n, me, codenode,
                        memlet=dace.Memlet("{a}[{i}]".format(a=n, i=idx)),
                        dst_conn=conn)
                for arg in outputs:
                    n = state.add_write(arg)
                    conn, idx = nested_sdfg_outputs[arg]
                    state.add_memlet_path(
                        codenode, mx, n,
                        memlet=dace.Memlet("{a}[{i}]".format(a=n, i=idx)),
                        src_conn=conn)
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
        state.add_mapped_tasklet(
            tasklet_params['name'],
            map_indices,
            input_memlets,
            tasklet_params['code'],
            output_memlets,
            external_edges=True
        )


@oprepo.replaces_ufunc('ufunc')
def implement_ufunc(visitor: 'ProgramVisitor',
                    ast_node: ast.Call,
                    sdfg: SDFG,
                    state: SDFGState,
                    ufunc_name: str,
                    args: Sequence[UfuncInput],
                    kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name,
                                  num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name,
                                    num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name,
                                      num_inputs, num_outputs, num_args,
                                      args, kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg,
                                             ufunc_name, kwargs)
    
    # Validate data shapes and apply NumPy broadcasting rules
    inp_shapes = copy.deepcopy(inputs)
    if has_where:
        inp_shapes += [where]
    (out_shape, map_indices, out_indices, inp_indices) = _validate_shapes(
         visitor, ast_node, sdfg, ufunc_name, inp_shapes, outputs)
    
    # Infer result type
    result_type, casting = _result_type(
        [sdfg.arrays[arg]
        if isinstance(arg, str) and arg in sdfg.arrays else arg
        for arg in inputs], ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.DTYPE_TO_TYPECLASS.keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor, sdfg, state, inputs, outputs, map_indices,
                     inp_indices, out_indices, out_shape, tasklet_params,
                     has_where=has_where, where=where)

    return outputs


def _validate_keepdims_kword(visitor: 'ProgramVisitor',
                             ast_node: ast.Call,
                             ufunc_name: str,
                             kwargs: Dict[str, Any]) -> bool:
    """ Validates the 'keepdims' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: Boolean value of the 'keepdims' keyword argument
    """

    keepdims = False
    if 'keepdims' in kwargs.keys():
        keepdims = kwargs['keepdims']
        if not isinstance(keepdims, (Integral, bool)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Integer or boolean value expected for keyword argument "
                "'keepdims' in reduction operation {f} (got {v}).".format(
                    f=ufunc_name, v=keepdims))
        if not isinstance(keepdims, bool):
            keepdims = bool(keepdims)
    
    return keepdims


def _validate_axis_kword(
    visitor: 'ProgramVisitor',
    ast_node: ast.Call,
    sdfg: SDFG,
    inputs: List[UfuncInput],
    kwargs: Dict[str, Any],
    keepdims: bool
) -> Tuple[Tuple[int, ...], Union[Shape, None], Shape]:
    """ Validates the 'axis' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call
        :param keepdims: Boolean value of the 'keepdims' keyword argument

        :raises DaCeSyntaxError: When validation fails

        :returns: The value of the 'axis' keyword argument, the intermediate
        data shape (if needed), and the expected output shape
    """

    # Validate 'axis' keyword
    axis = (0,)
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        inp_shape = sdfg.arrays[inputs[0]].shape
    else:
        inp_shape = [1]
    if 'axis' in kwargs.keys():
        # Set to (0,) if the keyword arg value is None
        axis = kwargs['axis'] or axis
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
    if axis is not None:
        axis = tuple(pystr_to_symbolic(a) for a in axis)
        axis = tuple(normalize_axes(axis, len(inp_shape)))
        if len(axis) > len(inp_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Axis {a} is out of bounds for data of dimension {d}".format(
                    a=axis, d=inp_shape
                )
            )
        for a in axis:
            if a >= len(inp_shape):
                raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Axis {a} is out of bounds for data of dimension {d}".format(
                    a=a, d=inp_shape
                )
            )
        if keepdims:
            intermediate_shape = [d for i, d in enumerate(inp_shape)
                                  if i not in axis]
            expected_out_shape = [d if i not in axis else 1
                                  for i, d in enumerate(inp_shape)]
        else:
            intermediate_shape = None
            expected_out_shape = [d for i, d in enumerate(inp_shape)
                                  if i not in axis]
        expected_out_shape = expected_out_shape or [1]
    else:
        if keepdims:
            intermediate_shape = [1]
            expected_out_shape = [1] * len(inp_shape)
        else:
            intermediate_shape = None
            expected_out_shape = [1]
    
    return axis, intermediate_shape, expected_out_shape


@oprepo.replaces_ufunc('reduce')
def implement_ufunc_reduce(visitor: 'ProgramVisitor',
                           ast_node: ast.Call,
                           sdfg: SDFG,
                           state: SDFGState,
                           ufunc_name: str,
                           args: Sequence[UfuncInput],
                           kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'reduce' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name,
                                  num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name,
                                    num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name,
                                      num_inputs, num_outputs, num_args,
                                      args, kwargs)

    # Validate 'keepdims' keyword
    keepdims = _validate_keepdims_kword(visitor, ast_node, ufunc_name, kwargs)
    
    # Validate 'axis' keyword
    axis, intermediate_shape, expected_out_shape = _validate_axis_kword(
        visitor, ast_node, sdfg, inputs, kwargs, keepdims)

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
                visitor, ast_node,
                "Output parameter for reduction operation {f} does not have "
                "enough dimensions (output shape {o}, expected shape {e})."
                .format(f=ufunc_name, o=out_shape, e=expected_out_shape))
        if len(out_shape) > len(expected_out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Output parameter for reduction operation {f} has too many "
                "dimensions (output shape {o}, expected shape {e})."
                .format(f=ufunc_name, o=out_shape, e=expected_out_shape))
        if (list(out_shape) != list(expected_out_shape)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Output parameter for reduction operation {f} has non-reduction"
                " dimension not equal to the input one (output shape {o}, "
                "expected shape {e}).".format(
                    f=ufunc_name, o=out_shape, e=expected_out_shape))
    else:
        out_shape = expected_out_shape
    
    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str):
        datadesc = sdfg.arrays[arg]
        result_type = datadesc.dtype
    elif isinstance(arg, Number):
        result_type = dtypes.DTYPE_TO_TYPECLASS[type(arg)]
    elif isinstance(arg, sp.Basic):
        result_type = _sym_type(arg)

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type,
                             force_scalar=True)
    if keepdims:
        if (len(intermediate_shape) == 1 and intermediate_shape[0] == 1):
            intermediate_name = sdfg.temp_data_name()
            sdfg.add_scalar(intermediate_name, result_type, transient=True)
        else:
            intermediate_name, _ = sdfg.add_temp_transient(
                intermediate_shape, result_type)
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
                    state.add_mapped_tasklet(
                        name=state.label + "_reduce_initial",
                        map_ranges={
                            "__i{i}".format(i=i): "0:{s}".format(s=s)
                            for i, s in enumerate(inpdata.shape)
                            if i not in axis
                        },
                        inputs={
                            "__inp": dace.Memlet("{a}[{i}]".format(
                                a=inputs[0], i=','.join([
                                    "0" if i in axis else "__i{i}".format(i=i)
                                    for i in range(len(inpdata.shape))
                            ])))
                        },
                        outputs={
                            "__out": dace.Memlet("{a}[{i}]".format(
                                a=intermediate_name, i=','.join([
                                    "__i{i}".format(i=i)
                                    for i in range(len(inpdata.shape))
                                    if i not in axis
                            ])))
                        },
                        code="__out = __inp",
                        external_edges=True
                    )
                else:
                    r = state.add_read(inputs[0])
                    w = state.add_write(intermediate_name)
                    state.add.nedge(
                        r, w, dace.Memlet.from_array(inputs[0], inpdata))
                state = visitor._add_state(state.label + 'b')
            else: 
                initial = intermediate_name

    # Create subgraph
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        _reduce(sdfg, state, ufunc_impl['reduce'], inputs[0], intermediate_name,
                axis=axis, identity=initial)
    else:
        tasklet = state.add_tasklet(state.label + "_tasklet", {}, {'__out'},
                                    "__out = {}".format(inputs[0]))
        out_node = state.add_write(intermediate_name)
        datadesc = sdfg.arrays[intermediate_name]
        state.add_edge(tasklet, '__out', out_node, None,
                       dace.Memlet.from_array(intermediate_name, datadesc))
    
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
        state.add_nedge(
            intermediate_node, out_node,
            dace.Memlet.from_array(outputs[0], sdfg.arrays[outputs[0]]))

    return outputs


@oprepo.replaces_ufunc('accumulate')
def implement_ufunc_accumulate(visitor: 'ProgramVisitor',
                               ast_node: ast.Call,
                               sdfg: SDFG,
                               state: SDFGState,
                               ufunc_name: str,
                               args: Sequence[UfuncInput],
                               kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'accumulate' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name,
                                  num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name,
                                    num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name,
                                      num_inputs, num_outputs, num_args,
                                      args, kwargs)
    
    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str) and arg in sdfg.arrays.keys():
        datadesc = sdfg.arrays[arg]
        if not isinstance(datadesc, data.Array):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Cannot accumulate on a dace.data.Scalar or dace.data.Stream.")
        out_shape = datadesc.shape
        result_type = datadesc.dtype
    else:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Can accumulate only on a dace.data.Array.")
    
    # Validate 'axis' keyword argument
    axis = 0
    if 'axis' in kwargs.keys():
        axis = kwargs['axis'] or axis
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            axis = axis[0]
        if not isinstance(axis, Integral):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Value of keyword argument 'axis' in 'accumulate' method of {f}"
                " must be an integer (value {v}).".format(
                    f=ufunc_name, v=axis))
        if axis >= len(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Axis {a} is out of bounds for dace.data.Array of dimension "
                "{l}".format(a=axis, l=len(out_shape)))
        # Normalize negative axis
        axis = normalize_axes([axis], len(out_shape))[0]
    
    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Create subgraph
    shape = datadesc.shape
    map_range = {"__i{}".format(i): "0:{}".format(s)
                 for i, s in enumerate(shape) if i != axis}
    input_idx = ','.join(["__i{}".format(i)
                          if i != axis else "0:{}".format(shape[i])
                          for i in range(len(shape))])
    output_idx = ','.join(["__i{}".format(i)
                           if i != axis else "0:{}".format(shape[i])
                           for i in range(len(shape))])

    nested_sdfg = dace.SDFG(state.label + "_for_loop")
    nested_sdfg._temp_transients = sdfg._temp_transients
    inpconn = nested_sdfg.temp_data_name()
    outconn = nested_sdfg.temp_data_name()
    shape = [datadesc.shape[axis]]
    strides = [datadesc.strides[axis]]
    nested_sdfg.add_array(inpconn, shape, result_type, strides=strides)
    nested_sdfg.add_array(outconn, shape, result_type, strides=strides)
    
    init_state = nested_sdfg.add_state(label="init")
    r = init_state.add_read(inpconn)
    w = init_state.add_write(outconn)
    init_state.add_nedge(r, w, dace.Memlet("{a}[{i}] -> {oi}".format(
        a=inpconn, i='0', oi='0')))

    body_state = nested_sdfg.add_state(label="body")
    r1 = body_state.add_read(inpconn)
    r2 = body_state.add_read(outconn)
    w = body_state.add_write(outconn)
    t = body_state.add_tasklet(
        name=state.label + "_for_loop_tasklet",
        inputs=ufunc_impl['inputs'],
        outputs=ufunc_impl['outputs'],
        code=ufunc_impl['code']
    )

    loop_idx = "__i{}".format(axis)
    loop_idx_m1 = "__i{} - 1".format(axis)
    body_state.add_edge(r1, None, t, '__in1',
                        dace.Memlet("{a}[{i}]".format(a=inpconn, i=loop_idx)))
    body_state.add_edge(
        r2, None, t, '__in2',
        dace.Memlet("{a}[{i}]".format(a=outconn, i=loop_idx_m1)))
    body_state.add_edge(t, '__out', w, None,
                        dace.Memlet("{a}[{i}]".format(a=outconn, i=loop_idx)))

    init_expr = str(1)
    cond_expr = "__i{i} < {s}".format(i=axis, s=shape[0])
    incr_expr = "__i{} + 1".format(axis)
    nested_sdfg.add_loop(init_state, body_state, None, loop_idx,
                         init_expr, cond_expr, incr_expr)

    sdfg._temp_transients = nested_sdfg._temp_transients

    r = state.add_read(inputs[0])
    w = state.add_write(outputs[0])
    codenode = state.add_nested_sdfg(nested_sdfg, sdfg,
                                        {inpconn}, {outconn})
    me, mx = state.add_map(state.label + '_map', map_range)
    state.add_memlet_path(
        r, me, codenode,
        memlet=dace.Memlet("{a}[{i}]".format(a=inputs[0], i=input_idx)),
        dst_conn=inpconn)
    state.add_memlet_path(
        codenode, mx, w,
        memlet=dace.Memlet("{a}[{i}]".format(a=outputs[0], i=output_idx)),
        src_conn=outconn)

    return outputs


@oprepo.replaces_ufunc('outer')
def implement_ufunc_outer(visitor: 'ProgramVisitor',
                          ast_node: ast.Call,
                          sdfg: SDFG,
                          state: SDFGState,
                          ufunc_name: str,
                          args: Sequence[UfuncInput],
                          kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'outer' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name,
                                  num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name,
                                    num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name,
                                      num_inputs, num_outputs, num_args,
                                      args, kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg,
                                             ufunc_name, kwargs)

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
                map_vars.extend(["__i{i}_{j}".format(i=i, j=j)
                                 for j in range(len(shape))])
                map_range.update({
                    "__i{i}_{j}".format(i=i, j=j): "0:{}".format(sz)
                    for j, sz in enumerate(shape)})
                input_idx = ','.join(["__i{i}_{j}".format(i=i, j=j)
                                      for j in range(len(shape))])
                if output_idx:
                    output_idx = ','.join([output_idx, input_idx])
                else:
                    output_idx = input_idx
            else:
                raise mem_parser.DaceSyntaxError(
                    visitor, ast_node,
                    "Unsuported data type {t} in 'outer' method of NumPy ufunc "
                    "{f}.".format(t=type(datadesc), f=ufunc_name))
        elif isinstance(arg, (Number, sp.Basic)):
            input_idx = None
        input_indices.append(input_idx)

    if has_where and not isinstance(where, bool):
        where_shape = sdfg.arrays[where].shape
        try:
            bcast_out_shape, _, _, bcast_inp_indices = _broadcast(
                [out_shape, where_shape])
        except SyntaxError:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "'where' shape {w} could not be broadcast together with 'out' "
                "shape {o}.".format(w=where_shape, o=out_shape)
            )
        if list(bcast_out_shape) != list(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node,
                "Broadcasting 'where' shape {w} together with expected 'out' "
                "shape {o} resulted in a different output shape {no}. This is "
                "currently unsupported.".format(
                    w=where_shape, o=out_shape, no=bcast_out_shape))
        where_idx = bcast_inp_indices[1]
        for i in range(len(out_shape)):
            where_idx = where_idx.replace("__i{}".format(i), map_vars[i])
        input_indices.append(where_idx)
    else:
        input_indices.append(None)
    
    # Infer result type
    result_type, casting = _result_type(
        [sdfg.arrays[arg]
        if isinstance(arg, str) and arg in sdfg.arrays else arg
        for arg in inputs], ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.DTYPE_TO_TYPECLASS.keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor, sdfg, state, inputs, outputs, map_range,
                     input_indices, output_idx, out_shape, tasklet_params,
                     has_where=has_where, where=where)

    return outputs


# Datatype converter #########################################################

def _make_datatype_converter(typeclass: str):
    if typeclass in {"int", "float", "complex"}:
        dtype = dtypes.DTYPE_TO_TYPECLASS[eval(typeclass)]
    else:
        dtype = dtypes.DTYPE_TO_TYPECLASS[eval("np.{}".format(typeclass))]
    @oprepo.replaces(typeclass)
    @oprepo.replaces("dace.{}".format(typeclass))
    @oprepo.replaces("numpy.{}".format(typeclass))
    def _converter(sdfg: SDFG, state: SDFGState, arg: UfuncInput):
        return _datatype_converter(sdfg, state, arg, dtype=dtype)


for typeclass in dtypes.TYPECLASS_STRINGS:
    _make_datatype_converter(typeclass)


def _datatype_converter(sdfg: SDFG,
                        state: SDFGState,
                        arg: UfuncInput,
                        dtype: dtypes.typeclass) -> UfuncOutput:
    """ Out-of-place datatype conversion of the input argument.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param arg: Input argument
        :param dtype: Datatype to convert input argument into

        :returns: dace.data.Array of same size as input or dace.data.Scalar
    """

    # Get shape and indices
    (out_shape, map_indices, out_indices, inp_indices) = _validate_shapes(
         None, None, sdfg, None, [arg], [None])

    # Create output data
    outputs = _create_output(sdfg, [arg], [None], out_shape, dtype)

    # Set tasklet parameters
    impl = {
        'name': "_convert_to_{}_".format(dtype.to_string()),
        'inputs': ['__inp'],
        'outputs': ['__out'],
        'code': "__out = dace.{}(__inp)".format(dtype.to_string())
    }
    tasklet_params = _set_tasklet_params(impl, [arg])

    # Visitor input only needed when `has_where == True`.
    _create_subgraph(None, sdfg, state, [arg], outputs, map_indices,
                     inp_indices, out_indices, out_shape, tasklet_params,
                     has_where=False, where=None)
    
    return outputs
