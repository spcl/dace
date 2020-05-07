import dace

import ast
import copy
import itertools
from functools import reduce
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
        input_memlet = Memlet(inarr, input_subset.num_elements(), input_subset,
                              1)
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
        input_memlet = Memlet(inarr, input_subset.num_elements(), input_subset,
                              1)
        output_subset = parse_memlet_subset(sdfg.arrays[outarr],
                                            ast.parse(out_array).body[0].value,
                                            {})
        output_memlet = Memlet(outarr, output_subset.num_elements(),
                               output_subset, 1)

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
def _exp(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'exp')


@oprepo.replaces('sin')
@oprepo.replaces('dace.sin')
@oprepo.replaces('numpy.sin')
def _sin(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sin')


@oprepo.replaces('cos')
@oprepo.replaces('dace.cos')
@oprepo.replaces('numpy.cos')
def _cos(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'cos')


@oprepo.replaces('sqrt')
@oprepo.replaces('dace.sqrt')
@oprepo.replaces('numpy.sqrt')
def _sqrt(sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sqrt')


@oprepo.replaces('log')
@oprepo.replaces('dace.log')
@oprepo.replaces('numpy.log')
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
    return _simple_call(sdfg, state, input, 'real',
                        _complex_to_scalar(inptype))


@oprepo.replaces('imag')
@oprepo.replaces('dace.imag')
@oprepo.replaces('numpy.imag')
def _imag(sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return _simple_call(sdfg, state, input, 'imag',
                        _complex_to_scalar(inptype))


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
def _max(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    # HACK: reduce doesn't work if identity isn't specified (at the moment)
    return _reduce(sdfg,
                   state,
                   "lambda x, y: max(x, y)",
                   a,
                   axis=axis,
                   identity=-9999999999999)


@oprepo.replaces('numpy.min')
def _min(sdfg: SDFG, state: SDFGState, a: str, axis=None):
    # HACK: reduce doesn't work if identity isn't specified (at the moment)
    return _reduce(sdfg,
                   state,
                   "lambda x, y: min(x, y)",
                   a,
                   axis=axis,
                   identity=999999999999)


@oprepo.replaces('numpy.argmax')
def _argmax(sdfg: SDFG,
            state: SDFGState,
            a: str,
            axis,
            result_type=dace.int32):
    return _argminmax(sdfg,
                      state,
                      a,
                      axis,
                      func="max",
                      result_type=result_type)


@oprepo.replaces('numpy.argmin')
def _argmin(sdfg: SDFG,
            state: SDFGState,
            a: str,
            axis,
            result_type=dace.int32):
    return _argminmax(sdfg,
                      state,
                      a,
                      axis,
                      func="min",
                      result_type=result_type)


def _argminmax(sdfg: SDFG,
               state: SDFGState,
               a: str,
               axis,
               func,
               result_type=dace.int32):
    nest = NestedCall(sdfg, state)

    assert func in ['min', 'max']

    if axis is None or type(axis) is not int:
        raise SyntaxError('Axis must be an int')

    a_arr = sdfg.arrays[a]

    if not 0 <= axis < len(a_arr.shape):
        raise SyntaxError("Expected 0 <= axis < len({}.shape), got {}".format(
            a, axis))

    reduced_shape = list(copy.deepcopy(a_arr.shape))
    reduced_shape.pop(axis)

    val_and_idx = dace.struct('_val_and_idx', val=a_arr.dtype, idx=result_type)

    # convert to array of structs
    structs, structs_arr = sdfg.add_temp_transient(a_arr.shape, val_and_idx)

    # HACK: at the time of writing, the reduce op on structs doesn't work unless we init the output
    # array for that reason we will init the output array with (-1, -inf) in the loop. This should
    # be removed once the issue is resolved
    reduced_structs, reduced_struct_arr = sdfg.add_temp_transient(
        reduced_shape, val_and_idx)

    code = ("__out = _val_and_idx(val=__in, idx=__i{})\n".format(axis) +
            "__init = _val_and_idx(val={}1e38, idx=-1)".format('-' if func ==
                                                               'max' else ''))

    state.add_mapped_tasklet(
        name="_arg{}_convert_".format(func),
        map_ranges={
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(a_arr.shape)
        },
        inputs={
            "__in":
            Memlet.simple(
                a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))
        },
        code=code,
        outputs={
            '__init':
            Memlet.simple(
                reduced_structs, ','.join('__i%d' % i
                                          for i in range(len(a_arr.shape))
                                          if i != axis)),
            '__out':
            Memlet.simple(
                structs,
                ','.join('__i%d' % i for i in range(len(a_arr.shape))))
        },
        external_edges=True)

    # reduce array of structs
    nest(_reduce)(  # comment for yapf
        "lambda x, y: _val_and_idx(val={}(x.val, y.val), idx=(x.idx if x.val {} y.val else y.idx))"
        .format(func, '>' if func == 'max' else '<'),
        structs,
        out_array=reduced_structs,
        axis=axis)

    # map to int64
    out, outarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape,
                                          dace.int64)
    nest(_elementwise)("lambda x: x.idx", reduced_structs, out_array=out)

    return nest, out


##############################################################################
# Python operation replacements ##############################################
##############################################################################


def _assignop(sdfg: SDFG, state: SDFGState, op1: str, opcode: str,
              opname: str):
    """ Implements a general element-wise array assignment operator. """
    arr1 = sdfg.arrays[op1]

    name, _ = sdfg.add_temp_transient(arr1.shape, arr1.dtype, arr1.storage)
    write_memlet = None
    if opcode:
        write_memlet = Memlet.simple(
            name,
            ','.join(['__i%d' % i for i in range(len(arr1.shape))]),
            wcr_str='lambda x, y: x %s y' % opcode)
    else:
        write_memlet = Memlet.simple(
            name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
    state.add_mapped_tasklet(
        "_%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr1.shape)}, {
             '__in1':
             Memlet.simple(
                 op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
         },
        '__out = __in1', {'__out': write_memlet},
        external_edges=True)
    return name


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

    all_idx_dict = {k: "0:" + str(v) for k, v in all_idx_dict.items()}

    return out_shape, all_idx_dict, to_string(all_idx), to_string(
        a1_idx), to_string(a2_idx)


def _binop(sdfg: SDFG, state: SDFGState, op1: str, op2: str, opcode: str,
           opname: str, restype: dace.typeclass):
    """ Implements a general element-wise array binary operator. """
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    out_shape, all_idx_dict, all_idx, arr1_idx, arr2_idx = _broadcast_together(
        arr1.shape, arr2.shape)

    name, _ = sdfg.add_temp_transient(out_shape, restype, arr1.storage)
    state.add_mapped_tasklet("_%s_" % opname,
                             all_idx_dict, {
                                 '__in1': Memlet.simple(op1, arr1_idx),
                                 '__in2': Memlet.simple(op2, arr2_idx)
                             },
                             '__out = __in1 %s __in2' % opcode,
                             {'__out': Memlet.simple(name, all_idx)},
                             external_edges=True)
    return name


# Defined as a function in order to include the op and the opcode in the closure
def _makeassignop(op, opcode):
    @oprepo.replaces_operator('Array', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: str,
            op2=None):
        return _assignop(sdfg, state, op1, opcode, op)


def _makeunop(op, opcode):
    @oprepo.replaces_operator('Array', op)
    def _op(visitor: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            op1: str,
            op2=None):
        return _unop(sdfg, state, op1, opcode, op)


@oprepo.replaces_operator('int', 'USub', None)
@oprepo.replaces_operator('float', 'USub', None)
def _neg(visitor: 'ProgramVisitor',
         sdfg: SDFG,
         state: SDFGState,
         op1: Union[int, float],
         op2=None):
    return -op1


@oprepo.replaces_operator('symbol', 'Add', 'int')
@oprepo.replaces_operator('symbol', 'Add', 'float')
def _addsym(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
            op1: symbolic.symbol, op2: Union[int, float]):
    return op1 + op2


def _is_scalar(sdfg: SDFG, arrname: str):
    """ Checks whether array is pseudo-scalar (shape=(1,)). """
    shape = sdfg.arrays[arrname].shape
    if len(shape) == 1 and shape[0] == 1:
        return True
    return False


def _is_op_boolean(op: str):
    if op in {'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE'}:
        return True
    return False


def _array_x_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState,
                   op1: str, op2: str, op: str, opcode: str):

    arr1 = sdfg.arrays[op1]
    type1 = arr1.dtype.type
    isscal1 = _is_scalar(sdfg, op1)
    isnum1 = isscal1 and (op1 in visitor.numbers.values())
    if isnum1:
        type1 = inverse_dict_lookup(visitor.numbers, op1)
    arr2 = sdfg.arrays[op2]
    type2 = arr2.dtype.type
    isscal2 = _is_scalar(sdfg, op2)
    isnum2 = isscal2 and (op2 in visitor.numbers.values())
    if isnum2:
        type2 = inverse_dict_lookup(visitor.numbers, op2)
    if _is_op_boolean(op):
        restype = dace.bool
    else:
        restype = dace.DTYPE_TO_TYPECLASS[np.result_type(type1, type2).type]

    if isscal1 and isscal2:
        arr1 = sdfg.arrays[op1]
        arr2 = sdfg.arrays[op2]
        op3, arr3 = sdfg.add_temp_transient([1], restype, arr2.storage)
        tasklet = state.add_tasklet('_SS%s_' % op, {'s1', 's2'}, {'s3'},
                                    's3 = s1 %s s2' % opcode)
        n1 = state.add_read(op1)
        n2 = state.add_read(op2)
        n3 = state.add_write(op3)
        state.add_edge(n1, None, tasklet, 's1',
                       dace.Memlet.from_array(op1, arr1))
        state.add_edge(n2, None, tasklet, 's2',
                       dace.Memlet.from_array(op2, arr2))
        state.add_edge(tasklet, 's3', n3, None,
                       dace.Memlet.from_array(op3, arr3))
        return op3
    else:
        return _binop(sdfg, state, op1, op2, opcode, op, restype)


def _makebinop(op, opcode):
    @oprepo.replaces_operator('Array', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_x_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_x_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_x_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str,
            op2: str):
        return _array_x_binop(visitor, sdfg, state, op1, op2, op, opcode)


# Define all standard Python augmented assignment operators
for op, opcode in [
    ('None', None),
    ('Add', '+'),
    ('Sub', '-'),
    ('Mult', '*'),
    ('Div', '/'),
    ('FloorDiv', '//'),
    ('Mod', '%'),
    ('Pow', '**'),
    ('LShift', '<<'),
    ('RShift', '>>'),
    ('BitOr', '|'),
    ('BitXor', '^'),
    ('BitAnd', '&'),
]:
    _makeassignop(op, opcode)

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
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]
    # TODO: Apply numpy broadcast rules
    if len(arr1.shape) > 3 or len(arr2.shape) > 3:
        raise SyntaxError('Matrix multiplication of tensors of dimensions > 3 '
                          'not supported')
    if arr1.shape[-1] != arr2.shape[-2]:
        raise SyntaxError('Matrix dimension mismatch %s != %s' %
                          (arr1.shape[-1], arr2.shape[-2]))

    import dace.libraries.blas as blas  # Avoid import loop
    from dace.libraries.blas.nodes.matmul import get_batchmm_opts

    # Determine batched multiplication
    bopt = get_batchmm_opts(arr1.shape, arr1.strides, arr2.shape, arr2.strides,
                            None, None)
    if bopt:
        output_shape = (bopt['b'], arr1.shape[-2], arr2.shape[-1])
    else:
        output_shape = (arr1.shape[-2], arr2.shape[-1])

    type1 = arr1.dtype.type
    type2 = arr2.dtype.type
    restype = dace.DTYPE_TO_TYPECLASS[np.result_type(type1, type2).type]

    op3, arr3 = sdfg.add_temp_transient(output_shape, restype, arr1.storage)

    acc1 = state.add_read(op1)
    acc2 = state.add_read(op2)
    acc3 = state.add_write(op3)

    tasklet = blas.MatMul('_MatMult_', restype)
    state.add_node(tasklet)
    state.add_edge(acc1, None, tasklet, '_a',
                   dace.Memlet.from_array(op1, arr1))
    state.add_edge(acc2, None, tasklet, '_b',
                   dace.Memlet.from_array(op2, arr2))
    state.add_edge(tasklet, '_c', acc3, None,
                   dace.Memlet.from_array(op3, arr3))

    return op3
