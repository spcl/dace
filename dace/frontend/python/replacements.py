import dace

import ast
import copy
from functools import reduce
from typing import Any, Dict, Union, Callable

import dace
from dace.config import Config
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.frontend.common import op_repository as oprepo
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic

import numpy as np

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
            input: str,
            output=None,
            axis=None,
            identity=None):
    # TODO(later): If output is None, derive the output size from the input and create a new node
    if output is None:
        inarr = input
        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
        input_subset = _parse_memlet_subset(sdfg.arrays[inarr],
                                            ast.parse(input).body[0].value, {})
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
        inarr = input
        outarr = output

        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)

        # Compute memlets
        input_subset = _parse_memlet_subset(sdfg.arrays[inarr],
                                            ast.parse(input).body[0].value, {})
        input_memlet = Memlet(inarr, input_subset.num_elements(), input_subset,
                              1)
        output_subset = _parse_memlet_subset(sdfg.arrays[outarr],
                                             ast.parse(output).body[0].value,
                                             {})
        output_memlet = Memlet(outarr, output_subset.num_elements(),
                               output_subset, 1)

    # Create reduce subgraph
    inpnode = state.add_read(inarr)
    rednode = state.add_reduce(redfunction, axis, identity)
    outnode = state.add_write(outarr)
    state.add_nedge(inpnode, rednode, input_memlet)
    state.add_nedge(rednode, outnode, output_memlet)

    if output is None:
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


def _binop(sdfg: SDFG, state: SDFGState, op1: str, op2: str, opcode: str,
           opname: str, restype: dace.typeclass):
    """ Implements a general element-wise array binary operator. """
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]
    if (len(arr1.shape) != len(arr2.shape)
            or any(s1 != s2 for s1, s2 in zip(arr1.shape, arr2.shape))):
        raise SyntaxError('Array sizes must match')

    name, _ = sdfg.add_temp_transient(arr1.shape, restype, arr1.storage)
    state.add_mapped_tasklet(
        "_%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr1.shape)}, {
             '__in1':
             Memlet.simple(
                 op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))])),
             '__in2':
             Memlet.simple(
                 op2, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
         },
        '__out = __in1 %s __in2' % opcode, {
            '__out':
            Memlet.simple(
                name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))
        },
        external_edges=True)
    return name


def _scalarbinop(sdfg: SDFG,
                 state: SDFGState,
                 scalop: str,
                 arrop: str,
                 opcode: str,
                 opname: str,
                 restype: dace.typeclass,
                 reverse: bool = False):
    """ Implements a general Scalar-Array binary operator. """
    scalar = sdfg.arrays[scalop]
    arr = sdfg.arrays[arrop]

    name, _ = sdfg.add_temp_transient(arr.shape, restype, arr.storage)
    state.add_mapped_tasklet(
        "_SA%s_" % opname,
        {'__i%d' % i: '0:%s' % s
         for i, s in enumerate(arr.shape)}, {
             '__in1':
             Memlet.simple(scalop, '0'),
             '__in2':
             Memlet.simple(
                 arrop, ','.join(['__i%d' % i
                                  for i in range(len(arr.shape))])),
         },
        '__out = %s %s %s' % ('__in2' if reverse else '__in1', opcode,
                              '__in1' if reverse else '__in2'),
        {
            '__out':
            Memlet.simple(
                name, ','.join(['__i%d' % i for i in range(len(arr.shape))]))
        },
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


def _inverse_dict_lookup(dict: Dict[str, Any], value: Any):
    """ Finds the first key in a dictionary with the input value. """
    for k, v in dict.items():
        if v == value:
            return k
    return None


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
        type1 = _inverse_dict_lookup(visitor.numbers, op1)
    arr2 = sdfg.arrays[op2]
    type2 = arr2.dtype.type
    isscal2 = _is_scalar(sdfg, op2)
    isnum2 = isscal2 and (op2 in visitor.numbers.values())
    if isnum2:
        type2 = _inverse_dict_lookup(visitor.numbers, op2)
    if _is_op_boolean(op):
        restype = dace.bool
    else:
        restype = dace.DTYPE_TO_TYPECLASS[np.result_type(type1, type2).type]

    if isscal1:
        if isscal2:
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
            return _scalarbinop(sdfg, state, op1, op2, opcode, op, restype)
    else:
        if isscal2:
            return _scalarbinop(sdfg, state, op2, op1, opcode, op, restype,
                                True)
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
