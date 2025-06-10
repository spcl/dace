# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements of reduction operations, which cover both NumPy's Mathematical Functions (e.g., ``numpy.sum``)
and Sorting, Searching, and Counting Functions (e.g., ``numpy.argmax``).
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.nested_call import NestedCall
from dace.frontend.python.replacements.utils import ProgramVisitor, normalize_axes
from dace import dtypes, nodes, subsets, symbolic, Memlet, SDFG, SDFGState

import copy
import functools
from numbers import Integral, Number
from typing import Any, Dict, Callable, Optional, Union


@oprepo.replaces('dace.reduce')
def reduce(pv: ProgramVisitor,
           sdfg: SDFG,
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
            axis = tuple(symbolic.pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_shape = None

        # check if we are reducing along all axes
        if axis is not None and len(axis) == len(input_subset.size()):
            reduce_all = all(x == y for x, y in zip(axis, range(len(input_subset.size()))))
        else:
            reduce_all = False

        if axis is None or reduce_all:
            output_shape = [1]
        else:
            output_subset = copy.deepcopy(input_subset)
            output_subset.pop(axis)
            output_shape = output_subset.size()
        if (len(output_shape) == 1 and output_shape[0] == 1):
            outarr = sdfg.temp_data_name()
            outarr, arr = sdfg.add_scalar(outarr, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage, transient=True)
        else:
            outarr, arr = sdfg.add_temp_transient(output_shape, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage)
        output_memlet = Memlet.from_array(outarr, arr)
    else:
        inarr = in_array
        outarr = out_array

        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(symbolic.pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        # Compute memlets
        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_subset = subsets.Range.from_array(sdfg.arrays[outarr])
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


@oprepo.replaces('numpy.sum')
def _sum(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return reduce(pv, sdfg, state, "lambda x, y: x + y", a, axis=axis, identity=0)


@oprepo.replaces('sum')
def _sum_array(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str):
    # sum(numpy_array) is equivalent to np.sum(numpy_array, axis=0)
    return reduce(pv, sdfg, state, "lambda x, y: x + y", a, axis=0, identity=0)


@oprepo.replaces('numpy.any')
def _any(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return reduce(pv, sdfg, state, "lambda x, y: x or y", a, axis=axis, identity=0)


@oprepo.replaces('numpy.all')
def _all(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return reduce(pv, sdfg, state, "lambda x, y: x and y", a, axis=axis, identity=0)


@oprepo.replaces('numpy.mean')
def _mean(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)

    sum = nest(_sum)(a, axis=axis)

    if axis is None:
        div_amount = functools.reduce(lambda x, y: x * y, (d for d in sdfg.arrays[a].shape))
    elif isinstance(axis, (tuple, list)):
        axis = normalize_axes(axis, len(sdfg.arrays[a].shape))
        # each entry needs to be divided by the size of the reduction
        div_amount = functools.reduce(lambda x, y: x * y, (d for i, d in enumerate(sdfg.arrays[a].shape) if i in axis))
    else:
        div_amount = sdfg.arrays[a].shape[axis]

    return nest, nest(elementwise)("lambda x: x / ({})".format(div_amount), sum)


@oprepo.replaces('numpy.max')
@oprepo.replaces('numpy.amax')
def _max(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, initial=None):
    initial = initial if initial is not None else dtypes.min_value(sdfg.arrays[a].dtype)
    return reduce(pv, sdfg, state, "lambda x, y: max(x, y)", a, axis=axis, identity=initial)


@oprepo.replaces('numpy.min')
@oprepo.replaces('numpy.amin')
def _min(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, initial=None):
    initial = initial if initial is not None else dtypes.max_value(sdfg.arrays[a].dtype)
    return reduce(pv, sdfg, state, "lambda x, y: min(x, y)", a, axis=axis, identity=initial)


@oprepo.replaces_method('Array', 'max')
@oprepo.replaces_method('Scalar', 'max')
@oprepo.replaces_method('View', 'max')
def _ndarray_max(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    from dace.frontend.python.replacements.ufunc import implement_ufunc_reduce  # Avoid import loop
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'maximum', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'min')
@oprepo.replaces_method('Scalar', 'min')
@oprepo.replaces_method('View', 'min')
def _ndarray_min(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    from dace.frontend.python.replacements.ufunc import implement_ufunc_reduce  # Avoid import loop
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'minimum', [arr], kwargs)[0]


def _minmax2(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, b: str, ismin=True):
    """ Implements the min or max function with 2 scalar arguments. """
    from dace.frontend.python.replacements.array_creation_dace import _define_local_scalar
    from dace.frontend.python.replacements.operators import result_type

    in_conn = set()
    out_conn = {'__out'}

    if isinstance(a, str) and a in sdfg.arrays.keys():
        desc_a = sdfg.arrays[a]
        read_a = state.add_read(a)
        conn_a = '__in_a'
        in_conn.add(conn_a)
    else:
        desc_a = a
        read_a = None
        conn_a = symbolic.symstr(a)

    if isinstance(b, str) and b in sdfg.arrays.keys():
        desc_b = sdfg.arrays[b]
        read_b = state.add_read(b)
        conn_b = '__in_b'
        in_conn.add(conn_b)
    else:
        desc_b = b
        read_b = None
        conn_b = symbolic.symstr(b)

    dtype_c, [cast_a, cast_b] = result_type([desc_a, desc_b])
    arg_a, arg_b = "{in1}".format(in1=conn_a), "{in2}".format(in2=conn_b)
    if cast_a:
        arg_a = "{ca}({in1})".format(ca=str(cast_a).replace('::', '.'), in1=conn_a)
    if cast_b:
        arg_b = "{cb}({in2})".format(cb=str(cast_b).replace('::', '.'), in2=conn_b)

    func = 'min' if ismin else 'max'
    tasklet = nodes.Tasklet(f'__{func}2', in_conn, out_conn, f'__out = {func}({arg_a}, {arg_b})')

    c = _define_local_scalar(pv, sdfg, state, dtype_c)
    desc_c = sdfg.arrays[c]
    write_c = state.add_write(c)
    if read_a:
        state.add_edge(read_a, None, tasklet, '__in_a', Memlet.from_array(a, desc_a))
    if read_b:
        state.add_edge(read_b, None, tasklet, '__in_b', Memlet.from_array(b, desc_b))
    state.add_edge(tasklet, '__out', write_c, None, Memlet.from_array(c, desc_c))

    return c


# NOTE: We support only the version of Python max that takes scalar arguments.
# For iterable arguments one must use the equivalent NumPy methods.
@oprepo.replaces('max')
def _pymax(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: Union[str, Number, symbolic.symbol], *args):
    left_arg = a
    current_state = state
    for i, b in enumerate(args):
        if i > 0:
            pv._add_state('__min2_%d' % i)
            pv.last_block.set_default_lineinfo(pv.current_lineinfo)
            current_state = pv.last_block
        left_arg = _minmax2(pv, sdfg, current_state, left_arg, b, ismin=False)
    return left_arg


# NOTE: We support only the version of Python min that takes scalar arguments.
# For iterable arguments one must use the equivalent NumPy methods.
@oprepo.replaces('min')
def _pymin(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: Union[str, Number, symbolic.symbol], *args):
    left_arg = a
    current_state = state
    for i, b in enumerate(args):
        if i > 0:
            pv._add_state('__min2_%d' % i)
            pv.last_block.set_default_lineinfo(pv.current_lineinfo)
            current_state = pv.last_block
        left_arg = _minmax2(pv, sdfg, current_state, left_arg, b)
    return left_arg


@oprepo.replaces('numpy.argmax')
def _argmax(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            a: str,
            axis: Optional[int] = None,
            result_type=dtypes.int32):
    return _argminmax(pv, sdfg, state, a, axis, func="max", result_type=result_type)


@oprepo.replaces('numpy.argmin')
def _argmin(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            a: str,
            axis: Optional[int] = None,
            result_type=dtypes.int32):
    return _argminmax(pv, sdfg, state, a, axis, func="min", result_type=result_type)


def _argminmax(pv: ProgramVisitor,
               sdfg: SDFG,
               state: SDFGState,
               a: str,
               axis: Optional[int],
               func: str,
               result_type: dtypes.typeclass = dtypes.int32,
               return_both: bool = False):
    nest = NestedCall(pv, sdfg, state)

    assert func in ['min', 'max']

    # Flatten the array if axis is not given
    if axis is None:
        from dace.frontend.python.replacements.array_manipulation import flat  # Avoid import loop
        axis = 0
        a = flat(pv, sdfg, state, a)

    if not isinstance(axis, Integral):
        raise SyntaxError('Axis must be an int')

    a_arr = sdfg.arrays[a]

    if not 0 <= axis < len(a_arr.shape):
        raise SyntaxError("Expected 0 <= axis < len({}.shape), got {}".format(a, axis))

    reduced_shape = list(copy.deepcopy(a_arr.shape))
    reduced_shape.pop(axis)
    if not reduced_shape:
        reduced_shape = [1]

    val_and_idx = dtypes.struct('_val_and_idx', idx=result_type, val=a_arr.dtype)

    # HACK: since identity cannot be specified for structs, we have to init the output array
    reduced_structs, reduced_struct_arr = sdfg.add_temp_transient(reduced_shape, val_and_idx)

    code = "__init = _val_and_idx(val={}, idx=-1)".format(
        dtypes.min_value(a_arr.dtype) if func == 'max' else dtypes.max_value(a_arr.dtype))

    reduced_expr = ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis)
    reduced_maprange = {'__i%d' % i: '0:%s' % n for i, n in enumerate(a_arr.shape) if i != axis}
    if not reduced_expr:
        reduced_expr = '0'
        reduced_maprange = {'__i0': '0:1'}
    nest.add_state().add_mapped_tasklet(name="_arg{}_convert_".format(func),
                                        map_ranges=reduced_maprange,
                                        inputs={},
                                        code=code,
                                        outputs={'__init': Memlet.simple(reduced_structs, reduced_expr)},
                                        external_edges=True)

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_reduce_".format(func),
        map_ranges={
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(a_arr.shape)
        },
        inputs={'__in': Memlet.simple(a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))},
        code="__out = _val_and_idx(idx={}, val=__in)".format("__i%d" % axis),
        outputs={
            '__out':
            Memlet.simple(reduced_structs,
                          reduced_expr,
                          wcr_str=("lambda x, y:"
                                   "_val_and_idx(val={}(x.val, y.val), "
                                   "idx=(y.idx if x.val {} y.val else x.idx))").format(
                                       func, '<' if func == 'max' else '>'))
        },
        external_edges=True)

    if return_both:
        outidx, outidxarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        outval, outvalarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, a_arr.dtype)

        nest.add_state().add_mapped_tasklet(name="_arg{}_extract_".format(func),
                                            map_ranges=reduced_maprange,
                                            inputs={'__in': Memlet.simple(reduced_structs, reduced_expr)},
                                            code="__out_val = __in.val\n__out_idx = __in.idx",
                                            outputs={
                                                '__out_val': Memlet.simple(outval, reduced_expr),
                                                '__out_idx': Memlet.simple(outidx, reduced_expr)
                                            },
                                            external_edges=True)

        return nest, (outval, outidx)

    else:
        # map to result_type
        out, outarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        nest(elementwise)("lambda x: x.idx", reduced_structs, out_array=out)
        return nest, out


@oprepo.replaces_method('Array', 'argmax')
@oprepo.replaces_method('Scalar', 'argmax')
@oprepo.replaces_method('View', 'argmax')
def _ndarray_argmax(pv: ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    arr: str,
                    axis: int = None,
                    out: str = None) -> str:
    nest, newarr = _argmax(pv, sdfg, state, arr, axis)
    if out:
        r = state.add_read(newarr)
        w = state.add_write(out)
        state.add_nedge(r, w, Memlet.from_array(newarr, sdfg.arrays[newarr]))
        newarr = out
    return newarr


@oprepo.replaces_method('Array', 'argmin')
@oprepo.replaces_method('Scalar', 'argmin')
@oprepo.replaces_method('View', 'argmin')
def _ndarray_argmin(pv: ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    arr: str,
                    axis: int = None,
                    out: str = None) -> str:
    nest, newarr = _argmin(pv, sdfg, state, arr, axis)
    if out:
        r = state.add_read(newarr)
        w = state.add_write(out)
        state.add_nedge(r, w, Memlet.from_array(newarr, sdfg.arrays[newarr]))
        newarr = out
    return newarr
