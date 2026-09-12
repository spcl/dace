# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements of reduction operations, which cover both NumPy's Mathematical Functions (e.g., ``numpy.sum``)
and Sorting, Searching, and Counting Functions (e.g., ``numpy.argmax``).
"""
import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.nested_call import NestedCall
from dace.frontend.python.replacements.utils import ProgramVisitor, normalize_axes
from dace import data, dtypes, nodes, subsets, symbolic, Memlet, SDFG, SDFGState

import copy
import functools
import numpy
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
            outarr = pv.get_target_name()
            outarr, arr = sdfg.add_scalar(outarr,
                                          sdfg.arrays[inarr].dtype,
                                          sdfg.arrays[inarr].storage,
                                          transient=True,
                                          find_new_name=True)
        else:
            outarr, arr = pv.add_temp_transient(output_shape, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage)
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
    state.add_edge(inpnode, None, rednode, '_in', input_memlet)
    state.add_edge(rednode, '_out', outnode, None, output_memlet)

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


def as_bool(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    """The operand's truth values. ``any``/``all`` reduce over these, not over the operand itself."""
    from dace.frontend.python.replacements.array_manipulation import _ndarray_astype  # Avoid import loop

    if sdfg.arrays[arr].dtype == dtypes.bool_:
        return arr
    return _ndarray_astype(pv, sdfg, state, arr, dtypes.bool_)


@oprepo.replaces('numpy.any')
def _any(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return reduce(pv, sdfg, state, "lambda x, y: x or y", as_bool(pv, sdfg, state, a), axis=axis, identity=0)


@oprepo.replaces('numpy.all')
def _all(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    """``np.all``: AND over the operand's truth values.

    The identity of AND is TRUE, and it has to be true in the dtype the accumulator carries.
    Reducing the operand directly made that dtype-dependent -- and with an identity of 0 the very
    first ``x and y`` was false, which pinned every result to False for every input. Reducing the
    BOOL cast instead makes the accumulator bool, so the identity is unambiguous and the answer
    comes back in bool, the way NumPy reports it.
    """
    return reduce(pv, sdfg, state, "lambda x, y: x and y", as_bool(pv, sdfg, state, a), axis=axis, identity=1)


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


@oprepo.replaces('numpy.prod')
def _prod(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return reduce(pv, sdfg, state, "lambda x, y: x * y", a, axis=axis, identity=1)


def reduced_axes(shape, axis) -> list:
    """The axes a reduction over ``axis`` consumes, ``axis=None`` meaning all of them."""
    if axis is None:
        return list(range(len(shape)))
    return normalize_axes(axis if isinstance(axis, (tuple, list)) else [axis], len(shape))


@oprepo.replaces('numpy.var')
def _var(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, ddof: int = 0):
    """``np.var``, as the TWO-PASS mean-then-deviation NumPy computes.

    The one-pass ``E[x^2] - E[x]^2`` form is one reduction cheaper and loses every significant
    digit once the mean dominates the spread, which is the regime a normalization kernel runs in.
    """
    from dace.frontend.python.replacements.array_manipulation import reshape  # Avoid import loop
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop
    from dace.frontend.python.replacements.operators import _array_array_binop  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    shape = list(sdfg.arrays[a].shape)
    axes = reduced_axes(shape, axis)

    mean = nest(_mean)(a, axis=axis)
    # Keep the reduced axes at extent 1 so the deviation broadcasts back over ANY axis, not just
    # the trailing one a right-aligned binop would accept.
    keepdims = [1 if i in axes else extent for i, extent in enumerate(shape)]
    deviation = nest(_array_array_binop)(a, nest(reshape)(mean, keepdims), 'Sub', '-')
    total = nest(_sum)(nest(elementwise)("lambda x: x * x", deviation), axis=axis)
    count = functools.reduce(lambda x, y: x * y, (shape[i] for i in axes))
    return nest, nest(elementwise)("lambda x: x / ({})".format(count - ddof), total)


@oprepo.replaces('numpy.std')
def _std(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, ddof: int = 0):
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    return nest, nest(elementwise)("lambda x: sqrt(x)", nest(_var)(a, axis=axis, ddof=ddof))


@oprepo.replaces('numpy.ptp')
def _ptp(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    from dace.frontend.python.replacements.operators import _array_array_binop  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    return nest, nest(_array_array_binop)(nest(_max)(a, axis=axis), nest(_min)(a, axis=axis), 'Sub', '-')


@oprepo.replaces('numpy.average')
def _average(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, weights=None):
    if weights is not None:
        raise NotImplementedError('numpy.average with weights is not supported; write the weighted sum out')
    return _mean(pv, sdfg, state, a, axis=axis)


@oprepo.replaces('numpy.count_nonzero')
def _count_nonzero(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    from dace.frontend.python.replacements.array_manipulation import _ndarray_astype  # Avoid import loop
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    flags = nest(elementwise)("lambda x: 1 if x != 0 else 0", a)
    # NumPy counts in an integer; the flags carry the operand's dtype, so the cast is the last step.
    return nest, nest(_ndarray_astype)(nest(_sum)(flags, axis=axis), dtypes.int64)


def nan_filled(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, fill: str) -> str:
    """``a`` with every NaN replaced by ``fill``. ``x != x`` is the NaN test that needs no header."""
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop

    if not isinstance(sdfg.arrays[a].dtype,
                      dtypes.typeclass) or sdfg.arrays[a].dtype not in (dtypes.float16, dtypes.float32, dtypes.float64):
        raise NotImplementedError('the nan-aware reductions are supported for floating-point arrays')
    return elementwise(pv, sdfg, state, "lambda x: x if x == x else ({})".format(fill), a)


@oprepo.replaces('numpy.nansum')
def _nansum(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    nest = NestedCall(pv, sdfg, state)
    return nest, nest(_sum)(nest(nan_filled)(a, '0'), axis=axis)


@oprepo.replaces('numpy.nanprod')
def _nanprod(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    nest = NestedCall(pv, sdfg, state)
    return nest, nest(_prod)(nest(nan_filled)(a, '1'), axis=axis)


@oprepo.replaces('numpy.nanmax')
def _nanmax(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    nest = NestedCall(pv, sdfg, state)
    fill = dtypes.min_value(sdfg.arrays[a].dtype)
    return nest, nest(_max)(nest(nan_filled)(a, str(fill)), axis=axis)


@oprepo.replaces('numpy.nanmin')
def _nanmin(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    nest = NestedCall(pv, sdfg, state)
    fill = dtypes.max_value(sdfg.arrays[a].dtype)
    return nest, nest(_min)(nest(nan_filled)(a, str(fill)), axis=axis)


@oprepo.replaces('numpy.nanmean')
def _nanmean(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None):
    """Sum of the non-NaN entries over the count of them -- NOT the plain mean of a filled array,
    which would divide by the NaNs too."""
    from dace.frontend.python.replacements.misc import elementwise  # Avoid import loop
    from dace.frontend.python.replacements.operators import _array_array_binop  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    total = nest(_sum)(nest(nan_filled)(a, '0'), axis=axis)
    present = nest(_sum)(nest(elementwise)("lambda x: 1 if x == x else 0", a), axis=axis)
    return nest, nest(_array_array_binop)(total, present, 'Div', '/')


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
    from dace.frontend.python.replacements.array_manipulation import flat  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)

    assert func in ['min', 'max']

    # Flatten the array if axis is not given
    if axis is None:
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

    reduced_expr = ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis)
    reduced_maprange = {'__i%d' % i: '0:%s' % n for i, n in enumerate(a_arr.shape) if i != axis}
    if not reduced_expr:
        reduced_expr = '0'
        reduced_maprange = {'__i0': '0:1'}

    # Two scalar reductions, no struct. A ``_val_and_idx`` struct needs a Custom WCR, which
    # ``ExpandReduceOpenMP`` refuses (it falls back to the serial expansion), whose combine has to
    # hand-break ties and got it wrong under parallel reduction, and whose member read
    # (``best.idx``) is not a symbolic expression -- propagating it emitted an undeclared name.
    # Reducing the extremum and then the smallest index attaining it uses only builtin WCRs, is
    # first-occurrence by construction whatever order the reduction runs in, and returns scalars.
    extremum, _ = pv.add_temp_transient(reduced_shape, a_arr.dtype)
    # Both reductions need their identity written first: a fresh transient is zero-filled, and
    # ``min(0, ...)`` / ``max(0, ...)`` would clamp every all-positive / all-negative input.
    nest.add_state().add_mapped_tasklet(
        name="_arg{}_value_init_".format(func),
        map_ranges=reduced_maprange,
        inputs={},
        code="__out = {}".format(dtypes.min_value(a_arr.dtype) if func == 'max' else dtypes.max_value(a_arr.dtype)),
        outputs={'__out': Memlet.simple(extremum, reduced_expr)},
        external_edges=True)
    nest.add_state().add_mapped_tasklet(
        name="_arg{}_value_".format(func),
        map_ranges={
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(a_arr.shape)
        },
        inputs={'__in': Memlet.simple(a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))},
        code="__out = __in",
        outputs={'__out': Memlet.simple(extremum, reduced_expr, wcr_str="lambda x, y: {}(x, y)".format(func))},
        external_edges=True)

    # ``result_type``'s largest value is the identity: any real index is smaller, so a min-reduction
    # over the indices attaining the extremum yields the FIRST one, as numpy specifies.
    outidx, _ = pv.add_temp_transient(reduced_shape, result_type, output_index=0 if return_both else None)
    nest.add_state().add_mapped_tasklet(name="_arg{}_index_init_".format(func),
                                        map_ranges=reduced_maprange,
                                        inputs={},
                                        code="__out = {}".format(dtypes.max_value(result_type)),
                                        outputs={'__out': Memlet.simple(outidx, reduced_expr)},
                                        external_edges=True)
    nest.add_state().add_mapped_tasklet(
        name="_arg{}_index_".format(func),
        map_ranges={
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(a_arr.shape)
        },
        inputs={
            '__in': Memlet.simple(a, ','.join('__i%d' % i for i in range(len(a_arr.shape)))),
            '__best': Memlet.simple(extremum, reduced_expr)
        },
        code="__out = {} if __in == __best else {}".format('__i%d' % axis, dtypes.max_value(result_type)),
        outputs={'__out': Memlet.simple(outidx, reduced_expr, wcr_str="lambda x, y: min(x, y)")},
        external_edges=True)

    if return_both:
        outval, _ = pv.add_temp_transient(reduced_shape, a_arr.dtype, output_index=1)
        nest.add_state().add_mapped_tasklet(name="_arg{}_value_out_".format(func),
                                            map_ranges=reduced_maprange,
                                            inputs={'__in': Memlet.simple(extremum, reduced_expr)},
                                            code="__out = __in",
                                            outputs={'__out': Memlet.simple(outval, reduced_expr)},
                                            external_edges=True)
        return nest, (outval, outidx)
    return nest, outidx


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


#: numpy's dtype rule for the cumulative functions: an integer input accumulates in the widest
#: integer of its OWN signedness -- bool and every signed width in int64, every unsigned width in
#: uint64 -- and every other dtype keeps itself. Not the elementwise promotion rule, which is why
#: it is spelled out here rather than deferred to ``result_type``.
def cumulative_dtype(dtype: dtypes.typeclass) -> dtypes.typeclass:
    """Result dtype of ``numpy.cumsum``/``numpy.cumprod`` over ``dtype``."""
    kind = dtype.as_numpy_dtype().kind
    if kind == 'u':
        return dtypes.uint64
    if kind in ('b', 'i'):
        return dtypes.int64
    return dtype


def cumulative(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis, dtype, op, funcname: str) -> str:
    """``numpy.cumsum``/``numpy.cumprod`` lowered onto the ``Scan`` library node.

    A prefix scan is NOT a reduction: every partial result stays visible, so ``Reduce`` cannot
    express it. The libnode already carries the implementations that matter -- the blocked OpenMP
    header on CPU, ``gpucub::DeviceScan`` on GPU, a portable loop -- so the frontend's whole job is to
    name the op and wire the memlets. Lowering the recurrence into an explicit loop here would hand
    every backend the shape only one of them wants.

    Only the LAST axis is lowered, which is the one the operand's layout makes contiguous. A scan
    along an inner axis is a strided chain per outer index; the libnode's ``stride`` property
    expresses one such chain, not a batch of them. Over a rank > 1 operand the lowering is a Map
    across the leading axes with the scan inside, which is also where the parallelism is -- the
    recurrence itself is sequential.
    """
    from dace.libraries.standard.nodes.scan import Scan  # Avoid import loop
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise SyntaxError(f'{funcname} expects an array operand, got {a}')
    desc = sdfg.arrays[a]
    if not isinstance(desc, data.Array):
        raise SyntaxError(f'{funcname} expects an array operand, got a {type(desc).__name__}')
    rank = len(desc.shape)
    if axis is None:
        # numpy FLATTENS an axis-less cumulative over a rank > 1 operand. Refusing is deliberate:
        # the flatten is a reshape only when the operand is contiguous, and quietly scanning the
        # last axis instead would return the right SHAPE holding the wrong numbers.
        if rank != 1:
            raise NotImplementedError(f'{funcname} without an axis flattens a {rank}-D operand; '
                                      f'pass axis={rank - 1} to scan the last axis instead')
        axis = 0
    else:
        axis = normalize_axes((axis, ), rank)[0]
    if axis != rank - 1:
        raise NotImplementedError(f'{funcname} is lowered along the last axis only; axis={axis} of a '
                                  f'{rank}-D operand is a strided chain per outer index')
    if dtype is None:
        out_dtype = cumulative_dtype(desc.dtype)
    elif isinstance(dtype, dtypes.typeclass):
        out_dtype = dtype
    else:
        out_dtype = dtypes.typeclass(numpy.dtype(dtype).type)

    indices = ', '.join(f'__i{d}' for d in range(rank))
    source, read = a, None
    if out_dtype != desc.dtype:
        # The libnode scans in ONE dtype and numpy's rule can widen it, so cast first: the
        # accumulation then happens at the output width, which is what numpy does. The cast's WRITE
        # node is reused as the scan's read node -- a second access node for the same array would
        # leave the two unordered inside one state, which is a race, not a copy.
        source, _ = pv.add_temp_transient(desc.shape, out_dtype, storage=desc.storage)
        _tasklet, _entry, cast_exit = state.add_mapped_tasklet(f'{funcname}_cast', {
            f'__i{d}': f'0:{s}'
            for d, s in enumerate(desc.shape)
        }, {'__inp': Memlet.simple(a, indices)},
                                                               '__out = __inp',
                                                               {'__out': Memlet.simple(source, indices)},
                                                               external_edges=True)
        read = next(e.dst for e in state.out_edges(cast_exit))
    if read is None:
        read = state.add_read(source)

    out, _out_desc = pv.add_temp_transient(desc.shape, out_dtype, storage=desc.storage)
    scan = Scan(funcname, op=op)
    length = desc.shape[axis]
    write = state.add_write(out)
    if rank == 1:
        state.add_edge(read, None, scan, Scan.INPUT_CONNECTOR_NAME, Memlet.simple(source, f'0:{length}'))
        state.add_edge(scan, Scan.OUTPUT_CONNECTOR_NAME, write, None, Memlet.simple(out, f'0:{length}'))
        return out
    outer = {f'__s{d}': f'0:{s}' for d, s in enumerate(desc.shape[:-1])}
    entry, exit_node = state.add_map(f'{funcname}_batch', outer)
    row = ', '.join([*outer, f'0:{length}'])
    state.add_memlet_path(read, entry, scan, dst_conn=Scan.INPUT_CONNECTOR_NAME, memlet=Memlet.simple(source, row))
    state.add_memlet_path(scan, exit_node, write, src_conn=Scan.OUTPUT_CONNECTOR_NAME, memlet=Memlet.simple(out, row))
    return out


@oprepo.replaces('dace.cumsum')
@oprepo.replaces('numpy.cumsum')
@oprepo.replaces('numpy.cumulative_sum')
def _cumsum(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, dtype=None, out=None):
    from dace.libraries.standard.nodes.scan import ScanOp  # Avoid import loop
    if out is not None:
        raise NotImplementedError('numpy.cumsum(out=...) is not supported; assign the result instead')
    return cumulative(pv, sdfg, state, a, axis, dtype, ScanOp.SUM, 'cumsum')


@oprepo.replaces('dace.cumprod')
@oprepo.replaces('numpy.cumprod')
@oprepo.replaces('numpy.cumulative_prod')
def _cumprod(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, axis=None, dtype=None, out=None):
    from dace.libraries.standard.nodes.scan import ScanOp  # Avoid import loop
    if out is not None:
        raise NotImplementedError('numpy.cumprod(out=...) is not supported; assign the result instead')
    return cumulative(pv, sdfg, state, a, axis, dtype, ScanOp.PRODUCT, 'cumprod')
