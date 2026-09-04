# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for array-filling methods (zeros, ones, etc.)
"""
import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, broadcast_together, step_state, sym_type
from dace.frontend.python.replacements.array_creation_dace import promote_size_scalars_in_shape
from dace.frontend.python.replacements.operators import result_type
from dace.sdfg.type_inference import infer_expr_type
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

import ast
import copy
import math
from numbers import Number, Integral
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import sympy as sp


@oprepo.replaces('numpy.copy')
def _numpy_copy(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str):
    """ Creates a copy of array a.
    """
    if a not in sdfg.arrays.keys():
        raise DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=a))
    sample = sdfg.arrays[a]
    if isinstance(sample, data.Array) and isinstance(sample, data.View):
        # A slice (e.g. path[:, 1]) is an ArrayView, a concrete subclass that the
        # generic transient dispatch below does not recognize (it keys on exact
        # type). The view's own shape is already the sliced shape, so materialize
        # the copy directly as a plain array of that shape and dtype.
        name, desc = sdfg.add_transient(pv.get_target_name(), sample.shape, sample.dtype, find_new_name=True)
    else:
        # TODO: The whole AddTransientMethod class should be move in replacements.py
        from dace.frontend.python.newast import _add_transient_data
        name, desc = _add_transient_data(pv, sdfg, sample)
    rnode = state.add_read(a)
    wnode = state.add_write(name)
    state.add_nedge(rnode, wnode, Memlet.from_array(name, desc))
    return name


@oprepo.replaces_method('Array', 'copy')
@oprepo.replaces_method('Scalar', 'copy')
@oprepo.replaces_method('View', 'copy')
def _ndarray_copy(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _numpy_copy(pv, sdfg, state, arr)


@oprepo.replaces('numpy.full')
def _numpy_full(pv: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                shape: Shape,
                fill_value: Union[sp.Expr, Number, data.Scalar],
                dtype: dtypes.typeclass = None):
    """ Creates and array of the specified shape and initializes it with
        the fill value.
    """
    if isinstance(shape, Number) or symbolic.issymbolic(shape):
        shape = [shape]
    is_data = False
    if isinstance(fill_value, (Number, np.bool_)):
        vtype = dtypes.dtype_to_typeclass(type(fill_value))
    elif isinstance(fill_value, sp.Expr):
        vtype = sym_type(fill_value)
    else:
        is_data = True
        vtype = sdfg.arrays[fill_value].dtype
    dtype = dtype or vtype

    # Handle one-dimensional inputs
    if isinstance(shape, (Number, str)) or symbolic.issymbolic(shape):
        shape = [shape]

    shape, promoted = promote_size_scalars_in_shape(pv, sdfg, shape)
    if promoted:
        # Promotion opens a state to carry the symbol assignment; the fill has to follow it.
        state = pv.last_block
    if any(isinstance(s, str) for s in shape):
        raise DaceSyntaxError(
            pv, None, f'Data-dependent shape {shape} is currently not allowed. Only constants '
            'and symbolic values can be used.')

    name = pv.get_target_name()
    name, _ = sdfg.add_transient(name, shape, dtype, find_new_name=True)

    if is_data:
        state.add_mapped_tasklet('_numpy_full_', {
            "__i{}".format(i): "0: {}".format(s)
            for i, s in enumerate(shape)
        },
                                 dict(__inp=Memlet(data=fill_value, subset='0')),
                                 "__out = __inp",
                                 dict(__out=Memlet.simple(name, ",".join(["__i{}".format(i)
                                                                          for i in range(len(shape))]))),
                                 external_edges=True)
    else:
        state.add_mapped_tasklet('_numpy_full_', {
            "__i{}".format(i): "0: {}".format(s)
            for i, s in enumerate(shape)
        }, {},
                                 "__out = {}".format(fill_value),
                                 dict(__out=Memlet.simple(name, ",".join(["__i{}".format(i)
                                                                          for i in range(len(shape))]))),
                                 external_edges=True)

    return name


@oprepo.replaces('numpy.full_like')
def _numpy_full_like(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     a: str,
                     fill_value: Number,
                     dtype: dtypes.typeclass = None,
                     shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with the fill value.
    """
    if a not in sdfg.arrays:
        raise DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=a))
    desc = sdfg.arrays[a]
    dtype = dtype or desc.dtype
    shape = shape or desc.shape
    return _numpy_full(pv, sdfg, state, shape, fill_value, dtype)


@oprepo.replaces_method('Array', 'fill')
@oprepo.replaces_method('Scalar', 'fill')
@oprepo.replaces_method('View', 'fill')
def _ndarray_fill(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, value: Union[str, Number,
                                                                                           sp.Expr]) -> str:
    assert arr in sdfg.arrays

    if isinstance(value, sp.Expr):
        raise NotImplementedError(
            f"{arr}.fill is not implemented for symbolic expressions ({value}).")  # Look at `full`.

    if isinstance(value, (Number, np.bool_)):
        body = value
        inputs = {}
    elif isinstance(value, str) and value in sdfg.arrays:
        value_array = sdfg.arrays[value]
        if not isinstance(value_array, data.Scalar):
            raise DaceSyntaxError(pv, None,
                                  f"{arr}.fill requires a scalar argument, but {type(value_array)} was given.")
        body = '__inp'
        inputs = {'__inp': Memlet(data=value, subset='0')}
    else:
        raise DaceSyntaxError(pv, None, f"Unsupported argument '{value}' for {arr}.fill.")

    shape = sdfg.arrays[arr].shape
    state.add_mapped_tasklet(
        '_numpy_fill_',
        map_ranges={
            f"__i{dim}": f"0:{s}"
            for dim, s in enumerate(shape)
        },
        inputs=inputs,
        code=f"__out = {body}",
        outputs={'__out': Memlet.simple(arr, ",".join([f"__i{dim}" for dim in range(len(shape))]))},
        external_edges=True)

    return arr


@oprepo.replaces('numpy.ones')
def _numpy_ones(pv: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                shape: Shape,
                dtype: dtypes.typeclass = dtypes.float64):
    """ Creates and array of the specified shape and initializes it with ones.
    """
    return _numpy_full(pv, sdfg, state, shape, 1.0, dtype)


@oprepo.replaces('numpy.ones_like')
def _numpy_ones_like(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     a: str,
                     dtype: dtypes.typeclass = None,
                     shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with ones.
    """
    return _numpy_full_like(pv, sdfg, state, a, 1.0, dtype, shape)


@oprepo.replaces('numpy.zeros')
def _numpy_zeros(pv: ProgramVisitor,
                 sdfg: SDFG,
                 state: SDFGState,
                 shape: Shape,
                 dtype: dtypes.typeclass = dtypes.float64):
    """ Creates and array of the specified shape and initializes it with zeros.
    """
    return _numpy_full(pv, sdfg, state, shape, 0.0, dtype)


@oprepo.replaces('numpy.zeros_like')
def _numpy_zeros_like(pv: ProgramVisitor,
                      sdfg: SDFG,
                      state: SDFGState,
                      a: str,
                      dtype: dtypes.typeclass = None,
                      shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with zeros.
    """
    return _numpy_full_like(pv, sdfg, state, a, 0.0, dtype, shape)


@oprepo.replaces('numpy.eye')
def eye(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, N, M=None, k=0, dtype=dtypes.float64):
    M = M or N
    name = pv.get_target_name()
    name, _ = sdfg.add_transient(name, [N, M], dtype, find_new_name=True)

    state.add_mapped_tasklet('eye',
                             dict(__i0='0:%s' % N, __i1='0:%s' % M), {},
                             'val = 1 if __i0 == (__i1 - %s) else 0' % k,
                             dict(val=Memlet.simple(name, '__i0, __i1')),
                             external_edges=True)

    return name


@oprepo.replaces('numpy.identity')
def _numpy_identity(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, n, dtype=dtypes.float64):
    """ Generates the nxn identity matrix. """
    return eye(pv, sdfg, state, n, dtype=dtype)


@oprepo.replaces('numpy.arange')
@oprepo.replaces('dace.arange')
def _arange(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            *args,
            dtype: dtypes.typeclass = None,
            like: Optional[str] = None):
    """ Implementes numpy.arange """

    start = 0
    stop = None
    step = 1
    if len(args) == 1:
        stop = args[0]
        if isinstance(stop, Number):
            start = type(stop)(0)
    elif len(args) == 2:
        start, stop = args
    else:
        start, stop, step = args

    # A string bound is a name, not a value. An SDFG symbol names a symbolic extent directly; a size-1
    # container (``K = nclusters; np.arange(K)``) is read into a symbol on an interstate edge, the same
    # mechanism ``numpy.full`` uses to size its output. Any other data would take the extent from array
    # contents, which cannot size the output.
    for kind, value in (('start', start), ('stop', stop), ('step', step)):
        if not isinstance(value, str):
            continue
        if value not in sdfg.symbols and not (value in sdfg.arrays and sdfg.arrays[value].total_size == 1):
            raise TypeError(f'Cannot compile numpy.arange with a scalar {kind} value "{value}" (only constants and '
                            'symbolic expressions are supported). Please use numpy.linspace instead.')

    (start, stop, step), promoted = promote_size_scalars_in_shape(pv, sdfg, (start, stop, step))
    if promoted:
        # Promotion opens a state to carry the symbol assignment; the map has to follow it.
        state = pv.last_block
    start, stop, step = [symbolic.pystr_to_symbolic(v) if isinstance(v, str) else v for v in (start, stop, step)]
    # Type inference below reads the call arguments, which have no case for a name.
    args = (stop, ) if len(args) == 1 else (start, stop, step)[:len(args)]

    actual_step = step
    if isinstance(start, Number) and isinstance(stop, Number):
        actual_step = type(start + step)(start + step) - start

    if any(not isinstance(s, Number) for s in [start, stop, step]):
        if step == 1:  # Common case where ceiling is not necessary
            shape = (stop - start, )
        else:
            shape = (symbolic.int_ceil(stop - start, step), )
    else:
        shape = (np.int64(np.ceil((stop - start) / step)), )

    # Infer dtype from input arguments
    if dtype is None:
        dtype, _ = result_type(args)

    # TODO: Unclear what 'like' does
    # if 'like' is not None:
    #     outname, outarr = sdfg.add_temp_transient_like(sdfg.arrays[like], name=pv.get_target_name())
    #     outarr.shape = shape

    if not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.dtype_to_typeclass(dtype)
    outname = pv.get_target_name()
    outname, outarr = sdfg.add_transient(outname, shape, dtype, find_new_name=True)

    start = f'{dtype.ctype}({start})'
    actual_step = f'{dtype.ctype}({actual_step})'

    state.add_mapped_tasklet(name="_numpy_arange_",
                             map_ranges={'__i': f"0:{shape[0]}"},
                             inputs={},
                             code=f"__out = {start} + __i * {actual_step}",
                             outputs={'__out': Memlet(f"{outname}[__i]")},
                             external_edges=True)

    return outname


def _add_axis_to_shape(shape: Sequence[symbolic.SymbolicType], axis: int,
                       axis_value: Any) -> List[symbolic.SymbolicType]:
    if axis > len(shape):
        raise ValueError(f'axis {axis} is out of bounds for array of dimension {len(shape)}')
    if axis < 0:
        naxis = len(shape) + 1 + axis
        if naxis < 0 or naxis > len(shape):
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {len(shape)}')
        axis = naxis

    # Make a new shape list with the inserted dimension
    new_shape = [None] * (len(shape) + 1)
    for i in range(len(shape) + 1):
        if i == axis:
            new_shape[i] = axis_value
        elif i < axis:
            new_shape[i] = shape[i]
        else:
            new_shape[i] = shape[i - 1]

    return new_shape


@oprepo.replaces('numpy.linspace')
def _linspace(pv: ProgramVisitor,
              sdfg: SDFG,
              state: SDFGState,
              start: Union[Number, symbolic.SymbolicType, str],
              stop: Union[Number, symbolic.SymbolicType, str],
              num: Union[Integral, symbolic.SymbolicType] = 50,
              endpoint: bool = True,
              retstep: bool = False,
              dtype: dtypes.typeclass = None,
              axis: int = 0):
    """ Implements numpy.linspace """
    # Argument checks
    if not isinstance(num, (Integral, sp.Basic)):
        raise TypeError('numpy.linspace can only be compiled when the ``num`` argument is symbolic or constant.')
    if not isinstance(axis, Integral):
        raise TypeError('numpy.linspace can only be compiled when the ``axis`` argument is constant.')

    # Start and stop are broadcast together, then, a new dimension is added to axis (taken from ``ndim + 1``),
    # along which the numbers are filled.
    def _endpoint_shape(x):
        # numpy.linspace of 0-d (scalar) endpoints is 1-D of length ``num``. A DaCe Scalar reports shape
        # (1,), which would otherwise add a spurious trailing axis -- making the result (num, 1), a column
        # that then mis-broadcasts against other 1-D arrays. Treat a true scalar endpoint as contributing
        # no dimensions; a genuine size-1 *array* endpoint keeps its (1,) shape (numpy also returns a column).
        if isinstance(x, str) and x in sdfg.arrays and not isinstance(sdfg.arrays[x], data.Scalar):
            return sdfg.arrays[x].shape
        return []

    start_shape = _endpoint_shape(start)
    stop_shape = _endpoint_shape(stop)

    shape, ranges, outind, ind1, ind2 = broadcast_together(start_shape, stop_shape)
    shape_with_axis = _add_axis_to_shape(shape, axis, num)
    ranges_with_axis = _add_axis_to_shape(ranges, axis, ('__sind', f'0:{symbolic.symstr(num)}'))
    if outind:
        outind_with_axis = _add_axis_to_shape(outind.split(', '), axis, '__sind')
    else:
        outind_with_axis = ['__sind']

    if dtype is None:
        # Infer output type from start and stop
        start_type = sdfg.arrays[start] if (isinstance(start, str) and start in sdfg.arrays) else start
        stop_type = sdfg.arrays[stop] if (isinstance(stop, str) and stop in sdfg.arrays) else stop

        dtype, _ = result_type((start_type, stop_type), 'Add')

        # From the NumPy documentation: The inferred dtype will never be an integer; float is chosen even if the
        # arguments would produce an array of integers.
        if dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32,
                     dtypes.uint64):
            dtype = dtypes.dtype_to_typeclass(float)
    elif not isinstance(dtype, dtypes.typeclass):
        # An explicitly passed dtype arrives as a numpy type; the tasklet body below needs ``ctype``.
        dtype = dtypes.dtype_to_typeclass(dtype)

    outname = pv.get_target_name()
    outname, _ = sdfg.add_transient(outname, shape_with_axis, dtype, find_new_name=True)

    if endpoint == True:
        num -= 1

    # Fill in input memlets as necessary
    def _endpoint_index(name, ind):
        # A broadcast index if the endpoint carries one; else a scalar endpoint (no broadcast axis) must
        # still read its single element explicitly ([0]) so memlet propagation sees a concrete subset
        # rather than a bare, subset-less scalar memlet (which trips propagation with a None range).
        if ind:
            return f'[{ind}]'
        return '[0]' if isinstance(sdfg.arrays[name], data.Scalar) else ''

    inputs = {}
    if isinstance(start, str) and start in sdfg.arrays:
        inputs['__start'] = Memlet(f'{start}{_endpoint_index(start, ind1)}')
        startcode = '__start'
    else:
        startcode = symbolic.symstr(start)

    if isinstance(stop, str) and stop in sdfg.arrays:
        inputs['__stop'] = Memlet(f'{stop}{_endpoint_index(stop, ind2)}')
        stopcode = '__stop'
    else:
        stopcode = symbolic.symstr(stop)

    # Create tasklet code based on inputs
    # Compute the step first (``i * (delta/num)``), matching numpy.linspace's ``start + arange*step`` order.
    # Multiplying before dividing (``i*delta/num``) is a different floating-point rounding and is not
    # bit-exact with numpy.
    code = f'__out = {startcode} + __sind * ({dtype.ctype}({stopcode} - {startcode}) / {dtype.ctype}({symbolic.symstr(num)}))'

    state.add_mapped_tasklet(name="linspace",
                             map_ranges=ranges_with_axis,
                             inputs=inputs,
                             code=code,
                             outputs={'__out': Memlet(f"{outname}[{','.join(outind_with_axis)}]")},
                             external_edges=True)

    if retstep == False:
        return outname

    # Return step if requested

    # Handle scalar outputs
    if not ranges:
        ranges = [('__unused', '0:1')]
    out_index = f'[{outind}]'

    stepname = pv.get_target_name() + "_step"
    if len(shape) > 0:
        stepname, _ = sdfg.add_transient(stepname, shape, dtype, find_new_name=True)
    else:
        stepname, _ = sdfg.add_scalar(stepname, dtype, transient=True, find_new_name=True)
        out_index = '[0]'

    state.add_mapped_tasklet('retstep',
                             ranges,
                             copy.deepcopy(inputs),
                             f'__out = {dtype.ctype}({stopcode} - {startcode}) / {dtype.ctype}({symbolic.symstr(num)})',
                             {'__out': Memlet(f"{stepname}{out_index}")},
                             external_edges=True)

    return outname, stepname


@oprepo.replaces('numpy.logspace')
def logspace(pv: ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             start: Union[Number, symbolic.SymbolicType, str],
             stop: Union[Number, symbolic.SymbolicType, str],
             num: Union[Integral, symbolic.SymbolicType] = 50,
             endpoint: bool = True,
             base: Union[Number, symbolic.SymbolicType] = 10.0,
             dtype: dtypes.typeclass | None = None,
             axis: int = 0) -> str:
    """``np.logspace`` is ``base ** linspace(...)``; the exponents come from the native :func:`_linspace`."""
    if isinstance(base, str):
        raise ValueError(f'numpy.logspace needs a constant or symbolic ``base``, not the array "{base}"')

    exponents = _linspace(pv, sdfg, state, start, stop, num, endpoint, False, None, axis)
    expdesc = sdfg.arrays[exponents]
    if dtype is None:
        basetype = sym_type(base) if symbolic.issymbolic(base) else dtypes.dtype_to_typeclass(type(base))
        dtype = dtypes.result_type_of(expdesc.dtype, basetype)
    elif not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.dtype_to_typeclass(dtype)

    outname, _ = sdfg.add_transient(pv.get_target_name(), expdesc.shape, dtype, find_new_name=True)
    index = ', '.join(f'__i{d}' for d in range(len(expdesc.shape)))
    state = step_state(pv, state)
    state.add_mapped_tasklet(name='logspace',
                             map_ranges={
                                 f'__i{d}': f'0:{symbolic.symstr(s)}'
                                 for d, s in enumerate(expdesc.shape)
                             },
                             inputs={'__exp': Memlet(f'{exponents}[{index}]')},
                             code=f'__out = {dtype.ctype}({symbolic.symstr(base)}) ** __exp',
                             outputs={'__out': Memlet(f'{outname}[{index}]')},
                             external_edges=True)
    return outname


@oprepo.replaces('numpy.geomspace')
def geomspace(pv: ProgramVisitor,
              sdfg: SDFG,
              state: SDFGState,
              start: Number,
              stop: Number,
              num: Integral = 50,
              endpoint: bool = True,
              dtype: dtypes.typeclass | None = None,
              axis: int = 0) -> str:
    """``np.geomspace`` is a base-10 :func:`logspace` between ``log10`` endpoints, both ends pinned.

    The endpoints have to be constants because their logarithms become the linspace bounds, which are
    formed at parse time.
    """
    if not isinstance(start, Number) or not isinstance(stop, Number):
        raise ValueError('numpy.geomspace needs compile-time constant ``start`` and ``stop`` values, since their '
                         'logarithms become the linspace bounds')
    if start == 0 or stop == 0:
        raise ValueError('numpy.geomspace: ``start`` and ``stop`` must be nonzero')
    if (start < 0) != (stop < 0):
        raise ValueError('numpy.geomspace: ``start`` and ``stop`` must have the same sign')
    if not isinstance(num, Integral):
        raise ValueError('numpy.geomspace needs a compile-time constant ``num``')
    if not isinstance(axis, Integral) or int(axis) not in (0, -1):
        raise ValueError('numpy.geomspace of scalar endpoints is one-dimensional, so ``axis`` must be 0')

    sign = -1 if start < 0 else 1
    lo, hi = sign * start, sign * stop
    exponents = _linspace(pv, sdfg, state, math.log10(lo), math.log10(hi), num, endpoint, False, None, 0)
    if dtype is None:
        dtype = sdfg.arrays[exponents].dtype
    elif not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.dtype_to_typeclass(dtype)

    # numpy restores the exact arguments at both ends; the round trip through log10 does not.
    body = f'{dtype.ctype}(10.0) ** __exp'
    if endpoint and num > 1:
        body = f'({dtype.ctype}({hi}) if __i0 == {int(num) - 1} else ({body}))'
    if num > 0:
        body = f'({dtype.ctype}({lo}) if __i0 == 0 else {body})'
    if sign < 0:
        body = f'-({body})'

    outname, _ = sdfg.add_transient(pv.get_target_name(), [num], dtype, find_new_name=True)
    state = step_state(pv, state)
    state.add_mapped_tasklet(name='geomspace',
                             map_ranges={'__i0': f'0:{symbolic.symstr(num)}'},
                             inputs={'__exp': Memlet(f'{exponents}[__i0]')},
                             code=f'__out = {body}',
                             outputs={'__out': Memlet(f'{outname}[__i0]')},
                             external_edges=True)
    return outname


def index_dtype(dtype: Any) -> dtypes.typeclass:
    """Index arrays default to int64 and are never narrowed on our own initiative."""
    if dtype is None:
        return dtypes.int64
    return dtype if isinstance(dtype, dtypes.typeclass) else dtypes.dtype_to_typeclass(dtype)


def static_extents(name: str, shape: Shape) -> list[Any]:
    """Extents that a descriptor can carry: constants and symbols, never array contents."""
    if isinstance(shape, (Number, str)) or symbolic.issymbolic(shape):
        shape = [shape]
    extents = list(shape)
    for extent in extents:
        if isinstance(extent, str):
            raise ValueError(f'{name} cannot use the data-dependent extent "{extent}"; only constants and '
                             'symbolic values can size the result')
    return extents


@oprepo.replaces('numpy.indices')
def indices(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            dimensions: Shape,
            dtype: dtypes.typeclass | None = None,
            sparse: bool = False) -> str | list[str]:
    """``np.indices``: every value is the map index of its own axis, so one map per axis fills the grid."""
    dims = static_extents('numpy.indices', dimensions)
    itype = index_dtype(dtype)
    ndim = len(dims)

    if sparse:
        # The sparse form keeps every axis (length 1 off the own axis), so it broadcasts like the dense one.
        outnames = []
        for axis in range(ndim):
            shape = [1] * ndim
            shape[axis] = dims[axis]
            name, _ = sdfg.add_transient(pv.get_target_name(output_index=axis), shape, itype, find_new_name=True)
            index = ', '.join(f'__i{axis}' if d == axis else '0' for d in range(ndim))
            state.add_mapped_tasklet(name=f'indices_{axis}',
                                     map_ranges={f'__i{axis}': f'0:{symbolic.symstr(dims[axis])}'},
                                     inputs={},
                                     code=f'__out = __i{axis}',
                                     outputs={'__out': Memlet(f'{name}[{index}]')},
                                     external_edges=True)
            outnames.append(name)
        return outnames

    outname, _ = sdfg.add_transient(pv.get_target_name(), [ndim] + dims, itype, find_new_name=True)
    index = ', '.join(f'__i{d}' for d in range(ndim))
    for axis in range(ndim):
        state.add_mapped_tasklet(name=f'indices_{axis}',
                                 map_ranges={
                                     f'__i{d}': f'0:{symbolic.symstr(s)}'
                                     for d, s in enumerate(dims)
                                 },
                                 inputs={},
                                 code=f'__out = __i{axis}',
                                 outputs={'__out': Memlet(f'{outname}[{axis}, {index}]')},
                                 external_edges=True)
    return outname


@oprepo.replaces('numpy.ix_')
def ix_(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args: str) -> list[str]:
    """``np.ix_``: the open mesh is the same data seen through one axis, i.e. a reshape per argument.

    A boolean argument is refused: numpy turns it into its nonzero positions, whose length is a
    property of the values rather than of the descriptor.
    """
    from dace.frontend.python.replacements.array_manipulation import reshape  # Avoid import loop

    ndim = len(args)
    if ndim == 0:
        raise ValueError('numpy.ix_ needs at least one index array')

    outnames = []
    for axis, arg in enumerate(args):
        if isinstance(arg, (list, tuple)) and len(arg) == 1:
            arg = arg[0]
        if not isinstance(arg, str) or arg not in sdfg.arrays:
            raise ValueError('numpy.ix_ expects one-dimensional index arrays')
        desc = sdfg.arrays[arg]
        if desc.dtype in (dtypes.bool, dtypes.bool_):
            raise ValueError('numpy.ix_ with a boolean mask is unsupported: the result extent would be '
                             'data-dependent')
        if isinstance(desc, data.Scalar) or len(desc.shape) != 1:
            raise ValueError('numpy.ix_: cross index must be 1 dimensional')
        newshape = [1] * ndim
        newshape[axis] = desc.shape[0]
        outnames.append(reshape(pv, sdfg, state, arg, newshape))
    return outnames


@oprepo.replaces('numpy.ravel_multi_index')
def ravel_multi_index(pv: ProgramVisitor,
                      sdfg: SDFG,
                      state: SDFGState,
                      multi_index: Sequence[Any],
                      dims: Shape,
                      mode: StringLiteral | str = 'raise',
                      order: StringLiteral | str = 'C') -> str:
    """``np.ravel_multi_index`` is the dot product of the indices with the strides of ``dims``.

    ``mode='raise'`` is computed unchecked: a generated kernel has no way to raise, and every mode
    agrees on in-range indices.
    """
    if not isinstance(multi_index, (list, tuple)):
        raise ValueError('numpy.ravel_multi_index expects a sequence of index arrays, one per dimension')
    extents = static_extents('numpy.ravel_multi_index', dims)
    if len(multi_index) != len(extents):
        raise ValueError(f'numpy.ravel_multi_index: {len(multi_index)} indices given for {len(extents)} dimensions')
    mode, order = str(mode), str(order)
    if mode not in ('raise', 'wrap', 'clip'):
        raise ValueError(f'numpy.ravel_multi_index: unknown mode "{mode}"')
    if order not in ('C', 'F'):
        raise ValueError(f'numpy.ravel_multi_index: unknown order "{order}"')

    ndim = len(extents)
    if order == 'C':
        strides = [data._prod(extents[d + 1:]) for d in range(ndim)]
    else:
        strides = [data._prod(extents[:d]) for d in range(ndim)]

    # Right-aligned broadcast of the index operands, as numpy does before the dot product.
    shapes = []
    for index in multi_index:
        if isinstance(index, str) and index in sdfg.arrays:
            desc = sdfg.arrays[index]
            shapes.append([] if isinstance(desc, data.Scalar) else list(desc.shape))
        elif isinstance(index, Number) or symbolic.issymbolic(index):
            shapes.append([])
        else:
            raise ValueError(f'numpy.ravel_multi_index cannot use "{index}" as an index')
    rank = max((len(s) for s in shapes), default=0)
    outshape = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for d, extent in enumerate(shape):
            if not symbolic.inequal_symbols(outshape[offset + d], extent):
                continue
            if outshape[offset + d] == 1:
                outshape[offset + d] = extent
            elif extent != 1:
                raise ValueError('numpy.ravel_multi_index: the index shapes cannot be broadcast together')

    inputs = {}
    terms = []
    for pos, index in enumerate(multi_index):
        if isinstance(index, str):
            shape = shapes[pos]
            offset = rank - len(shape)
            subset = ', '.join('0' if shape[d] == 1 else f'__i{offset + d}' for d in range(len(shape))) or '0'
            inputs[f'__in{pos}'] = Memlet(f'{index}[{subset}]')
            value = f'__in{pos}'
        else:
            value = f'({symbolic.symstr(index)})'
        extent = symbolic.symstr(extents[pos])
        if mode == 'wrap':
            value = f'(({value}) % ({extent}))'
        elif mode == 'clip':
            last = symbolic.symstr(symbolic.pystr_to_symbolic(f'({extent}) - 1'))
            value = f'(0 if ({value}) < 0 else (({last}) if ({value}) > ({last}) else ({value})))'
        stride = symbolic.symstr(strides[pos])
        terms.append(f'({value})' if strides[pos] == 1 else f'({value}) * ({stride})')

    if rank == 0:
        outname, _ = sdfg.add_scalar(pv.get_target_name(), dtypes.int64, transient=True, find_new_name=True)
        ranges, outsubset = {'__i0': '0:1'}, '0'
    else:
        outname, _ = sdfg.add_transient(pv.get_target_name(), outshape, dtypes.int64, find_new_name=True)
        ranges = {f'__i{d}': f'0:{symbolic.symstr(s)}' for d, s in enumerate(outshape)}
        outsubset = ', '.join(f'__i{d}' for d in range(rank))

    state.add_mapped_tasklet(name='ravel_multi_index',
                             map_ranges=ranges,
                             inputs=inputs,
                             code='__out = ' + ' + '.join(terms),
                             outputs={'__out': Memlet(f'{outname}[{outsubset}]')},
                             external_edges=True)
    return outname


@oprepo.replaces('numpy.triu_indices')
def triu_indices(pv: ProgramVisitor,
                 sdfg: SDFG,
                 state: SDFGState,
                 n: Integral,
                 k: Integral = 0,
                 m: Integral | None = None) -> list[str]:
    """``np.triu_indices`` in row-major order, as at most two maps over the kept rectangle and trapezoid.

    Rows below ``-k`` keep every column, the rest start at ``i + k``; each row's flat offset is then a
    closed form of the row index, so no compaction pass is needed. ``n``, ``k`` and ``m`` have to be
    constants because the number of kept pairs is otherwise not a static extent.
    """
    m = n if m is None else m
    if not all(isinstance(v, Integral) for v in (n, k, m)):
        raise ValueError('numpy.triu_indices needs compile-time constant ``n``, ``k`` and ``m``: the number of '
                         'upper-triangle indices is otherwise not a static extent')
    n, k, m = int(n), int(k), int(m)
    if n < 0 or m < 0:
        raise ValueError('numpy.triu_indices: ``n`` and ``m`` must be non-negative')

    full = min(max(-k, 0), n)  # rows entirely inside the triangle
    last = min(max(m - k, 0), n)  # first row with nothing left of it
    count = full * m + sum(m - i - k for i in range(full, last))

    rows, _ = sdfg.add_transient(pv.get_target_name(output_index=0), [count], dtypes.int64, find_new_name=True)
    cols, _ = sdfg.add_transient(pv.get_target_name(output_index=1), [count], dtypes.int64, find_new_name=True)

    def emit(label: str, ranges: dict[str, str], offset: symbolic.SymbolicType) -> None:
        subset = symbolic.symstr(offset)
        for name, code in ((rows, '__out = __i'), (cols, '__out = __j')):
            state.add_mapped_tasklet(name=f'triu_indices_{label}_{name}',
                                     map_ranges=dict(ranges),
                                     inputs={},
                                     code=code,
                                     outputs={'__out': Memlet(f'{name}[{subset}]')},
                                     external_edges=True)

    i, j = symbolic.pystr_to_symbolic('__i'), symbolic.pystr_to_symbolic('__j')
    if full > 0:
        emit('full', {'__i': f'0:{full}', '__j': f'0:{m}'}, i * m + j)
    if last > full:
        # Row ``i`` starts at column ``i + k`` and the rows before it have lost 0, 1, ... columns.
        d = i - full
        offset = full * m + d * (m - k - full) - symbolic.int_floor(d * (d - 1), 2) + j - i - k
        emit('trapezoid', {'__i': f'{full}:{last}', '__j': f'{symbolic.symstr(i + k)}:{m}'}, offset)
    return [rows, cols]


@oprepo.replaces('numpy.fromfunction')
def fromfunction(pv: ProgramVisitor,
                 sdfg: SDFG,
                 state: SDFGState,
                 function: str,
                 shape: Shape,
                 dtype: dtypes.typeclass | None = None,
                 like: Any = None,
                 **kwargs: Any) -> str:
    """``np.fromfunction`` INLINES the callable: its parameters are the map indices of each axis.

    Only an inline arithmetic lambda can be inlined. A named callable reaches this replacement as data
    rather than as source, and a call or attribute in the body would need memlets that the index
    variables alone cannot supply, so both are refused instead of falling back to the interpreter.
    """
    if kwargs:
        raise ValueError('numpy.fromfunction with extra keyword arguments for the callable is unsupported')
    if like is not None:
        raise ValueError('numpy.fromfunction does not support the ``like`` argument')
    if not isinstance(function, str):
        raise ValueError('numpy.fromfunction needs an inline lambda; a named callable cannot be inlined')
    try:
        lam = ast.parse(function.strip(), mode='eval').body
    except SyntaxError:
        lam = None
    if not isinstance(lam, ast.Lambda):
        raise ValueError('numpy.fromfunction needs an inline lambda over the index variables')
    fargs = lam.args
    if fargs.posonlyargs or fargs.vararg or fargs.kwonlyargs or fargs.kwarg or fargs.defaults:
        raise ValueError('numpy.fromfunction needs a lambda with one plain parameter per dimension')

    extents = static_extents('numpy.fromfunction', shape)
    params = [a.arg for a in fargs.args]
    if len(params) != len(extents):
        raise ValueError(f'numpy.fromfunction: the lambda takes {len(params)} parameters for a rank-{len(extents)} '
                         'shape')

    itype = dtypes.float64 if dtype is None else index_dtype(dtype)
    if not np.issubdtype(itype.type, np.number) or np.issubdtype(itype.type, np.complexfloating):
        raise ValueError('numpy.fromfunction is supported for real index dtypes')

    for node in ast.walk(lam.body):
        if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
            raise ValueError('numpy.fromfunction can only inline an arithmetic lambda over its index variables')

    constants = {}
    for node in ast.walk(lam.body):
        if not isinstance(node, ast.Name) or node.id in params:
            continue
        if node.id in sdfg.arrays:
            raise ValueError(f'numpy.fromfunction cannot read the array "{node.id}" inside its lambda')
        if node.id in sdfg.symbols or node.id in sdfg.constants:
            continue
        value = pv.globals.get(node.id)
        if isinstance(value, symbolic.symbol):
            continue
        if not isinstance(value, Number):
            raise ValueError(f'numpy.fromfunction cannot resolve the name "{node.id}" inside its lambda')
        constants[node.id] = astutils.create_constant(value)

    # Typing sees bare index names; the emitted body casts them, because numpy hands the callable
    # values of ``dtype`` and integer indices would silently truncate the arithmetic.
    named = {**constants, **{p: ast.Name(id=f'__i{d}', ctx=ast.Load()) for d, p in enumerate(params)}}
    cast = {**constants, **{p: ast.parse(f'{itype.ctype}(__i{d})').body[0].value for d, p in enumerate(params)}}
    typed_body = astutils.ASTFindReplace(named).visit(astutils.copy_tree(lam.body))
    code_body = astutils.ASTFindReplace(cast).visit(astutils.copy_tree(lam.body))

    symbol_types = {**sdfg.symbols, **{f'__i{d}': itype for d in range(len(params))}}
    outtype = infer_expr_type(astutils.unparse(typed_body), symbol_types) or itype

    outname, _ = sdfg.add_transient(pv.get_target_name(), extents, outtype, find_new_name=True)
    index = ', '.join(f'__i{d}' for d in range(len(extents)))
    state.add_mapped_tasklet(name='fromfunction',
                             map_ranges={
                                 f'__i{d}': f'0:{symbolic.symstr(s)}'
                                 for d, s in enumerate(extents)
                             },
                             inputs={},
                             code=f'__out = {astutils.unparse(code_body)}',
                             outputs={'__out': Memlet(f'{outname}[{index}]')},
                             external_edges=True)
    return outname
