# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for array-filling methods (zeros, ones, etc.)
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, sym_type, broadcast_together
from dace.frontend.python.replacements.operators import result_type
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

import copy
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
    # TODO: The whole AddTransientMethod class should be move in replacements.py
    from dace.frontend.python.newast import _add_transient_data
    name, desc = _add_transient_data(sdfg, sdfg.arrays[a])
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

    if any(isinstance(s, str) for s in shape):
        raise DaceSyntaxError(
            pv, None, f'Data-dependent shape {shape} is currently not allowed. Only constants '
            'and symbolic values can be used.')

    name, _ = sdfg.add_temp_transient(shape, dtype)

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
    name, _ = sdfg.add_temp_transient([N, M], dtype)

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

    if isinstance(start, str):
        raise TypeError(f'Cannot compile numpy.arange with a scalar start value "{start}" (only constants and symbolic '
                        'expressions are supported). Please use numpy.linspace instead.')
    if isinstance(stop, str):
        raise TypeError(f'Cannot compile numpy.arange with a scalar stop value "{stop}" (only constants and symbolic '
                        'expressions are supported). Please use numpy.linspace instead.')
    if isinstance(step, str):
        raise TypeError(f'Cannot compile numpy.arange with a scalar step value "{step}" (only constants and symbolic '
                        'expressions are supported). Please use numpy.linspace instead.')

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
    #     outname, outarr = sdfg.add_temp_transient_like(sdfg.arrays[like])
    #     outarr.shape = shape

    if not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.dtype_to_typeclass(dtype)
    outname, outarr = sdfg.add_temp_transient(shape, dtype)

    start = f'decltype(__out)({start})'
    actual_step = f'decltype(__out)({actual_step})'

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
    start_shape = sdfg.arrays[start].shape if (isinstance(start, str) and start in sdfg.arrays) else []
    stop_shape = sdfg.arrays[stop].shape if (isinstance(stop, str) and stop in sdfg.arrays) else []

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

    outname, _ = sdfg.add_temp_transient(shape_with_axis, dtype)

    if endpoint == True:
        num -= 1

    # Fill in input memlets as necessary
    inputs = {}
    if isinstance(start, str) and start in sdfg.arrays:
        index = f'[{ind1}]' if ind1 else ''
        inputs['__start'] = Memlet(f'{start}{index}')
        startcode = '__start'
    else:
        startcode = symbolic.symstr(start)

    if isinstance(stop, str) and stop in sdfg.arrays:
        index = f'[{ind2}]' if ind2 else ''
        inputs['__stop'] = Memlet(f'{stop}{index}')
        stopcode = '__stop'
    else:
        stopcode = symbolic.symstr(stop)

    # Create tasklet code based on inputs
    code = f'__out = {startcode} + __sind * decltype(__out)({stopcode} - {startcode}) / decltype(__out)({symbolic.symstr(num)})'

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

    if len(shape) > 0:
        stepname, _ = sdfg.add_temp_transient(shape, dtype)
    else:
        stepname, _ = sdfg.add_scalar(sdfg.temp_data_name(), dtype, transient=True)
        out_index = '[0]'

    state.add_mapped_tasklet(
        'retstep',
        ranges,
        copy.deepcopy(inputs),
        f'__out = decltype(__out)({stopcode} - {startcode}) / decltype(__out)({symbolic.symstr(num)})',
        {'__out': Memlet(f"{stepname}{out_index}")},
        external_edges=True)

    return outname, stepname
