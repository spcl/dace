# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import sympy
import warnings

from dace import data, dtypes, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements import _np_result_type, _sym_type
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from numbers import Number, Integral
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'
Scalar = Union[str, Number, symbolic.SymbolicType]
Size = Union[int, symbolic.SymbolicType]


def _is_scalar(desc: data.Data) -> bool:
    return isinstance(desc, data.Scalar) or (isinstance(desc, (data.Array, data.View)) and desc.shape == (1, ))


def _normalize_axis_index(axis: int, ndims: int) -> int:
    if axis < 0:
        axis += ndims
    if axis < 0 or axis >= ndims:
        raise ValueError(f"Axis {axis} is out of bounds for array of dimension {ndims}.")
    return axis


@oprepo.replaces('numpy.linspace')
def linspace(visitor: ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             start: Scalar,
             stop: Scalar,
             num: Size = 50,
             endpoint: bool = True,
             retstep: bool = False,
             dtype: dtypes.typeclass = None,
             axis=0) -> str:
    """ Implements numpy.linspace.
    
        The method supports symbolic start, stop, and num arguments.
        However, it does not support array-like start, stop arguments, the retstep, and the axis argument.
        If endpoint is True and num is symbolic, then it is assumed that num is greater than 1.
    """

    start_conn = None
    start_desc = None
    start_type = start
    if isinstance(start, str):
        if start not in sdfg.arrays:
            raise ValueError(f"Data {start} not found in SDFG.")
        start_desc = sdfg.arrays[start]
        if not _is_scalar(start_desc):
            raise NotImplementedError("DaCe's np.linspace implementation does not support array-like start argument.")
        start_conn = '__inp0'
        start_type = start_desc.dtype.type
    elif isinstance(start, (sympy.Basic, symbolic.SymExpr)):
        start_type = _sym_type(start).type

    stop_conn = None
    stop_desc = None
    stop_type = stop
    if isinstance(stop, str):
        if stop not in sdfg.arrays:
            raise ValueError(f"Data {stop} not found in SDFG.")
        stop_desc = sdfg.arrays[stop]
        if not _is_scalar(stop_desc):
            raise NotImplementedError("DaCe's np.linspace implementation does not support array-like stop argument.")
        stop_conn = '__inp1'
        stop_type = stop_desc.dtype.type
    elif isinstance(stop, (sympy.Basic, symbolic.SymExpr)):
        stop_type = _sym_type(stop).type

    if retstep:
        raise NotImplementedError("DaCe's np.linspace implementation does not support retstep argument.")

    if axis != 0:
        raise NotImplementedError("DaCe's np.linspace implementation does not support axis argument.")

    num_type = num
    if isinstance(num, Number):
        if isinstance(num, Integral):
            if num < 0:
                raise ValueError(f"Number of samples, {num}, must be non-negative.")
            num_type = float(num)
        else:
            raise TypeError(f"Number of samples, {num}, must be an integer.")
    else:
        num_type = _sym_type(num).type

    div = num - 1 if endpoint else num

    dt = _np_result_type([start_type, stop_type, num_type])
    if dtype is None:
        dtype = dt

    param = '__i0'
    out_conn = '__out'

    out_name, out_arr = sdfg.add_transient('tmp', (num, ), dtype, find_new_name=True)

    # NOTE: We do not need special care of integer output types due to implicit casting.
    code = ''
    if isinstance(div, Number) and isinstance(start, Number) and isinstance(stop, Number):
        delta = stop - start
        step = delta / div
        if step == 0:
            code = f"{out_conn} = (({param}) / ({div})) * ({delta}) + ({start})"
        else:
            code = f"{out_conn} = ({param}) * ({step}) + ({start})"
    else:
        delta = f"{stop_conn if stop_conn else stop} - {start_conn if start_conn else start}"
        code = f"{out_conn} = ({param}) * ({delta}) / ({div}) + {start_conn if start_conn else start}"

    length = num
    if endpoint:
        if isinstance(num, Integral):
            if num > 1:
                length = num - 1
        else:
            length = num - 1
            warnings.warn(f"Assuming that the number of samples, {num}, is greater than 1.")

    inputs = {}
    if start_conn:
        inputs[start_conn] = Memlet.from_array(start, start_desc)
    if stop_conn:
        inputs[stop_conn] = Memlet.from_array(stop, stop_desc)
    outputs = {out_conn: Memlet(data=out_name, subset=param)}

    state.add_mapped_tasklet('linspace', {param: f'0:{length}'}, inputs, code, outputs, external_edges=True)

    if length != num:
        inputs = set()
        if stop_conn:
            inputs.add(stop_conn)
            code = f"{out_conn} = {stop_conn}"
        else:
            code = f"{out_conn} = {stop}"
        tasklet = state.add_tasklet('linspace_last', inputs, {out_conn}, code)
        if stop_conn:
            stop_node = state.add_access(stop)
            state.add_edge(stop_node, None, tasklet, stop_conn, Memlet.from_array(stop, stop_desc))
        out_node = state.add_access(out_name)
        state.add_edge(tasklet, out_conn, out_node, None, Memlet(data=out_name, subset=f'{length}'))

    return out_name


@oprepo.replaces('numpy.concatenate')
def concatenate(visitor: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                arrays: Sequence[str],
                axis: int = 0,
                out: str = None,
                dtype: dtypes.typeclass = None,
                casting: str = 'same_kind') -> Union[None, str]:
    """ Implements numpy.concatenate.

        Doesn't support axis = None.
    """

    # Get array descriptors
    arr_descs = [sdfg.arrays[a] for a in arrays]

    if not arr_descs:
        raise ValueError('At least one is array is needed to concatenate.')
    
    if axis is None:
        raise NotImplementedError("DaCe's np.concatenate implementation does not support axis=None.")
    
    # TODO (minor): NumPy returns a more detailed error message here.
    ndims = {i: len(arr_desc.shape) for i, arr_desc in enumerate(arr_descs)}
    if len(set(ndims.values())) != 1:
        raise ValueError('All input arrays must have the same number of dimensions.')
    
    result_ndim = ndims[0]
    axis = _normalize_axis_index(axis, result_ndim)

    # TODO (minor): NumPy returns a more detailed error message here.
    shapes = {i: tuple(s for j, s in enumerate(arr_desc.shape) if j != axis) for i, arr_desc in enumerate(arr_descs)}
    if len(set(shapes.values())) != 1:
        raise ValueError('All input array dimensions except for the concatenation axis must match exactly.')

    concat_dim = sum(arr_desc.shape[axis] for arr_desc in arr_descs)

    if result_ndim > 1:
        result_shape = tuple(list(shapes[0]).insert(axis, concat_dim))
    else:
        result_shape = (concat_dim, )

    if out is None:
        # TODO: Casting
        if dtype is None:
            dtype = arr_descs[0].dtype
        out_name, out_arr = sdfg.add_transient('tmp', result_shape, dtype, find_new_name=True)
    else:
        out_arr = sdfg.arrays[out]
        out_name = out
        if dtype is not None:
            raise TypeError("concatenate() only takes `out` or `dtype` as an argument, but both were provided.")
        if result_ndim != len(out_arr.shape):
            raise ValueError("Output array has wrong dimensionality.")
        if result_shape != out_arr.shape:
            raise ValueError("Output array has wrong shape.")
    
    index = 0
    out_node = state.add_access(out_name)
    for arr_name, arr_desc in zip(arrays, arr_descs):
        arr_node = state.add_access(arr_name)
        subset = ','.join(f'0:{s}' if i != axis else f'{index}:{index + s}' for i, s in enumerate(arr_desc.shape))
        other_subset = ','.join(f'0:{s}' for s in arr_desc.shape)
        state.add_edge(arr_node, None, out_node, None, Memlet(data=out_name, subset=subset, other_subset=other_subset))
        index += arr_desc.shape[axis]
    
    if out is None:
        return out_name


@oprepo.replaces('numpy.stack')
def stack(visitor: ProgramVisitor,
          sdfg: SDFG,
          state: SDFGState,
          arrays: Sequence[str],
          axis: int = 0,
          out: str = None,
          dtype: dtypes.typeclass = None,
          casting: str = 'same_kind') -> Union[None, str]:
    """ Implements numpy.stack. """
    
    # Get array descriptors
    arr_descs = [sdfg.arrays[a] for a in arrays]

    if not arr_descs:
        raise ValueError('At least one is array is needed to stack.')
    
    shapes = {arr.shape for arr in arr_descs}
    if len(shapes) != 1:
        raise ValueError('All input arrays must have the same shape.')
    
    result_ndim = len(arr_descs[0].shape) + 1
    axis = _normalize_axis_index(axis, result_ndim)

    if axis == result_ndim - 1:
        from dace.frontend.python.replacements import reshape
        arrays = [reshape(visitor, sdfg, state, arr, (*desc.shape, 1), 'A') for arr, desc in zip(arrays, arr_descs)]
    
    return concatenate(visitor, sdfg, state, arrays, axis, out, dtype, casting)


@oprepo.replaces('numpy.hstack')
def hstack(visitor: ProgramVisitor,
          sdfg: SDFG,
          state: SDFGState,
          arrays: Sequence[str],
          axis: int = 0,
          out: str = None,
          dtype: dtypes.typeclass = None,
          casting: str = 'same_kind') -> Union[None, str]:
    """ Implements numpy.hstack. """

    # Get array descriptors
    arr_descs = [sdfg.arrays[a] for a in arrays]

    if arr_descs and len(arr_descs[0].shape) == 1:
        return concatenate(visitor, sdfg, state, arrays, 0, out, dtype, casting)
    else:
        return concatenate(visitor, sdfg, state, arrays, 1, out, dtype, casting)
