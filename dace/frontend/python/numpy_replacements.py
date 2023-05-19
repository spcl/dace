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
             axis=0):
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


@oprepo.replaces('numpy.hstack')
def hstack(visitor: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState):
    """ Implements numpy.hstack."""
    pass
