# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for array-filling methods (zeros, ones, etc.)
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, sym_type
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

from numbers import Number
from typing import Union

import numpy as np
import sympy as sp


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
    is_data = False
    if isinstance(fill_value, (Number, np.bool_)):
        vtype = dtypes.dtype_to_typeclass(type(fill_value))
    elif isinstance(fill_value, sp.Expr):
        vtype = sym_type(fill_value)
    else:
        is_data = True
        vtype = sdfg.arrays[fill_value].dtype
    dtype = dtype or vtype
    name, _ = sdfg.add_temp_transient(shape, dtype)

    if is_data:
        state.add_mapped_tasklet('_numpy_full_', {"__i{}".format(i): "0: {}".format(s)
                                                  for i, s in enumerate(shape)},
                                 dict(__inp=Memlet(data=fill_value, subset='0')),
                                 "__out = __inp",
                                 dict(__out=Memlet.simple(name, ",".join(["__i{}".format(i)
                                                                          for i in range(len(shape))]))),
                                 external_edges=True)
    else:
        state.add_mapped_tasklet('_numpy_full_', {"__i{}".format(i): "0: {}".format(s)
                                                  for i, s in enumerate(shape)}, {},
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
    if a not in sdfg.arrays.keys():
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
        map_ranges={f"__i{dim}": f"0:{s}"
                    for dim, s in enumerate(shape)},
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
                             dict(i='0:%s' % N, j='0:%s' % M), {},
                             'val = 1 if i == (j - %s) else 0' % k,
                             dict(val=Memlet.simple(name, 'i, j')),
                             external_edges=True)

    return name


@oprepo.replaces('numpy.identity')
def _numpy_identity(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, n, dtype=dtypes.float64):
    """ Generates the nxn identity matrix. """
    return eye(pv, sdfg, state, n, dtype=dtype)


@oprepo.replaces('numpy.arange')
@oprepo.replaces('dace.arange')
def _arange(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args, **kwargs):
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

    actual_step = step
    if isinstance(start, Number) and isinstance(stop, Number):
        actual_step = type(start + step)(start + step) - start

    if any(not isinstance(s, Number) for s in [start, stop, step]):
        shape = (symbolic.int_ceil(stop - start, step), )
    else:
        shape = (np.ceil((stop - start) / step), )

    if not isinstance(shape[0], Number) and ('dtype' not in kwargs or kwargs['dtype'] == None):
        raise NotImplementedError("The current implementation of numpy.arange requires that the output dtype is given "
                                  "when at least one of (start, stop, step) is symbolic.")
    # TODO: Unclear what 'like' does
    # if 'like' in kwargs and kwargs['like'] != None:
    #     outname, outarr = sdfg.add_temp_transient_like(sdfg.arrays[kwargs['like']])
    #     outarr.shape = shape
    if 'dtype' in kwargs and kwargs['dtype'] != None:
        dtype = kwargs['dtype']
        if not isinstance(dtype, dtypes.typeclass):
            dtype = dtypes.dtype_to_typeclass(dtype)
        outname, outarr = sdfg.add_temp_transient(shape, dtype)
    else:
        dtype = dtypes.dtype_to_typeclass(type(shape[0]))
        outname, outarr = sdfg.add_temp_transient(shape, dtype)

    state.add_mapped_tasklet(name="_numpy_arange_",
                             map_ranges={'__i': f"0:{shape[0]}"},
                             inputs={},
                             code=f"__out = {start} + __i * {actual_step}",
                             outputs={'__out': Memlet(f"{outname}[__i]")},
                             external_edges=True)

    return outname
