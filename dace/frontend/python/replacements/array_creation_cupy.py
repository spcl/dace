# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Adds CuPy support for array creation functions.
"""
from dace.frontend.common import op_repository as oprepo
import dace.frontend.python.memlet_parser as mem_parser
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, sym_type
from dace import dtypes, symbolic, Memlet, SDFG, SDFGState

from numbers import Number

import numpy as np


@oprepo.replaces("cupy._core.core.ndarray")
@oprepo.replaces("cupy.ndarray")
def _define_cupy_local(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """Defines a local array in a DaCe program."""
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    name, _ = pv.add_temp_transient(shape, dtype, storage=dtypes.StorageType.GPU_Global)
    return name


@oprepo.replaces('cupy.full')
def _cupy_full(pv: ProgramVisitor,
               sdfg: SDFG,
               state: SDFGState,
               shape: Shape,
               fill_value: symbolic.SymbolicType,
               dtype: dtypes.typeclass = None):
    """ Creates and array of the specified shape and initializes it with
        the fill value.
    """
    if isinstance(fill_value, (Number, np.bool_)):
        vtype = dtypes.dtype_to_typeclass(type(fill_value))
    elif symbolic.issymbolic(fill_value):
        vtype = sym_type(fill_value)
    else:
        raise mem_parser.DaceSyntaxError(pv, None, "Fill value {f} must be a number!".format(f=fill_value))
    dtype = dtype or vtype
    name, _ = pv.add_temp_transient(shape, dtype, storage=dtypes.StorageType.GPU_Global)

    state.add_mapped_tasklet('_cupy_full_', {
        "__i{}".format(i): "0: {}".format(s)
        for i, s in enumerate(shape)
    }, {},
                             "__out = {}".format(fill_value),
                             dict(__out=Memlet.simple(name, ",".join(["__i{}".format(i) for i in range(len(shape))]))),
                             external_edges=True)

    return name


@oprepo.replaces('cupy.zeros')
def _cupy_zeros(pv: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                shape: Shape,
                dtype: dtypes.typeclass = dtypes.float64):
    """ Creates and array of the specified shape and initializes it with zeros.
    """
    return _cupy_full(pv, sdfg, state, shape, 0.0, dtype)


@oprepo.replaces('cupy.empty_like')
def _cupy_empty_like(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     prototype: str,
                     dtype: dtypes.typeclass = None,
                     shape: Shape = None):
    if prototype not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=prototype))
    desc = sdfg.arrays[prototype]
    name, newdesc = sdfg.add_temp_transient_like(desc, name=pv.get_target_name())
    if dtype is not None:
        newdesc.dtype = dtype
    if shape is not None:
        newdesc.shape = shape
    return name


@oprepo.replaces('cupy.empty')
@oprepo.replaces('cupy_empty')
def _cupy_empty(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_cupy_local(pv, sdfg, state, shape, dtype)
