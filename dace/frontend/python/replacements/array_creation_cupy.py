# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Adds CuPy support for array creation functions.
"""
from dace.frontend.common import op_repository as oprepo
import dace.frontend.python.memlet_parser as mem_parser
from dace.frontend.python.replacements.array_creation_dace import _normalize_allocator_shape
from dace.frontend.python.replacements.type_inference import _get_desc
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, sym_type
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

from numbers import Number

import numpy as np


def _normalize_cupy_dtype(dtype: dtypes.typeclass):
    if dtype is None:
        return None
    if isinstance(dtype, dtypes.typeclass):
        return dtype
    try:
        return dtypes.dtype_to_typeclass(dtype)
    except (TypeError, ValueError):
        return None


def _cupy_array_descriptor(shape: Shape, dtype: dtypes.typeclass):
    out_shape = _normalize_allocator_shape(shape)
    out_dtype = _normalize_cupy_dtype(dtype)
    if out_shape is None or out_dtype is None:
        return None
    return data.Array(out_dtype, out_shape, storage=dtypes.StorageType.GPU_Global, transient=True)


@oprepo.replaces("cupy._core.core.ndarray")
@oprepo.replaces("cupy.ndarray")
def _define_cupy_local(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """Defines a local array in a DaCe program."""
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    name, _ = pv.add_temp_transient(shape, dtype, storage=dtypes.StorageType.GPU_Global)
    return name


@oprepo.infers_descriptor("cupy._core.core.ndarray")
@oprepo.infers_descriptor("cupy.ndarray")
def _infer_cupy_local(input_descs, shape: Shape, dtype: dtypes.typeclass, **_kw):
    return _cupy_array_descriptor(shape, dtype)


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


@oprepo.infers_descriptor('cupy.full')
def _infer_cupy_full(input_descs,
                     shape: Shape,
                     fill_value: symbolic.SymbolicType,
                     dtype: dtypes.typeclass = None,
                     **_kw):
    if dtype is None:
        if isinstance(fill_value, (Number, np.bool_)):
            dtype = dtypes.dtype_to_typeclass(type(fill_value))
        elif symbolic.issymbolic(fill_value):
            dtype = sym_type(fill_value)
        else:
            return None
    return _cupy_array_descriptor(shape, dtype)


@oprepo.replaces('cupy.zeros')
def _cupy_zeros(pv: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                shape: Shape,
                dtype: dtypes.typeclass = dtypes.float64):
    """ Creates and array of the specified shape and initializes it with zeros.
    """
    return _cupy_full(pv, sdfg, state, shape, 0.0, dtype)


@oprepo.infers_descriptor('cupy.zeros')
def _infer_cupy_zeros(input_descs, shape: Shape, dtype: dtypes.typeclass = dtypes.float64, **_kw):
    return _cupy_array_descriptor(shape, dtype)


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


@oprepo.infers_descriptor('cupy.empty_like')
def _infer_cupy_empty_like(input_descs, prototype: str, dtype: dtypes.typeclass = None, shape: Shape = None, **_kw):
    desc = _get_desc(input_descs, prototype)
    if not isinstance(desc, data.Data):
        return None
    result = desc.clone()
    if dtype is not None:
        out_dtype = _normalize_cupy_dtype(dtype)
        if out_dtype is None:
            return None
        result.dtype = out_dtype
    if shape is not None:
        out_shape = _normalize_allocator_shape(shape)
        if out_shape is None:
            return None
        result.shape = out_shape
    result.storage = dtypes.StorageType.GPU_Global
    result.transient = True
    return result


@oprepo.replaces('cupy.empty')
@oprepo.replaces('cupy_empty')
def _cupy_empty(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_cupy_local(pv, sdfg, state, shape, dtype)


@oprepo.infers_descriptor('cupy.empty')
@oprepo.infers_descriptor('cupy_empty')
def _infer_cupy_empty(input_descs, shape: Shape, dtype: dtypes.typeclass, **_kw):
    return _cupy_array_descriptor(shape, dtype)
