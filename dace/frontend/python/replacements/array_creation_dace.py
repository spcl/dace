# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains definitions of new data containers (arrays, locals, streams) as per DaCe's API, as well as several
array creation functions for NumPy that reuse the same functionality.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, Size, sym_type
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

from copy import deepcopy as dcpy
from numbers import Integral
from typing import Any, Optional

import numpy as np


def _normalize_allocator_shape(shape: Shape):
    if isinstance(shape, Integral) or symbolic.issymbolic(shape):
        return [shape]
    if not isinstance(shape, (list, tuple)):
        return None
    return list(shape)


def infer_array_creation_descriptor(obj: Any,
                                    *,
                                    dtype: dtypes.typeclass = None,
                                    copy: bool = True,
                                    order: StringLiteral = StringLiteral('K'),
                                    subok: bool = False,
                                    ndmin: int = 0,
                                    like: Any = None) -> Optional[data.Data]:
    if like is not None:
        return None

    if dtype is not None and not isinstance(dtype, dtypes.typeclass):
        try:
            dtype = dtypes.typeclass(dtype)
        except TypeError:
            return None

    try:
        if dtype is None:
            arr = np.array(obj, copy=copy, order=str(order), subok=subok, ndmin=ndmin)
        else:
            arr = np.array(obj, dtype.as_numpy_dtype(), copy=copy, order=str(order), subok=subok, ndmin=ndmin)
    except Exception:
        return None

    try:
        descriptor = data.create_datadescriptor(arr)
    except TypeError:
        scalar_dtype = dtypes.typeclass(np.asarray(arr).dtype.type)
        if getattr(arr, 'shape', tuple()) == tuple():
            descriptor = data.Scalar(scalar_dtype, transient=True)
        else:
            return None
    descriptor.transient = True
    return descriptor


def infer_dynamic_literal_descriptor(obj: Any,
                                     sdfg: SDFG,
                                     *,
                                     dtype: dtypes.typeclass = None,
                                     ndmin: int = 0) -> Optional[data.Array]:
    shape_dtype = _infer_dynamic_literal_shape_dtype(obj, sdfg)
    if shape_dtype is None:
        return None

    shape, inferred_dtype = shape_dtype
    result_dtype = dtype or inferred_dtype
    if result_dtype is None:
        return None

    out_shape = list(shape)
    if len(out_shape) < ndmin:
        out_shape = [1] * (ndmin - len(out_shape)) + out_shape
    return data.Array(result_dtype, out_shape, transient=True)


def populate_dynamic_literal_array(state: SDFGState, sdfg: SDFG, outname: str, obj: Any) -> None:
    outdesc = sdfg.arrays[outname]
    constant_array = _entire_constant_literal_array(obj, outdesc)
    if constant_array is not None:
        const_name = sdfg.find_new_constant(f'{outname}_literal')
        sdfg.add_constant(const_name, constant_array)
        sdfg.arrays[const_name] = sdfg.constants_prop[const_name][0]
        sdfg.arrays[const_name].transient = True
        read = state.add_read(const_name)
        write = state.add_write(outname)
        subset = ', '.join(f'0:{dim}' for dim in constant_array.shape)
        state.add_edge(read, None, write, None, Memlet.simple(const_name, subset, other_subset_str=subset))
        return

    write = state.add_write(outname)
    counter = 0

    def emit(value: Any, index: tuple[int, ...]) -> None:
        nonlocal counter
        if isinstance(value, (list, tuple)):
            for child_idx, child in enumerate(value):
                emit(child, index + (child_idx, ))
            return

        tasklet_name = f'{outname}_literal_{counter}'
        counter += 1
        subset = ', '.join(str(i) for i in index)
        if isinstance(value, str) and value in sdfg.arrays:
            desc = sdfg.arrays[value]
            read = state.add_read(value)
            tasklet = state.add_tasklet(tasklet_name, {'__inp'}, {'__out'}, '__out = __inp')
            state.add_edge(read, None, tasklet, '__inp', Memlet.from_array(value, desc))
            state.add_edge(tasklet, '__out', write, None, Memlet.simple(outname, subset))
            return

        tasklet = state.add_tasklet(tasklet_name, set(), {'__out'}, f'__out = {_literal_code(value)}')
        state.add_edge(tasklet, '__out', write, None, Memlet.simple(outname, subset))

    emit(obj, tuple())


def _entire_constant_literal_array(obj: Any, outdesc: data.Array) -> Optional[np.ndarray]:
    if not _is_entire_literal_constant(obj):
        return None
    npdtype = outdesc.dtype.as_numpy_dtype()
    result = np.array(obj, dtype=npdtype)
    if tuple(result.shape) != tuple(outdesc.shape):
        try:
            result = result.reshape(tuple(outdesc.shape))
        except ValueError:
            return None
    return result


def _is_entire_literal_constant(obj: Any) -> bool:
    if isinstance(obj, (list, tuple)):
        return all(_is_entire_literal_constant(v) for v in obj)
    return isinstance(obj, (np.generic, bool, int, float, complex))


def _infer_dynamic_literal_shape_dtype(obj: Any, sdfg: SDFG) -> Optional[tuple[tuple[int, ...], dtypes.typeclass]]:
    if isinstance(obj, (list, tuple)):
        child_shapes: list[tuple[int, ...]] = []
        child_dtype: Optional[dtypes.typeclass] = None
        for element in obj:
            shape_dtype = _infer_dynamic_literal_shape_dtype(element, sdfg)
            if shape_dtype is None:
                return None
            element_shape, element_dtype = shape_dtype
            child_shapes.append(element_shape)
            child_dtype = element_dtype if child_dtype is None else dtypes.result_type_of(child_dtype, element_dtype)

        if not child_shapes:
            return ((0, ), dtypes.float64)

        first_shape = child_shapes[0]
        if any(shape != first_shape for shape in child_shapes[1:]):
            return None
        return ((len(obj), ) + first_shape, child_dtype)

    dtype = _dynamic_literal_scalar_dtype(obj, sdfg)
    if dtype is None:
        return None
    return (tuple(), dtype)


def _dynamic_literal_scalar_dtype(obj: Any, sdfg: SDFG) -> Optional[dtypes.typeclass]:
    if isinstance(obj, np.generic):
        return dtypes.typeclass(obj.dtype.type)
    if isinstance(obj, bool):
        return dtypes.bool
    if isinstance(obj, (int, float, complex)):
        return dtypes.typeclass(type(obj))
    if symbolic.issymbolic(obj):
        return sym_type(obj)
    if isinstance(obj, str):
        if obj in sdfg.arrays:
            desc = sdfg.arrays[obj]
            if isinstance(desc, data.Scalar):
                return desc.dtype
            if isinstance(desc, data.Array) and tuple(desc.shape) == (1, ):
                return desc.dtype
            return None
        if obj in sdfg.symbols:
            return sdfg.symbols[obj]
        try:
            parsed = symbolic.pystr_to_symbolic(obj)
        except Exception:
            return None
        if symbolic.issymbolic(parsed):
            return sym_type(parsed)
    return None


def _literal_code(value: Any) -> str:
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, str):
        return value
    if symbolic.issymbolic(value):
        return symbolic.symstr(value)
    return repr(value)


@oprepo.infers_descriptor('dace.define_local')
@oprepo.infers_descriptor('dace.ndarray')
@oprepo.infers_descriptor('numpy.ndarray')
@oprepo.infers_descriptor('numpy.empty')
def _infer_local_array_descriptor(input_descs, shape: Shape, dtype: dtypes.typeclass, **_kw):
    del input_descs
    out_shape = _normalize_allocator_shape(shape)
    if out_shape is None or dtype is None:
        return None
    if not isinstance(dtype, dtypes.typeclass):
        try:
            dtype = dtypes.dtype_to_typeclass(dtype)
        except (TypeError, ValueError):
            return None
    return data.Array(dtype, out_shape, transient=True)


@oprepo.infers_descriptor('dace.define_local_scalar')
def _infer_local_scalar_descriptor(input_descs, dtype: dtypes.typeclass, **_kw):
    del input_descs
    if dtype is None:
        return None
    if not isinstance(dtype, dtypes.typeclass):
        try:
            dtype = dtypes.dtype_to_typeclass(dtype)
        except (TypeError, ValueError):
            return None
    return data.Scalar(dtype, transient=True)


@oprepo.infers_descriptor('dace.define_local_structure')
def _infer_local_structure_descriptor(input_descs, dtype: data.Structure, **_kw):
    del input_descs
    if dtype is None:
        return None
    descriptor = dcpy(dtype)
    descriptor.transient = True
    return descriptor


def _normalize_inferred_dtype(dtype: dtypes.typeclass) -> Optional[dtypes.typeclass]:
    if dtype is None:
        return None
    if isinstance(dtype, dtypes.typeclass):
        return dtype
    try:
        return dtypes.dtype_to_typeclass(dtype)
    except (TypeError, ValueError):
        return None


@oprepo.infers_descriptor('dace.define_stream')
def _infer_stream_descriptor(input_descs, dtype: dtypes.typeclass, buffer_size: Size = 1, **_kw):
    out_dtype = _normalize_inferred_dtype(dtype)
    if out_dtype is None:
        return None
    return data.Stream(out_dtype, buffer_size, transient=True)


@oprepo.infers_descriptor('dace.define_streamarray')
@oprepo.infers_descriptor('dace.stream')
def _infer_streamarray_descriptor(input_descs, shape: Shape, dtype: dtypes.typeclass, buffer_size: Size = 1, **_kw):
    out_shape = _normalize_allocator_shape(shape)
    out_dtype = _normalize_inferred_dtype(dtype)
    if out_shape is None or out_dtype is None:
        return None
    return data.Stream(out_dtype, buffer_size, shape=out_shape, transient=True)


@oprepo.infers_descriptor('numpy.array')
@oprepo.infers_descriptor('dace.array')
def _infer_literal_array_descriptor(input_descs,
                                    obj: Any,
                                    dtype: dtypes.typeclass = None,
                                    copy: bool = True,
                                    order: StringLiteral = StringLiteral('K'),
                                    subok: bool = False,
                                    ndmin: int = 0,
                                    like: Any = None,
                                    **_kw):
    del input_descs
    return infer_array_creation_descriptor(obj,
                                           dtype=dtype,
                                           copy=copy,
                                           order=order,
                                           subok=subok,
                                           ndmin=ndmin,
                                           like=like)


@oprepo.replaces('dace.define_local')
@oprepo.replaces('dace.ndarray')
def _define_local_ex(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     shape: Shape,
                     dtype: dtypes.typeclass,
                     strides: Optional[Shape] = None,
                     storage: dtypes.StorageType = dtypes.StorageType.Default,
                     lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope):
    """ Defines a local array in a DaCe program. """
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    if strides is not None:
        if not isinstance(strides, (list, tuple)):
            strides = [strides]
        strides = [int(s) if isinstance(s, Integral) else s for s in strides]
    name = pv.get_target_name()
    name, _ = sdfg.add_transient(name,
                                 shape,
                                 dtype,
                                 strides=strides,
                                 storage=storage,
                                 lifetime=lifetime,
                                 find_new_name=True)
    return name


@oprepo.replaces('numpy.ndarray')
def _define_local(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """ Defines a local array in a DaCe program. """
    return _define_local_ex(pv, sdfg, state, shape, dtype)


@oprepo.replaces('dace.define_local_scalar')
def _define_local_scalar(pv: ProgramVisitor,
                         sdfg: SDFG,
                         state: SDFGState,
                         dtype: dtypes.typeclass,
                         storage: dtypes.StorageType = dtypes.StorageType.Default,
                         lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope):
    """ Defines a local scalar in a DaCe program. """
    name = pv.get_target_name()
    name, desc = sdfg.add_scalar(name, dtype, transient=True, storage=storage, lifetime=lifetime, find_new_name=True)
    pv.variables[name] = name
    return name


@oprepo.replaces('dace.define_local_structure')
def _define_local_structure(pv: ProgramVisitor,
                            sdfg: SDFG,
                            state: SDFGState,
                            dtype: data.Structure,
                            storage: dtypes.StorageType = dtypes.StorageType.Default,
                            lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope):
    """ Defines a local structure in a DaCe program. """
    name = pv.get_target_name()
    desc = dcpy(dtype)
    desc.transient = True
    desc.storage = storage
    desc.lifetime = lifetime
    name = sdfg.add_datadesc(name, desc, find_new_name=True)
    pv.variables[name] = name
    return name


@oprepo.replaces('dace.define_stream')
def _define_stream(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dtype: dtypes.typeclass, buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = pv.get_target_name()
    name, _ = sdfg.add_stream(name, dtype, buffer_size=buffer_size, transient=True, find_new_name=True)
    return name


@oprepo.replaces('dace.define_streamarray')
@oprepo.replaces('dace.stream')
def _define_streamarray(pv: ProgramVisitor,
                        sdfg: SDFG,
                        state: SDFGState,
                        shape: Shape,
                        dtype: dtypes.typeclass,
                        buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = pv.get_target_name()
    name, _ = sdfg.add_stream(name, dtype, shape=shape, buffer_size=buffer_size, transient=True, find_new_name=True)
    return name


@oprepo.replaces('numpy.array')
@oprepo.replaces('dace.array')
def _define_literal_ex(pv: ProgramVisitor,
                       sdfg: SDFG,
                       state: SDFGState,
                       obj: Any,
                       dtype: dtypes.typeclass = None,
                       copy: bool = True,
                       order: StringLiteral = StringLiteral('K'),
                       subok: bool = False,
                       ndmin: int = 0,
                       like: Any = None,
                       storage: Optional[dtypes.StorageType] = None,
                       lifetime: Optional[dtypes.AllocationLifetime] = None):
    """ Defines a literal array in a DaCe program. """
    if like is not None:
        raise NotImplementedError('"like" argument unsupported for numpy.array')

    name = pv.get_target_name()
    if dtype is not None and not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.typeclass(dtype)

    # From existing data descriptor
    if isinstance(obj, str):
        desc = dcpy(sdfg.arrays[obj])
        if dtype is not None:
            desc.dtype = dtype
        dynamic_literal = False
    else:  # From literal / constant
        desc = infer_array_creation_descriptor(obj,
                                               dtype=dtype,
                                               copy=copy,
                                               order=order,
                                               subok=subok,
                                               ndmin=ndmin,
                                               like=like)
        dynamic_literal = desc is None
        if dynamic_literal:
            desc = infer_dynamic_literal_descriptor(obj, sdfg, dtype=dtype, ndmin=ndmin)
            if desc is None:
                raise DaceSyntaxError(pv, None, 'Could not infer numpy.array descriptor from literal input')
        else:
            if dtype is None:
                arr = np.array(obj, copy=copy, order=str(order), subok=subok, ndmin=ndmin)
            else:
                npdtype = dtype.as_numpy_dtype()
                arr = np.array(obj, npdtype, copy=copy, order=str(order), subok=subok, ndmin=ndmin)

    # Set extra properties
    desc.transient = True
    if storage is not None:
        desc.storage = storage
    if lifetime is not None:
        desc.lifetime = lifetime

    name = sdfg.add_datadesc(name, desc, find_new_name=True)

    # If using existing array, make copy. Otherwise, make constant
    if isinstance(obj, str):
        # Make copy
        rnode = state.add_read(obj)
        wnode = state.add_write(name)
        state.add_nedge(rnode, wnode, Memlet.from_array(name, desc))
    elif dynamic_literal:
        populate_dynamic_literal_array(state, sdfg, name, obj)
    else:
        # Make constant
        sdfg.add_constant(name, arr, desc)

    return name


@oprepo.replaces('numpy.empty')
def _numpy_empty(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dtypes.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_local(pv, sdfg, state, shape, dtype)


@oprepo.replaces('numpy.empty_like')
def _numpy_empty_like(pv: ProgramVisitor,
                      sdfg: SDFG,
                      state: SDFGState,
                      prototype: str,
                      dtype: dtypes.typeclass = None,
                      shape: Shape = None):
    """ Creates an unitialized array of the same shape and dtype as prototype.
        The optional dtype and shape inputs allow overriding the corresponding
        attributes of prototype.
    """
    if prototype not in sdfg.arrays.keys():
        raise DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=prototype))
    desc = sdfg.arrays[prototype]
    dtype = dtype or desc.dtype
    shape = shape or desc.shape
    return _define_local(pv, sdfg, state, shape, dtype)
