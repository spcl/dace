# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains definitions of new data containers (arrays, locals, streams) as per DaCe's API, as well as several
array creation functions for NumPy that reuse the same functionality.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape, Size
from dace import data, dtypes, Memlet, SDFG, SDFGState

from copy import deepcopy as dcpy
from numbers import Integral
from typing import Any, Optional

import numpy as np


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
    name, _ = sdfg.add_temp_transient(shape, dtype, strides=strides, storage=storage, lifetime=lifetime)
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
    name = sdfg.temp_data_name()
    _, desc = sdfg.add_scalar(name, dtype, transient=True, storage=storage, lifetime=lifetime)
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
    name = sdfg.temp_data_name()
    desc = dcpy(dtype)
    desc.transient = True
    desc.storage = storage
    desc.lifetime = lifetime
    sdfg.add_datadesc(name, desc)
    pv.variables[name] = name
    return name


@oprepo.replaces('dace.define_stream')
def _define_stream(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dtype: dtypes.typeclass, buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_stream(name, dtype, buffer_size=buffer_size, transient=True)
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
    name = sdfg.temp_data_name()
    sdfg.add_stream(name, dtype, shape=shape, buffer_size=buffer_size, transient=True)
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

    name = sdfg.temp_data_name()
    if dtype is not None and not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.typeclass(dtype)

    # From existing data descriptor
    if isinstance(obj, str):
        desc = dcpy(sdfg.arrays[obj])
        if dtype is not None:
            desc.dtype = dtype
    else:  # From literal / constant
        if dtype is None:
            arr = np.array(obj, copy=copy, order=str(order), subok=subok, ndmin=ndmin)
        else:
            npdtype = dtype.as_numpy_dtype()
            arr = np.array(obj, npdtype, copy=copy, order=str(order), subok=subok, ndmin=ndmin)
        desc = data.create_datadescriptor(arr)

    # Set extra properties
    desc.transient = True
    if storage is not None:
        desc.storage = storage
    if lifetime is not None:
        desc.lifetime = lifetime

    sdfg.add_datadesc(name, desc)

    # If using existing array, make copy. Otherwise, make constant
    if isinstance(obj, str):
        # Make copy
        rnode = state.add_read(obj)
        wnode = state.add_write(name)
        state.add_nedge(rnode, wnode, Memlet.from_array(name, desc))
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
