# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Data descriptor creation functions.

This module contains functions for creating data descriptors from arbitrary objects,
as well as functions for creating arrays from descriptors.
"""
import ctypes

from numbers import Number
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from numpy.typing import ArrayLike
except (ModuleNotFoundError, ImportError):
    ArrayLike = Any

from dace import dtypes, symbolic
from dace.data.core import Array, Data, Scalar


def create_datadescriptor(obj, no_custom_desc=False):
    """ Creates a data descriptor from various types of objects.

        :see: dace.data.Data
    """
    if isinstance(obj, Data):
        return obj
    elif not no_custom_desc and hasattr(obj, '__descriptor__'):
        return obj.__descriptor__()
    elif not no_custom_desc and hasattr(obj, 'descriptor'):
        return obj.descriptor
    elif type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor":
        # special case for torch tensors. Maybe __array__ could be used here for a more
        # general solution, but torch doesn't support __array__ for cuda tensors.
        try:
            # If torch is importable, define translations between typeclasses and torch types. These are reused by daceml.
            # conversion happens here in pytorch:
            # https://github.com/pytorch/pytorch/blob/143ef016ee1b6a39cf69140230d7c371de421186/torch/csrc/utils/tensor_numpy.cpp#L237
            import torch
            TYPECLASS_TO_TORCH_DTYPE = {
                dtypes.bool_: torch.bool,
                dtypes.int8: torch.int8,
                dtypes.int16: torch.int16,
                dtypes.int32: torch.int32,
                dtypes.int64: torch.int64,
                dtypes.uint8: torch.uint8,
                dtypes.float16: torch.float16,
                dtypes.float32: torch.float32,
                dtypes.float64: torch.float64,
                dtypes.complex64: torch.complex64,
                dtypes.complex128: torch.complex128,
            }

            TORCH_DTYPE_TO_TYPECLASS = {v: k for k, v in TYPECLASS_TO_TORCH_DTYPE.items()}

            storage = dtypes.StorageType.GPU_Global if obj.device.type == 'cuda' else dtypes.StorageType.Default

            return Array(dtype=TORCH_DTYPE_TO_TYPECLASS[obj.dtype],
                         strides=obj.stride(),
                         shape=tuple(obj.shape),
                         storage=storage)
        except ImportError:
            raise ValueError("Attempted to convert a torch.Tensor, but torch could not be imported")
    elif dtypes.is_array(obj) and (hasattr(obj, '__array_interface__') or hasattr(obj, '__cuda_array_interface__')):
        if dtypes.is_gpu_array(obj):
            interface = obj.__cuda_array_interface__
            storage = dtypes.StorageType.GPU_Global
        else:
            interface = obj.__array_interface__
            storage = dtypes.StorageType.Default

        if hasattr(obj, 'dtype') and obj.dtype.fields is not None:  # Struct
            dtype = dtypes.struct('unnamed', **{k: dtypes.typeclass(v[0].type) for k, v in obj.dtype.fields.items()})
        else:
            if np.dtype(interface['typestr']).type is np.void:  # Struct from __array_interface__
                if 'descr' in interface:
                    dtype = dtypes.struct('unnamed', **{
                        k: dtypes.typeclass(np.dtype(v).type)
                        for k, v in interface['descr']
                    })
                else:
                    raise TypeError(f'Cannot infer data type of array interface object "{interface}"')
            else:
                dtype = dtypes.typeclass(np.dtype(interface['typestr']).type)
        itemsize = np.dtype(interface['typestr']).itemsize
        if len(interface['shape']) == 0:
            return Scalar(dtype, storage=storage)
        return Array(dtype=dtype,
                     shape=interface['shape'],
                     strides=(tuple(s // itemsize for s in interface['strides']) if interface['strides'] else None),
                     storage=storage)
    elif isinstance(obj, (list, tuple)):
        # Lists and tuples are cast to numpy
        obj = np.array(obj)

        if obj.dtype.fields is not None:  # Struct
            dtype = dtypes.struct('unnamed', **{k: dtypes.typeclass(v[0].type) for k, v in obj.dtype.fields.items()})
        else:
            dtype = dtypes.typeclass(obj.dtype.type)
        return Array(dtype=dtype, strides=tuple(s // obj.itemsize for s in obj.strides), shape=obj.shape)
    elif type(obj).__module__ == "cupy" and type(obj).__name__ == "ndarray":
        # special case for CuPy and HIP, which does not support __cuda_array_interface__
        storage = dtypes.StorageType.GPU_Global
        dtype = dtypes.typeclass(obj.dtype.type)
        itemsize = obj.itemsize
        return Array(dtype=dtype, shape=obj.shape, strides=tuple(s // itemsize for s in obj.strides), storage=storage)
    elif symbolic.issymbolic(obj):
        return Scalar(symbolic.symtype(obj))
    elif isinstance(obj, dtypes.typeclass):
        return Scalar(obj)
    elif (obj is int or obj is float or obj is complex or obj is bool or obj is None):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, type) and issubclass(obj, np.number):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, (Number, np.number, np.bool_)):
        return Scalar(dtypes.typeclass(type(obj)))
    elif obj is type(None):
        # NoneType is void *
        return Scalar(dtypes.pointer(dtypes.typeclass(None)))
    elif isinstance(obj, str) or obj is str:
        return Scalar(dtypes.string)
    elif callable(obj):
        # Cannot determine return value/argument types from function object
        return Scalar(dtypes.callback(None))

    raise TypeError(f'Could not create a DaCe data descriptor from object {obj}. '
                    'If this is a custom object, consider creating a `__descriptor__` '
                    'adaptor method to the type hint or object itself.')


def make_array_from_descriptor(descriptor: Array,
                               original_array: Optional[ArrayLike] = None,
                               symbols: Optional[Dict[str, Any]] = None) -> ArrayLike:
    """
    Creates an array that matches the given data descriptor, and optionally copies another array to it.

    :param descriptor: The data descriptor to create the array from.
    :param original_array: An optional array to fill the content of the return value with.
    :param symbols: An optional symbol mapping between symbol names and their values. Used for creating arrays
                    with symbolic sizes.
    :return: A NumPy-compatible array (CuPy for GPU storage) with the specified size and strides.
    """
    symbols = symbols or {}

    free_syms = set(map(str, descriptor.free_symbols)) - symbols.keys()
    if free_syms:
        raise NotImplementedError(f'Cannot make Python references to arrays with undefined symbolic sizes: {free_syms}')

    if descriptor.storage == dtypes.StorageType.GPU_Global:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError):
            raise NotImplementedError('GPU memory can only be allocated in Python if cupy is installed')

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = cp.ndarray(shape=[total_size], dtype=dtype)
            view = cp.ndarray(shape=shape,
                              dtype=dtype,
                              memptr=buffer.data,
                              strides=[s * dtype.itemsize for s in strides])
            return view

        def copy_array(dst, src):
            dst[:] = cp.asarray(src)

    else:

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = np.ndarray([total_size], dtype=dtype)
            view = np.ndarray(shape, dtype, buffer=buffer, strides=[s * dtype.itemsize for s in strides])
            return view

        def copy_array(dst, src):
            dst[:] = src

    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    evaluated_shape = tuple(symbolic.evaluate(s, symbols) for s in descriptor.shape)
    evaluated_size = symbolic.evaluate(descriptor.total_size, symbols)
    evaluated_strides = tuple(symbolic.evaluate(s, symbols) for s in descriptor.strides)
    view = create_array(evaluated_shape, npdtype, evaluated_size, evaluated_strides)
    if original_array is not None:
        copy_array(view, original_array)

    return view


def make_reference_from_descriptor(descriptor: Array,
                                   original_array: ctypes.c_void_p,
                                   symbols: Optional[Dict[str, Any]] = None) -> ArrayLike:
    """
    Creates an array that matches the given data descriptor from the given pointer. Shares the memory
    with the argument (does not create a copy).

    :param descriptor: The data descriptor to create the array from.
    :param original_array: The array whose memory the return value would be used in.
    :param symbols: An optional symbol mapping between symbol names and their values. Used for referencing arrays
                    with symbolic sizes.
    :return: A NumPy-compatible array (CuPy for GPU storage) with the specified size and strides, sharing memory
             with the pointer specified in ``original_array``.
    """
    symbols = symbols or {}
    # Filter symbols to sympy symbols and constants
    symbols = {k: v for k, v in symbols.items() if isinstance(v, (Number, symbolic.sympy.Basic))}

    original_array: int = ctypes.cast(original_array, ctypes.c_void_p).value

    free_syms = set(map(str, descriptor.free_symbols)) - symbols.keys()
    if free_syms:
        raise NotImplementedError(f'Cannot make Python references to arrays with undefined symbolic sizes: {free_syms}')

    if descriptor.storage == dtypes.StorageType.GPU_Global:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError):
            raise NotImplementedError('GPU memory can only be referenced in Python if cupy is installed')

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = dtypes.ptrtocupy(original_array, descriptor.dtype.as_ctypes(), (total_size, ))
            view = cp.ndarray(shape=shape,
                              dtype=dtype,
                              memptr=buffer.data,
                              strides=[s * dtype.itemsize for s in strides])
            return view

    else:

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = dtypes.ptrtonumpy(original_array, descriptor.dtype.as_ctypes(), (total_size, ))
            view = np.ndarray(shape, dtype, buffer=buffer, strides=[s * dtype.itemsize for s in strides])
            return view

    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    evaluated_shape = tuple(symbolic.evaluate(s, symbols) for s in descriptor.shape)
    evaluated_size = symbolic.evaluate(descriptor.total_size, symbols)
    evaluated_strides = tuple(symbolic.evaluate(s, symbols) for s in descriptor.strides)
    return create_array(evaluated_shape, npdtype, evaluated_size, evaluated_strides)
