# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Ctypes interoperability for data descriptors.

This module contains functions for converting data descriptors to ctypes.
"""
import ctypes
import warnings

from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp

from dace import config, dtypes, symbolic


def make_ctypes_argument(arg: Any,
                         argtype: 'Data',
                         name: Optional[str] = None,
                         allow_views: Optional[bool] = None,
                         symbols: Optional[Dict[str, Any]] = None,
                         callback_retval_references: Optional[List[Any]] = None,
                         argument_to_pyobject: Optional[Dict[Any, Any]] = None) -> Any:
    """
    Converts a given argument to the expected ``ctypes`` type for passing to compiled SDFG functions.

    :param arg: The argument to convert.
    :param argtype: The expected data descriptor type of the argument.
    :param name: The name of the argument (for error messages).
    :param allow_views: Whether to allow views and references as input. If False, raises an error if a view or
                        reference is passed. If None (default), uses the global configuration setting
                        ``compiler.allow_view_arguments``.
    :param symbols: An optional symbol mapping between symbol names and their values. Used for evaluating symbolic
                    sizes in callback arguments.
    :param callback_retval_references: A list to store references to callback return values (to avoid garbage
                                       collection of said return values). This object must be kept alive until the
                                       SDFG call is complete.
    :param argument_to_pyobject: A dictionary to map ctypes arguments back to their original Python objects.
                                 If given, this function will update the dictionary with the mapping for the current argument.
    :return: The argument converted to the appropriate ctypes type.
    """
    # Import here to avoid circular imports
    from dace.data.core import Array, ContainerArray, Structure

    if allow_views is None:
        no_view_arguments = not config.Config.get_bool('compiler', 'allow_view_arguments')
    else:
        no_view_arguments = not allow_views
    a = name or '<unknown>'
    atype = argtype

    result = arg
    is_array = dtypes.is_array(arg)
    is_ndarray = isinstance(arg, np.ndarray)
    is_dtArray = isinstance(argtype, Array)
    if not is_array and is_dtArray:
        if isinstance(arg, list):
            print(f'WARNING: Casting list argument "{a}" to ndarray')
        elif arg is None:
            if atype.optional is False:  # If array cannot be None
                raise TypeError(f'Passing a None value to a non-optional array in argument "{a}"')
            # Otherwise, None values are passed as null pointers below
        elif isinstance(arg, ctypes._Pointer):
            pass
        elif isinstance(arg, str):
            # Cast to bytes
            result = ctypes.c_char_p(arg.encode('utf-8'))
        else:
            raise TypeError(f'Passing an object (type {type(arg).__name__}) to an array in argument "{a}"')
    elif is_array and not is_dtArray:
        # GPU scalars and return values are pointers, so this is fine
        if atype.storage != dtypes.StorageType.GPU_Global and not a.startswith('__return'):
            raise TypeError(f'Passing an array to a scalar (type {atype.dtype.ctype}) in argument "{a}"')
    elif (is_dtArray and is_ndarray and not isinstance(atype, ContainerArray)
          and atype.dtype.as_numpy_dtype() != arg.dtype):
        # Make exception for vector types
        if (isinstance(atype.dtype, dtypes.vector) and atype.dtype.vtype.as_numpy_dtype() == arg.dtype):
            pass
        else:
            print(f'WARNING: Passing {arg.dtype} array argument "{a}" to a {atype.dtype.type.__name__} array')
    elif is_dtArray and is_ndarray and arg.base is not None and not '__return' in a and no_view_arguments:
        raise TypeError(f'Passing a numpy view (e.g., sub-array or "A.T") "{a}" to DaCe '
                        'programs is not allowed in order to retain analyzability. '
                        'Please make a copy with "numpy.copy(...)". If you know what '
                        'you are doing, you can override this error in the '
                        'configuration by setting compiler.allow_view_arguments '
                        'to True.')
    elif (not isinstance(atype, (Array, Structure)) and not isinstance(atype.dtype, dtypes.callback)
          and not isinstance(arg, (atype.dtype.type, sp.Basic))
          and not (isinstance(arg, symbolic.symbol) and arg.dtype == atype.dtype)):
        is_int = isinstance(arg, int)
        if is_int and atype.dtype.type == np.int64:
            pass
        elif (is_int and atype.dtype.type == np.int32 and abs(arg) <= (1 << 31) - 1):
            pass
        elif (is_int and atype.dtype.type == np.uint32 and arg >= 0 and arg <= (1 << 32) - 1):
            pass
        elif isinstance(arg, float) and atype.dtype.type == np.float64:
            pass
        elif isinstance(arg, bool) and atype.dtype.type == np.bool_:
            pass
        elif (isinstance(arg, str) or arg is None) and atype.dtype == dtypes.string:
            if arg is None:
                result = ctypes.c_char_p(None)
            else:
                # Cast to bytes
                result = ctypes.c_char_p(arg.encode('utf-8'))
        else:
            warnings.warn(f'Casting scalar argument "{a}" from {type(arg).__name__} to {atype.dtype.type}')
            result = atype.dtype.type(arg)

    # Call a wrapper function to make NumPy arrays from pointers.
    if isinstance(argtype.dtype, dtypes.callback):
        result = argtype.dtype.get_trampoline(result, symbols or {}, callback_retval_references, argument_to_pyobject)
    # List to array
    elif isinstance(result, list) and isinstance(argtype, Array):
        result = np.array(result, dtype=argtype.dtype.type)
    # Null pointer
    elif result is None and isinstance(argtype, Array):
        result = ctypes.c_void_p(0)

    # Retain only the element datatype for upcoming checks and casts
    actype = argtype.dtype.as_ctypes()

    try:
        if dtypes.is_array(result):  # `c_void_p` is subclass of `ctypes._SimpleCData`.
            result = ctypes.c_void_p(dtypes.array_interface_ptr(result, atype.storage))
        elif not isinstance(result, (ctypes._SimpleCData, ctypes._Pointer)):
            result = actype(result)
        else:
            pass
    except TypeError as ex:
        raise TypeError(f'Invalid type for scalar argument "{a}": {ex}')

    # Map the ctypes argument back to the original Python object
    if argument_to_pyobject is not None:
        try:
            addr = result.value
        except AttributeError:
            addr = ctypes.addressof(result)
        argument_to_pyobject[addr] = arg

    return result
