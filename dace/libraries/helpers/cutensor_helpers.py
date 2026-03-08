# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helper functions for cuTENSOR data types and enumerations.
Based on cuTENSOR 1.7.0 documentation:
https://docs.nvidia.com/cuda/cutensor/1.7.0/api/types.html
"""
import numpy as np
from dace import dtypes, data
from dace.data import Array
from typing import Any, Dict, Tuple


def dtype_to_cutensordatatype(dtype: dtypes.typeclass) -> str:
    """
    Returns the cuTENSOR data type enum string (cudaDataType_t) corresponding
    to the given dace data type.

    cuTENSOR supports the following tensor data types (as of v1.7.0):
        - FP16, BF16, FP32, FP64
        - Complex FP32, Complex FP64
    Integer types are not supported for tensor storage.

    :param dtype: A dace typeclass instance.
    :return: A string like 'CUTENSOR_COMPUTE_16F', 'CUTENSOR_COMPUTE_32F', etc.
    :raises TypeError: If the dtype is not supported by cuTENSOR.
    """
    mapping = {
        dtypes.float16: 'CUTENSOR_COMPUTE_16F',
        dtypes.float32: 'CUTENSOR_COMPUTE_32F',
        dtypes.float64: 'CUTENSOR_COMPUTE_64F',
        dtypes.complex64: 'CUTENSOR_COMPUTE_32F',
        dtypes.complex128: 'CUTENSOR_COMPUTE_64F',
        dtypes.uint8: 'CUTENSOR_COMPUTE_8U',
        dtypes.int8: 'CUTENSOR_COMPUTE_8I',
        dtypes.uint32: 'CUTENSOR_COMPUTE_32U',
        dtypes.int32: 'CUTENSOR_COMPUTE_32I',
    }
    if dtype not in mapping:
        raise TypeError(f'cuTENSOR does not support tensor data type: {dtype}')
    return mapping[dtype]


def to_cutensor_computetype(dtype: dtypes.typeclass) -> str:
    """
    Returns the cuTENSOR compute type suffix (cutensorComputeType_t)
    corresponding to the given dace data type.

    Compute types determine the precision used for internal arithmetic.
    They may differ from the tensor storage type. cuTENSOR supports
    floating-point and integer compute types (as of v1.7.0).

    :param dtype: A dace typeclass instance.
    :return: A string like '16F', '32I', etc., suitable for constructing
             the full enum, e.g., 'CUTENSOR_COMPUTE_32F'.
    :raises TypeError: If the dtype has no corresponding compute type.
    """
    mapping = {
        # Floating-point
        dtypes.float16: '16F',
        dtypes.float32: '32F',
        dtypes.float64: '64F',
        dtypes.complex64: '32F',           # complex uses same compute as real
        dtypes.complex128: '64F',
        # Integer (signed and unsigned)
        dtypes.uint8: '8U',
        dtypes.int8: '8I',
        dtypes.uint32: '32U',
        dtypes.int32: '32I',
        # Additional types may be added as needed (e.g., 16-bit integers)
    }
    if dtype not in mapping:
        raise TypeError(f'No cuTENSOR compute type defined for {dtype}')
    return mapping[dtype]


def cutensor_type_metadata(dtype: dtypes.typeclass) -> Tuple[str, str, str]:
    """
    Returns type metadata for cuTENSOR operations.

    :param dtype: A dace typeclass instance.
    :return: A 3-tuple of:
             - cuTENSOR compute type suffix (str)
             - CUDA C type name (str)
             - Human-readable name (str)
    :raises TypeError: If the dtype is not supported.
    """
    # Compute type suffix (for CUTENSOR_COMPUTE_*)
    compute_suffix = to_cutensor_computetype(dtype)

    # CUDA C type names for kernel arguments
    cuda_c_type = {
        dtypes.float16: '__half',
        dtypes.float32: 'float',
        dtypes.float64: 'double',
        dtypes.complex64: 'cuComplex',
        dtypes.complex128: 'cuDoubleComplex',
        dtypes.uint8: 'uint8_t',
        dtypes.int8: 'int8_t',
        dtypes.uint32: 'uint32_t',
        dtypes.int32: 'int32_t',
    }.get(dtype, None)

    if cuda_c_type is None:
        raise TypeError(f'No CUDA C type mapping for {dtype}')

    # Human-readable name
    type_name = {
        dtypes.float16: 'Half',
        dtypes.float32: 'Float',
        dtypes.float64: 'Double',
        dtypes.complex64: 'Complex64',
        dtypes.complex128: 'Complex128',
        dtypes.uint8: 'Uint8',
        dtypes.int8: 'Int8',
        dtypes.uint32: 'Uint32',
        dtypes.int32: 'Int32',
    }.get(dtype, None)

    if type_name is None:
        raise TypeError(f'No type name for {dtype}')

    return compute_suffix, cuda_c_type, type_name


# Optional: utility to get alignment or workspace preferences if needed.
def cutensor_get_alignment(dtype: dtypes.typeclass) -> int:
    """
    Returns the required alignment in bytes for a tensor of the given dtype
    in cuTENSOR operations.
    """
    # Usually 16-byte alignment is recommended for performance.
    # But we can return the size of the type as a safe default.
    return dtype.bytes
