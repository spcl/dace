# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from dace import dtypes, data
from dace.data import Array
from typing import Any, Dict, Tuple


def to_blastype(dtype):
    """ Returns a BLAS character that corresponds to the input type.
        Used in MKL/CUBLAS calls. """

    if dtype == np.float16:
        return 'H'
    elif dtype == np.float32:
        return 'S'
    elif dtype == np.float64:
        return 'D'
    elif dtype == np.complex64:
        return 'C'
    elif dtype == np.complex128:
        return 'Z'
    else:
        raise TypeError('Type %s not supported in BLAS operations' % dtype.__name__)


def cublas_type_metadata(dtype: dtypes.typeclass) -> Tuple[str, str, str]:
    """ 
    Returns type metadata on a given dace dtype. 
    
    :return: A 3 tuple of (BLAS letter, CUDA C type, Name in dace runtime).
    """

    if dtype == dtypes.float16:
        return 'H', '__half', 'Half'
    elif dtype == dtypes.float32:
        return 'S', 'float', 'Float'
    elif dtype == dtypes.float64:
        return 'D', 'double', 'Double'
    elif dtype == dtypes.complex64:
        return 'C', 'cuComplex', 'Complex64'
    elif dtype == dtypes.complex128:
        return 'Z', 'cuDoubleComplex', 'Complex128'
    else:
        raise TypeError('Type %s not supported in BLAS operations' % str(dtype))


def dtype_to_cudadatatype(dtype: dtypes.typeclass) -> str:
    types = {
        dtypes.float16: 'CUDA_R_16F',
        dtypes.float32: 'CUDA_R_32F',
        dtypes.complex64: 'CUDA_C_32F',
        dtypes.float64: 'CUDA_R_64F',
        dtypes.complex128: 'CUDA_C_64F',
        dtypes.int8: 'CUDA_R_8I',
        dtypes.uint8: 'CUDA_R_8U',
        dtypes.int32: 'CUDA_R_32I',
        #dtypes.complex32: 'CUDA_C_16F',
        #dtypes.bfloat16: 'CUDA_R_16BF',
        #dtypes.bcomplex32: 'CUDA_C_16BF',
        #dtypes.icomplex16: 'CUDA_C_8I',
        #dtypes.iucomplex16: 'CUDA_C_8U',
        #dtypes.icomplex64: 'CUDA_C_32I',
    }
    return types[dtype]


def to_cublas_computetype(dtype: dtypes.typeclass) -> str:
    types = {
        dtypes.float16: '16F',
        dtypes.float32: '32F',
        dtypes.float64: '64F',
        dtypes.complex64: '32F',
        dtypes.complex128: '64F',
        dtypes.int8: '32I',
        dtypes.int16: '32I',
        dtypes.int32: '32I',
        dtypes.int64: '32I',
        dtypes.uint8: '32I',
        dtypes.uint16: '32I',
        dtypes.uint32: '32I',
        dtypes.uint64: '32I',
    }
    return types[dtype]


def get_gemm_opts(a_strides, b_strides, c_strides) -> Dict[str, Any]:
    """
    Returns GEMM argument order, transposition, and leading dimensions
    based on column-major storage from dace arrays.

    :param a: Data descriptor for the first matrix.
    :param b: Data descriptor for the second matrix.
    :param c: Data descriptor for the output matrix.
    :return: A dictionary with the following keys: swap (if True, a and b
             should be swapped); lda, ldb, ldc (leading dimensions); ta, tb
             (whether GEMM should be called with OP_N or OP_T).
    """
    # possible order (C, row based) of dimensions in input array
    # and computed result based on
    # 1. N/T - transpose flag in cublas
    # 2. LR/RL - order in which A and B are passed into cublas
    #     k m, n k -> n m (LR, N, N)
    #     m k, n k -> n m (LR, T, N)
    #     k m, k n -> n m (LR, N, T)
    #     m k, k n -> n m (LR, T, T)
    #     m k, k n -> m n (RL, N, N)
    #     m k, n k -> m n (RL, N, T)
    #     k m, k n -> m n (RL, T, N)
    #     k m, n k -> m n (RL, T, T)
    #       |    |      |
    #     use these 3 to detect correct option

    sAM, sAK = a_strides[-2:]
    sBK, sBN = b_strides[-2:]
    sCM, sCN = c_strides[-2:]

    opts = {
        'mkm': {
            'swap': False,
            'lda': sAK,
            'ldb': sBN,
            'ldc': sCN,
            'ta': 'N',
            'tb': 'N'
        },
        'kkm': {
            'swap': False,
            'lda': sAM,
            'ldb': sBN,
            'ldc': sCN,
            'ta': 'T',
            'tb': 'N'
        },
        'mnm': {
            'swap': False,
            'lda': sAK,
            'ldb': sBK,
            'ldc': sCN,
            'ta': 'N',
            'tb': 'T'
        },
        'knm': {
            'swap': False,
            'lda': sAM,
            'ldb': sBK,
            'ldc': sCN,
            'ta': 'T',
            'tb': 'T'
        },
        'knn': {
            'swap': True,
            'lda': sAM,
            'ldb': sBK,
            'ldc': sCM,
            'ta': 'N',
            'tb': 'N'
        },
        'kkn': {
            'swap': True,
            'lda': sAM,
            'ldb': sBN,
            'ldc': sCM,
            'ta': 'N',
            'tb': 'T'
        },
        'mnn': {
            'swap': True,
            'lda': sAK,
            'ldb': sBK,
            'ldc': sCM,
            'ta': 'T',
            'tb': 'N'
        },
        'mkn': {
            'swap': True,
            'lda': sAK,
            'ldb': sBN,
            'ldc': sCM,
            'ta': 'T',
            'tb': 'T'
        },
    }

    if sAM == 1:
        optA = 'm'
    elif sAK == 1:
        optA = 'k'
    else:
        raise Exception("sAM or sAK should be 1")

    if sBN == 1:
        optB = 'n'
    elif sBK == 1:
        optB = 'k'
    else:
        raise Exception("sBK or sBN should be 1")

    if sCM == 1:
        optC = 'm'
    elif sCN == 1:
        optC = 'n'
    else:
        raise Exception("sCM or sCN should be 1")

    return opts[optA + optB + optC]


def check_access(schedule: dtypes.ScheduleType, *descs: data.Data):
    """ If schedule cannot access all passed descriptors, through an error.

        :param schedule: the schedule.
        :param descs: the descriptors to check.
    """
    for desc in descs:
        if not dtypes.can_access(schedule, desc.storage):
            raise ValueError(f"Schedule mismatch: {schedule} cannot access {desc.storage}")
