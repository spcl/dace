import numpy as np
from dace.data import Array
from typing import Any, Dict


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
        raise TypeError('Type %s not supported in BLAS operations' %
                        dtype.__name__)


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


def _get_matmul_inputs(node, state, sdfg):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_a = None
    res_b = None
    for edge in state.in_edges(node):
        if edge.dst_conn in ["_a", "_b"]:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [
                s for i, s in enumerate(outer_array.strides) if i in squeezed
            ]
            res = edge, outer_array, size, strides
            if edge.dst_conn == "_a":
                res_a = res
            else:
                res_b = res
    if res_a is None or res_b is None:
        raise ValueError("Matrix multiplication input connectors \"_a\" and "
                         "\"_b\" not found.")
    return res_a, res_b
