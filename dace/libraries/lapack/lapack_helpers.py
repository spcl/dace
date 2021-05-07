# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from dace import dtypes
from typing import Any, Dict, Tuple


def to_lapacktype(dtype):
    """ Returns a LAPACK character that corresponds to the input type.
        Used in MKL/OpenBLAS/CUDATOOLKIT calls. """

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
        raise TypeError('Type %s not supported in LAPACK operations' %
                        dtype.__name__)


def cuda_type_metadata(dtype: dtypes.typeclass) -> Tuple[str, str, str]:
    """ 
    Returns type metadata on a given dace dtype. 
    :return: A 3 tuple of (LAPACK letter, CUDA C type, Name in dace runtime).
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
        raise TypeError('Type %s not supported in LAPACK operations' %
                        dtype.__name__)
