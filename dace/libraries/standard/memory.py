# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helper functions for memory movements
"""

# ---------- ----------
# NUMPY
# ---------- ----------
def aligned_ndarray(arr, alignment=64):
    """
    Allocates a and returns a copy of ``arr`` as an ``alignment``-byte aligned
    array. Useful for aligned vectorized access.
    
    Based on https://stackoverflow.com/a/20293172/6489142
    """
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Using aligned_ndarray requires numpy to be installed.')

    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment // arr.itemsize
    buf = np.empty(arr.size + extra, dtype=arr.dtype)
    ofs = (-buf.ctypes.data % alignment) // arr.itemsize
    result = buf[ofs:ofs + arr.size].reshape(arr.shape)
    np.copyto(result, arr)
    assert (result.ctypes.data % alignment) == 0
    return result