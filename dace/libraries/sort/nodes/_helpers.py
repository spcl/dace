# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Internal helpers for the :mod:`dace.libraries.sort` library nodes."""
import dace

_INTEGER_DTYPES = {
    dace.int8,
    dace.int16,
    dace.int32,
    dace.int64,
    dace.uint8,
    dace.uint16,
    dace.uint32,
    dace.uint64,
}


def is_integer_dtype(dtype: dace.dtypes.typeclass) -> bool:
    """``True`` if ``dtype`` is one of DaCe's integer ``typeclass`` singletons."""
    return dtype in _INTEGER_DTYPES
