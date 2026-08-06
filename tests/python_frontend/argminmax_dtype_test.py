# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.argmax``/``np.argmin`` are the only frontend constructs that build a ``dtypes.struct`` dtype
(the ``_val_and_idx`` pair the reduction carries), so they are where a struct that cannot stringify
takes down a parse."""
import numpy as np

import dace
from dace import dtypes

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def argmax_kernel(a: dace.float64[N], b: dace.int64[1]):
    b[0] = np.argmax(a)


def test_argminmax_struct_dtype_stringifies():
    sdfg = argmax_kernel.to_sdfg(simplify=False)
    structs = [desc.dtype for desc in sdfg.arrays.values() if isinstance(desc.dtype, dtypes.struct)]
    assert structs, 'np.argmax no longer builds a struct dtype -- this test guards the wrong construct'
    for dtype in structs:
        assert dtype.to_string() == dtype.name


def test_struct_pointer_vector_stringify():
    assert dtypes.struct('_val_and_idx', idx=dace.int32, val=dace.float64).to_string() == '_val_and_idx'
    # Both wrap another typeclass without going through ``typeclass.__init__``.
    assert dtypes.pointer(dace.float32).to_string()
    assert dtypes.vector(dace.float32, 4).to_string()


def test_argmax_matches_numpy():
    a = np.random.rand(64)
    b = np.zeros(1, dtype=np.int64)
    argmax_kernel(a=a, b=b, N=64)
    assert b[0] == np.argmax(a)


if __name__ == '__main__':
    test_argminmax_struct_dtype_stringifies()
    test_struct_pointer_vector_stringify()
    test_argmax_matches_numpy()
