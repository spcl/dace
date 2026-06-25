# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``compute`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'M': 2000, 'N': 2000}
INPUT_ARGS = ('M', 'N')
ARRAY_ARGS = ('array_1', 'array_2', 'a', 'b', 'c', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


def initialize(M, N, datatype=np.int64):
    from numpy.random import default_rng
    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    out = np.empty((M, N), dtype=np.int64)
    return (array_1, array_2, a, b, c, out)


def reference(array_1, array_2, a, b, c, out):
    out[:] = np.clip(array_1, 2, 10) * a + array_2 * b + c


@dc.program
def kernel(array_1: dc.int64[M, N], array_2: dc.int64[M, N], a: dc.int64, b: dc.int64, c: dc.int64, out: dc.int64[M,
                                                                                                                  N]):
    out[:] = np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c


CORPUS = dict(name='compute',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
