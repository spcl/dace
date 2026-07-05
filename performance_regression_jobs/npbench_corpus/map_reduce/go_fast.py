# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``go_fast`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'N': 2000}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('a', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

N = dc.symbol('N', dtype=dc.int64)


def initialize(N, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, N), dtype=datatype)
    out = np.zeros((N, N), dtype=datatype)
    return (x, out)


def reference(a, out):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    out[:] = a + trace


@dc.program
def kernel(a: dc_float[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace


CORPUS = dict(name='go_fast',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
