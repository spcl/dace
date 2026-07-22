# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``covariance2`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'M': 500, 'N': 600}
INPUT_ARGS = ('M', 'N')
ARRAY_ARGS = ('float_n', 'data', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


def initialize(M, N, datatype=np.float32):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: i * j / M, (N, M), dtype=datatype)
    out = np.zeros((M, M), dtype=datatype)
    return (float_n, data, out)


def reference(M, float_n, data, out):
    mean = np.mean(data, axis=0)
    centered = data - mean
    out[:] = np.transpose(centered) @ centered / (float_n - 1.0)


@dc.program
def kernel(float_n: dc_float, data: dc_float[N, M]):
    """DaCe equivalent of `np.cov(np.transpose(data))` (the numpy
    covariance2 reference): mean-center along axis 0, then compute
    `data.T @ data / (float_n - 1.0)`. M is resolved symbolically from
    the data shape (mirrors covariance_dace.py).
    """
    mean = np.mean(data, axis=0)
    centered = data - mean
    return centered.T @ centered / (float_n - 1.0)


CORPUS = dict(name='covariance2',
              dwarf='dense_linear_algebra',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
