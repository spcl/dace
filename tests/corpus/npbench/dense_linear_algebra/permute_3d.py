# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``permute_3d`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'N': 128}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('A', 'B')
SCALARS = {}
OUTPUT_ARGS = ('B', )

N = dc.symbol('N', dtype=dc.int64)


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i * N * N + j * N + k * N) / N, (N, N, N), dtype=datatype)
    B = np.zeros((N, N, N), dtype=datatype)
    return (A, B)


def reference(A, B):
    """B[i, j, k] = A[k, j, i] — swap the first and last axes."""
    B[:] = np.transpose(A, (2, 1, 0))


@dc.program
def kernel(A: dc_float[N, N, N], B: dc_float[N, N, N]):
    for i, j, k in dc.map[0:N, 0:N, 0:N]:
        B[i, j, k] = A[k, j, i]


CORPUS = dict(name='permute_3d',
              dwarf='dense_linear_algebra',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
