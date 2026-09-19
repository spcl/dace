# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``permute_3d`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'N': 128}
#: This kernel is local to this corpus, not an npbench benchmark, so there is no upstream paper row
#: to copy. At N=128 the whole transpose runs in tens of microseconds, below OpenMP fork/join, so
#: the measurement was of the runtime rather than of the code. 512 rather than 1000 because the
#: arrays are (N, N, N): 512 gives 1.1 GB each and about 3.2 GB for the pair plus the reference's
#: transpose, where 1000 would want 8 GB each and 24 GB on the one rank that draws this kernel.
PAPER_SIZES = {'N': 512}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('A', 'B')
SCALARS = {}
OUTPUT_ARGS = ('B', )

N = dc.symbol('N', dtype=dc.int64)


def initialize(N, datatype=np.float64):
    # Broadcast rather than np.fromfunction: fromfunction hands the lambda three FULL (N, N, N)
    # index grids, so at the paper size it wants 24 GB of temporaries to fill an 8 GB array. These
    # reshaped ranges broadcast to the same values, materializing (N, N, N) once, and the division
    # is in place so there is no second copy.
    i = np.arange(N, dtype=datatype).reshape(N, 1, 1)
    j = np.arange(N, dtype=datatype).reshape(1, N, 1)
    k = np.arange(N, dtype=datatype).reshape(1, 1, N)
    A = i * (N * N) + j * N + k * N
    A /= N
    B = np.zeros((N, N, N), dtype=datatype)
    return (A, B)


def reference(A, B):
    """B[i, j, k] = A[k, j, i] -- swap the first and last axes."""
    B[:] = np.transpose(A, (2, 1, 0))


@dc.program
def kernel(A: dc_float[N, N, N], B: dc_float[N, N, N]):
    for i, j, k in dc.map[0:N, 0:N, 0:N]:
        B[i, j, k] = A[k, j, i]


CORPUS = dict(name='permute_3d',
              dwarf='dense_linear_algebra',
              sizes=SIZES,
              paper_sizes=PAPER_SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
