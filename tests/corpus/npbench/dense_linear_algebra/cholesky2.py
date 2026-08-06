# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``cholesky2`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'N': 1000}
PAPER_SIZES = {'N': 2000}
INPUT_ARGS = ('N', )
ARRAY_ARGS = ('A', )
SCALARS = {}
OUTPUT_ARGS = ('A', )

N = dc.symbol('N', dtype=dc.int64)


def initialize(N, datatype=np.float64):
    A = np.zeros((N, N), dtype=datatype)
    for i in range(N):
        A[i, :i + 1] = np.fromfunction(lambda j: -j % N / N + 1, (i + 1, ), dtype=datatype)
        A[i, i + 1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    return A


def reference(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


# The strict upper triangle is written inline rather than by a nested ``triu`` program, so the map
# bound is the literal ``i + 1``. A nested program cannot carry the offset soundly today: passing it
# as a scalar PARAMETER emits a free symbol literally named ``scalar`` that the call site never
# binds, and canonicalize then resolves that unbound symbol to a DATA CONTAINER -- the map range
# became ``i + <array name>`` and codegen emitted ``double* + double*``. Inlining keeps the map
# bound a compile-time constant and needs no symbol promotion at all.
@dc.program
def kernel(A: dc_float[N, N]):
    B = np.zeros_like(A)
    for i in dc.map[0:N]:
        for j in dc.map[i + 1:N]:
            B[i, j] = A[i, j]
    A[:] = np.linalg.cholesky(A) + B


CORPUS = dict(name='cholesky2',
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
