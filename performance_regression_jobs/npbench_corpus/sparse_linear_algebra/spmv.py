# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``spmv`` (sparse_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'M': 4096, 'N': 4096, 'nnz': 8192}
INPUT_ARGS = ('M', 'N', 'nnz')
# initialize returns the CSR triplet (indptr, indices, data) then x, y -- name them
# so both the numpy reference (A_data/A_indices/A_indptr/x/y) and the dace kernel
# resolve their parameters by name.
ARRAY_ARGS = ('A_indptr', 'A_indices', 'A_data', 'x', 'y')
SCALARS = {}
OUTPUT_ARGS = ('y', )

M, N, nnz = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'nnz'))


def initialize(M, N, nnz, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, ), dtype=datatype)
    from scipy.sparse import random
    matrix = random(M, N, density=nnz / (M * N), format='csr', dtype=datatype, random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data
    y = np.zeros(M, dtype=datatype)
    return (rows, cols, vals, x, y)


def reference(A_data, A_indices, A_indptr, x, y):
    M = A_indptr.shape[0] - 1
    for i in range(M):
        cols = A_indices[A_indptr[i]:A_indptr[i + 1]]
        vals = A_data[A_indptr[i]:A_indptr[i + 1]]
        y[i] = vals @ x[cols]


@dc.program
def kernel(A_data: dc_float[nnz], A_indices: dc.uint32[nnz], A_indptr: dc.uint32[M + 1], x: dc_float[N]):
    y = np.empty(M, A_data.dtype)
    for i in range(M):
        start = dc.define_local_scalar(dc.uint32)
        stop = dc.define_local_scalar(dc.uint32)
        start = A_indptr[i]
        stop = A_indptr[i + 1]
        cols = A_indices[start:stop]
        vals = A_data[start:stop]
        y[i] = vals @ x[cols]
    return y


CORPUS = dict(name='spmv',
              dwarf='sparse_linear_algebra',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
