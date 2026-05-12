# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import scipy
import tests.codegen.sve.common as common
import pytest

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


@dace.program
def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
         b: dace.float32[H]):

    @dace.mapscope(_[0:H])
    def compute_row(i):

        @dace.map(_[A_row[i]:A_row[i + 1]])
        def compute(j):
            a << A_val[j]
            in_x << x[A_col[j]]
            out >> b(1, lambda x, y: x + y)[i]

            out = a * in_x


@pytest.mark.sve
def test_spmv():
    W = 64
    H = 64
    nnz = 640

    print('Sparse Matrix-Vector Multiplication %dx%d (%d non-zero elements)' % (W, H, nnz))

    A_row = dace.ndarray([H + 1], dtype=dace.uint32)
    A_col = dace.ndarray([nnz], dtype=dace.uint32)
    A_val = dace.ndarray([nnz], dtype=dace.float32)

    x = dace.ndarray([W], dace.float32)
    b = dace.ndarray([H], dace.float32)

    # Assuming uniform sparsity distribution across rows
    nnz_per_row = nnz // H
    nnz_last_row = nnz_per_row + (nnz % H)
    if nnz_last_row > W:
        print('Too many nonzeros per row')
        exit(1)

    # RANDOMIZE SPARSE MATRIX
    A_row[0] = dace.uint32(0)
    A_row[1:H] = dace.uint32(nnz_per_row)
    A_row[-1] = dace.uint32(nnz_last_row)
    A_row = np.cumsum(A_row, dtype=np.uint32)

    # Fill column data
    for i in range(H - 1):
        A_col[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(W, nnz_per_row, replace=False))
    # Fill column data for last row
    A_col[nnz_per_row * (H - 1):] = np.sort(np.random.choice(W, nnz_last_row, replace=False))

    A_val[:] = np.random.rand(nnz).astype(dace.float32.type)
    #########################

    x[:] = np.random.rand(W).astype(dace.float32.type)
    b[:] = dace.float32(0)

    # Setup regression
    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), shape=(H, W))

    sdfg = common.vectorize(spmv)

    sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=H, W=W, nnz=nnz)

    if dace.Config.get_bool('profiling'):
        dace.timethis('spmv', 'scipy', 0, A_sparse.dot, x)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(H)
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5
