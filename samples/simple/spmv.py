# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple program showing the `dace.map` syntax and profiling. """

import argparse
import dace
import numpy as np
try:
    import scipy.sparse as sp
except (ImportError, ModuleNotFoundError):
    print('This sample requires scipy to validate. Validation will be disabled')
    sp = None

# Define sparse array sizes
W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


# Define dace program with type hints to enable Ahead-Of-Time compilation
@dace.program
def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W]):
    b = np.zeros([H], dtype=np.float32)

    for i in dace.map[0:H]:
        for j in dace.map[A_row[i]:A_row[i + 1]]:
            b[i] += A_val[j] * x[A_col[j]]

    return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-W", type=int, nargs="?", default=64)
    parser.add_argument("-H", type=int, nargs="?", default=64)
    parser.add_argument("-nnz", type=int, nargs="?", default=640)
    args = parser.parse_args()

    print(f'Sparse Matrix-Vector Multiplication {args.H}x{args.W} ({args.nnz} non-zero elements)')

    # Setup inputs
    A_row = np.empty([args.H + 1], dtype=np.uint32)
    A_col = np.empty([args.nnz], dtype=np.uint32)
    A_val = np.random.rand(args.nnz).astype(np.float32)
    x = np.random.rand(args.W).astype(np.float32)

    # Assuming uniform sparsity distribution across rows
    nnz_per_row = args.nnz // args.H
    nnz_last_row = nnz_per_row + (args.nnz % args.H)
    if nnz_last_row > args.W:
        print('Too many nonzeros per row')
        exit(1)

    # Randomize sparse matrix structure
    A_row[0] = dace.uint32(0)
    A_row[1:args.H] = dace.uint32(nnz_per_row)
    A_row[-1] = dace.uint32(nnz_last_row)
    A_row = np.cumsum(A_row, dtype=np.uint32)

    # Fill column data
    for i in range(args.H - 1):
        A_col[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(args.W, nnz_per_row, replace=False))
    # Fill column data for last row
    A_col[nnz_per_row * (args.H - 1):] = np.sort(np.random.choice(args.W, nnz_last_row, replace=False))

    #########################

    # Run program
    b = spmv(A_row, A_col, A_val, x)

    # Check for correctness
    if sp is not None:
        A_sparse = sp.csr_matrix((A_val, A_col, A_row), shape=(args.H, args.W))
        expected = A_sparse.dot(x)
        diff = np.linalg.norm(expected - b) / float(args.H)
        print("Difference:", diff)
