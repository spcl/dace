# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest
import scipy

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


@dace.program(dace.uint32[H + 1], dace.uint32[nnz], dace.float32[nnz], dace.float32[W], dace.float32[H])
def spmv(A_row, A_col, A_val, x, b):
    @dace.mapscope(_[0:H])
    def compute_row(i):
        @dace.map(_[A_row[i]:A_row[i + 1]])
        def compute(j):
            a << A_val[j]
            in_x << x[A_col[j]]
            out >> b(1, lambda x, y: x + y)[i]

            out = a * in_x


@pytest.mark.gpu
def test_dynamic_map():
    height = 1024
    width = 1024

    # Prepare spmv SDFG for GPU
    sdfg = spmv.to_sdfg()
    sdfg.apply_gpu_transformations()

    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential:
            node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic

    # Fill input data
    # each row has up (including) 256 elements
    A_row = np.random.randint(257, size=height + 1, dtype=dace.uint32.type)
    A_row[0] = 0
    A_row = np.cumsum(A_row, dtype=dace.uint32.type)

    # Column data
    A_col = dace.ndarray([A_row[height]], dtype=dace.uint32)
    for i in range(height):
        A_col[A_row[i]:A_row[i + 1]] = np.sort(np.random.choice(width, A_row[i + 1] - A_row[i], replace=False))

    # values
    A_val = np.random.rand(A_row[height]).astype(dace.float32.type)

    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), dtype=dace.float32.type, shape=(1024, 1024))

    x = np.random.rand(width).astype(dace.float32.type)
    b = np.zeros(height, dtype=dace.float32.type)

    sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=A_sparse.shape[0], W=A_sparse.shape[1], nnz=A_sparse.nnz)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(height)
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


if __name__ == '__main__':
    test_dynamic_map()
