# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

M, N, nnz = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'nnz'))


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@dc.program
def spmv_kernel(A_row: dc.uint32[M + 1], A_col: dc.uint32[nnz], A_val: dc.float64[nnz], x: dc.float64[N]):
    y = np.empty(M, A_val.dtype)

    for i in range(M):
        start = dc.define_local_scalar(dc.uint32)
        stop = dc.define_local_scalar(dc.uint32)
        start = A_row[i]
        stop = A_row[i + 1]
        cols = A_col[start:stop]
        vals = A_val[start:stop]
        y[i] = vals @ x[cols]

    return y


def initialize(M, N, nnz):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    from scipy.sparse import random

    matrix = random(M, N, density=nnz / (M * N), format='csr', dtype=np.float64, random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data

    return rows, cols, vals, x


def ground_truth(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


def run_spmv(device_type: dace.dtypes.DeviceType):
    '''
    Runs SPMV for the given device
    :return: the SDFG
    '''

    # Initialize data (custom size)
    M, N, nnz = 2048, 2048, 4096
    A_rows, A_cols, A_vals, x = initialize(M, N, nnz)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = spmv_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        y = sdfg(A_rows, A_cols, np.copy(A_vals), x, M=M, N=N, nnz=nnz)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = spmv_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

        sdfg.specialize(dict(M=M, N=N, nnz=nnz))
        y = sdfg(A_rows, A_cols, np.copy(A_vals), x)

    # Compute ground truth and Validate result
    y_ref = ground_truth(A_rows, A_cols, A_vals, x)
    assert np.allclose(y, y_ref)
    return sdfg


def test_cpu():
    run_spmv(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_spmv(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Missing free symbol")
@fpga_test(assert_ii_1=False)
def test_fpga():
    run_spmv(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_spmv(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_spmv(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_spmv(dace.dtypes.DeviceType.FPGA)