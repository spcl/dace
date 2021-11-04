# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def kernel(A: dc.float32[N, N]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


def init_data(N):
    A = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            A[i, j] = (-j % N) / N + 1
        for j in range(i + 1, N):
            A[i, j] = 0.0
        A[i, i] = 1.0

    A[:] = A @ np.transpose(A)
    return A


def ground_truth(N, A):
    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


def run_cholesky(device_type: dace.dtypes.DeviceType):
    '''
    Runs Cholesky for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench medium size)
    N = 400
    A = init_data(N)
    gt_A = np.copy(A)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A=A, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(strict=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        platform = dace.config.Config.get("compiler", "fpga" ,"vendor")
        if platform == "intel_fpga":
            Dot.default_implementation = "FPGA_Accumulate"
        else:
            Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

        sdfg(A=A, N=N)

    # Compute ground truth and validate result
    ground_truth(N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    assert diff < 1e-6
    return sdfg


def test_cpu():
    run_cholesky(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_cholesky(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_cholesky(dace.dtypes.DeviceType.FPGA)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--target",
        default='cpu',
        choices=['cpu', 'gpu', 'fpga'],
        help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_cholesky(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_cholesky(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_cholesky(dace.dtypes.DeviceType.FPGA)
