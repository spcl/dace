# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusion, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def lu_kernel(A: dc.float32[N, N]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def ground_truth(N, A):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def init_data(N):

    A = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            A[i, j] = (-j % N) / N + 1
        for j in range(i + 1, N):
            A[i, j] = 0.0
        A[i, i] = 1.0

    B = np.empty((N, N), dtype=np.float32)
    B[:] = A @ np.transpose(A)
    A[:] = B

    return A


def run_lu(device_type: dace.dtypes.DeviceType):
    '''
    Runs LU for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = 40
    A = init_data(N)
    gt_A = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = lu_kernel.to_sdfg()
        dace_res = sdfg(A=A, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = lu_kernel.to_sdfg(coarsen=True)

        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        platform = dace.config.Config.get("compiler", "fpga", "vendor")
        if platform == "intel_fpga":
            Dot.default_implementation = "FPGA_Accumulate"
        else:
            Dot.default_implementation = "FPGA_PartialSums"

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)
        fpga_auto_opt.fpga_global_to_local(sdfg)

        sdfg.specialize(dict(N=N))
        dace_res = sdfg(A=A)

    # Compute ground truth and validate result
    ground_truth(N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    assert diff < 1e-5
    return sdfg


def test_cpu():
    run_lu(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_lu(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_lu(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_lu(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_lu(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_lu(dace.dtypes.DeviceType.FPGA)
