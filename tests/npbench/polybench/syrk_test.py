# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
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

# M, N
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, C: dc.float32[N, N], A: dc.float32[N, M]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def init_data(N, M):

    alpha = np.float32(1.5)
    beta = np.float32(1.2)
    C = np.empty((N, N), dtype=np.float32)
    A = np.empty((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            A[i, j] = ((i * j + 1) % N) / N
    for i in range(N):
        for j in range(N):
            C[i, j] = ((i * j + 2) % M) / M

    return alpha, beta, C, A


def ground_truth(N, M, alpha, beta, C, A):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def run_syrk(device_type: dace.dtypes.DeviceType):
    '''
    Runs Syrk for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A = init_data(N, M)
    gt_C = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha=alpha, beta=beta, C=C, A=A, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)
        sdfg.specialize(dict(N=N, M=M))
        # run program
        sdfg(alpha=alpha, beta=beta, C=C, A=A)

    # Compute ground truth and validate result
    ground_truth(N, M, alpha, beta, gt_C, A)
    assert np.allclose(C, gt_C)
    return sdfg


def test_cpu():
    run_syrk(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_syrk(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_syrk(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_syrk(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_syrk(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_syrk(dace.dtypes.DeviceType.FPGA)
