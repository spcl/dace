# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary

# Data set sizes
# M, N
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def syr2k_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N], A: dc.float64[N, M], B: dc.float64[N, M]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k])


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 3) % N) / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % M) / M, (N, M), dtype=datatype)

    return alpha, beta, C, A, B


def ground_truth(alpha, beta, C, A, B):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k])


def run_syr2k(device_type: dace.dtypes.DeviceType):
    '''
    Runs Syr2k for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A, B = initialize(M, N)
    C_ref = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = syr2k_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, C, A, B, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = syr2k_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # No libnodes expansion for this kernel
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(M=M, N=N))
        sdfg(alpha, beta, C, A, B)

    # Compute ground truth and validate
    ground_truth(alpha, beta, C_ref, A, B)
    assert np.allclose(C, C_ref)
    return sdfg


def test_cpu():
    run_syr2k(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_syr2k(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_syr2k(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_syr2k(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_syr2k(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_syr2k(dace.dtypes.DeviceType.FPGA)
