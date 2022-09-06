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
# N
sizes = {"mini": 30, "small": 90, "medium": 250, "large": 1300, "extra-large": 2800}

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def gesummv_kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], B: dc.float64[N, N], x: dc.float64[N]):

    return alpha * A @ x + beta * B @ x


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return alpha, beta, A, B, x


def run_gesummv(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gesummv for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["small"]
    alpha, beta, A, B, x = initialize(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gesummv_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        C = sdfg(alpha, beta, A, B, x, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gesummv_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        C = sdfg(alpha, beta, A, B, x)

    # Compute ground truth and validate
    C_ref = gesummv_kernel.f(alpha, beta, A, B, x)
    assert np.allclose(C, C_ref)
    return sdfg


def test_cpu():
    run_gesummv(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gesummv(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_gesummv(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gesummv(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gesummv(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gesummv(dace.dtypes.DeviceType.FPGA)
