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
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def trisolv_kernel(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N]):
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


def initialize(N, datatype=np.float64):
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N), dtype=datatype)
    x = np.full((N, ), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N, ), dtype=datatype)
    return L, x, b


def ground_truth(L, x, b):
    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


def run_trisolv(device_type: dace.dtypes.DeviceType):
    '''
    Runs trisolv for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    L, x, b = initialize(N)
    x_ref = np.copy(x)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = trisolv_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(L, x, np.copy(b), N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = trisolv_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(L, x, np.copy(b))

    # Compute ground truth and validate
    ground_truth(L, x_ref, b)
    assert np.allclose(x, x_ref)
    return sdfg


def test_cpu():
    run_trisolv(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_trisolv(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_trisolv(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_trisolv(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_trisolv(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_trisolv(dace.dtypes.DeviceType.FPGA)
