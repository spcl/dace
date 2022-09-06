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
# NI, NJ, NK
sizes = {
    "mini": (20, 25, 30),
    "small": (60, 70, 80),
    "medium": (200, 220, 240),
    "large": (1000, 1100, 1200),
    "extra-large": (2000, 2300, 2600)
}

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))


@dc.program
def gemm_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ], A: dc.float64[NI, NK], B: dc.float64[NK,
                                                                                                                 NJ]):

    C[:] = alpha * A @ B + beta * C


def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


def run_gemm(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gemm for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    NI, NJ, NK = sizes["small"]
    alpha, beta, C, A, B = initialize(NI, NJ, NK)
    C_ref = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gemm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, C, A, B, NI=NI, NJ=NJ, NK=NK)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gemm_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(NI=NI, NJ=NJ, NK=NK))
        sdfg(alpha, beta, C, A, B)

    # Compute ground truth and validate
    gemm_kernel.f(alpha, beta, C_ref, A, B)
    assert np.allclose(C, C_ref)
    return sdfg


def test_cpu():
    run_gemm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gemm(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_gemm(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gemm(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gemm(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gemm(dace.dtypes.DeviceType.FPGA)
