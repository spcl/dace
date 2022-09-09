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
# NI, NJ, NK, NL
sizes = {
    "mini": (16, 18, 22, 24),
    "small": (40, 50, 70, 80),
    "medium": (180, 190, 210, 220),
    "large": (800, 900, 1100, 1200),
    "extra-large": (1600, 1800, 2200, 2400)
}

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK', 'NL'))


@dc.program
def k2mm_kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK], B: dc.float64[NK, NJ],
                C: dc.float64[NJ, NL], D: dc.float64[NI, NL]):

    D[:] = alpha * A @ B @ C + beta * D


def initialize(NI, NJ, NK, NL, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, (NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: ((i * (j + 3) + 1) % NL) / NL, (NJ, NL), dtype=datatype)
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, (NI, NL), dtype=datatype)

    return alpha, beta, A, B, C, D


def run_k2mm(device_type: dace.dtypes.DeviceType):
    '''
    Runs 2MM for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    NI, NJ, NK, NL = sizes["small"]
    alpha, beta, A, B, C, D = initialize(NI, NJ, NK, NL)
    D_ref = np.copy(D)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = k2mm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, A, B, C, D, NI=NI, NJ=NJ, NK=NK, NL=NL)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = k2mm_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(NI=NI, NJ=NJ, NK=NK, NL=NL))
        sdfg(alpha, beta, A, B, C, D)
    # Compute ground truth and validate
    k2mm_kernel.f(alpha, beta, A, B, C, D_ref)
    assert np.allclose(D, D_ref)
    return sdfg


def test_cpu():
    run_k2mm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_k2mm(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_k2mm(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_k2mm(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_k2mm(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_k2mm(dace.dtypes.DeviceType.FPGA)
