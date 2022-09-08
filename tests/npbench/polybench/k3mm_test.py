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
# NI, NJ, NK, NL, NM
sizes = {
    "mini": (16, 18, 20, 22, 24),
    "small": (40, 50, 60, 70, 80),
    "medium": (180, 190, 200, 210, 220),
    "large": (800, 900, 1000, 1100, 1200),
    "extra-large": (1600, 1800, 2000, 2200, 2400)
}

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64)
                      for s in ('NI', 'NJ', 'NK', 'NL', 'NM'))


@dc.program
def k3mm_kernel(A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NM],
           D: dc.float64[NM, NL]):

    return A @ B @ C @ D


def initialize(NI, NJ, NK, NL, NM, datatype=np.float64):
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / (5 * NI), (NI, NK),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ),
                        (NK, NJ),
                        dtype=datatype)
    C = np.fromfunction(lambda i, j: (i * (j + 3) % NL) / (5 * NL), (NJ, NM),
                        dtype=datatype)
    D = np.fromfunction(lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK),
                        (NM, NL),
                        dtype=datatype)

    return A, B, C, D


def run_k3mm(device_type: dace.dtypes.DeviceType):
    '''
    Runs 3MM for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    NI, NJ, NK, NL, NM = sizes["small"]
    A, B, C, D = initialize(NI, NJ, NK, NL, NM)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = k3mm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        E = sdfg(A, B, C, D, NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = k3mm_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM))
        E= sdfg(A, B, C, D)
    # Compute ground truth and validate
    E_ref = k3mm_kernel.f( A, B, C, D)
    assert np.allclose(E, E_ref)
    return sdfg


def test_cpu():
    run_k3mm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_k3mm(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_k3mm(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_k3mm(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_k3mm(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_k3mm(dace.dtypes.DeviceType.FPGA)
