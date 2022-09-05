# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


def initialize(M, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M), dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return A, p, r


@dc.program
def bicg_kernel(A: dc.float64[N, M], p: dc.float64[M], r: dc.float64[N]):
    return r @ A, A @ p


def run_bicg(device_type: dace.dtypes.DeviceType):
    '''
    Runs BiCG for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    M, N = (116, 124)
    A, p, r = initialize(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = bicg_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        s, q = sdfg(A, p, r, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        # Note: currently the kernel uses double-precision floating point numbers
        sdfg = bicg_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        
        sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory],
                                                         [{}, {
                                                             'storage': dace.StorageType.FPGA_Local
                                                         }],
                                                         print_report=True)
        assert sm_applied == 8  # 3 inlines and 3 Streaming memories

        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(M=M, N=N))
        s, q = sdfg(A, p, r)

    # Compute ground truth and Validate result
    s_ref, q_ref = bicg_kernel.f(A, p, r)
    assert np.allclose(s, s_ref)
    assert np.allclose(q, q_ref)
    return sdfg


def test_cpu():
    run_bicg(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_bicg(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_bicg(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_bicg(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_bicg(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_bicg(dace.dtypes.DeviceType.FPGA)
