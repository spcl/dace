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
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float32[M, N], x: dc.float32[N]):
    return (A @ x) @ A


def init_data(M, N):
    fn = np.float32(N)
    A = np.empty((M, N), dtype=np.float32)
    x = np.empty((N, ), dtype=np.float32)
    y = np.empty((N, ), dtype=np.float32)
    for i in range(N):
        x[i] = 1 + (i / fn)
    for i in range(M):
        for j in range(N):
            A[i, j] = ((i + j) % N) / (5 * M)
    return A, x, y


def run_atax(device_type: dace.dtypes.DeviceType):
    '''
    Runs ATAX for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench medium size)
    M, N = (390, 410)
    A, x, y_ref = init_data(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        y = sdfg(A, x, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(coarsen=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        sm_applied = sdfg.apply_transformations_repeated(
            [InlineSDFG, StreamingMemory],
            [{}, {
                'storage': dace.StorageType.FPGA_Local
            }],
            print_report=True)
        assert sm_applied == 6  # 3 inlines and 3 Streaming memories

        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(M=M, N=N))
        y = sdfg(A, x)

    # Compute ground truth and Validate result
    y_ref = kernel.f(A, x)
    assert np.allclose(y, y_ref)
    return sdfg


def test_cpu():
    run_atax(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_atax(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_atax(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--target",
                        default='cpu',
                        choices=['cpu', 'gpu', 'fpga'],
                        help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_atax(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_atax(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_atax(dace.dtypes.DeviceType.FPGA)
