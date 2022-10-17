# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusion
from dace.transformation.auto.auto_optimize import auto_optimize

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def kernel(TSTEPS: dc.int32, A: dc.float32[N, N], B: dc.float32[N, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


def init_data(N):
    A = np.empty((N, N), dtype=np.float32)
    B = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            A[i, j] = i * (j + 2) / N
            B[i, j] = i * (j + 3) / N
    return A, B


def run_jacobi_2d(device_type: dace.dtypes.DeviceType):
    '''
    Runs jacobi_2d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    TSTEPS, N = (20, 30)
    A, B = init_data(N)
    np_A, np_B = np.copy(A), np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A=A, B=B, TSTEPS=TSTEPS, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        sdfg.apply_transformations_repeated([MapFusion])
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory],
                                                         [{}, {
                                                             'storage': dace.StorageType.FPGA_Local
                                                         }],
                                                         print_report=True)

        assert sm_applied > 0

        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.states()[0].location["is_FPGA_kernel"] = False
        # we need to specialize both the top-level SDFG and the nested SDFG
        for sd in sdfg.all_sdfgs_recursive():
            sd.specialize(dict(N=N))
        # run program
        sdfg(A=A, B=B, TSTEPS=TSTEPS)

    # Compute ground truth and validate result
    kernel.f(TSTEPS, np_A, np_B)
    assert np.allclose(A, np_A)
    assert np.allclose(B, np_B)
    return sdfg


def test_cpu():
    run_jacobi_2d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_jacobi_2d(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_jacobi_2d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_jacobi_2d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_jacobi_2d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_jacobi_2d(dace.dtypes.DeviceType.FPGA)
