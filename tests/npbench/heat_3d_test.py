# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition, MapFusion
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
import argparse

import numpy as np
import dace as dc


N = dc.symbol('N', dtype=dc.int32)


@dc.program
def heat_3d_kernel(TSTEPS: dc.int32, A: dc.float32[N, N, N],
           B: dc.float32[N, N, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1, 1:-1] = (
                0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                         A[:-2, 1:-1, 1:-1]) +
                0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                         A[1:-1, :-2, 1:-1]) +
                0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                         A[1:-1, 1:-1, 0:-2]) +
                A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1, 1:-1] = (
                0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                         B[:-2, 1:-1, 1:-1]) +
                0.125 * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                         B[1:-1, :-2, 1:-1]) +
                0.125 * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                         B[1:-1, 1:-1, 0:-2]) +
                B[1:-1, 1:-1, 1:-1])



def init_data(N):

    A = np.empty((N, N, N), dtype=np.float32)
    B = np.empty((N, N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i, j, k] = B[i, j, k] = (i + j + (N - k)) * 10 / N

    return A, B


def run_heat_3d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Heat-3d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench medium size)
    TSTEPS, N = 100, 40
    A, B = init_data(N)
    gt_A = np.copy(A)
    gt_B = np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = heat_3d_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS=TSTEPS, A=A, B=B, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = heat_3d_kernel.to_sdfg(strict=True)
        sdfg.apply_transformations_repeated([MapFusion])
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1
        # sdfg.view()
        # sm_applied = sdfg.apply_transformations_repeated(
        #     [InlineSDFG, StreamingComposition],
        #     [{}, {
        #         'storage': dace.StorageType.FPGA_Local
        #     }],
        #     print_report=True)
        # sdfg.view()
        # assert sm_applied == 2
        #
        sdfg.apply_transformations_repeated([InlineSDFG])
        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)
        # # In this case, we want to generate the top-level state as an host-based state,
        # # not an FPGA kernel. We need to explicitly indicate that
        # sdfg.states()[0].location["is_FPGA_kernel"] = False
        # sdfg.states()[0].nodes()[0].sdfg.specialize(dict(W=W, H=H))
        sdfg.specialize(dict(N=N))
        sdfg(A=A, B=B, TSTEPS=TSTEPS)

    # Compute ground truth and validate result
    heat_3d_kernel.f(TSTEPS, gt_A, gt_B)
    assert np.allclose(A, gt_A)
    assert np.allclose(B, gt_B)
    # diff_ex = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    # diff_ey = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    # diff_hz = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    # tol = 1e-6
    #
    # assert diff_ex < tol
    # assert diff_ey < tol
    # assert diff_hz < tol

    return sdfg



def test_cpu():
    run_heat_3d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_heat_3d(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_heat_3d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--target",
        default='cpu',
        choices=['cpu', 'gpu', 'fpga'],
        help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_heat_3d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_heat_3d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_heat_3d(dace.dtypes.DeviceType.FPGA)