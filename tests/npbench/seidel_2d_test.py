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
def seidel_2d_kernel(TSTEPS: dc.int32, A: dc.float32[N, N]):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                           A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def ground_truth(TSTEPS, N, A):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:N - 1] += (A[i - 1, :N - 2] + A[i - 1, 1:N - 1] +
                              A[i - 1, 2:] + A[i, 2:] + A[i + 1, :N - 2] +
                              A[i + 1, 1:N - 1] + A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def init_data(N):

    A = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            A[i, j] = (i * (j + 2) + 2) / N

    return A


def run_seidel_2d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Seidel_2d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    TSTEPS, N = 40, 120
    A = init_data(N)
    gt_A = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = seidel_2d_kernel.to_sdfg()
        sdfg.apply_strict_transformations()

        # FAILS with auto optimization (due to greedy fuse)
        # sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS=TSTEPS, A=A, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = seidel_2d_kernel.to_sdfg(strict=True)
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
        sdfg(A=A, B=B)

    # Compute ground truth and validate result
    ground_truth(TSTEPS, N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    print(diff)
    assert np.allclose(A, gt_A)

    return sdfg



def test_cpu():
    run_seidel_2d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_seidel_2d(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_seidel_2d(dace.dtypes.DeviceType.FPGA)


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
        run_seidel_2d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_seidel_2d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_seidel_2d(dace.dtypes.DeviceType.FPGA)