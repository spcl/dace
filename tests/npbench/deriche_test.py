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

W, H = (dc.symbol(s, dtype=dc.int32) for s in ('W', 'H'))


@dc.program
def deriche_kernel(alpha: dc.float32, imgIn: dc.float32[W, H]):
    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (
            1.0 + alpha * np.exp(-alpha) - np.exp(np.float32(2.0) * alpha))
    a1 = k
    a5 = k
    a2 = k * np.exp(-alpha) * (alpha - 1.0)
    a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = k * np.exp(-alpha) * (alpha + 1.0)
    a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = - k * np.exp(-2.0 * alpha)
    a8 = - k * np.exp(-2.0 * alpha)
    b1 = np.float32(2.0) ** (-alpha)
    b2 = - np.exp(-2.0 * alpha)
    c1 = 1
    c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] +
                    b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] +
                    b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] +
                    b1 * y1[i - 1, :] + b2 * y1[i - 2, :])

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] +
                    b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = c2 * (y1 + y2)

    return imgOut


def ground_truth(W, H, alpha, imgIn, imgOut, y1, y2):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (
            1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0**(-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = c2 = 1


    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] +
                    b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2[:, H - 1] = 0.0
    y2[:, H - 2] = a3 * imgIn[:, H - 1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] +
                    b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut[:] = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] +
                    b1 * y1[i - 1, :] + b2 * y1[i - 2, :])


    y2[W - 1, :] = 0.0
    y2[W - 2, :] = a7 * imgOut[W - 1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] +
                    b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = c2 * (y1 + y2)


def init_data(W, H):

    alpha = np.float32(0.25)
    imgIn = np.empty((W, H), dtype=np.float32)
    imgOut = np.empty((W, H), dtype=np.float32)
    y1 = np.empty((W, H), dtype=np.float32)
    y2 = np.empty((W, H), dtype=np.float32)
    for i in range(W):
        for j in range(H):
            imgIn[i, j] = ((313 * i + 991 * j) % 65536) / 65535.0

    return alpha, imgIn, imgOut, y1, y2




def run_deriche(device_type: dace.dtypes.DeviceType):
    '''
    Runs Deriche for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench medium size)
    W, H = 720, 480
    alpha, imgIn, imgOut, y1, y2 = init_data(W, H)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = deriche_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        dace_res = sdfg(alpha=alpha, imgIn=imgIn, W=W, H=H)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = deriche_kernel.to_sdfg(strict=True)
        sdfg.apply_transformations_repeated([MapFusion])
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1
        # sdfg.view()
        # sm_applied = sdfg.apply_transformations_repeated(
        #     [InlineSDFG, StreamingMemory],
        #     [{}, {
        #         'storage': dace.StorageType.FPGA_Local
        #     }],
        #     print_report=True)
        #
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
        sdfg.specialize(dict(W=W, H=H))
        dace_res = sdfg(imgIn=imgIn, alpha=alpha)

    # Compute ground truth and validate result
    ground_truth(W, H, alpha, imgIn, imgOut, y1, y2)
    assert np.allclose(dace_res, imgOut)
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
    run_deriche(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_deriche(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_deriche(dace.dtypes.DeviceType.FPGA)


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
        run_deriche(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_deriche(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_deriche(dace.dtypes.DeviceType.FPGA)
