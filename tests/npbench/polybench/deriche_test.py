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

W, H = (dc.symbol(s, dtype=dc.int64) for s in ('W', 'H'))


@dc.program
def deriche_kernel(alpha: dc.float64, imgIn: dc.float64[W, H]):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = k
    a5 = k
    a2 = k * np.exp(-alpha) * (alpha - 1.0)
    a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = k * np.exp(-alpha) * (alpha + 1.0)
    a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = -k * np.exp(-2.0 * alpha)
    a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0**(-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = 1
    c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] + b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] + b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] + b1 * y1[i - 1, :] + b2 * y1[i - 2, :])

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] + b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = c2 * (y1 + y2)

    return imgOut


def initialize(W, H, datatype=np.float64):
    alpha = datatype(0.25)
    imgIn = np.fromfunction(lambda i, j: ((313 * i + 991 * j) % 65536) / 65535.0, (W, H), dtype=datatype)

    return alpha, imgIn


def ground_truth(alpha, imgIn):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0**(-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, imgIn.shape[1]):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] + b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(imgIn.shape[1] - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] + b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, imgIn.shape[0]):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] + b1 * y1[i - 1, :] + b2 * y1[i - 2, :])

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(imgIn.shape[0] - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] + b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = c2 * (y1 + y2)

    return imgOut


def run_deriche(device_type: dace.dtypes.DeviceType):
    '''
    Runs Deriche for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    W, H = (192, 128)
    alpha, imgIn = initialize(W, H)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = deriche_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        imgOut = sdfg(alpha, imgIn, W=W, H=H)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        # Note: currently the kernel uses double-precision floating point numbers and
        # works for Xilinx
        sdfg = deriche_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(W=W, H=H))
        imgOut = sdfg(alpha, imgIn)

    # Compute ground truth and validate result

    imgOut_ref = ground_truth(alpha, imgIn)
    assert np.allclose(imgOut, imgOut_ref)
    return sdfg


def test_cpu():
    run_deriche(dace.dtypes.DeviceType.CPU)

@pytest.mark.skip(reason="GPU AutoOpt support")
@pytest.mark.gpu
def test_gpu():
    run_deriche(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False, intel=False)
def test_fpga():
    return run_deriche(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_deriche(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_deriche(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_deriche(dace.dtypes.DeviceType.FPGA)
