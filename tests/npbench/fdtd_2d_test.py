# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition, MapFusion
from dace.transformation.auto.auto_optimize import auto_optimize
import argparse

TMAX, NX, NY = (dc.symbol(s, dtype=dc.int32) for s in ('TMAX', 'NX', 'NY'))


@dc.program
def kernel(ex: dc.float32[NX, NY], ey: dc.float32[NX, NY], hz: dc.float32[NX, NY], _fict_: dc.float32[TMAX]):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])


def init_data(TMAX, NX, NY):

    ex = np.empty((NX, NY), dtype=np.float32)
    ey = np.empty((NX, NY), dtype=np.float32)
    hz = np.empty((NX, NY), dtype=np.float32)
    _fict_ = np.empty((TMAX, ), dtype=np.float32)
    for i in range(TMAX):
        _fict_[i] = i
    for i in range(NX):
        for j in range(NY):
            ex[i, j] = (i * (j + 1)) / NX
            ey[i, j] = (i * (j + 2)) / NY
            hz[i, j] = (i * (j + 3)) / NX

    return ex, ey, hz, _fict_


def ground_truth(TMAX, NX, NY, ex, ey, hz, _fict_):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:NX - 1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :NY - 1])
        hz[:NX - 1, :NY - 1] -= 0.7 * (ex[:NX - 1, 1:] - ex[:NX - 1, :NY - 1] + ey[1:, :NY - 1] - ey[:NX - 1, :NY - 1])


def run_fdtd_2d(device_type: dace.dtypes.DeviceType):
    """
    Runs FDTD-2D for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    TMAX, NX, NY = (20, 20, 30)

    ex, ey, hz, _fict_ = init_data(TMAX, NX, NY)
    gt_ex, gt_ey, gt_hz = np.copy(ex), np.copy(ey), np.copy(hz)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(ex=ex, ey=ey, hz=hz, _fict_=_fict_, TMAX=TMAX, NX=NX, NY=NY)
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

        sdfg.apply_transformations_repeated([InlineSDFG])
        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.states()[0].location["is_FPGA_kernel"] = False

        sdfg(ex=ex, ey=ey, hz=hz, _fict_=_fict_, TMAX=TMAX, NX=NX, NY=NY)

    # Compute ground truth and validate result
    ground_truth(TMAX, NX, NY, gt_ex, gt_ey, gt_hz, _fict_=_fict_)
    diff_ex = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    diff_ey = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    diff_hz = np.linalg.norm(gt_ex - ex) / np.linalg.norm(gt_ex)
    tol = 1e-6

    assert diff_ex < tol
    assert diff_ey < tol
    assert diff_hz < tol

    return sdfg


def test_cpu():
    run_fdtd_2d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_fdtd_2d(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_fdtd_2d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_fdtd_2d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_fdtd_2d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_fdtd_2d(dace.dtypes.DeviceType.FPGA)
