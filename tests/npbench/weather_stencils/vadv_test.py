# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize
# Sample constants
BET_M = 0.5
BET_P = 0.5

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@dc.program
def vadv_kernel(utens_stage: dc.float64[I, J, K], u_stage: dc.float64[I, J, K], wcon: dc.float64[I + 1, J, K],
                u_pos: dc.float64[I, J, K], utens: dc.float64[I, J, K], dtr_stage: dc.float64):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs[:] = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] -
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                      u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        # as_ = gav * BET_M
        as_[:] = gav * BET_M
        # acol = gav * BET_P
        acol[:] = gav * BET_P
        # bcol = dtr_stage - acol
        bcol[:] = dtr_stage - acol

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        # datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    dtr_stage = 3. / 20.

    # Define arrays
    utens_stage = rng.random((I, J, K))
    u_stage = rng.random((I, J, K))
    wcon = rng.random((I + 1, J, K))
    u_pos = rng.random((I, J, K))
    utens = rng.random((I, J, K))

    return dtr_stage, utens_stage, u_stage, wcon, u_pos, utens


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def ground_truth(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2]
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                   u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


def run_vadv(device_type: dace.dtypes.DeviceType):
    '''
    Runs VAdv for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    I, J, K = 60, 60, 40
    dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = initialize(I, J, K)
    utens_stage_ref = np.copy(utens_stage)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = vadv_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage, I=I, J=J, K=K)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = vadv_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(I=I, J=J, K=K))
        sdfg(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)

    # Compute ground truth and validate
    ground_truth(utens_stage_ref, u_stage, wcon, u_pos, utens, dtr_stage)
    assert np.allclose(utens_stage, utens_stage_ref)
    return sdfg


def test_cpu():
    run_vadv(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_vadv(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Variable not defined")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_vadv(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_vadv(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_vadv(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_vadv(dace.dtypes.DeviceType.FPGA)
