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

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
@dc.program
def hdiff_kernel(in_field: dc.float64[I + 4, J + 4, K], out_field: dc.float64[I, J, K], coeff: dc.float64[I, J, K]):
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    # res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(
        (res1 * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res1,
    )

    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(
        (res2 * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res2,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] +
                                                                           fly_field[:, 1:, :] - fly_field[:, :-1, :])


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K))
    out_field = rng.random((I, J, K))
    coeff = rng.random((I, J, K))

    return in_field, out_field, coeff


def ground_truth(in_field, out_field, coeff):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(
        (res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res,
    )

    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(
        (res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] +
                                                                           fly_field[:, 1:, :] - fly_field[:, :-1, :])


def run_hdiff(device_type: dace.dtypes.DeviceType):
    '''
    Runs Hdiff for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    I, J, K = 64, 64, 60
    in_field, out_field, coeff = initialize(I, J, K)
    out_field_ref = np.copy(out_field)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = hdiff_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(in_field, out_field, coeff, I=I, J=J, K=K)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = hdiff_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(I=I, J=J, K=K))
        sdfg(in_field, out_field, coeff)

    # Compute ground truth and validate
    ground_truth(in_field, out_field_ref, coeff)
    assert np.allclose(out_field, out_field_ref)
    return sdfg


def test_cpu():
    run_hdiff(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_hdiff(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_hdiff(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_hdiff(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_hdiff(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_hdiff(dace.dtypes.DeviceType.FPGA)
