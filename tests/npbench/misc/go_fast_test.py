# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def go_fast_kernel(a: dc.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, N), dtype=np.float64)
    return x


def ground_truth(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


def run_go_fast(device_type: dace.dtypes.DeviceType):
    '''
    Runs go_fast for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N = 2000
    a = initialize(N)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = go_fast_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(a, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = go_fast_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        out = sdfg(a)

    # Compute ground truth and validate
    out_ref = ground_truth(a)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    run_go_fast(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_go_fast(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Operand type in binary expressions")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_go_fast(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_go_fast(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_go_fast(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_go_fast(dace.dtypes.DeviceType.FPGA)
