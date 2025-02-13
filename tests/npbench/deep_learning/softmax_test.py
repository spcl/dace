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

N, H, SM = (dc.symbol(s, dc.int64) for s in ('N', 'H', 'SM'))


# Numerically-stable version of softmax
@dc.program
def softmax_kernel(x: dc.float32[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x


def ground_truth(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def run_softmax(device_type: dace.dtypes.DeviceType):
    '''
    Runs Softmax for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, H, SM = 16, 16, 128
    x = initialize(N, H, SM)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = softmax_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(x, N=N, H=H, SM=SM)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = softmax_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, H=H, SM=SM))
        out = sdfg(x)

    # Compute ground truth and validate
    out_ref = ground_truth(x)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    run_softmax(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_softmax(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_softmax(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_softmax(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_softmax(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_softmax(dace.dtypes.DeviceType.FPGA)
