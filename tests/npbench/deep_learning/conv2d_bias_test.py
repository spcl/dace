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

C_in, C_out, H, K, N, W = (dc.symbol(s, dc.int64) for s in ('C_in', 'C_out', 'H', 'K', 'N', 'W'))


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in, C_out]):

    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


@dc.program
def conv2d_bias_kernel(input: dc.float32[N, H, W, C_in], weights: dc.float32[K, K, C_in, C_out],
                       bias: dc.float32[C_out]):
    return conv2d(input, weights) + bias


def initialize(C_in, C_out, H, K, N, W):
    from numpy.random import default_rng
    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out, ), dtype=np.float32)
    return input, weights, bias


def conv2d_np(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def conv2d_bias_np(input, weights, bias):
    return conv2d_np(input, weights) + bias


def run_conv2d_bias(device_type: dace.dtypes.DeviceType):
    '''
    Runs conv2d_bias for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, C_in, C_out, K, H, W = 8, 3, 16, 2, 32, 32
    input, weight, bias = initialize(C_in, C_out, H, K, N, W)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = conv2d_bias_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(input, weight, bias, C_in=C_in, C_out=C_out, H=H, K=K, N=N, W=W)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = conv2d_bias_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(C_in=C_in, C_out=C_out, H=H, K=K, N=N, W=W))
        out = sdfg(input, weight, bias)

    # Compute ground truth and validate
    out_ref = conv2d_bias_np(input, weight, bias)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    run_conv2d_bias(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_conv2d_bias(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_conv2d_bias(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_conv2d_bias(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_conv2d_bias(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_conv2d_bias(dace.dtypes.DeviceType.FPGA)
