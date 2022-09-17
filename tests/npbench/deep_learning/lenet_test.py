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

N, H, W, C_before_fc1, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                                 for s in ('N', 'H', 'W', 'C_before_fc1', 'S0', 'S1', 'S2', 'S3', 'S4',
                                                           'S5'))


@dc.program
def relu2(x: dc.float32[S0, S1]):
    return np.maximum(x, 0)


@dc.program
def relu4(x: dc.float32[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float32[S0, S1, S2, S3], weights: dc.float32[S4, S4, S3, S5]):
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + S4, j:j + S4, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# 2x2 maxpool operator, as used in LeNet-5
@dc.program
def maxpool2d(x: dc.float32[S0, S1, S2, S3]):
    output = np.ndarray([S0, S1 // 2, S2 // 2, S3], dtype=np.float32)
    for i in range(S1 // 2):
        for j in range(S2 // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
@dc.program
def lenet5_kernel(input: dc.float32[N, H, W, 1], conv1: dc.float32[5, 5, 1, 6], conv1bias: dc.float32[6],
                  conv2: dc.float32[5, 5, 6,
                                    16], conv2bias: dc.float32[16], fc1w: dc.float32[C_before_fc1,
                                                                                     120], fc1b: dc.float32[120],
                  fc2w: dc.float32[120, 84], fc2b: dc.float32[84], fc3w: dc.float32[84, 10], fc3b: dc.float32[10]):
    x1 = relu4(conv2d(input, conv1) + conv1bias)
    x2 = maxpool2d(x1)
    x3 = relu4(conv2d(x2, conv2) + conv2bias)
    x4 = maxpool2d(x3)
    x5 = np.reshape(x4, (N, C_before_fc1))
    x6 = relu2(x5 @ fc1w + fc1b)
    x7 = relu2(x6 @ fc2w + fc2b)
    return x7 @ fc3w + fc3b


def initialize(N, H, W):
    from numpy.random import default_rng
    rng = default_rng(42)

    H_conv1 = H - 4
    W_conv1 = W - 4
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4
    W_conv2 = W_pool1 - 4
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2

    # NHWC data layout
    input = rng.random((N, H, W, 1), dtype=np.float32)
    # Weights
    conv1 = rng.random((5, 5, 1, 6), dtype=np.float32)
    conv1bias = rng.random((6, ), dtype=np.float32)
    conv2 = rng.random((5, 5, 6, 16), dtype=np.float32)
    conv2bias = rng.random((16, ), dtype=np.float32)
    fc1w = rng.random((C_before_fc1, 120), dtype=np.float32)
    fc1b = rng.random((120, ), dtype=np.float32)
    fc2w = rng.random((120, 84), dtype=np.float32)
    fc2b = rng.random((84, ), dtype=np.float32)
    fc3w = rng.random((84, 10), dtype=np.float32)
    fc3b = rng.random((10, ), dtype=np.float32)

    return (input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1)


def relu_np(x):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
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


# 2x2 maxpool operator, as used in LeNet-5
def maxpool2d_np(x):
    output = np.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
def lenet5_np(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_np(conv2d_np(input, conv1) + conv1bias)
    x = maxpool2d_np(x)
    x = relu_np(conv2d_np(x, conv2) + conv2bias)
    x = maxpool2d_np(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_np(x @ fc1w + fc1b)
    x = relu_np(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


def run_lenet(device_type: dace.dtypes.DeviceType):
    '''
    Runs lenet for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, H, W = 4, 28, 28
    input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = initialize(N, H, W)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = lenet5_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(input,
                   conv1,
                   conv1bias,
                   conv2,
                   conv2bias,
                   fc1w,
                   fc1b,
                   fc2w,
                   fc2b,
                   fc3w,
                   fc3b,
                   N=N,
                   H=H,
                   W=W,
                   C_before_fc1=C_before_fc1)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = lenet5_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, H=W, W=W, C_before_fc1=C_before_fc1))
        out = sdfg(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b)

    # Compute ground truth and validate
    out_ref = lenet5_np(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    # Serialization causes issues, we temporarily disable it
    with set_temporary("testing", "serialization", value=False):
        run_lenet(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Code error")
@pytest.mark.gpu
def test_gpu():
    run_lenet(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Dynamic memory allocation")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_lenet(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_lenet(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_lenet(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_lenet(dace.dtypes.DeviceType.FPGA)
