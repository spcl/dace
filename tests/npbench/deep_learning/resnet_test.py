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

N, H, W, C1, C2, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                           for s in ('N', 'H', 'W', 'C1', 'C2', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5'))


@dc.program
def relu(x: dc.float32[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float32[S0, S1, S2, S3], weights: dc.float32[S4, S4, S3, S5]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + S4, j:j + S4, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Batch normalization operator, as used in ResNet
@dc.program
def batchnorm2d(x: dc.float32[S0, S1, S2, S3]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float32(S0))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float32(S0))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@dc.program
def resnet_basicblock(input: dc.float32[N, H, W, C1], conv1: dc.float32[1, 1, C1, C2], conv2: dc.float32[3, 3, C2, C2],
                      conv3: dc.float32[1, 1, C2, C1]):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.zeros((N, H + 2, W + 2, C2), dtype=np.float32)

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x1 = relu(x)

    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    return relu(x6 + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@dc.program
def resnet_basicblock_gpu(out: dc.float32[N, H, W, C1], input: dc.float32[N, H, W, C1], conv1: dc.float32[1, 1, C1, C2],
                          conv2: dc.float32[3, 3, C2, C2], conv3: dc.float32[1, 1, C2, C1]):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.ndarray((N, H + 2, W + 2, C2), dtype=np.float32)
    padded[:] = 0

    # padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    padded[:, 1:H + 1, 1:W + 1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x1 = relu(x)

    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    return relu(x6 + input)


def initialize(N, W, H, C1, C2):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float32)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float32)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float32)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float32)
    return (input, conv1, conv2, conv3)


###### Reference implementation (numpy)


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


# Batch normalization operator, as used in ResNet
def batchnorm2d_np(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock_np(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_np(input, conv1)
    x = batchnorm2d_np(padded)
    x = relu_np(x)

    x = conv2d_np(x, conv2)
    x = batchnorm2d_np(x)
    x = relu_np(x)
    x = conv2d_np(x, conv3)
    x = batchnorm2d_np(x)
    return relu_np(x + input)


def run_resnet(device_type: dace.dtypes.DeviceType):
    '''
    Runs resnet for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, W, H, C1, C2 = 8, 14, 14, 32, 8
    input, conv1, conv2, conv3 = initialize(N, W, H, C1, C2)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = resnet_basicblock.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(input=input, conv1=conv1, conv2=conv2, conv3=conv3, N=N, W=W, H=H, C1=C1, C2=C2)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = resnet_basicblock.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, W=W, H=H, C1=C1, C2=C2))
        out = sdfg(input=input, conv1=conv1, conv2=conv2, conv3=conv3)

    # Compute ground truth and validate
    out_ref = resnet_basicblock_np(input, conv1, conv2, conv3)
    assert np.allclose(out, out_ref, rtol=1e-4, atol=1e-5)
    return sdfg


def test_cpu():
    run_resnet(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Dynamic memory allocation")
@pytest.mark.gpu
def test_gpu():
    run_resnet(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Dynamic memory allocation")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_resnet(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_resnet(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_resnet(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_resnet(dace.dtypes.DeviceType.FPGA)
