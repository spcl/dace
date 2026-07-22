# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``resnet`` (ml) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'N': 8, 'W': 14, 'H': 14, 'C1': 32, 'C2': 8}
INPUT_ARGS = ('N', 'W', 'H', 'C1', 'C2')
ARRAY_ARGS = ('input', 'conv1', 'conv2', 'conv3', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

N, H, W, C1, C2, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                           for s in ('N', 'H', 'W', 'C1', 'C2', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5'))


def initialize(N, W, H, C1, C2, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)
    input = rng.random((N, H, W, C1), dtype=datatype)
    conv1 = rng.random((1, 1, C1, C2), dtype=datatype)
    conv2 = rng.random((3, 3, C2, C2), dtype=datatype)
    conv3 = rng.random((1, 1, C2, C1), dtype=datatype)
    out = np.zeros((N, H, W, C1), dtype=datatype)
    return (input, conv1, conv2, conv3, out)


# Numpy reference helpers (distinct names so they don't collide with the dace
# ``@dc.program`` operators of the same role used by the kernel below).
def relu_np(x):
    return np.maximum(x, 0)


def conv2d_np(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.zeros((N, H_out, W_out, C_out), dtype=input.dtype)
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )
    return output


def batchnorm2d_np(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


def reference(input, conv1, conv2, conv3, out):
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]), dtype=input.dtype)
    padded[:, 1:-1, 1:-1, :] = conv2d_np(input, conv1)
    x = batchnorm2d_np(padded)
    x = relu_np(x)
    x = conv2d_np(x, conv2)
    x = batchnorm2d_np(x)
    x = relu_np(x)
    x = conv2d_np(x, conv3)
    x = batchnorm2d_np(x)
    out[:] = relu_np(x + input)


@dc.program
def relu(x: dc_float[S0, S1, S2, S3]):
    return np.maximum(x, 0)


@dc.program
def conv2d(input: dc_float[S0, S1, S2, S3], weights: dc_float[S4, S4, S3, S5]):
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=dc_float)
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(input[:, i:i + S4, j:j + S4, :, np.newaxis] * weights[np.newaxis, :, :, :],
                                        axis=(1, 2, 3))
    return output


@dc.program
def batchnorm2d(x: dc_float[S0, S1, S2, S3]):
    mean = np.ndarray((1, S1, S2, S3), dtype=dc_float)
    mean[:] = np.mean(x, axis=0)
    std = np.ndarray((1, S1, S2, S3), dtype=dc_float)
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / dc_float(S0))
    return (x - mean) / np.sqrt(std + 1e-05)


@dc.program
def resnet_basicblock(input: dc_float[N, H, W, C1], conv1: dc_float[1, 1, C1, C2], conv2: dc_float[3, 3, C2, C2],
                      conv3: dc_float[1, 1, C2, C1]):
    padded = np.zeros((N, H + 2, W + 2, C2), dtype=dc_float)
    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x1 = relu(x)
    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    return relu(x6 + input)


@dc.program
def kernel(out: dc_float[N, H, W, C1], input: dc_float[N, H, W, C1], conv1: dc_float[1, 1, C1, C2],
           conv2: dc_float[3, 3, C2, C2], conv3: dc_float[1, 1, C2, C1]):
    padded = np.ndarray((N, H + 2, W + 2, C2), dtype=dc_float)
    padded[:] = 0
    padded[:, 1:H + 1, 1:W + 1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x1 = relu(x)
    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    return relu(x6 + input)


CORPUS = dict(name='resnet',
              dwarf='ml',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
