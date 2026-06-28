# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``lenet`` (ml) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

# C_before_fc1 = 16 * (H_pool2 * W_pool2). The dace kernel uses it as a free symbol
# in ``np.reshape(x, (N, C_before_fc1))`` that cannot be inferred from a transient,
# so it must be bound explicitly. With the harness size-cap clamping H=W=28 -> 16,
# the conv/pool chain gives H_pool2 = W_pool2 = 1, hence C_before_fc1 = 16.
SIZES = {'N': 4, 'H': 28, 'W': 28, 'C_before_fc1': 16}
INPUT_ARGS = ('N', 'H', 'W')
ARRAY_ARGS = ('input', 'conv1', 'conv1bias', 'conv2', 'conv2bias', 'fc1w', 'fc1b', 'fc2w', 'fc2b', 'fc3w', 'fc3b',
              'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

N, H, W, C_before_fc1, S0, S1, S2, S3, S4, S5 = (dc.symbol(s, dtype=dc.int64)
                                                 for s in ('N', 'H', 'W', 'C_before_fc1', 'S0', 'S1', 'S2', 'S3', 'S4',
                                                           'S5'))


def initialize(N, H, W, datatype=np.float32):
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
    input = rng.random((N, H, W, 1), dtype=datatype)
    conv1 = rng.random((5, 5, 1, 6), dtype=datatype)
    conv1bias = rng.random((6, ), dtype=datatype)
    conv2 = rng.random((5, 5, 6, 16), dtype=datatype)
    conv2bias = rng.random((16, ), dtype=datatype)
    fc1w = rng.random((C_before_fc1, 120), dtype=datatype)
    fc1b = rng.random((120, ), dtype=datatype)
    fc2w = rng.random((120, 84), dtype=datatype)
    fc2b = rng.random((84, ), dtype=datatype)
    fc3w = rng.random((84, 10), dtype=datatype)
    fc3b = rng.random((10, ), dtype=datatype)
    out = np.zeros((N, 10), dtype=datatype)
    # C_before_fc1 is encoded in fc1w.shape[0]; the dace kernel infers the matching
    # symbol from that input array's shape, and the reference recovers it the same way.
    return (input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, out)


# Numpy reference helpers (distinct names so they don't collide with the dace
# ``@dc.program`` operators -- the dace ``conv2d``/``maxpool2d`` below are used by
# the kernel; the numpy ``lenet5`` reference must use these numpy versions).
def relu_np(x):
    return np.maximum(x, 0)


def conv2d_np(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=input.dtype)
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )
    return output


def maxpool2d_np(x):
    output = np.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output


def reference(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, out):
    C_before_fc1 = fc1w.shape[0]
    x = relu_np(conv2d_np(input, conv1) + conv1bias)
    x = maxpool2d_np(x)
    x = relu_np(conv2d_np(x, conv2) + conv2bias)
    x = maxpool2d_np(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_np(x @ fc1w + fc1b)
    x = relu_np(x @ fc2w + fc2b)
    out[:] = x @ fc3w + fc3b


@dc.program
def relu2(x: dc_float[S0, S1]):
    return np.maximum(x, 0)


@dc.program
def relu4(x: dc_float[S0, S1, S2, S3]):
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
def maxpool2d(x: dc_float[S0, S1, S2, S3]):
    output = np.ndarray([S0, S1 // 2, S2 // 2, S3], dtype=dc_float)
    for i in range(S1 // 2):
        for j in range(S2 // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
    return output


@dc.program
def kernel(input: dc_float[N, H, W, 1], conv1: dc_float[5, 5, 1, 6], conv1bias: dc_float[6],
           conv2: dc_float[5, 5, 6, 16], conv2bias: dc_float[16], fc1w: dc_float[C_before_fc1, 120],
           fc1b: dc_float[120], fc2w: dc_float[120, 84], fc2b: dc_float[84], fc3w: dc_float[84,
                                                                                            10], fc3b: dc_float[10]):
    x1 = relu4(conv2d(input, conv1) + conv1bias)
    x2 = maxpool2d(x1)
    x3 = relu4(conv2d(x2, conv2) + conv2bias)
    x4 = maxpool2d(x3)
    x5 = np.reshape(x4, (N, C_before_fc1))
    x6 = relu2(x5 @ fc1w + fc1b)
    x7 = relu2(x6 @ fc2w + fc2b)
    return x7 @ fc3w + fc3b


CORPUS = dict(name='lenet',
              dwarf='ml',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
