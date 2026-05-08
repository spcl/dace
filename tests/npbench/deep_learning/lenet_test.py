# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

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


def conv2d_lax(jnp, lax, input, weights):
    # Kernel size, number of input images, and output dimensions.
    K = weights.shape[0]  # Assuming square kernel of size K x K.
    N = input.shape[0]  # Batch size.
    H_out = input.shape[1] - K + 1  # Output height.
    W_out = input.shape[2] - K + 1  # Output width.
    C_out = weights.shape[3]  # Number of output channels.

    # Allocate output array.
    output = jnp.empty((N, H_out, W_out, C_out), dtype=input.dtype)

    # Row update: iterate over output rows.
    def row_update(out, i):
        # Column update: iterate over output columns.
        def col_update(out, j):
            # Extract a patch from 'input' at the given (i, j) position.
            patch = lax.dynamic_slice(input, (0, i, j, 0), (N, K, K, input.shape[-1]))
            # Expand dims on the patch to broadcast with weights.
            # weights: shape (K, K, in_channels, C_out)
            # patch[..., None] becomes shape (N, K, K, in_channels, 1)
            # We add a new leading dimension to weights to broadcast:
            conv = jnp.sum(patch[..., None] * weights[None, :, :, :], axis=(1, 2, 3))
            # conv now has shape (N, C_out). Update output at (0, i, j, 0).
            out = lax.dynamic_update_slice(out, conv[:, None, None, :], (0, i, j, 0))
            return out, None

        out, _ = lax.scan(col_update, out, jnp.arange(W_out))
        return out, None

    output, _ = lax.scan(row_update, output, jnp.arange(H_out))
    return output


def maxpool2d_lax(jnp, lax, x):
    output = jnp.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)

    def row_update(output, i):

        def col_update(output, j):
            input_slice = lax.dynamic_slice(x, (0, 2 * i, 2 * j, 0), (x.shape[0], 2, 2, x.shape[3]))
            output = lax.dynamic_update_slice(output, jnp.max(input_slice, axis=(1, 2))[:, None, None, :], (0, i, j, 0))
            return output, None

        output, _ = lax.scan(col_update, output, jnp.arange(x.shape[2] // 2))
        return output, None

    output, _ = lax.scan(row_update, output, jnp.arange(x.shape[1] // 2))

    return output


def jax_relu(jnp, x):
    return jnp.maximum(x, 0)


def lenet_jax_kernel(jnp, lax, input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b):
    C_before_fc1 = fc1w.shape[0]
    N = input.shape[0]
    x = jax_relu(jnp, conv2d_lax(jnp, lax, input, conv1) + conv1bias)
    x = maxpool2d_lax(jnp, lax, x)
    x = jax_relu(jnp, conv2d_lax(jnp, lax, x, conv2) + conv2bias)
    x = maxpool2d_lax(jnp, lax, x)
    x = jnp.reshape(x, (N, C_before_fc1))
    x = jax_relu(jnp, x @ fc1w + fc1b)
    x = jax_relu(jnp, x @ fc2w + fc2b)
    return jnp.sum(x @ fc3w + fc3b)


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

    # Compute ground truth and validate
    out_ref = lenet5_np(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1)
    assert np.allclose(out, out_ref)
    return sdfg


def run_lenet_autodiff():
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    # Initialize data (npbench test size)
    N, H, W = 4, 16, 16
    input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = initialize(N, H, W)

    # Initialize gradient computation data
    gradient_input = np.zeros_like(input, dtype=np.float32)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(input: dc.float32[N, H, W, 1], conv1: dc.float32[5, 5, 1, 6], conv1bias: dc.float32[6],
                        conv2: dc.float32[5, 5, 6, 16], conv2bias: dc.float32[16], fc1w: dc.float32[C_before_fc1, 120],
                        fc1b: dc.float32[120], fc2w: dc.float32[120, 84], fc2b: dc.float32[84],
                        fc3w: dc.float32[84, 10], fc3b: dc.float32[10]):
        result = lenet5_kernel(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b)
        return np.sum(result)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["input"], outputs=["__return"])

    sdfg(input,
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
         C_before_fc1=C_before_fc1,
         gradient_input=gradient_input,
         gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_kernel = lambda input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b: lenet_jax_kernel(
        jnp, lax, input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=0))
    jax_grad_input = jax_grad(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b)
    np.testing.assert_allclose(gradient_input, jax_grad_input, rtol=1e-6)


def test_cpu(monkeypatch):
    # Serialization causes issues, we temporarily disable it
    monkeypatch.setenv("DACE_testing_serialization", "0")
    run_lenet(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="std::runtime_error")
@pytest.mark.gpu
def test_gpu():
    run_lenet(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_lenet_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_lenet(dace.dtypes.DeviceType.CPU)
        run_lenet_autodiff()
    elif target == "gpu":
        run_lenet(dace.dtypes.DeviceType.GPU)
