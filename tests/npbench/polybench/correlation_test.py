# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import os
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

# Data set sizes
# M, N
sizes = {"mini": (28, 32), "small": (80, 100), "medium": (240, 260), "large": (1200, 1400), "extra-large": (2600, 3000)}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def correlation_kernel(float_n: dc.float64, data: dc.float64[N, M]):

    mean = np.mean(data, axis=0)
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.mean(np.subtract(data, mean)**2, axis=0))
    stddev[stddev <= 0.1] = 1.0
    # data -= mean
    np.subtract(data, mean, out=data)
    # data /= np.sqrt(float_n) * stddev
    np.divide(data, np.sqrt(float_n) * stddev, out=data)
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        # corr[i, i+1:M] = np.transpose(data[:, i+1:M]) @ data[:, i]
        corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
        corr[i + 1:M, i] = corr[i, i + 1:M]

    return corr


def initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=datatype)

    return float_n, data


def correlation_jax_kernel(float_n, data):
    mean = jnp.mean(data, axis=0)
    M = data.shape[1]
    stddev = jnp.sqrt(jnp.mean(jnp.subtract(data, mean)**2, axis=0))
    stddev = jnp.where(stddev <= 0.1, 1.0, stddev)
    data = jnp.subtract(data, mean)
    data = jnp.divide(data, jnp.sqrt(float_n) * stddev)
    corr = jnp.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr = corr.at[i, i + 1:M].set(data[:, i] @ data[:, i + 1:M])
        corr = corr.at[i + 1:M, i].set(corr[i, i + 1:M])
    return jnp.sum(corr)


def ground_truth(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    return corr


def run_correlation(device_type: dace.dtypes.DeviceType):
    '''
    Runs correlation for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    float_n, data = initialize(M, N)
    float_n_ref = np.copy(float_n)
    data_ref = np.copy(data)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = correlation_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        last_value = os.environ.get('DACE_testing_serialization', '0')
        os.environ['DACE_testing_serialization'] = '0'
        corr = sdfg(float_n, data, M=M, N=N)
        os.environ['DACE_testing_serialization'] = last_value

    elif device_type == dace.dtypes.DeviceType.FPGA:
        pass  # Not Yet Implemented

    # Compute ground truth and validate result

    corr_ref = ground_truth(M, float_n_ref, data_ref)
    diff = corr_ref - corr
    assert np.abs(diff).max() <= 10e-10
    return sdfg


def run_correlation_autodiff():
    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    float_n, data = initialize(M, N)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float64)
    gradient_data = np.zeros_like(data)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(float_n: dc.float64, data: dc.float64[N, M]):
        corr = correlation_kernel(float_n, data)
        return np.sum(corr)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["__return"], autooptimize=False)
    sdfg(float_n, data, M=M, N=N, gradient_data=gradient_data, gradient___return=gradient___return)
    
    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(correlation_jax_kernel, argnums=1), static_argnums=(0,))
    float_n_jax = float_n
    data_jax = np.copy(initialize(M, N)[1]).astype(np.float64)  # Fresh copy of data
    S_jax = S.astype(np.float64)
    jax_grad_data = jax_grad(float_n_jax, data_jax)
    np.testing.assert_allclose(gradient_data, jax_grad_data, rtol=1e-5, atol=1e-8)


def test_cpu():
    run_correlation(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_correlation(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_correlation_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_correlation(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_correlation(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_correlation(dace.dtypes.DeviceType.FPGA)
