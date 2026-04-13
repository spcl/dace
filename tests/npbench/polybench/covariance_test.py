# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import os
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusionVertical, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.libraries.standard import Reduce
from dace.libraries.blas import Gemv
from dace.autodiff import add_backward_pass

# Data set sizes
# M, N
sizes = {"mini": (28, 32), "small": (80, 100), "medium": (240, 260), "large": (1200, 1400), "extra-large": (2600, 3000)}

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def covariance_kernel(float_n: dc.float32, data: dc.float32[N, M]):

    mean = np.mean(data, axis=0)
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    # for i in range(M):
    #     cov[i, i:M] = data[:, i] @ data[:, i:M]

    return cov


def covariance_jax_kernel(jnp, float_n, data):
    mean = jnp.mean(data, axis=0)
    M = data.shape[1]
    data -= mean
    cov = jnp.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov = cov.at[i:M, i].set(data[:, i] @ data[:, i:M] / (float_n - 1.0))
        cov = cov.at[i, i:M].set(data[:, i] @ data[:, i:M] / (float_n - 1.0))
    return jnp.sum(cov)


def ground_truth(M, N, float_n, data):

    mean = np.empty((M, ), dtype=data.dtype)
    for j in range(M):
        mean[j] = 0.0
        for i in range(N):
            mean[j] += data[i, j]
        mean[j] /= float_n

    for i in range(N):
        for j in range(M):
            data[i, j] -= mean[j]

    cov = np.empty((M, M), dtype=data.dtype)
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = 0.0
            for k in range(N):
                cov[i, j] += data[k, i] * data[k, j]
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

    return cov


def init_data(M, N):

    float_n = np.float32(N)
    data = np.empty((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            data[i, j] = (i * j) / M

    return float_n, data


def run_covariance(device_type: dace.dtypes.DeviceType):
    """
    Runs Covariance for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    float_n, data = init_data(M, N)

    gt_data = np.copy(data)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = covariance_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        dace_res = sdfg(float_n=float_n, data=data, M=M, N=N)

    # Compute ground truth and validate result
    gt_res = ground_truth(M, N, float_n, gt_data)
    assert np.allclose(gt_res, dace_res)
    return sdfg


def run_covariance_autodiff():
    import jax
    import jax.numpy as jnp

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    float_n, data = init_data(M, N)
    data_jax = np.copy(data)

    # Initialize gradient computation data
    gradient_data = np.zeros_like(data)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(float_n: dc.float32, data: dc.float32[N, M]):
        cov = covariance_kernel(float_n, data)
        return np.sum(cov)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["__return"])
    sdfg(float_n, data, M=M, N=N, gradient_data=gradient_data, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_kernel = lambda float_n, data: covariance_jax_kernel(jnp, float_n, data)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=1), static_argnums=(0, ))
    jax_grad_data = jax_grad(float_n, data_jax)
    np.testing.assert_allclose(gradient_data, jax_grad_data, rtol=1e-5, atol=1e-8)


def test_cpu(monkeypatch):
    # Serialization causes issues, we temporarily disable it
    monkeypatch.setenv("DACE_testing_serialization", "0")
    run_covariance(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_covariance(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    # Serialization causes issues, we temporarily disable it
    # TODO: open an issue to fix the serialization stability problem
    last_value = os.environ.get('DACE_testing_serialization', '0')
    os.environ['DACE_testing_serialization'] = '0'
    run_covariance_autodiff()
    os.environ['DACE_testing_serialization'] = last_value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_covariance(dace.dtypes.DeviceType.CPU)
        run_covariance_autodiff()
    elif target == "gpu":
        run_covariance(dace.dtypes.DeviceType.GPU)
