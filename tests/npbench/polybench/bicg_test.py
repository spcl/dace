# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

# Data set sizes
# M, N
sizes = {
    "mini": (38, 42),
    "small": (116, 124),
    "medium": (390, 410),
    "large": (1900, 2100),
    "extra-large": (1800, 2200)
}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


def initialize(M, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M), dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return A, p, r


@dc.program
def bicg_kernel(A: dc.float64[N, M], p: dc.float64[M], r: dc.float64[N]):
    return r @ A, A @ p


def bicg_jax_kernel(jnp, A, p, r):
    B, D = r @ A, A @ p
    return jnp.sum(D)


def run_bicg(device_type: dace.dtypes.DeviceType):
    '''
    Runs BiCG for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    M, N = sizes["small"]
    A, p, r = initialize(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = bicg_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        s, q = sdfg(A, p, r, M=M, N=N)

    # Compute ground truth and Validate result
    s_ref, q_ref = bicg_kernel.f(A, p, r)
    assert np.allclose(s, s_ref)
    assert np.allclose(q, q_ref)
    return sdfg


def run_bicg_autodiff():
    import jax
    import jax.numpy as jnp

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A, p, r = initialize(M, N)

    # Initialize gradient computation data
    B = np.zeros((M, ), dtype=np.float64)
    D = np.zeros((N, ), dtype=np.float64)
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float64[N, M], p: dc.float64[M], r: dc.float64[N]):
        B, D = bicg_kernel(A, p, r)
        return np.sum(D)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(A, p, r, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_kernel = lambda A, p, r: bicg_jax_kernel(jnp, A, p, r)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=0))
    jax_grad_A = jax_grad(A, p, r)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_bicg(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_bicg(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_bicg_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_bicg(dace.dtypes.DeviceType.CPU)
        run_bicg_autodiff()
    elif target == "gpu":
        run_bicg(dace.dtypes.DeviceType.GPU)
