# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.config import set_temporary
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

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float32[M, N], x: dc.float32[N]):
    return (A @ x) @ A


def init_data(M, N):
    fn = np.float32(N)
    A = np.empty((M, N), dtype=np.float32)
    x = np.empty((N, ), dtype=np.float32)
    y = np.empty((N, ), dtype=np.float32)
    for i in range(N):
        x[i] = 1 + (i / fn)
    for i in range(M):
        for j in range(N):
            A[i, j] = ((i + j) % N) / (5 * M)
    return A, x, y


def atax_jax_kernel(jnp, A, x):
    B = (A @ x) @ A
    return jnp.sum(B)


def run_atax(device_type: dace.dtypes.DeviceType):
    """
    Runs ATAX for the given device

    :return: the SDFG
    """

    # Initialize data (polybench small size)
    M, N = sizes["small"]
    A, x, y_ref = init_data(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        y = sdfg(A, x, M=M, N=N)

    # Compute ground truth and Validate result
    y_ref = kernel.f(A, x)
    assert np.allclose(y, y_ref)
    return sdfg


def run_atax_autodiff():
    import jax
    import jax.numpy as jnp

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A, x, y = init_data(M, N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float32[M, N], x: dc.float32[N]):
        y = kernel(A, x)
        return np.sum(y)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(A, x, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_kernel = lambda A, x: atax_jax_kernel(jnp, A, x)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=0))
    jax_grad_A = jax_grad(A, x)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-6, atol=1e-6)


def test_cpu():
    run_atax(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_atax(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_atax_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_atax(dace.dtypes.DeviceType.CPU)
        run_atax_autodiff()
    elif target == "gpu":
        run_atax(dace.dtypes.DeviceType.GPU)
