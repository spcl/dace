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
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def symm_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N], A: dc.float64[M, M], B: dc.float64[M, N]):

    temp2 = np.empty((N, ), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i + j) % 100) / M, (M, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((N + i - j) % 100) / M, (M, N), dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, :i + 1] = np.fromfunction(lambda j: ((i + j) % 100) / M, (i + 1, ), dtype=datatype)
        A[i, i + 1:] = -999

    return alpha, beta, C, A, B


def symm_jax_kernel(jnp, lax, alpha, beta, C, A, B):
    temp2 = jnp.empty((C.shape[1], ), dtype=C.dtype)
    C = C * beta

    def row_update_body(carry, i):
        C, temp2 = carry

        def col_update_body(carry_inner, j):
            C, temp2 = carry_inner

            A_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0.0)
            B_slice = jnp.where(jnp.arange(B.shape[0]) < i, B[:, j], 0.0)

            updated_col = C[:, j] + (alpha * B[i, j] * A_slice)
            C = lax.dynamic_update_slice(C, updated_col[:, None], (0, j))
            temp2 = temp2.at[j].set(B_slice @ A_slice)
            return (C, temp2), jnp.array(0)

        (C, temp2), _ = lax.scan(col_update_body, (C, temp2), jnp.arange(C.shape[1]))
        C = C.at[i, :].add(alpha * B[i, :] * A[i, i] + alpha * temp2)
        return (C, temp2), jnp.array(0)

    (C, temp2), _ = lax.scan(row_update_body, (C, temp2), jnp.arange(C.shape[0]))
    return jnp.sum(C)


def ground_truth(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


def run_symm(device_type: dace.dtypes.DeviceType):
    '''
    Runs Symm for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A, B = initialize(M, N)
    C_ref = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = symm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, C, A, B, M=M, N=N)
    # Compute ground truth and validate
    ground_truth(alpha, beta, C_ref, A, B)
    assert np.allclose(C, C_ref)
    return sdfg


def run_symm_autodiff():
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A, B = initialize(M, N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N], A: dc.float64[M, M],
                        B: dc.float64[M, N]):
        symm_kernel(alpha, beta, C, A, B)
        return np.sum(C)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(alpha, beta, C, A, B, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_kernel = lambda alpha, beta, C, A, B: symm_jax_kernel(jnp, lax, alpha, beta, C, A, B)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=3), static_argnums=(0, 1))
    alpha, beta, C_jax, A_jax, B_jax = initialize(M, N)
    jax_grad_A = jax_grad(alpha, beta, C_jax, A_jax, B_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_symm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_symm(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_symm_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_symm(dace.dtypes.DeviceType.CPU)
        run_symm_autodiff()
    elif target == "gpu":
        run_symm(dace.dtypes.DeviceType.GPU)
