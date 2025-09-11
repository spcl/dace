# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
import jax.lax as lax

# Data set sizes
# M, N
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def syr2k_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N], A: dc.float64[N, M], B: dc.float64[N, M]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k])


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 3) % N) / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % M) / M, (N, M), dtype=datatype)

    return alpha, beta, C, A, B


def syr2k_jax_kernel(alpha, beta, C, A, B):
    m = A.shape[0]  # outer loop range
    n = A.shape[1]  # inner loop range

    def outer_body_fun(carry, i):
        # Unpack loop variables for the outer loop.
        alpha, beta, C, A, B = carry

        # Outer-loop update: scale row i of C by beta, but only for columns < i+1.
        C_slice = jnp.where(jnp.arange(C.shape[1]) < (i + 1),
                            C[i, :],
                            0.0)
        C_slice = C_slice * beta
        C_slice = jnp.where(jnp.arange(C.shape[1]) < (i + 1),
                            C_slice,
                            C[i, :])
        C = lax.dynamic_update_slice(C, C_slice[None, :], (i, 0))

        # Define the inner scan that will update row i of C using index k.
        def inner_body_fun(inner_carry, k):
            # Unpack inner loop variables.
            alpha_inner, C_inner, A_inner, B_inner = inner_carry

            # For A_update_slice and B_update_slice, only entries for indices < i+1 are used.
            A_update_slice = jnp.where(jnp.arange(A_inner.shape[0]) < (i + 1),
                                       A_inner[:, k],
                                       0.0)
            A_update_slice = A_update_slice * (alpha_inner * B_inner[i, k])

            B_update_slice = jnp.where(jnp.arange(B_inner.shape[0]) < (i + 1),
                                       B_inner[:, k],
                                       0.0)
            B_update_slice = B_update_slice * (alpha_inner * A_inner[i, k])

            # Compute an update for row i of C: take its current values (only for indices < i+1)
            # and add the contributions from A_update_slice and B_update_slice.
            C_update_slice = jnp.where(jnp.arange(C_inner.shape[1]) < (i + 1),
                                       C_inner[i, :],
                                       0.0)
            C_update_slice = C_update_slice + A_update_slice + B_update_slice
            # For indices not less than i+1, keep the original C[i, :].
            C_update_slice = jnp.where(jnp.arange(C_inner.shape[1]) < (i + 1),
                                       C_update_slice,
                                       C_inner[i, :])
            # Update row i of C.
            C_inner = lax.dynamic_update_slice(C_inner, C_update_slice[None, :], (i, 0))
            return (alpha_inner, C_inner, A_inner, B_inner), None

        # Run the inner scan over k from 0 to n-1.
        (alpha, C, A, B), _ = lax.scan(inner_body_fun, (alpha, C, A, B), jnp.arange(n))
        return (alpha, beta, C, A, B), None

    # Run the outer scan over i from 0 to m-1.
    (alpha, beta, C, A, B), _ = lax.scan(outer_body_fun, (alpha, beta, C, A, B), jnp.arange(m))
    return jnp.sum(C)


def ground_truth(alpha, beta, C, A, B):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k])


def run_syr2k(device_type: dace.dtypes.DeviceType):
    '''
    Runs Syr2k for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A, B = initialize(M, N)
    C_ref = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = syr2k_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, C, A, B, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = syr2k_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # No libnodes expansion for this kernel
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(M=M, N=N))
        sdfg(alpha, beta, C, A, B)

    # Compute ground truth and validate
    ground_truth(alpha, beta, C_ref, A, B)
    assert np.allclose(C, C_ref)
    return sdfg


def run_syr2k_autodiff():
    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A, B = initialize(M, N)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float64)
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N], A: dc.float64[N, M], B: dc.float64[N, M]):
        syr2k_kernel(alpha, beta, C, A, B)
        return np.sum(C)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=False)
    sdfg(alpha, beta, C, A, B, S, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)
    
    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(syr2k_jax_kernel, argnums=3), static_argnums=(0, 1))
    A_jax = A.astype(np.float64)
    B_jax = B.astype(np.float64)
    C_jax = np.copy(initialize(M, N)[2]).astype(np.float64)  # Fresh copy of C
    S_jax = S.astype(np.float64)
    jax_grad_A = jax_grad(alpha, beta, C_jax, A_jax, B_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-5, atol=1e-8)


def test_cpu():
    run_syr2k(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_syr2k(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_syr2k_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_syr2k(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_syr2k(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_syr2k(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_syr2k(dace.dtypes.DeviceType.FPGA)
