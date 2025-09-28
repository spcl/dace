# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusionVertical, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
import jax.lax as lax

# M, N
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, C: dc.float32[N, N], A: dc.float32[N, M]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def init_data(N, M):

    alpha = np.float32(1.5)
    beta = np.float32(1.2)
    C = np.empty((N, N), dtype=np.float32)
    A = np.empty((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            A[i, j] = ((i * j + 1) % N) / N
    for i in range(N):
        for j in range(N):
            C[i, j] = ((i * j + 2) % M) / M

    return alpha, beta, C, A


def syrk_jax_kernel(alpha, beta, C, A):
    m = A.shape[0]  # number of rows
    n = A.shape[1]  # number of columns

    def outer_body_fun(carry, i):
        # Unpack outer loop carry.
        alpha, beta, C, A = carry

        # Outer loop update: scale row i of C by beta for indices < i+1.
        col_mask = jnp.arange(C.shape[1]) < (i + 1)
        C_slice = jnp.where(col_mask, C[i, :], 0.0)
        C_slice = C_slice * beta
        # Preserve the original values for indices >= i+1.
        C_slice = jnp.where(col_mask, C_slice, C[i, :])
        C = lax.dynamic_update_slice(C, C_slice[None, :], (i, 0))

        # Define the inner loop which updates row i of C using column updates from A.
        def inner_body_fun(inner_carry, k):
            alpha_inner, C_inner, A_inner = inner_carry

            # Compute an update slice from A[:, k] for rows < i+1.
            row_mask = jnp.arange(A_inner.shape[0]) < (i + 1)
            A_update_slice = jnp.where(row_mask, A_inner[:, k], 0.0)
            A_update_slice = A_update_slice * (alpha_inner * A_inner[i, k])

            # Update C[i, :] by adding the A_update_slice, only for columns < i+1.
            col_mask_inner = jnp.arange(C_inner.shape[1]) < (i + 1)
            C_update_slice = jnp.where(col_mask_inner, C_inner[i, :], 0.0)
            C_update_slice = C_update_slice + A_update_slice
            C_update_slice = jnp.where(col_mask_inner, C_update_slice, C_inner[i, :])
            C_inner = lax.dynamic_update_slice(C_inner, C_update_slice[None, :], (i, 0))
            return (alpha_inner, C_inner, A_inner), None

        # Run the inner loop over k = 0,..., n-1.
        (alpha, C, A), _ = lax.scan(inner_body_fun, (alpha, C, A), jnp.arange(n))
        return (alpha, beta, C, A), None

    # Run the outer loop over i = 0,..., m-1.
    (alpha, beta, C, A), _ = lax.scan(outer_body_fun, (alpha, beta, C, A), jnp.arange(m))
    return jnp.sum(C)


def ground_truth(N, M, alpha, beta, C, A):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def run_syrk(device_type: dace.dtypes.DeviceType):
    """
    Runs Syrk for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, beta, C, A = init_data(N, M)
    gt_C = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha=alpha, beta=beta, C=C, A=A, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)
        sdfg.specialize(dict(N=N, M=M))
        # run program
        sdfg(alpha=alpha, beta=beta, C=C, A=A)

    # Compute ground truth and validate result
    ground_truth(N, M, alpha, beta, gt_C, A)
    assert np.allclose(C, gt_C)
    return sdfg


def run_syrk_autodiff():
    # Initialize data (polybench mini size) - note the order swap for this test
    M, N = sizes["mini"]
    alpha, beta, C, A = init_data(N, M)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float32, beta: dc.float32, C: dc.float32[N, N], A: dc.float32[N, M]):
        kernel(alpha, beta, C, A)
        return np.sum(C)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(alpha=alpha, beta=beta, C=C, A=A, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(syrk_jax_kernel, argnums=3), static_argnums=(0, 1))
    alpha, beta, C_jax, A_jax = init_data(N, M)
    jax_grad_A = jax_grad(alpha, beta, C_jax, A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-6, atol=1e-5)


def test_cpu():
    run_syrk(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_syrk(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    run_syrk_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_syrk(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_syrk(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_syrk(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_syrk(dace.dtypes.DeviceType.FPGA)
