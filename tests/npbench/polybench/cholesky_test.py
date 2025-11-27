# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

# Data set sizes
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def kernel(A: dc.float32[N, N]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


def init_data(N):
    A = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            A[i, j] = (-j % N) / N + 1
        for j in range(i + 1, N):
            A[i, j] = 0.0
        A[i, i] = 1.0

    A[:] = A @ np.transpose(A)
    return A


def cholesky_jax_kernel(jnp, lax, A):
    A = A.at[0, 0].set(jnp.sqrt(A[0, 0]))

    def row_update_body(A, i):

        def col_update_body(A, j):

            def do_update(_):
                mask = jnp.arange(A.shape[1]) < j
                A_i_slice = jnp.where(mask, A[i, :], 0)
                A_j_slice = jnp.where(mask, A[j, :], 0)
                dot_product = jnp.dot(A_i_slice, A_j_slice)
                new_val = (A[i, j] - dot_product) / A[j, j]
                return A.at[i, j].set(new_val)

            A = lax.cond(j < i, do_update, lambda _: A, operand=None)
            return A, None

        A, _ = lax.scan(col_update_body, A, jnp.arange(A.shape[0]))

        mask = jnp.arange(A.shape[1]) < i
        A_i_slice = jnp.where(mask, A[i, :], 0)
        dot_product = jnp.dot(A_i_slice, A_i_slice)
        A = A.at[i, i].set(jnp.sqrt(A[i, i] - dot_product))
        return A, None

    A, _ = lax.scan(row_update_body, A, jnp.arange(1, A.shape[0]))
    return jnp.sum(A)


def ground_truth(N, A):
    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


def run_cholesky(device_type: dace.dtypes.DeviceType):
    """
    Runs Cholesky for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    A = init_data(N)
    gt_A = np.copy(A)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A=A, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        platform = dace.config.Config.get("compiler", "fpga", "vendor")
        if platform == "intel_fpga":
            Dot.default_implementation = "FPGA_Accumulate"
        else:
            Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

        sdfg(A=A, N=N)

    # Compute ground truth and validate result
    ground_truth(N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    assert diff < 1e-6
    return sdfg


def run_cholesky_autodiff():
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    # Initialize data (polybench mini size)
    N = 20
    A = init_data(N)
    A_jax = jnp.copy(A)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float32[N, N]):
        kernel(A)
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg(simplify=True)
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(A, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_kernel = lambda A: cholesky_jax_kernel(jnp, lax, A)
    jax_grad = jax.jit(jax.grad(jax_kernel))
    jax_grad_A = jax_grad(A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-4, atol=1e-4)


def test_cpu():
    run_cholesky(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_cholesky(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_cholesky_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_cholesky(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_cholesky(dace.dtypes.DeviceType.CPU)
        run_cholesky_autodiff()
    elif target == "gpu":
        run_cholesky(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_cholesky(dace.dtypes.DeviceType.FPGA)
