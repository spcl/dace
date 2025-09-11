# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
import jax.lax as lax

# Data set sizes
# M, N
# Note: these have been swapped to improve numerical stability
sizes = {"mini": (30, 20), "small": (80, 60), "medium": (240, 200), "large": (1200, 1000), "extra-large": (2600, 2000)}

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))


@dc.program
def gramschmidt_kernel(A: dc.float64[M, N]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


def initialize(M, N, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return A


def gramschmidt_jax_kernel(A):
    n = A.shape[1]
    Q = jnp.zeros_like(A)
    R = jnp.zeros((n, n), dtype=A.dtype)

    # Outer loop: iterate over k = 0, 1, ..., n-1.
    def body_fun(carry, k):
        Q, R, A = carry

        # Compute the norm for the k-th column and update R and Q.
        nrm = jnp.dot(A[:, k], A[:, k])
        R = R.at[k, k].set(jnp.sqrt(nrm))
        Q = Q.at[:, k].set(A[:, k] / R[k, k])

        # Inner loop: iterate over j = 0,1,...,n-1, but update only when j >= k+1.
        def inner_body_fun(carry_inner, j):
            Q, R, A = carry_inner

            def do_update(_):
                # Update R[k, j] with dot(Q[:, k], A[:, j])
                new_R = R.at[k, j].set(jnp.dot(Q[:, k], A[:, j]))
                # Then update A[:, j] by subtracting Q[:, k] * new_R[k, j]
                new_A = A.at[:, j].add(-Q[:, k] * new_R[k, j])
                return (Q, new_R, new_A)

            def no_update(_):
                return (Q, R, A)

            # Only perform the update if j >= k+1.
            Q, R, A = lax.cond(j >= (k + 1), do_update, no_update, operand=None)
            return (Q, R, A), None

        (Q, R, A), _ = lax.scan(inner_body_fun, (Q, R, A), jnp.arange(n))
        return (Q, R, A), None

    (Q, R, A), _ = lax.scan(body_fun, (Q, R, A), jnp.arange(n))
    return jnp.sum(A)


def ground_truth(A):

    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


def run_gramschmidt(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gesummv for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A = initialize(M, N)
    A_ref = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gramschmidt_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        Q, R = sdfg(A, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gramschmidt_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(M=M, N=N))
        Q, R = sdfg(A)

    # Compute ground truth and validate
    Q_ref, R_ref = ground_truth(A_ref)
    assert np.allclose(Q, Q_ref)
    assert np.allclose(R, R_ref)
    return sdfg


def run_gramschmidt_autodiff():
    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A = initialize(M, N)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float64)
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float64[M, N]):
        Q, R = gramschmidt_kernel(A)
        return np.sum(A)  # Sum the modified A matrix

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=False)
    sdfg(A, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)
    
    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(gramschmidt_jax_kernel))
    A_jax = np.copy(initialize(M, N)).astype(np.float64)  # Fresh copy of A
    S_jax = S.astype(np.float64)
    jax_grad_A = jax_grad(A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-5, atol=1e-8)


def test_cpu():
    run_gramschmidt(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gramschmidt(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_gramschmidt_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_gramschmidt(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gramschmidt(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gramschmidt(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gramschmidt(dace.dtypes.DeviceType.FPGA)
