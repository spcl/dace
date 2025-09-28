# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
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
from dace.config import set_temporary
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
import jax.lax as lax

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 40), "small": (40, 120), "medium": (100, 400), "large": (500, 2000), "extra-large": (1000, 4000)}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def seidel_2d_kernel(TSTEPS: dc.int64, A: dc.float64[N, N]):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 2) + 2) / N, (N, N), dtype=datatype)

    return A


def seidel_2d_jax_kernel(TSTEPS, A):
    """JAX implementation using efficient lax.scan operations"""
    N = A.shape[0]

    def loop1_body(A, t):

        def loop2_body(A, i):
            update_val = (A[i, 1:-1] + (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] +
                                        A[i + 1, 1:-1] + A[i + 1, 2:]))
            A = A.at[i, 1:-1].set(update_val)

            def loop3_body(A, j):
                new_val = (A[i, j] + A[i, j - 1]) / 9.0
                A = A.at[i, j].set(new_val)
                return A, None

            A, _ = lax.scan(loop3_body, A, jnp.arange(1, N - 1))
            return A, None

        A, _ = lax.scan(loop2_body, A, jnp.arange(1, N - 1))
        return A, None

    A, _ = lax.scan(loop1_body, A, jnp.arange(TSTEPS - 1))
    return jnp.sum(A)


def ground_truth(TSTEPS, N, A):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def run_seidel_2d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Seidel 2d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    TSTEPS, N = sizes["mini"]
    A = initialize(N)
    A_ref = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = seidel_2d_kernel.to_sdfg()
        # sdfg = auto_optimize(sdfg, device_type) # TBD
        sdfg(TSTEPS, A, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = seidel_2d_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(TSTEPS, A)

    # Compute ground truth and validate
    ground_truth(
        TSTEPS,
        N,
        A_ref,
    )
    print(np.linalg.norm(A - A_ref) / np.linalg.norm(A_ref))
    assert np.allclose(A, A_ref)
    return sdfg


def run_seidel_2d_autodiff():
    # Initialize data (test size for efficiency)
    TSTEPS, N = (2, 8)
    A = initialize(N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output using __return pattern
    @dc.program
    def autodiff_kernel(TSTEPS: dc.int64, A: dc.float64[N, N]):
        for t in range(0, TSTEPS - 1):
            for i in range(1, N - 1):
                A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] +
                               A[i + 1, 1:-1] + A[i + 1, 2:])
                for j in range(1, N - 1):
                    A[i, j] += A[i, j - 1]
                    A[i, j] /= 9.0
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(TSTEPS, A, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(seidel_2d_jax_kernel, argnums=1), static_argnums=(0, ))
    A_jax = initialize(N)
    jax_grad_A = jax_grad(TSTEPS, A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_seidel_2d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_seidel_2d(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    run_seidel_2d_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_seidel_2d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_seidel_2d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_seidel_2d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_seidel_2d(dace.dtypes.DeviceType.FPGA)
