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

from tests.npbench.polybench.jacobi_2d_test import run_jacobi_2d_autodiff

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 30), "small": (40, 120), "medium": (100, 400), "large": (500, 2000), "extra-large": (1000, 4000)}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def jacobi_1d_kernel(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def jacobi_1d_jax_kernel(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B = B.at[1:-1].set(0.33333 * (A[:-2] + A[1:-1] + A[2:]))
        A = A.at[1:-1].set(0.33333 * (B[:-2] + B[1:-1] + B[2:]))

    return jax.block_until_ready(jnp.sum(A))


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i: (i + 2) / N, (N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N, ), dtype=datatype)

    return A, B


def ground_truth(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def run_jacobi_1d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Jacobi 1d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    TSTEPS, N = sizes["small"]
    A, B = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = jacobi_1d_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS, A, B, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = jacobi_1d_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(TSTEPS=TSTEPS, A=A, B=B)

    # Compute ground truth and validate
    ground_truth(TSTEPS, A_ref, B_ref)
    assert np.allclose(A, A_ref)
    return sdfg


def run_jacobi_1d_autodiff():
    # Initialize data (polybench mini size)
    TSTEPS, N = (20, 30)
    A, B = initialize(N)
    jax_A, jax_B = np.copy(A), np.copy(B)

    # Intiialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):
        jacobi_1d_kernel(TSTEPS, A, B)
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(TSTEPS, A, B, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(jacobi_1d_jax_kernel, argnums=1), static_argnums=0)
    jax_grad_A = jax_grad(TSTEPS, jax_A, jax_B)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_jacobi_1d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_jacobi_1d(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    run_jacobi_1d_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_jacobi_1d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_jacobi_1d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_jacobi_1d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_jacobi_1d(dace.dtypes.DeviceType.FPGA)
