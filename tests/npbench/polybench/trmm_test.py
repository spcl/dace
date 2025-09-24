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
sizes = {"mini": (20, 30), "small": (60, 80), "medium": (200, 240), "large": (1000, 1200), "extra-large": (2000, 2600)}

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))


@dc.program
def trmm_kernel(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N]):

    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N), dtype=datatype)

    return alpha, A, B


def trmm_jax_kernel(alpha, A, B):

    def outer_body(carry, i):
        B = carry

        def inner_body(B, j):

            mask = (jnp.arange(A.shape[0]) > i).astype(A.dtype)
            dot_val = jnp.sum(A[:, i] * B[:, j] * mask)
            new_val = B[i, j] + dot_val
            B = B.at[i, j].set(new_val)
            return B, jnp.array(0)

        B, _ = lax.scan(inner_body, B, jnp.arange(B.shape[1]))
        return B, jnp.array(0)

    B, _ = lax.scan(outer_body, B, jnp.arange(B.shape[0]))
    B = B * alpha
    return jnp.sum(B)


def ground_truth(alpha, A, B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha


def run_trmm(device_type: dace.dtypes.DeviceType):
    '''
    Runs trmm for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, A, B = initialize(M, N)
    B_ref = np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = trmm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, A, B, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = trmm_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(M=M, N=N))
        sdfg(alpha, A, B)

    # Compute ground truth and validate
    ground_truth(alpha, A, B_ref)
    assert np.allclose(B, B_ref)
    return sdfg


def run_trmm_autodiff():
    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    alpha, A, B = initialize(M, N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N]):
        trmm_kernel(alpha, A, B)
        return np.sum(B)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=True)
    sdfg(alpha, A, B, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(trmm_jax_kernel, argnums=1), static_argnums=(0, ))
    alpha, A_jax, B_jax = initialize(M, N)
    jax_grad_A = jax_grad(alpha, A_jax, B_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_trmm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_trmm(dace.dtypes.DeviceType.GPU)


@pytest.mark.ad
def test_autodiff():
    run_trmm_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_trmm(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_trmm(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_trmm(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_trmm(dace.dtypes.DeviceType.FPGA)
