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
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def ludcmp_kernel(A: dc.float64[N, N], b: dc.float64[N]):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x, y


def initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, :i + 1] = np.fromfunction(lambda j: (-j % N) / N + 1, (i + 1, ), dtype=datatype)
        A[i, i + 1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    fn = datatype(N)
    b = np.fromfunction(lambda i: (i + 1) / fn / 2.0 + 4.0, (N, ), dtype=datatype)

    return A, b


def ground_truth(A, b):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(A.shape[0]):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, A.shape[0]):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(A.shape[0]):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x, y


def ludcmp_jax_kernel(A, b):
    n = A.shape[0]
    x = jnp.zeros_like(b)
    y = jnp.zeros_like(b)

    def outer_loop_body_1(A, i):

        def inner_loop_1_body(A, j):

            def update():
                A_slice_1 = jnp.where(jnp.arange(n) < j, A[i, :], 0.0)
                A_slice_2 = jnp.where(jnp.arange(n) < j, A[:, j], 0.0)
                new_val = (A[i, j] - A_slice_1 @ A_slice_2) / A[j, j]
                return A.at[i, j].set(new_val)

            A = lax.cond(j < i, lambda _: update(), lambda _: A, operand=None)
            return A, None

        def inner_loop_2_body(A, j):

            def update():
                A_slice_1 = jnp.where(jnp.arange(n) < i, A[i, :], 0.0)
                A_slice_2 = jnp.where(jnp.arange(n) < i, A[:, j], 0.0)
                new_val = A[i, j] - A_slice_1 @ A_slice_2
                return A.at[i, j].set(new_val)

            A = lax.cond(j >= i, lambda _: update(), lambda _: A, operand=None)
            return A, None

        A, _ = lax.scan(inner_loop_1_body, A, jnp.arange(n))
        A, _ = lax.scan(inner_loop_2_body, A, jnp.arange(n))
        return A, None

    A, _ = lax.scan(outer_loop_body_1, A, jnp.arange(n))

    def loop_body_2_scan(loop_vars, i):
        A, y, b = loop_vars
        A_slice = jnp.where(jnp.arange(n) < i, A[i, :], 0.0)
        y_slice = jnp.where(jnp.arange(n) < i, y, 0.0)
        new_y = b[i] - A_slice @ y_slice
        y = y.at[i].set(new_y)
        return (A, y, b), None

    (A, y, b), _ = lax.scan(loop_body_2_scan, (A, y, b), jnp.arange(n))

    def loop_body_3_scan(loop_vars, t):
        A, x, y = loop_vars
        i = n - 1 - t  # reverse order
        A_slice = jnp.where(jnp.arange(n) > i, A[i, :], 0.0)
        x_slice = jnp.where(jnp.arange(n) > i, x, 0.0)
        new_x = (y[i] - A_slice @ x_slice) / A[i, i]
        x = x.at[i].set(new_x)
        return (A, x, y), None

    (A, x, y), _ = lax.scan(loop_body_3_scan, (A, x, y), jnp.arange(n))

    return jnp.sum(x)


def run_ludcmp_autodiff():
    # Initialize data (polybench mini size)
    N = sizes["mini"]
    A, b = initialize(N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float64[N, N], b: dc.float64[N]):
        x, y = ludcmp_kernel(A, b)
        return np.sum(x)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=True)
    sdfg(A, b, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(ludcmp_jax_kernel, argnums=0))
    A_jax, b_jax = initialize(N)
    jax_grad_A = jax_grad(A_jax, b_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def run_ludcmp(device_type: dace.dtypes.DeviceType):
    '''
    Runs Ludcmp for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    A, b = initialize(N)
    A_ref = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = ludcmp_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        x, y = sdfg(A, b, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = ludcmp_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        x, y = sdfg(A, b)

    # Compute ground truth and validate
    x_ref, y_ref = ground_truth(A_ref, b)
    assert np.allclose(x, x_ref)
    assert np.allclose(y, y_ref)
    return sdfg


def test_cpu():
    run_ludcmp(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_ludcmp(dace.dtypes.DeviceType.GPU)


@pytest.mark.ad
def test_autodiff():
    run_ludcmp_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_ludcmp(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_ludcmp(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_ludcmp(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_ludcmp(dace.dtypes.DeviceType.FPGA)
