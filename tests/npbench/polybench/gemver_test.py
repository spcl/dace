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

# Data set sizes
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def gemver_kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N],
                  u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N],
                  z: dc.float64[N]):

    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N, ), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N, ), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N, ), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N, ), dtype=datatype)
    w = np.zeros((N, ), dtype=datatype)
    x = np.zeros((N, ), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N, ), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N, ), dtype=datatype)

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


def gemver_jax_kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += jnp.outer(u1, v1) + jnp.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
    return jnp.sum(w)


def run_gemver(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gemver for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["small"]
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = initialize(N)
    A_ref = np.copy(A)
    w_ref = np.copy(w)
    x_ref = np.copy(x)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gemver_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, A, np.copy(u1), v1, u2, v2, w, x, y, z, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gemver_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(alpha, beta, A, np.copy(u1), v1, u2, v2, w, x, y, z)

    # Compute ground truth and validate
    gemver_kernel.f(alpha, beta, A_ref, u1, v1, u2, v2, w_ref, x_ref, y, z)
    assert np.allclose(A, A_ref)
    assert np.allclose(x, x_ref)
    assert np.allclose(w, w_ref)

    return sdfg


def run_gemver_autodiff():
    # Initialize data (polybench mini size)
    N = sizes["mini"]
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = initialize(N)
    A_jax, u1_jax, v1_jax, u2_jax, v2_jax, w_jax, x_jax, y_jax, z_jax = map(np.copy, (A, u1, v1, u2, v2, w, x, y, z))

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N],
                        u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N],
                        z: dc.float64[N]):
        gemver_kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
        return np.sum(w)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(alpha,
         beta,
         A,
         np.copy(u1),
         v1,
         u2,
         v2,
         w,
         x,
         y,
         z,
         N=N,
         gradient_A=gradient_A,
         gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(gemver_jax_kernel, argnums=2))
    jax_grad_A = jax_grad(alpha, beta, A_jax, u1_jax, v1_jax, u2_jax, v2_jax, w_jax, x_jax, y_jax, z_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_gemver(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gemver(dace.dtypes.DeviceType.GPU)


@pytest.mark.ad
def test_autodiff():
    run_gemver_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_gemver(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gemver(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gemver(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gemver(dace.dtypes.DeviceType.FPGA)
