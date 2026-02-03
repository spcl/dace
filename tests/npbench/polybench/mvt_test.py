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
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def mvt_kernel(x1: dc.float64[N], x2: dc.float64[N], y_1: dc.float64[N], y_2: dc.float64[N], A: dc.float64[N, N]):

    x1 += A @ y_1
    x2 += y_2 @ A


def initialize(N, datatype=np.float64):
    x1 = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, (N, ), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, (N, ), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, (N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)

    return x1, x2, y_1, y_2, A


def mvt_jax_kernel(jnp, x1, x2, y_1, y_2, A):
    x1 += A @ y_1
    x2 += y_2 @ A
    return jnp.sum(x2)


def run_mvt(device_type: dace.dtypes.DeviceType):
    '''
    Runs MVT for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["small"]
    x1, x2, y_1, y_2, A = initialize(N)
    x1_ref = np.copy(x1)
    x2_ref = np.copy(x2)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = mvt_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(x1, x2, y_1, y_2, A, N=N)
    # Compute ground truth and validate
    mvt_kernel.f(x1_ref, x2_ref, y_1, y_2, A)
    assert np.allclose(x1, x1_ref)
    assert np.allclose(x2, x2_ref)
    return sdfg


def run_mvt_autodiff():
    import jax
    import jax.numpy as jnp

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    x1, x2, y_1, y_2, A = initialize(N)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(x1: dc.float64[N], x2: dc.float64[N], y_1: dc.float64[N], y_2: dc.float64[N], A: dc.float64[N,
                                                                                                                    N]):
        mvt_kernel(x1, x2, y_1, y_2, A)
        return np.sum(x2)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(x1, x2, y_1, y_2, A, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_kernel = lambda x1, x2, y_1, y_2, A: mvt_jax_kernel(jnp, x1, x2, y_1, y_2, A)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=4))
    x1_jax, x2_jax, y_1_jax, y_2_jax, A_jax = initialize(N)
    jax_grad_A = jax_grad(x1_jax, x2_jax, y_1_jax, y_2_jax, A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_mvt(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_mvt(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_mvt_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_mvt(dace.dtypes.DeviceType.CPU)
        run_mvt_autodiff()
    elif target == "gpu":
        run_mvt(dace.dtypes.DeviceType.GPU)
