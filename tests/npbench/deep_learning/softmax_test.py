# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

N, H, SM = (dc.symbol(s, dc.int64) for s in ('N', 'H', 'SM'))


# Numerically-stable version of softmax
@dc.program
def softmax_kernel(x: dc.float32[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x


def ground_truth(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def softmax_jax_kernel(jnp, x):
    tmp_max = jnp.max(x, axis=-1, keepdims=True)
    tmp_out = jnp.exp(x - tmp_max)
    tmp_sum = jnp.sum(tmp_out, axis=-1, keepdims=True)
    return jnp.sum(tmp_out / tmp_sum)


def run_softmax(device_type: dace.dtypes.DeviceType):
    '''
    Runs Softmax for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N, H, SM = 16, 16, 128
    x = initialize(N, H, SM)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = softmax_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(x, N=N, H=H, SM=SM)

    # Compute ground truth and validate
    out_ref = ground_truth(x)
    assert np.allclose(out, out_ref)
    return sdfg


def run_softmax_autodiff():
    import jax
    import jax.numpy as jnp

    # Initialize data (npbench test size)
    N, H, SM = 4, 4, 32
    x = initialize(N, H, SM)
    out = np.zeros_like(x)

    # Initialize gradient computation data
    gradient_x = np.zeros_like(x)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def softmax_autodiff_kernel(x: dc.float32[N, H, SM, SM]):
        return np.sum(softmax_kernel(x))

    # Add the backward pass to the SDFG
    sdfg = softmax_autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["x"], outputs=["__return"])
    sdfg(x, out, N=N, H=H, SM=SM, gradient_x=gradient_x, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_kernel = lambda x: softmax_jax_kernel(jnp, x)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=0))
    jax_grad_x = jax_grad(x)
    np.testing.assert_allclose(gradient_x, jax_grad_x, atol=1e-6)


def test_cpu():
    run_softmax(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_softmax(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_softmax_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_softmax(dace.dtypes.DeviceType.CPU)
        run_softmax_autodiff()
    elif target == "gpu":
        run_softmax(dace.dtypes.DeviceType.GPU)
