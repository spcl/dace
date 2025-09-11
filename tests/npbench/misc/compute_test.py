# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


M, N = (dace.symbol(s, dtype=dace.int64) for s in ('M', 'N'))


@dace.program
def compute(array_1: dace.int64[M, N], array_2: dace.int64[M, N], a: dace.int64, b: dace.int64, c: dace.int64):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c


def initialize(M, N):
    from numpy.random import default_rng
    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    return array_1, array_2, a, b, c


def run_compute(device_type: dace.dtypes.DeviceType):
    '''
    Runs compute for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    M, N = (2000, 2000)
    array_1, array_2, a, b, c = initialize(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = compute.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val = sdfg(array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = compute.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        sdfg.expand_library_nodes()

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

        sdfg.specialize(dict(M=M, N=N))
        val = sdfg(array_1=array_1, array_2=array_2, a=a, b=b, c=c)

    # Compute ground truth and Validate result
    ref = compute.f(array_1, array_2, a, b, c)
    assert (np.allclose(val, ref) or relerror(val, ref) < 1e-10)
    return sdfg


def compute_jax_kernel(array_1, array_2, a, b, c, S):
    return jnp.sum(jnp.minimum(jnp.maximum(array_1, 2), 10) * a + array_2 * b + c)

def run_compute_autodiff():
    # Initialize forward data (using smaller size for AD test)
    M, N = (20, 20)
    array_1, array_2, a, b, c = initialize(M, N)
    # Convert to float64 for AD
    array_1 = array_1.astype(np.float64)
    array_2 = array_2.astype(np.float64)
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float64)
    gradient_array_2 = np.zeros_like(array_2)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dace.program
    def autodiff_kernel(array_1: dace.float64[M, N], array_2: dace.float64[M, N], 
                        a: dace.float64, b: dace.float64, c: dace.float64):
        result = compute(array_1, array_2, a, b, c)
        return np.sum(result)
    
    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["array_2"], outputs=["__return"], autooptimize=True)
    sdfg(array_1, array_2, a, b, c, M=M, N=N, gradient_array_2=gradient_array_2, gradient___return=gradient___return)
    
    # Enable float64 support
    jax.config.update("jax_enable_x64", True)
    
    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(compute_jax_kernel, argnums=1))
    jax_grad_array_2 = jax_grad(array_1, array_2, a, b, c, S)
    np.testing.assert_allclose(gradient_array_2, jax_grad_array_2, rtol=1e-5, atol=1e-8)

def test_cpu():
    run_compute(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_compute(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_compute_autodiff()

@pytest.mark.skip(reason="Compiler error")
@fpga_test(assert_ii_1=False)
def test_fpga():
    run_compute(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_compute(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_compute(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_compute(dace.dtypes.DeviceType.FPGA)
