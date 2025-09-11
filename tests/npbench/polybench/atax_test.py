# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
# M, N
sizes = {
    "mini": (38, 42),
    "small": (116, 124),
    "medium": (390, 410),
    "large": (1900, 2100),
    "extra-large": (1800, 2200)
}

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float32[M, N], x: dc.float32[N]):
    return (A @ x) @ A


def init_data(M, N):
    fn = np.float32(N)
    A = np.empty((M, N), dtype=np.float32)
    x = np.empty((N, ), dtype=np.float32)
    y = np.empty((N, ), dtype=np.float32)
    for i in range(N):
        x[i] = 1 + (i / fn)
    for i in range(M):
        for j in range(N):
            A[i, j] = ((i + j) % N) / (5 * M)
    return A, x, y


def atax_jax_kernel(A, x, B, S):
    B = (A @ x) @ A
    return jnp.sum(B)


def run_atax(device_type: dace.dtypes.DeviceType):
    """
    Runs ATAX for the given device

    :return: the SDFG
    """

    # Initialize data (polybench small size)
    M, N = sizes["small"]
    A, x, y_ref = init_data(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        y = sdfg(A, x, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory],
                                                         [{}, {
                                                             'storage': dace.StorageType.FPGA_Local
                                                         }],
                                                         print_report=True)
        assert sm_applied == 6  # 3 inlines and 3 Streaming memories

        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(M=M, N=N))
        y = sdfg(A, x)

    # Compute ground truth and Validate result
    y_ref = kernel.f(A, x)
    assert np.allclose(y, y_ref)
    return sdfg


def run_atax_autodiff():
    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A, x, y = init_data(M, N)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float32)
    B = np.zeros((N,), dtype=np.float32)
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float32[M, N], x: dc.float32[N]):
        y = (A @ x) @ A
        return np.sum(y)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=False)
    sdfg(A, x, M=M, N=N, gradient_A=gradient_A, gradient___return=gradient___return)
    
    # Enable float32 support (matching the original test data type)
    jax.config.update("jax_enable_x64", False)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(atax_jax_kernel, argnums=0))
    A_jax = A.astype(np.float32)
    x_jax = x.astype(np.float32) 
    B_jax = B.astype(np.float32)
    S_jax = S.astype(np.float32)
    jax_grad_A = jax_grad(A_jax, x_jax, B_jax, S_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-5, atol=1e-6)


def test_cpu():
    run_atax(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_atax(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_atax_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_atax(dace.dtypes.DeviceType.FPGA)


@xilinx_test(assert_ii_1=False)
def test_xilinx_decoupled_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return run_atax(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_atax(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_atax(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_atax(dace.dtypes.DeviceType.FPGA)
