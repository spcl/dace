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
# NI, NJ, NK
sizes = {
    "mini": (20, 25, 30),
    "small": (60, 70, 80),
    "medium": (200, 220, 240),
    "large": (1000, 1100, 1200),
    "extra-large": (2000, 2300, 2600)
}

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))


@dc.program
def gemm_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ], A: dc.float64[NI, NK], B: dc.float64[NK,
                                                                                                                 NJ]):
    C[:] = alpha * A @ B + beta * C


def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


def gemm_jax_kernel(alpha, beta, A, B, C):
    return jnp.sum(alpha * A @ B + beta * C)


def run_gemm(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gemm for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    NI, NJ, NK = sizes["small"]
    alpha, beta, C, A, B = initialize(NI, NJ, NK)
    C_ref = np.copy(C)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gemm_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(alpha, beta, C, A, B, NI=NI, NJ=NJ, NK=NK)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gemm_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(NI=NI, NJ=NJ, NK=NK))
        sdfg(alpha, beta, C, A, B)

    # Compute ground truth and validate
    gemm_kernel.f(alpha, beta, C_ref, A, B)
    assert np.allclose(C, C_ref)
    return sdfg


def run_gemm_autodiff():
    # Initialize data (polybench mini size)
    NI, NJ, NK = sizes["mini"]
    alpha, beta, C, A, B = initialize(NI, NJ, NK)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ], A: dc.float64[NI, NK],
                        B: dc.float64[NK, NJ]):
        gemm_kernel(alpha, beta, C, A, B)
        return np.sum(C)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=True)
    sdfg(alpha, beta, C, A, B, NI=NI, NJ=NJ, NK=NK, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(gemm_jax_kernel, argnums=2), static_argnums=(0, 1))
    jax_grad_A = jax_grad(alpha, beta, A, B, C)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_gemm(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gemm(dace.dtypes.DeviceType.GPU)


@pytest.mark.ad
def test_autodiff():
    run_gemm_autodiff()


@fpga_test(assert_ii_1=False, xilinx=False)
def test_fpga():
    return run_gemm(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gemm(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gemm(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gemm(dace.dtypes.DeviceType.FPGA)
