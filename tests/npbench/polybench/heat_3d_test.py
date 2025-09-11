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

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 10), "small": (40, 20), "medium": (100, 40), "large": (500, 120), "extra-large": (1000, 200)}

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def heat_3d_kernel(TSTEPS: dc.int64, A: dc.float64[N, N, N], B: dc.float64[N, N, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N), dtype=datatype)
    B = np.copy(A)

    return A, B


def heat_3d_jax_kernel(TSTEPS, A, B, S):
    def time_step(carry, t):
        A, B = carry
        # First, update B using the current A.
        B_new = B.at[1:-1, 1:-1, 1:-1].set(
            0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) +
            0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) +
            0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, :-2]) +
            A[1:-1, 1:-1, 1:-1]
        )
        # Then, update A using the new B.
        A_new = A.at[1:-1, 1:-1, 1:-1].set(
            0.125 * (B_new[2:, 1:-1, 1:-1] - 2.0 * B_new[1:-1, 1:-1, 1:-1] + B_new[:-2, 1:-1, 1:-1]) +
            0.125 * (B_new[1:-1, 2:, 1:-1] - 2.0 * B_new[1:-1, 1:-1, 1:-1] + B_new[1:-1, :-2, 1:-1]) +
            0.125 * (B_new[1:-1, 1:-1, 2:] - 2.0 * B_new[1:-1, 1:-1, 1:-1] + B_new[1:-1, 1:-1, :-2]) +
            B_new[1:-1, 1:-1, 1:-1]
        )
        return (A_new, B_new), None

    # Scan over time steps from 1 to TSTEPS-1.
    (A_final, B_final), _ = lax.scan(time_step, (A, B), jnp.arange(1, TSTEPS))
    return jnp.sum(A_final)


def ground_truth(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def run_heat_3d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gesummv for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    TSTEPS, N = sizes["small"]
    A, B = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)

    def count_maps(sdfg: dc.SDFG) -> int:
        nb_maps = 0
        for _, state in sdfg.all_nodes_recursive():
            node: dc.SDFGState
            for node in state.nodes():
                if isinstance(node, dc.sdfg.nodes.MapEntry):
                    nb_maps += 1
        return nb_maps

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = heat_3d_kernel.to_sdfg()
        initial_maps = count_maps(sdfg)
        sdfg = auto_optimize(sdfg, device_type)
        after_maps = count_maps(sdfg)
        assert after_maps < initial_maps, f"Expected less maps, initially {initial_maps} many maps, but after optimization {after_maps}"
        sdfg(TSTEPS, A, B, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = heat_3d_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(TSTEPS, A, B)

    # Compute ground truth and validate
    ground_truth(TSTEPS, A_ref, B_ref)
    assert np.allclose(A, A_ref)
    return sdfg


def run_heat_3d_autodiff():
    # Initialize data (polybench small size)
    TSTEPS, N = sizes["small"]
    A, B = initialize(N)
    
    # Initialize gradient computation data
    S = np.zeros((1,), dtype=np.float64)
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones_like(S)
    
    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(TSTEPS: dc.int64, A: dc.float64[N, N, N], B: dc.float64[N, N, N]):
        heat_3d_kernel(TSTEPS, A, B)
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"], autooptimize=False)
    sdfg(TSTEPS, A, B, S, N=N, gradient_A=gradient_A, gradient___return=gradient___return)
    
    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(heat_3d_jax_kernel, argnums=1), static_argnums=(0,))
    A_jax = np.copy(initialize(N)[0]).astype(np.float64)  # Fresh copy of A
    B_jax = np.copy(initialize(N)[1]).astype(np.float64)  # Fresh copy of B
    S_jax = S.astype(np.float64)
    jax_grad_A = jax_grad(TSTEPS, A_jax, B_jax, S_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-5, atol=1e-8)


def test_cpu():
    run_heat_3d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_heat_3d(dace.dtypes.DeviceType.GPU)


@pytest.mark.daceml
def test_autodiff():
    run_heat_3d_autodiff()


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_heat_3d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_heat_3d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_heat_3d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_heat_3d(dace.dtypes.DeviceType.FPGA)
