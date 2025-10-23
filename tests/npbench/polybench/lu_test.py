# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusionVertical, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp
import jax.lax as lax

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def lu_kernel(A: dc.float32[N, N]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def ground_truth(N, A):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def init_data(N):

    A = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            A[i, j] = (-j % N) / N + 1
        for j in range(i + 1, N):
            A[i, j] = 0.0
        A[i, i] = 1.0

    B = np.empty((N, N), dtype=np.float32)
    B[:] = A @ np.transpose(A)
    A[:] = B

    return A


def lu_jax_kernel(A):
    n = A.shape[0]

    def outer_loop_body(A, i):

        def inner_loop_1_body(A, j):

            def update_fn(_):
                mask = jnp.arange(n) < j
                A_slice_1 = jnp.where(mask, A[i, :], 0.0)
                A_slice_2 = jnp.where(mask, A[:, j], 0.0)
                new_val = (A[i, j] - A_slice_1 @ A_slice_2) / A[j, j]
                return A.at[i, j].set(new_val)

            A = lax.cond(j < i, lambda _: update_fn(None), lambda _: A, operand=None)
            return A, None

        def inner_loop_2_body(A, j):

            def update_fn(_):
                mask = jnp.arange(n) < i
                A_slice_1 = jnp.where(mask, A[i, :], 0.0)
                A_slice_2 = jnp.where(mask, A[:, j], 0.0)
                new_val = A[i, j] - A_slice_1 @ A_slice_2
                return A.at[i, j].set(new_val)

            A = lax.cond(j >= i, lambda _: update_fn(None), lambda _: A, operand=None)
            return A, None

        A, _ = lax.scan(inner_loop_1_body, A, jnp.arange(n))
        A, _ = lax.scan(inner_loop_2_body, A, jnp.arange(n))
        return A, None

    A, _ = lax.scan(outer_loop_body, A, jnp.arange(n))
    return jnp.sum(A)


def run_lu(device_type: dace.dtypes.DeviceType):
    """
    Runs LU for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    N = 40
    A = init_data(N)
    gt_A = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = lu_kernel.to_sdfg()
        auto_optimize(sdfg, device=device_type)
        dace_res = sdfg(A=A, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = lu_kernel.to_sdfg(simplify=True)

        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        platform = dace.config.Config.get("compiler", "fpga", "vendor")
        if platform == "intel_fpga":
            Dot.default_implementation = "FPGA_Accumulate"
        else:
            Dot.default_implementation = "FPGA_PartialSums"

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)
        fpga_auto_opt.fpga_global_to_local(sdfg)

        sdfg.specialize(dict(N=N))
        dace_res = sdfg(A=A)

    # Compute ground truth and validate result
    ground_truth(N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    assert diff < 1e-5
    return sdfg


def run_lu_autodiff():
    # Initialize data (polybench mini size)
    N = 10
    A = init_data(N)
    A_jax = jnp.copy(A)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float32)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float32[N, N]):
        lu_kernel(A)
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(A, N=N, gradient_A=gradient_A, gradient___return=gradient___return)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(lu_jax_kernel))
    jax_grad_A = jax_grad(A_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A, rtol=1e-5, atol=1e-5)


def test_cpu():
    run_lu(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_lu(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    run_lu_autodiff()


@fpga_test(assert_ii_1=False, xilinx=False)
def test_fpga():
    return run_lu(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_lu(dace.dtypes.DeviceType.CPU)
        run_lu_autodiff()
    elif target == "gpu":
        run_lu(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_lu(dace.dtypes.DeviceType.FPGA)
