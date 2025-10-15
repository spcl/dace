# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

# Data set sizes
# NQ, NR, NP
sizes = {
    "mini": (8, 10, 12),
    "small": (20, 25, 30),
    "medium": (40, 50, 60),
    "large": (140, 150, 160),
    "extra-large": (220, 250, 270)
}

NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NQ', 'NP'))


@dc.program
def doitgen_kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):

    # Ideal - not working because Matmul with dim > 3 unsupported
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, NP)) @ C4, (NQ, NP))


def initialize(NR, NQ, NP, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=datatype)

    return A, C4


def doitgen_jax_kernel(A, C4):
    NR = A.shape[0]
    NQ = A.shape[1]
    NP = A.shape[2]
    for r in range(NR):
        A = A.at[r, :, :].set(jnp.reshape(jnp.reshape(A[r], (NQ, NP)) @ C4, (NQ, NP)))
    return jnp.sum(A)


def ground_truth(NR, NQ, NP, A, C4):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


def run_doitgen(device_type: dace.dtypes.DeviceType):
    '''
    Runs Doitgen for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    NQ, NR, NP = sizes["mini"]
    A, C4 = initialize(NR, NQ, NP)
    A_ref = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = doitgen_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A, C4, NR=NR, NQ=NQ, NP=NP)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = doitgen_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.states()[0].location["is_FPGA_kernel"] = False
        # we need to specialize both the top-level SDFG and the nested SDFG
        sdfg.specialize(dict(NR=NR, NQ=NQ, NP=NP))
        sdfg.states()[0].nodes()[0].sdfg.specialize(dict(NR=NR, NQ=NQ, NP=NP))
        # TODO: add support for `long long` in Intel FPGA, set systolic array size
        sdfg(A, C4)

    # Compute ground truth and Validate result
    ground_truth(NR, NQ, NP, A_ref, C4)
    assert np.allclose(A, A_ref)
    return sdfg


def run_doitgen_autodiff():
    # Initialize data (polybench mini size)
    NQ, NR, NP = sizes["mini"]
    A, C4 = initialize(NR, NQ, NP)

    # Initialize gradient computation data
    gradient_A = np.zeros_like(A)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):
        doitgen_kernel(A, C4)
        return np.sum(A)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["__return"])
    sdfg(A, C4, NR=NR, NQ=NQ, NP=NP, gradient_A=gradient_A, gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_grad = jax.jit(jax.grad(doitgen_jax_kernel, argnums=0))
    A_jax, C4_jax = initialize(NR, NQ, NP)
    jax_grad_A = jax_grad(A_jax, C4_jax)
    np.testing.assert_allclose(gradient_A, jax_grad_A)


def test_cpu():
    run_doitgen(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_doitgen(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    run_doitgen_autodiff()


@pytest.mark.skip(reason="long long support for IntelFPGA")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_doitgen(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_doitgen(dace.dtypes.DeviceType.CPU)
        run_doitgen_autodiff()
    elif target == "gpu":
        run_doitgen(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_doitgen(dace.dtypes.DeviceType.FPGA)
