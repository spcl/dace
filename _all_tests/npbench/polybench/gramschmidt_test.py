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

# Data set sizes
# M, N
# Note: these have been swapped to improve numerical stability
sizes = {"mini": (30, 20), "small": (80, 60), "medium": (240, 200), "large": (1200, 1000), "extra-large": (2600, 2000)}

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))


@dc.program
def gramschmidt_kernel(A: dc.float64[M, N]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


def initialize(M, N, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return A


def ground_truth(A):

    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


def run_gramschmidt(device_type: dace.dtypes.DeviceType):
    '''
    Runs Gesummv for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    M, N = sizes["mini"]
    A = initialize(M, N)
    A_ref = np.copy(A)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = gramschmidt_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        Q, R = sdfg(A, M=M, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = gramschmidt_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(M=M, N=N))
        Q, R = sdfg(A)

    # Compute ground truth and validate
    Q_ref, R_ref = ground_truth(A_ref)
    assert np.allclose(Q, Q_ref)
    assert np.allclose(R, R_ref)
    return sdfg


def test_cpu():
    run_gramschmidt(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_gramschmidt(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_gramschmidt(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_gramschmidt(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_gramschmidt(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_gramschmidt(dace.dtypes.DeviceType.FPGA)
