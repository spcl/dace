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
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def flip(A: dc.float64[M]):
    B = np.ndarray((M, ), dtype=np.float64)
    for i in dc.map[0:M]:
        B[i] = A[M - 1 - i]
    return B


@dc.program
def durbin_kernel(r: dc.float64[N]):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * flip(y[:k])
        y[k] = alpha

    return y


def initialize(N, datatype=np.float64):
    r = np.fromfunction(lambda i: N + 1 - i, (N, ), dtype=datatype)
    return r


def ground_truth(r):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y


def run_durbin(device_type: dace.dtypes.DeviceType):
    '''
    Runs Durbin for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["small"]
    r = initialize(N)
    y_ref = ground_truth(r)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = durbin_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        y = sdfg(r, N=N)
        assert np.allclose(y, y_ref)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = durbin_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        y = sdfg(r)
        assert np.allclose(y, y_ref, atol=1e-6)

    return sdfg


@pytest.mark.skip(reason="Validation error")
def test_cpu():
    run_durbin(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_durbin(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_durbin(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_durbin(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_durbin(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_durbin(dace.dtypes.DeviceType.FPGA)
