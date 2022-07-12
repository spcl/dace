# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusion, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.libraries.standard import Reduce
from dace.libraries.blas import Gemv

M, N = (dc.symbol(s, dtype=dc.int32) for s in ('M', 'N'))


@dc.program
def covariance_kernel(float_n: dc.float32, data: dc.float32[N, M]):

    mean = np.mean(data, axis=0)
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    # for i in range(M):
    #     cov[i, i:M] = data[:, i] @ data[:, i:M]

    return cov


def ground_truth(M, N, float_n, data):

    mean = np.empty((M, ), dtype=data.dtype)
    for j in range(M):
        mean[j] = 0.0
        for i in range(N):
            mean[j] += data[i, j]
        mean[j] /= float_n

    for i in range(N):
        for j in range(M):
            data[i, j] -= mean[j]

    cov = np.empty((M, M), dtype=data.dtype)
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = 0.0
            for k in range(N):
                cov[i, j] += data[k, i] * data[k, j]
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

    return cov


def init_data(M, N):

    float_n = np.float32(N)
    data = np.empty((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            data[i, j] = (i * j) / M

    return float_n, data


def run_covariance(device_type: dace.dtypes.DeviceType):
    """
    Runs Covariance for the given device

    :return: the SDFG
    """

    # Initialize data (polybench small size)
    M, N = (80, 100)
    float_n, data = init_data(M, N)

    gt_data = np.copy(data)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = covariance_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        dace_res = sdfg(float_n=float_n, data=data, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = covariance_kernel.to_sdfg(simplify=False)
        sdfg.simplify()
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations([InlineSDFG])

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        # Reduce.default_implementation = "FPGAPartialReduction"
        Gemv.default_implementation = "FPGA_Accumulate"

        sdfg.expand_library_nodes()
        sdfg.apply_transformations([InlineSDFG])

        # Other FPGA auto opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # Specialize the SDFG
        sdfg.specialize(dict(N=N, M=M))

        # run program
        dace_res = sdfg(float_n=float_n, data=data)

    # Compute ground truth and validate result
    gt_res = ground_truth(M, N, float_n, gt_data)
    assert np.allclose(gt_res, dace_res)
    return sdfg


def test_cpu():
    run_covariance(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_covariance(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_covariance(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_covariance(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_covariance(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_covariance(dace.dtypes.DeviceType.FPGA)
