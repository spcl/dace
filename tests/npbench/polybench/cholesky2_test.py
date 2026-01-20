# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.auto.auto_optimize import auto_optimize

# Data set sizes
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

N = dc.symbol('N', dtype=dc.int64)
k = dc.symbol('k', dtype=dc.int64)


@dc.program
def triu(A: dc.float64[N, N]):
    B = np.zeros_like(A)
    for i in dc.map[0:N]:
        for j in dc.map[i + k:N]:
            B[i, j] = A[i, j]
    return B


@dc.program
def cholesky2_kernel(A: dc.float64[N, N]):
    A[:] = np.linalg.cholesky(A) + triu(A, k=1)


def init_data(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, :i + 1] = np.fromfunction(lambda j: (-j % N) / N + 1, (i + 1, ), dtype=datatype)
        A[i, i + 1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)

    return A


def ground_truth(N, A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


def run_cholesky2(device_type: dace.dtypes.DeviceType):
    '''
    Runs Cholesky2 for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    A = init_data(N)
    gt_A = np.copy(A)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = cholesky2_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A=A, N=N)
    # Compute ground truth and validate result
    ground_truth(N, gt_A)
    diff = np.linalg.norm(gt_A - A) / np.linalg.norm(gt_A)
    assert diff < 1e-6
    return sdfg


def test_cpu():
    run_cholesky2(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_cholesky2(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_cholesky2(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_cholesky2(dace.dtypes.DeviceType.GPU)
