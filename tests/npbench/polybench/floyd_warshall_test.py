# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.transformation.dataflow import StreamingMemory, MapFusionVertical, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize

# Data set sizes
# N
sizes = {"mini": 60, "small": 180, "medium": 500, "large": 2800, "extra-large": 5600}

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def kernel(path: dc.int32[N, N]):

    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))


def init_data(N):

    path = np.empty((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            path[i, j] = i * j % 7 + 1
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return path


def ground_truth(path, N):

    for k in range(N):
        for i in range(N):
            tmp = path[i, k] + path[k, :]
            cond = path[i, :] >= tmp
            path[i, cond] = tmp[cond]


def run_floyd_warshall(device_type: dace.dtypes.DeviceType):
    """
    Runs Floyd Warshall for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    N = sizes["mini"]
    path = init_data(N)
    gt_path = np.copy(path)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(path=path, N=N)

    # Compute ground truth and validate result
    ground_truth(gt_path, N)
    assert np.allclose(path, gt_path)
    return sdfg


def test_cpu():
    run_floyd_warshall(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_floyd_warshall(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_floyd_warshall(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_floyd_warshall(dace.dtypes.DeviceType.GPU)
