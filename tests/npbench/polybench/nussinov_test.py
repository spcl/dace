# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.dataflow import StreamingMemory, MapFusionVertical, StreamingComposition, PruneConnectors
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.config import set_temporary

N = dc.symbol('N', dtype=dc.int32)


@dc.program
def match(b1: dc.int32, b2: dc.int32):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


@dc.program
def kernel(seq: dc.int32[N]):

    table = np.zeros((N, N), np.int32)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = np.maximum(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = np.maximum(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = np.maximum(table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    table[i, j] = np.maximum(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = np.maximum(table[i, j], table[i, k] + table[k + 1, j])

    return table


def match_gt(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


def ground_truth(N, seq):
    table = np.zeros((N, N), np.int32)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] + match_gt(seq[i], seq[j]))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])
    return table


def init_data(N):

    seq = np.empty((N, ), dtype=np.int32)
    table = np.empty((N, N), dtype=np.int32)
    for i in range(N):
        seq[i] = (i + 1) % 4
    for i in range(N):
        for j in range(N):
            table[i, j] = 0

    return seq, table


def run_nussinov(device_type: dace.dtypes.DeviceType):
    """
    Runs Nussinov for the given device

    :return: the SDFG
    """

    # Initialize data (polybench mini size)
    N = 60
    seq, table = init_data(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        dace_res = sdfg(seq=seq, N=N)

    # Compute ground truth and validate result
    gt_res = ground_truth(N, seq)

    assert np.allclose(dace_res, gt_res)
    return sdfg


def test_cpu():
    run_nussinov(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_nussinov(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_nussinov(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_nussinov(dace.dtypes.DeviceType.GPU)
