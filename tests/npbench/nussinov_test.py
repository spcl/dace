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
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_aopt

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
                    table[i, j] = np.maximum(
                        table[i, j],
                        table[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    table[i, j] = np.maximum(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = np.maximum(table[i, j],
                                         table[i, k] + table[k + 1, j])

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
                    table[i,
                          j] = max(table[i, j],
                                   table[i + 1, j - 1] + match_gt(seq[i], seq[j]))
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
    '''
    Runs Nussinov for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = 60
    seq, table = init_data(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = kernel.to_sdfg()
        # sdfg.apply_strict_transformations()

        # Applying this will let the validation fail
        # The problem seems to be in Loop2Map. With this version of it fc7ff0b8
        # then autoopt works again
        sdfg = auto_optimize(sdfg, device_type)
        dace_res = sdfg(seq=seq, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(strict=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sm_applied = sdfg.apply_transformations_repeated(
            [InlineSDFG, StreamingMemory],
            [{}, {
                'storage': dace.StorageType.FPGA_Local
            }],
            print_report=True)
        # assert sm_applied == 1
        sc_applied = sdfg.apply_transformations_repeated(
            [InlineSDFG, StreamingComposition],
            [{}, {
                'storage': dace.StorageType.FPGA_Local
            }],
            print_report=True)
        # assert sc_applied == 1

        # Prune connectors after Streaming Comp
        pruned_conns = sdfg.apply_transformations_repeated(
            PruneConnectors, options=[{
                'remove_unused_containers': True
            }])

        # assert pruned_conns == 1

        # fpga_aopt.fpga_global_to_local(sdfg)
        fpga_aopt.fpga_rr_interleave_containers_to_banks(sdfg)

        # # In this case, we want to generate the top-level state as an host-based state,
        # # not an FPGA kernel. We need to explicitly indicate that
        # sdfg.states()[0].location["is_FPGA_kernel"] = False
        # we need to specialize both the top-level SDFG and the nested SDFG
        sdfg.specialize(dict(N=N))
        # sdfg.states()[0].nodes()[0].sdfg.specialize(dict(N=N))
        # run program
        dace_res = sdfg(seq=seq)

    # Compute ground truth and validate result
    gt_res = ground_truth(N, seq)
    print(dace_res)
    print(gt_res)
    assert np.allclose(dace_res, gt_res)
    return sdfg


def test_cpu():
    run_nussinov(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_nussinov(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_nussinov(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--target",
                        default='cpu',
                        choices=['cpu', 'gpu', 'fpga'],
                        help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_nussinov(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_nussinov(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_nussinov(dace.dtypes.DeviceType.FPGA)
