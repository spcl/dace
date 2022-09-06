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
    '''
    Runs Floyd Warshall for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench mini size)
    N = 60
    path = init_data(N)
    gt_path = np.copy(path)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(path=path, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = kernel.to_sdfg(simplify=True)
        # sdfg.apply_transformations_repeated([MapFusion])
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory],
                                                         [{}, {
                                                             'storage': dace.StorageType.FPGA_Local
                                                         }],
                                                         print_report=True)
        sc_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingComposition],
                                                         [{}, {
                                                             'storage': dace.StorageType.FPGA_Local
                                                         }],
                                                         print_report=True,
                                                         permissive=True)
        assert sc_applied == 1

        # Prune connectors after Streaming Composition
        pruned_conns = sdfg.apply_transformations_repeated(PruneConnectors,
                                                           options=[{
                                                               'remove_unused_containers': True
                                                           }])

        assert pruned_conns == 1

        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.states()[0].location["is_FPGA_kernel"] = False
        # we need to specialize both the top-level SDFG and the nested SDFG
        sdfg.specialize(dict(N=N))
        sdfg.states()[0].nodes()[0].sdfg.specialize(dict(N=N))
        # run program
        sdfg(path=path)

    # Compute ground truth and validate result
    ground_truth(gt_path, N)
    assert np.allclose(path, gt_path)
    return sdfg


def test_cpu():
    run_floyd_warshall(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_floyd_warshall(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_floyd_warshall(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_floyd_warshall(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_floyd_warshall(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_floyd_warshall(dace.dtypes.DeviceType.FPGA)
