# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import numpy as np

from dace.transformation.passes.nested_gpu_device_map_lowering import NestedGPUDeviceMapLowering


@pytest.mark.gpu
@pytest.mark.parametrize('block_size', [None, '64,8,1'])
def test_double_nest_thread_specialization_noncontiguous_blocks(block_size):

    @dace.program
    def double_nest_thread_specialization(A: dace.float64[128, 64, 32], c: dace.float64):
        for i in dace.map[0:128]:
            for j in dace.map[0:32]:
                with dace.tasklet:
                    out >> A[i, j, 0]
                    out = 5.0
                if c > 2.0:
                    for k in dace.map[0:16]:
                        with dace.tasklet:
                            out2 >> A[i, j, k]
                            out2 = 5.0

            for j in dace.map[33:60]:
                with dace.tasklet:
                    inp << A[i, j, 0]
                    out >> A[i, j, 0]
                    out = 6.0 + inp
                if c > 2.0:
                    for k in dace.map[16:32]:
                        with dace.tasklet:
                            inp2 << A[i, j, k]
                            out2 >> A[i, j, k]
                            out2 = 6.0 + inp2

    sdfg = double_nest_thread_specialization.to_sdfg()
    sdfg.apply_gpu_transformations()

    # Ensure all nested maps set grid dimensions
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            n.schedule = dace.ScheduleType.GPU_Device

    a = np.random.rand(128, 64, 32)
    expected = np.copy(a)
    expected[:, :32, :16] = 5.0
    expected[:, 33:60, 16:32] += 6.0

    sdfg.validate()
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    sdfg.validate()

    num_device_maps = len({
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
    })
    assert num_device_maps == 1

    if block_size is not None:
        with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value=block_size):
            sdfg(a)
    else:
        sdfg(a)

    assert np.allclose(a, expected)


@pytest.mark.gpu
@pytest.mark.parametrize('block_size', [None, '64,8,1'])
def test_thread_specialization_noncontiguous_blocks(block_size):

    @dace.program
    def thread_specialization(A: dace.float64[128, 64]):
        for i in dace.map[0:128]:
            for j in dace.map[0:32]:
                with dace.tasklet:
                    out >> A[i, j]
                    out = 5

            for j in dace.map[33:60]:
                with dace.tasklet:
                    inp << A[i, j]
                    out >> A[i, j]
                    out = 6 + inp

    sdfg = thread_specialization.to_sdfg()
    sdfg.apply_gpu_transformations()

    # Ensure all nested maps set grid dimensions
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            n.schedule = dace.ScheduleType.GPU_Device
    sdfg.save("x1.sdfg")

    a = np.random.rand(128, 64)
    expected = np.copy(a)
    expected[:, :32] = 5
    expected[:, 33:60] += 6

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    num_device_maps = len({
        n
        for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
    })
    assert num_device_maps == 1
    sdfg.save("x2.sdfg")

    if block_size is not None:
        with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value=block_size):
            sdfg(a)
    else:
        sdfg(a)

    assert np.allclose(a, expected)


if __name__ == '__main__':
    test_thread_specialization_noncontiguous_blocks(None)
    test_thread_specialization_noncontiguous_blocks('64,8,1')
    test_double_nest_thread_specialization_noncontiguous_blocks(None)
    test_double_nest_thread_specialization_noncontiguous_blocks('64,8,1')
