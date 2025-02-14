# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import numpy as np


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

    a = np.random.rand(128, 64)
    expected = np.copy(a)
    expected[:, :32] = 5
    expected[:, 33:60] += 6

    if block_size is not None:
        with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value=block_size):
            sdfg(a)
    else:
        sdfg(a)

    assert np.allclose(a, expected)


if __name__ == '__main__':
    test_thread_specialization_noncontiguous_blocks(None)
    test_thread_specialization_noncontiguous_blocks('64,8,1')
