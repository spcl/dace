# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU_Device map whose range bounds arrive through dynamic-range connectors named after their
scalar containers, as the frontend emits for ``for jl in range(kidia, kfdia + 1)``."""
import re

import numpy as np
import pytest

import dace
from dace import dtypes

N = dace.symbol('N', dace.int64)


def dynamic_range_kernel_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('dynamic_range_kernel_dynamic_range_kernel_sdfg')
    sdfg.add_scalar('kidia', dace.int32)
    sdfg.add_scalar('kfdia', dace.int32)
    sdfg.add_array('a', (N, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('b', (N, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('kernel', {'jl': 'kidia:kfdia + 1'}, schedule=dtypes.ScheduleType.GPU_Device)
    for bound in ('kidia', 'kfdia'):
        entry.add_in_connector(bound)
        state.add_edge(state.add_read(bound), None, entry, bound, dace.Memlet(f'{bound}[0]'))
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('a[jl]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='out', memlet=dace.Memlet('b[jl]'))
    return sdfg


@pytest.mark.gpu
def test_dynamic_range_bounds_reach_the_kernel():
    import cupy as cp
    sdfg = dynamic_range_kernel_sdfg()
    host_code = '\n'.join(obj.clean_code for obj in sdfg.generate_code() if obj.language == 'cpp')
    self_initialized = re.search(r'^\s*int (kidia|kfdia) = \1;', host_code, re.MULTILINE)
    assert self_initialized is None, 'a bound is shadowed by a self-initialized copy'

    n = 32
    a = cp.arange(n, dtype=np.float64)
    b = cp.zeros(n, dtype=np.float64)
    sdfg(a=a, b=b, kidia=np.int32(4), kfdia=np.int32(20), N=n)
    expected = np.zeros(n)
    expected[4:21] = 2.0 * np.arange(4, 21)
    assert np.array_equal(b.get(), expected)


if __name__ == '__main__':
    test_dynamic_range_bounds_reach_the_kernel()
