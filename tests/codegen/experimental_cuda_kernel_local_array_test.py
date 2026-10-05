# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel-local array the experimental codegen lifts out of the kernel is handed on through the kernel exit as
the same container; that hand-over moves no data."""
import numpy as np
import pytest

import dace
from dace import dtypes

N = dace.symbol('N')
M = dace.symbol('M')


def _kernel_local_array_sdfg() -> dace.SDFG:
    """B[i, :] = A[i, :]^2 + 1 per kernel thread, staged through a per-thread array of M elements."""
    sdfg = dace.SDFG('kernel_local_array')
    sdfg.add_array('A', (N, M), dace.float64)
    sdfg.add_array('B', (N, M), dace.float64)
    sdfg.add_transient('gpu_A', (N, M), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient('gpu_B', (N, M), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient('tmp', (M, ), dace.float64, storage=dtypes.StorageType.Register)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('kernel', dict(i='0:N'), schedule=dtypes.ScheduleType.GPU_Device)
    fill_entry, fill_exit = state.add_map('fill', dict(j='0:M'), schedule=dtypes.ScheduleType.Sequential)
    drain_entry, drain_exit = state.add_map('drain', dict(j='0:M'), schedule=dtypes.ScheduleType.Sequential)
    square = state.add_tasklet('square', {'a'}, {'b'}, 'b = a * a')
    increment = state.add_tasklet('increment', {'a'}, {'b'}, 'b = a + 1')
    tmp = state.add_access('tmp')
    gpu_a = state.add_access('gpu_A')
    gpu_b = state.add_access('gpu_B')
    state.add_nedge(state.add_read('A'), gpu_a, dace.Memlet('A[0:N, 0:M]'))
    state.add_nedge(gpu_b, state.add_write('B'), dace.Memlet('B[0:N, 0:M]'))
    state.add_memlet_path(gpu_a, entry, fill_entry, square, dst_conn='a', memlet=dace.Memlet('gpu_A[i, j]'))
    state.add_memlet_path(square, fill_exit, tmp, src_conn='b', memlet=dace.Memlet('tmp[j]'))
    state.add_memlet_path(tmp, drain_entry, increment, dst_conn='a', memlet=dace.Memlet('tmp[j]'))
    state.add_memlet_path(increment, drain_exit, exit_node, gpu_b, src_conn='b', memlet=dace.Memlet('gpu_B[i, j]'))
    sdfg.validate()
    return sdfg


def test_a_kernel_local_array_is_not_copied_onto_itself():
    sdfg = _kernel_local_array_sdfg()
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        objects = sdfg.generate_code()
    kernels = [code.clean_code for code in objects if '__global__' in code.clean_code]
    assert len(kernels) == 1, 'expected exactly one kernel'

    assert 'Copy' not in kernels[0], 'the lifted array is copied onto itself inside the kernel'


@pytest.mark.gpu
def test_a_kernel_local_array_keeps_each_threads_values():
    sdfg = _kernel_local_array_sdfg()
    rng = np.random.default_rng(0)
    a = rng.random((96, 40))
    b = np.zeros_like(a)

    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg(A=a, B=b, N=96, M=40)

    np.testing.assert_allclose(b, a * a + 1)


if __name__ == '__main__':
    test_a_kernel_local_array_is_not_copied_onto_itself()
    test_a_kernel_local_array_keeps_each_threads_values()
