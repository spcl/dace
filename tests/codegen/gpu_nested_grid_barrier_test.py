# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel whose body is a nested SDFG with consecutive device maps has to declare its grid barrier.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""
import re

import pytest

import dace

# The grid barrier is emitted by the legacy CUDA code generator only.
pytestmark = pytest.mark.old_gpu_codegen_only

N = 32
W = 4


def kernel_with_consecutive_maps(nested: bool) -> dace.SDFG:
    sdfg = dace.SDFG('grid_barrier_nested' if nested else 'grid_barrier_flat')
    for name in ('A', 'T', 'B'):
        sdfg.add_array(name, [W, N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    kernel_entry, kernel_exit = state.add_map('kernel', dict(i=f'0:{W}'), dace.ScheduleType.GPU_Device)

    if nested:
        inner = dace.SDFG('inner')
        inner.add_symbol('i', dace.int64)
        for name in ('A', 'T', 'B'):
            inner.add_array(name, [W, N], dace.float64, storage=dace.StorageType.GPU_Global)
        body = inner.add_state()
    else:
        body = state

    first_entry, first_exit = body.add_map('first', dict(j=f'0:{N}'), dace.ScheduleType.GPU_Device)
    second_entry, second_exit = body.add_map('second', dict(j=f'0:{N}'), dace.ScheduleType.GPU_Device)
    first = body.add_tasklet('first', {'a'}, {'t'}, 't = a * 5')
    second = body.add_tasklet('second', {'t'}, {'b'}, 'b = t + 1')
    tmp = body.add_access('T')
    if nested:
        body.add_memlet_path(body.add_read('A'), first_entry, first, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
        body.add_memlet_path(first, first_exit, tmp, src_conn='t', memlet=dace.Memlet('T[i, j]'))
        body.add_memlet_path(tmp, second_entry, second, dst_conn='t', memlet=dace.Memlet('T[i, j]'))
        body.add_memlet_path(second, second_exit, body.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i, j]'))
        node = state.add_nested_sdfg(inner, {'A'}, {'T', 'B'}, {'i': 'i'})
        state.add_memlet_path(state.add_read('A'),
                              kernel_entry,
                              node,
                              dst_conn='A',
                              memlet=dace.Memlet(f'A[0:{W}, 0:{N}]'))
        state.add_memlet_path(node,
                              kernel_exit,
                              state.add_write('T'),
                              src_conn='T',
                              memlet=dace.Memlet(f'T[0:{W}, 0:{N}]'))
        state.add_memlet_path(node,
                              kernel_exit,
                              state.add_write('B'),
                              src_conn='B',
                              memlet=dace.Memlet(f'B[0:{W}, 0:{N}]'))
    else:
        state.add_memlet_path(state.add_read('A'),
                              kernel_entry,
                              first_entry,
                              first,
                              dst_conn='a',
                              memlet=dace.Memlet('A[i, j]'))
        state.add_memlet_path(first, first_exit, tmp, src_conn='t', memlet=dace.Memlet('T[i, j]'))
        state.add_memlet_path(tmp, second_entry, second, dst_conn='t', memlet=dace.Memlet('T[i, j]'))
        state.add_memlet_path(second,
                              second_exit,
                              kernel_exit,
                              state.add_write('B'),
                              src_conn='b',
                              memlet=dace.Memlet('B[i, j]'))
    sdfg.validate()
    return sdfg


def check_barrier(nested: bool):
    sdfg = kernel_with_consecutive_maps(nested)
    code = next(obj.clean_code for obj in sdfg.generate_code() if obj.language == 'cu')
    assert '__gbar.Sync();' in code
    assert re.search(r'dace::GridBarrier\s+__gbar|GridBarrier\s*&\s*__gbar', code)


def test_grid_barrier_in_kernel():
    check_barrier(nested=False)


def test_grid_barrier_in_nested_sdfg():
    check_barrier(nested=True)


if __name__ == '__main__':
    test_grid_barrier_in_kernel()
    test_grid_barrier_in_nested_sdfg()
