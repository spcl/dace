# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
import numpy as np
import sympy as sp


def _construct_graph(tbsize_1=None, tbsize_2=None) -> dace.SDFG:
    """
    Construct a graph for the tests.
    :param tbsize_1: The dimensions of the thread-block map in the first
                        map, or None for no thread-block map.
    :param tbsize_2: The dimensions of the thread-block map in the second
                        map, or None for no thread-block map.
    :return: SDFG for test.
    """
    wsize = 32
    size = 128 // 32
    dims = (wsize, size)
    ind = 'i, j'
    int_ceil = dace.symbolic.int_ceil
    sdfg = dace.SDFG('kernel_fusion_test')
    sdfg.add_array('hA', dims, dace.float64)
    sdfg.add_transient('A', dims, dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp',
                       dims,
                       dace.float64,
                       storage=dace.StorageType.GPU_Global,
                       lifetime=dace.AllocationLifetime.SDFG)
    sdfg.add_transient('B', dims, dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('hB', dims, dace.float64)
    state = sdfg.add_state()

    # For compatibility, schedule fused kernel with the smallest concurrent
    # number of blocks
    fme, fmx = state.add_map('fused_kernel', dict(i=f'0:{wsize}'), dace.ScheduleType.GPU_Device)
    ime1, imx1 = state.add_map('kernel_a', dict(j='0:%s' % (size if tbsize_1 is None else int_ceil(size, tbsize_1))),
                               dace.ScheduleType.GPU_Device)
    ime2, imx2 = state.add_map('kernel_b', dict(j='0:%s' % (size if tbsize_2 is None else int_ceil(size, tbsize_2))),
                               dace.ScheduleType.GPU_Device)

    tasklet1 = state.add_tasklet('code_a', {'a'}, {'t'}, 't = a * 5')
    tasklet2 = state.add_tasklet('code_b', {'t'}, {'b'}, 'b = t + 1')

    a = state.add_access('A')
    tmp = state.add_access('tmp')
    b = state.add_access('B')

    # Add copy-in/copy-out edges
    ha = state.add_read('hA')
    hb = state.add_write('hB')
    state.add_nedge(ha, a, dace.Memlet('A'))
    state.add_nedge(b, hb, dace.Memlet('B'))

    # Add thread-block maps and edges as necessary
    if tbsize_1 is not None:
        tbme1, tbmx1 = state.add_map('block_a', dict(k='0:%s' % tbsize_1), dace.ScheduleType.GPU_ThreadBlock)
        state.add_memlet_path(a,
                              fme,
                              ime1,
                              tbme1,
                              tasklet1,
                              dst_conn='a',
                              memlet=dace.Memlet.simple('A', f'i, {size} - 1 - (j*{tbsize_1} + k)'))
        state.add_memlet_path(tasklet1,
                              tbmx1,
                              imx1,
                              tmp,
                              src_conn='t',
                              memlet=dace.Memlet.simple('tmp', f'i, j*{tbsize_1} + k'))
    else:
        state.add_memlet_path(a,
                              fme,
                              ime1,
                              tasklet1,
                              dst_conn='a',
                              memlet=dace.Memlet.simple('A', f'i, {size} - 1 - j'))
        state.add_memlet_path(tasklet1, imx1, tmp, src_conn='t', memlet=dace.Memlet.simple('tmp', ind))

    if tbsize_2 is not None:
        tbme2, tbmx2 = state.add_map('block_a', dict(k='0:%s' % tbsize_2), dace.ScheduleType.GPU_ThreadBlock)
        state.add_memlet_path(tmp,
                              ime2,
                              tbme2,
                              tasklet2,
                              dst_conn='t',
                              memlet=dace.Memlet.simple('tmp', f'i, j*{tbsize_2} + k'))
        state.add_memlet_path(tasklet2,
                              tbmx2,
                              imx2,
                              fmx,
                              b,
                              src_conn='b',
                              memlet=dace.Memlet.simple('B', f'i, j*{tbsize_2} + k'))
    else:
        state.add_memlet_path(tmp, ime2, tasklet2, dst_conn='t', memlet=dace.Memlet.simple('tmp', ind))
        state.add_memlet_path(tasklet2, imx2, fmx, b, src_conn='b', memlet=dace.Memlet.simple('B', ind))
    return sdfg


def _check_results(sdfg: dace.SDFG):
    A = np.random.rand(32, 4)
    B = np.random.rand(32, 4)
    sdfg(hA=A, hB=B)
    assert np.allclose(B, A[:, ::-1] * 5 + 1)


@pytest.mark.gpu
def test_fused_notb():
    sdfg = _construct_graph(None, None)
    _check_results(sdfg)


@pytest.mark.gpu
def test_fused_tb():
    sdfg = _construct_graph(2, 2)
    _check_results(sdfg)


@pytest.mark.gpu
def test_fused_mixedtb():
    sdfg = _construct_graph(2, None)
    _check_results(sdfg)


if __name__ == '__main__':
    test_fused_notb()
    test_fused_tb()
    test_fused_mixedtb()
