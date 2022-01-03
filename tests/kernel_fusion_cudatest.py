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
    size = 2048
    int_ceil = sp.Function('int_ceil')
    sdfg = dace.SDFG('kernel_fusion_test')
    sdfg.add_array('hA', [size], dace.float64)
    sdfg.add_transient('A', [size], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp', [size],
                       dace.float64,
                       storage=dace.StorageType.GPU_Global,
                       lifetime=dace.AllocationLifetime.SDFG)
    sdfg.add_transient('B', [size], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('hB', [size], dace.float64)
    state = sdfg.add_state()

    # For compatibility, schedule fused kernel with the smallest concurrent
    # number of blocks
    fme, fmx = state.add_map('fused_kernel', dict(i='0:2'), dace.ScheduleType.GPU_Device)
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
    state.add_nedge(ha, a, dace.Memlet.simple('A', '0:%s' % size))
    state.add_nedge(b, hb, dace.Memlet.simple('B', '0:%s' % size))

    # Add thread-block maps and edges as necessary
    if tbsize_1 is not None:
        tbme1, tbmx1 = state.add_map('block_a', dict(k='0:%s' % tbsize_1), dace.ScheduleType.GPU_ThreadBlock)
        state.add_memlet_path(a,
                              fme,
                              ime1,
                              tbme1,
                              tasklet1,
                              dst_conn='a',
                              memlet=dace.Memlet.simple('A', '%s - 1 - (j*%s + k)' % (size, tbsize_1)))
        state.add_memlet_path(tasklet1,
                              tbmx1,
                              imx1,
                              tmp,
                              src_conn='t',
                              memlet=dace.Memlet.simple('tmp', 'j*%s + k' % tbsize_1))
    else:
        state.add_memlet_path(a, fme, ime1, tasklet1, dst_conn='a', memlet=dace.Memlet.simple('A', '%s - 1 - j' % size))
        state.add_memlet_path(tasklet1, imx1, tmp, src_conn='t', memlet=dace.Memlet.simple('tmp', 'j'))

    if tbsize_2 is not None:
        tbme2, tbmx2 = state.add_map('block_a', dict(k='0:%s' % tbsize_2), dace.ScheduleType.GPU_ThreadBlock)
        state.add_memlet_path(tmp,
                              ime2,
                              tbme2,
                              tasklet2,
                              dst_conn='t',
                              memlet=dace.Memlet.simple('tmp', '(j*%s + k)' % tbsize_2))
        state.add_memlet_path(tasklet2,
                              tbmx2,
                              imx2,
                              fmx,
                              b,
                              src_conn='b',
                              memlet=dace.Memlet.simple('B', 'j*%s + k' % tbsize_2))
    else:
        state.add_memlet_path(tmp, ime2, tasklet2, dst_conn='t', memlet=dace.Memlet.simple('tmp', 'j'))
        state.add_memlet_path(tasklet2, imx2, fmx, b, src_conn='b', memlet=dace.Memlet.simple('B', 'j'))
    return sdfg


def _check_results(sdfg: dace.SDFG):
    A = np.random.rand(2048)
    B = np.random.rand(2048)
    sdfg(hA=A, hB=B)
    assert np.allclose(B, A[::-1] * 5 + 1)


@pytest.mark.gpu
def test_fused_notb():
    sdfg = _construct_graph(None, None)
    _check_results(sdfg)


@pytest.mark.gpu
def test_fused_tb():
    sdfg = _construct_graph(64, 64)
    _check_results(sdfg)


@pytest.mark.gpu
def test_fused_mixedtb():
    sdfg = _construct_graph(256, None)
    _check_results(sdfg)


if __name__ == '__main__':
    test_fused_notb()
    test_fused_tb()
    test_fused_mixedtb()
