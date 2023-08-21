# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_read_write_set():
    sdfg = dace.SDFG('graph')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_array('C', [10], dace.float64)
    state = sdfg.add_state('state')
    task1 = state.add_tasklet('work1', {'A'}, {'B'}, 'B = A + 1')
    task2 = state.add_tasklet('work2', {'B'},  {'C'}, 'C = B + 1')
    read_a = state.add_access('A')
    rw_b = state.add_access('B')
    write_c = state.add_access('C')
    state.add_memlet_path(read_a, task1, dst_conn='A', memlet=dace.Memlet('A[2]'))
    state.add_memlet_path(task1, rw_b, src_conn='B', memlet=dace.Memlet('B[2]'))
    state.add_memlet_path(rw_b, task2, dst_conn='B', memlet=dace.Memlet('B[2]'))
    state.add_memlet_path(task2, write_c, src_conn='C', memlet=dace.Memlet('C[2]'))

    assert 'B' not in state.read_and_write_sets()[0]


def test_read_write_set_y_formation():
    sdfg = dace.SDFG('graph')
    state = sdfg.add_state('state')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_array('C', [2], dace.float64)
    task1 = state.add_tasklet('work1', {'A'}, {'B'}, 'B = A + 1')
    task2 = state.add_tasklet('work2', {'B'},  {'C'}, 'C += B + 1')
    task3 = state.add_tasklet('work3', {'A'},  {'B'}, 'B = A + 2')
    read_a = state.add_access('A')
    rw_b = state.add_access('B')
    write_c = state.add_access('C')
    state.add_memlet_path(read_a, task1, dst_conn='A', memlet=dace.Memlet(data='A', subset='0'))
    state.add_memlet_path(read_a, task3, dst_conn='A', memlet=dace.Memlet(data='A', subset='1'))
    state.add_memlet_path(task1, rw_b, src_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(task3, rw_b, src_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(rw_b, task2, dst_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(task2, write_c, src_conn='C', memlet=dace.Memlet(data='C', subset='0'))

    assert 'B' not in state.read_and_write_sets()[0]


if __name__ == '__main__':
    test_read_write_set()
    test_read_write_set_y_formation()
