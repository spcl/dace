# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import numpy as np

from dace.transformation.dataflow import RemoveIntermediateWrite


def test_write_before_map_exit():

    sdfg = dace.SDFG('test_write_before_map_exit')
    sdfg.add_array('A', (10, ), dace.int32)
    sdfg.add_array('B', (10, ), dace.int32)

    state = sdfg.add_state('state')
    me, mx = state.add_map('map', dict(i='0:10'))
    a_access = state.add_read('A')
    b_access = state.add_write('B')
    tasklet = state.add_tasklet('tasklet', {'__inp'}, {'__out'}, '__out = __inp')
    state.add_memlet_path(a_access, me, tasklet, memlet=dace.Memlet(data='A', subset='i'), dst_conn='__inp')
    state.add_edge(tasklet, '__out', b_access, None, dace.Memlet(data='B', subset='i'))
    state.add_edge(b_access, None, mx, None, dace.Memlet())

    A = np.arange(10, dtype=np.int32)
    ref = A

    before_val = np.empty((10, ), dtype=np.int32)
    after_val = np.empty((10, ), dtype=np.int32)

    sdfg_before = copy.deepcopy(sdfg)
    sdfg_before(A=A, B=before_val)
    assert np.allclose(before_val, ref)

    sdfg.apply_transformations_repeated(RemoveIntermediateWrite)
    sdfg(A=A, B=after_val)
    assert np.allclose(after_val, ref)


def test_write_before_nested_map_exit():

    sdfg = dace.SDFG('test_write_before_nested_map_exit')
    sdfg.add_array('A', (10, 10), dace.int32)
    sdfg.add_array('B', (10, 10), dace.int32)

    state = sdfg.add_state('state')
    me0, mx0 = state.add_map('map', dict(i='0:10'))
    me1, mx1 = state.add_map('map2', dict(j='0:10'))
    a_access = state.add_read('A')
    b_access = state.add_write('B')
    tasklet = state.add_tasklet('tasklet', {'__inp'}, {'__out'}, '__out = __inp')
    state.add_memlet_path(a_access, me0, me1, tasklet, memlet=dace.Memlet(data='A', subset='i, j'), dst_conn='__inp')
    state.add_edge(tasklet, '__out', b_access, None, dace.Memlet(data='B', subset='i, j'))
    state.add_nedge(b_access, mx1, dace.Memlet())
    state.add_nedge(mx1, mx0, dace.Memlet())

    A = np.arange(100, dtype=np.int32).reshape((10, 10)).copy()
    ref = A

    before_val = np.empty((10, 10), dtype=np.int32)
    after_val = np.empty((10, 10), dtype=np.int32)

    sdfg_before = copy.deepcopy(sdfg)
    sdfg_before(A=A, B=before_val)
    assert np.allclose(before_val, ref)

    sdfg.apply_transformations_repeated(RemoveIntermediateWrite)
    sdfg(A=A, B=after_val)
    assert np.allclose(after_val, ref)


def test_write_before_nested_map_exit_2():

    sdfg = dace.SDFG('test_write_before_nested_map_exit_2')
    sdfg.add_array('A', (10, 10), dace.int32)
    sdfg.add_array('B', (10, 10), dace.int32)
    sdfg.add_array('C', (10, ), dace.int32, transient=True)

    state = sdfg.add_state('state')
    me0, mx0 = state.add_map('map', dict(i='0:10'))
    me1, mx1 = state.add_map('map2', dict(j='0:10'))
    a_access = state.add_read('A')
    b_access = state.add_write('B')
    c_access = state.add_write('C')
    tasklet0 = state.add_tasklet('tasklet0', {'__inp'}, {'__out'}, '__out = __inp')
    tasklet1 = state.add_tasklet('tasklet1', {'__inp'}, {'__out'}, '__out = __inp')
    state.add_memlet_path(a_access, me0, me1, tasklet0, memlet=dace.Memlet(data='A', subset='i, j'), dst_conn='__inp')
    state.add_memlet_path(a_access, me0, me1, tasklet1, memlet=dace.Memlet(data='A', subset='i, j'), dst_conn='__inp')
    state.add_edge(tasklet0, '__out', b_access, None, dace.Memlet(data='B', subset='i, j'))
    state.add_edge(tasklet1, '__out', c_access, None, dace.Memlet(data='C', subset='j'))
    state.add_nedge(b_access, mx1, dace.Memlet())
    state.add_nedge(c_access, mx1, dace.Memlet())
    state.add_nedge(mx1, mx0, dace.Memlet())

    A = np.arange(100, dtype=np.int32).reshape((10, 10)).copy()
    ref = A

    before_val = np.empty((10, 10), dtype=np.int32)
    after_val = np.empty((10, 10), dtype=np.int32)

    sdfg_before = copy.deepcopy(sdfg)
    sdfg_before(A=A, B=before_val)
    assert np.allclose(before_val, ref)

    sdfg.apply_transformations_repeated(RemoveIntermediateWrite)
    c_nodes = [n for n in state.data_nodes() if n.data == 'C']
    assert len(c_nodes) == 1
    assert len(state.edges_between(tasklet1, c_nodes[0])) == 0
    assert len(state.edges_between(c_nodes[0], mx1)) == 0
    assert len(state.edges_between(mx1, c_nodes[0])) == 1
    assert len(state.edges_between(c_nodes[0], mx0)) == 1
    assert len(state.edges_between(mx0, c_nodes[0])) == 0
    sdfg(A=A, B=after_val)
    assert np.allclose(after_val, ref) 


if __name__ == '__main__':
    test_write_before_map_exit()
    test_write_before_nested_map_exit()
    test_write_before_nested_map_exit_2()
