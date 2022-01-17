# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests dereferencing issues with tasklets that use dynamic memlets. """
import dace
import numpy as np


def test_dynamic_memlets():
    """ Tests dynamic memlet dereferencing on one value. """
    sdfg = dace.SDFG('test')
    state = sdfg.add_state('state')
    sdfg.add_array('out_arr1', dtype=dace.float64, shape=(3, 3))
    sdfg.add_array('out_arr2', dtype=dace.float64, shape=(3, 3))
    tasklet = state.add_tasklet('tasklet', inputs={}, outputs={'o1', 'o2'}, code='o1 = 1.0; o2 = 2 * o1')
    map_entry, map_exit = state.add_map('map', ndrange=dict(i='0:3', j='0:3'))
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet,
                          map_exit,
                          state.add_write('out_arr1'),
                          src_conn='o1',
                          memlet=dace.Memlet.simple('out_arr1', subset_str='i,j'))
    state.add_memlet_path(tasklet,
                          map_exit,
                          state.add_write('out_arr2'),
                          src_conn='o2',
                          memlet=dace.Memlet.simple('out_arr2', subset_str='i,j'))
    sdfg.validate()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, (dace.nodes.Tasklet, dace.nodes.MapExit)):
                for edge in state.out_edges(node):
                    edge.data.dynamic = True

    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)
    sdfg(out_arr1=A, out_arr2=B)
    assert np.allclose(A, 1)
    assert np.allclose(B, 2)


def test_dynamic_memlets_subset():
    """ 
    Tests dynamic memlet dereferencing when subset/pointer is used
    in tasklet connector.
    """
    sdfg = dace.SDFG('test')
    state = sdfg.add_state('state')
    sdfg.add_array('out_arr1', dtype=dace.float64, shape=(3, 3))
    sdfg.add_array('out_arr2', dtype=dace.float64, shape=(3, 3))
    tasklet = state.add_tasklet('tasklet', inputs={}, outputs={'o1', 'o2'}, code='o1 = 1.0; o2[i, j] = 2 * o1')
    map_entry, map_exit = state.add_map('map', ndrange=dict(i='0:3', j='0:3'))
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet,
                          map_exit,
                          state.add_write('out_arr1'),
                          src_conn='o1',
                          memlet=dace.Memlet.simple('out_arr1', subset_str='i,j'))
    state.add_memlet_path(tasklet,
                          map_exit,
                          state.add_write('out_arr2'),
                          src_conn='o2',
                          memlet=dace.Memlet('out_arr2[0:3, 0:3]'))
    sdfg.validate()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, (dace.nodes.Tasklet, dace.nodes.MapExit)):
                for edge in state.out_edges(node):
                    edge.data.dynamic = True

    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)
    sdfg(out_arr1=A, out_arr2=B)
    assert np.allclose(A, 1)
    assert np.allclose(B, 2)


if __name__ == '__main__':
    test_dynamic_memlets()
    test_dynamic_memlets_subset()
