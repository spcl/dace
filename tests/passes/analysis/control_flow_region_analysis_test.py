# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests analysis passes related to control flow regions (control_flow_region_analysis.py). """

import dace
from dace.memlet import Memlet
from dace.sdfg.sdfg import SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis.control_flow_region_analysis import StateDataDependence


def test_state_data_dependence_with_contained_read():
    sdfg = SDFG('myprog')
    N = dace.symbol('N')
    sdfg.add_array('A', (N, ), dace.float32)
    sdfg.add_array('B', (N, ), dace.float32)
    mystate = sdfg.add_state('mystate', is_start_block=True)
    b_read = mystate.add_access('B')
    b_write_second_half = mystate.add_access('B')
    b_write_first_half = mystate.add_access('B')
    a_read_write = mystate.add_access('A')
    first_entry, first_exit = mystate.add_map('map_one', {'i': '0:0.5*N'})
    second_entry, second_exit = mystate.add_map('map_two', {'i': '0:0.5*N'})
    t1 = mystate.add_tasklet('t1', {'i1'}, {'o1'}, 'o1 = i1 + 1.0')
    t2 = mystate.add_tasklet('t2', {'i1'}, {'o1'}, 'o1 = i1 - 1.0')
    t3 = mystate.add_tasklet('t3', {'i1'}, {'o1'}, 'o1 = i1 * 2.0')
    mystate.add_memlet_path(b_read, first_entry, t1, memlet=Memlet('B[i]'), dst_conn='i1')
    mystate.add_memlet_path(b_read, first_entry, t2, memlet=Memlet('B[i]'), dst_conn='i1')
    mystate.add_memlet_path(t1, first_exit, a_read_write, memlet=Memlet('A[i]'), src_conn='o1')
    mystate.add_memlet_path(t2, first_exit, b_write_second_half, memlet=Memlet('B[N - (i + 1)]'), src_conn='o1')
    mystate.add_memlet_path(a_read_write, second_entry, t3, memlet=Memlet('A[i]'), dst_conn='i1')
    mystate.add_memlet_path(t3, second_exit, b_write_first_half, memlet=Memlet('B[i]'), src_conn='o1')

    res = {}
    Pipeline([StateDataDependence()]).apply_pass(sdfg, res)
    state_data_deps = res[StateDataDependence.__name__][0][sdfg.states()[0]]

    assert len(state_data_deps[0]) == 1
    read_memlet: Memlet = list(state_data_deps[0])[0]
    assert read_memlet.data == 'B'
    assert read_memlet.subset[0][0] == 0
    assert read_memlet.subset[0][1] == 0.5 * N - 1

    assert len(state_data_deps[1]) == 3


def test_state_data_dependence_with_contained_read_in_map():
    sdfg = SDFG('myprog')
    N = dace.symbol('N')
    sdfg.add_array('A', (N, ), dace.float32)
    sdfg.add_transient('tmp', (N, ), dace.float32)
    sdfg.add_array('B', (N, ), dace.float32)
    mystate = sdfg.add_state('mystate', is_start_block=True)
    a_read = mystate.add_access('A')
    tmp = mystate.add_access('tmp')
    b_write = mystate.add_access('B')
    m_entry, m_exit = mystate.add_map('my_map', {'i': 'N'})
    t1 = mystate.add_tasklet('t1', {'i1'}, {'o1'}, 'o1 = i1 * 2.0')
    t2 = mystate.add_tasklet('t2', {'i1'}, {'o1'}, 'o1 = i1 - 1.0')
    mystate.add_memlet_path(a_read, m_entry, t1, memlet=Memlet('A[i]'), dst_conn='i1')
    mystate.add_memlet_path(t1, tmp, memlet=Memlet('tmp[i]'), src_conn='o1')
    mystate.add_memlet_path(tmp, t2, memlet=Memlet('tmp[i]'), dst_conn='i1')
    mystate.add_memlet_path(t2, m_exit, b_write, memlet=Memlet('B[i]'), src_conn='o1')

    res = {}
    Pipeline([StateDataDependence()]).apply_pass(sdfg, res)
    state_data_deps = res[StateDataDependence.__name__][0][sdfg.states()[0]]

    assert len(state_data_deps[0]) == 1
    read_memlet: Memlet = list(state_data_deps[0])[0]
    assert read_memlet.data == 'A'

    assert len(state_data_deps[1]) == 2
    out_containers = [m.data for m in state_data_deps[1]]
    assert 'B' in out_containers
    assert 'tmp' in out_containers
    assert 'A' not in out_containers


def test_state_data_dependence_with_non_contained_read_in_map():
    sdfg = SDFG('myprog')
    N = dace.symbol('N')
    sdfg.add_array('A', (N, ), dace.float32)
    sdfg.add_array('tmp', (N, ), dace.float32)
    sdfg.add_array('B', (N, ), dace.float32)
    mystate = sdfg.add_state('mystate', is_start_block=True)
    a_read = mystate.add_access('A')
    tmp = mystate.add_access('tmp')
    b_write = mystate.add_access('B')
    m_entry, m_exit = mystate.add_map('my_map', {'i': '0:ceil(N/2)'})
    t1 = mystate.add_tasklet('t1', {'i1'}, {'o1'}, 'o1 = i1 * 2.0')
    t2 = mystate.add_tasklet('t2', {'i1'}, {'o1'}, 'o1 = i1 - 1.0')
    mystate.add_memlet_path(a_read, m_entry, t1, memlet=Memlet('A[i]'), dst_conn='i1')
    mystate.add_memlet_path(t1, tmp, memlet=Memlet('tmp[i]'), src_conn='o1')
    mystate.add_memlet_path(tmp, t2, memlet=Memlet('tmp[i+ceil(N/2)]'), dst_conn='i1')
    mystate.add_memlet_path(t2, m_exit, b_write, memlet=Memlet('B[i]'), src_conn='o1')

    res = {}
    Pipeline([StateDataDependence()]).apply_pass(sdfg, res)
    state_data_deps = res[StateDataDependence.__name__][0][sdfg.states()[0]]

    assert len(state_data_deps[0]) == 2
    in_containers = [m.data for m in state_data_deps[0]]
    assert 'A' in in_containers
    assert 'tmp' in in_containers
    assert 'B' not in in_containers

    assert len(state_data_deps[1]) == 2
    out_containers = [m.data for m in state_data_deps[1]]
    assert 'B' in out_containers
    assert 'tmp' in out_containers
    assert 'A' not in out_containers


if __name__ == '__main__':
    test_state_data_dependence_with_contained_read()
    test_state_data_dependence_with_contained_read_in_map()
    test_state_data_dependence_with_non_contained_read_in_map()
