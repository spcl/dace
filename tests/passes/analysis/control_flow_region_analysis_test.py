# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests analysis passes related to control flow regions (control_flow_region_analysis.py). """


import dace
from dace.memlet import Memlet
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis.control_flow_region_analysis import CFGDataDependence, StateDataDependence


def test_simple_state_data_dependence_with_self_contained_read():
    sdfg = SDFG('myprog')
    N = dace.symbol('N')
    sdfg.add_array('A', (N,), dace.float32)
    sdfg.add_array('B', (N,), dace.float32)
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

    propagate_memlets_sdfg(sdfg)

    res = {}
    Pipeline([StateDataDependence()]).apply_pass(sdfg, res)
    state_data_deps = res[StateDataDependence.__name__][0][sdfg.states()[0]]

    assert len(state_data_deps[0]) == 1
    read_memlet: Memlet = list(state_data_deps[0])[0]
    assert read_memlet.data == 'B'
    assert read_memlet.subset[0][0] == 0
    assert read_memlet.subset[0][1] == 0.5 * N - 1

    assert len(state_data_deps[1]) == 3


'''
def test_nested_cf_region_data_dependence():
    N = dace.symbol('N')

    @dace.program
    def myprog(A: dace.float64[N], B: dace.float64):
        for i in range(N):
            with dace.tasklet:
                in1 << B[i]
                out1 >> A[i]
                out1 = in1 + 1
        for i in range(N):
            with dace.tasklet:
                in1 << A[i]
                out1 >> B[i]
                out1 = in1 * 2

    myprog.use_experimental_cfg_blocks = True

    sdfg = myprog.to_sdfg()

    res = {}
    pipeline = Pipeline([CFGDataDependence()])
    pipeline.__experimental_cfg_block_compatible__ = True
    pipeline.apply_pass(sdfg, res)

    print(sdfg)
    '''


if __name__ == '__main__':
    test_simple_state_data_dependence_with_self_contained_read()
    #test_nested_cf_region_data_dependence()
