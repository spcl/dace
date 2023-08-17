# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_inout_connector_validation_success():

    sdfg = dace.SDFG("test_inout_connector_validation_success")
    sdfg.add_array("A", [1], dace.int32)
    sdfg.add_array("B", [1], dace.int32)

    nsdfg = dace.SDFG("nested_sdfg")
    nsdfg.add_array("C", [1], dace.int32)

    nstate = nsdfg.add_state()
    read_c = nstate.add_access("C")
    write_c = nstate.add_access("C")
    tasklet = nstate.add_tasklet("tasklet", {"__inp"}, {"__out"}, "__out = __inp + 5")
    nstate.add_edge(read_c, None, tasklet, '__inp', dace.Memlet.from_array('C', nsdfg.arrays['C']))
    nstate.add_edge(tasklet, '__out', write_c, None, dace.Memlet.from_array('C', nsdfg.arrays['C']))

    state = sdfg.add_state()
    read_b = state.add_access("B")
    write_b = state.add_access("B")
    tasklet = state.add_nested_sdfg(nsdfg, sdfg, {"C"}, {"C"})
    state.add_edge(read_b, None, tasklet, 'C', dace.Memlet.from_array('B', sdfg.arrays['B']))
    state.add_edge(tasklet, 'C', write_b, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    try:
        sdfg.validate()
    except dace.sdfg.InvalidSDFGError:
        assert False, "SDFG should validate"

    return


def test_inout_connector_validation_fail():

    sdfg = dace.SDFG("test_inout_connector_validation_fail")
    sdfg.add_array("A", [1], dace.int32)
    sdfg.add_array("B", [1], dace.int32)

    nsdfg = dace.SDFG("nested_sdfg")
    nsdfg.add_array("C", [1], dace.int32)

    nstate = nsdfg.add_state()
    read_c = nstate.add_access("C")
    write_c = nstate.add_access("C")
    tasklet = nstate.add_tasklet("tasklet", {"__inp"}, {"__out"}, "__out = __inp + 5")
    nstate.add_edge(read_c, None, tasklet, '__inp', dace.Memlet.from_array('C', nsdfg.arrays['C']))
    nstate.add_edge(tasklet, '__out', write_c, None, dace.Memlet.from_array('C', nsdfg.arrays['C']))

    state = sdfg.add_state()
    read_a = state.add_access("A")
    write_b = state.add_access("B")
    tasklet = state.add_nested_sdfg(nsdfg, sdfg, {"C"}, {"C"})
    state.add_edge(read_a, None, tasklet, 'C', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(tasklet, 'C', write_b, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    try:
        sdfg.validate()
    except dace.sdfg.InvalidSDFGError:
        return

    assert False, "SDFG should not validate"


def test_nested_sdfg_with_transient_connector():

    sdfg = dace.SDFG('nested_main')
    sdfg.add_array('A', [2], dace.float32)

    def mystate(state, src, dst):
        src_node = state.add_read(src)
        dst_node = state.add_write(dst)
        tasklet = state.add_tasklet('aaa2', {'a'}, {'b'}, 'b = a + 1')

        # input path (src->tasklet[a])
        state.add_memlet_path(src_node, tasklet, dst_conn='a', memlet=dace.Memlet(data=src, subset='0'))
        # output path (tasklet[b]->dst)
        state.add_memlet_path(tasklet, dst_node, src_conn='b', memlet=dace.Memlet(data=dst, subset='0'))


    sub_sdfg = dace.SDFG('nested_sub')
    sub_sdfg.add_scalar('sA', dace.float32)
    sub_sdfg.add_scalar('sB', dace.float32, transient=True)
    sub_sdfg.add_scalar('sC', dace.float32, transient=True)

    state0 = sub_sdfg.add_state('subs0')
    mystate(state0, 'sA', 'sB')
    state1 = sub_sdfg.add_state('subs1')
    mystate(state1, 'sB', 'sC')

    sub_sdfg.add_edge(state0, state1, dace.InterstateEdge())


    state = sdfg.add_state('s0')
    me, mx = state.add_map('mymap', dict(k='0:2'))
    nsdfg = state.add_nested_sdfg(sub_sdfg, sdfg, {'sA'}, {'sC'})
    Ain = state.add_read('A')
    Aout = state.add_write('A')

    state.add_memlet_path(Ain, me, nsdfg, memlet=dace.Memlet(data='A', subset='k'), dst_conn='sA')
    state.add_memlet_path(nsdfg, mx, Aout, memlet=dace.Memlet(data='A', subset='k'), src_conn='sC')
 
    try:
        sdfg.validate()
    except dace.sdfg.InvalidSDFGError:
        return

    assert False, "SDFG should not validate"


if __name__ == "__main__":
    test_inout_connector_validation_success()
    test_inout_connector_validation_fail()
    test_nested_sdfg_with_transient_connector()
