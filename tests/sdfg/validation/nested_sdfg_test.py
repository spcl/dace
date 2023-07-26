# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_inout_connector():

    sdfg = dace.SDFG("test_inout_connector")
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


if __name__ == "__main__":
    test_inout_connector()
