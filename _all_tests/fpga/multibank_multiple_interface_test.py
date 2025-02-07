# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation.dataflow.map_unroll import MapUnroll
from dace import dtypes, subsets
import dace
from dace import memlet
from dace.fpga_testing import xilinx_test
import numpy as np
from dace.sdfg import SDFG
from dace.transformation.interstate import InlineSDFG
from dace.config import set_temporary
# Checks multiple interfaces attached to the same HBM/DDR-bank.


def four_interface_to_2_banks(mem_type, decouple_interfaces):
    sdfg = SDFG("test_4_interface_to_2_banks_" + mem_type)
    state = sdfg.add_state()

    _, desc_a = sdfg.add_array("a", [2, 2], dace.int32)
    desc_a.location["memorytype"] = mem_type
    desc_a.location["bank"] = "0:2"
    acc_read1 = state.add_read("a")
    acc_write1 = state.add_write("a")

    t1 = state.add_tasklet("r1", set(["_x1", "_x2"]), set(["_y1"]), "_y1 = _x1 + _x2")

    m1_in, m1_out = state.add_map("m", {"k": "0:2"}, dtypes.ScheduleType.Unrolled)

    state.add_memlet_path(acc_read1, m1_in, t1, memlet=memlet.Memlet("a[0, 0]"), dst_conn="_x1")
    state.add_memlet_path(acc_read1, m1_in, t1, memlet=memlet.Memlet("a[1, 0]"), dst_conn="_x2")
    state.add_memlet_path(t1, m1_out, acc_write1, memlet=memlet.Memlet("a[0, 1]"), src_conn="_y1")

    sdfg.apply_fpga_transformations()
    assert sdfg.apply_transformations(InlineSDFG) == 1
    assert sdfg.apply_transformations(MapUnroll) == 1
    for node in sdfg.states()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.Tasklet):
            sdfg.states()[0].out_edges(node)[0].data.subset = subsets.Range.from_string("1, 1")
            break

    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=decouple_interfaces):
        bank_assignment = sdfg.generate_code()[3].clean_code
        # if we are not decoupling array interfaces we will use less mem interfaces
        assert bank_assignment.count("sp") == 6 if decouple_interfaces else 4
        assert bank_assignment.count(mem_type + "[0]") == 3 if decouple_interfaces else 2
        assert bank_assignment.count(mem_type + "[1]") == 3 if decouple_interfaces else 2

    a = np.zeros([2, 2], np.int32)
    a[0, 0] = 2
    a[1, 0] = 3
    sdfg(a=a)
    assert a[0, 1] == 5

    return sdfg


@xilinx_test(assert_ii_1=False)
def test_4_interface_to_2_banks_ddr_non_decoupled_interfaces():
    return four_interface_to_2_banks(mem_type="DDR", decouple_interfaces=False)


@xilinx_test(assert_ii_1=False)
def test_4_interface_to_2_banks_ddr_decoupled_interfaces():
    return four_interface_to_2_banks(mem_type="DDR", decouple_interfaces=True)


@xilinx_test(assert_ii_1=False)
def test_4_interface_to_2_banks_hbm_non_decoupled_interface():
    return four_interface_to_2_banks(mem_type="HBM", decouple_interfaces=False)


@xilinx_test(assert_ii_1=False)
def test_4_interface_to_2_banks_hbm_decoupled_interface():
    return four_interface_to_2_banks(mem_type="HBM", decouple_interfaces=True)


if __name__ == "__main__":
    test_4_interface_to_2_banks_hbm_decoupled_interface(None)
    test_4_interface_to_2_banks_hbm_non_decoupled_interface(None)
    test_4_interface_to_2_banks_ddr_decoupled_interfaces(None)
    test_4_interface_to_2_banks_ddr_non_decoupled_interfaces(None)
