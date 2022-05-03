# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace import subsets
from dace.fpga_testing import xilinx_test
import numpy as np
import pytest
from dace.config import set_temporary


# A test checking wcr-reduction with HBM/DDR arrays as inputs and output


def create_multibank_reduce_sdfg(
    name,
    mem_type,
    banks=2,
):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = dace.SDFG(name + "_" + mem_type)
    state = sdfg.add_state('red_' + mem_type, True)

    in1 = sdfg.add_array("in1", [banks, N, M], dace.float32)
    in2 = sdfg.add_array("in2", [banks, N, M], dace.float32)
    out = sdfg.add_array("out", [banks, N], dace.float32)
    in1[1].location["memorytype"] = mem_type
    in2[1].location["memorytype"] = mem_type
    out[1].location["memorytype"] = mem_type
    in1[1].location["bank"] = f"0:{banks}"
    in2[1].location["bank"] = f"{banks}:{2*banks}"
    out[1].location["bank"] = f"{2*banks}:{3*banks}"

    read_in1 = state.add_read("in1")
    read_in2 = state.add_read("in2")
    out_write = state.add_write("out")
    tmp_in1_memlet = dace.Memlet(f"in1[k, i, j]")
    tmp_in2_memlet = dace.Memlet(f"in2[k, i, j]")
    tmp_out_memlet = dace.Memlet(f"out[k, i]", wcr="lambda x,y: x+y")

    outer_entry, outer_exit = state.add_map("vadd_outer_map", dict(k=f'0:{banks}'))
    map_entry, map_exit = state.add_map("vadd_inner_map", dict(i="0:N", j="0:M"))
    tasklet = state.add_tasklet("mul", dict(__in1=None, __in2=None), dict(__out=None), '__out = __in1 * __in2')
    outer_entry.map.schedule = dace.ScheduleType.Unrolled

    state.add_memlet_path(read_in1, outer_entry, map_entry, tasklet, memlet=tmp_in1_memlet, dst_conn="__in1")
    state.add_memlet_path(read_in2, outer_entry, map_entry, tasklet, memlet=tmp_in2_memlet, dst_conn="__in2")
    state.add_memlet_path(tasklet, map_exit, outer_exit, out_write, memlet=tmp_out_memlet, src_conn="__out")

    sdfg.apply_fpga_transformations()
    return sdfg


def create_test_set(N, M, banks):
    in1 = np.random.rand(*[banks, N, M]).astype('f')
    in2 = np.random.rand(*[banks, N, M]).astype('f')
    expected = np.sum(in1 * in2, axis=2, dtype=np.float32)
    out = np.zeros((banks, N), dtype=np.float32)
    return (in1, in2, expected, out)


def exec_test(N, M, banks, mem_type, name):
    in1, in2, expected, target = create_test_set(N, M, banks)
    sdfg = create_multibank_reduce_sdfg(name, mem_type, banks)
    sdfg(in1=in1, in2=in2, out=target, N=N, M=M)
    assert np.allclose(expected, target, rtol=1e-6)
    return sdfg


@xilinx_test()
def test_hbm_reduce_2x3_2b():
    return exec_test(2, 3, 2, "hbm", "red_2x3_2b")

@xilinx_test()
def test_hbm_reduce_2x3_2b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(2, 3, 2, "hbm", "red_2x3_2b_decoupled")


@xilinx_test()
def test_hbm_reduce_10x50_4b():
    return exec_test(10, 50, 4, "hbm", "red_10x50_4b")

@xilinx_test()
def test_hbm_reduce_10x50_4b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(10, 50, 4, "hbm", "red_10x50_4b_decoupled")


@xilinx_test()
def test_hbm_reduce_red_1x50_1b():
    return exec_test(1, 50, 1, "hbm", "red_1x50_1b")

@xilinx_test()
def test_hbm_reduce_red_1x50_1b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(1, 50, 1, "hbm", "red_1x50_1b_decoupled")


@xilinx_test()
def test_hbm_reduce_red_1x40_8b():
    return exec_test(1, 40, 8, "hbm", "red_1x40_8b")

@xilinx_test()
def test_hbm_reduce_red_1x40_8b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(1, 40, 8, "hbm", "red_1x40_8b_decoupled")


@xilinx_test()
def test_hbm_reduce_red_2x40_6b():
    return exec_test(2, 40, 6, "hbm", "red_2x40_6b")

@xilinx_test()
def test_hbm_reduce_red_2x40_6b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(2, 40, 6, "hbm", "red_2x40_6b_decoupled")


@xilinx_test()
def test_ddr_reduce_2x3_2b():
    return exec_test(2, 3, 2, "ddr", "red_2x3_2b")

@xilinx_test()
def test_ddr_reduce_2x3_2b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(2, 3, 2, "ddr", "red_2x3_2b_decoupled")


@xilinx_test()
def test_ddr_reduce_10x50_4b():
    return exec_test(10, 50, 4, "ddr", "red_10x50_4b")

@xilinx_test()
def test_ddr_reduce_10x50_4b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(10, 50, 4, "ddr", "red_10x50_4b_decoupled")


@xilinx_test()
def test_ddr_reduce_red_1x50_1b():
    return exec_test(1, 50, 1, "ddr", "red_1x50_1b")

@xilinx_test()
def test_ddr_reduce_red_1x50_1b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(1, 50, 1, "ddr", "red_1x50_1b_decoupled")


@xilinx_test()
def test_ddr_reduce_red_1x40_8b():
    return exec_test(1, 40, 8, "ddr", "red_1x40_8b")

@xilinx_test()
def test_ddr_reduce_red_1x40_8b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(1, 40, 8, "ddr", "red_1x40_8b_decoupled")


@xilinx_test()
def test_ddr_reduce_red_2x40_6b():
    return exec_test(2, 40, 6, "ddr", "red_2x40_6b")

@xilinx_test()
def test_ddr_reduce_red_2x40_6b_decouple_array_interfaces():
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return exec_test(2, 40, 6, "ddr", "red_2x40_6b_decoupled")


if __name__ == "__main__":
    test_hbm_reduce_2x3_2b(None)
    test_hbm_reduce_10x50_4b(None)
    test_hbm_reduce_red_1x50_1b(None)
    test_hbm_reduce_red_1x40_8b(None)
    test_hbm_reduce_red_2x40_6b(None)
    test_ddr_reduce_2x3_2b(None)
    test_ddr_reduce_10x50_4b(None)
    test_ddr_reduce_red_1x50_1b(None)
    test_ddr_reduce_red_1x40_8b(None)
    test_ddr_reduce_red_2x40_6b(None)
