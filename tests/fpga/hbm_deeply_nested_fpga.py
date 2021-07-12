# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace import subsets

# A test checking HBM in the context of nested maps and nested sdfgs


def create_deeply_nested_sdfg():
    sdfg = dace.SDFG("deepnest_test")
    state: dace.SDFGState = sdfg.add_state("init")
    xarr = state.add_array("x", [4, 10], dace.float32)
    sdfg.arrays["x"].location["memorytype"] = "hbm"
    sdfg.arrays["x"].location["bank"] = "0:4"
    yarr = state.add_array("y", [4, 10], dace.float32)
    sdfg.arrays["y"].location["memorytype"] = "hbm"
    sdfg.arrays["y"].location["bank"] = "4:8"

    top_map_entry, top_map_exit = state.add_map("topmap", dict(k="0:2"))
    top_map_entry.schedule = dtypes.ScheduleType.Unrolled

    nsdfg = dace.SDFG("nest")
    nstate = nsdfg.add_state("nested_state")
    x_read = nstate.add_array("xin", [4, 10], dace.float32,
                              dtypes.StorageType.FPGA_Global)
    x_write = nstate.add_array("xout", [4, 10], dace.float32,
                               dtypes.StorageType.FPGA_Global)
    nsdfg.arrays["xin"].location["memorytype"] = "hbm"
    nsdfg.arrays["xin"].location["bank"] = "0:4"
    nsdfg.arrays["xout"].location["memorytype"] = "hbm"
    nsdfg.arrays["xout"].location["bank"] = "4:8"
    map_entry, map_exit = nstate.add_map("map1", dict(w="0:2"))
    map_entry.schedule = dtypes.ScheduleType.Unrolled
    imap_entry, imap_exit = nstate.add_map("map2", dict(i="0:10"))
    nope = nstate.add_tasklet("nop", dict(_in=None), dict(_out=None),
                              "_out = _in")
    input_mem = mem.Memlet("xin[2*k+w, i]")
    output_mem = mem.Memlet("xout[2*k+w, i]")
    nstate.add_memlet_path(x_read,
                           map_entry,
                           imap_entry,
                           nope,
                           memlet=input_mem,
                           dst_conn="_in")
    nstate.add_memlet_path(nope,
                           imap_exit,
                           map_exit,
                           x_write,
                           memlet=output_mem,
                           src_conn="_out")
    nsdfg_node = state.add_nested_sdfg(nsdfg, state, set(["xin"]),
                                       set(['xout']))

    state.add_memlet_path(xarr,
                          top_map_entry,
                          nsdfg_node,
                          memlet=mem.Memlet.from_array("x", sdfg.arrays["x"]),
                          dst_conn="xin")
    state.add_memlet_path(nsdfg_node,
                          top_map_exit,
                          yarr,
                          memlet=mem.Memlet.from_array("y", sdfg.arrays["y"]),
                          src_conn="xout")
    sdfg.apply_fpga_transformations()

    return sdfg


def exec_deeply_nested_test():
    sdfg = create_deeply_nested_sdfg()
    a = np.zeros((4, 10), np.float32)
    a[2, 4:9] += 1
    a[3, 3:8] += 2
    a[0, 7] += 3
    c = np.ones((4, 10), np.float32)
    sdfg(x=a, y=c)
    assert np.allclose(a, c, 10e-6)


if __name__ == "__main__":
    exec_deeply_nested_test()
