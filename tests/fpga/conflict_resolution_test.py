# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests whether conflict resolution is handled correctly on both local and
# global memory containers from within an FPGA kernel.

import dace
import numpy as np
from dace.fpga_testing import fpga_test


def make_sdfg():

    N = dace.symbol("N")

    sdfg = dace.SDFG("fpga_conflict_resolution")

    sdfg.add_array("host_memory", [N], dace.int32)
    sdfg.add_array("global_memory", [N], dace.int32, transient=True, storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("local_memory", [1], dace.int32, transient=True, storage=dace.StorageType.FPGA_Local)

    state = sdfg.add_state("fpga_conflict_resolution")

    # Copy memory to FPGA
    pre_state = sdfg.add_state("pre_state")
    pre_host = pre_state.add_read("host_memory")
    pre_device = pre_state.add_write("global_memory")
    pre_state.add_memlet_path(pre_host, pre_device, memlet=dace.Memlet("global_memory[0:N]"))
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())

    # Copy memory back
    post_state = sdfg.add_state("post_state")
    post_device = post_state.add_read("global_memory")
    post_host = post_state.add_write("host_memory")
    post_state.add_memlet_path(post_device, post_host, memlet=dace.Memlet("global_memory[0:N]"))
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    gmem_read = state.add_read("global_memory")
    gmem_write = state.add_write("global_memory")

    local_init = state.add_access("local_memory")
    local_write = state.add_access("local_memory")

    # Initialize local memory
    init_tasklet = state.add_tasklet("init", {}, {"out"}, "out = 0")
    state.add_memlet_path(init_tasklet, local_init, src_conn="out", memlet=dace.Memlet("local_memory[0]"))

    # Accumulate on local memory
    acc_entry, acc_exit = state.add_map("wcr_local", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)
    acc_tasklet = state.add_tasklet("wcr_local", {"gmem"}, {"lmem"}, "lmem = gmem")
    state.add_memlet_path(gmem_read, acc_entry, acc_tasklet, dst_conn="gmem", memlet=dace.Memlet("global_memory[i]"))
    state.add_memlet_path(local_init, acc_entry, memlet=dace.Memlet())
    state.add_memlet_path(acc_tasklet,
                          acc_exit,
                          local_write,
                          src_conn="lmem",
                          memlet=dace.Memlet("local_memory[0]", wcr="lambda a, b: a + b"))

    # Write with conflict into global memory
    wcr_entry, wcr_exit = state.add_map("wcr_global", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)
    wcr_tasklet = state.add_tasklet("wcr_global", {"lmem"}, {"gmem"}, "gmem = lmem")
    state.add_memlet_path(local_write, wcr_entry, wcr_tasklet, dst_conn="lmem", memlet=dace.Memlet("local_memory[0]"))
    state.add_memlet_path(wcr_tasklet,
                          wcr_exit,
                          gmem_write,
                          src_conn="gmem",
                          memlet=dace.Memlet("global_memory[i]", wcr="lambda a, b: a + b"))

    return sdfg


@fpga_test()
def test_fpga_wcr():
    sdfg = make_sdfg()
    size = 128
    host_memory = np.arange(size, dtype=np.int32)
    reference = host_memory.copy()
    sdfg(host_memory=host_memory, N=size)
    assert all(np.sum(reference) + reference == host_memory)
    return sdfg


if __name__ == "__main__":
    test_fpga_wcr(None)
