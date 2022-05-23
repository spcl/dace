import argparse
import dace
import numpy as np
import re
from dace.fpga_testing import intel_fpga_test

DTYPE = dace.float32


def make_sdfg():

    N = dace.symbol("N", DTYPE)
    P = dace.symbol("P", DTYPE)

    sdfg = dace.SDFG("autorun_test")

    pre_state = sdfg.add_state("host_to_device")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("device_to_host")

    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    sdfg.add_array("arr_host", (N, ), DTYPE)
    sdfg.add_array("arr", (N, ), DTYPE, storage=dace.StorageType.FPGA_Global, transient=True)

    # Copy from host to device
    pre_host = pre_state.add_read("arr_host")
    pre_device = pre_state.add_write("arr")
    pre_state.add_memlet_path(pre_host, pre_device, memlet=dace.Memlet("arr[0:N]"))

    # Copy from device to host
    post_device = post_state.add_read("arr")
    post_host = post_state.add_write("arr_host")
    post_state.add_memlet_path(post_device, post_host, memlet=dace.Memlet("arr_host[0:N]"))

    sdfg.add_stream("pipe_in", DTYPE, storage=dace.StorageType.FPGA_Local, transient=True)

    # Read from memory into a stream
    memory_read = state.add_read("arr")
    pipe_in_write = state.add_write("pipe_in")
    state.add_memlet_path(memory_read, pipe_in_write, memlet=dace.Memlet("arr[0:N]", other_subset="0"))

    sdfg.add_stream("pipes_systolic", DTYPE, shape=(P + 1, ), storage=dace.StorageType.FPGA_Local, transient=True)

    # Simple processing element that can be autorun
    pipe_in_read = state.add_read("pipe_in")
    entry_add, exit_add = state.add_map("add", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)
    tasklet_add = state.add_tasklet("add", {"val_in"}, {"val_out"}, "val_out = val_in + 9")
    state.add_memlet_path(pipe_in_read, entry_add, tasklet_add, dst_conn="val_in", memlet=dace.Memlet("pipe_in[0]"))
    pipe_systolic_write_head = state.add_write("pipes_systolic")
    state.add_memlet_path(tasklet_add,
                          exit_add,
                          pipe_systolic_write_head,
                          src_conn="val_out",
                          memlet=dace.Memlet("pipes_systolic[0]"))

    # Systolic array which can be autorun
    unroll_entry, unroll_exit = state.add_map("systolic_array", {"p": "0:P"},
                                              schedule=dace.ScheduleType.FPGA_Device,
                                              unroll=True)
    pipe_unroll_read = state.add_read("pipes_systolic")
    state.add_memlet_path(unroll_entry, pipe_unroll_read, memlet=dace.Memlet())
    systolic_entry, systolic_exit = state.add_map("add_systolic", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)
    systolic_tasklet = state.add_tasklet("add_systolic", {"val_in"}, {"val_out"}, "val_out = 2 * val_in")
    state.add_memlet_path(pipe_unroll_read,
                          systolic_entry,
                          systolic_tasklet,
                          dst_conn="val_in",
                          memlet=dace.Memlet("pipes_systolic[p]"))
    pipe_unroll_write = state.add_write("pipes_systolic")
    state.add_memlet_path(systolic_tasklet,
                          systolic_exit,
                          pipe_unroll_write,
                          src_conn="val_out",
                          memlet=dace.Memlet("pipes_systolic[p + 1]"))
    state.add_memlet_path(pipe_unroll_write, unroll_exit, memlet=dace.Memlet())

    # Write back to memory
    pipe_systolic_read_tail = state.add_read("pipes_systolic")
    memory_write = state.add_write("arr")
    state.add_memlet_path(pipe_systolic_read_tail, memory_write, memlet=dace.Memlet("arr[0:N]", other_subset="P"))

    return sdfg


@intel_fpga_test()
def test_autorun():

    n = 128
    p = 4

    sdfg = make_sdfg()
    sdfg.specialize({"N": 128, "P": 4})

    arr = np.ones((128, ), dtype=DTYPE.type)

    for c in (c for c in sdfg.generate_code() if c.language == "cl"):
        if len(re.findall(r"__attribute__\(\(autorun\)\)", c.code)) != 2:
            raise RuntimeError("Autogen attributes not found.")

    sdfg(arr_host=arr)

    if any(arr != 2**4 * 10):
        raise ValueError("Verification failed.")

    return sdfg


if __name__ == "__main__":
    test_autorun(None)
