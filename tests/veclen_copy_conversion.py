#!/usr/bin/env python3
import copy
import dace


def make_sdfg(tasklet_code=None,
              name="veclen_copy_conversion",
              dtype=dace.float32,
              veclen=16):

    if tasklet_code is None:
        tasklet_code = "_out = _in"

    n = dace.symbol("N")

    sdfg = dace.SDFG(name)

    pre_state = sdfg.add_state(name + "_pre")
    state = sdfg.add_state(name)
    post_state = sdfg.add_state(name + "_post")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    _, desc_input_host = sdfg.add_array("a", (n, ), dtype)
    _, desc_output_host = sdfg.add_array("b", (n, ), dtype)
    desc_input_device = copy.copy(desc_input_host)
    desc_input_device.storage = dace.StorageType.FPGA_Global
    desc_input_device.location["bank"] = 0
    desc_input_device.transient = True
    desc_output_device = copy.copy(desc_output_host)
    desc_output_device.storage = dace.StorageType.FPGA_Global
    desc_output_device.location["bank"] = 1
    desc_output_device.transient = True
    sdfg.add_datadesc("a_device", desc_input_device)
    sdfg.add_datadesc("b_device", desc_output_device)

    # Host to device
    pre_read = pre_state.add_read("a")
    pre_write = pre_state.add_write("a_device")
    pre_state.add_memlet_path(pre_read,
                              pre_write,
                              memlet=dace.Memlet.simple(pre_write, "0:N"))

    # Device to host
    post_read = post_state.add_read("a_device")
    post_write = post_state.add_write("a")
    post_state.add_memlet_path(post_read,
                               post_write,
                               memlet=dace.Memlet.simple(
                                   post_write, "0:N"))

    # Compute state
    read_memory = state.add_read("a_device")
    write_memory = state.add_write("b_device")

    # Memory streams
    sdfg.add_stream("a_stream",
                    dtype,
                    veclen=veclen,
                    storage=dace.StorageType.FPGA_Local,
                    transient=True)
    sdfg.add_stream("b_stream",
                    dtype,
                    veclen=veclen,
                    storage=dace.StorageType.FPGA_Local,
                    transient=True)
    produce_input_stream = state.add_write("a_stream")
    consume_input_stream = state.add_read("a_stream")
    produce_output_stream = state.add_write("b_stream")
    consume_output_stream = state.add_write("b_stream")

    tasklet = state.add_tasklet(name, {"_in"}, {"_out"}, tasklet_code)

    # Iterative map
    entry, exit = state.add_map(name, {
        "i": "0:N/{}".format(veclen),
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    # Unrolled map
    unroll_entry, unroll_exit = state.add_map(
        name + "_unroll", {"u": "0:{}".format(veclen)},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    # Container-to-container copies between arrays and streams
    state.add_memlet_path(read_memory,
                          produce_input_stream,
                          memlet=dace.Memlet.simple(produce_input_stream.data,
                                                    "0",
                                                    other_subset_str="0:N",
                                                    num_accesses=n / veclen,
                                                    veclen=veclen))
    state.add_memlet_path(consume_output_stream,
                          write_memory,
                          memlet=dace.Memlet.simple(write_memory.data,
                                                    "0:N",
                                                    other_subset_str="0",
                                                    num_accesses=n / veclen,
                                                    veclen=veclen))

    # Container-to-container copy from vectorized stream to non-vectorized
    # buffer
    sdfg.add_array("a_buffer", (veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    sdfg.add_array("b_buffer", (veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    a_buffer = state.add_access("a_buffer")
    b_buffer = state.add_access("b_buffer")

    # Input stream to buffer
    state.add_memlet_path(consume_input_stream,
                          entry,
                          a_buffer,
                          memlet=dace.Memlet.simple(a_buffer.data,
                                                    "0:{}".format(veclen),
                                                    other_subset_str="0",
                                                    veclen=veclen,
                                                    num_accesses=1))
    # Buffer to tasklet
    state.add_memlet_path(a_buffer,
                          unroll_entry,
                          tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet.simple(a_buffer.data,
                                                    "u",
                                                    veclen=1,
                                                    num_accesses=1))

    # Tasklet to buffer
    state.add_memlet_path(tasklet,
                          unroll_exit,
                          b_buffer,
                          src_conn="_out",
                          memlet=dace.Memlet.simple(b_buffer.data,
                                                    "u",
                                                    veclen=1,
                                                    num_accesses=1))

    # Buffer to output stream
    state.add_memlet_path(b_buffer,
                          exit,
                          produce_output_stream,
                          memlet=dace.Memlet.simple(
                              produce_output_stream.data,
                              "0",
                              other_subset_str="0:{}".format(veclen),
                              veclen=veclen,
                              num_accesses=1))

    return sdfg


if __name__ == "__main__":

    sdfg = make_sdfg()
    sdfg.save("test.sdfg")
