#!/usr/bin/env python3
import copy
import dace


def make_sdfg(name="fpga_stencil_test",
              dtype=dace.float32,
              veclen=8):

    n = dace.symbol("N")
    m = dace.symbol("M")

    sdfg = dace.SDFG(name)

    pre_state = sdfg.add_state(name + "_pre")
    state = sdfg.add_state(name)
    post_state = sdfg.add_state(name + "_post")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    _, desc_input_host = sdfg.add_array("a", (n, m), dtype)
    _, desc_output_host = sdfg.add_array("b", (n, m), dtype)
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
                              memlet=dace.Memlet.simple(pre_write,
                                                        "0:N, 0:M",
                                                        veclen=veclen))

    # Device to host
    post_read = post_state.add_read("b_device")
    post_write = post_state.add_write("b")
    post_state.add_memlet_path(post_read,
                               post_write,
                               memlet=dace.Memlet.simple(post_write,
                                                         "0:N, 0:M",
                                                         veclen=veclen))

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

    tasklet = state.add_tasklet(name, {"_north", "_west", "_east", "_south"},
                                {"result"}, """\
north = _north if i >= 1 else 1
west = _west if {W}*j + u >= 1 else 1
east = _east if {W}*j + u < M - 1 else 1
south = _south if i < N - 1 else 1

result = 0.25 * (north + west + east + south)""".format(W=veclen))

    entry, exit = state.add_pipeline(name, {
        "i": "0:N",
        "j": "0:M/{}".format(veclen),
    },
                                     schedule=dace.ScheduleType.FPGA_Device,
                                     init_size=m / veclen,
                                     init_overlap=False,
                                     drain_size=m / veclen,
                                     drain_overlap=True)

    # Unrolled map
    unroll_entry, unroll_exit = state.add_map(
        name + "_unroll", {"u": "0:{}".format(veclen)},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    # Container-to-container copies between arrays and streams
    state.add_memlet_path(read_memory,
                          produce_input_stream,
                          memlet=dace.Memlet.simple(read_memory.data,
                                                    "0:N, 0:M",
                                                    other_subset_str="0",
                                                    num_accesses=n *
                                                    (m / veclen),
                                                    veclen=veclen))
    state.add_memlet_path(consume_output_stream,
                          write_memory,
                          memlet=dace.Memlet.simple(write_memory.data,
                                                    "0:N, 0:M",
                                                    other_subset_str="0",
                                                    num_accesses=n *
                                                    (m / veclen),
                                                    veclen=veclen))

    # Container-to-container copy from vectorized stream to non-vectorized
    # buffer
    sdfg.add_array("input_buffer", (veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    sdfg.add_array("shift_register", (2 * m + veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_ShiftRegister,
                   transient=True)
    sdfg.add_array("output_buffer", (veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    sdfg.add_array("output_buffer_packed", (veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    input_buffer = state.add_access("input_buffer")
    shift_register = state.add_access("shift_register")
    output_buffer = state.add_access("output_buffer")
    output_buffer_packed = state.add_access("output_buffer_packed")

    # Only write if not initializing
    read_tasklet = state.add_tasklet(
        name + "_conditional_read", {"_in"}, {"_out"},
        "if not {}:\n\t_out = _in".format(entry.pipeline.drain_condition()))

    # Input stream to buffer
    state.add_memlet_path(consume_input_stream,
                          entry,
                          read_tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet.simple(
                              consume_input_stream.data,
                              "0",
                              veclen=veclen,
                              num_accesses=-1))
    state.add_memlet_path(read_tasklet,
                          input_buffer,
                          src_conn="_out",
                          memlet=dace.Memlet.simple(
                              input_buffer.data,
                              "0:{}".format(veclen),
                              veclen=veclen,
                              num_accesses=1))
    state.add_memlet_path(input_buffer,
                          shift_register,
                          memlet=dace.Memlet.simple(
                              shift_register.data,
                              "2*M*{W}: (2*M + 1)*{W}".format(W=veclen),
                              other_subset_str="0:{}".format(veclen),
                              veclen=veclen,
                              num_accesses=1))

    # Stencils accesses
    state.add_memlet_path(shift_register,
                          unroll_entry,
                          tasklet,
                          dst_conn="_north",
                          memlet=dace.Memlet.simple(shift_register.data,
                                                    "u",  # North
                                                    veclen=1,
                                                    num_accesses=1))
    state.add_memlet_path(shift_register,
                          unroll_entry,
                          tasklet,
                          dst_conn="_west",
                          memlet=dace.Memlet.simple(shift_register.data,
                                                    "u + M - 1",  # West
                                                    veclen=1,
                                                    num_accesses=1))
    state.add_memlet_path(shift_register,
                          unroll_entry,
                          tasklet,
                          dst_conn="_east",
                          memlet=dace.Memlet.simple(shift_register.data,
                                                    "u + M + 1",  # East
                                                    veclen=1,
                                                    num_accesses=1))
    state.add_memlet_path(shift_register,
                          unroll_entry,
                          tasklet,
                          dst_conn="_south",
                          memlet=dace.Memlet.simple(shift_register.data,
                                                    "u + 2 * M",  # South
                                                    veclen=1,
                                                    num_accesses=1))

    # Tasklet to buffer
    state.add_memlet_path(tasklet,
                          unroll_exit,
                          output_buffer,
                          src_conn="result",
                          memlet=dace.Memlet.simple(output_buffer.data,
                                                    "u",
                                                    veclen=1,
                                                    num_accesses=1))

    # Pack buffer
    state.add_memlet_path(output_buffer,
                          output_buffer_packed,
                          memlet=dace.Memlet.simple(output_buffer.data,
                                                    "0:{}".format(veclen),
                                                    veclen=veclen,
                                                    num_accesses=1))

    # Only write if not initializing
    write_tasklet = state.add_tasklet(
        name + "_conditional_write", {"_in"}, {"_out"},
        "if not {}:\n\t_out = _in".format(entry.pipeline.init_condition()))

    # Buffer to output stream
    state.add_memlet_path(output_buffer_packed,
                          write_tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet.simple(
                              output_buffer_packed.data,
                              "0:{}".format(veclen),
                              veclen=veclen,
                              num_accesses=1))

    # Buffer to output stream
    state.add_memlet_path(write_tasklet,
                          exit,
                          produce_output_stream,
                          src_conn="_out",
                          memlet=dace.Memlet.simple(
                              produce_output_stream.data,
                              "0",
                              veclen=veclen,
                              num_accesses=-1))

    return sdfg


if __name__ == "__main__":

    import numpy as np

    dtype = np.float32
    n = 16
    m = 16

    jacobi = make_sdfg(dtype=dtype)
    jacobi.specialize({"N": n, "M": m})

    a = np.arange(n*m, dtype=dtype).reshape((n, m))
    b = np.empty((n, m), dtype=dtype)

    jacobi(a=a, b=b)

    padded = np.ones((n + 2, m + 2), dtype)
    padded[1:-1, 1:-1] = a
    ref = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                  padded[1:-1, :-2] + padded[1:-1, 2:])

    if (b != ref).any():
        raise ValueError("Unexpected output.")
