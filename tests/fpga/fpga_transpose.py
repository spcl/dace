# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace


# Builds an SDFG that implements transposition of a 2D array on the FPGA
def make_sdfg(name="transpose"):

    n = dace.symbol("N")
    m = dace.symbol("M")

    sdfg = dace.SDFG(name)

    pre_state = sdfg.add_state(name + "_pre")
    state = sdfg.add_state(name)
    post_state = sdfg.add_state(name + "_post")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    _, desc_input_host = sdfg.add_array("a_input", (n, m), dace.float64)
    _, desc_output_host = sdfg.add_array("a_output", (m, n), dace.float64)
    desc_input_device = copy.copy(desc_input_host)
    desc_input_device.storage = dace.StorageType.FPGA_Global
    desc_input_device.location["bank"] = 0
    desc_input_device.transient = True
    desc_output_device = copy.copy(desc_output_host)
    desc_output_device.storage = dace.StorageType.FPGA_Global
    desc_output_device.location["bank"] = 1
    desc_output_device.transient = True
    sdfg.add_datadesc("a_input_device", desc_input_device)
    sdfg.add_datadesc("a_output_device", desc_output_device)

    # Host to device
    pre_read = pre_state.add_read("a_input")
    pre_write = pre_state.add_write("a_input_device")
    pre_state.add_memlet_path(pre_read,
                              pre_write,
                              memlet=dace.Memlet.simple(pre_write, "0:N, 0:M"))

    # Device to host
    post_read = post_state.add_read("a_output_device")
    post_write = post_state.add_write("a_output")
    post_state.add_memlet_path(post_read,
                               post_write,
                               memlet=dace.Memlet.simple(
                                   post_write, "0:N, 0:M"))

    # Compute state
    read = state.add_read("a_input_device")
    write = state.add_write("a_output_device")

    # Trivial tasklet
    tasklet = state.add_tasklet(name, {"_in"}, {"_out"}, "_out = _in")

    entry, exit = state.add_map(name, {
        "i": "0:N",
        "j": "0:M",
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    state.add_memlet_path(read,
                          entry,
                          tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet.simple("a_input_device",
                                                    "i, j",
                                                    num_accesses=1))
    state.add_memlet_path(tasklet,
                          exit,
                          write,
                          src_conn="_out",
                          memlet=dace.Memlet.simple("a_output_device",
                                                    "j, i",
                                                    num_accesses=1))

    return sdfg
