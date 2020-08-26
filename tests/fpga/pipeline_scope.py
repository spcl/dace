#!/usr/bin/env python3
import copy
import dace


def make_sdfg(dtype, name="pipeline_test"):

    n = dace.symbol("N")
    k = dace.symbol("K")
    m = dace.symbol("M")

    sdfg = dace.SDFG(name)

    pre_state = sdfg.add_state(name + "_pre")
    state = sdfg.add_state(name)
    post_state = sdfg.add_state(name + "_post")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    _, desc_input_host = sdfg.add_array("a", (n, k, m), dtype)
    _, desc_output_host = sdfg.add_array("b", (n, k, m), dtype)
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
                                                        "0:N, 0:K, 0:M"))

    # Device to host
    post_read = post_state.add_read("b_device")
    post_write = post_state.add_write("b")
    post_state.add_memlet_path(post_read,
                               post_write,
                               memlet=dace.Memlet.simple(post_write,
                                                         "0:N, 0:K, 0:M"))

    # Compute state
    read_memory = state.add_read("a_device")
    write_memory = state.add_write("b_device")

    # Memory streams
    sdfg.add_stream("a_stream",
                    dtype,
                    storage=dace.StorageType.FPGA_Local,
                    transient=True)
    sdfg.add_stream("b_stream",
                    dtype,
                    storage=dace.StorageType.FPGA_Local,
                    transient=True)
    produce_input_stream = state.add_write("a_stream")
    consume_input_stream = state.add_read("a_stream")
    produce_output_stream = state.add_write("b_stream")
    consume_output_stream = state.add_write("b_stream")

    entry, exit = state.add_pipeline(name, {
        "n": "0:N",
        "k": "0:K",
        "m": "0:M",
    },
                                     schedule=dace.ScheduleType.FPGA_Device,
                                     init_size=k * m,
                                     init_overlap=True,
                                     drain_size=k * m,
                                     drain_overlap=True)

    tasklet = state.add_tasklet(
        name, {"_in"}, {"_out"},
        """_out = _in + (1 if {} else (3 if {} else 2))""".format(
            entry.pipeline.init_condition(), entry.pipeline.drain_condition()))

    # Container-to-container copies between arrays and streams
    state.add_memlet_path(read_memory,
                          produce_input_stream,
                          memlet=dace.Memlet.simple(read_memory.data,
                                                    "0:N, 0:K, 0:M",
                                                    other_subset_str="0",
                                                    num_accesses=n * k * m))
    state.add_memlet_path(consume_output_stream,
                          write_memory,
                          memlet=dace.Memlet.simple(write_memory.data,
                                                    "0:N, 0:K, 0:M",
                                                    other_subset_str="0",
                                                    num_accesses=n * k * m))

    # Input stream to buffer
    state.add_memlet_path(consume_input_stream,
                          entry,
                          tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet.simple(
                              consume_input_stream.data,
                              "0",
                              num_accesses=-1))

    # Buffer to output stream
    state.add_memlet_path(tasklet,
                          exit,
                          produce_output_stream,
                          src_conn="_out",
                          memlet=dace.Memlet.simple(
                              produce_output_stream.data,
                              "0",
                              num_accesses=-1))

    return sdfg


if __name__ == "__main__":

    import numpy as np

    dtype = np.float64
    n = 16
    k = 24
    m = 32

    jacobi = make_sdfg(dtype=dtype)
    jacobi.specialize({"N": n, "K": k, "M": m})

    a = np.arange(n*k*m, dtype=dtype).reshape((n, k, m))
    b = np.empty((n, k, m), dtype=dtype)

    jacobi(a=a, b=b)

    ref = copy.copy(a)
    ref[0, :, :] += 1
    ref[1:-1, :, :] += 2
    ref[-1, :, :] += 3

    if (b != ref).any():
        print(b)
        print(ref)
        raise ValueError("Unexpected output.")
