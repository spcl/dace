# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
from dace.fpga_testing import intel_fpga_test

SIZE = dace.symbol("N")
VECTOR_LENGTH = dace.symbol("W")
DTYPE = dace.float64


def make_copy_to_fpga_state(sdfg):

    state = sdfg.add_state("copy_to_device")

    A_host = sdfg.add_array("A", [SIZE // VECTOR_LENGTH.get()], dtype=dace.vector(DTYPE, VECTOR_LENGTH.get()))

    A_device = sdfg.add_array("A_device", [SIZE],
                              dtype=dace.vector(DTYPE, VECTOR_LENGTH.get()),
                              transient=True,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    read = state.add_read("A")
    write = state.add_write("A_device")

    state.add_memlet_path(read,
                          write,
                          memlet=dace.memlet.Memlet.simple("A_device",
                                                           "0:N//{}".format(VECTOR_LENGTH.get()),
                                                           num_accesses=SIZE // VECTOR_LENGTH.get()))

    return state


def make_copy_to_host_state(sdfg):

    state = sdfg.add_state("copy_to_host")

    B_device = sdfg.add_array("B_device", [SIZE],
                              dtype=dace.vector(DTYPE, VECTOR_LENGTH.get()),
                              transient=True,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    B_host = sdfg.add_array("B", [SIZE // VECTOR_LENGTH.get()], dtype=dace.vector(DTYPE, VECTOR_LENGTH.get()))

    read = state.add_read("B_device")
    write = state.add_write("B")

    state.add_memlet_path(read,
                          write,
                          memlet=dace.memlet.Memlet.simple("B",
                                                           "0:N//{}".format(VECTOR_LENGTH.get()),
                                                           num_accesses=SIZE // VECTOR_LENGTH.get()))

    return state


def make_fpga_state(sdfg, vectorize_connector):

    state = sdfg.add_state("fpga_state")

    sdfg.add_array("input_buffer", (VECTOR_LENGTH.get(), ),
                   DTYPE,
                   transient=True,
                   storage=dace.StorageType.FPGA_Registers)
    sdfg.add_array("output_buffer", (VECTOR_LENGTH.get(), ),
                   DTYPE,
                   transient=True,
                   storage=dace.StorageType.FPGA_Registers)

    read_input = state.add_read("A_device")
    read_buffer = state.add_access("input_buffer")
    write_buffer = state.add_access("output_buffer")
    write_output = state.add_write("B_device")

    outer_entry, outer_exit = state.add_map("outer_map", {"i": "0:N/W"}, schedule=dace.ScheduleType.FPGA_Device)

    # Test read from packed memory to an unpacked buffer
    if vectorize_connector:
        outputs = {"a_unpacked": dace.vector(DTYPE, VECTOR_LENGTH.get())}
    else:
        outputs = {"a_unpacked"}  # Infers an array
    unpack_tasklet = state.add_tasklet("unpack_tasklet", {"a"}, outputs, "a_unpacked = a")
    state.add_memlet_path(read_input,
                          outer_entry,
                          unpack_tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet.simple("A_device", "i"))
    state.add_memlet_path(unpack_tasklet,
                          read_buffer,
                          src_conn="a_unpacked",
                          memlet=dace.Memlet.simple(read_buffer.data, "0:{}".format(VECTOR_LENGTH.get())))

    unroll_entry, unroll_exit = state.add_map("shuffle_map", {"w": "0:W"},
                                              schedule=dace.ScheduleType.FPGA_Device,
                                              unroll=True)

    tasklet = state.add_tasklet("shuffle_tasklet", {"a"}, {"b"}, "b = a")

    state.add_memlet_path(read_buffer,
                          unroll_entry,
                          tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet.simple("input_buffer", "(w + W // 2) % W", num_accesses=1))

    state.add_memlet_path(tasklet,
                          unroll_exit,
                          write_buffer,
                          src_conn="b",
                          memlet=dace.Memlet.simple("output_buffer", "w", num_accesses=1))

    # Test writing from unpacked to packed from inside tasklet
    if vectorize_connector:
        outputs = {"b": dace.vector(DTYPE, VECTOR_LENGTH.get())}
    else:
        outputs = {"b"}
    pack_tasklet = state.add_tasklet("pack_tasklet", outputs, {"b_packed"}, "b_packed = b")
    state.add_memlet_path(write_buffer,
                          pack_tasklet,
                          dst_conn="b",
                          memlet=dace.Memlet.simple(write_buffer.data, "0:{}".format(VECTOR_LENGTH.get())))

    # Write back out to memory from unpacked to packed memory
    state.add_memlet_path(pack_tasklet,
                          outer_exit,
                          write_output,
                          src_conn="b_packed",
                          memlet=dace.Memlet.simple("B_device", "i"))

    return state


def make_sdfg(name=None, vectorize_connector=False):

    if name is None:
        name = "veclen_conversion"

    sdfg = dace.SDFG(name)

    pre_state = make_copy_to_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)
    compute_state = make_fpga_state(sdfg, vectorize_connector)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


@intel_fpga_test()
def test_veclen_conversion():

    size = 128
    vector_length = 4

    SIZE.set(size)
    VECTOR_LENGTH.set(vector_length)

    if size % vector_length != 0:
        raise ValueError("Size {} must be divisible by vector length {}.".format(size, vector_length))

    sdfg = make_sdfg(vectorize_connector=False)
    sdfg.specialize({"W": vector_length})

    A = np.arange(size, dtype=np.float64)
    B = np.zeros((size, ), dtype=np.float64)

    sdfg(A=A, B=B, N=SIZE)

    mid = vector_length // 2

    for i in range(size // vector_length):
        expected = np.concatenate(
            (A[i * vector_length + mid:(i + 1) * vector_length], A[i * vector_length:i * vector_length + mid]))
        if any(B[i * vector_length:(i + 1) * vector_length] != expected):
            raise ValueError("Shuffle failed: {} (should be {})".format(B, expected))

    return sdfg


if __name__ == "__main__":
    test_veclen_conversion(None)
