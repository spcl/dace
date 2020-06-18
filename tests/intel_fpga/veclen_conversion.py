#!/usr/bin/env python3
import argparse
import dace
import numpy as np

SIZE = dace.symbol("N")
VECTOR_LENGTH = dace.symbol("W")
DTYPE = np.float64


def make_copy_to_fpga_state(sdfg):

    state = sdfg.add_state("copy_to_device")

    A_host = sdfg.add_array("A", [SIZE], dtype=DTYPE)

    A_device = sdfg.add_array("A_device", [SIZE],
                              dtype=DTYPE,
                              transient=True,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    read = state.add_read("A")
    write = state.add_write("A_device")

    state.add_memlet_path(read,
                          write,
                          memlet=dace.memlet.Memlet.simple(
                              "A_device",
                              "0:N",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=SIZE))

    return state


def make_copy_to_host_state(sdfg):

    state = sdfg.add_state("copy_to_host")

    B_device = sdfg.add_array("B_device", [SIZE],
                              dtype=DTYPE,
                              transient=True,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    B_host = sdfg.add_array("B", [SIZE], dtype=DTYPE)

    read = state.add_read("B_device")
    write = state.add_write("B")

    state.add_memlet_path(read,
                          write,
                          memlet=dace.memlet.Memlet.simple(
                              "B",
                              "0:N",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=SIZE))

    return state


def make_fpga_state(sdfg):

    state = sdfg.add_state("fpga_state")

    input_buffer = sdfg.add_array("input_buffer", (VECTOR_LENGTH, ),
                                  DTYPE,
                                  transient=True,
                                  storage=dace.StorageType.FPGA_Registers)
    output_buffer = sdfg.add_array("output_buffer", (VECTOR_LENGTH, ),
                                   DTYPE,
                                   transient=True,
                                   storage=dace.StorageType.FPGA_Registers)

    read_input = state.add_read("A_device")
    read_buffer = state.add_access("input_buffer")
    write_buffer = state.add_access("output_buffer")
    write_output = state.add_write("B_device")

    outer_entry, outer_exit = state.add_map(
        "outer_map", {"i": "0:N/W"}, schedule=dace.ScheduleType.FPGA_Device)

    # Test read from packed memory to an unpacked buffer
    unpack_tasklet = state.add_tasklet("unpack_tasklet", {"a"}, {"a_unpacked"},
                                       "a_unpacked = a")
    state.add_memlet_path(read_input,
                          outer_entry,
                          unpack_tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet.simple(
                              "A_device",
                              "i*W",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=VECTOR_LENGTH))
    state.add_memlet_path(unpack_tasklet,
                          read_buffer,
                          src_conn="a_unpacked",
                          memlet=dace.Memlet.simple(
                              "input_buffer",
                              "0",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=VECTOR_LENGTH))

    unroll_entry, unroll_exit = state.add_map(
        "shuffle_map", {"w": "0:W"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    tasklet = state.add_tasklet("shuffle_tasklet", {"a"}, {"b"}, "b = a")

    state.add_memlet_path(read_buffer,
                          unroll_entry,
                          tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet.simple("input_buffer",
                                                    "(w + W // 2) % W",
                                                    veclen=1,
                                                    num_accesses=1))

    state.add_memlet_path(tasklet,
                          unroll_exit,
                          write_buffer,
                          src_conn="b",
                          memlet=dace.Memlet.simple("output_buffer",
                                                    "w",
                                                    veclen=1,
                                                    num_accesses=1))

    # Test writing from unpacked to packed from inside tasklet
    pack_tasklet = state.add_tasklet("pack_tasklet", {"b"}, {"b_packed"},
                                     "b_packed = b")
    state.add_memlet_path(write_buffer,
                          pack_tasklet,
                          dst_conn="b",
                          memlet=dace.Memlet.simple(
                              write_buffer,
                              "0",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=VECTOR_LENGTH))

    # Write back out to memory from unpacked to packed memory
    state.add_memlet_path(pack_tasklet,
                          outer_exit,
                          write_output,
                          src_conn="b_packed",
                          memlet=dace.Memlet.simple(
                              "B_device",
                              "W*i",
                              veclen=VECTOR_LENGTH.get(),
                              num_accesses=VECTOR_LENGTH))

    return state


def make_sdfg(name=None):

    if name is None:
        name = "veclen_conversion"

    sdfg = dace.SDFG(name)

    pre_state = make_copy_to_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)
    compute_state = make_fpga_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-size", type=int, default=128)
    parser.add_argument("-vector_length", type=int, default=4)
    args = parser.parse_args()

    SIZE.set(args.size)
    VECTOR_LENGTH.set(args.vector_length)

    if args.size % args.vector_length != 0:
        raise ValueError(
            "Size {} must be divisible by vector length {}.".format(
                args.size, args.vector_length))

    sdfg = make_sdfg()
    sdfg.specialize({"W": args.vector_length})

    # Initialize arrays: Randomize A and B, zero C
    A = np.arange(args.size, dtype=np.float64)
    B = np.zeros((args.size, ), dtype=np.float64)

    sdfg(A=A, B=B, N=SIZE)

    mid = args.vector_length // 2

    for i in range(args.size // args.vector_length):
        expected = np.concatenate(
            (A[i * args.vector_length + mid:(i + 1) * args.vector_length],
             A[i * args.vector_length:i * args.vector_length + mid]))
        if any(B[i * args.vector_length:(i + 1) *
                 args.vector_length] != expected):
            raise ValueError("Shuffle failed: {} (should be {})".format(
                B, expected))
