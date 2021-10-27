# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Computes C = A @ B + C.

This implementation is based on the HLS implementation from:
    https://github.com/spcl/gemm_hls
It uses the compute and I/O optimal strategy described in the FPGA'20 paper:
    "Flexible Communication Avoiding Matrix Multiplication on FPGA with
     High-Level Synthesis".
"""

import click
import copy
import dace
import numpy as np
from dace.libraries.standard import Gearbox
from dace.transformation.interstate import InlineSDFG

MINIMUM_CHANNEL_DEPTH = 8

# Symbols used in this implementation:
#
#   N:  Number of rows of A and C.
#   K:  Number of columns of A and rows of B.
#   M:  Number of columns of B and C.
#   TN: The tile size in N, which must divide the size in N.
#   TM: The tile size in M, which must divide the size in M.
#   P:  The number of (vertically unrolled) processing elements, and
#       consequently one of the two degrees of parallelism in the kernel. Must
#       divide the tile size TN.
#   W:  The vectorization width, being the other degree of parallelism. Must
#       divide the tile size TM.


def make_copy_to_fpga_state(sdfg, vtype):
    """
    Creates the pre-state where the matrices are transferred to the FPGA.
    """

    state = sdfg.add_state("copy_to_device")
    dtype = vtype.base_type
    # mem_veclen is the vectorization width necessary to create a 512-bit
    # interface to memory, and mtype is the corresponding type.
    mem_veclen = 64 // dtype.bytes
    mtype = dace.vector(dtype, mem_veclen)

    # Host data has plain data types
    sdfg.add_array("A", ["N", "K"], dtype=dtype)
    sdfg.add_array("B", ["K", "M"], dtype=dtype)
    sdfg.add_array("C", ["N", "M"], dtype=dtype)
    A_host = state.add_read("A")
    B_host = state.add_read("B")
    C_host = state.add_read("C")

    # On the device, vector B and C will be vectorized along rows. A is read
    # column-wise, so it is not vectorized.
    sdfg.add_array("A_device", ["N", f"K//{mem_veclen}"],
                   dtype=mtype,
                   transient=True,
                   location={
                       "memorytype": "DDR",
                       "bank": 1
                   },
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_device", ["K", f"M//{mem_veclen}"],
                   dtype=mtype,
                   transient=True,
                   location={
                       "memorytype": "DDR",
                       "bank": 1
                   },
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("C_device", ["N", f"M//{mem_veclen}"],
                   dtype=mtype,
                   transient=True,
                   location={
                       "memorytype": "DDR",
                       "bank": 1
                   },
                   storage=dace.dtypes.StorageType.FPGA_Global)
    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_memlet_path(
        A_host,
        A_device,
        memlet=dace.Memlet(f"A_device[0:N, 0:K//{mem_veclen}]"))
    state.add_memlet_path(
        B_host,
        B_device,
        memlet=dace.Memlet(f"B_device[0:K, 0:M//{mem_veclen}]"))
    state.add_memlet_path(
        C_host,
        C_device,
        memlet=dace.Memlet(f"C_device[0:N, 0:M//{mem_veclen}]"))

    return state


def make_copy_to_host_state(sdfg, vtype):
    """
    Creates the post-state where C is copied back to the host.
    """

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_memlet_path(C_device, C_host, memlet=dace.Memlet("C[0:N, 0:M]"))

    return state


def make_read_A(sdfg, state, vtype):
    """
    Creates the memory read from A, which performs in-memory transposition by
    reading 512-bit wide vectors of A, then piping them into separate streams
    that are popped one at a time and sent to the kernel.
    """

    # Deduce types
    dtype = vtype.base_type
    mem_veclen = 64 // dtype.bytes

    # Unpack vector into a register
    sdfg.add_array("transpose_reg", (mem_veclen, ),
                   dtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)

    # Add a stream for each element in the vector
    sdfg.add_stream(
        "transpose",
        dtype,
        # Allow loading the next column while the previous is being
        # used
        buffer_size="2 * TN",
        shape=(mem_veclen, ),
        storage=dace.StorageType.FPGA_Local,
        transient=True)

    # Read each element into a buffer to unpack the vector into individual
    # elements
    mem = state.add_read("A_device")
    entry, exit = state.add_map("read_A", {
        "n0": "0:N//TN",
        "m": "0:M//TM",
        "k0": f"0:K//{mem_veclen}",
        "n1": "0:TN",
    },
                                schedule=dace.ScheduleType.FPGA_Device)
    buffer_access = state.add_access("transpose_reg")
    state.add_memlet_path(mem,
                          entry,
                          buffer_access,
                          memlet=dace.Memlet("A_device[n0 * TN + n1, k0]"))

    # Now stick each element into a separate stream
    unroll_entry, unroll_exit = state.add_map(
        "unpack_A", {"k1": f"0:{mem_veclen}"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)
    unroll_tasklet = state.add_tasklet("unpack_A", {"from_memory"}, {"to_pipe"},
                                       "to_pipe = from_memory")
    unroll_write = state.add_write("transpose")
    state.add_memlet_path(buffer_access,
                          unroll_entry,
                          unroll_tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet(f"transpose_reg[k1]"))
    state.add_memlet_path(unroll_tasklet,
                          unroll_exit,
                          exit,
                          unroll_write,
                          src_conn="to_pipe",
                          memlet=dace.Memlet(f"transpose[k1]"))

    # A separate processing element will pop from the streams one at a time
    transpose_read = state.add_read("transpose")
    transpose_entry, transpose_exit = state.add_map(
        "transpose_A", {
            "n0": "0:N//TN",
            "m": "0:M//TM",
            "k0": f"0:K//{mem_veclen}",
            "k1": f"0:{mem_veclen}",
            "n1": "0:TN",
        },
        schedule=dace.ScheduleType.FPGA_Device)
    pipe_out = state.add_write("A_pipe")
    tasklet = state.add_tasklet("transpose_A", {"from_transpose"},
                                {"to_kernel"}, "to_kernel = from_transpose")
    state.add_memlet_path(transpose_read,
                          transpose_entry,
                          tasklet,
                          dst_conn="from_transpose",
                          memlet=dace.Memlet(f"transpose[k1]"))
    state.add_memlet_path(tasklet,
                          transpose_exit,
                          pipe_out,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("A_pipe[0]"))


def make_read_B(sdfg, state, vtype):

    # Deduce types
    dtype = vtype.base_type
    mem_veclen = 64 // dtype.bytes
    mtype = dace.vector(dtype, mem_veclen)

    entry, exit = state.add_map("read_B", {
        "n0": "0:N//TN",
        "m0": "0:M//TM",
        "k": "0:K",
        "m1": f"0:TM//{mem_veclen}"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("B_device")
    to_feeder = state.add_write("B_to_feeder")
    tasklet = state.add_tasklet("read_B", {"from_memory"}, {"to_feeder"},
                                "to_feeder = from_memory")
    state.add_memlet_path(
        mem,
        entry,
        tasklet,
        dst_conn="from_memory",
        memlet=dace.Memlet(f"B_device[k, m0 * (TM//{mem_veclen}) + m1]"))

    if mem_veclen > vtype.veclen:

        # Data arrives as 512-bit wide vectors, and will be converted to the
        # vector length of the kernel

        sdfg.add_stream("B_to_converter",
                        dtype=mtype,
                        buffer_size=MINIMUM_CHANNEL_DEPTH,
                        storage=dace.StorageType.FPGA_Local,
                        transient=True)
        to_converter_write = state.add_write("B_to_converter")
        state.add_memlet_path(tasklet,
                              exit,
                              to_converter_write,
                              src_conn="to_feeder",
                              memlet=dace.Memlet("B_to_converter[0]"))

        # Convert 512-bit vectors to whatever width the kernel uses
        to_converter_read = state.add_read("B_to_converter")
        gearbox = Gearbox(f"(N//TN) * (M//TM) * K * (TM//{mem_veclen})",
                          "convert_B", dace.ScheduleType.FPGA_Device)
        state.add_memlet_path(to_converter_read,
                              gearbox,
                              dst_conn="from_memory",
                              memlet=dace.Memlet(f"B_to_converter[0]",
                                                 dynamic=True))
        state.add_memlet_path(gearbox,
                              to_feeder,
                              src_conn="to_feeder",
                              memlet=dace.Memlet("B_to_feeder[0]",
                                                 dynamic=True))

    else:

        # If the kernel uses the full memory width, just send the data directly
        # without any conversion
        state.add_memlet_path(tasklet,
                              exit,
                              to_feeder,
                              src_conn="to_feeder",
                              memlet=dace.Memlet(f"B_to_feeder[0]"))


def make_feed_B(sdfg, state, vtype):
    """
    This module will buffer the values read from the B matrix, sending them
    multiple times to the kernel for each row of A in the current outer product.
    """

    entry, exit = state.add_map("feed_B", {
        "n0": "0:N//TN",
        "m0": "0:M//TM",
        "k": "0:K",
        "n1": "0:TN//P",
        "m1": "0:TM//W"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    sdfg.add_array("feed_B_buffer", ("TM//W", ),
                   vtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)
    buffer_read = state.add_read("feed_B_buffer")
    buffer_write = state.add_write("feed_B_buffer")

    read = state.add_read("B_to_feeder")
    write = state.add_write("B_pipe")
    tasklet = state.add_tasklet(
        "feed_B", {"from_memory", "buffer_in"}, {"to_kernel", "buffer_out"}, """
val = buffer_in
if n1 == 0:
    val = from_memory
to_kernel = val
buffer_out = val""")

    state.add_memlet_path(read,
                          entry,
                          tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet("B_to_feeder[0]", dynamic=True))
    state.add_memlet_path(buffer_read,
                          entry,
                          tasklet,
                          dst_conn="buffer_in",
                          memlet=dace.Memlet("feed_B_buffer[m1]"))
    state.add_memlet_path(tasklet,
                          exit,
                          buffer_write,
                          src_conn="buffer_out",
                          memlet=dace.Memlet("feed_B_buffer[m1]"))
    state.add_memlet_path(tasklet,
                          exit,
                          write,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("B_pipe[0]"))


def make_write_C(sdfg, state, vtype):

    # Deduce types
    dtype = vtype.base_type
    mem_veclen = 64 // dtype.bytes
    mtype = dace.vector(dtype, mem_veclen)

    from_kernel = state.add_read("C_pipe")
    mem_read = state.add_read("C_device")
    mem_write = state.add_write("C_device")

    if mem_veclen > vtype.veclen:

        # We need to convert from the kernel vectorization length to 512-bit
        # vectors that are written back to memory

        gearbox = Gearbox(f"(N//TN) * (M//TM) * TN * (TM//{mem_veclen})",
                          "convert_C",
                          schedule=dace.ScheduleType.FPGA_Device)
        sdfg.add_stream("C_from_converter",
                        mtype,
                        buffer_size=f"TM//{mem_veclen}",
                        storage=dace.StorageType.FPGA_Local,
                        transient=True)
        converter_write = state.add_write("C_from_converter")
        state.add_memlet_path(from_kernel,
                              gearbox,
                              dst_conn="from_kernel",
                              memlet=dace.Memlet(f"C_pipe[0]", dynamic=True))
        state.add_memlet_path(gearbox,
                              converter_write,
                              src_conn="to_memory",
                              memlet=dace.Memlet("C_from_converter[0]",
                                                 dynamic=True))

        to_writer = state.add_read("C_from_converter")
        to_writer_subset = "C_from_converter[0]"

    else:

        # Just send the data directly to the reader
        to_writer = from_kernel
        to_writer_subset = "C_pipe[0]"

    entry, exit = state.add_map("write_C", {
        "n0": "0:N//TN",
        "m0": "0:M//TM",
        "n1": "0:TN",
        "m1": f"0:TM//{mem_veclen}"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    tasklet = state.add_tasklet("write_C", {"from_kernel", "prev"},
                                {"to_memory"}, "to_memory = from_kernel + prev")
    state.add_memlet_path(to_writer,
                          entry,
                          tasklet,
                          dst_conn="from_kernel",
                          memlet=dace.Memlet(to_writer_subset))

    state.add_memlet_path(
        mem_read,
        entry,
        tasklet,
        dst_conn="prev",
        memlet=dace.Memlet(
            f"C_device[n0 * TN + n1, m0 * (TM//{mem_veclen}) + m1]"))

    state.add_memlet_path(
        tasklet,
        exit,
        mem_write,
        src_conn="to_memory",
        memlet=dace.Memlet(
            f"C_device[n0 * TN + n1, m0 * (TM//{mem_veclen}) + m1]"))


def make_compute(sdfg, state, vtype):

    dtype = vtype.base_type

    # Pipes connecting the systolic array
    A_pipe_in = state.add_read("A_pipe")
    A_pipe_out = state.add_write("A_pipe")
    B_pipe_in = state.add_read("B_pipe")
    B_pipe_out = state.add_write("B_pipe")
    C_pipe_in = state.add_read("C_pipe")
    C_pipe_out = state.add_write("C_pipe")

    # Instantiate the buffer for A, and initialize it
    sdfg.add_array("A_buffer", ("2 * (TN//P)", ),
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Registers)
    init_A = state.add_access("A_buffer")
    init_entry, init_exit = state.add_map(
        "init", {
            "n0": "0:TN//P",
            "n1": "0:P-p"
        },
        schedule=dace.ScheduleType.FPGA_Device)
    init_tasklet = state.add_tasklet(
        "init_A", {"from_prev"}, {"to_buffer", "to_next"}, """\
val = from_prev
if n1 == 0:
    to_buffer = val
elif p < P - 1:
    to_next = val""")
    state.add_memlet_path(A_pipe_in,
                          init_entry,
                          init_tasklet,
                          dst_conn="from_prev",
                          memlet=dace.Memlet("A_pipe[p]"))
    state.add_memlet_path(init_tasklet,
                          init_exit,
                          init_A,
                          src_conn="to_buffer",
                          memlet=dace.Memlet(f"A_buffer[n0]", dynamic=True))
    state.add_memlet_path(init_tasklet,
                          init_exit,
                          A_pipe_out,
                          src_conn="to_next",
                          memlet=dace.Memlet(f"A_pipe[p + 1]", dynamic=True))

    # Now instantiate the body of the computation
    outer_entry, outer_exit = state.add_map(
        "tiles", {
            "n0": "0:N//TN",
            "m0": "0:M//TM"
        },
        schedule=dace.ScheduleType.FPGA_Device)

    # Make a dummy edge to bring the initialization buffer into scope
    state.add_memlet_path(init_A, outer_entry, memlet=dace.Memlet())

    # Loop over the reduced dimension
    k_entry, k_exit = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)

    # Loops over the tile content
    inner_entry, inner_exit = state.add_map(
        "inner", {
            "n1": "0:TN//P",
            "m1": "0:TM//W"
        },
        schedule=dace.ScheduleType.FPGA_Device)

    # Double buffering scheme of A
    update_A = state.add_access("A_buffer")
    buffer_tasklet = state.add_tasklet(
        "double_buffer_A", {"from_prev"}, {"to_buffer", "to_next"}, """\
if (n0 < (N/TN) - 1 or m0 < (M/TM) - 1 or k < K - 1) and m1 >= p and m1 < P:
  val = from_prev
  if m1 == p:
    to_buffer = val
  elif p < P - 1:
    to_next = val""")
    state.add_memlet_path(A_pipe_in,
                          outer_entry,
                          k_entry,
                          inner_entry,
                          buffer_tasklet,
                          dst_conn="from_prev",
                          memlet=dace.Memlet(f"A_pipe[p]", dynamic=True))
    state.add_memlet_path(buffer_tasklet,
                          update_A,
                          src_conn="to_buffer",
                          memlet=dace.Memlet(
                              f"A_buffer[n1 + (1 - (k % 2)) * (TN//P)]",
                              dynamic=True))
    state.add_memlet_path(buffer_tasklet,
                          inner_exit,
                          k_exit,
                          outer_exit,
                          A_pipe_out,
                          src_conn="to_next",
                          memlet=dace.Memlet(f"A_pipe[p + 1]", dynamic=True))

    # Instantiate the "big" buffer of the output, where most of our fast memory
    # will be spent
    sdfg.add_array("C_buffer", ("TN/P", "TM/W"),
                   vtype,
                   storage=dace.StorageType.FPGA_Local,
                   transient=True)

    # Now the tasklet performing the actual computation
    compute_tasklet = state.add_tasklet(
        "multiply_add", {"a_in", "b_in", "c_in"}, {"b_out", "c_out"}, """\
if p < P - 1:
    b_out = b_in
c_val = c_in
if k == 0:
  c_val = 0
c_out = c_val + a_in * b_in""")
    C_buffer_read = state.add_read("C_buffer")
    C_buffer_write = state.add_access("C_buffer")
    state.add_memlet_path(
        update_A,
        compute_tasklet,
        dst_conn="a_in",
        memlet=dace.Memlet(f"A_buffer[n1 + (k % 2) * (TN//P)]"))
    state.add_memlet_path(B_pipe_in,
                          outer_entry,
                          k_entry,
                          inner_entry,
                          compute_tasklet,
                          dst_conn="b_in",
                          memlet=dace.Memlet("B_pipe[p]"))
    state.add_memlet_path(C_buffer_read,
                          outer_entry,
                          k_entry,
                          inner_entry,
                          compute_tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("C_buffer[n1, m1]"))
    state.add_memlet_path(compute_tasklet,
                          inner_exit,
                          k_exit,
                          outer_exit,
                          B_pipe_out,
                          src_conn="b_out",
                          memlet=dace.Memlet("B_pipe[p + 1]", dynamic=True))
    state.add_memlet_path(compute_tasklet,
                          inner_exit,
                          k_exit,
                          C_buffer_write,
                          src_conn="c_out",
                          memlet=dace.Memlet("C_buffer[n1, m1]"))

    # Now we need to write C out after each tile has been processed
    write_entry, write_exit = state.add_map(
        "write_C", {"n1": "0:TN//P"}, schedule=dace.ScheduleType.FPGA_Device)

    # We need to enforce sequentiality between these loops
    write_sdfg = dace.SDFG("write_C")
    write_sdfg_node = state.add_nested_sdfg(write_sdfg, sdfg,
                                            {"buffer_in", "forward_in"},
                                            {"forward_out"})
    state.add_memlet_path(C_buffer_write,
                          write_entry,
                          write_sdfg_node,
                          dst_conn="buffer_in",
                          memlet=dace.Memlet("C_buffer[n1, 0:TM/W]"))
    state.add_memlet_path(C_pipe_in,
                          outer_entry,
                          write_entry,
                          write_sdfg_node,
                          dst_conn="forward_in",
                          memlet=dace.Memlet("C_pipe[p + 1]", dynamic=True))
    state.add_memlet_path(write_sdfg_node,
                          write_exit,
                          outer_exit,
                          C_pipe_out,
                          src_conn="forward_out",
                          memlet=dace.Memlet("C_pipe[p]", dynamic=True))
    write_sdfg.add_stream("forward_in",
                          vtype,
                          buffer_size=MINIMUM_CHANNEL_DEPTH,
                          storage=dace.StorageType.FPGA_Local,
                          transient=False)
    write_sdfg.add_stream("forward_out",
                          vtype,
                          buffer_size=MINIMUM_CHANNEL_DEPTH,
                          storage=dace.StorageType.FPGA_Local,
                          transient=False)
    write_sdfg.add_array("buffer_in", ("TM//W", ),
                         vtype,
                         transient=False,
                         storage=dace.StorageType.FPGA_Local)
    # Send results from this PE
    send_state = write_sdfg.add_state("send_C")
    send_read = send_state.add_read("buffer_in")
    send_write = send_state.add_write("forward_out")
    send_tasklet = send_state.add_tasklet("send_C", {"from_buffer"},
                                          {"to_next"}, "to_next = from_buffer")
    send_entry, send_exit = send_state.add_map(
        "send_C", {"m1": "0:TM//W"}, schedule=dace.ScheduleType.FPGA_Device)
    send_state.add_memlet_path(send_read,
                               send_entry,
                               send_tasklet,
                               dst_conn="from_buffer",
                               memlet=dace.Memlet("buffer_in[m1]"))
    send_state.add_memlet_path(send_tasklet,
                               send_exit,
                               send_write,
                               src_conn="to_next",
                               memlet=dace.Memlet("forward_out[0]"))
    # And finally forward results from earlier PEs
    forward_state = write_sdfg.add_state("forward_C")
    forward_read = forward_state.add_read("forward_in")
    forward_write = forward_state.add_read("forward_out")
    forward_tasklet = forward_state.add_tasklet(
        "forward_C", {"from_prev"}, {"to_next"}, """\
if p < P - 1:
    to_next = from_prev""")
    forward_entry, forward_exit = forward_state.add_map(
        "forward_C", {
            "n1": "0:P - p - 1",
            "m1": "0:TM//W"
        },
        schedule=dace.ScheduleType.FPGA_Device)
    # These must be dynamic so the compiler can optimize out the write from the
    # last processing element
    forward_state.add_memlet_path(forward_read,
                                  forward_entry,
                                  forward_tasklet,
                                  dst_conn="from_prev",
                                  memlet=dace.Memlet("forward_in[0]",
                                                     dynamic=True))
    forward_state.add_memlet_path(forward_tasklet,
                                  forward_exit,
                                  forward_write,
                                  src_conn="to_next",
                                  memlet=dace.Memlet("forward_out[0]",
                                                     dynamic=True))
    # Enforce sending own data before forwarding
    write_sdfg.add_edge(send_state, forward_state, dace.InterstateEdge())

    # Unroll processing elements
    unroll_entry, unroll_exit = state.add_map(
        "unroll_processing_elements", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    # Bring data nodes into scope
    state.add_memlet_path(unroll_entry, A_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(unroll_entry, B_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(unroll_entry, C_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(unroll_entry,
                          C_buffer_read,
                          memlet=dace.memlet.Memlet())
    state.add_memlet_path(A_pipe_out, unroll_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(B_pipe_out, unroll_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(C_pipe_out, unroll_exit, memlet=dace.memlet.Memlet())

    # Propagate symbols
    write_sdfg.symbols = copy.deepcopy(sdfg.symbols)
    write_sdfg.add_symbol("p", sdfg.symbols["P"])
    write_sdfg_node.symbol_mapping = {k: k for k in sdfg.free_symbols}
    write_sdfg_node.symbol_mapping["p"] = "p"


def make_fpga_state(sdfg, vtype):

    state = sdfg.add_state("gemm")

    sdfg.add_stream("A_pipe",
                    vtype.base_type,
                    transient=True,
                    shape=("P + 1", ),
                    storage=dace.dtypes.StorageType.FPGA_Local,
                    buffer_size="P")
    sdfg.add_stream("B_pipe",
                    vtype,
                    transient=True,
                    shape=("P + 1", ),
                    buffer_size=MINIMUM_CHANNEL_DEPTH,
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("B_to_feeder",
                    vtype,
                    transient=True,
                    buffer_size=MINIMUM_CHANNEL_DEPTH,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream("C_pipe",
                    vtype,
                    transient=True,
                    shape=("P + 1", ),
                    buffer_size=MINIMUM_CHANNEL_DEPTH,
                    storage=dace.dtypes.StorageType.FPGA_Local)

    make_read_A(sdfg, state, vtype)
    make_read_B(sdfg, state, vtype)
    make_feed_B(sdfg, state, vtype)
    make_compute(sdfg, state, vtype)
    make_write_C(sdfg, state, vtype)

    return state


def make_sdfg(name, vtype):

    sdfg = dace.SDFG(name)

    pre_state = make_copy_to_fpga_state(sdfg, vtype)
    compute_state = make_fpga_state(sdfg, vtype)
    post_state = make_copy_to_host_state(sdfg, vtype)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    if vtype.bytes < 64:
        sdfg.expand_library_nodes()
        assert sdfg.apply_transformations_repeated(InlineSDFG) == 2

    return sdfg


@click.command()
@click.argument("N", type=int)
@click.argument("K", type=int)
@click.argument("M", type=int)
@click.argument("num-pes", type=int)
@click.argument("vector-width", type=int)
@click.option("--dtype", default="float32")
@click.option("--tile-size-n",
              type=int,
              default=None,
              help=("Must be a multiple of the number of processing elements, "
                    "and must divide the size in N."))
@click.option("--tile-size-m",
              type=int,
              default=None,
              help=("Must be a multiple of the vector size, and must divide"
                    " the size in M."))
@click.option("--specialize/--no-specialize",
              default=False,
              help="Fix matrix sizes at compile time.")
def cli(n, k, m, num_pes, dtype, tile_size_n, tile_size_m, vector_width,
        specialize):

    # Some reasonable default values for tile sizes
    if not tile_size_n:
        tile_size_n = n // num_pes
    if not tile_size_m:
        tile_size_m = min(m, 1024)

    # Rename
    P = num_pes
    W = vector_width
    TN = tile_size_n
    TM = tile_size_m

    dtype = getattr(dace.dtypes, dtype)  # Convert from string to typeclass
    vtype = dace.vector(dtype, vector_width)

    if TN % P != 0:
        raise ValueError(
            f"Tile size in N {TN} must be divisible by the number of processing elements {P}."
        )
    if TM % W != 0:
        raise ValueError(
            f"Tile size in M {TM} must be divisible by the vectorization width {W}."
        )
    if n % TN != 0:
        raise ValueError(
            f"Size in N {n} must be divisible by the tile size in N {TN}.")
    if n % TM != 0:
        raise ValueError(
            f"Size in M {m} must be divisible by the tile size in M {TM}.")
    if (dtype.bytes * TM) % 64 != 0:
        raise ValueError(f"Tile size in M {TM} must be a multiple of 64 bytes.")
    if (dtype.bytes * k) % 64 != 0:
        raise ValueError(f"Size in K {K} must be a multiple of 64 bytes.")

    dtype = dtype.type  # Convert from typeclass to NumPy type

    if specialize:
        name = (f"gemm_fpga_systolic_vectorized_d{num_pes}_"
                f"w{vector_width}_{tile_size_n}x{tile_size_m}_{n}x{k}x{m}")
    else:
        name = (f"gemm_fpga_systolic_vectorized_d{num_pes}_"
                f"w{vector_width}_{tile_size_n}x{tile_size_m}_NxKxM")

    sdfg = make_sdfg(name, vtype)

    # Specialize compile time constants
    sdfg.specialize({"P": P, "W": W, "TN": TN, "TM": TM})
    if specialize:
        sdfg.specialize({"N": n, "K": k, "M": m})

    print(f"Matrix multiplication {n}x{k}x{m} "
          f"with {num_pes} PEs "
          f"and vectorization width {vector_width}, "
          f"and tile sizes {TN}x{TM}.")

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([n, k], dtype=dtype)
    B = np.ndarray([k, m], dtype=dtype)
    C = np.ndarray([n, m], dtype=dtype)
    A[:] = np.random.rand(n, k).astype(dtype)
    B[:] = np.random.rand(k, m).astype(dtype)
    C[:] = np.random.rand(n, m).astype(dtype)

    # Compute reference result
    C_reference = A @ B + C

    # Run DaCe program
    if specialize:
        sdfg(A=A, B=B, C=C)
    else:
        sdfg(A=A, B=B, C=C, N=n, K=k, M=m)

    # Verify results
    if not np.allclose(C, C_reference):
        raise ValueError("Verification failed.")
    else:
        print("Results successfully verified.")


if __name__ == "__main__":
    cli()
