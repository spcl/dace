# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", [N, K], dtype=dace.float32)
    sdfg.add_array("B", [K, M], dtype=dace.float32)
    sdfg.add_array("C", [N, M], dtype=dace.float32)

    A_host = state.add_read("A")
    B_host = state.add_read("B")
    C_host = state.add_read("C")

    sdfg.add_array("A_device", [N, K], dtype=dace.float32, transient=True, storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M], dtype=dace.float32, transient=True, storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M], dtype=dace.float32, transient=True, storage=dace.StorageType.FPGA_Global)

    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_edge(A_host, None, A_device, None, dace.Memlet("A_device"))
    state.add_edge(B_host, None, B_device, None, dace.Memlet("B_device"))
    state.add_edge(C_host, None, C_device, None, dace.Memlet("C_device"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_edge(C_device, None, C_host, None, dace.Memlet("C"))

    return state


def make_fpga_state(sdfg):

    state = sdfg.add_state("mm")

    A = state.add_read("A_device")
    B = state.add_read("B_device")
    C = state.add_write("C_device")

    A_pipe_in = state.add_stream("A_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    B_pipe_in = state.add_stream("B_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    C_pipe_in = state.add_stream("C_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream("A_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    C_pipe_out = state.add_stream("C_pipe", dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)

    state.add_memlet_path(A, A_pipe_out, memlet=dace.Memlet("A_device"))

    read_b_entry, read_b_exit = state.add_map("read_b", {
        "n": "0:N",
        "k": "0:K",
        "m": "0:M"
    },
                                              schedule=dace.ScheduleType.FPGA_Device)
    read_b_tasklet = state.add_tasklet("read_b", {"mem"}, {"s"}, "s = mem")
    state.add_memlet_path(B, read_b_entry, read_b_tasklet, dst_conn="mem", memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(read_b_tasklet, read_b_exit, B_pipe_out, src_conn="s", memlet=dace.Memlet("B_pipe[0]"))

    state.add_memlet_path(C_pipe_in, C, src_conn="mem", memlet=dace.Memlet("C_device"))

    ###########################################################################

    n_entry, n_exit = state.add_map("outer_map", {"n": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)
    km_entry, km_exit = state.add_map("inner_map", {"k": "0:K", "m": "0:M"}, schedule=dace.ScheduleType.FPGA_Device)

    sdfg.add_array("output_buffer", [M], dtype=dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    sdfg.add_array("A_reg", [1], dtype=dace.float32, transient=True, storage=dace.StorageType.FPGA_Local)
    output_buffer_read = state.add_read("output_buffer")
    output_buffer_write = state.add_write("output_buffer")
    read_a_reg = state.add_read("A_reg")
    write_a_reg = state.add_write("A_reg")

    tasklet = state.add_tasklet(
        "multiply_accumulate", {"a_mem", "a_reg_in", "b", "c_in"}, {"a_reg_out", "c_out"}, """\
a = a_mem if m == 0 else a_reg_in
a_reg_out = a
prev = 0 if k == 0 else c_in
c_out = prev + a * b""")

    state.add_memlet_path(A_pipe_in,
                          n_entry,
                          km_entry,
                          tasklet,
                          dst_conn="a_mem",
                          memlet=dace.Memlet("A_pipe[0]", dynamic=True))

    state.add_memlet_path(B_pipe_in, n_entry, km_entry, tasklet, dst_conn="b", memlet=dace.Memlet("B_pipe[0]"))

    state.add_memlet_path(read_a_reg, n_entry, km_entry, tasklet, dst_conn="a_reg_in", memlet=dace.Memlet("A_reg[0]"))

    state.add_memlet_path(output_buffer_read,
                          km_entry,
                          tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("output_buffer[m]"))

    # Make sure it's in scope
    state.add_memlet_path(n_entry, output_buffer_read, memlet=dace.Memlet())

    state.add_memlet_path(tasklet,
                          km_exit,
                          output_buffer_write,
                          src_conn="c_out",
                          memlet=dace.Memlet("output_buffer[m]"))

    state.add_memlet_path(tasklet, km_exit, n_exit, write_a_reg, src_conn="a_reg_out", memlet=dace.Memlet("A_reg[0]"))

    state.add_memlet_path(output_buffer_write, n_exit, C_pipe_out, memlet=dace.Memlet("output_buffer[0:M]"))

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("mm_fpga_stream_{}x{}x{}".format(N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("mm_fpga_stream_NxKx{}".format(M.get()))

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = vars(parser.parse_args())

    if not args["specialize"]:
        M.set(args["M"])
        # M must always be specialized, as it's used for the static buffer size
        sdfg = make_sdfg(False)
        sdfg.specialize(dict(M=M))
        N.set(args["N"])
        K.set(args["K"])
    else:
        M.set(args["M"])
        N.set(args["N"])
        K.set(args["K"])
        sdfg = make_sdfg(True)
        sdfg.specialize(dict(M=M, N=N, K=K))

    print("Matrix multiplication {}x{}x{} ({}specialized)".format(M.get(), N.get(), K.get(),
                                                                  "" if args["specialize"] else "not "))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = 1  # np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    B[:] = 1  # np.random.rand(K.get(), M.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    A_regression = np.ndarray([N.get(), K.get()], dtype=np.float32)
    B_regression = np.ndarray([K.get(), M.get()], dtype=np.float32)
    C_regression = np.ndarray([N.get(), M.get()], dtype=np.float32)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    if args["specialize"]:
        sdfg(A=A, B=B, C=C)
    else:
        sdfg(A=A, B=B, C=C, N=N, K=K)

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
