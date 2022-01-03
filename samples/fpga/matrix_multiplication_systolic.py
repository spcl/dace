# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import click
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
K = dace.symbol("K")
M = dace.symbol("M")
P = dace.symbol("P")


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

    sdfg.add_array("A_device", [N, K], dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M], dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M], dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_memlet_path(A_host, A_device, memlet=dace.Memlet("A_device[0:N, 0:K]"))
    state.add_memlet_path(B_host, B_device, memlet=dace.Memlet("B_device[0:K, 0:M]"))
    state.add_memlet_path(C_host, C_device, memlet=dace.Memlet("C_device[0:N, 0:M]"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_memlet_path(C_device, C_host, memlet=dace.Memlet("C[0:N, 0:M]"))

    return state


def make_read_A(state):

    entry, exit = state.add_map("read_A", {
        "n0": "0:N/P",
        "k": "0:K",
        "n1": "0:P"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("A_device")
    pipe = state.add_write("A_pipe")
    tasklet = state.add_tasklet("read_A", {"from_memory"}, {"to_kernel"}, "to_kernel = from_memory")

    state.add_memlet_path(mem, entry, tasklet, dst_conn="from_memory", memlet=dace.Memlet("A_device[n0 * P + n1, k]"))
    state.add_memlet_path(tasklet, exit, pipe, src_conn="to_kernel", memlet=dace.Memlet("A_pipe[0]"))


def make_read_B(state):

    entry, exit = state.add_map("read_B", {
        "n": "0:N/P",
        "k": "0:K",
        "m": "0:M"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("B_device")
    pipe = state.add_write("B_pipe")
    tasklet = state.add_tasklet("read_B", {"from_memory"}, {"to_kernel"}, "to_kernel = from_memory")

    state.add_memlet_path(mem, entry, tasklet, dst_conn="from_memory", memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(tasklet, exit, pipe, src_conn="to_kernel", memlet=dace.Memlet("B_pipe[0]"))


def make_write_C(state):

    pipe = state.add_read("C_pipe")
    mem = state.add_write("C_device")

    state.add_memlet_path(pipe, mem, memlet=dace.Memlet("C_device[0:N, 0:M]", other_subset="P - 1"))


def make_compute(sdfg, state):

    A_pipe_in = state.add_read("A_pipe")
    A_pipe_out = state.add_write("A_pipe")
    B_pipe_in = state.add_read("B_pipe")
    B_pipe_out = state.add_write("B_pipe")
    C_pipe_in = state.add_read("C_pipe")
    C_pipe_out = state.add_write("C_pipe")

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    }, schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"}, schedule=dace.ScheduleType.FPGA_Device)
    entry_a, exit_a = state.add_map("buffer_A", {"n1": "0:P"}, schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M"}, schedule=dace.ScheduleType.FPGA_Device)
    entry_c, exit_c = state.add_map("write_C", {"n1": "0:P", "m": "0:M"}, schedule=dace.ScheduleType.FPGA_Device)

    # Instantiate buffers
    sdfg.add_scalar("A_reg", dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Registers)
    A_reg = state.add_write("A_reg")
    sdfg.add_array("C_buffer", [M], dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_in = state.add_read("C_buffer")
    C_buffer_out = state.add_write("C_buffer")

    buffer_a_tasklet = state.add_tasklet("buffer_a", {"a_in"}, {"a_reg", "a_out"}, """\
if n1 == P - p - 1:
    a_reg = a_in
if p < P - 1:
    a_out = a_in""")
    state.add_memlet_path(A_pipe_in,
                          entry_n0,
                          entry_k,
                          entry_a,
                          buffer_a_tasklet,
                          memlet=dace.Memlet("A_pipe[p]", dynamic=False),
                          dst_conn="a_in")
    state.add_memlet_path(buffer_a_tasklet,
                          exit_a,
                          A_reg,
                          memlet=dace.Memlet("A_reg[0]", dynamic=True),
                          src_conn="a_reg")
    state.add_memlet_path(buffer_a_tasklet,
                          exit_a,
                          exit_k,
                          exit_n0,
                          A_pipe_out,
                          memlet=dace.Memlet("A_pipe[p + 1]", dynamic=True),
                          src_conn="a_out")

    compute_tasklet = state.add_tasklet(
        "multiply_add", {"a_in", "b_in", "c_in"}, {"b_out", "c_out"}, """\
c_prev = 0 if k == 0 else c_in
c_out = c_prev + a_in * b_in
if p < P - 1:
    b_out = b_in""")

    state.add_memlet_path(A_reg, entry_m, compute_tasklet, dst_conn="a_in", memlet=dace.Memlet("A_reg[0]"))
    state.add_memlet_path(B_pipe_in,
                          entry_n0,
                          entry_k,
                          entry_m,
                          compute_tasklet,
                          memlet=dace.Memlet("B_pipe[p]", dynamic=False),
                          dst_conn="b_in")
    state.add_memlet_path(compute_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          B_pipe_out,
                          memlet=dace.Memlet("B_pipe[p + 1]", dynamic=True),
                          src_conn="b_out")
    state.add_memlet_path(C_buffer_in,
                          entry_k,
                          entry_m,
                          compute_tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("C_buffer[m]"))
    state.add_memlet_path(entry_n0, C_buffer_in, memlet=dace.Memlet())
    state.add_memlet_path(compute_tasklet,
                          exit_m,
                          exit_k,
                          C_buffer_out,
                          memlet=dace.Memlet("C_buffer[m]"),
                          src_conn="c_out")
    state.add_memlet_path(C_buffer_out, exit_n0, memlet=dace.Memlet())

    # Write back
    write_c_tasklet = state.add_tasklet("write_c", {"buffer_in", "forward_in"}, {"c_out"}, """\
if n1 <= p:
    c_out = forward_in if p > 0 and n1 > 0 else buffer_in""")
    state.add_memlet_path(C_buffer_out,
                          entry_c,
                          write_c_tasklet,
                          memlet=dace.Memlet("C_buffer[m]", dynamic=True),
                          dst_conn="buffer_in")
    state.add_memlet_path(C_pipe_in,
                          entry_n0,
                          entry_c,
                          write_c_tasklet,
                          memlet=dace.Memlet("C_pipe[p-1]", dynamic=True),
                          dst_conn="forward_in")
    state.add_memlet_path(write_c_tasklet,
                          exit_c,
                          exit_n0,
                          C_pipe_out,
                          memlet=dace.Memlet("C_pipe[p]", dynamic=True),
                          src_conn="c_out")

    # Unroll processing elements
    compute_entry, compute_exit = state.add_map("unroll_compute", {"p": "0:P"},
                                                schedule=dace.ScheduleType.FPGA_Device,
                                                unroll=True)

    # Bring data nodes into scope
    state.add_memlet_path(compute_entry, A_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(compute_entry, B_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(compute_entry, C_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(A_pipe_out, compute_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(B_pipe_out, compute_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(C_pipe_out, compute_exit, memlet=dace.memlet.Memlet())


def make_fpga_state(sdfg):

    state = sdfg.add_state("mm")

    sdfg.add_stream("A_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P + 1, ),
                    storage=dace.dtypes.StorageType.FPGA_Local,
                    buffer_size="P")
    sdfg.add_stream("B_pipe", dace.float32, transient=True, shape=(P + 1, ), storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("C_pipe", dace.float32, transient=True, shape=(P, ), storage=dace.dtypes.StorageType.FPGA_Local)

    make_read_A(state)
    make_read_B(state)
    make_compute(sdfg, state)
    make_write_C(state)

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("mm_fpga_systolic_{}_{}x{}x{}".format(P.get(), N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("mm_fpga_systolic_{}_NxKx{}".format(P.get(), M.get()))

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


def run_matmul_systolic(m, n, k, p, specialize):
    print("==== Program start ====")

    if not specialize:
        P.set(p)
        M.set(m)
        # M must always be specialized, as it's used for the static buffer size
        sdfg = make_sdfg(False)
        sdfg.specialize(dict(P=p, M=m))
        N.set(n)
        K.set(k)
    else:
        P.set(p)
        M.set(m)
        N.set(n)
        K.set(k)
        sdfg = make_sdfg(True)
        sdfg.specialize(dict(P=p, M=m, N=n, K=k))

    print("Matrix multiplication {}x{}x{} with {} PEs ({}specialized)".format(M.get(), N.get(), K.get(), P.get(),
                                                                              "" if specialize else "not "))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    B[:] = np.random.rand(K.get(), M.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    A_regression = np.ndarray([N.get(), K.get()], dtype=np.float32)
    B_regression = np.ndarray([K.get(), M.get()], dtype=np.float32)
    C_regression = np.ndarray([N.get(), M.get()], dtype=np.float32)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    if specialize:
        sdfg(A=A, B=B, C=C)
    else:
        sdfg(A=A, B=B, C=C, N=N, K=K)

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")

    print("==== Program end ====")

    return sdfg


@click.command()
@click.argument("M", type=int)
@click.argument("N", type=int)
@click.argument("K", type=int)
@click.argument("P", type=int)
@click.option("--specialize/--no-specialize", default=False, help="Fix all loop bounds at compile time/in hardware")
def cli(m, n, k, p, specialize):
    run_matmul_systolic(m, n, k, p, specialize)


if __name__ == "__main__":
    cli()
