# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("mm_fpga_pipelined_{}x{}x{}".format(N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("mm_fpga_pipelined_NxKx{}".format(M.get()))

    ###########################################################################
    # Copy data to FPGA

    pre_state = sdfg.add_state("pre_mm")

    A_host = pre_state.add_array("A", [N, K], dtype=dace.float32)
    B_host = pre_state.add_array("B", [K, M], dtype=dace.float32)
    C_host = pre_state.add_array("C", [N, M], dtype=dace.float32)

    A_device = pre_state.add_array("A_device", [N, K],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)
    B_device = pre_state.add_array("B_device", [K, M],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)
    C_device = pre_state.add_array("C_device", [N, M],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)

    pre_state.add_edge(A_host, None, A_device, None, dace.memlet.Memlet.simple(A_device, "0:N, 0:K"))
    pre_state.add_edge(B_host, None, B_device, None, dace.memlet.Memlet.simple(B_device, "0:K, 0:M"))
    pre_state.add_edge(C_host, None, C_device, None, dace.memlet.Memlet.simple(C_device, "0:N, 0:M"))

    ###########################################################################
    # Compute

    state = sdfg.add_state("mm")
    sdfg.add_edge(pre_state, state, dace.sdfg.InterstateEdge())

    A = state.add_array("A_device", [N, K],
                        dtype=dace.float32,
                        transient=True,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    B = state.add_array("B_device", [K, M],
                        dtype=dace.float32,
                        transient=True,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    C = state.add_array("C_device", [N, M],
                        dtype=dace.float32,
                        transient=True,
                        storage=dace.dtypes.StorageType.FPGA_Global)

    C_buffer_in = state.add_array("C_buffer", [M],
                                  dtype=dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_out = state.add_array("C_buffer", [M],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Local)

    n_entry, n_exit = state.add_map("Map_N", {"n": "0:N"}, schedule=dace.dtypes.ScheduleType.FPGA_Device)
    k_entry, k_exit = state.add_map("Map_K", {"k": "0:K"}, schedule=dace.dtypes.ScheduleType.FPGA_Device)
    m_entry, m_exit = state.add_map("Map_M", {"m": "0:M"}, schedule=dace.dtypes.ScheduleType.FPGA_Device)

    state.add_nedge(n_entry, C_buffer_in, dace.memlet.Memlet())

    ###########################################################################
    # Nested SDFG

    nested_sdfg = dace.SDFG("zero_or_wcr")

    if_state = nested_sdfg.add_state("if_state")
    then_state = nested_sdfg.add_state("then_state")
    else_state = nested_sdfg.add_state("else_state")
    end_state = nested_sdfg.add_state("end_state")
    nested_sdfg.add_edge(
        if_state, then_state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string("k == 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(
        if_state, else_state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string("k != 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(then_state, end_state, dace.sdfg.InterstateEdge())
    nested_sdfg.add_edge(else_state, end_state, dace.sdfg.InterstateEdge())

    # These are identical, they only differ in their confres
    then_tasklet = then_state.add_tasklet("multiply", {"a", "b"}, {"c_out"}, "c_out = a * b")
    else_tasklet = else_state.add_tasklet("multiply", {"a", "b", "c_in"}, {"c_out"}, "c_out = c_in + a * b")

    # Add scalar I/O
    then_A_val = then_state.add_scalar("A_val", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    then_B_val = then_state.add_scalar("B_val", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    then_C_out = then_state.add_scalar("C_out", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)

    else_A_val = else_state.add_scalar("A_val", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    else_B_val = else_state.add_scalar("B_val", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    else_C_in = else_state.add_scalar("C_in", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    else_C_out = else_state.add_scalar("C_out", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)

    # Memlets
    then_a_val_memlet = dace.memlet.Memlet.simple(then_A_val, "0")
    then_b_val_memlet = dace.memlet.Memlet.simple(then_B_val, "0")
    then_c_out_memlet = dace.memlet.Memlet.simple(then_C_out, "0")

    else_a_val_memlet = dace.memlet.Memlet.simple(else_A_val, "0")
    else_b_val_memlet = dace.memlet.Memlet.simple(else_B_val, "0")
    else_c_in_memlet = dace.memlet.Memlet.simple(else_C_in, "0")
    else_c_out_memlet = dace.memlet.Memlet.simple(else_C_out, "0")

    # Draw paths within each state
    then_state.add_memlet_path(then_A_val, then_tasklet, memlet=then_a_val_memlet, dst_conn="a")
    then_state.add_memlet_path(then_B_val, then_tasklet, memlet=then_b_val_memlet, dst_conn="b")
    then_state.add_memlet_path(then_tasklet, then_C_out, memlet=then_c_out_memlet, src_conn="c_out")

    else_state.add_memlet_path(else_A_val, else_tasklet, memlet=else_a_val_memlet, dst_conn="a")
    else_state.add_memlet_path(else_B_val, else_tasklet, memlet=else_b_val_memlet, dst_conn="b")
    else_state.add_memlet_path(else_C_in, else_tasklet, memlet=else_c_in_memlet, dst_conn="c_in")
    else_state.add_memlet_path(else_tasklet, else_C_out, memlet=else_c_out_memlet, src_conn="c_out")

    tasklet = state.add_nested_sdfg(nested_sdfg, sdfg, {"A_val", "B_val", "C_in"}, {"C_out"})

    ###########################################################################
    # Compute continued

    # tasklet = state.add_tasklet("multiply", {"a", "b"}, {"c"}, "c = a * b")

    read_a_memlet = dace.memlet.Memlet.simple(A, "n, k")
    read_b_memlet = dace.memlet.Memlet.simple(B, "k, m")
    read_c_memlet = dace.memlet.Memlet.simple(C_buffer_in, "m")

    state.add_memlet_path(A, n_entry, k_entry, m_entry, tasklet, memlet=read_a_memlet, dst_conn="A_val")
    state.add_memlet_path(B, n_entry, k_entry, m_entry, tasklet, memlet=read_b_memlet, dst_conn="B_val")
    state.add_memlet_path(C_buffer_in, k_entry, m_entry, tasklet, memlet=read_c_memlet, dst_conn="C_in")

    write_buffer_memlet = dace.memlet.Memlet.simple(C_buffer_out, "m")

    state.add_memlet_path(tasklet, m_exit, k_exit, C_buffer_out, memlet=write_buffer_memlet, src_conn="C_out")

    write_c_memlet = dace.memlet.Memlet.simple(C, "n, 0:M")

    state.add_memlet_path(C_buffer_out, n_exit, C, memlet=write_c_memlet)

    ###########################################################################
    # Copy back result

    post_state = sdfg.add_state("post_mm")
    sdfg.add_edge(state, post_state, dace.sdfg.InterstateEdge())

    C_device = post_state.add_array("C_device", [N, M],
                                    dtype=dace.float32,
                                    transient=True,
                                    storage=dace.dtypes.StorageType.FPGA_Global)

    C_host = post_state.add_array("C", [N, M], dtype=dace.float32)

    post_state.add_edge(C_device, None, C_host, None, dace.memlet.Memlet.simple(C_device, "0:N, 0:M"))

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
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float32.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    if args["specialize"]:
        sdfg(A=A, B=B, C=C)
    else:
        sdfg(A=A, B=B, C=C, N=N, K=K)

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
