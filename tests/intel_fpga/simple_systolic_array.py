# Simple systolic array of P processing element, each one increments by 1 the incoming element

import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
P = dace.symbol("P")


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    A_host = state.add_array("A", [N], dtype=dace.int32)

    A_device = state.add_array(
        "A_device", [N],
        dtype=dace.int32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_edge(A_host, None, A_device, None,
                   dace.memlet.Memlet.simple(A_device, "0:N"))
    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    A_device = state.add_array(
        "A_device", [N],
        dtype=dace.int32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    A_host = state.add_array("A", [N], dtype=dace.int32)

    state.add_edge(A_device, None, A_host, None,
                   dace.memlet.Memlet.simple(A_host, "0:N"))

    return state


def make_read_A_sdfg():

    sdfg = dace.SDFG("array_read_A")

    n_inner_begin = sdfg.add_state("n_inner_begin")
    n_inner_entry = sdfg.add_state("n_inner_entry")
    n_inner_end = sdfg.add_state("n_inner_end")

    loop_body = sdfg.add_state("read_memory")

    sdfg.add_edge(
        n_inner_begin,
        n_inner_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(
        n_inner_entry,
        loop_body,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        n_inner_entry,
        n_inner_end,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(
        loop_body,
        n_inner_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": "n + 1"}))

    mem = loop_body.add_array(
        "mem", [N],
        dtype=dace.int32,
        storage=dace.dtypes.StorageType.FPGA_Global)

    pipe = loop_body.add_stream(
        "pipe", dace.int32, storage=dace.dtypes.StorageType.FPGA_Local)

    loop_body.add_memlet_path(
        mem,
        pipe,
        memlet=dace.memlet.Memlet(
            pipe,
            dace.symbolic.pystr_to_symbolic("1"),
            dace.properties.SubsetProperty.from_string("0"),
            1,
            other_subset=dace.properties.SubsetProperty.from_string("n")))

    return sdfg


def make_write_A_sdfg():

    sdfg = dace.SDFG("array_write_A")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    loop_body = sdfg.add_state("write_memory")

    sdfg.add_edge(
        n_begin,
        n_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": 0}))

    sdfg.add_edge(
        n_entry,
        loop_body,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(
        loop_body,
        n_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": "n + 1"}))

    sdfg.add_edge(
        n_entry,
        n_end,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))

    mem = loop_body.add_array(
        "mem", [N],
        dtype=dace.int32,
        storage=dace.dtypes.StorageType.FPGA_Global)

    pipe = loop_body.add_stream(
        "pipe", dace.int32, storage=dace.dtypes.StorageType.FPGA_Local)

    loop_body.add_memlet_path(
        pipe,
        mem,
        memlet=dace.memlet.Memlet(
            mem,
            dace.symbolic.pystr_to_symbolic("1"),
            dace.properties.SubsetProperty.from_string("n"),
            1,
            other_subset=dace.properties.SubsetProperty.from_string("0")))

    return sdfg


def make_compute_sdfg():

    sdfg = dace.SDFG("gemm_compute")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    state = sdfg.add_state("compute")

    # Data nodes
    A_pipe_in = state.add_stream(
        "A_stream_in", dace.int32, storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream(
        "A_stream_out", dace.int32, storage=dace.dtypes.StorageType.FPGA_Local)

    # N-loop
    sdfg.add_edge(
        n_begin,
        n_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(
        n_entry,
        state,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        n_entry,
        n_end,
        dace.graph.edges.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))

    # Backtrack two loops
    sdfg.add_edge(
        state,
        n_entry,
        dace.graph.edges.InterstateEdge(assignments={"n": "n + 1"}))

    # Compute tasklet

    compute_tasklet = state.add_tasklet("add", {"a_in"}, {"a_out"},
                                        "a_out = a_in +1")

    state.add_memlet_path(
        A_pipe_in,
        compute_tasklet,
        memlet=dace.memlet.Memlet(
            A_pipe_in, dace.symbolic.pystr_to_symbolic("-1"),
            dace.properties.SubsetProperty.from_string("0"), 1),
        dst_conn="a_in")
    state.add_memlet_path(
        compute_tasklet,
        A_pipe_out,
        memlet=dace.memlet.Memlet(
            A_pipe_out, dace.symbolic.pystr_to_symbolic("-1"),
            dace.properties.SubsetProperty.from_string("0"), 1),
        src_conn="a_out")

    return sdfg


def make_fpga_state(sdfg):

    state = sdfg.add_state("simple_array")

    read_A_sdfg = make_read_A_sdfg()
    read_A_sdfg_node = state.add_nested_sdfg(read_A_sdfg, sdfg, {"mem"},
                                             {"pipe"})

    compute_sdfg = make_compute_sdfg()
    compute_sdfg_node = state.add_nested_sdfg(
        compute_sdfg, sdfg, {"A_stream_in"}, {"A_stream_out"})

    write_A_sdfg = make_write_A_sdfg()
    write_A_sdfg_node = state.add_nested_sdfg(write_A_sdfg, sdfg, {"pipe"},
                                              {"mem"})

    A_IN = state.add_array(
        "A_device", [N],
        dtype=dace.int32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    A_OUT = state.add_array(
        "A_device", [N],
        dtype=dace.int32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    A_pipe_read = state.add_stream(
        "A_pipe",
        dace.int32,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_in = state.add_stream(
        "A_pipe",
        dace.int32,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_write = state.add_stream(
        "A_pipe",
        dace.int32,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream(
        "A_pipe",
        dace.int32,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)

    compute_entry, compute_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    # Bring data nodes into scope
    state.add_memlet_path(
        compute_entry, A_pipe_in, memlet=dace.memlet.EmptyMemlet())
    state.add_memlet_path(
        A_pipe_out, compute_exit, memlet=dace.memlet.EmptyMemlet())

    # Connect data nodes
    state.add_memlet_path(
        A_pipe_in,
        compute_sdfg_node,
        dst_conn="A_stream_in",
        memlet=dace.memlet.Memlet(
            A_pipe_in, dace.symbolic.pystr_to_symbolic("N/P"),
            dace.properties.SubsetProperty.from_string("p"), 1))
    state.add_memlet_path(
        compute_sdfg_node,
        A_pipe_out,
        src_conn="A_stream_out",
        memlet=dace.memlet.Memlet(
            A_pipe_out, dace.symbolic.pystr_to_symbolic("N/P"),
            dace.properties.SubsetProperty.from_string("p + 1"), 1))

    state.add_memlet_path(
        A_IN,
        read_A_sdfg_node,
        dst_conn="mem",
        memlet=dace.memlet.Memlet(
            A_IN, dace.symbolic.pystr_to_symbolic("N"),
            dace.properties.SubsetProperty.from_string("0:N"), 1))
    state.add_memlet_path(
        read_A_sdfg_node,
        A_pipe_read,
        src_conn="pipe",
        memlet=dace.memlet.Memlet(
            A_pipe_in, dace.symbolic.pystr_to_symbolic("N"),
            dace.properties.SubsetProperty.from_string("0"), 1))

    state.add_memlet_path(
        A_pipe_write,
        write_A_sdfg_node,
        dst_conn="pipe",
        memlet=dace.memlet.Memlet(
            A_pipe_out, dace.symbolic.pystr_to_symbolic("N"),
            dace.properties.SubsetProperty.from_string("P"), 1))
    state.add_memlet_path(
        write_A_sdfg_node,
        A_OUT,
        src_conn="mem",
        memlet=dace.memlet.Memlet(
            A_OUT, dace.symbolic.pystr_to_symbolic("N"),
            dace.properties.SubsetProperty.from_string("0:N"), 1))

    return state


def make_sdfg(specialized):

    sdfg = dace.SDFG("simple_systolic_array_{}".format(P.get()))

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.graph.edges.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.graph.edges.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("P", type=int)

    args = vars(parser.parse_args())

    P.set(args["P"])
    N.set(args["N"])
    sdfg = make_sdfg(False)
    sdfg.specialize(dict(P=P, N=N))

    print("Simple Systolic array")

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get()], dtype=dace.int32.type)
    A[:] = np.random.randint(0, 1000, N.get()).astype(dace.int32.type)

    A_Exp = A + P.get()

    sdfg.draw_to_file()
    sdfg(A=A)
    # print("A: ", A)
    # print("A_Exp: ", A_Exp)
    diff = np.abs(A_Exp - A)
    diff_total = np.sum(diff)
    highest_diff = np.max(diff)
    wrong_elements = np.transpose(np.nonzero(diff >= 0.01))

    print("==== Program end ====")

    if diff_total >= 0.01:
        print("Verification failed!")
        exit(1)
    else:
        print("Results verified successfully.")
    exit(0)
