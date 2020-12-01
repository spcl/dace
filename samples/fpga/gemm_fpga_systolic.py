# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
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

    sdfg.add_array("A_device", [N, K],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_memlet_path(A_host,
                          A_device,
                          memlet=dace.Memlet("A_device[0:N, 0:K]"))
    state.add_memlet_path(B_host,
                          B_device,
                          memlet=dace.Memlet("B_device[0:K, 0:M]"))
    state.add_memlet_path(C_host,
                          C_device,
                          memlet=dace.Memlet("C_device[0:N, 0:M]"))

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
    tasklet = state.add_tasklet("read_A", {"from_memory"}, {"to_kernel"},
                                "to_kernel = from_memory")

    state.add_memlet_path(mem,
                          entry,
                          tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet("A_device[n0 * P + n1, k]"))
    state.add_memlet_path(tasklet,
                          exit,
                          pipe,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("A_pipe[0]"))


def make_read_B(state):

    entry, exit = state.add_map("read_B", {
        "n": "0:N/P",
        "k": "0:K",
        "m": "0:M"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("B_device")
    pipe = state.add_write("B_pipe")
    tasklet = state.add_tasklet("read_B", {"from_memory"}, {"to_kernel"},
                                "to_kernel = from_memory")

    state.add_memlet_path(mem,
                          entry,
                          tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(tasklet,
                          exit,
                          pipe,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("B_pipe[0]"))


def make_write_C(state):

    pipe = state.add_read("C_pipe")
    mem = state.add_write("C_device")

    state.add_memlet_path(pipe, mem, memlet=dace.Memlet("C_device[0:N, 0:M]", other_subset="P"))


def make_compute_sdfg():

    sdfg = dace.SDFG("gemm_compute")

    n_begin = sdfg.add_state("n0_begin")
    n_entry = sdfg.add_state("n0_entry")
    n_end = sdfg.add_state("n0_end")

    k_begin = sdfg.add_state("k_begin")
    k_entry = sdfg.add_state("k_entry")
    k_end = sdfg.add_state("k_end")

    a_begin = sdfg.add_state("a_begin")
    a_entry = sdfg.add_state("a_entry")
    a_end = sdfg.add_state("a_end")

    buffer_a_state = sdfg.add_state("read_a")

    m_begin = sdfg.add_state("m_begin")
    m_entry = sdfg.add_state("m_entry")
    m_end = sdfg.add_state("m_end")

    state = sdfg.add_state("compute")

    c_n1_begin = sdfg.add_state("c_n1_begin")
    c_n1_entry = sdfg.add_state("c_n1_entry")
    c_n1_end = sdfg.add_state("c_n1_end")

    c_m_begin = sdfg.add_state("c_m_begin")
    c_m_entry = sdfg.add_state("c_m_entry")
    c_m_end = sdfg.add_state("c_m_end")

    write_c_state = sdfg.add_state("write_c")

    # Data nodes
    B_pipe_in = state.add_stream("B_stream_in",
                                 dace.float32,
                                 storage=dace.dtypes.StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_stream_out",
                                  dace.float32,
                                  storage=dace.dtypes.StorageType.FPGA_Local)

    # N-loop
    sdfg.add_edge(n_begin, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n0": 0}))
    sdfg.add_edge(
        n_entry, k_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n0 < N / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        n_entry, n_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n0 >= N / P", language=dace.dtypes.Language.Python)))

    # K-loop
    sdfg.add_edge(k_begin, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": 0}))
    sdfg.add_edge(
        k_entry, a_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k < K", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, k_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k >= K", language=dace.dtypes.Language.Python)))

    # Buffer A-loop
    sdfg.add_edge(a_begin, a_entry,
                  dace.sdfg.InterstateEdge(assignments={"n1": 0}))
    sdfg.add_edge(
        a_entry, buffer_a_state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n1 < P - p", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(buffer_a_state, a_entry,
                  dace.sdfg.InterstateEdge(assignments={"n1": "n1 + 1"}))
    sdfg.add_edge(
        a_entry, a_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n1 >= P - p", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(a_end, m_begin, dace.sdfg.InterstateEdge())

    # Inner M-loop
    sdfg.add_edge(m_begin, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": 0}))
    sdfg.add_edge(
        m_entry, state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m < M", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        m_entry, m_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m >= M", language=dace.dtypes.Language.Python)))

    # Backtrack two loops
    sdfg.add_edge(state, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": "m + 1"}))
    sdfg.add_edge(m_end, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": "k + 1"}))

    # Continue to next sequential loop
    sdfg.add_edge(k_end, c_n1_begin, dace.sdfg.InterstateEdge())

    # Two C-loops
    sdfg.add_edge(c_n1_begin, c_n1_entry,
                  dace.sdfg.InterstateEdge(assignments={"n1": 0}))
    sdfg.add_edge(
        c_n1_entry, c_m_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n1 < p + 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        c_n1_entry, c_n1_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n1 >= p + 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(c_m_begin, c_m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": 0}))
    sdfg.add_edge(
        c_m_entry, write_c_state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m < M", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        c_m_entry, c_m_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m >= M", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(write_c_state, c_m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": "m + 1"}))
    sdfg.add_edge(c_m_end, c_n1_entry,
                  dace.sdfg.InterstateEdge(assignments={"n1": "n1 + 1"}))

    # Backtrack
    sdfg.add_edge(c_n1_end, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n0": "n0 + 1"}))

    # Scalar buffer for A
    A_pipe_in = buffer_a_state.add_stream(
        "A_stream_in", dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_out = buffer_a_state.add_stream(
        "A_stream_out",
        dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Local)
    A_reg_out = buffer_a_state.add_scalar(
        "A_reg",
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    buffer_a_tasklet = buffer_a_state.add_tasklet(
        "buffer_a", {"a_in"}, {"a_reg", "a_out"}, "a_input = float(0)"
        "\nif n1 == P-p-1 or p < P - 1: a_input = a_in"
        "\nif n1 == P - p - 1:"
        "\n\ta_reg = a_input"
        "\nelse:"
        "\n\tif p < P - 1:"
        "\n\t\ta_out = a_input")
    buffer_a_state.add_memlet_path(A_pipe_in,
                                   buffer_a_tasklet,
                                   memlet=dace.memlet.Memlet.simple(
                                       A_pipe_in, "0", num_accesses=-1),
                                   dst_conn="a_in")
    buffer_a_state.add_memlet_path(buffer_a_tasklet,
                                   A_reg_out,
                                   memlet=dace.memlet.Memlet.simple(
                                       A_reg_out, "0", num_accesses=-1),
                                   src_conn="a_reg")
    buffer_a_state.add_memlet_path(buffer_a_tasklet,
                                   A_pipe_out,
                                   memlet=dace.memlet.Memlet.simple(
                                       A_pipe_out, "0", num_accesses=-1),
                                   src_conn="a_out")

    ###########################################################################
    # Nested SDFG

    nested_sdfg = dace.SDFG("gemm_nested")

    if_state_c = nested_sdfg.add_state("if_state_c")
    then_state_c = nested_sdfg.add_state("then_state_c")
    else_state_c = nested_sdfg.add_state("else_state_c")
    compute_state = nested_sdfg.add_state("compute_state")
    nested_sdfg.add_edge(
        if_state_c, then_state_c,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k == 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(
        if_state_c, else_state_c,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k != 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(then_state_c, compute_state,
                         dace.sdfg.InterstateEdge())
    nested_sdfg.add_edge(else_state_c, compute_state,
                         dace.sdfg.InterstateEdge())

    compute_tasklet = compute_state.add_tasklet(
        "multiply_add", {"a_in", "b_in", "c_in"}, {"b_out", "c_out"},
        "c_out = c_in + a_in * b_in\nif p < P - 1:\n\tb_out = b_in")

    # Then state C
    zero_tasklet = then_state_c.add_tasklet("zero_C_buffer", {}, {"c_out"},
                                            "c_out = 0")
    C_val_out_then = then_state_c.add_scalar(
        "C_val",
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)
    then_state_c.add_memlet_path(zero_tasklet,
                                 C_val_out_then,
                                 src_conn="c_out",
                                 memlet=dace.memlet.Memlet.simple(
                                     C_val_out_then, "0"))

    # Else state C
    C_in = else_state_c.add_scalar("C_in",
                                   dtype=dace.float32,
                                   storage=dace.dtypes.StorageType.FPGA_Local)
    C_val_out_else = else_state_c.add_scalar(
        "C_val",
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)
    else_state_c.add_memlet_path(C_in,
                                 C_val_out_else,
                                 memlet=dace.memlet.Memlet.simple(
                                     C_val_out_else, "0"))

    # Compute state
    A_val_in = compute_state.add_scalar(
        "A_val_in",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    B_in = compute_state.add_stream(
        "B_in",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    B_out = compute_state.add_stream(
        "B_out",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    C_val_in = compute_state.add_scalar(
        "C_val",
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    C_out = compute_state.add_scalar(
        "C_out",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    compute_state.add_memlet_path(A_val_in,
                                  compute_tasklet,
                                  memlet=dace.memlet.Memlet.simple(
                                      A_val_in, "0"),
                                  dst_conn="a_in")
    compute_state.add_memlet_path(B_in,
                                  compute_tasklet,
                                  memlet=dace.memlet.Memlet.simple(B_in, "0"),
                                  dst_conn="b_in")
    compute_state.add_memlet_path(compute_tasklet,
                                  B_out,
                                  memlet=dace.memlet.Memlet.simple(
                                      B_out, "0", num_accesses=-1),
                                  src_conn="b_out")
    compute_state.add_memlet_path(C_val_in,
                                  compute_tasklet,
                                  memlet=dace.memlet.Memlet.simple(
                                      C_val_in, "0"),
                                  dst_conn="c_in")
    compute_state.add_memlet_path(compute_tasklet,
                                  C_out,
                                  memlet=dace.memlet.Memlet.simple(C_out, "0"),
                                  src_conn="c_out")

    tasklet = state.add_nested_sdfg(nested_sdfg, sdfg,
                                    {"A_val_in", "B_in", "C_in"},
                                    {"B_out", "C_out"})

    ###########################################################################
    # Compute continued

    A_reg_in = state.add_scalar("A_reg",
                                dtype=dace.float32,
                                transient=True,
                                storage=dace.dtypes.StorageType.FPGA_Registers)

    C_buffer_in = state.add_array("C_buffer", [M],
                                  dtype=dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_out = state.add_array("C_buffer", [M],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Local)

    state.add_memlet_path(B_pipe_in,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(B_pipe_in,
                                                           "0",
                                                           num_accesses=-1),
                          dst_conn="B_in")
    state.add_memlet_path(tasklet,
                          B_pipe_out,
                          memlet=dace.memlet.Memlet.simple(B_pipe_out,
                                                           "0",
                                                           num_accesses=-1),
                          src_conn="B_out")
    state.add_memlet_path(C_buffer_in,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(C_buffer_in,
                                                           "m",
                                                           num_accesses=-1),
                          dst_conn="C_in")
    state.add_memlet_path(tasklet,
                          C_buffer_out,
                          memlet=dace.memlet.Memlet.simple(C_buffer_out, "m"),
                          src_conn="C_out")
    state.add_memlet_path(A_reg_in,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(A_reg_in,
                                                           "0",
                                                           num_accesses=-1),
                          dst_conn="A_val_in")

    ###########################################################################
    # Write back state

    C_pipe_in = write_c_state.add_stream(
        "C_stream_in", dace.float32, storage=dace.dtypes.StorageType.FPGA_Local)
    C_pipe_out = write_c_state.add_stream(
        "C_stream_out",
        dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_write = write_c_state.add_array(
        "C_buffer", [M],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)
    write_c_tasklet = write_c_state.add_tasklet(
        "write_c", {"buffer_in", "forward_in"}, {"c_out"}, "if n1 == p:"
        "\n\tc_out = buffer_in"
        "\nelse:"
        "\n\tif p > 0:"
        "\n\t\tc_out = forward_in")
    write_c_state.add_memlet_path(C_buffer_write,
                                  write_c_tasklet,
                                  memlet=dace.memlet.Memlet.simple(
                                      C_buffer_in, "m", num_accesses=-1),
                                  dst_conn="buffer_in")
    write_c_state.add_memlet_path(C_pipe_in,
                                  write_c_tasklet,
                                  memlet=dace.memlet.Memlet.simple(
                                      C_pipe_in, "0", num_accesses=-1),
                                  dst_conn="forward_in")
    write_c_state.add_memlet_path(write_c_tasklet,
                                  C_pipe_out,
                                  memlet=dace.memlet.Memlet.simple(
                                      C_pipe_out, "0"),
                                  src_conn="c_out")

    return sdfg


def make_fpga_state(sdfg):

    state = sdfg.add_state("gemm")

    sdfg.add_stream("A_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P + 1, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("B_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P + 1, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("C_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P + 1, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)

    make_read_A(state)
    make_read_B(state)

    compute_sdfg = make_compute_sdfg()
    compute_sdfg_node = state.add_nested_sdfg(
        compute_sdfg, sdfg, {"A_stream_in", "B_stream_in", "C_stream_in"},
        {"A_stream_out", "B_stream_out", "C_stream_out"})

    write_C_sdfg = make_write_C(state)

    A_pipe_in = state.add_read("A_pipe")
    B_pipe_in = state.add_read("B_pipe")
    C_pipe_in = state.add_stream("C_pipe",
                                 dace.float32,
                                 transient=True,
                                 shape=(P + 1, ),
                                 storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream("A_pipe",
                                  dace.float32,
                                  transient=True,
                                  shape=(P + 1, ),
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_pipe",
                                  dace.float32,
                                  transient=True,
                                  shape=(P + 1, ),
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    C_pipe_out = state.add_stream("C_pipe",
                                  dace.float32,
                                  transient=True,
                                  shape=(P + 1, ),
                                  storage=dace.dtypes.StorageType.FPGA_Local)

    compute_entry, compute_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    # Bring data nodes into scope
    state.add_memlet_path(compute_entry, A_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(compute_entry, B_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(compute_entry, C_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(A_pipe_out, compute_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(B_pipe_out, compute_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(C_pipe_out, compute_exit, memlet=dace.memlet.Memlet())

    # Connect data nodes
    state.add_memlet_path(A_pipe_in,
                          compute_sdfg_node,
                          dst_conn="A_stream_in",
                          memlet=dace.memlet.Memlet.simple(
                              A_pipe_in,
                              'p',
                              num_accesses=dace.symbolic.pystr_to_symbolic(
                                  "(N / P) * K * (P - p)")))
    state.add_memlet_path(
        B_pipe_in,
        compute_sdfg_node,
        dst_conn="B_stream_in",
        memlet=dace.memlet.Memlet.simple(
            B_pipe_in,
            'p',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K * M / P")))
    state.add_memlet_path(
        C_pipe_in,
        compute_sdfg_node,
        dst_conn="C_stream_in",
        memlet=dace.memlet.Memlet.simple(
            C_pipe_in,
            'p',
            num_accesses=dace.symbolic.pystr_to_symbolic("(N / P) * M * p")))
    state.add_memlet_path(compute_sdfg_node,
                          A_pipe_out,
                          src_conn="A_stream_out",
                          memlet=dace.memlet.Memlet.simple(
                              A_pipe_out,
                              'p + 1',
                              num_accesses=dace.symbolic.pystr_to_symbolic(
                                  "(N / P) * K * (P - p - 1)")))
    state.add_memlet_path(compute_sdfg_node,
                          B_pipe_out,
                          src_conn="B_stream_out",
                          memlet=dace.memlet.Memlet.simple(
                              B_pipe_out,
                              'p + 1',
                              num_accesses=dace.symbolic.pystr_to_symbolic(
                                  "(p // (P - 1)) * (N / P) * K * M")))
    state.add_memlet_path(compute_sdfg_node,
                          C_pipe_out,
                          src_conn="C_stream_out",
                          memlet=dace.memlet.Memlet.simple(
                              C_pipe_out,
                              'p + 1',
                              num_accesses=dace.symbolic.pystr_to_symbolic(
                                  "(N / P) * M * (p + 1)")))

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("gemm_fpga_systolic_{}_{}x{}x{}".format(
            P.get(), N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("gemm_fpga_systolic_{}_NxKx{}".format(
            P.get(), M.get()))

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("P", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = vars(parser.parse_args())

    if not args["specialize"]:
        P.set(args["P"])
        M.set(args["M"])
        # M must always be specialized, as it's used for the static buffer size
        sdfg = make_sdfg(False)
        sdfg.specialize(dict(P=P, M=M))
        N.set(args["N"])
        K.set(args["K"])
    else:
        P.set(args["P"])
        M.set(args["M"])
        N.set(args["N"])
        K.set(args["K"])
        sdfg = make_sdfg(True)
        sdfg.specialize(dict(P=P, M=M, N=N, K=K))

    print("Matrix multiplication {}x{}x{} with {} PEs ({}specialized)".format(
        M.get(), N.get(), K.get(), P.get(),
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
    np.dot(A_regression, B_regression, C_regression)

    diff = np.abs(C_regression - C)
    diff_total = np.sum(diff)
    highest_diff = np.max(diff)
    wrong_elements = np.transpose(np.nonzero(diff >= 0.01))

    print("==== Program end ====")

    if diff_total >= 0.01:
        print("Verification failed!")
        print("Total difference: {}".format(diff_total))
        print("Incorrect elements: {} / {}".format(wrong_elements.shape[0],
                                                   N.get() * M.get()))
        print("Highest difference: {}".format(highest_diff))
        print("** Result:\n", C)
        print("** Reference:\n", C_regression)
        print("Type \"debug\" to enter debugger, "
              "or any other string to quit (timeout in 10 seconds)")
        read, _, _ = select.select([sys.stdin], [], [], 10)
        if len(read) > 0 and sys.stdin.readline().strip().lower() == "debug":
            print("Entering debugger...")
            pdb.set_trace()
        else:
            print("Exiting...")
        exit(1)
    else:
        print("Results verified successfully.")
    exit(0)
