import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    A_host = state.add_array("A", [N, K], dtype=dace.float32)
    B_host = state.add_array("B", [K, M], dtype=dace.float32)
    C_host = state.add_array("C", [N, M], dtype=dace.float32)

    A_device = state.add_array("A_device", [N, K],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    B_device = state.add_array("B_device", [K, M],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    C_device = state.add_array("C_device", [N, M],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_edge(A_host, None, A_device, None,
                   dace.memlet.Memlet.simple(A_device, "0:N, 0:K"))
    state.add_edge(B_host, None, B_device, None,
                   dace.memlet.Memlet.simple(B_device, "0:K, 0:M"))
    state.add_edge(C_host, None, C_device, None,
                   dace.memlet.Memlet.simple(C_device, "0:N, 0:M"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_array("C_device", [N, M],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    C_host = state.add_array("C", [N, M], dtype=dace.float32)

    state.add_edge(C_device, None, C_host, None,
                   dace.memlet.Memlet.simple(C_host, "0:N, 0:M"))

    return state


def make_read_A_sdfg():

    sdfg = dace.SDFG("gemm_read_A")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    k_begin = sdfg.add_state("k_begin")
    k_entry = sdfg.add_state("k_entry")
    k_end = sdfg.add_state("k_end")

    loop_body = sdfg.add_state("read_memory")

    sdfg.add_edge(n_begin, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(k_begin, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": 0}))

    sdfg.add_edge(
        n_entry, k_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k < K", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(k_end, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": "n + 1"}))
    sdfg.add_edge(loop_body, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": "k + 1"}))

    sdfg.add_edge(
        n_entry, n_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, k_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k >= K", language=dace.dtypes.Language.Python)))

    mem = loop_body.add_array("mem", [N, K],
                              dtype=dace.float32,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    pipe = loop_body.add_stream("pipe",
                                dace.float32,
                                storage=dace.dtypes.StorageType.FPGA_Local)

    loop_body.add_memlet_path(mem,
                              pipe,
                              memlet=dace.memlet.Memlet.simple(
                                  pipe, '0', other_subset_str="n, k"))

    return sdfg


def make_read_B_sdfg():

    sdfg = dace.SDFG("gemm_read_B")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    k_begin = sdfg.add_state("k_begin")
    k_entry = sdfg.add_state("k_entry")
    k_end = sdfg.add_state("k_end")

    m_begin = sdfg.add_state("m_begin")
    m_entry = sdfg.add_state("m_entry")
    m_end = sdfg.add_state("m_end")

    loop_body = sdfg.add_state("read_memory")

    sdfg.add_edge(n_begin, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(k_begin, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": 0}))
    sdfg.add_edge(m_begin, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": 0}))

    sdfg.add_edge(
        n_entry, k_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, m_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k < K", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        m_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m < M", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(k_end, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": "n + 1"}))
    sdfg.add_edge(m_end, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(loop_body, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": "m + 1"}))

    sdfg.add_edge(
        n_entry, n_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, k_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k >= K", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        m_entry, m_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m >= M", language=dace.dtypes.Language.Python)))

    mem = loop_body.add_array("mem", [K, M],
                              dtype=dace.float32,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    pipe = loop_body.add_stream("pipe",
                                dace.float32,
                                storage=dace.dtypes.StorageType.FPGA_Local)

    loop_body.add_memlet_path(mem,
                              pipe,
                              memlet=dace.memlet.Memlet.simple(
                                  pipe, '0', other_subset_str="k, m"))

    return sdfg


def make_write_C_sdfg():

    sdfg = dace.SDFG("gemm_write_C")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    m_begin = sdfg.add_state("m_begin")
    m_entry = sdfg.add_state("m_entry")
    m_end = sdfg.add_state("m_end")

    loop_body = sdfg.add_state("write_memory")

    sdfg.add_edge(n_begin, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(m_begin, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": 0}))

    sdfg.add_edge(
        n_entry, m_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        m_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m < M", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(m_end, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": "n + 1"}))
    sdfg.add_edge(loop_body, m_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": "m + 1"}))

    sdfg.add_edge(
        n_entry, n_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        m_entry, m_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m >= M", language=dace.dtypes.Language.Python)))

    mem = loop_body.add_array("mem", [N, M],
                              dtype=dace.float32,
                              storage=dace.dtypes.StorageType.FPGA_Global)

    pipe = loop_body.add_stream("pipe",
                                dace.float32,
                                storage=dace.dtypes.StorageType.FPGA_Local)

    loop_body.add_memlet_path(pipe,
                              mem,
                              memlet=dace.memlet.Memlet.simple(
                                  mem, "n, m", other_subset_str="0"))

    return sdfg


def make_compute_sdfg():

    sdfg = dace.SDFG("gemm_compute")

    n_begin = sdfg.add_state("n_begin")
    n_entry = sdfg.add_state("n_entry")
    n_end = sdfg.add_state("n_end")

    k_begin = sdfg.add_state("k_begin")
    k_entry = sdfg.add_state("k_entry")
    k_end = sdfg.add_state("k_end")

    m_begin = sdfg.add_state("m_begin")
    m_entry = sdfg.add_state("m_entry")
    m_end = sdfg.add_state("m_end")

    c_begin = sdfg.add_state("c_begin")
    c_entry = sdfg.add_state("c_entry")
    c_end = sdfg.add_state("c_end")

    state = sdfg.add_state("compute")

    write_c_state = sdfg.add_state("write_c")

    # Data nodes
    A_pipe = state.add_stream("A_stream_in",
                              dace.float32,
                              storage=dace.dtypes.StorageType.FPGA_Local)
    B_pipe = state.add_stream("B_stream_in",
                              dace.float32,
                              storage=dace.dtypes.StorageType.FPGA_Local)
    C_pipe = write_c_state.add_stream(
        "C_stream_out",
        dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Local)

    # N-loop
    sdfg.add_edge(n_begin, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": 0}))
    sdfg.add_edge(
        n_entry, k_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n < N", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        n_entry, n_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "n >= N", language=dace.dtypes.Language.Python)))

    # K-loop
    sdfg.add_edge(k_begin, k_entry,
                  dace.sdfg.InterstateEdge(assignments={"k": 0}))
    sdfg.add_edge(
        k_entry, m_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k < K", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        k_entry, k_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "k >= K", language=dace.dtypes.Language.Python)))

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
    sdfg.add_edge(k_end, c_begin, dace.sdfg.InterstateEdge())

    # Tight C-loop
    sdfg.add_edge(c_begin, c_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": 0}))
    sdfg.add_edge(
        c_entry, write_c_state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m < M", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        c_entry, c_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m >= M", language=dace.dtypes.Language.Python)))

    # End inner loop
    sdfg.add_edge(write_c_state, c_entry,
                  dace.sdfg.InterstateEdge(assignments={"m": "m + 1"}))

    # Backtrack
    sdfg.add_edge(c_end, n_entry,
                  dace.sdfg.InterstateEdge(assignments={"n": "n + 1"}))

    C_buffer_in = state.add_array("C_buffer", [M],
                                  dtype=dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_out = state.add_array("C_buffer", [M],
                                   dtype=dace.float32,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Local)
    C_buffer_write = write_c_state.add_array(
        "C_buffer", [M],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)

    ###########################################################################
    # Nested SDFG

    nested_sdfg = dace.SDFG("gemm_nested")

    if_state_c = nested_sdfg.add_state("if_state_c")
    then_state_c = nested_sdfg.add_state("then_state_c")
    else_state_c = nested_sdfg.add_state("else_state_c")
    if_state_a = nested_sdfg.add_state("if_state_a")
    then_state_a = nested_sdfg.add_state("then_state_a")
    # No else state is necessary, but control flow detection seems to be broken
    # for ifs with no else branch
    else_state_a = nested_sdfg.add_state("else_state_a")
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
    nested_sdfg.add_edge(then_state_c, if_state_a, dace.sdfg.InterstateEdge())
    nested_sdfg.add_edge(else_state_c, if_state_a, dace.sdfg.InterstateEdge())
    nested_sdfg.add_edge(
        if_state_a, then_state_a,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m == 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(
        if_state_a, else_state_a,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "m != 0", language=dace.dtypes.Language.Python)))
    nested_sdfg.add_edge(then_state_a, compute_state,
                         dace.sdfg.InterstateEdge())
    nested_sdfg.add_edge(else_state_a, compute_state,
                         dace.sdfg.InterstateEdge())

    compute_tasklet = compute_state.add_tasklet("multiply_add",
                                                {"a", "b", "c_in"}, {"c_out"},
                                                "c_out = c_in + a * b")

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

    # Then state A
    A_in = then_state_a.add_scalar(
        "A_in",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    A_val_out = then_state_a.add_scalar(
        "A_val_out",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    then_state_a.add_memlet_path(A_in,
                                 A_val_out,
                                 memlet=dace.memlet.Memlet.simple(
                                     A_val_out, "0", num_accesses=-1))

    # Compute state
    A_val_in = compute_state.add_scalar(
        "A_val_in",
        dtype=dace.float32,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    B_in = compute_state.add_scalar(
        "B_in",
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
                                  dst_conn="a")
    compute_state.add_memlet_path(B_in,
                                  compute_tasklet,
                                  memlet=dace.memlet.Memlet.simple(B_in, "0"),
                                  dst_conn="b")
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
                                    {"A_in", "A_val_in", "B_in", "C_in"},
                                    {"A_val_out", "C_out"})

    ###########################################################################
    # Compute continued

    # Scalar buffer for A
    A_reg_in = state.add_scalar("A_reg",
                                dtype=dace.float32,
                                transient=True,
                                lifetime=dace.dtypes.AllocationLifetime.SDFG,
                                storage=dace.dtypes.StorageType.FPGA_Registers)
    A_reg_out = state.add_scalar("A_reg",
                                 dtype=dace.float32,
                                 transient=True,
                                 lifetime=dace.dtypes.AllocationLifetime.SDFG,
                                 storage=dace.dtypes.StorageType.FPGA_Registers)

    state.add_memlet_path(A_pipe,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(A_pipe,
                                                           "0",
                                                           num_accesses=-1),
                          dst_conn="A_in")
    state.add_memlet_path(B_pipe,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(B_pipe, "0"),
                          dst_conn="B_in")
    state.add_memlet_path(C_buffer_in,
                          tasklet,
                          memlet=dace.memlet.Memlet.simple(C_buffer_in, "m"),
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
    state.add_memlet_path(tasklet,
                          A_reg_out,
                          memlet=dace.memlet.Memlet.simple(A_reg_out,
                                                           "0",
                                                           num_accesses=-1),
                          src_conn="A_val_out")

    ###########################################################################
    # Write back state

    write_c_state.add_memlet_path(C_buffer_write,
                                  C_pipe,
                                  memlet=dace.memlet.Memlet.simple(
                                      C_pipe, "0", other_subset_str="m"))

    return sdfg


def make_fpga_state(sdfg):

    state = sdfg.add_state("gemm")

    read_A_sdfg = make_read_A_sdfg()
    read_A_sdfg_node = state.add_nested_sdfg(read_A_sdfg, sdfg, {"mem"},
                                             {"pipe"})

    read_B_sdfg = make_read_B_sdfg()
    read_B_sdfg_node = state.add_nested_sdfg(read_B_sdfg, sdfg, {"mem"},
                                             {"pipe"})

    compute_sdfg = make_compute_sdfg()
    compute_sdfg_node = state.add_nested_sdfg(compute_sdfg, sdfg,
                                              {"A_stream_in", "B_stream_in"},
                                              {"C_stream_out"})

    write_C_sdfg = make_write_C_sdfg()
    write_C_sdfg_node = state.add_nested_sdfg(write_C_sdfg, sdfg, {"pipe"},
                                              {"mem"})

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

    A_pipe_in = state.add_stream("A_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.dtypes.StorageType.FPGA_Local)
    B_pipe_in = state.add_stream("B_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.dtypes.StorageType.FPGA_Local)
    C_pipe_in = state.add_stream("C_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.dtypes.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream("A_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    C_pipe_out = state.add_stream("C_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)

    state.add_memlet_path(A,
                          read_A_sdfg_node,
                          dst_conn="mem",
                          memlet=dace.memlet.Memlet.simple(A, "0:N, 0:K"))
    state.add_memlet_path(
        read_A_sdfg_node,
        A_pipe_out,
        src_conn="pipe",
        memlet=dace.memlet.Memlet.simple(
            A_pipe_out,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K")))
    state.add_memlet_path(
        A_pipe_in,
        compute_sdfg_node,
        dst_conn="A_stream_in",
        memlet=dace.memlet.Memlet.simple(
            A_pipe_in,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K")))

    state.add_memlet_path(
        B,
        read_B_sdfg_node,
        dst_conn="mem",
        memlet=dace.memlet.Memlet.simple(
            B,
            "0:K, 0:M",
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K * M")))
    state.add_memlet_path(
        read_B_sdfg_node,
        B_pipe_out,
        src_conn="pipe",
        memlet=dace.memlet.Memlet.simple(
            B_pipe_out,
            "0",
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K * M")))
    state.add_memlet_path(
        B_pipe_in,
        compute_sdfg_node,
        dst_conn="B_stream_in",
        memlet=dace.memlet.Memlet.simple(
            B_pipe_in,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * K * M")))

    state.add_memlet_path(write_C_sdfg_node,
                          C,
                          src_conn="mem",
                          memlet=dace.memlet.Memlet.simple(C, "0:N, 0:M"))
    state.add_memlet_path(
        C_pipe_in,
        write_C_sdfg_node,
        dst_conn="pipe",
        memlet=dace.memlet.Memlet.simple(
            C_pipe_in,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * M")))
    state.add_memlet_path(
        compute_sdfg_node,
        C_pipe_out,
        src_conn="C_stream_out",
        memlet=dace.memlet.Memlet.simple(
            C_pipe_out,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("N * M")))

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("gemm_fpga_stream_{}x{}x{}".format(
            N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("gemm_fpga_stream_NxKx{}".format(M.get()))

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

    print("Matrix multiplication {}x{}x{} ({}specialized)".format(
        M.get(), N.get(), K.get(), "" if args["specialize"] else "not "))

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
