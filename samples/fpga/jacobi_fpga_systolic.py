# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np
import select
import sys
from scipy import ndimage

W = dace.symbol("W")
H = dace.symbol("H")
T = dace.symbol("T")
P = dace.symbol("P")  # Number of processing elements
dtype = dace.float32


def add_tmp(state):
    return state.add_array("tmp", (2, H, W),
                           dtype,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Global)


def make_init_state(sdfg):
    state = sdfg.add_state("init")

    a0 = state.add_array("A", (H, W), dtype)
    tmp0 = add_tmp(state)
    state.add_memlet_path(a0,
                          tmp0,
                          memlet=dace.memlet.Memlet.simple(tmp0, "0, 0:H, 0:W"))

    a1 = state.add_array("A", (H, W), dtype)
    tmp1 = add_tmp(state)
    state.add_memlet_path(a1,
                          tmp1,
                          memlet=dace.memlet.Memlet.simple(tmp1, "1, 0:H, 0:W"))

    return state


def make_finalize_state(sdfg, even):
    state = sdfg.add_state("finalize_" + ("even" if even else "odd"))

    tmp = add_tmp(state)
    a = state.add_array("A", (H, W), dtype)
    state.add_memlet_path(tmp,
                          a,
                          memlet=dace.memlet.Memlet.simple(
                              tmp, "{}, 0:H, 0:W".format(0 if even else 1)))

    return state


def make_compute_sdfg():
    sdfg = dace.SDFG("compute")

    time_begin = sdfg.add_state("time_begin")
    time_entry = sdfg.add_state("time_entry")
    time_end = sdfg.add_state("time_end")

    y_begin = sdfg.add_state("y_begin")
    y_entry = sdfg.add_state("y_entry")
    y_end = sdfg.add_state("y_end")

    x_begin = sdfg.add_state("x_begin")
    x_entry = sdfg.add_state("x_entry")
    x_end = sdfg.add_state("x_end")

    pre_shift = sdfg.add_state("pre_shift")
    loop_body = sdfg.add_state("compute_body")
    post_shift = sdfg.add_state("post_shift")

    sdfg.add_edge(time_begin, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": 0}))
    sdfg.add_edge(y_begin, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": 1}))
    sdfg.add_edge(x_begin, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": 1}))

    sdfg.add_edge(
        time_entry, y_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t < T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, x_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y < H", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, pre_shift,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x < W", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(y_end, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": "t + 1"}))
    sdfg.add_edge(x_end, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": "y + 1"}))
    sdfg.add_edge(pre_shift, loop_body, dace.sdfg.InterstateEdge())
    sdfg.add_edge(loop_body, post_shift, dace.sdfg.InterstateEdge())
    sdfg.add_edge(post_shift, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": "x + 1"}))

    sdfg.add_edge(
        time_entry, time_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t >= T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, y_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y >= H", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, x_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x >= W", language=dace.dtypes.Language.Python)))

    stream_in = pre_shift.add_stream(
        "stream_in", dtype, 1, storage=dace.dtypes.StorageType.FPGA_Global)
    stream_out = loop_body.add_stream(
        "stream_out", dtype, 1, storage=dace.dtypes.StorageType.FPGA_Global)

    rows_in = pre_shift.add_array("row_buffers", (2, W),
                                  dtype,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local,
                                  lifetime=dace.dtypes.AllocationLifetime.SDFG)
    rows_out = post_shift.add_array(
        "row_buffers", (2, W),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)

    window_buffer_in = post_shift.add_array(
        "sliding_window", (3, 3),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)
    window_buffer_out = pre_shift.add_array(
        "sliding_window", (3, 3),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)
    window_compute_in = loop_body.add_array(
        "sliding_window", (3, 3),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)
    window_shift_in = post_shift.add_array(
        "sliding_window", (3, 3),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)
    window_shift_out = post_shift.add_array(
        "sliding_window", (3, 3),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers,
        lifetime=dace.dtypes.AllocationLifetime.SDFG)

    code = """\
res = 0.0
if y >= 3 and x >= 3 and y < H - 1 and x < W - 1:
    res = float(0.2) * (window[0, 1] + window[1, 0] + window[1, 1] + window[1, 2] + window[2, 1])
elif y >= 2 and x >= 2:
    res = window[1, 1]
if (y >= 3 and x >= 3 and y < H - 1 and x < W - 1) or (y >= 2 and x >= 2):
    result = res"""

    tasklet = loop_body.add_tasklet("compute", {"window"}, {"result"}, code)

    # Input window
    loop_body.add_memlet_path(window_compute_in,
                              tasklet,
                              dst_conn="window",
                              memlet=dace.memlet.Memlet.simple(
                                  window_compute_in, "0:3, 0:3"))

    # Output result (conditional write)
    out_memlet = dace.memlet.Memlet.simple(stream_out, "0", num_accesses=-1)
    loop_body.add_memlet_path(tasklet,
                              stream_out,
                              src_conn="result",
                              memlet=out_memlet)

    # Read row buffer
    read_row_memlet = dace.memlet.Memlet.simple(rows_in,
                                                "0:2, x",
                                                num_accesses=2,
                                                other_subset_str="0:2, 2")
    pre_shift.add_memlet_path(rows_in,
                              window_buffer_out,
                              memlet=read_row_memlet)

    # Read from memory
    read_memory_memlet = dace.memlet.Memlet.simple(stream_in,
                                                   "0",
                                                   num_accesses=-1)
    read_memory_tasklet = pre_shift.add_tasklet(
        "skip_last", {"read"}, {"window_buffer"},
        "if y < H - 1 and x < W - 1:\n\twindow_buffer = read")
    pre_shift.add_memlet_path(stream_in,
                              read_memory_tasklet,
                              memlet=read_memory_memlet,
                              dst_conn="read")
    pre_shift.add_memlet_path(read_memory_tasklet,
                              window_buffer_out,
                              memlet=dace.memlet.Memlet.simple(
                                  window_buffer_out, "2, 2"),
                              src_conn="window_buffer")

    # Shift window
    shift_window_memlet = dace.memlet.Memlet.simple(window_shift_in,
                                                    '0:3, 1:3',
                                                    other_subset_str='0:3, 0:2')
    post_shift.add_memlet_path(window_shift_in,
                               window_shift_out,
                               memlet=shift_window_memlet)

    # To row buffer
    write_row_memlet = dace.memlet.Memlet.simple(window_buffer_in,
                                                 '1:3, 2',
                                                 other_subset_str='0:2, x')
    post_shift.add_memlet_path(window_buffer_in,
                               rows_out,
                               memlet=write_row_memlet)

    return sdfg


def make_read_sdfg():
    sdfg = dace.SDFG("read_memory_sdfg")

    time_begin = sdfg.add_state("time_begin")
    time_entry = sdfg.add_state("time_entry")
    time_end = sdfg.add_state("time_end")

    y_begin = sdfg.add_state("y_begin")
    y_entry = sdfg.add_state("y_entry")
    y_end = sdfg.add_state("y_end")

    x_begin = sdfg.add_state("x_begin")
    x_entry = sdfg.add_state("x_entry")
    x_end = sdfg.add_state("x_end")

    loop_body = sdfg.add_state("read_memory")

    sdfg.add_edge(time_begin, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": 0}))
    sdfg.add_edge(y_begin, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": 1}))
    sdfg.add_edge(x_begin, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": 1}))

    sdfg.add_edge(
        time_entry, y_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t < T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, x_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y < H - 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x < W - 1", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(y_end, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": "t + 1"}))
    sdfg.add_edge(x_end, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": "y + 1"}))
    sdfg.add_edge(loop_body, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": "x + 1"}))

    sdfg.add_edge(
        time_entry, time_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t >= T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, y_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y >= H - 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, x_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x >= W - 1", language=dace.dtypes.Language.Python)))

    mem_read = loop_body.add_array("mem_read", (2, H, W),
                                   dtype,
                                   storage=dace.dtypes.StorageType.FPGA_Global)
    pipe = loop_body.add_stream("pipe",
                                dtype,
                                1,
                                storage=dace.dtypes.StorageType.FPGA_Global)

    # Read from memory
    read_memory_memlet = dace.memlet.Memlet.simple(mem_read,
                                                   "t%2, y, x",
                                                   other_subset_str="0")
    loop_body.add_memlet_path(mem_read, pipe, memlet=read_memory_memlet)

    return sdfg


def make_write_sdfg():
    sdfg = dace.SDFG("write_memory_sdfg")

    time_begin = sdfg.add_state("time_begin")
    time_entry = sdfg.add_state("time_entry")
    time_end = sdfg.add_state("time_end")

    y_begin = sdfg.add_state("y_begin")
    y_entry = sdfg.add_state("y_entry")
    y_end = sdfg.add_state("y_end")

    x_begin = sdfg.add_state("x_begin")
    x_entry = sdfg.add_state("x_entry")
    x_end = sdfg.add_state("x_end")

    loop_body = sdfg.add_state("write_memory")

    sdfg.add_edge(time_begin, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": 0}))
    sdfg.add_edge(y_begin, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": 1}))
    sdfg.add_edge(x_begin, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": 1}))

    sdfg.add_edge(
        time_entry, y_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t < T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, x_begin,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y < H - 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x < W - 1", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(y_end, time_entry,
                  dace.sdfg.InterstateEdge(assignments={"t": "t + 1"}))
    sdfg.add_edge(x_end, y_entry,
                  dace.sdfg.InterstateEdge(assignments={"y": "y + 1"}))
    sdfg.add_edge(loop_body, x_entry,
                  dace.sdfg.InterstateEdge(assignments={"x": "x + 1"}))

    sdfg.add_edge(
        time_entry, time_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "t >= T / P", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        y_entry, y_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "y >= H - 1", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        x_entry, x_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "x >= W - 1", language=dace.dtypes.Language.Python)))

    pipe = loop_body.add_stream("pipe",
                                dtype,
                                1,
                                storage=dace.dtypes.StorageType.FPGA_Global)
    mem_write = loop_body.add_array("mem_write", (2, H, W),
                                    dtype,
                                    storage=dace.dtypes.StorageType.FPGA_Global)

    # Read from memory
    write_memory_memlet = dace.memlet.Memlet.simple(
        pipe, '0', other_subset_str="1 - t%2, y, x")
    loop_body.add_memlet_path(pipe, mem_write, memlet=write_memory_memlet)

    return sdfg


def make_outer_compute_state(sdfg):
    state = sdfg.add_state("fpga_outer_state")

    tmp_in = add_tmp(state)
    pipes_memory_read = state.add_stream(
        "pipes",
        dtype,
        1,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)
    pipes_read = state.add_stream("pipes",
                                  dtype,
                                  1,
                                  transient=True,
                                  shape=(P + 1, ),
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    pipes_write = state.add_stream("pipes",
                                   dtype,
                                   1,
                                   transient=True,
                                   shape=(P + 1, ),
                                   storage=dace.dtypes.StorageType.FPGA_Local)
    pipes_memory_write = state.add_stream(
        "pipes",
        dtype,
        1,
        transient=True,
        shape=(P + 1, ),
        storage=dace.dtypes.StorageType.FPGA_Local)

    read_sdfg = make_read_sdfg()
    read_sdfg_node = state.add_nested_sdfg(read_sdfg, sdfg, {"mem_read"},
                                           {"pipe"})
    compute_sdfg = make_compute_sdfg()
    compute_sdfg_node = state.add_nested_sdfg(compute_sdfg, sdfg, {"stream_in"},
                                              {"stream_out"})
    write_sdfg = make_write_sdfg()
    write_sdfg_node = state.add_nested_sdfg(write_sdfg, sdfg, {"pipe"},
                                            {"mem_write"})

    tmp_out = add_tmp(state)

    state.add_memlet_path(tmp_in,
                          read_sdfg_node,
                          dst_conn="mem_read",
                          memlet=dace.memlet.Memlet.simple(
                              tmp_in, "0:2, 0:H, 0:W"))
    state.add_memlet_path(
        read_sdfg_node,
        pipes_memory_write,
        src_conn="pipe",
        memlet=dace.memlet.Memlet.simple(
            pipes_memory_write,
            '0',
            num_accesses=dace.symbolic.pystr_to_symbolic("(T/P)*H*W")))

    compute_entry, compute_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)
    state.add_memlet_path(compute_entry,
                          pipes_read,
                          memlet=dace.memlet.Memlet())
    state.add_memlet_path(
        pipes_read,
        compute_sdfg_node,
        dst_conn="stream_in",
        memlet=dace.memlet.Memlet.simple(
            pipes_read,
            'p',
            num_accesses=dace.symbolic.pystr_to_symbolic("(T/P)*H*W")))
    state.add_memlet_path(
        compute_sdfg_node,
        pipes_write,
        src_conn="stream_out",
        memlet=dace.memlet.Memlet.simple(
            pipes_write,
            'p + 1',
            num_accesses=dace.symbolic.pystr_to_symbolic("(T/P)*H*W")))
    state.add_memlet_path(pipes_write,
                          compute_exit,
                          memlet=dace.memlet.Memlet())

    state.add_memlet_path(
        pipes_memory_read,
        write_sdfg_node,
        dst_conn="pipe",
        memlet=dace.memlet.Memlet.simple(
            pipes_memory_read,
            'P',
            num_accesses=dace.symbolic.pystr_to_symbolic("(T/P)*H*W")))
    state.add_memlet_path(write_sdfg_node,
                          tmp_out,
                          src_conn="mem_write",
                          memlet=dace.memlet.Memlet.simple(
                              tmp_out, "0:2, 0:H, 0:W"))

    return state


def make_sdfg(specialize_all):
    name = "jacobi_fpga_systolic_{}_{}x{}x{}".format(
        P.get(), ("H" if not specialize_all else H.get()), W.get(),
        ("T" if not specialize_all else T.get()))

    sdfg = dace.SDFG(name)
    sdfg.add_symbol('T', dace.int32)
    init_state = make_init_state(sdfg)

    fpga_state = make_outer_compute_state(sdfg)

    finalize_even = make_finalize_state(sdfg, True)
    finalize_odd = make_finalize_state(sdfg, False)

    sdfg.add_edge(init_state, fpga_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(
        fpga_state, finalize_even,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "(T / P) % 2 == 0", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        fpga_state, finalize_odd,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "(T / P) % 2 == 1", language=dace.dtypes.Language.Python)))

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("H", type=int, nargs="?", default=64)
    parser.add_argument("W", type=int, nargs="?", default=8192)
    parser.add_argument("T", type=int, nargs="?", default=16)
    parser.add_argument("P", type=int, nargs="?", default=8)
    parser.add_argument("-specialize_all",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = vars(parser.parse_args())

    # Width and number of PEs must be known at compile time, as it will
    # influence the hardware layout
    W.set(args["W"])
    P.set(args["P"])
    if args["specialize_all"]:
        print("Specializing H and T...")
        H.set(args["H"])
        T.set(args["T"])

    jacobi = make_sdfg(args["specialize_all"])
    jacobi.specialize(dict(W=W, P=P))

    if not args["specialize_all"]:
        H.set(args["H"])
        T.set(args["T"])
    else:
        jacobi.specialize(dict(H=H, T=T))

    if T.get() % P.get() != 0:
        raise ValueError(
            "Iteration must be divisable by number of processing elements")

    print("Jacobi Stencil {}x{} ({} steps) with {} PEs{}".format(
        H.get(), W.get(), T.get(), P.get(),
        (" (fully specialized)" if args["specialize_all"] else "")))

    A = dace.ndarray([H, W], dtype=dace.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    A[2:H.get() - 2, 2:W.get() - 2] = 1
    regression = np.ndarray([H.get() - 4, W.get() - 4], dtype=np.float32)
    regression[:] = A[2:H.get() - 2, 2:W.get() - 2]

    #############################################
    # Run DaCe program

    if args["specialize_all"]:
        jacobi(A=A)
    else:
        jacobi(A=A, H=H, T=T)

    # Regression
    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]],
                      dtype=np.float32)
    for i in range(T.get()):
        regression = ndimage.convolve(regression,
                                      kernel,
                                      mode='constant',
                                      cval=0.0)

    residual = np.linalg.norm(A[2:H.get() - 2, 2:W.get() - 2] -
                              regression) / (H.get() * W.get())
    print("Residual:", residual)
    diff = np.abs(A[2:H.get() - 2, 2:W.get() - 2] - regression)
    wrong_elements = np.transpose(np.nonzero(diff >= 0.01))
    highest_diff = np.max(diff)

    print("==== Program end ====")
    if residual >= 0.01 or highest_diff >= 0.01:
        print("Verification failed!")
        print("Residual: {}".format(residual))
        print("Incorrect elements: {} / {}".format(wrong_elements.shape[0],
                                                   H.get() * W.get()))
        print("Highest difference: {}".format(highest_diff))
        print("** Result:\n", A[:min(6, H.get()), :min(6, W.get())])
        print("** Reference:\n", regression[:min(4, H.get()), :min(4, W.get())])
        print("Type \"debug\" to enter debugger, "
              "or any other string to quit (timeout in 10 seconds)")
        read, _, _ = select.select([sys.stdin], [], [], 10)
        if len(read) > 0 and sys.stdin.readline().strip().lower() == "debug":
            print("Entering debugger...")
            import pdb

            pdb.set_trace()
        else:
            print("Exiting...")
        exit(1)
    exit(0)
