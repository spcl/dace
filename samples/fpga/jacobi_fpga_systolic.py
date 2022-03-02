# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def make_init_state(sdfg):
    state = sdfg.add_state("init")

    a0 = state.add_read("A")
    tmp0 = state.add_write("tmp")
    state.add_memlet_path(a0, tmp0, memlet=dace.Memlet.simple(tmp0, "0, 0:H, 0:W"))

    a1 = state.add_read("A")
    tmp1 = state.add_write("tmp")
    state.add_memlet_path(a1, tmp1, memlet=dace.Memlet.simple(tmp1, "1, 0:H, 0:W"))

    return state


def make_finalize_state(sdfg, even):
    state = sdfg.add_state("finalize_" + ("even" if even else "odd"))

    tmp = state.add_read("tmp")
    a = state.add_write("A")
    state.add_memlet_path(tmp, a, memlet=dace.Memlet.simple(tmp, "{}, 0:H, 0:W".format(0 if even else 1)))

    return state


def make_compute_sdfg():
    sdfg = dace.SDFG("compute")

    pre_shift = sdfg.add_state("pre_shift")
    loop_body = sdfg.add_state("compute_body")
    post_shift = sdfg.add_state("post_shift")

    sdfg.add_edge(pre_shift, loop_body, dace.sdfg.InterstateEdge())
    sdfg.add_edge(loop_body, post_shift, dace.sdfg.InterstateEdge())

    sdfg.add_stream("stream_in", dtype, storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("stream_out", dtype, storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_array("row_buffers", (2, W), dtype, storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_array("sliding_window", (3, 3), dtype, storage=dace.dtypes.StorageType.FPGA_Registers)

    stream_in = pre_shift.add_read("stream_in")
    stream_out = loop_body.add_write("stream_out")

    rows_in = pre_shift.add_read("row_buffers")
    rows_out = post_shift.add_write("row_buffers")

    window_buffer_in = post_shift.add_read("sliding_window")
    window_buffer_out = pre_shift.add_write("sliding_window")
    window_compute_in = loop_body.add_read("sliding_window")
    window_shift_in = post_shift.add_read("sliding_window")
    window_shift_out = post_shift.add_write("sliding_window")

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
                              memlet=dace.Memlet.simple(window_compute_in, "0:3, 0:3"))

    # Output result (conditional write)
    out_memlet = dace.Memlet.simple(stream_out, "0", num_accesses=-1)
    loop_body.add_memlet_path(tasklet, stream_out, src_conn="result", memlet=out_memlet)

    # Read row buffer
    read_row_memlet = dace.Memlet.simple(rows_in, "0:2, x", num_accesses=2, other_subset_str="0:2, 2")
    pre_shift.add_memlet_path(rows_in, window_buffer_out, memlet=read_row_memlet)

    # Read from memory
    read_memory_memlet = dace.Memlet(f"{stream_in.data}[0]", dynamic=True)
    read_memory_tasklet = pre_shift.add_tasklet("skip_last", {"read"}, {"window_buffer"},
                                                "if y < H - 1 and x < W - 1:\n\twindow_buffer = read")
    pre_shift.add_memlet_path(stream_in, read_memory_tasklet, memlet=read_memory_memlet, dst_conn="read")
    pre_shift.add_memlet_path(read_memory_tasklet,
                              window_buffer_out,
                              memlet=dace.Memlet.simple(window_buffer_out, "2, 2"),
                              src_conn="window_buffer")

    # Shift window
    shift_window_memlet = dace.Memlet.simple(window_shift_in, '0:3, 1:3', other_subset_str='0:3, 0:2')
    post_shift.add_memlet_path(window_shift_in, window_shift_out, memlet=shift_window_memlet)

    # To row buffer
    write_row_memlet = dace.Memlet.simple(window_buffer_in, '1:3, 2', other_subset_str='0:2, x')
    post_shift.add_memlet_path(window_buffer_in, rows_out, memlet=write_row_memlet)

    return sdfg


def make_outer_compute_state(sdfg):
    state = sdfg.add_state("fpga_outer_state")

    tmp_in = state.add_read("tmp")
    pipes_memory_read = state.add_stream("pipes",
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
    pipes_memory_write = state.add_stream("pipes",
                                          dtype,
                                          1,
                                          transient=True,
                                          shape=(P + 1, ),
                                          storage=dace.dtypes.StorageType.FPGA_Local)

    # Read memory
    read_entry, read_exit = state.add_map("read", {
        "t": "0:T/P",
        "y": "1:H-1",
        "x": "1:W-1"
    },
                                          schedule=dace.ScheduleType.FPGA_Device)
    read_tasklet = state.add_tasklet("read", {"mem"}, {"to_kernel"}, "to_kernel = mem")
    state.add_memlet_path(tmp_in,
                          read_entry,
                          read_tasklet,
                          dst_conn="mem",
                          memlet=dace.Memlet(f"{tmp_in.data}[t % 2, y, x]"))
    state.add_memlet_path(read_tasklet,
                          read_exit,
                          pipes_memory_write,
                          src_conn="to_kernel",
                          memlet=dace.Memlet(f"{pipes_memory_write.data}[0]")),

    # Compute
    compute_sdfg = make_compute_sdfg()
    compute_sdfg_node = state.add_nested_sdfg(compute_sdfg, sdfg, {"stream_in", "sliding_window", "row_buffers"},
                                              {"stream_out", "sliding_window", "row_buffers"})
    systolic_entry, systolic_exit = state.add_map("unroll_compute", {"p": "0:P"},
                                                  schedule=dace.ScheduleType.FPGA_Device,
                                                  unroll=True)
    state.add_memlet_path(systolic_entry, pipes_read, memlet=dace.Memlet())
    sdfg.add_array("_sliding_window", (3, 3), dtype, transient=True, storage=dace.dtypes.StorageType.FPGA_Registers)
    sliding_window_read = state.add_read("_sliding_window")
    sliding_window_write = state.add_write("_sliding_window")
    sdfg.add_array("_row_buffers", (2, W), dtype, storage=dace.dtypes.StorageType.FPGA_Local, transient=True)
    row_buffers_read = state.add_read("_row_buffers")
    row_buffers_write = state.add_write("_row_buffers")
    compute_entry, compute_exit = state.add_map("compute", {
        "t": "0:T/P",
        "y": "1:H",
        "x": "1:W"
    },
                                                schedule=dace.ScheduleType.FPGA_Device)
    state.add_memlet_path(pipes_read,
                          compute_entry,
                          compute_sdfg_node,
                          dst_conn="stream_in",
                          memlet=dace.Memlet(f"{pipes_read.data}[p]", dynamic=True))
    state.add_memlet_path(systolic_entry, sliding_window_read, memlet=dace.Memlet())
    state.add_memlet_path(sliding_window_read,
                          compute_entry,
                          compute_sdfg_node,
                          dst_conn="sliding_window",
                          memlet=dace.Memlet(f"_sliding_window[0:3, 0:3]"))
    state.add_memlet_path(sliding_window_write, systolic_exit, memlet=dace.Memlet())
    state.add_memlet_path(compute_sdfg_node,
                          compute_exit,
                          sliding_window_write,
                          src_conn="sliding_window",
                          memlet=dace.Memlet(f"_sliding_window[0:3, 0:3]"))
    state.add_memlet_path(systolic_entry, row_buffers_read, memlet=dace.Memlet())
    state.add_memlet_path(row_buffers_read,
                          compute_entry,
                          compute_sdfg_node,
                          dst_conn="row_buffers",
                          memlet=dace.Memlet(f"_row_buffers[0:2, 0:W]"))
    state.add_memlet_path(row_buffers_write, systolic_exit, memlet=dace.Memlet())
    state.add_memlet_path(compute_sdfg_node,
                          compute_exit,
                          row_buffers_write,
                          src_conn="row_buffers",
                          memlet=dace.Memlet(f"_row_buffers[0:2, 0:W]"))
    state.add_memlet_path(compute_sdfg_node,
                          compute_exit,
                          pipes_write,
                          src_conn="stream_out",
                          memlet=dace.Memlet(f"{pipes_write.data}[p + 1]", dynamic=True))
    state.add_memlet_path(pipes_write, systolic_exit, memlet=dace.Memlet())

    # Write memory
    write_entry, write_exit = state.add_map("write", {
        "t": "0:T/P",
        "y": "1:H-1",
        "x": "1:W-1"
    },
                                            schedule=dace.ScheduleType.FPGA_Device)
    write_tasklet = state.add_tasklet("write", {"from_kernel"}, {"mem"}, "mem = from_kernel")
    tmp_out = state.add_write("tmp")
    state.add_memlet_path(pipes_memory_read,
                          write_entry,
                          write_tasklet,
                          dst_conn="from_kernel",
                          memlet=dace.Memlet(f"{pipes_memory_read}[P]"))
    state.add_memlet_path(write_tasklet,
                          write_exit,
                          tmp_out,
                          src_conn="mem",
                          memlet=dace.Memlet(f"{tmp_out.data}[1 - t % 2, y, x]"))

    return state


def make_sdfg(specialize_all):
    name = "jacobi_fpga_systolic_{}_{}x{}x{}".format(P.get(), ("H" if not specialize_all else H.get()), W.get(),
                                                     ("T" if not specialize_all else T.get()))

    sdfg = dace.SDFG(name)
    sdfg.add_symbol('T', dace.int32)

    sdfg.add_array("A", (H, W), dtype)
    sdfg.add_array("tmp", (2, H, W), dtype, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)

    init_state = make_init_state(sdfg)

    fpga_state = make_outer_compute_state(sdfg)

    finalize_even = make_finalize_state(sdfg, True)
    finalize_odd = make_finalize_state(sdfg, False)

    sdfg.add_edge(init_state, fpga_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(
        fpga_state, finalize_even,
        dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
            "(T / P) % 2 == 0", language=dace.dtypes.Language.Python)))
    sdfg.add_edge(
        fpga_state, finalize_odd,
        dace.sdfg.InterstateEdge(condition=dace.properties.CodeProperty.from_string(
            "(T / P) % 2 == 1", language=dace.dtypes.Language.Python)))

    return sdfg


def run_jacobi(w: int, h: int, t: int, p: int, specialize_all: bool = False):
    print("==== Program start ====")

    # Width and number of PEs must be known at compile time, as it will
    # influence the hardware layout
    W.set(w)
    P.set(p)
    if specialize_all:
        print("Specializing H and T...")
        H.set(h)
        T.set(t)

    jacobi = make_sdfg(specialize_all)
    jacobi.specialize(dict(W=W, P=P))

    if not specialize_all:
        H.set(h)
        T.set(t)
    else:
        jacobi.specialize(dict(H=H, T=T))

    if T.get() % P.get() != 0:
        raise ValueError("Iteration must be divisable by number of processing elements")

    print("Jacobi Stencil {}x{} ({} steps) with {} PEs{}".format(H.get(), W.get(), T.get(), P.get(),
                                                                 (" (fully specialized)" if specialize_all else "")))

    A = dace.ndarray([H, W], dtype=dace.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    A[2:H.get() - 2, 2:W.get() - 2] = 1
    regression = np.ndarray([H.get() - 4, W.get() - 4], dtype=np.float32)
    regression[:] = A[2:H.get() - 2, 2:W.get() - 2]

    #############################################
    # Run DaCe program

    if specialize_all:
        jacobi(A=A)
    else:
        jacobi(A=A, H=H, T=T)

    # Regression
    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]], dtype=np.float32)
    for i in range(T.get()):
        regression = ndimage.convolve(regression, kernel, mode='constant', cval=0.0)

    residual = np.linalg.norm(A[2:H.get() - 2, 2:W.get() - 2] - regression) / (H.get() * W.get())
    print("Residual:", residual)
    diff = np.abs(A[2:H.get() - 2, 2:W.get() - 2] - regression)
    wrong_elements = np.transpose(np.nonzero(diff >= 0.01))
    highest_diff = np.max(diff)

    print("==== Program end ====")
    if residual >= 0.01 or highest_diff >= 0.01:
        print("Verification failed!")
        print("Residual: {}".format(residual))
        print("Incorrect elements: {} / {}".format(wrong_elements.shape[0], H.get() * W.get()))
        print("Highest difference: {}".format(highest_diff))
        print("** Result:\n", A[:min(6, H.get()), :min(6, W.get())])
        print("** Reference:\n", regression[:min(5, H.get()), :min(4, W.get())])
        raise RuntimeError("Validation failed.")

    return jacobi


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("H", type=int, nargs="?", default=64)
    parser.add_argument("W", type=int, nargs="?", default=8192)
    parser.add_argument("T", type=int, nargs="?", default=16)
    parser.add_argument("P", type=int, nargs="?", default=8)
    parser.add_argument("-specialize_all",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = parser.parse_args()

    run_jacobi(args.H, args.W, args.T, args.P, args.specialize_all)
