#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

from dace import SDFG, Memlet, EmptyMemlet
from dace.dtypes import StorageType, ScheduleType
from dace.subsets import Indices

W = dace.symbol("W")
H = dace.symbol("H")
P = dace.symbol("P")
num_bins = dace.symbol("num_bins")
num_bins.set(256)
dtype = dace.float32
itype = dace.uint32


def make_copy_to_fpga_state(sdfg):

    state = sdfg.add_state("copy_to_fpga")

    a_host = state.add_array("A", (H, W), dtype)
    a_device = state.add_array(
        "A_device", (H, W),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    hist_host = state.add_array("hist", (num_bins, ), itype)
    hist_device = state.add_array(
        "hist_device", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(
        a_host,
        a_device,
        memlet=dace.memlet.Memlet.simple(a_device, "0:H, 0:W", veclen=P.get()))
    state.add_memlet_path(
        hist_host,
        hist_device,
        memlet=dace.memlet.Memlet.simple(hist_device, "0:num_bins"))

    return state


def make_copy_to_host_state(sdfg):

    state = sdfg.add_state("copy_to_host")

    hist_device = state.add_array(
        "hist_device", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    hist_host = state.add_array("hist", (num_bins, ), itype)

    state.add_memlet_path(
        hist_device,
        hist_host,
        memlet=dace.memlet.Memlet.simple(
            hist_host, "0:num_bins", veclen=P.get()))

    return state


def make_compute_state(sdfg):

    state = sdfg.add_state("histogram_fpga")

    a = state.add_stream(
        "A_pipe_in", dtype, storage=dace.dtypes.StorageType.FPGA_Local)
    hist = state.add_array(
        "hist_buffer", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)

    entry, exit = state.add_map(
        "map", {
            "h": "0:H",
            "w": "0:W:P"
        }, schedule=ScheduleType.FPGA_Device)

    tasklet = state.add_tasklet("compute", {"a"}, {"out"},
                                "out[int(float(num_bins) * a)] = 1")

    read_memlet = dace.memlet.Memlet.simple(a, "0")
    write_memlet = dace.memlet.Memlet.simple(
        hist,
        "0:num_bins",
        wcr_str="lambda a, b: a + b",
        wcr_identity=0,
        num_accesses=1)

    state.add_memlet_path(a, entry, tasklet, memlet=read_memlet, dst_conn="a")
    state.add_memlet_path(
        tasklet, exit, hist, memlet=write_memlet, src_conn="out")

    return state


def make_init_buffer_state(sdfg):

    state = sdfg.add_state("init_buffer")

    hist_buffer = state.add_array(
        "hist_buffer", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)

    entry, exit = state.add_map(
        "init_map", {"i": "0:num_bins"}, schedule=ScheduleType.FPGA_Device)
    tasklet = state.add_tasklet("zero", {}, {"out"}, "out = 0")
    state.add_nedge(entry, tasklet, dace.memlet.EmptyMemlet())
    state.add_memlet_path(
        tasklet,
        exit,
        hist_buffer,
        src_conn="out",
        memlet=dace.memlet.Memlet.simple(hist_buffer, "i"))

    return state


def make_write_buffer_state(sdfg):

    state = sdfg.add_state("write_buffer")

    hist_buffer = state.add_array(
        "hist_buffer", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Local)
    hist_dram = state.add_stream(
        "hist_pipe_out", itype, storage=dace.dtypes.StorageType.FPGA_Local)

    state.add_memlet_path(
        hist_buffer,
        hist_dram,
        memlet=dace.memlet.Memlet.simple(
            hist_dram, "0:num_bins", num_accesses=1))

    return state


def make_compute_nested_sdfg(parent):

    sdfg = SDFG("histogram_compute")

    init_state = make_init_buffer_state(sdfg)
    compute_state = make_compute_state(sdfg)
    finalize_state = make_write_buffer_state(sdfg)

    sdfg.add_edge(init_state, compute_state, dace.graph.edges.InterstateEdge())
    sdfg.add_edge(compute_state, finalize_state,
                  dace.graph.edges.InterstateEdge())

    return sdfg


def make_sdfg(specialize):

    if specialize:
        sdfg = SDFG("histogram_fpga_parallel_{}_{}x{}".format(
            P.get(), H.get(), W.get()))
    else:
        sdfg = SDFG("histogram_fpga_parallel_{}".format(P.get()))

    copy_to_fpga_state = make_copy_to_fpga_state(sdfg)

    state = sdfg.add_state("compute")

    # Compute module
    nested_sdfg = make_compute_nested_sdfg(state)
    tasklet = state.add_nested_sdfg(nested_sdfg, sdfg, {"A_pipe_in"},
                                    {"hist_pipe_out"})
    A_pipes_out = state.add_stream(
        "A_pipes",
        dtype,
        shape=(P, ),
        transient=True,
        storage=StorageType.FPGA_Local)
    A_pipes_in = state.add_stream(
        "A_pipes",
        dtype,
        shape=(P, ),
        transient=True,
        storage=StorageType.FPGA_Local)
    hist_pipes_out = state.add_stream(
        "hist_pipes",
        itype,
        shape=(P, ),
        transient=True,
        storage=StorageType.FPGA_Local)
    unroll_entry, unroll_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)
    state.add_memlet_path(unroll_entry, A_pipes_in, memlet=EmptyMemlet())
    state.add_memlet_path(hist_pipes_out, unroll_exit, memlet=EmptyMemlet())
    state.add_memlet_path(
        A_pipes_in,
        tasklet,
        dst_conn="A_pipe_in",
        memlet=Memlet.simple(A_pipes_in, "p", num_accesses="W*H"))
    state.add_memlet_path(
        tasklet,
        hist_pipes_out,
        src_conn="hist_pipe_out",
        memlet=Memlet.simple(hist_pipes_out, "p", num_accesses="num_bins"))

    # Read module
    a_device = state.add_array(
        "A_device", (H, W),
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    read_entry, read_exit = state.add_map(
        "read_map", {
            "h": "0:H",
            "w": "0:W:P"
        },
        schedule=ScheduleType.FPGA_Device)
    a_val = state.add_array(
        "A_val", (P, ), dtype, transient=True, storage=StorageType.FPGA_Local)
    read_unroll_entry, read_unroll_exit = state.add_map(
        "read_unroll", {"p": "0:P"},
        schedule=ScheduleType.FPGA_Device,
        unroll=True)
    read_tasklet = state.add_tasklet("read", {"A_in"}, {"A_pipe"},
                                     "A_pipe = A_in[p]")
    state.add_memlet_path(
        a_device,
        read_entry,
        a_val,
        memlet=Memlet(
            a_val,
            num_accesses=1,
            subset=Indices(["0"]),
            vector_length=P.get(),
            other_subset=Indices(["h", "w"])))
    state.add_memlet_path(
        a_val,
        read_unroll_entry,
        read_tasklet,
        dst_conn="A_in",
        memlet=Memlet.simple(a_val, "0", veclen=P.get(), num_accesses=1))
    state.add_memlet_path(
        read_tasklet,
        read_unroll_exit,
        read_exit,
        A_pipes_out,
        src_conn="A_pipe",
        memlet=Memlet.simple(A_pipes_out, "p"))

    # Write module
    hist_pipes_in = state.add_stream(
        "hist_pipes",
        itype,
        shape=(P, ),
        transient=True,
        storage=StorageType.FPGA_Local)
    hist_device_out = state.add_array(
        "hist_device", (num_bins, ),
        itype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)
    merge_entry, merge_exit = state.add_map(
        "merge", {"nb": "0:num_bins"}, schedule=ScheduleType.FPGA_Device)
    merge_reduce = state.add_reduce(
        "lambda a, b: a + b", (0, ), "0", schedule=ScheduleType.FPGA_Device)
    state.add_memlet_path(
        hist_pipes_in,
        merge_entry,
        merge_reduce,
        memlet=Memlet.simple(hist_pipes_in, "0:P", num_accesses=P))
    state.add_memlet_path(
        merge_reduce,
        merge_exit,
        hist_device_out,
        memlet=dace.memlet.Memlet.simple(hist_device_out, "nb"))

    copy_to_host_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(copy_to_fpga_state, state, dace.graph.edges.InterstateEdge())
    sdfg.add_edge(state, copy_to_host_state, dace.graph.edges.InterstateEdge())

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("H", type=int)
    parser.add_argument("W", type=int)
    parser.add_argument("P", type=int)
    parser.add_argument(
        "-specialize",
        default=False,
        action="store_true",
        help="Fix all symbols at compile time/in hardware")
    args = vars(parser.parse_args())

    if args["specialize"]:
        H.set(args["H"])
        W.set(args["W"])
        P.set(args["P"])
        histogram = make_sdfg(True)
        histogram.specialize(dict(H=H, W=W, P=P, num_bins=num_bins))
    else:
        P.set(args["P"])
        histogram = make_sdfg(False)
        histogram.specialize(dict(P=P, num_bins=num_bins))
        H.set(args["H"])
        W.set(args["W"])

    print("Histogram {}x{} ({}specialized)".format(
        H.get(), W.get(), "" if args["specialize"] else "not "))

    histogram.draw_to_file()

    A = dace.ndarray([H, W], dtype=dtype)
    hist = dace.ndarray([num_bins], dtype=itype)

    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    hist[:] = itype(0)

    if args["specialize"]:
        histogram(A=A, hist=hist)
    else:
        histogram(A=A, H=H, W=W, hist=hist)

    if dace.Config.get_bool('profiling'):
        dace.timethis('histogram', 'numpy', (H.get() * W.get()), np.histogram,
                      A, num_bins)

    ref = np.histogram(A, bins=num_bins.get(), range=(0.0, 1.0))[0]
    diff = np.linalg.norm(ref[1:-1] - hist[1:-1])

    print("Difference:", diff)
    if diff > 1e-5:
        print("** Kernel")
        print(hist)
        print("** Reference")
        print(ref)
        print("Validation failed.")
    print("==== Program end ====")

    exit(0 if diff <= 1e-5 else 1)
