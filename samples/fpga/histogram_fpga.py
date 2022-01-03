# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol("W")
H = dace.symbol("H")
num_bins = dace.symbol("num_bins")
num_bins.set(256)
dtype = dace.float32


def make_copy_to_fpga_state(sdfg):

    state = sdfg.add_state("copy_to_fpga")

    a_host = state.add_array("A", (H, W), dtype)
    a_device = state.add_array("A_device", (H, W), dtype, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    hist_host = state.add_array("hist", (num_bins, ), dace.uint32)
    hist_device = state.add_array("hist_device", (num_bins, ),
                                  dace.uint32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(a_host, a_device, memlet=dace.memlet.Memlet.simple(a_device, "0:H, 0:W"))
    state.add_memlet_path(hist_host, hist_device, memlet=dace.memlet.Memlet.simple(hist_device, "0:num_bins"))

    return state


def make_copy_to_host_state(sdfg):

    state = sdfg.add_state("copy_to_host")

    hist_device = state.add_array("hist_device", (num_bins, ),
                                  dace.uint32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Global)
    hist_host = state.add_array("hist", (num_bins, ), dace.uint32)

    state.add_memlet_path(hist_device, hist_host, memlet=dace.memlet.Memlet.simple(hist_host, "0:num_bins"))

    return state


def make_compute_state(sdfg):

    state = sdfg.add_state("histogram_fpga")

    a = state.add_array("A_in", (H, W), dtype, storage=dace.dtypes.StorageType.FPGA_Global)
    hist = state.add_array("hist_buffer", (num_bins, ),
                           dace.uint32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)

    entry, exit = state.add_map("map", {"i": "0:H", "j": "0:W"})

    tasklet = state.add_tasklet("compute", {"a"}, {"out"}, "out[int(float(num_bins) * a)] = 1")

    read_memlet = dace.memlet.Memlet.simple(a, "i, j")
    write_memlet = dace.memlet.Memlet.simple(hist, "0:num_bins", wcr_str="lambda a, b: a + b")

    state.add_memlet_path(a, entry, tasklet, memlet=read_memlet, dst_conn="a")
    state.add_memlet_path(tasklet, exit, hist, memlet=write_memlet, src_conn="out")

    return state


def make_init_buffer_state(sdfg):

    state = sdfg.add_state("init_buffer")

    hist_buffer = state.add_array("hist_buffer", (num_bins, ),
                                  dace.uint32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)

    entry, exit = state.add_map("init_map", {"i": "0:num_bins"})
    tasklet = state.add_tasklet("zero", {}, {"out"}, "out = 0")
    state.add_nedge(entry, tasklet, dace.memlet.Memlet())
    state.add_memlet_path(tasklet,
                          exit,
                          hist_buffer,
                          src_conn="out",
                          memlet=dace.memlet.Memlet.simple(hist_buffer, "i"))

    return state


def make_write_buffer_state(sdfg):

    state = sdfg.add_state("write_buffer")

    hist_buffer = state.add_array("hist_buffer", (num_bins, ),
                                  dace.uint32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Local)
    hist_dram = state.add_array("hist_out", (num_bins, ), dace.uint32, storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(hist_buffer, hist_dram, memlet=dace.memlet.Memlet.simple(hist_dram, "0:num_bins"))

    return state


def make_nested_sdfg(parent):

    sdfg = dace.SDFG("compute")

    init_state = make_init_buffer_state(sdfg)
    compute_state = make_compute_state(sdfg)
    finalize_state = make_write_buffer_state(sdfg)

    sdfg.add_edge(init_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, finalize_state, dace.sdfg.InterstateEdge())

    return sdfg


def make_sdfg(specialize):

    if specialize:
        sdfg = dace.SDFG("histogram_fpga_{}x{}".format(H.get(), W.get()))
    else:
        sdfg = dace.SDFG("histogram_fpga")

    copy_to_fpga_state = make_copy_to_fpga_state(sdfg)

    state = sdfg.add_state("compute")
    nested_sdfg = make_nested_sdfg(state)
    tasklet = state.add_nested_sdfg(nested_sdfg, sdfg, {"A_in"}, {"hist_out"})
    a_device = state.add_array("A_device", (H, W), dtype, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    hist_device = state.add_array("hist_device", (num_bins, ),
                                  dace.uint32,
                                  transient=True,
                                  storage=dace.dtypes.StorageType.FPGA_Global)
    state.add_memlet_path(a_device, tasklet, dst_conn="A_in", memlet=dace.memlet.Memlet.simple(a_device, "0:H, 0:W"))
    state.add_memlet_path(tasklet,
                          hist_device,
                          src_conn="hist_out",
                          memlet=dace.memlet.Memlet.simple(hist_device, "0:num_bins"))

    copy_to_host_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(copy_to_fpga_state, state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(state, copy_to_host_state, dace.sdfg.InterstateEdge())

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("H", type=int)
    parser.add_argument("W", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all symbols at compile time/in hardware")
    args = vars(parser.parse_args())

    if args["specialize"]:
        H.set(args["H"])
        W.set(args["W"])
        histogram = make_sdfg(True)
        histogram.specialize(dict(H=H, W=W, num_bins=num_bins))
    else:
        histogram = make_sdfg(False)
        histogram.specialize(dict(num_bins=num_bins))
        H.set(args["H"])
        W.set(args["W"])

    print("Histogram {}x{} ({}specialized)".format(H.get(), W.get(), "" if args["specialize"] else "not "))

    A = dace.ndarray([H, W], dtype=dtype)
    hist = dace.ndarray([num_bins], dtype=dace.uint32)

    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    hist[:] = dace.uint32(0)

    if args["specialize"]:
        histogram(A=A, hist=hist)
    else:
        histogram(A=A, H=H, W=W, hist=hist)

    if dace.Config.get_bool('profiling'):
        dace.timethis('histogram', 'numpy', (H.get() * W.get()), np.histogram, A, num_bins)

    diff = np.linalg.norm(np.histogram(A, bins=num_bins.get(), range=(0.0, 1.0))[0][1:-1] - hist[1:-1])

    print("Difference:", diff)
    if diff > 1e-5:
        print("Validation failed.")
    print("==== Program end ====")

    exit(0 if diff <= 1e-5 else 1)
