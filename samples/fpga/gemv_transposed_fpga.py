#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
import select
import sys

N = dace.symbol("N")
M = dace.symbol("M")
dtype = dace.float64

# This implementation of transposed DGEMV assumes that the two vectors (x and y) fit into
# FPGA fast memory


def make_init_state(sdfg):

    state = sdfg.add_state("init")

    a_host = state.add_array("A", (M, N), dtype)
    a_device = state.add_array("A_device", (M, N),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    x_host = state.add_array("x", (M, ), dtype)
    x_device = state.add_array("x_device", (M, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    y_host = state.add_array("y", (M, ), dtype)
    y_device = state.add_array("y_device", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(a_host,
                          a_device,
                          memlet=dace.memlet.Memlet.simple(
                              a_device, "0:N, 0:M"))

    state.add_memlet_path(x_host,
                          x_device,
                          memlet=dace.memlet.Memlet.simple(x_device, "0:M"))

    state.add_memlet_path(y_host,
                          y_device,
                          memlet=dace.memlet.Memlet.simple(y_device, "0:N"))

    return state


def make_finalize_state(sdfg):

    state = sdfg.add_state("finalize")

    y_host = state.add_array("y", (M, ), dtype)
    y_device = state.add_array("y_device", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(y_device,
                          y_host,
                          memlet=dace.memlet.Memlet.simple(y_host, "0:N"))

    return state


def make_load_state(sdfg):

    state = sdfg.add_state("load")

    y = state.add_array("y_nested", (N, ),
                        dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    y_buffer = state.add_array("y_buffer", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Local)

    state.add_memlet_path(y,
                          y_buffer,
                          memlet=dace.memlet.Memlet.simple(y_buffer, "0:N"))

    return state


def make_store_state(sdfg):

    state = sdfg.add_state("store")

    y_buffer = state.add_array("y_buffer", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Local)
    y = state.add_array("y_nested", (N, ),
                        dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(y_buffer,
                          y,
                          memlet=dace.memlet.Memlet.simple(y, "0:N"))

    return state


def make_compute_state(sdfg):

    state = sdfg.add_state("compute")

    a = state.add_array("A_nested", (M, N),
                        dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    x = state.add_array("x_nested", (M, ),
                        dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    y_buffer = state.add_array("y_buffer", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Local)

    cols_entry, cols_exit = state.add_map(
        "cols", {"m": "0:M"}, schedule=dace.ScheduleType.Sequential)
    rows_entry, rows_exit = state.add_map("rows", {"n": "0:N"})

    tasklet = state.add_tasklet("update", {"a", "x_in"}, {"update"},
                                "update = a * x_in")

    wcr_memlet = dace.memlet.Memlet.simple(y_buffer,
                                           "n",
                                           wcr_str="lambda a, b: a + b",
                                           wcr_identity=0)

    state.add_memlet_path(a,
                          cols_entry,
                          rows_entry,
                          tasklet,
                          dst_conn="a",
                          memlet=dace.memlet.Memlet.simple(a, "m, n"))
    state.add_memlet_path(x,
                          cols_entry,
                          rows_entry,
                          tasklet,
                          dst_conn="x_in",
                          memlet=dace.memlet.Memlet.simple(x, "m"))
    state.add_memlet_path(tasklet,
                          rows_exit,
                          cols_exit,
                          y_buffer,
                          src_conn="update",
                          memlet=wcr_memlet)

    return state


def make_outer_compute_state(sdfg):

    state = sdfg.add_state("gemv_transposed")

    nested_sdfg = dace.SDFG("gemv_transposed")
    load_state = make_load_state(nested_sdfg)
    compute_state = make_compute_state(nested_sdfg)
    store_state = make_store_state(nested_sdfg)
    nested_sdfg.add_edge(load_state, compute_state,
                         dace.graph.edges.InterstateEdge())
    nested_sdfg.add_edge(compute_state, store_state,
                         dace.graph.edges.InterstateEdge())

    tasklet = state.add_nested_sdfg(nested_sdfg, sdfg,
                                    {"A_nested", "x_nested"}, {"y_nested"})

    a_device = state.add_array("A_device", (M, N),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    x_device = state.add_array("x_device", (M, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    y_device = state.add_array("y_device", (N, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(a_device,
                          tasklet,
                          dst_conn="A_nested",
                          memlet=dace.memlet.Memlet.simple(
                              a_device, "0:M, 0:N"))
    state.add_memlet_path(x_device,
                          tasklet,
                          dst_conn="x_nested",
                          memlet=dace.memlet.Memlet.simple(x_device, "0:M"))
    state.add_memlet_path(tasklet,
                          y_device,
                          src_conn="y_nested",
                          memlet=dace.memlet.Memlet.simple(y_device, "0:N"))

    return state


def make_sdfg(specialize):

    if specialize:
        name = "gemv_transposed_{}x{}".format(N.get(), M.get())
    else:
        name = "gemv_transposed_{}xM".format(N.get())

    sdfg = dace.SDFG(name)

    init_state = make_init_state(sdfg)
    fpga_state = make_outer_compute_state(sdfg)
    finalize_state = make_finalize_state(sdfg)

    sdfg.add_edge(init_state, fpga_state, dace.graph.edges.InterstateEdge())
    sdfg.add_edge(fpga_state, finalize_state,
                  dace.graph.edges.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("M", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Also fix M in hardware")
    args = vars(parser.parse_args())

    N.set(args["N"])
    if args["specialize"]:
        print("Specializing M...")
        M.set(args["M"])

    gemv = make_sdfg(args["specialize"])
    gemv.specialize(dict(N=N))

    if not args["specialize"]:
        M.set(args["M"])
    else:
        gemv.specialize(dict(M=M))

    print("Running GEMV {}x{} ({}specialized)".format(
        N.get(), M.get(), ("" if args["specialize"] else "not ")))

    A = dace.ndarray([M, N], dtype=dtype)
    x = dace.ndarray([M], dtype=dtype)
    y = dace.ndarray([N], dtype=dtype)

    # Intialize: randomize A, x and y
    # A[:, :] = np.random.rand(M.get(), N.get()).astype(dtype.type)
    # x[:] = np.random.rand(M.get()).astype(dtype.type)
    # y[:] = np.random.rand(N.get()).astype(dtype.type)
    A[:, :] = 1
    x[:] = 1
    y[:] = 0

    # Regression
    regression = np.matmul(np.transpose(A), x) + y

    #############################################
    # Run DaCe program

    if args["specialize"]:
        gemv(A=A, x=x, y=x)
    else:
        gemv(A=A, M=M, x=x, y=y)

    residual = np.linalg.norm(y - regression) / (N.get() * M.get())
    print("Residual:", residual)
    diff = np.abs(y - regression)
    wrong_elements = np.transpose(np.nonzero(diff >= 0.01))
    highest_diff = np.max(diff)

    print("==== Program end ====")
    if residual >= 0.01 or highest_diff >= 0.01:
        print("Verification failed!")
        print("Residual: {}".format(residual))
        print("Incorrect elements: {} / {}".format(wrong_elements.shape[0],
                                                   (N.get() * M.get())))
        print("Highest difference: {}".format(highest_diff))
        print("** Result:\n", y)
        print("** Reference:\n", regression)
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
