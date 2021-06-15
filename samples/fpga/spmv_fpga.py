# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np
import scipy
import pdb
import select
import sys

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')
itype = dace.dtypes.uint32
dtype = dace.dtypes.float32


def make_pre_state(sdfg):

    state = sdfg.add_state("pre_state")

    a_row_host = state.add_array("A_row", (H + 1, ), itype)
    a_row_device = state.add_array("A_row_device", (H + 1, ),
                                   itype,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)

    a_col_host = state.add_array("A_col", (nnz, ), itype)
    a_col_device = state.add_array("A_col_device", (nnz, ),
                                   itype,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)

    a_val_host = state.add_array("A_val", (nnz, ), dtype)
    a_val_device = state.add_array("A_val_device", (nnz, ),
                                   dtype,
                                   transient=True,
                                   storage=dace.dtypes.StorageType.FPGA_Global)

    x_host = state.add_array("x", (W, ), dtype)
    x_device = state.add_array("x_device", (W, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)

    state.add_memlet_path(a_row_host,
                          a_row_device,
                          memlet=dace.memlet.Memlet.simple(
                              a_row_device, "0:H+1"))

    state.add_memlet_path(a_col_host,
                          a_col_device,
                          memlet=dace.memlet.Memlet.simple(
                              a_col_device, "0:nnz"))

    state.add_memlet_path(a_val_host,
                          a_val_device,
                          memlet=dace.memlet.Memlet.simple(
                              a_val_device, "0:nnz"))

    state.add_memlet_path(x_host,
                          x_device,
                          memlet=dace.memlet.Memlet.simple(x_device, "0:W"))

    return state


def make_post_state(sdfg):

    state = sdfg.add_state("post_state")

    b_device = state.add_array("b_device", (H, ),
                               dtype,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Global)
    b_host = state.add_array("b", (H, ), dtype)

    state.add_memlet_path(b_device,
                          b_host,
                          memlet=dace.memlet.Memlet.simple(b_host, "0:H"))

    return state


def make_nested_sdfg(parent):

    sdfg = dace.SDFG("spmv_inner")

    set_zero_state = sdfg.add_state("set_zero")
    set_zero_b = set_zero_state.add_scalar(
        "b_buffer",
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    set_zero_tasklet = set_zero_state.add_tasklet("set_zero", {}, {"b_out"},
                                                  "b_out = 0")
    set_zero_state.add_memlet_path(set_zero_tasklet,
                                   set_zero_b,
                                   src_conn="b_out",
                                   memlet=dace.memlet.Memlet.simple(
                                       set_zero_b, "0"))

    write_back_state = sdfg.add_state("write_back")
    write_back_b_buffer = write_back_state.add_scalar(
        "b_buffer",
        dtype,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Registers)
    write_back_b = write_back_state.add_scalar(
        "b_write", dtype, storage=dace.dtypes.StorageType.FPGA_Registers)
    write_back_state.add_memlet_path(write_back_b_buffer,
                                     write_back_b,
                                     memlet=dace.memlet.Memlet.simple(
                                         write_back_b, "0"))

    state = sdfg.add_state("compute_cols")

    sdfg.add_edge(set_zero_state, state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(state, write_back_state, dace.sdfg.InterstateEdge())

    compute_entry, compute_exit = state.add_map("compute_col",
                                                {"j": "rowptr:rowend"})

    indirection_tasklet = state.add_tasklet("indirection",
                                            {"x_val_in", "col_index"},
                                            {"lookup"},
                                            "lookup = x_val_in[col_index]")

    x_in = state.add_scalar("x_in",
                            dtype,
                            storage=dace.dtypes.StorageType.FPGA_Registers,
                            transient=True)

    compute_tasklet = state.add_tasklet("compute", {"a", "x_val_in"}, {"out"},
                                        "out = a * x_val_in")

    b_buffer = state.add_scalar("b_buffer",
                                dtype,
                                transient=True,
                                storage=dace.dtypes.StorageType.FPGA_Registers)
    rowptr = state.add_scalar("row_begin",
                              itype,
                              storage=dace.dtypes.StorageType.FPGA_Registers)
    rowend = state.add_scalar("row_end",
                              itype,
                              storage=dace.dtypes.StorageType.FPGA_Registers)
    a_val = state.add_array("A_val_read", (nnz, ),
                            dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
    a_col = state.add_array("A_col_read", (nnz, ),
                            itype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
    x = state.add_array("x_read", (W, ),
                        dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global)

    compute_entry.add_in_connector("rowptr")
    state.add_memlet_path(rowptr,
                          compute_entry,
                          dst_conn="rowptr",
                          memlet=dace.memlet.Memlet.simple(rowptr, "0"))

    compute_entry.add_in_connector("rowend")
    state.add_memlet_path(rowend,
                          compute_entry,
                          dst_conn="rowend",
                          memlet=dace.memlet.Memlet.simple(rowend, "0"))

    state.add_memlet_path(a_val,
                          compute_entry,
                          compute_tasklet,
                          dst_conn="a",
                          memlet=dace.memlet.Memlet.simple(a_val, "j"))

    state.add_memlet_path(x,
                          compute_entry,
                          indirection_tasklet,
                          dst_conn="x_val_in",
                          memlet=dace.memlet.Memlet.simple(x, "0:W"))

    state.add_memlet_path(a_col,
                          compute_entry,
                          indirection_tasklet,
                          dst_conn="col_index",
                          memlet=dace.memlet.Memlet.simple(a_col, "j"))

    state.add_memlet_path(indirection_tasklet,
                          x_in,
                          src_conn="lookup",
                          memlet=dace.memlet.Memlet.simple(x_in, "0"))

    state.add_memlet_path(x_in,
                          compute_tasklet,
                          dst_conn="x_val_in",
                          memlet=dace.memlet.Memlet.simple(x_in, "0"))

    state.add_memlet_path(compute_tasklet,
                          compute_exit,
                          b_buffer,
                          src_conn="out",
                          memlet=dace.memlet.Memlet.simple(
                              b_buffer, "0", wcr_str="lambda a, b: a + b"))

    return sdfg


def make_main_state(sdfg):

    state = sdfg.add_state("spmv")

    a_row = state.add_array("A_row_device", (H + 1, ),
                            itype,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Global)
    a_col = state.add_array("A_col_device", (nnz, ),
                            itype,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Global)
    a_val = state.add_array("A_val_device", (nnz, ),
                            dtype,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Global)
    x = state.add_array("x_device", (W, ),
                        dtype,
                        transient=True,
                        storage=dace.dtypes.StorageType.FPGA_Global)
    b = state.add_array("b_device", (H, ),
                        dtype,
                        transient=True,
                        storage=dace.dtypes.StorageType.FPGA_Global)

    row_entry, row_exit = state.add_map(
        "compute_row", {"i": "0:H"},
        schedule=dace.dtypes.ScheduleType.FPGA_Device)

    rowptr = state.add_scalar("rowptr",
                              itype,
                              storage=dace.dtypes.StorageType.FPGA_Registers,
                              transient=True)
    rowend = state.add_scalar("rowend",
                              itype,
                              storage=dace.dtypes.StorageType.FPGA_Registers,
                              transient=True)

    nested_sdfg = make_nested_sdfg(state)
    nested_sdfg_tasklet = state.add_nested_sdfg(
        nested_sdfg, sdfg,
        {"row_begin", "row_end", "A_val_read", "A_col_read", "x_read"},
        {"b_write"})

    state.add_memlet_path(a_row,
                          row_entry,
                          rowptr,
                          memlet=dace.memlet.Memlet.simple(
                              rowptr, "0", other_subset_str="i"))
    state.add_memlet_path(rowptr,
                          nested_sdfg_tasklet,
                          dst_conn="row_begin",
                          memlet=dace.memlet.Memlet.simple(rowptr, "0"))

    state.add_memlet_path(a_row,
                          row_entry,
                          rowend,
                          memlet=dace.memlet.Memlet.simple(
                              rowend, "0", other_subset_str="i + 1"))
    state.add_memlet_path(rowend,
                          nested_sdfg_tasklet,
                          dst_conn="row_end",
                          memlet=dace.memlet.Memlet.simple(rowend, "0"))

    state.add_memlet_path(a_val,
                          row_entry,
                          nested_sdfg_tasklet,
                          dst_conn="A_val_read",
                          memlet=dace.memlet.Memlet.simple(a_val, "0:nnz"))

    state.add_memlet_path(x,
                          row_entry,
                          nested_sdfg_tasklet,
                          dst_conn="x_read",
                          memlet=dace.memlet.Memlet.simple(x, "0:W"))

    state.add_memlet_path(a_col,
                          row_entry,
                          nested_sdfg_tasklet,
                          dst_conn="A_col_read",
                          memlet=dace.memlet.Memlet.simple(a_col, "0:nnz"))

    state.add_memlet_path(nested_sdfg_tasklet,
                          row_exit,
                          b,
                          src_conn="b_write",
                          memlet=dace.memlet.Memlet.simple(b, "i"))

    return state


def make_sdfg(specialize):

    if specialize:
        name = "spmv_fpga_{}x{}x{}".format(H.get(), W.get(), nnz.get())
    else:
        name = "spmv_fpga"
    sdfg = dace.SDFG(name)

    pre_state = make_pre_state(sdfg)
    main_state = make_main_state(sdfg)
    post_state = make_post_state(sdfg)

    sdfg.add_edge(pre_state, main_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(main_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int)
    parser.add_argument("H", type=int)
    parser.add_argument("nnz", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all symbols at compile time/in hardware")
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])
    nnz.set(args["nnz"])

    print(
        'Sparse Matrix-Vector Multiplication {}x{} ({} non-zero elements, {}specialized)'
        .format(W.get(), H.get(), nnz.get(),
                "not " if not args["specialize"] else ""))

    A_row = dace.ndarray([H + 1], dtype=itype)
    A_col = dace.ndarray([nnz], dtype=itype)
    A_val = dace.ndarray([nnz], dtype=dtype)

    x = dace.ndarray([W], dtype)
    b = dace.ndarray([H], dtype)

    # Assuming uniform sparsity distribution across rows
    nnz_per_row = nnz.get() // H.get()
    nnz_last_row = nnz_per_row + (nnz.get() % H.get())
    if nnz_last_row > W.get():
        print('Too many nonzeros per row')
        exit(1)

    # RANDOMIZE SPARSE MATRIX
    A_row[0] = itype(0)
    A_row[1:H.get()] = itype(nnz_per_row)
    A_row[-1] = itype(nnz_last_row)
    A_row = np.cumsum(A_row, dtype=itype.type)

    # Fill column data
    for i in range(H.get() - 1):
        A_col[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(W.get(), nnz_per_row, replace=False))
    # Fill column data for last row
    A_col[nnz_per_row * (H.get() - 1):] = np.sort(
        np.random.choice(W.get(), nnz_last_row, replace=False))

    A_val[:] = np.random.rand(nnz.get()).astype(dtype.type)
    #########################

    x[:] = np.random.rand(W.get()).astype(dtype.type)
    #b[:] = dtype(0)

    # Setup regression
    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row),
                                       shape=(H.get(), W.get()))

    spmv = make_sdfg(args["specialize"])
    if args["specialize"]:
        spmv.specialize(dict(H=H, W=W, nnz=nnz))
    spmv(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=H, W=W, nnz=nnz)

    if dace.Config.get_bool('profiling'):
        dace.timethis('spmv', 'scipy', 0, A_sparse.dot, x)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(H.get())
    print("Difference:", diff)
    if diff >= 1e-5:
        print("Validation failed.")
        print("Result:")
        print(b)
        print("Reference:")
        print(A_sparse.dot(x))
        print("Type \"debug\" to enter debugger, "
              "or any other string to quit (timeout in 10 seconds)")
        read, _, _ = select.select([sys.stdin], [], [], 10)
        if len(read) > 0 and sys.stdin.readline().strip().lower() == "debug":
            print("Entering debugger...")
            pdb.set_trace()
        else:
            print("Exiting...")
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
