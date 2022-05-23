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

from dace.sdfg import SDFG, InterstateEdge
from dace.memlet import Memlet
from dace.dtypes import AllocationLifetime, ScheduleType, StorageType, Language
from dace.properties import CodeProperty

cols = dace.symbol("cols")
rows = dace.symbol("rows")
nnz = dace.symbol("nnz")
itype = dace.dtypes.uint32
dtype = dace.dtypes.float32


def make_pre_state(sdfg):

    state = sdfg.add_state("pre_state")

    a_row_host = state.add_array("A_row", (rows + 1, ), itype)
    a_row_device = state.add_array("A_row_device", (rows + 1, ), itype, transient=True, storage=StorageType.FPGA_Global)

    a_col_host = state.add_array("A_col", (nnz, ), itype)
    a_col_device = state.add_array("A_col_device", (nnz, ), itype, transient=True, storage=StorageType.FPGA_Global)

    a_val_host = state.add_array("A_val", (nnz, ), dtype)
    a_val_device = state.add_array("A_val_device", (nnz, ), dtype, transient=True, storage=StorageType.FPGA_Global)

    x_host = state.add_array("x", (cols, ), dtype)
    x_device = state.add_array("x_device", (cols, ), dtype, transient=True, storage=StorageType.FPGA_Global)

    state.add_memlet_path(a_row_host, a_row_device, memlet=dace.memlet.Memlet.simple(a_row_device, "0:rows+1"))

    state.add_memlet_path(a_col_host, a_col_device, memlet=dace.memlet.Memlet.simple(a_col_device, "0:nnz"))

    state.add_memlet_path(a_val_host, a_val_device, memlet=dace.memlet.Memlet.simple(a_val_device, "0:nnz"))

    state.add_memlet_path(x_host, x_device, memlet=dace.memlet.Memlet.simple(x_device, "0:cols"))

    return state


def make_post_state(sdfg):

    state = sdfg.add_state("post_state")

    b_device = state.add_array("b_device", (rows, ), dtype, transient=True, storage=StorageType.FPGA_Global)
    b_host = state.add_array("b", (rows, ), dtype)

    state.add_memlet_path(b_device, b_host, memlet=dace.memlet.Memlet.simple(b_host, "0:rows"))

    return state


def make_write_sdfg():

    sdfg = SDFG("spmv_write")

    begin = sdfg.add_state("begin")
    entry = sdfg.add_state("entry")
    state = sdfg.add_state("body")
    end = sdfg.add_state("end")

    sdfg.add_edge(begin, entry, InterstateEdge(assignments={"h": "0"}))

    sdfg.add_edge(entry, state,
                  InterstateEdge(condition=CodeProperty.from_string("h < rows", language=Language.Python)))

    sdfg.add_edge(entry, end, InterstateEdge(condition=CodeProperty.from_string("h >= rows", language=Language.Python)))

    sdfg.add_edge(state, entry, InterstateEdge(assignments={"h": "h + 1"}))

    result_to_write_in = state.add_stream("b_pipe", dtype, storage=StorageType.FPGA_Local)
    b = state.add_array("b_mem", (rows, ), dtype, storage=StorageType.FPGA_Global)

    state.add_memlet_path(result_to_write_in, b, memlet=Memlet.simple(b, "h"))

    return sdfg


def make_iteration_space(sdfg):

    pre_state = sdfg.add_state("pre_state")

    rows_begin = sdfg.add_state("rows_begin")
    rows_entry = sdfg.add_state("rows_entry")
    rows_end = sdfg.add_state("rows_end")

    shift_rowptr = sdfg.add_state("shift_rowptr")
    read_rowptr = sdfg.add_state("read_rowptr")

    cols_begin = sdfg.add_state("cols_begin")
    cols_entry = sdfg.add_state("cols_entry")
    cols_end = sdfg.add_state("cols_end")

    body = sdfg.add_state("compute")

    post_state = sdfg.add_state("post_state")

    sdfg.add_edge(pre_state, rows_begin, InterstateEdge())

    sdfg.add_edge(rows_begin, rows_entry, InterstateEdge(assignments={"h": "0"}))
    sdfg.add_edge(rows_entry, shift_rowptr,
                  InterstateEdge(condition=CodeProperty.from_string("h < rows", language=Language.Python)))
    sdfg.add_edge(rows_entry, rows_end,
                  InterstateEdge(condition=CodeProperty.from_string("h >= rows", language=Language.Python)))

    sdfg.add_edge(shift_rowptr, read_rowptr, InterstateEdge())
    sdfg.add_edge(read_rowptr, cols_begin, InterstateEdge())

    sdfg.add_edge(cols_begin, cols_entry, InterstateEdge(assignments={"c": "0"}))
    sdfg.add_edge(
        cols_entry, body,
        InterstateEdge(condition=CodeProperty.from_string("c < row_end - row_begin", language=Language.Python)))
    sdfg.add_edge(
        cols_entry, cols_end,
        InterstateEdge(condition=CodeProperty.from_string("c >= row_end - row_begin", language=Language.Python)))

    sdfg.add_edge(body, cols_entry, InterstateEdge(assignments={"c": "c + 1"}))
    sdfg.add_edge(cols_end, post_state, InterstateEdge())
    sdfg.add_edge(post_state, rows_entry, InterstateEdge(assignments={"h": "h + 1"}))

    row_end_first = pre_state.add_scalar("row_end", itype, transient=True, storage=StorageType.FPGA_Registers)
    row_pipe_first = pre_state.add_stream("row_pipe", itype, storage=StorageType.FPGA_Local)
    pre_state.add_memlet_path(row_pipe_first, row_end_first, memlet=Memlet.simple(row_end_first, "0"))

    row_end_shift = shift_rowptr.add_scalar("row_end", itype, transient=True, storage=StorageType.FPGA_Registers)
    row_begin_shift = shift_rowptr.add_scalar("row_begin",
                                              itype,
                                              transient=True,
                                              lifetime=AllocationLifetime.SDFG,
                                              storage=StorageType.FPGA_Registers)
    shift_rowptr.add_memlet_path(row_end_shift, row_begin_shift, memlet=Memlet.simple(row_begin_shift, "0"))

    row_pipe = read_rowptr.add_stream("row_pipe", itype, storage=StorageType.FPGA_Local)
    row_end = read_rowptr.add_scalar("row_end", itype, transient=True, storage=StorageType.FPGA_Registers)
    read_rowptr.add_memlet_path(row_pipe, row_end, memlet=Memlet.simple(row_end, "0"))

    return pre_state, body, post_state


def make_compute_nested_sdfg():

    sdfg = SDFG("spmv_compute_nested")

    if_state = sdfg.add_state("if")
    then_state = sdfg.add_state("then")
    else_state = sdfg.add_state("else")
    end_state = sdfg.add_state("end")

    sdfg.add_edge(if_state, then_state,
                  InterstateEdge(condition=CodeProperty.from_string("c == 0", language=Language.Python)))
    sdfg.add_edge(if_state, else_state,
                  InterstateEdge(condition=CodeProperty.from_string("c != 0", language=Language.Python)))
    sdfg.add_edge(then_state, end_state, InterstateEdge())
    sdfg.add_edge(else_state, end_state, InterstateEdge())

    a_in = if_state.add_scalar("a_in", dtype, storage=StorageType.FPGA_Registers)
    x_in = if_state.add_scalar("x_in", dtype, storage=StorageType.FPGA_Registers)
    b_tmp_out = if_state.add_scalar("b_tmp", dtype, transient=True, storage=StorageType.FPGA_Registers)
    tasklet = if_state.add_tasklet("compute", {"_a_in", "_x_in"}, {"_b_out"}, "_b_out = _a_in * _x_in")
    if_state.add_memlet_path(a_in, tasklet, dst_conn="_a_in", memlet=Memlet.simple(a_in, "0"))
    if_state.add_memlet_path(x_in, tasklet, dst_conn="_x_in", memlet=Memlet.simple(x_in, "0"))
    if_state.add_memlet_path(tasklet, b_tmp_out, src_conn="_b_out", memlet=Memlet.simple(b_tmp_out, "0"))

    b_tmp_then_in = then_state.add_scalar("b_tmp", dtype, transient=True, storage=StorageType.FPGA_Registers)
    b_then_out = then_state.add_scalar("b_out", dtype, storage=StorageType.FPGA_Registers)
    then_state.add_memlet_path(b_tmp_then_in, b_then_out, memlet=Memlet.simple(b_then_out, "0"))

    b_tmp_else_in = else_state.add_scalar("b_tmp", dtype, transient=True, storage=StorageType.FPGA_Registers)
    b_else_in = else_state.add_scalar("b_in", dtype, storage=StorageType.FPGA_Registers)
    b_else_out = else_state.add_scalar("b_out", dtype, storage=StorageType.FPGA_Registers)
    else_tasklet = else_state.add_tasklet("b_wcr", {"_b_in", "b_prev"}, {"_b_out"}, "_b_out = b_prev + _b_in")
    else_state.add_memlet_path(b_tmp_else_in, else_tasklet, dst_conn="_b_in", memlet=Memlet.simple(b_tmp_else_in, "0"))
    else_state.add_memlet_path(b_else_in, else_tasklet, dst_conn="b_prev", memlet=Memlet.simple(b_else_in, "0"))
    else_state.add_memlet_path(else_tasklet, b_else_out, src_conn="_b_out", memlet=Memlet.simple(b_else_out, "0"))

    return sdfg


def make_compute_sdfg():

    sdfg = SDFG("spmv_compute")

    pre_state, body, post_state = make_iteration_space(sdfg)

    a_pipe = body.add_stream("a_pipe", dtype, storage=StorageType.FPGA_Local)
    x_pipe = body.add_stream("x_pipe", dtype, storage=StorageType.FPGA_Local)
    b_buffer_in = body.add_scalar("b_buffer", dtype, transient=True, storage=StorageType.FPGA_Registers)
    b_buffer_out = body.add_scalar("b_buffer", dtype, transient=True, storage=StorageType.FPGA_Registers)
    nested_sdfg = make_compute_nested_sdfg()
    tasklet = body.add_nested_sdfg(nested_sdfg, sdfg, {"a_in", "x_in", "b_in"}, {"b_out"})
    body.add_memlet_path(a_pipe, tasklet, dst_conn="a_in", memlet=Memlet.simple(a_pipe, "0"))
    body.add_memlet_path(b_buffer_in, tasklet, dst_conn="b_in", memlet=Memlet.simple(b_buffer_in, "0"))
    body.add_memlet_path(x_pipe, tasklet, dst_conn="x_in", memlet=Memlet.simple(x_pipe, "0"))
    body.add_memlet_path(tasklet, b_buffer_out, src_conn="b_out", memlet=Memlet.simple(b_buffer_out, "0"))

    b_buffer_post_in = post_state.add_scalar("b_buffer", dtype, transient=True, storage=StorageType.FPGA_Registers)
    b_pipe = post_state.add_stream("b_pipe", dtype, storage=StorageType.FPGA_Local)
    post_state.add_memlet_path(b_buffer_post_in, b_pipe, memlet=Memlet.simple(b_pipe, "0"))

    return sdfg


def make_read_x():

    sdfg = SDFG("spmv_read_x")

    pre_state, body, post_state = make_iteration_space(sdfg)

    x_mem = body.add_array("x_mem", (cols, ), dtype, storage=StorageType.FPGA_Global)
    col_pipe = body.add_stream("col_pipe", itype, storage=StorageType.FPGA_Local)
    compute_pipe = body.add_stream("compute_pipe", dtype, storage=StorageType.FPGA_Local)

    tasklet = body.add_tasklet("read_x", {"x_in", "col_in"}, {"x_out"}, "x_out = x_in[col_in]")

    body.add_memlet_path(x_mem, tasklet, dst_conn="x_in", memlet=Memlet.simple(x_mem, "0:cols"))
    body.add_memlet_path(col_pipe, tasklet, dst_conn="col_in", memlet=Memlet.simple(col_pipe, "0"))
    body.add_memlet_path(tasklet, compute_pipe, src_conn="x_out", memlet=Memlet.simple(compute_pipe, "0"))

    return sdfg


def make_read_val():

    sdfg = SDFG("spmv_read_val")

    pre_state, body, post_state = make_iteration_space(sdfg)

    a_val_mem = body.add_array("A_val_mem", (nnz, ), dtype, storage=StorageType.FPGA_Global)
    compute_pipe = body.add_stream("compute_pipe", dtype, storage=StorageType.FPGA_Local)

    tasklet = body.add_tasklet("read_val", {"a_in"}, {"a_out"}, "a_out = a_in[row_begin + c]")

    body.add_memlet_path(a_val_mem, tasklet, dst_conn="a_in", memlet=Memlet.simple(a_val_mem, "0:nnz"))
    body.add_memlet_path(tasklet, compute_pipe, src_conn="a_out", memlet=Memlet.simple(compute_pipe, "0"))

    return sdfg


def make_read_col():

    sdfg = SDFG("spmv_read_col")

    pre_state, body, post_state = make_iteration_space(sdfg)

    a_col = body.add_array("A_col_mem", (nnz, ), itype, storage=StorageType.FPGA_Global)
    col_pipe = body.add_stream("col_pipe", itype, storage=StorageType.FPGA_Local)

    tasklet = body.add_tasklet("read_col", {"col_in"}, {"col_out"}, "col_out = col_in[row_begin + c]")

    body.add_memlet_path(a_col, tasklet, dst_conn="col_in", memlet=Memlet.simple(a_col, "0:nnz"))
    body.add_memlet_path(tasklet, col_pipe, src_conn="col_out", memlet=Memlet.simple(col_pipe, "0"))

    return sdfg


def make_read_row():

    sdfg = SDFG("spmv_read_row")

    begin = sdfg.add_state("begin")
    entry = sdfg.add_state("entry")
    end = sdfg.add_state("end")
    body = sdfg.add_state("body")

    sdfg.add_edge(begin, entry, InterstateEdge(assignments={"h": "0"}))
    sdfg.add_edge(entry, body,
                  InterstateEdge(condition=CodeProperty.from_string("h < rows + 1", language=Language.Python)))
    sdfg.add_edge(entry, end,
                  InterstateEdge(condition=CodeProperty.from_string("h >= rows + 1", language=Language.Python)))
    sdfg.add_edge(body, entry, InterstateEdge(assignments={"h": "h + 1"}))

    a_row_mem = body.add_array("A_row_mem", (rows + 1, ), itype, storage=StorageType.FPGA_Global)
    to_val_pipe = body.add_stream("to_val_pipe", itype, storage=StorageType.FPGA_Local)
    to_col_pipe = body.add_stream("to_col_pipe", itype, storage=StorageType.FPGA_Local)
    to_compute_pipe = body.add_stream("to_compute_pipe", itype, storage=StorageType.FPGA_Local)
    to_x_pipe = body.add_stream("to_x_pipe", itype, storage=StorageType.FPGA_Local)
    tasklet = body.add_tasklet(
        "read_row", {"row_in"}, {"to_val_out", "to_col_out", "to_compute_out", "to_x_out"}, "to_val_out = row_in\n"
        "to_col_out = row_in\n"
        "to_compute_out = row_in\n"
        "to_x_out = row_in")

    body.add_memlet_path(a_row_mem, tasklet, dst_conn="row_in", memlet=Memlet.simple(a_row_mem, "h"))
    body.add_memlet_path(tasklet, to_val_pipe, src_conn="to_val_out", memlet=Memlet.simple(to_val_pipe, "0"))
    body.add_memlet_path(tasklet, to_col_pipe, src_conn="to_col_out", memlet=Memlet.simple(to_col_pipe, "0"))
    body.add_memlet_path(tasklet,
                         to_compute_pipe,
                         src_conn="to_compute_out",
                         memlet=Memlet.simple(to_compute_pipe, "0"))
    body.add_memlet_path(tasklet, to_x_pipe, src_conn="to_x_out", memlet=Memlet.simple(to_x_pipe, "0"))

    return sdfg


def make_main_state(sdfg):

    state = sdfg.add_state("spmv")

    # Read row pointers and send to value and column readers
    a_row = state.add_array("A_row_device", (rows + 1, ), itype, transient=True, storage=StorageType.FPGA_Global)
    row_to_val_out = state.add_stream("row_to_val", itype, transient=True, storage=StorageType.FPGA_Local)
    row_to_col_out = state.add_stream("row_to_col", itype, transient=True, storage=StorageType.FPGA_Local)
    row_to_x_out = state.add_stream("row_to_x", itype, transient=True, storage=StorageType.FPGA_Local)
    row_to_compute_out = state.add_stream("row_to_compute", itype, transient=True, storage=StorageType.FPGA_Local)
    read_row_sdfg = make_read_row()
    read_row_tasklet = state.add_nested_sdfg(read_row_sdfg, sdfg, {"A_row_mem"},
                                             {"to_val_pipe", "to_col_pipe", "to_x_pipe", "to_compute_pipe"})
    state.add_memlet_path(a_row,
                          read_row_tasklet,
                          memlet=dace.memlet.Memlet.simple(a_row, "0:rows+1"),
                          dst_conn="A_row_mem")
    state.add_memlet_path(read_row_tasklet,
                          row_to_val_out,
                          memlet=dace.memlet.Memlet.simple(row_to_val_out, "0", num_accesses=-1),
                          src_conn="to_val_pipe")
    state.add_memlet_path(read_row_tasklet,
                          row_to_col_out,
                          memlet=dace.memlet.Memlet.simple(row_to_col_out, "0", num_accesses=-1),
                          src_conn="to_col_pipe")
    state.add_memlet_path(read_row_tasklet,
                          row_to_x_out,
                          memlet=dace.memlet.Memlet.simple(row_to_x_out, "0", num_accesses=-1),
                          src_conn="to_x_pipe")
    state.add_memlet_path(read_row_tasklet,
                          row_to_compute_out,
                          memlet=dace.memlet.Memlet.simple(row_to_compute_out, "0", num_accesses=-1),
                          src_conn="to_compute_pipe")

    # Read columns of A using row pointers and send to x reader
    a_col = state.add_array("A_col_device", (nnz, ), itype, transient=True, storage=StorageType.FPGA_Global)
    row_to_col_in = state.add_stream("row_to_col", itype, transient=True, storage=StorageType.FPGA_Local)
    col_to_x_out = state.add_stream("col_to_x", itype, transient=True, storage=StorageType.FPGA_Local)
    read_col_sdfg = make_read_col()
    read_col_tasklet = state.add_nested_sdfg(read_col_sdfg, sdfg, {"A_col_mem", "row_pipe"}, {"col_pipe"})
    state.add_memlet_path(a_col,
                          read_col_tasklet,
                          memlet=dace.memlet.Memlet.simple(a_col, "0:nnz"),
                          dst_conn="A_col_mem")
    state.add_memlet_path(row_to_col_in,
                          read_col_tasklet,
                          memlet=dace.memlet.Memlet.simple(row_to_col_in, "0", num_accesses=-1),
                          dst_conn="row_pipe")
    state.add_memlet_path(read_col_tasklet,
                          col_to_x_out,
                          memlet=dace.memlet.Memlet.simple(col_to_x_out, "0", num_accesses=-1),
                          src_conn="col_pipe")

    # Read values of A using row pointers and send to compute
    a_val = state.add_array("A_val_device", (nnz, ), dtype, transient=True, storage=StorageType.FPGA_Global)
    row_to_val_in = state.add_stream("row_to_val", itype, transient=True, storage=StorageType.FPGA_Local)
    val_to_compute_out = state.add_stream("val_to_compute", dtype, transient=True, storage=StorageType.FPGA_Local)
    read_val_sdfg = make_read_val()
    read_val_tasklet = state.add_nested_sdfg(read_val_sdfg, sdfg, {"A_val_mem", "row_pipe"}, {"compute_pipe"})
    state.add_memlet_path(a_val,
                          read_val_tasklet,
                          dst_conn="A_val_mem",
                          memlet=dace.memlet.Memlet.simple(a_val, "0:nnz"))
    state.add_memlet_path(row_to_val_in,
                          read_val_tasklet,
                          dst_conn="row_pipe",
                          memlet=dace.memlet.Memlet.simple(row_to_val_in, "0", num_accesses=-1))
    state.add_memlet_path(read_val_tasklet,
                          val_to_compute_out,
                          src_conn="compute_pipe",
                          memlet=dace.memlet.Memlet.simple(val_to_compute_out, "0", num_accesses=-1))

    # Read values of x using column pointers and send to compute
    x = state.add_array("x_device", (cols, ), dtype, transient=True, storage=StorageType.FPGA_Global)
    row_to_x_in = state.add_stream("row_to_x", itype, transient=True, storage=StorageType.FPGA_Local)
    col_to_x_in = state.add_stream("col_to_x", itype, transient=True, storage=StorageType.FPGA_Local)
    x_to_compute_out = state.add_stream("x_to_compute", dtype, transient=True, storage=StorageType.FPGA_Local)
    read_x_sdfg = make_read_x()
    read_x_tasklet = state.add_nested_sdfg(read_x_sdfg, sdfg, {"x_mem", "col_pipe", "row_pipe"}, {"compute_pipe"})
    state.add_memlet_path(x, read_x_tasklet, dst_conn="x_mem", memlet=dace.memlet.Memlet.simple(x, "0:cols"))
    state.add_memlet_path(col_to_x_in,
                          read_x_tasklet,
                          dst_conn="col_pipe",
                          memlet=dace.memlet.Memlet.simple(col_to_x_in, "0", num_accesses=-1))
    state.add_memlet_path(row_to_x_in,
                          read_x_tasklet,
                          dst_conn="row_pipe",
                          memlet=dace.memlet.Memlet.simple(row_to_x_in, "0", num_accesses=-1))
    state.add_memlet_path(read_x_tasklet,
                          x_to_compute_out,
                          src_conn="compute_pipe",
                          memlet=dace.memlet.Memlet.simple(x_to_compute_out, "0", num_accesses=-1))

    # Receive values of A and x and compute resulting values of b
    row_to_compute_in = state.add_stream("row_to_compute", itype, transient=True, storage=StorageType.FPGA_Local)
    val_to_compute_in = state.add_stream("val_to_compute", dtype, transient=True, storage=StorageType.FPGA_Local)
    x_to_compute_in = state.add_stream("x_to_compute", dtype, transient=True, storage=StorageType.FPGA_Local)
    result_to_write_out = state.add_stream("result_to_write", dtype, transient=True, storage=StorageType.FPGA_Local)
    compute_sdfg = make_compute_sdfg()
    compute_tasklet = state.add_nested_sdfg(compute_sdfg, sdfg, {"row_pipe", "a_pipe", "x_pipe"}, {"b_pipe"})
    state.add_memlet_path(row_to_compute_in,
                          compute_tasklet,
                          dst_conn="row_pipe",
                          memlet=dace.memlet.Memlet.simple(row_to_compute_out, "0", num_accesses=-1))
    state.add_memlet_path(val_to_compute_in,
                          compute_tasklet,
                          dst_conn="a_pipe",
                          memlet=dace.memlet.Memlet.simple(val_to_compute_in, "0", num_accesses=-1))
    state.add_memlet_path(x_to_compute_in,
                          compute_tasklet,
                          dst_conn="x_pipe",
                          memlet=dace.memlet.Memlet.simple(x_to_compute_in, "0", num_accesses=-1))
    state.add_memlet_path(compute_tasklet,
                          result_to_write_out,
                          src_conn="b_pipe",
                          memlet=dace.memlet.Memlet.simple(result_to_write_out, "0", num_accesses=-1))

    # Write back values of b
    result_to_write_in = state.add_stream("result_to_write", dtype, transient=True, storage=StorageType.FPGA_Local)
    b = state.add_array("b_device", (rows, ), dtype, transient=True, storage=StorageType.FPGA_Global)
    write_sdfg = make_write_sdfg()
    write_tasklet = state.add_nested_sdfg(write_sdfg, sdfg, {"b_pipe"}, {"b_mem"})
    state.add_memlet_path(result_to_write_in,
                          write_tasklet,
                          dst_conn="b_pipe",
                          memlet=dace.memlet.Memlet.simple(result_to_write_in, "0", num_accesses=-1))
    state.add_memlet_path(write_tasklet, b, src_conn="b_mem", memlet=dace.memlet.Memlet.simple(b, "0:rows"))

    return state


def make_nested_compute_state(sdfg):

    state = sdfg.add_state("spmv")

    row_entry, row_exit = state.add_map("compute_row", {"i": "0:rows"}, schedule=ScheduleType.FPGA_Device)

    rowptr = state.add_scalar("rowptr", itype, storage=StorageType.FPGA_Registers, transient=True)
    rowend = state.add_scalar("rowend", itype, storage=StorageType.FPGA_Registers, transient=True)

    nested_sdfg = make_nested_sdfg(sdfg)
    nested_sdfg_tasklet = state.add_nested_sdfg(nested_sdfg, sdfg,
                                                {"row_begin", "row_end", "A_val_read", "A_col_read", "x_read"},
                                                {"b_write"})

    state.add_memlet_path(a_row, row_entry, rowptr, memlet=dace.memlet.Memlet.simple(rowptr, "0", other_subset_str="i"))
    state.add_memlet_path(rowptr,
                          nested_sdfg_tasklet,
                          dst_conn="row_begin",
                          memlet=dace.memlet.Memlet.simple(rowptr, "0"))

    state.add_memlet_path(a_row,
                          row_entry,
                          rowend,
                          memlet=dace.memlet.Memlet.simple(rowend, "0", other_subset_str="i + 1"))
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
                          memlet=dace.memlet.Memlet.simple(x, "0:cols"))

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
        name = "spmv_fpga_stream_{}x{}x{}".format(rows.get(), cols.get(), nnz.get())
    else:
        name = "spmv_fpga_stream"
    sdfg = dace.SDFG(name)

    pre_state = make_pre_state(sdfg)
    main_state = make_main_state(sdfg)
    post_state = make_post_state(sdfg)

    sdfg.add_edge(pre_state, main_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(main_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


def run_spmv(size_w, size_h, num_nonzero, specialize):

    cols.set(size_w)
    rows.set(size_h)
    nnz.set(num_nonzero)

    print("Sparse Matrix-Vector Multiplication {}x{} "
          "({} non-zero elements, {}specialized)".format(cols.get(), rows.get(), nnz.get(),
                                                         "not " if not specialize else ""))

    A_row = dace.ndarray([rows + 1], dtype=itype)
    A_col = dace.ndarray([nnz], dtype=itype)
    A_val = dace.ndarray([nnz], dtype=dtype)

    x = dace.ndarray([cols], dtype)
    b = dace.ndarray([rows], dtype)

    # Assuming uniform sparsity distribution across rows
    nnz_per_row = nnz.get() // rows.get()
    nnz_last_row = nnz_per_row + (nnz.get() % rows.get())
    if nnz_last_row > cols.get():
        print("Too many nonzeros per row")
        exit(1)

    # RANDOMIZE SPARSE MATRIX
    A_row[0] = itype(0)
    A_row[1:rows.get()] = itype(nnz_per_row)
    A_row[-1] = itype(nnz_last_row)
    A_row = np.cumsum(A_row, dtype=itype.type)

    # Fill column data
    for i in range(rows.get() - 1):
        A_col[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(cols.get(), nnz_per_row, replace=False))
    # Fill column data for last row
    A_col[nnz_per_row * (rows.get() - 1):] = np.sort(np.random.choice(cols.get(), nnz_last_row, replace=False))

    A_val[:] = np.random.rand(nnz.get()).astype(dtype.type)
    #########################

    x[:] = np.random.rand(cols.get()).astype(dtype.type)
    #b[:] = dtype(0)

    # Setup regression
    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), shape=(rows.get(), cols.get()))

    spmv = make_sdfg(specialize)
    if specialize:
        spmv.specialize(dict(rows=rows, cols=cols, nnz=nnz))
    spmv(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, rows=rows, cols=cols, nnz=nnz)

    if dace.Config.get_bool("profiling"):
        dace.timethis("spmv", "scipy", 0, A_sparse.dot, x)

    diff = np.linalg.norm(A_sparse.dot(x) - b) / float(rows.get())
    print("Difference:", diff)
    if diff >= 1e-5:
        print("Validation failed.")
        print("Result:")
        print(b)
        print("Reference:")
        print(A_sparse.dot(x))
        print("Type \"debug\" to enter debugger, " "or any other string to quit (timeout in 10 seconds)")
        read, _, _ = select.select([sys.stdin], [], [], 10)
        if len(read) > 0 and sys.stdin.readline().strip().lower() == "debug":
            print("Entering debugger...")
            pdb.set_trace()
        else:
            print("Exiting...")
    print("==== Program end ====")
    if diff > 1e-5:
        raise RuntimeError("Validation failed.")

    return spmv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cols", type=int)
    parser.add_argument("rows", type=int)
    parser.add_argument("nnz", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all symbols at compile time/in hardware")
    args = parser.parse_args()

    run_spmv(args.cols, args.rows, args.nnz, args.specialize)
