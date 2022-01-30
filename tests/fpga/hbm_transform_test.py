# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
 
from dace.fpga_testing import xilinx_test 
from numpy.lib import math
from dace.sdfg.state import SDFGState
import numpy as np
from dace import dtypes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from typing import List, Tuple
from dace.sdfg import SDFG, nodes
import dace
from dace.transformation.dataflow import HbmTransform
from dace.transformation.interstate import NestSDFG
from functools import reduce


def set_assignment(sdfg: SDFG, assignments: List[Tuple[str, str, str]]):
    for array, memorytype, bank in assignments:
        desc = sdfg.arrays[array]
        desc.location["memorytype"] = memorytype
        desc.location["bank"] = bank


def rand_float(input_shape):
    a = np.random.rand(*input_shape)
    a = a.astype(np.float32)
    #a = np.ones(input_shape, np.float32)
    return a


def _exec_hbmtransform(sdfg_source,
                       assign,
                       nest=False,
                       num_apply=1,
                       apply_to=None):
    sdfg = sdfg_source()
    set_assignment(sdfg, assign)
    if apply_to is None:
        assert sdfg.apply_transformations_repeated(HbmTransform, {
            "new_dim": "kw",
            "move_to_FPGA_global": False
        },
                                                   validate=False) == num_apply
        if num_apply == 0:
            return sdfg
    else:
        for map_entry in apply_to(sdfg):
            HbmTransform.apply_to(sdfg, {
                "new_dim": "kw",
                "move_to_FPGA_global": False
            },
                                  save=False,
                                  _map_entry=map_entry)
    if nest:
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.Default:
                desc.storage = dtypes.StorageType.FPGA_Global
        sdfg.apply_transformations(NestSDFG, validate=False)
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.FPGA_Global:
                desc.storage = dtypes.StorageType.Default
    sdfg.apply_fpga_transformations(validate=False)
    sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
    csdfg = sdfg.compile()
    return (csdfg, sdfg)


def create_vadd_sdfg(name,
                     array_shape=dace.symbol("n"),
                     map_range=dace.symbol("n")):
    @dace.program
    def vadd(x: dace.float32[array_shape], y: dace.float32[array_shape],
             z: dace.float32[array_shape]):
        for i in dace.map[0:map_range]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                zout >> z[i]
                zout = xin + yin

    sdfg = vadd.to_sdfg()
    sdfg.name = name
    sdfg.apply_strict_transformations()
    return sdfg


def create_multi_access_sdfg(name):
    N = dace.symbol("N")

    @dace.program
    def sth(z: dace.float32[N], x: dace.float32[N], y: dace.float32[N],
            w: dace.float32[N], o1: dace.float32[N], o2: dace.float32[N]):
        for i in dace.map[0:N]:
            o1[i] = z[i] + x[i]
        for i in dace.map[0:N]:
            o2[i] = w[i] + y[i]

    sdfg = sth.to_sdfg()
    sdfg.name = name
    sdfg.apply_strict_transformations()
    return sdfg


def create_nd_sdfg(name):
    n = dace.symbol("n")
    m = dace.symbol("m")

    @dace.program
    def nd_sdfg(x: dace.float32[n, m], y: dace.float32[m, n],
                z: dace.float32[n, m]):
        for i in dace.map[0:n]:
            for j in dace.map[0:m]:
                with dace.tasklet:
                    yin << y[j, i]
                    xin << x[i, j]
                    zout >> z[i, j]
                    zout = yin + xin

    sdfg = nd_sdfg.to_sdfg()
    sdfg.name = name
    sdfg.apply_strict_transformations()
    return sdfg


def create_gemv_blas_sdfg(name, tile_size_y=None, tile_size_x=None, m=None):
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def gemv(A: dace.float32[M, N], x: dace.float32[N], y: dace.float32[M]):
        y[:] = A @ x

    sdfg = gemv.to_sdfg()
    sdfg.apply_strict_transformations()
    if m is not None:
        sdfg.specialize({M: m})
    libnode = list(
        filter(lambda x: isinstance(x, nodes.LibraryNode),
               sdfg.nodes()[0].nodes()))[0]
    libnode.expand(sdfg, sdfg.nodes()[0])
    libnode = list(
        filter(lambda x: isinstance(x, nodes.LibraryNode),
               sdfg.nodes()[0].nodes()))[0]
    libnode.implementation = "FPGA_TilesByColumn"
    libnode.expand(sdfg,
                   sdfg.nodes()[0],
                   tile_size_y=tile_size_y,
                   tile_size_x=tile_size_x)
    sdfg.apply_strict_transformations()
    sdfg.name = name
    return sdfg


def validate_vadd_sdfg(csdfg, input_shape):
    a = rand_float(input_shape)
    b = rand_float(input_shape)
    c = rand_float(input_shape)
    expect = a + b

    csdfg(x=a, y=b, z=c, n=reduce(lambda x, y: x * y, input_shape))
    assert np.allclose(expect, c)


def validate_gemv_sdfg(csdfg, matrix_shape, x_shape, y_shape):
    # A and potentially y is assumed to be split along dim 0
    A = rand_float(matrix_shape)
    x = rand_float(x_shape)
    y = rand_float(y_shape)
    expect = np.matmul(A, x)

    csdfg(A=A, x=x, y=y, M=matrix_shape[0] * matrix_shape[1], N=matrix_shape[2])
    if len(y_shape) == 1:
        y = np.reshape(y, [matrix_shape[0], matrix_shape[1]])
    assert np.allclose(y, expect)


def validate_nd_sdfg(csdfg, m, n, divide_m=1, divide_n=1):
    A = np.zeros([divide_m * divide_n, n // divide_n, m // divide_m],
                 np.float32)
    B = np.zeros([divide_m * divide_n, m // divide_m, n // divide_n],
                 np.float32)
    Z = np.zeros([divide_m * divide_n, n // divide_n, m // divide_m],
                 np.float32)
    expect = np.zeros([divide_m * divide_n, n // divide_n, m // divide_m],
                      np.float32)

    for k_i in range(1, divide_n + 1):
        for k_j in range(1, divide_m + 1):
            for i in range(n // divide_n):
                for j in range(m // divide_m):
                    index = k_i * k_j - 1
                    A[index, i, j] = np.random.random()
                    B[index, j, i] = np.random.random()
                    expect[index, i, j] = A[index, i, j] + B[index, j, i]

    csdfg(x=A, y=B, z=Z, m=m, n=n)
    assert np.allclose(expect, Z)


@xilinx_test(run_synthesis=False)
def test_axpy_unroll_3():
    csdfg, sdfg = _exec_hbmtransform(lambda: create_vadd_sdfg("axpy_unroll_3"),
                                     [("x", "HBM", "3:6"), ("y", "HBM", "0:3"),
                                      ("z", "HBM", "6:9")])
    validate_vadd_sdfg(csdfg, [3, 20])
    return sdfg


@xilinx_test(run_synthesis=False)
def test_axpy_unroll_mixed():
    csdfg, sdfg = _exec_hbmtransform(lambda: create_vadd_sdfg("axpy_mixed"),
                                     [("x", "DDR", "0"), ("y", "HBM", "0:2"),
                                      ("z", "HBM", "0:2")])
    validate_vadd_sdfg(csdfg, [2, 20])
    return sdfg


@xilinx_test(run_synthesis=False)
def test_nd_split():
    csdfg, sdfg = _exec_hbmtransform(lambda: create_nd_sdfg("nd_split"),
                                     [("x", "HBM", "0:10"),
                                      ("y", "HBM", "10:20"),
                                      ("z", "HBM", "20:30")])
    validate_nd_sdfg(csdfg, 10, 10, divide_n=10)
    return sdfg


@xilinx_test(run_synthesis=False)
def test_nd_split_inner():
    def apply_to(sdfg):
        state: SDFGState = sdfg.start_state
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node.map.params[0] == "j":
                return [node]

    csdfg, sdfg = _exec_hbmtransform(lambda: create_nd_sdfg("nd_split_inner"),
                                     [("x", "HBM", "0:10"),
                                      ("y", "HBM", "10:20"),
                                      ("z", "HBM", "20:30")],
                                     apply_to=apply_to)
    validate_nd_sdfg(csdfg, 10, 10, divide_m=10)
    return sdfg


@xilinx_test(run_synthesis=False)
def test_gemv_blas_1():
    csdfg, sdfg = _exec_hbmtransform(
        lambda: create_gemv_blas_sdfg("gemv_1", 32), [("x", "HBM", "31:32"),
                                                      ("y", "HBM", "30:31"),
                                                      ("A", "HBM", "0:30")],
        True)
    validate_gemv_sdfg(csdfg, [30, 32, 5], [5], [32 * 30])
    return sdfg


@xilinx_test(run_synthesis=False)
def test_gemv_blas_2():
    csdfg, sdfg = _exec_hbmtransform(
        lambda: create_gemv_blas_sdfg("gemv_2", 32), [("x", "HBM", "31:32"),
                                                      ("y", "HBM", "15:30"),
                                                      ("A", "HBM", "0:15")],
        True)
    validate_gemv_sdfg(csdfg, [15, 32, 5], [5], [15, 32])
    return sdfg


@xilinx_test(run_synthesis=False)
def test_multiple_applications():
    _, sdfg = _exec_hbmtransform(
        lambda: create_multi_access_sdfg("multi_access"),
        [("x", "HBM", "0:2"), ("y", "HBM", "2:4"), ("z", "HBM", "4:6"),
         ("w", "HBM", "10:12"), ("o1", "HBM", "6:8"), ("o2", "HBM", "8:10")],
        num_apply=2)
    return sdfg


# This does not run with synthesis enabled
@xilinx_test(run_synthesis=False)
def test_axpy_unroll_1():
    # This SDFG is fine, but we would do nothing at all
    sdfg = _exec_hbmtransform(lambda: create_vadd_sdfg("axpy_unroll_1"),
                              [("x", "DDR", "0"), ("y", "HBM", "0:1"),
                               ("z", "DDR", "1")],
                              num_apply=0)
    sdfg.compile(
    )  # We still have to compile for pytest, so the build folder exists
    return sdfg


# This does not run with synthesis enabled
@xilinx_test(run_synthesis=False)
def test_axpy_inconsistent_no_apply():
    sdfg = _exec_hbmtransform(lambda: create_vadd_sdfg("axpy_inconsistent"),
                              [("x", "HBM", "0:2"), ("y", "DDR", "0"),
                               ("z", "HBM", "0:3")],
                              num_apply=0)
    set_assignment(sdfg, [("x", "DDR", "0"), ("y", "HBM", "0:1"),
                          ("z", "DDR", "1")])
    sdfg.compile(
    )  # We still have to compile for pytest, so the build folder exists
    return sdfg


if __name__ == "__main__":
    test_axpy_unroll_3(None)
    test_axpy_unroll_1(None)
    test_axpy_unroll_mixed(None)
    test_nd_split(None)
    test_nd_split_inner(None)
    test_gemv_blas_1(None)
    test_gemv_blas_2(None)
    test_axpy_inconsistent_no_apply(None)
    test_multiple_applications(None)
