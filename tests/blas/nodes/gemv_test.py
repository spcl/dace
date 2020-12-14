#!/usr/bin/env python3
# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
import dace.libraries.blas.utility.fpga_helper as streaming


# ---------- ----------
# Pure graph program (CPU)
# ---------- ----------
def pure_graph(dtype, transposed, expansion, veclen):
    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("gemv")

    if veclen != 1:
        raise NotImplementedError("Vectorization not implemented for pure.")

    # alpha and beta are symbols
    sdfg.add_symbol("alpha", dtype)
    sdfg.add_symbol("beta", dtype)

    state = sdfg.add_state("gemv_compute")

    A_rows = n
    A_cols = m
    x_size = n if transposed else m
    y_size = m if transposed else n

    sdfg.add_array('A', shape=[A_rows, A_cols], dtype=dtype)
    sdfg.add_array('x', shape=[x_size], dtype=dtype)
    sdfg.add_array('y', shape=[y_size], dtype=dtype)

    A = state.add_read("A")
    x = state.add_read("x")
    result = state.add_write("y")

    gemv_node = blas.Gemv("gemv", dtype=dace.float32, transA=transposed)
    gemv_node.implementation = expansion

    state.add_memlet_path(A,
                          gemv_node,
                          dst_conn="_A",
                          memlet=Memlet.simple(
                              A, "0:{}, 0:{}".format(A_rows, A_cols)))
    state.add_memlet_path(x,
                          gemv_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x, "0:{}".format(x_size)))
    y = state.add_read("y")
    state.add_memlet_path(y,
                          gemv_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y, "0:{}".format(y_size)))
    state.add_memlet_path(gemv_node,
                          result,
                          src_conn="_y",
                          memlet=Memlet.simple(result, "0:{}".format(y_size)))
    return sdfg


# ---------- ----------
# Intel FPGA graph
# ---------- ----------
def fpga_graph(dtype, transposed, expansion, vec_width):

    vtype = dace.vector(dtype, vec_width)

    sdfg = dace.SDFG("gemv_fpga_test")

    n = dace.symbol("n")
    m = dace.symbol("m")

    m /= vec_width

    # alpha and beta are symbols
    sdfg.add_symbol("alpha", dtype)
    sdfg.add_symbol("beta", dtype)

    A_rows = n
    A_cols = m
    x_size = n if transposed else m
    y_size = m if transposed else n

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", shape=[n, m], dtype=vtype)
    sdfg.add_array("x",
                   shape=[x_size],
                   dtype=vtype if not transposed else dtype)
    sdfg.add_array("y", shape=[y_size], dtype=vtype if transposed else dtype)

    in_host_A = copy_in_state.add_read("A")
    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")

    sdfg.add_array("device_A",
                   shape=[A_rows, A_cols],
                   dtype=vtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_x",
                   shape=[x_size],
                   dtype=vtype if not transposed else dtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_y",
                   shape=[y_size],
                   dtype=vtype if transposed else dtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_A = copy_in_state.add_write("device_A")
    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    copy_in_state.add_memlet_path(in_host_A,
                                  in_device_A,
                                  memlet=Memlet.simple(
                                      in_host_A,
                                      "0:{}, 0:{}".format(A_rows, A_cols)))
    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(in_host_x,
                                                       "0:{}".format(x_size)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(in_host_y,
                                                       "0:{}".format(y_size)))

    ###########################################################################
    # Copy data from FPGA

    copy_out_state = sdfg.add_state("copy_to_host")

    out_device = copy_out_state.add_read("device_y")
    out_host = copy_out_state.add_write("y")

    copy_out_state.add_memlet_path(out_device,
                                   out_host,
                                   memlet=Memlet.simple(out_host,
                                                        "0:{}".format(y_size)))

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("gemv_computation")
    # This should not be an FPGA kernel, rather the gemv_expanded nested SDFG should

    A = fpga_state.add_read("device_A")
    x = fpga_state.add_read("device_x")
    y_out = fpga_state.add_write("device_y")

    gemv_node = blas.Gemv("gemv",
                          dtype=dace.float32,
                          transA=transposed,
                          alpha=alpha,
                          beta=beta)
    gemv_node.implementation = expansion

    fpga_state.add_memlet_path(A,
                               gemv_node,
                               dst_conn="_A",
                               memlet=Memlet.simple(A,
                                                    "0:{}, 0:{}".format(n, m)))

    fpga_state.add_memlet_path(x,
                               gemv_node,
                               dst_conn="_x",
                               memlet=Memlet.simple(
                                   x, "0:{}".format("{}".format(x_size))))

    if beta != 0:
        y_in = fpga_state.add_read("device_y")
        fpga_state.add_memlet_path(y_in,
                                   gemv_node,
                                   dst_conn="_y",
                                   memlet=Memlet.simple(y_in,
                                                        "0:{}".format(y_size)))
    fpga_state.add_memlet_path(gemv_node,
                               y_out,
                               src_conn="_y",
                               memlet=Memlet.simple(y_out,
                                                    "0:{}".format(y_size)))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

    gemv_node.expand(sdfg, fpga_state, tile_m_size=16, tile_n_size=16, vec_width=vec_width)

    sdfg.fill_scope_connectors()
    return sdfg


def fpga_row_streamd_graph(dtype, vendor, n_tile=4, m_tile=4, vec_width=1):

    nRows = dace.symbol("n")
    mCols = dace.symbol("m")

    a = dace.symbol("alpha")
    b = dace.symbol("beta")

    rowTile = n_tile
    colTile = m_tile
    partial_width = 4

    vec_type = dace.vector(dtype, vec_width)
    single_vec_type = dace.vector(dtype, 1)

    test_sdfg = dace.SDFG("gemv_testing_stream_" + vendor)
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, dace.float32)

    if b != 0:
        test_sdfg.add_symbol(b.name, dace.float32)

    test_sdfg.add_array('A', shape=[nRows * mCols / vec_width], dtype=vec_type)
    test_sdfg.add_array('x', shape=[mCols / vec_width], dtype=vec_type)
    test_sdfg.add_array('y', shape=[nRows], dtype=single_vec_type)
    test_sdfg.add_array('res', shape=[nRows], dtype=single_vec_type)

    x_stream = streaming.StreamReadVector('x',
                                          mCols,
                                          dace.float32,
                                          veclen=vec_width,
                                          repeat='{}/{}'.format(nRows, rowTile))

    y_stream = None
    if b != 0:
        y_stream = streaming.StreamReadVector(
            'y',
            nRows,
            dace.float32,
            veclen=1,
        )

    A_stream = streaming.StreamReadMatrixFull('A',
                                              nRows,
                                              mCols,
                                              rowTile,
                                              colTile,
                                              dace.float32,
                                              tileByRow=True,
                                              veclen=vec_width)

    res_stream = streaming.StreamWriteVector('res', nRows, dace.float32)

    gemv_node = blas.gemv.Gemv("blas_gemv",
                               dtype=dace.float32,
                               partial_width=partial_width,
                               n=nRows,
                               m=mCols,
                               alpha=a,
                               beta=b,
                               streaming=True)
    gemv_node.implementation = 'TilesByRow'

    if y_stream is not None:
        preState, postState = streaming.fpga_setup_connect_streamers(
            test_sdfg, test_state, gemv_node, [x_stream, y_stream, A_stream],
            ['_x', '_y', '_A'], gemv_node, [res_stream], ['_res'])
    else:
        preState, postState = streaming.fpga_setup_connect_streamers(
            test_sdfg, test_state, gemv_node, [x_stream, A_stream],
            ['_x', '_A'], gemv_node, [res_stream], ['_res'])

    gemv_node.expand(test_sdfg, test_state, tile_m_size=colTile, tile_n_size=rowTile, vec_width=vec_width)

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg


def fpga_row_array_graph(dtype, vendor, n_tile=4, m_tile=4, vec_width=1):

    DATATYPE = dtype

    n = dace.symbol("n")
    m = dace.symbol("m")
    a = dace.symbol("alpha")
    b = dace.symbol("beta")

    vendor_mark = "x" if vendor == "xilinx" else "i"
    test_sdfg = dace.SDFG("gemv_test_array_" + vendor_mark)

    test_sdfg.add_symbol(a.name, dtype)
    if b != 0:
        test_sdfg.add_symbol(b.name, dtype)

    vec_type = dace.vector(dtype, vec_width)
    single_vec_type = dace.vector(dtype, 1)

    test_sdfg.add_array('A', shape=[n * m / vec_width], dtype=vec_type)
    test_sdfg.add_array('x', shape=[m / vec_width], dtype=vec_type)
    test_sdfg.add_array('y', shape=[n], dtype=dtype)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = test_sdfg.add_state("copy_to_device")

    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")
    in_host_A = copy_in_state.add_read("A")

    test_sdfg.add_array("device_x",
                        shape=[m / vec_width],
                        dtype=vec_type,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)
    test_sdfg.add_array("device_y",
                        shape=[n],
                        dtype=dtype,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)
    test_sdfg.add_array("device_A",
                        shape=[n * m / vec_width],
                        dtype=vec_type,
                        storage=dace.dtypes.StorageType.FPGA_Global,
                        transient=True)

    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")
    in_device_A = copy_in_state.add_write("device_A")

    copy_in_state.add_memlet_path(in_host_x,
                                  in_device_x,
                                  memlet=Memlet.simple(
                                      in_host_x, "0:{}/{}".format(m,
                                                                  vec_width)))
    copy_in_state.add_memlet_path(in_host_y,
                                  in_device_y,
                                  memlet=Memlet.simple(in_host_y,
                                                       "0:{}".format(n)))
    copy_in_state.add_memlet_path(in_host_A,
                                  in_device_A,
                                  memlet=Memlet.simple(
                                      in_host_A,
                                      "0:{}".format(n * m / vec_width)))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = test_sdfg.add_state("copy_to_host")

    # test_sdfg.add_array("device_r",
    #                     shape=[n],
    #                     dtype=dtype,
    #                     storage=dace.dtypes.StorageType.FPGA_Global,
    #                     transient=True)

    out_device = copy_out_state.add_read("device_y")
    out_host = copy_out_state.add_write("y")

    copy_out_state.add_memlet_path(out_device,
                                   out_host,
                                   memlet=Memlet.simple(out_host,
                                                        "0:{}".format(n)))

    ########################################################################
    # FPGA State

    fpga_state = test_sdfg.add_state("fpga_state")

    x = fpga_state.add_read("device_x")
    y = fpga_state.add_read("device_y")
    y_out = fpga_state.add_write("device_y")
    A = fpga_state.add_read("device_A")

    gemv_node = blas.gemv.Gemv("blas_gemv",
                               dtype=dace.float32,
                               partial_width=8,
                               n=n,
                               m=m,
                               alpha=a,
                               beta=b,
                               streaming=False)
    gemv_node.implementation = 'TilesByRow'

    fpga_state.add_memlet_path(x,
                               gemv_node,
                               dst_conn="_x",
                               memlet=Memlet.simple(
                                   x, "0:{}".format(m / vec_width)))
    fpga_state.add_memlet_path(y,
                               gemv_node,
                               dst_conn="_y",
                               memlet=Memlet.simple(y, "0:{}".format(n)))
    fpga_state.add_memlet_path(A,
                               gemv_node,
                               dst_conn="_A",
                               memlet=Memlet.simple(
                                   A, "0:{}".format(n * m / vec_width)))
    fpga_state.add_memlet_path(gemv_node,
                               y_out,
                               src_conn="_y",
                               memlet=Memlet.simple(y_out, "0:{}".format(n)))

    ######################################
    # Interstate edges
    test_sdfg.add_edge(copy_in_state, fpga_state,
                       dace.sdfg.sdfg.InterstateEdge())
    test_sdfg.add_edge(fpga_state, copy_out_state,
                       dace.sdfg.sdfg.InterstateEdge())

    #########
    # Validate
    test_sdfg.fill_scope_connectors()
    test_sdfg.validate()

    
    gemv_node.expand(test_sdfg, fpga_state, tile_m_size=m_tile, tile_n_size=n_tile, vec_width=vec_width)

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=256)
    parser.add_argument("M", type=int, nargs="?", default=256)
    parser.add_argument("alpha", type=int, nargs="?", default=1)
    parser.add_argument("beta", type=int, nargs="?", default=0)
    parser.add_argument("--n_tile", type=int, nargs="?", default=16)
    parser.add_argument("--m_tile", type=int, nargs="?", default=16)
    parser.add_argument("--transposed",
                        action="store_true",
                        default=False,
                        help="Compute GEMV with transposed matrix")
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vectorize", dest="vectorize", default=1, type=int)
    parser.add_argument("--cached", action="store_true", default=False)

    args = parser.parse_args()
    n = args.N
    m = args.M
    alpha = args.alpha
    beta = args.beta
    transposed = args.transposed

    if args.cached:
        print("Using compilation cache")
        dace.config.Config.set("compiler", "use_cache", value="true")

    if args.target == "pure":
        sdfg = pure_graph(dace.float32, transposed, "pure", args.vectorize)
    elif args.target == "intel_fpga":
        sdfg = fpga_graph(dace.float32, transposed, "IntelFPGA", args.vectorize)
    elif args.target == "xilinx":
        sdfg = fpga_row_streamd_graph(dace.float32,
                                      "xilinx",
                                      n_tile=args.n_tile,
                                      m_tile=args.m_tile,
                                      vec_width=args.vectorize)
    elif args.target == "tiles_by_row_array":
        sdfg = fpga_row_array_graph(dace.float32,
                                    "xilinx",
                                    n_tile=args.n_tile,
                                    m_tile=args.m_tile,
                                    vec_width=args.vectorize)
    elif args.target == "tiles_by_column":
        if not transposed:
            raise RuntimeError(
                "Matrix must be transposed to use this expansion, "
                "please set --transpose")
        sdfg = fpga_graph(dace.float32, transposed, "TilesByColumn",
                          args.vectorize)
    else:
        print("Unsupported target")
        exit(-1)

    A = np.random.rand(n, m).astype(np.float32)
    x = np.random.rand(n if transposed else m).astype(np.float32)
    y = np.random.rand(m if transposed else n).astype(np.float32)
    res = np.zeros(m if transposed else n).astype(np.float32)

    # A = np.ones((n,m), dtype=np.float32)
    # x = np.array([1,2,3,4], dtype=np.float32)
    # y = np.zeros(n, dtype=np.float32)

    y_copy = np.copy(y)
    ref = scipy.linalg.blas.sgemv(alpha, A, x, beta, y_copy, trans=transposed)

    if args.target == "xilinx":
        sdfg(A=A, x=x, y=y, n=n, m=m, res=res, alpha=alpha, beta=beta)
        diff = np.linalg.norm(res - ref) / (m if transposed else n)
        result = res
    else:
        sdfg(A=A, x=x, y=y, n=n, m=m, alpha=alpha, beta=beta)
        diff = np.linalg.norm(y - ref) / (m if transposed else n)
        result = y

    if diff >= 1e-5:
        print("Error")
        raise RuntimeError("Unexpected result returned from gemv operation: "
                           "got:\n{}\nexpected:\n{} on {}".format(
                               result, ref, args.target))
    else:
        print("Ok")
