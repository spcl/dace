#!/usr/bin/env python3

import numpy as np

import argparse
import scipy
import random

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
import dace.libraries.blas.utility.fpga_helper as streaming
from dace.libraries.blas.utility import memory_operations as memOps
from dace.transformation.interstate import GPUTransformSDFG

from dace.libraries.standard.memory import aligned_ndarray

from multiprocessing import Process, Queue

# ---------- ----------
# FPGA graph program
# ---------- ----------
def fpga_graph(veclen, precision, vendor, testCase="0"):

    DATATYPE = precision
    nRows = dace.symbol("nRows")
    mCols = dace.symbol("mCols")

    n = dace.symbol("n")
    m = dace.symbol("m")
    a = dace.symbol("alpha")

    # TODO: support more tile sizes via test config
    rowTile = 4
    colTile = 4
    vecM = veclen

    vendor_mark = "x" if vendor == "xilinx" else "i"
    test_sdfg = dace.SDFG("ger_test_" + vendor_mark + "_" + testCase)
    test_state = test_sdfg.add_state("test_state")

    test_sdfg.add_symbol(a.name, DATATYPE)

    test_sdfg.add_array('A', shape=[n*m], dtype=DATATYPE)
    test_sdfg.add_array('x', shape=[m], dtype=DATATYPE)
    test_sdfg.add_array('y', shape=[n], dtype=DATATYPE)
    test_sdfg.add_array('r', shape=[n*m], dtype=DATATYPE)

    A_stream = streaming.StreamReadMatrixFull(
        'A',
        n,
        m,
        rowTile,
        colTile,
        DATATYPE,
        tileByRow=True,
        veclen=vecM
    )

    y_stream = streaming.StreamReadVector(
        'y',
        n,
        DATATYPE
    )

    x_stream = streaming.StreamReadVector(
        'x',
        m,
        DATATYPE,
        repeat='{}/{}'.format(n, rowTile),
        veclen=vecM
    )

    res_stream = streaming.StreamWriteMatrixFull(
        'r',
        n,
        m,
        rowTile,
        colTile,
        DATATYPE,
        tileByRow=True
    )


    ger_node = blas.Ger(
        "blas_ger",
        dtype=DATATYPE,
        nTile = rowTile,
        mTile = colTile,
        n=n,
        m=m,
        vecWidthM=vecM,
        a=a
    )
    ger_node.implementation = 'fpga_stream'

    preState, postState = streaming.fpga_setup_connect_streamers(
        test_sdfg,
        test_state,
        ger_node,
        [x_stream, y_stream, A_stream],
        ['_x', '_y', '_A'],
        ger_node,
        [res_stream],
        ['_res']
    )

    test_sdfg.expand_library_nodes()

    mode = "simulation" if vendor == "xilinx" else "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    return test_sdfg


# ---------- ----------
# Pure graph program (CPU)
# ---------- ----------
def pure_graph(dtype):

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG(
        "ger_operation")  # rank 1 operation: r = alpha * x * yT + A

    state = sdfg.add_state("ger")

    sdfg.add_symbol("alpha", dtype)

    sdfg.add_array("x", shape=[m], dtype=dtype)
    sdfg.add_array("y", shape=[n], dtype=dtype)
    sdfg.add_array("A", shape=[m, n], dtype=dtype)
    sdfg.add_array("r", shape=[m, n], dtype=dtype)  # result

    x = state.add_read("x")
    y = state.add_read("y")
    A = state.add_read("A")
    result = state.add_write("r")

    ger_node = blas.Ger(name="ger", dtype=dtype)
    ger_node.implementation = "pure"

    state.add_memlet_path(x,
                          ger_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x, "0:m", num_accesses=m))
    state.add_memlet_path(y,
                          ger_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(A,
                          ger_node,
                          dst_conn="_A",
                          memlet=Memlet.simple(A,
                                               "0:m, 0:n",
                                               num_accesses=m * n))
    state.add_memlet_path(ger_node,
                          result,
                          src_conn="_res",
                          memlet=Memlet.simple(result,
                                               "0:m, 0:n",
                                               num_accesses=m * n))

    sdfg.validate()
    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=4)
    parser.add_argument("M", type=int, nargs="?", default=4)
    parser.add_argument("alpha", type=np.float32, nargs="?", default=1.0)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()
    n = args.N
    m = args.M
    alpha = args.alpha

    if args.target == "pure":
        sdfg = pure_graph(dace.float32)
    elif args.target == "intel_fpga":
        raise NotImplementedError()
    elif args.target == "xilinx":
        sdfg = fpga_graph(1, dace.float32, args.target, "0")
    else:
        print("Unsupported target")
        exit(-1)

    ger = sdfg.compile()
    sdfg.save('aoeu.sdfg')

    x = np.ndarray(m, dtype=np.float32)
    y = np.ndarray(n, dtype=np.float32)
    A = np.ndarray((m, n), dtype=np.float32)
    result = np.ndarray((m, n), dtype=np.float32)

    x[:] = np.random.rand(m).astype(np.float32)
    y[:] = np.random.rand(n).astype(np.float32)
    A[:] = np.random.rand(m, n).astype(np.float32)
    result[:] = np.zeros((m, n)).astype(np.float32)

    ger(alpha=alpha, x=x, y=y, A=A, r=result, m=m, n=n)

    ref = scipy.linalg.blas.sger(alpha=alpha, x=x, y=y, a=A)

    diff = np.linalg.norm(np.subtract(result, ref))
    if diff >= args.eps * n * m:
        print("Unexpected result returned from ger rank 1 operation ({}): "
            "got:\n{}\nexpected:\n{}\ndiffs:\n{}".format(diff,result, ref, ref-result))
    else:
        print("Ok")
