#!/usr/bin/env python3

import numpy as np

import argparse
import scipy
import random

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas

from dace.libraries.standard.memory import aligned_ndarray


def pure_graph(dtype):

    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("ger_test")

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

    ger_node = blas.Ger(name="ger")
    ger_node.implementation = "pure"

    state.add_memlet_path(x, ger_node, dst_conn="_x", memlet=Memlet(f"x[0:m]"))
    state.add_memlet_path(y, ger_node, dst_conn="_y", memlet=Memlet("y[0:n]"))
    state.add_memlet_path(A,
                          ger_node,
                          dst_conn="_A",
                          memlet=Memlet("A[0:m, 0:n]"))
    state.add_memlet_path(ger_node,
                          result,
                          src_conn="_res",
                          memlet=Memlet("r[0:m, 0:n]"))

    return sdfg


def fpga_graph(dtype, tile_size_x, tile_size_y, veclen, alpha, implementation):

    m = "m"
    n = f"n/{veclen}"

    vtype = dace.vector(dtype, veclen)

    sdfg = dace.SDFG(f"ger_test_w{veclen}_x{tile_size_x}_y{tile_size_y}")
    pre_state = sdfg.add_state("pre_ger_test")
    post_state = sdfg.add_state("post_ger_test")
    state = sdfg.add_state("ger_test")
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    sdfg.add_array("A", (m, n), vtype)
    sdfg.add_array("x", (m, ), dtype)
    sdfg.add_array("y", (n, ), vtype)
    sdfg.add_array("A_device", (m, n),
                   vtype,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("x_device", (m, ),
                   dtype,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("y_device", (n, ),
                   vtype,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    a_host = pre_state.add_read("A")
    a_device = pre_state.add_write("A_device")
    pre_state.add_memlet_path(a_host,
                              a_device,
                              memlet=dace.Memlet(f"A[0:{m}, 0:{n}]"))
    x_host = pre_state.add_read("x")
    x_device = pre_state.add_write("x_device")
    pre_state.add_memlet_path(x_host, x_device, memlet=dace.Memlet(f"x[0:{m}]"))
    y_host = pre_state.add_read("y")
    y_device = pre_state.add_write("y_device")
    pre_state.add_memlet_path(y_host, y_device, memlet=dace.Memlet(f"y[0:{n}]"))

    a_device = post_state.add_read("A_device")
    a_host = post_state.add_write("A")
    post_state.add_memlet_path(a_device,
                               a_host,
                               memlet=dace.Memlet(f"A[0:{m}, 0:{n}]"))

    read_a = state.add_read("A_device")
    read_x = state.add_read("x_device")
    read_y = state.add_read("y_device")
    write_a = state.add_write("A_device")

    ger_node = blas.Ger("ger",
                        alpha=alpha,
                        veclen=veclen,
                        n_tile=tile_size_x,
                        m_tile=tile_size_y)
    ger_node.implementation = implementation

    state.add_memlet_path(read_a,
                          ger_node,
                          dst_conn="_A",
                          memlet=dace.Memlet(f"A_device[0:{m}, 0:{n}]"))
    state.add_memlet_path(read_x,
                          ger_node,
                          dst_conn="_x",
                          memlet=dace.Memlet(f"x_device[0:{m}]"))
    state.add_memlet_path(read_y,
                          ger_node,
                          dst_conn="_y",
                          memlet=dace.Memlet(f"y_device[0:{n}]"))
    state.add_memlet_path(ger_node,
                          write_a,
                          src_conn="_res",
                          memlet=dace.Memlet(f"A_device[0:{m}, 0:{n}]"))

    return sdfg


def run_test(ger, target):

    x = np.ndarray(m, dtype=np.float32)
    y = np.ndarray(n, dtype=np.float32)
    A = np.ndarray((m, n), dtype=np.float32)
    result = np.ndarray((m, n), dtype=np.float32)

    x[:] = np.random.rand(m).astype(np.float32)
    y[:] = np.random.rand(n).astype(np.float32)
    A[:] = np.random.rand(m, n).astype(np.float32)

    ger(alpha=alpha, x=x, y=y, A=A, r=result, m=m, n=n)

    ref = scipy.linalg.blas.sger(alpha=alpha, x=x, y=y, a=A)

    diff = np.linalg.norm(np.subtract(result, ref))
    if diff >= args.eps * n * m:
        raise RuntimeError(
            "Unexpected result returned from ger rank 1 operation: "
            "got:\n{}\nexpected:\n{} on {}".format(result, ref, target))
    else:
        print("Ok")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=256)
    parser.add_argument("M", type=int, nargs="?", default=512)
    parser.add_argument("n_tile", type=int, nargs="?", default=16)
    parser.add_argument("m_tile", type=int, nargs="?", default=32)
    parser.add_argument("alpha", type=np.float32, nargs="?", default=1.0)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--veclen", type=int, default=8)
    args = parser.parse_args()
    n = args.N
    m = args.M
    n_tile = args.n_tile
    m_tile = args.m_tile
    alpha = str(args.alpha)
    veclen = args.veclen

    if args.target == "pure":

        sdfg = pure_graph(dace.float32)
        run_test(sdfg.compile(), args.target)

    elif args.target == "fpga":

        sdfg = fpga_graph(dace.float32, n_tile, m_tile, veclen, alpha, "FPGA")
        x = aligned_ndarray(np.random.rand(m).astype(np.float32))
        y = aligned_ndarray(np.random.rand(n).astype(np.float32))
        A = aligned_ndarray(np.random.rand(m, n).astype(np.float32))
        A_ref = A.copy()

        sdfg(x=x, y=y, A=A, m=dace.int32(m), n=dace.int32(n))

        A_ref = scipy.linalg.blas.sger(alpha=alpha, x=x, y=y, a=A_ref)

        diff = np.linalg.norm(A_ref - A)
        if diff >= args.eps * n * m:
            raise RuntimeError(f"Validation failed: {diff}")
        else:
            print("Validation successful.")

    else:
        print("Unsupported target")
        exit(-1)
