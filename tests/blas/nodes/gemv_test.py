import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas


# ---------- ----------
# Pure graph program (CPU)
# ---------- ----------
def pure_graph(dtype, transposed):
    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("gemv")

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

    gemv_node = blas.Gemv("gemv",
                          dtype=dace.float32,
                          transA=transposed)

    state.add_memlet_path(A,
                          gemv_node,
                          dst_conn="_A",
                          memlet=Memlet.simple(
                              A, "0:{}, 0:{}".format(A_rows, A_cols)))
    state.add_memlet_path(x,
                          gemv_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x,
                                               "0:{}".format(x_size)))
    y = state.add_read("y")
    state.add_memlet_path(y,
                          gemv_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y,
                                               "0:{}".format(y_size)))
    state.add_memlet_path(gemv_node,
                          result,
                          src_conn="_y",
                          memlet=Memlet.simple(result, "0:{}".format(y_size)))
    return sdfg


# ---------- ----------
# Intel FPGA graph
# ---------- ----------
def intel_fpga_graph(dtype, vec_width=4):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=16)
    parser.add_argument("M", type=int, nargs="?", default=16)
    parser.add_argument("alpha", type=int, nargs="?", default=1)
    parser.add_argument("beta", type=int, nargs="?", default=0)
    parser.add_argument("--transposed",
                        action="store_true",
                        default=False,
                        help="Compute GEMV with transposed matrix")
    parser.add_argument("--target", dest="target", default="pure")

    args = parser.parse_args()
    n = args.N
    m = args.M
    alpha = args.alpha
    beta = args.beta
    transposed = args.transposed
    if args.target == "pure":
        sdfg = pure_graph(dace.float32, transposed)
    elif args.target == "intel_fpga":
        sdfg = intel_fpga_graph(dace.float32)
    else:
        print("Unsupported target")
        exit(-1)

    A = np.random.rand(n, m).astype(np.float32)
    x = np.random.rand(n if transposed else m).astype(np.float32)
    y = np.random.rand(m if transposed else n).astype(np.float32)

    y_copy = np.copy(y)

    sdfg(A=A, x=x, y=y, n=n, m=m, alpha=alpha, beta=beta)

    ref = scipy.linalg.blas.sgemv(alpha, A, x, beta, y_copy, trans=transposed)

    diff = np.linalg.norm(y - ref) / (m if transposed else n)
    if diff >= 1e-5:
        print("Error")
    else:
        print("Ok")
