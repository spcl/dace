import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas


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
    parser.add_argument("N", type=int, nargs="?", default=64)
    parser.add_argument("M", type=int, nargs="?", default=64)
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
        raise NotImplementedError()
    else:
        print("Unsupported target")
        exit(-1)

    ger = sdfg.compile()

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
        print("Unexpected result returned from ger rank 1 operation: "
              "got:\n{}\nexpected:\n{}".format(result, ref))
    else:
        print("Ok")
