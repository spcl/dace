import argparse

import numpy as np
import scipy

import dace
from dace.libraries import blas
from dace.memlet import Memlet


def pure_graph(dtype, transposed, expansion, veclen, alpha, beta, expansion_args=None):

    sdfg = dace.SDFG(f"gemv_{expansion}_{dtype}_{transposed}_w{veclen}")

    m = dace.symbol("m")
    n = dace.symbol("n")
    n /= veclen
    vtype = dace.vector(dtype, veclen)

    state = sdfg.add_state("gemv_compute")

    A_rows = m
    A_cols = n
    x_size = n if not transposed else m
    y_size = m if not transposed else n

    sdfg.add_array("A", shape=[A_rows, A_cols], dtype=vtype)
    sdfg.add_array("x", shape=[x_size], dtype=dtype if transposed else vtype)
    sdfg.add_array("y", shape=[y_size], dtype=vtype if transposed else dtype)

    A = state.add_read("A")
    x = state.add_read("x")
    result = state.add_write("y")

    gemv_node = blas.Gemv("gemv", transA=transposed, alpha=alpha, beta=beta)
    gemv_node.implementation = expansion

    state.add_memlet_path(A, gemv_node, dst_conn="_A", memlet=Memlet(f"A[0:{A_rows}, 0:{A_cols}]"))
    state.add_memlet_path(x, gemv_node, dst_conn="_x", memlet=Memlet(f"x[0:{x_size}]"))
    state.add_memlet_path(gemv_node, result, src_conn="_y", memlet=Memlet(f"y[0:{y_size}]"))

    if expansion_args is not None:
        gemv_node.expand(state, **expansion_args)

    return sdfg


def run_gemv(
    target: str,
    n: int,
    m: int,
    alpha: float = 1,
    transposed: bool = False,
    vectorize: int = 1,
    tile_size_x: int = 32,
    tile_size_y: int = 32,
):

    beta = 0  # TODO: GEMV is not currently implemented for beta != 0
    if target == "pure":
        sdfg = pure_graph(dace.float32, transposed, "pure", vectorize, alpha, beta)
    else:
        raise ValueError("Unsupported target")

    A = np.random.rand(m, n).astype(np.float32)
    x = np.random.rand(n if not transposed else m).astype(np.float32)
    y = np.random.rand(m if not transposed else n).astype(np.float32)

    y_copy = np.copy(y)

    sdfg(A=A, x=x, y=y, n=n, m=m)

    ref = scipy.linalg.blas.sgemv(alpha, A, x, beta, y_copy, trans=transposed)

    diff = np.linalg.norm(y - ref) / (m if transposed else n)
    if diff >= 1e-5:
        raise RuntimeError("Validation failed.")

    return sdfg


def test_pure():
    run_gemv("pure", 256, 512, transposed=True)


def test_pure_with_a_register_vector():
    """A heap matrix and a register vector both live on the host, so the pure expansion multiplies them."""
    sdfg = dace.SDFG("gemv_register_operand")
    sdfg.add_array("A", [3, 3], dace.float64)
    sdfg.add_array("x", [3], dace.float64)
    sdfg.add_array("y", [3], dace.float64)
    sdfg.add_array("xr", [3], dace.float64, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    xr = state.add_access("xr")
    state.add_nedge(state.add_read("x"), xr, Memlet("x[0:3]"))
    node = blas.Gemv("gemv")
    node.implementation = "pure"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_A", Memlet("A[0:3, 0:3]"))
    state.add_edge(xr, None, node, "_x", Memlet("xr[0:3]"))
    state.add_edge(node, "_y", state.add_write("y"), None, Memlet("y[0:3]"))
    A, x, y = np.random.rand(3, 3), np.random.rand(3), np.zeros(3)
    sdfg(A=A, x=x, y=y)
    assert np.allclose(y, A @ x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=256)
    parser.add_argument("N", type=int, nargs="?", default=512)
    parser.add_argument("alpha", type=int, nargs="?", default=1)
    # parser.add_argument("beta", type=int, nargs="?", default=0)
    parser.add_argument("--transposed", action="store_true", default=False, help="Compute GEMV with transposed matrix")
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vectorize", dest="vectorize", default=1, type=int)
    parser.add_argument("--tile-size-x", type=int, default=32)
    parser.add_argument("--tile-size-y", type=int, default=32)

    args = parser.parse_args()

    run_gemv(
        args.target, args.N, args.M, args.alpha, args.transposed, args.vectorize, args.tile_size_x, args.tile_size_y
    )
