import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory


def pure_graph(dtype,
               transposed,
               expansion,
               veclen,
               alpha,
               beta,
               expansion_args=None):

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

    state.add_memlet_path(A,
                          gemv_node,
                          dst_conn="_A",
                          memlet=Memlet(f"A[0:{A_rows}, 0:{A_cols}]"))
    state.add_memlet_path(x,
                          gemv_node,
                          dst_conn="_x",
                          memlet=Memlet(f"x[0:{x_size}]"))
    state.add_memlet_path(gemv_node,
                          result,
                          src_conn="_y",
                          memlet=Memlet(f"y[0:{y_size}]"))

    if expansion_args is not None:
        gemv_node.expand(sdfg, state, **expansion_args)

    return sdfg


def fpga_graph(dtype, transposed, expansion, veclen, alpha, beta, tile_size_x,
               tile_size_y):
    sdfg = pure_graph(dtype, transposed, expansion, veclen, alpha, beta, {
        "tile_size_x": tile_size_x,
        "tile_size_y": tile_size_y
    })
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated(
        [InlineSDFG, StreamingMemory], [{}, {
            "storage": dace.StorageType.FPGA_Local
        }])
    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=256)
    parser.add_argument("N", type=int, nargs="?", default=512)
    parser.add_argument("alpha", type=int, nargs="?", default=1)
    # parser.add_argument("beta", type=int, nargs="?", default=0)
    parser.add_argument("--transposed",
                        action="store_true",
                        default=False,
                        help="Compute GEMV with transposed matrix")
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vectorize", dest="vectorize", default=1, type=int)
    parser.add_argument("--tile-size-x", type=int, default=32)
    parser.add_argument("--tile-size-y", type=int, default=32)

    args = parser.parse_args()
    m = args.M
    n = args.N
    alpha = args.alpha
    # beta = args.beta
    beta = 0  # TODO: GEMV is not currently implemented for beta != 0
    transposed = args.transposed
    if args.target == "pure":
        sdfg = pure_graph(dace.float32, transposed, "pure", args.vectorize,
                          alpha, beta)
    elif args.target == "tiles_by_column":
        if not transposed and args.vectorize > 1:
            raise NotImplementedError(
                "Non-transposed vectorized tile-by-column NYI.")
        sdfg = fpga_graph(dace.float32,
                          transposed,
                          "FPGA_TilesByColumn",
                          args.vectorize,
                          alpha,
                          beta,
                          tile_size_x=args.tile_size_x,
                          tile_size_y=args.tile_size_y)
    elif args.target == "accumulate":
        sdfg = fpga_graph(dace.float32,
                          transposed,
                          "FPGA_Accumulate",
                          args.vectorize,
                          alpha,
                          beta,
                          tile_size_x=args.tile_size_x,
                          tile_size_y=args.tile_size_y)
    else:
        print("Unsupported target")
        exit(-1)

    A = np.random.rand(m, n).astype(np.float32)
    x = np.random.rand(n if not transposed else m).astype(np.float32)
    y = np.random.rand(m if not transposed else n).astype(np.float32)

    y_copy = np.copy(y)

    sdfg(A=A, x=x, y=y, n=n, m=m)

    ref = scipy.linalg.blas.sgemv(alpha, A, x, beta, y_copy, trans=transposed)

    diff = np.linalg.norm(y - ref) / (m if transposed else n)
    if diff >= 1e-5:
        raise ValueError("Validation failed.")
