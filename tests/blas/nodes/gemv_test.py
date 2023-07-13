import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory

M = dace.symbol('M')
K = dace.symbol('K')
L = dace.symbol('L')
O = dace.symbol('O')

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
        gemv_node.expand(sdfg, state, **expansion_args)

    return sdfg


def fpga_graph(dtype, transposed, expansion, veclen, alpha, beta, tile_size_x, tile_size_y):
    sdfg = pure_graph(dtype, transposed, expansion, veclen, alpha, beta, {
        "tile_size_x": tile_size_x,
        "tile_size_y": tile_size_y
    })
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory], [{}, {"storage": dace.StorageType.FPGA_Local}])
    return sdfg


def run_gemv(target: str,
             n: int,
             m: int,
             alpha: float = 1,
             transposed: bool = False,
             vectorize: int = 1,
             tile_size_x: int = 32,
             tile_size_y: int = 32):

    beta = 0  # TODO: GEMV is not currently implemented for beta != 0
    if target == "pure":
        sdfg = pure_graph(dace.float32, transposed, "pure", vectorize, alpha, beta)
    elif target == "tiles_by_column":
        if not transposed and vectorize > 1:
            raise NotImplementedError("Non-transposed vectorized tile-by-column NYI.")
        sdfg = fpga_graph(dace.float32,
                          transposed,
                          "FPGA_TilesByColumn",
                          vectorize,
                          alpha,
                          beta,
                          tile_size_x=tile_size_x,
                          tile_size_y=tile_size_y)
    elif target == "accumulate":
        sdfg = fpga_graph(dace.float32,
                          transposed,
                          "FPGA_Accumulate",
                          vectorize,
                          alpha,
                          beta,
                          tile_size_x=tile_size_x,
                          tile_size_y=tile_size_y)
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


@fpga_test()
def test_gemv_fpga_tiles_by_column():
    return run_gemv("tiles_by_column", 256, 512, transposed=True, vectorize=4)


@fpga_test()
def test_gemv_fpga_accumulate():
    return run_gemv("accumulate", 256, 512, vectorize=4)


def test_gemv_symbolic():
    sdfg = dace.SDFG("gemv")
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", [M, K], dace.float64)
    B, B_arr = sdfg.add_array("B", [L], dace.float64)
    C, C_arr = sdfg.add_array("C", [O], dace.float64)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    libnode = blas.Gemv('_Gemv_', transA=False, alpha=1.0, beta=0.0)
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_A', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_x', dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_y', wC, None, dace.Memlet.from_array(C, C_arr))

    try:
        sdfg.validate()
    except dace.sdfg.InvalidSDFGNodeError:
        pass

    sdfg.specialize({
        "M": 32,
        "L": 128,
        "O": 32,
        "K": 128,
    })

    sdfg.validate()

    sdfg.specialize({
        "M": 32,
        "L": 128,
        "O": 33,
        "K": 127,
    })

    try:
        sdfg.validate()
    except dace.sdfg.InvalidSDFGNodeError:
        pass


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

    run_gemv(args.target, args.N, args.M, args.alpha, args.transposed, args.vectorize, args.tile_size_x,
             args.tile_size_y)

    test_gemv_symbolic()
