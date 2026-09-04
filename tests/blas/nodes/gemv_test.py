import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas


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


def blas_gemv_graph(rows, cols, beta=0):
    """A single transposed OpenBLAS gemv whose contraction extent is ``rows``."""
    sdfg = dace.SDFG(f"gemv_blas_{rows}_{cols}")
    state = sdfg.add_state("s", is_start_block=True)
    sdfg.add_array("A", shape=[rows, cols], dtype=dace.float64)
    sdfg.add_array("x", shape=[rows], dtype=dace.float64)
    sdfg.add_array("y", shape=[cols], dtype=dace.float64)
    node = blas.Gemv("gemv", transA=True, alpha=1, beta=beta)
    node.implementation = "OpenBLAS"
    state.add_memlet_path(state.add_read("A"), node, dst_conn="_A", memlet=Memlet(f"A[0:{rows}, 0:{cols}]"))
    state.add_memlet_path(state.add_read("x"), node, dst_conn="_x", memlet=Memlet(f"x[0:{rows}]"))
    if beta != 0:
        state.add_memlet_path(state.add_read("y"), node, dst_conn="_y", memlet=Memlet(f"y[0:{cols}]"))
    state.add_memlet_path(node, state.add_write("y"), src_conn="_y", memlet=Memlet(f"y[0:{cols}]"))
    return sdfg


def gemv_tasklet_code(sdfg):
    sdfg.expand_library_nodes()
    codes = [n.code.as_string for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
    assert len(codes) == 1, f"expected a single gemv tasklet, got {len(codes)}"
    return codes[0]


def test_blas_gemv_guards_an_extent_that_can_be_empty():
    """A contraction that may be 0 must not leave the output unwritten.

    BLAS returns immediately when either dimension is 0 and skips the ``beta`` scaling with it, so
    with ``beta == 0`` the caller is handed uninitialized storage. It is invisible at small sizes
    only because fresh pages read back as zero.
    """
    code = gemv_tasklet_code(blas_gemv_graph(dace.symbol("m"), dace.symbol("n")))
    assert "cblas_dgemv" in code
    assert "<= 0" in code, f"symbolic contraction must be guarded, got:\n{code}"


def test_blas_gemv_leaves_a_provably_nonempty_extent_a_bare_call():
    """A literal positive extent cannot hit the early return, so it keeps the bare library call."""
    code = gemv_tasklet_code(blas_gemv_graph(8, 12))
    assert "cblas_dgemv" in code
    assert "<= 0" not in code, f"literal extents need no guard, got:\n{code}"


def test_blas_gemv_empty_contraction_writes_the_output():
    """The numbers, not just the shape: an empty contraction with ``beta == 0`` zeroes ``y``."""
    csdfg = blas_gemv_graph(dace.symbol("m"), dace.symbol("n")).compile()
    y = np.full(6, 1e250)
    csdfg(A=np.zeros((0, 6)), x=np.zeros(0), y=y, m=0, n=6)
    assert np.all(y == 0.0), f"empty contraction left y unwritten: {y}"


def test_blas_gemv_nonempty_contraction_still_computes_the_product():
    """The guard must not shadow the real call."""
    rng = np.random.default_rng(0)
    A, x = rng.random((5, 6)), rng.random(5)
    y = np.zeros(6)
    blas_gemv_graph(dace.symbol("m"), dace.symbol("n")).compile()(A=A.copy(), x=x.copy(), y=y, m=5, n=6)
    assert np.allclose(y, A.T @ x), f"{y} != {A.T @ x}"


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
