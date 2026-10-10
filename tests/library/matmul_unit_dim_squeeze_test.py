# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SpecializeMatMul`` and ``Gemm`` must read the same view of an operand.

The dispatcher matches on squeezed sizes, so ``np.reshape(x, (NQ, 1, NP)) @ C4`` routes to ``Gemm``
as ``(NQ, NP) @ (NP, NP)``. ``Gemm.validate`` and the GEMM codegen used to re-read the raw subset,
see rank 3, and reject the operand -- "matrix-matrix product only supported on matrices". npbench's
doitgen hits this. Collapsing the unit dim is exact (one GEMM), and keeping it on ``Gemm`` preserves
``alpha``, ``beta`` and the WCR that ``BatchedMatMul`` drops.
"""

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.matmul import MatMul
from dace.transformation.auto.auto_optimize import auto_optimize

NR, NQ, NP = (dace.symbol(s, dtype=dace.int64) for s in ("NR", "NQ", "NP"))


@dace.program
def doitgen_reshape(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
    # npbench polybench/doitgen, verbatim: the (NQ, 1, NP) reshape is the point of the benchmark.
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))


def reference(A, C4):
    nr, nq, np_ = A.shape
    return np.reshape(np.reshape(A, (nr, nq, 1, np_)) @ C4, (nr, nq, np_))


def initialize(nr, nq, np_, seed=0):
    # Random, not polybench's ((i*j+k) % NP)/NP, which is all-zero at NP == 1 (a vacuous compare).
    rng = np.random.default_rng(seed)
    return rng.random((nr, nq, np_)), rng.random((np_, np_))


# NQ == 1 collapses a second dim when squeezed, where a "whichever view looks 2D" heuristic breaks.
SIZES = [(3, 4, 5), (1, 1, 1), (8, 10, 12), (5, 1, 7)]


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("sizes", SIZES)
def test_reshape_unit_dim_matmul(optimize, sizes):
    nr, nq, np_ = sizes
    A, C4 = initialize(nr, nq, np_)
    ref = reference(A, C4)
    assert np.abs(ref).max() > 0.0  # guard against a vacuous comparison

    sdfg = doitgen_reshape.to_sdfg(simplify=False)
    sdfg.simplify()
    if optimize:
        auto_optimize(sdfg, dace.dtypes.DeviceType.CPU, symbols=dict(NR=nr, NQ=nq, NP=np_))

    sdfg(A=A, C4=C4, NR=nr, NQ=nq, NP=np_)
    assert np.allclose(A, ref)


def test_unit_batch_keeps_alpha():
    """Routing a unit-batch product away from Gemm would silently drop alpha, beta and WCR."""
    from dace import memlet as mm

    sdfg = dace.SDFG("unit_batch_alpha")
    b, m, k, n = 1, 8, 6, 5
    sdfg.add_array("A", [b, m, k], dace.float64)
    sdfg.add_array("B", [k, n], dace.float64)
    sdfg.add_array("C", [b, m, n], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    node = MatMul("mm", alpha=2.0)
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a", mm.Memlet("A[0:1, 0:8, 0:6]"))
    state.add_edge(state.add_read("B"), None, node, "_b", mm.Memlet("B[0:6, 0:5]"))
    state.add_edge(node, "_c", state.add_write("C"), None, mm.Memlet("C[0:1, 0:8, 0:5]"))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    a, bmat, c = rng.random((b, m, k)), rng.random((k, n)), np.zeros((b, m, n))
    sdfg(A=a, B=bmat, C=c)
    assert np.allclose(c[0], 2.0 * (a[0] @ bmat))


if __name__ == "__main__":
    for optimize in (False, True):
        for sizes in SIZES:
            test_reshape_unit_dim_matmul(optimize, sizes)
    test_unit_batch_keeps_alpha()
