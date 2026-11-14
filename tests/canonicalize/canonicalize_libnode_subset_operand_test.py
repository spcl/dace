# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonical form of a product of slices has no View: the GEMM reads and writes the slices' subsets."""

import numpy as np

import dace
from dace import data as dt
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

R, C, K, M, N, I, J = (dace.symbol(s) for s in "RCKMNIJ")


@dace.program
def product_of_slices(A: dace.float64[R, K], B: dace.float64[K, N], Out: dace.float64[M, C]):
    Out[:, J : J + N] = A[I : I + M, :] @ B


def test_a_product_of_slices_canonicalizes_without_a_view():
    sdfg = product_of_slices.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    assert not [
        node.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.AccessNode) and isinstance(node.desc(state.sdfg), dt.View)
    ]
    ((gemm, state),) = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]
    assert {e.data.data for e in state.all_edges(gemm)} == {"A", "B", "Out"}
    finalize_for_target(sdfg, "cpu", validate=True)

    rng = np.random.default_rng(3)
    a, b, out = rng.random((9, 4)), rng.random((4, 5)), rng.random((3, 8))
    expected = out.copy()
    expected[:, 2:7] = a[4:7, :] @ b
    sdfg(A=a, B=b, Out=out, R=9, K=4, N=5, M=3, C=8, I=4, J=2)
    assert np.allclose(out, expected)
