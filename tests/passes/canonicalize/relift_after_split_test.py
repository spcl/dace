# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A recurrence that only a split exposes is still lifted to a ``Scan``.

The hybrid sparse kernel's body is ``if i < K: y[i] = s else: y[i] = s + y[i - 1]``: the lifting segment sees a
branch, not a scan. Parallelism derivation splits the loop at ``K``, and the ``[K:N)`` half is then exactly
``y[i] = s[i] + y[i - 1]``, the shape ``LoopToScan`` lifts. Before the re-lift it stayed a sequential loop of
``N - K`` dependent adds.
"""

import copy

import numpy as np

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.measure_parallelization import cpu_params

N, K, NNZ = (dace.symbol(name) for name in ("N", "K", "NNZ"))


@dace.program
def hybrid_sparse(
    row_ptr: dace.int32[N + 1], col: dace.int32[NNZ], val: dace.float64[NNZ], x: dace.float64[N], y: dace.float64[N]
):
    for i in range(N):
        s = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            s += val[k] * x[col[k]]
        if i < K:
            y[i] = s
        else:
            y[i] = s + y[i - 1]


@dace.program
def split_reduction(x: dace.float64[N], y: dace.float64[N], out: dace.float64[1]):
    for i in range(N):
        if i < K:
            y[i] = 2.0 * x[i]
        else:
            out[0] += x[i]


def reference(row_ptr, col, val, x, y, K):
    for i in range(y.shape[0]):
        s = val[row_ptr[i] : row_ptr[i + 1]] @ x[col[row_ptr[i] : row_ptr[i + 1]]]
        y[i] = s if i < K else s + y[i - 1]


def canonicalized(name: str) -> dace.SDFG:
    sdfg = hybrid_sparse.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params())
    return sdfg


def test_recurrence_exposed_by_split_is_a_scan():
    sdfg = canonicalized("relift_hybrid_structure")
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    scans = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, Scan)]
    assert not loops, [loop.label for loop in loops]
    assert len(scans) == 1, scans


def test_relifted_scan_matches_reference():
    fin = finalize_for_target(copy.deepcopy(canonicalized("relift_hybrid_numerics")), "cpu")
    compiled = fin.compile()
    rng = np.random.default_rng(1234)
    n = 97
    row_ptr = np.concatenate([[0], np.cumsum(rng.integers(0, 6, n))]).astype(np.int32)
    nnz = int(row_ptr[-1])
    col = rng.integers(0, n, nnz).astype(np.int32)
    val = rng.random(nnz)
    x = rng.random(n)
    # K = 1: the scan covers all but one row; K = 40: both halves; K > N: no scan iteration at all.
    for k in (1, 40, n + 5):
        expected = np.zeros(n)
        reference(row_ptr, col, val, x, expected, k)
        got = np.zeros(n)
        compiled(row_ptr=row_ptr, col=col, val=val, x=x, y=got, N=n, K=k, NNZ=nnz)
        assert np.allclose(got, expected), k


def test_reduction_exposed_by_split_is_lifted_like_an_unsplit_one():
    """The ``[K:N)`` half is ``out[0] += x[i]``: lifted to a ``Reduce`` as the lifting segment lifts any reduction."""
    stats = {}
    sdfg = split_reduction.to_sdfg(simplify=True)
    sdfg.name = "relift_split_reduction"
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(), stats=stats)
    assert stats["outcomes"].get("lift:LoopToReduce") == 1, stats
    compiled = finalize_for_target(copy.deepcopy(sdfg), "cpu").compile()
    rng = np.random.default_rng(1234)
    n = 97
    for k in (1, 40, n + 5):
        x, y, out = rng.random(n), np.zeros(n), np.zeros(1)
        compiled(x=x, y=y, out=out, N=n, K=k)
        assert np.allclose(y[: min(k, n)], 2.0 * x[: min(k, n)]) and not y[min(k, n) :].any(), k
        assert np.isclose(out[0], x[min(k, n) :].sum()), k


if __name__ == "__main__":
    test_recurrence_exposed_by_split_is_a_scan()
    test_relifted_scan_matches_reference()
    test_reduction_exposed_by_split_is_lifted_like_an_unsplit_one()
