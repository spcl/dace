# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A skewed wavefront body that stays a NestedSDFG keeps its reductions inside it.

lu's two fused ``j`` loops become one guarded body, so ``LoopToMap`` nests it and propagation
would copy the inner ``k``-map WCR onto the NestedSDFG boundary. ``WavefrontSkew`` folds each such
reduction into a transient scalar first, so no WCR leaves a NestedSDFG.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


@dace.program
def lu_small(A: dace.float64[N, N]):
    for i in range(0, N, 1):
        for j in range(0, i, 1):

            @dace.map
            def k_loop1(k: _[0:j]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> A(1, lambda x, y: x + y)[i, j]
                out = -i_in * j_in

            @dace.tasklet
            def div():
                ij_in << A[i, j]
                jj_in << A[j, j]
                out >> A[i, j]
                out = ij_in / jj_in

        for j in range(i, N, 1):

            @dace.map
            def k_loop2(k: _[0:i]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> A(1, lambda x, y: x + y)[i, j]
                out = -i_in * j_in


def lu_reference(a: np.ndarray) -> np.ndarray:
    ref = a.copy()
    for i in range(ref.shape[0]):
        for j in range(i):
            ref[i, j] -= ref[i, :j] @ ref[:j, j]
            ref[i, j] /= ref[j, j]
        for j in range(i, ref.shape[0]):
            ref[i, j] -= ref[i, :i] @ ref[:i, j]
    return ref


def illegal_wcr_sources(sdfg: dace.SDFG) -> list:
    bad = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for e in state.edges():
                if e.data.wcr is None or isinstance(e.src, (nodes.Tasklet, nodes.MapExit)):
                    continue
                if isinstance(e.src, nodes.AccessNode) and (
                    e.src.data.startswith("_wcr_priv")
                    or (e.src.desc(sd).transient and isinstance(e.src.desc(sd), dace.data.Scalar))
                ):
                    continue
                bad.append(f"{type(e.src).__name__} {e.src} -> {e.dst}: {e.data}")
    return bad


def dataflow_nodes(sdfg: dace.SDFG) -> list:
    return [n for sd in sdfg.all_sdfgs_recursive() for state in sd.states() for n in state.nodes()]


def is_skewed_into_nested_body(sdfg: dace.SDFG) -> bool:
    skewed = any(
        isinstance(r, LoopRegion) and r.loop_variable.startswith("_skew_t_") for r in sdfg.all_control_flow_regions()
    )
    return skewed and any(isinstance(n, nodes.NestedSDFG) for n in dataflow_nodes(sdfg))


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_skewed_lu_body_emits_no_nested_sdfg_wcr(target: str):
    sdfg = lu_small.to_sdfg(simplify=True)
    canonicalize(sdfg, target=target, validate=True)
    assert is_skewed_into_nested_body(sdfg)
    assert illegal_wcr_sources(sdfg) == []
    accumulators = [n.data for n in dataflow_nodes(sdfg) if isinstance(n, nodes.AccessNode)]
    assert any(name.startswith("_priv_A") for name in accumulators)


def test_skewed_lu_matches_numpy_on_cpu():
    sdfg = lu_small.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu", validate=True)
    n = 13
    rng = np.random.default_rng(0)
    a = rng.random((n, n)) + n * np.eye(n)
    ref = lu_reference(a)
    sdfg(A=a, N=n)
    np.testing.assert_allclose(a, ref, rtol=1e-12, atol=1e-12)
