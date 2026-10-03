# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToRankKUpdate`` lifts the hand-written polybench ``syrk`` / ``syr2k`` nests to ``Syrk`` / ``Syr2k``
BLAS nodes and is a strict no-op on every other shape.

The kernels under test are the REAL corpus kernels (``tests.corpus.polybench``), so the test cannot drift from the
kernels the pass is meant to recognise. The numerical oracle is the untransformed baseline SDFG at the corpus's
dtype-aware tolerance, run multithreaded.
"""
import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.syr2k import Syr2k
from dace.libraries.blas.nodes.syrk import Syrk
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.loop_to_rank_k_update import LoopToRankKUpdate
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.polybench import polybench

N = dace.symbol("N")
M = dace.symbol("M")
datatype = dace.float64

#: polybench ``sizes`` index 0 -- the mini dataset (M=20, N=30).
MINI = 0


def corpus_kernel(name):
    kernel = polybench.collect(name)[0]
    arrays, psize = polybench.make_inputs(kernel, size_index=MINI, cap=None)
    return kernel, arrays, psize


def lifted(sdfg, cls):
    """``(state, node)`` for every ``cls`` library node in ``sdfg``."""
    return [(st, n) for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes()
            if isinstance(n, cls)]


def loop_count(sdfg):
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def inputs_of(state, node):
    return {e.dst_conn: e.data.data for e in state.in_edges(node)}


def assert_matches_baseline(kernel, arrays, psize, sdfg):
    """The lifted SDFG reproduces the corpus baseline up to the reassociation a BLAS rank-k update performs."""
    ref = polybench.reference(kernel, arrays, psize)
    got = polybench.run(sdfg, arrays, psize)
    assert polybench.outputs_match(ref, got), "lifted SDFG diverged from the polybench baseline"


def test_syrk_nest_lifted_to_syrk_node():
    """The polybench syrk nest becomes exactly one ``Syrk`` node (and no ``Syr2k``) that matches the baseline."""
    kernel, arrays, psize = corpus_kernel("syrk")
    sdfg = polybench.fresh_sdfg(kernel)
    loops_before = loop_count(sdfg)
    assert LoopToRankKUpdate().apply_pass(sdfg, {}) == 1

    found = lifted(sdfg, Syrk)
    assert len(found) == 1 and lifted(sdfg, Syr2k) == []
    assert loop_count(sdfg) == loops_before - 2, "the outer row loop and the inner k loop are both spliced out"
    node = found[0][1]
    assert node.uplo == "L" and node.trans == "N"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_c", "_alpha", "_beta"}

    sdfg.expand_library_nodes()
    sdfg.validate()
    assert_matches_baseline(kernel, arrays, psize, sdfg)


def test_syr2k_nest_lifted_to_syr2k_node():
    """The polybench syr2k nest becomes exactly one ``Syr2k`` node (and no ``Syrk``) that matches the baseline."""
    kernel, arrays, psize = corpus_kernel("syr2k")
    sdfg = polybench.fresh_sdfg(kernel)
    loops_before = loop_count(sdfg)
    assert LoopToRankKUpdate().apply_pass(sdfg, {}) == 1

    found = lifted(sdfg, Syr2k)
    assert len(found) == 1 and lifted(sdfg, Syrk) == []
    assert loop_count(sdfg) == loops_before - 2, "the outer row loop and the inner k loop are both spliced out"
    node = found[0][1]
    assert node.uplo == "L" and node.trans == "N"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_b", "_c", "_alpha", "_beta"}

    sdfg.expand_library_nodes()
    sdfg.validate()
    assert_matches_baseline(kernel, arrays, psize, sdfg)


@pytest.mark.parametrize("name,cls,other", [("syrk", Syrk, Syr2k), ("syr2k", Syr2k, Syrk)])
def test_canonicalize_pipeline_lifts(name, cls, other):
    """The lift fires through the full ``canonicalize()`` recipe, starting from the raw frontend shape."""
    kernel, arrays, psize = corpus_kernel(name)
    sdfg = polybench.fresh_sdfg(kernel, simplify=False)
    canonicalize(sdfg)
    assert len(lifted(sdfg, cls)) == 1 and lifted(sdfg, other) == []
    assert_matches_baseline(kernel, arrays, psize, sdfg)


@pytest.mark.parametrize("name", ["syrk", "syr2k"])
def test_semantic_lifting_disabled_leaves_nest(name):
    """``semantic_lifting=False`` (the vectorizer path) leaves the nest alone."""
    kernel = corpus_kernel(name)[0]
    sdfg = polybench.fresh_sdfg(kernel, simplify=False)
    canonicalize(sdfg, semantic_lifting=False)
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []


def test_syr2k_nest_is_not_lifted_to_syrk():
    """syr2k's two cross-paired operands both reach the ``Syr2k`` node; no ``Syrk`` reads just one of them."""
    kernel = corpus_kernel("syr2k")[0]
    sdfg = polybench.fresh_sdfg(kernel)
    LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syrk) == []
    (state, node), = lifted(sdfg, Syr2k)
    assert inputs_of(state, node) == {"_a": "A", "_b": "B", "_c": "C", "_alpha": "alpha", "_beta": "beta"}


def test_syrk_nest_is_not_lifted_to_syr2k():
    """syrk's single self-paired operand reaches ``Syrk`` once; no ``Syr2k`` invents a second operand."""
    kernel = corpus_kernel("syrk")[0]
    sdfg = polybench.fresh_sdfg(kernel)
    LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syr2k) == []
    (state, node), = lifted(sdfg, Syrk)
    assert inputs_of(state, node) == {"_a": "A", "_c": "C", "_alpha": "alpha", "_beta": "beta"}


@dace.program
def syrk_then_syr2k(C: datatype[N, N], D: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1],
                    beta: datatype[1]):
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += alpha[0] * A[i, k] * A[:i + 1, k]
    for i in range(N):
        D[i, :i + 1] *= beta[0]
        for k in range(M):
            D[i, :i + 1] += A[:i + 1, k] * alpha[0] * B[i, k] + B[:i + 1, k] * alpha[0] * A[i, k]


def test_rank_k_and_rank_2k_nests_lifted_in_one_run():
    """One run lifts a rank-k nest to ``Syrk`` on C and a rank-2k nest to ``Syr2k`` on D, matching numpy."""
    sdfg = syrk_then_syr2k.to_sdfg(simplify=True)
    assert LoopToRankKUpdate().apply_pass(sdfg, {}) == 2

    (syrk_state, syrk_node), = lifted(sdfg, Syrk)
    (syr2k_state, syr2k_node), = lifted(sdfg, Syr2k)
    assert inputs_of(syrk_state, syrk_node)["_c"] == "C" and "_b" not in inputs_of(syrk_state, syrk_node)
    assert inputs_of(syr2k_state, syr2k_node)["_c"] == "D"
    assert loop_count(sdfg) == 0
    sdfg.validate()

    rng = np.random.default_rng(7)
    n, m = 30, 20
    A, B = rng.random((n, m)), rng.random((n, m))
    C, D = rng.random((n, n)), rng.random((n, n))
    alpha, beta = np.array([1.5]), np.array([1.2])
    lower = np.tril(np.ones((n, n), dtype=bool))
    want_c = np.where(lower, beta[0] * C + alpha[0] * (A @ A.T), C)
    want_d = np.where(lower, beta[0] * D + alpha[0] * (A @ B.T + B @ A.T), D)

    sdfg(C=C, D=D, A=A, B=B, alpha=alpha, beta=beta, N=n, M=m)
    np.testing.assert_allclose(C, want_c, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(D, want_d, rtol=1e-9, atol=1e-11)


@dace.program
def full_row_nest(C: datatype[N, N], A: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    # A full-row (non-triangular) update: a plain contraction, not a rank-k update.
    for i in range(N):
        C[i, :] *= beta[0]
        for k in range(M):
            C[i, :] += alpha[0] * A[i, k] * A[:, k]


def test_full_row_nest_not_matched():
    """A full-row slice is a gemm-shaped contraction without the triangular fingerprint: the SDFG stays as it was."""
    sdfg = full_row_nest.to_sdfg(simplify=True)
    hash_before, loops_before = sdfg.hash_sdfg(), loop_count(sdfg)
    assert not LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []
    assert loop_count(sdfg) == loops_before and sdfg.hash_sdfg() == hash_before


@dace.program
def asymmetric_nest(C: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    # Triangular and rank-k shaped, but ``A[i,k]*B[j,k]`` has no symmetric ``B[i,k]*A[j,k]`` partner.
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += alpha[0] * A[i, k] * B[:i + 1, k]


def test_asymmetric_pairing_not_matched():
    """A one-sided two-operand product is neither a rank-k nor a rank-2k update: the SDFG stays as it was."""
    sdfg = asymmetric_nest.to_sdfg(simplify=True)
    hash_before, loops_before = sdfg.hash_sdfg(), loop_count(sdfg)
    assert not LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []
    assert loop_count(sdfg) == loops_before and sdfg.hash_sdfg() == hash_before


def test_rank_k_resolver_refuses_a_wcr_write():
    """A WCR in-edge accumulates onto the destination; reading it as a plain def would make the
    matcher believe the state computes the increment alone."""
    from dace.transformation.passes.canonicalize.rank_k_match import StateValueResolver

    sdfg = dace.SDFG("wcr_probe")
    sdfg.add_array("C", [4, 4], dace.float64)
    sdfg.add_array("A", [4, 4], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("inc", {"__a"}, {"__o"}, "__o = __a")
    state.add_edge(state.add_read("A"), None, tasklet, "__a", dace.Memlet("A[0, 0]"))
    state.add_edge(tasklet, "__o", state.add_write("C"), None,
                   dace.Memlet(data="C", subset="0, 0", wcr="lambda x, y: x + y"))
    sdfg.validate()

    sink = next(n for n in state.data_nodes() if n.data == "C")
    zero = dace.symbolic.pystr_to_symbolic("0")
    with pytest.raises(ValueError, match="WCR"):
        StateValueResolver(state).value_at(sink, [zero, zero])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
