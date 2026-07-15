# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToSyrk`` / ``LoopToSyr2k`` lift the hand-written polybench ``syrk`` / ``syr2k``
nests to ``Syrk`` / ``Syr2k`` BLAS nodes, and are a strict no-op on every other shape.

The kernels under test are the REAL corpus kernels, reached through the polybench
registry (``tests.corpus.polybench``) rather than restated here, so the test cannot
drift from the kernels the passes are meant to recognise. The numerical oracle is the
corpus's own: the untransformed baseline SDFG, compared at the corpus's dtype-aware
tolerance.

Each pass is verified four ways:

* it fires exactly once on its own kernel, with the right ``uplo`` / ``trans`` and
  runtime ``alpha`` / ``beta`` connectors;
* the lifted SDFG still matches the corpus baseline numerically;
* it fires through the full ``canonicalize()`` pipeline, not just standalone (and is
  correctly suppressed by ``semantic_lifting=False``);
* it does NOT fire on the sibling kernel, on a full-row (gemm-shaped) nest, or on a
  triangular nest whose operand pairing is not symmetric.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

import dace
from dace.libraries.blas.nodes.syr2k import Syr2k
from dace.libraries.blas.nodes.syrk import Syrk
from dace.transformation.passes.canonicalize.loop_to_syr2k import LoopToSyr2k
from dace.transformation.passes.canonicalize.loop_to_syrk import LoopToSyrk
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.polybench import polybench

N = dace.symbol("N")
M = dace.symbol("M")
datatype = dace.float64

#: polybench ``sizes`` index 0 -- the mini dataset (M=20, N=30), enough to exercise the
#: triangle and the k reduction while staying fast.
MINI = 0


def corpus_kernel(name):
    """``(kernel, arrays, psize)`` for the polybench kernel ``name`` at the mini size."""
    kernel = polybench.collect(name)[0]
    arrays, psize = polybench.make_inputs(kernel, size_index=MINI, cap=None)
    return kernel, arrays, psize


def nodes_of(sdfg, cls):
    return [n for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes() if isinstance(n, cls)]


def assert_matches_baseline(kernel, arrays, psize, sdfg):
    """The lifted SDFG must reproduce the corpus baseline's outputs.

    NOT bit-exact, by construction: the point of the lift is to replace the kernel's
    strictly-ordered per-``k`` accumulation with a BLAS rank-k update, which reassociates
    (and threads) that sum. The residual is floating-point reassociation only -- a few
    ULP of the result magnitude -- and is checked at the corpus's own fp64 tolerance
    (rtol 1e-9 / atol 1e-11), the same bar every other polybench lift is held to.
    """
    ref = polybench.reference(kernel, arrays, psize)
    got = polybench.run(sdfg, arrays, psize)
    assert polybench.outputs_match(ref, got), "lifted SDFG diverged from the polybench baseline"


def test_syrk_nest_lifted_to_syrk_node():
    """The polybench syrk nest becomes exactly one ``Syrk`` node (uplo L, trans N,
    runtime alpha/beta connectors) and stays faithful to the corpus baseline."""
    kernel, arrays, psize = corpus_kernel("syrk")
    sdfg = polybench.fresh_sdfg(kernel)
    assert LoopToSyrk().apply_pass(sdfg, {}) == 1

    found = nodes_of(sdfg, Syrk)
    assert len(found) == 1
    node = found[0]
    assert node.uplo == "L" and node.trans == "N"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_c", "_alpha", "_beta"}

    sdfg.expand_library_nodes()
    sdfg.validate()
    assert_matches_baseline(kernel, arrays, psize, sdfg)


def test_syr2k_nest_lifted_to_syr2k_node():
    """The polybench syr2k nest becomes exactly one ``Syr2k`` node (uplo L, trans N,
    runtime alpha/beta connectors) and stays faithful to the corpus baseline."""
    kernel, arrays, psize = corpus_kernel("syr2k")
    sdfg = polybench.fresh_sdfg(kernel)
    assert LoopToSyr2k().apply_pass(sdfg, {}) == 1

    found = nodes_of(sdfg, Syr2k)
    assert len(found) == 1
    node = found[0]
    assert node.uplo == "L" and node.trans == "N"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_b", "_c", "_alpha", "_beta"}

    sdfg.expand_library_nodes()
    sdfg.validate()
    assert_matches_baseline(kernel, arrays, psize, sdfg)


@pytest.mark.parametrize("name,cls", [("syrk", Syrk), ("syr2k", Syr2k)])
def test_canonicalize_pipeline_lifts(name, cls):
    """The lift also fires through the full ``canonicalize()`` recipe -- which is where
    it has to fire to be worth anything -- starting from the raw frontend shape."""
    kernel, arrays, psize = corpus_kernel(name)
    sdfg = polybench.fresh_sdfg(kernel, simplify=False)
    canonicalize(sdfg)
    assert len(nodes_of(sdfg, cls)) == 1
    assert_matches_baseline(kernel, arrays, psize, sdfg)


@pytest.mark.parametrize("name", ["syrk", "syr2k"])
def test_semantic_lifting_disabled_leaves_nest(name):
    """``semantic_lifting=False`` (the vectorizer path) must leave the nest alone."""
    kernel, _, _ = corpus_kernel(name)
    sdfg = polybench.fresh_sdfg(kernel, simplify=False)
    canonicalize(sdfg, semantic_lifting=False)
    assert nodes_of(sdfg, Syrk) == [] and nodes_of(sdfg, Syr2k) == []


def test_syrk_pass_does_not_match_syr2k():
    """syr2k's two-operand cross-pairing is not a rank-k update."""
    kernel, _, _ = corpus_kernel("syr2k")
    sdfg = polybench.fresh_sdfg(kernel)
    assert not LoopToSyrk().apply_pass(sdfg, {})
    assert nodes_of(sdfg, Syrk) == []


def test_syr2k_pass_does_not_match_syrk():
    """syrk's single self-paired operand is not a rank-2k update."""
    kernel, _, _ = corpus_kernel("syrk")
    sdfg = polybench.fresh_sdfg(kernel)
    assert not LoopToSyr2k().apply_pass(sdfg, {})
    assert nodes_of(sdfg, Syr2k) == []


@dace.program
def full_row_nest(C: datatype[N, N], A: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    # A full-row (non-triangular) update: a plain contraction, not a rank-k update.
    for i in range(N):
        C[i, :] *= beta[0]
        for k in range(M):
            C[i, :] += alpha[0] * A[i, k] * A[:, k]


def test_full_row_nest_not_matched():
    """A full-row slice is a gemm-shaped contraction: the triangular fingerprint that
    makes the update a rank-k one is absent, so neither pass may fire."""
    sdfg = full_row_nest.to_sdfg(simplify=True)
    assert not LoopToSyrk().apply_pass(sdfg, {})
    assert not LoopToSyr2k().apply_pass(sdfg, {})
    assert nodes_of(sdfg, Syrk) == [] and nodes_of(sdfg, Syr2k) == []


@dace.program
def asymmetric_nest(C: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    # Triangular and rank-k shaped, but the product pairs two DIFFERENT operands
    # one-sidedly (``A[i,k]*B[j,k]`` with no symmetric ``B[i,k]*A[j,k]`` partner), so the
    # result is not symmetric and neither xSYRK nor xSYR2K computes it.
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += alpha[0] * A[i, k] * B[:i + 1, k]


def test_asymmetric_pairing_not_matched():
    """A one-sided two-operand product is not a symmetric rank-k / rank-2k update."""
    sdfg = asymmetric_nest.to_sdfg(simplify=True)
    assert not LoopToSyrk().apply_pass(sdfg, {})
    assert not LoopToSyr2k().apply_pass(sdfg, {})
    assert nodes_of(sdfg, Syrk) == [] and nodes_of(sdfg, Syr2k) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
