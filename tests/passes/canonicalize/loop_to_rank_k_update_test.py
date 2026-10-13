# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToRankKUpdate`` lifts the hand-written polybench ``syrk`` / ``syr2k`` nests to ``Syrk`` / ``Syr2k``
BLAS nodes and is a strict no-op on every other shape.

The kernels under test are the REAL corpus kernels (``tests.corpus.polybench``), so the test cannot drift from the
kernels the pass is meant to recognise. The numerical oracle is the untransformed baseline SDFG at the corpus's
dtype-aware tolerance, run multithreaded.

The phased recipe runs the lift on the NORMALIZED nest (``lift_semantics``), where the corpus slice spelling and
a plain scalar-loop spelling of the same kernel are different loop nests; the matcher must see one computation.
"""

import os
from typing import Any, NamedTuple

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pytest
import sympy

import dace
from dace.libraries.blas.nodes.syr2k import Syr2k
from dace.libraries.blas.nodes.syrk import Syrk
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import loop_iteration_value as liv
from dace.transformation.passes.canonicalize.loop_to_rank_k_update import LoopToRankKUpdate
from dace.transformation.passes.canonicalize.pipeline import build_stages, canonicalize
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
    return [
        (st, n) for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes() if isinstance(n, cls)
    ]


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


@pytest.mark.parametrize(
    "name,cls,other",
    [("syrk", Syrk, Syr2k), ("syr2k", Syr2k, Syrk)],
)
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
    ((state, node),) = lifted(sdfg, Syr2k)
    assert inputs_of(state, node) == {"_a": "A", "_b": "B", "_c": "C", "_alpha": "alpha", "_beta": "beta"}


def test_syrk_nest_is_not_lifted_to_syr2k():
    """syrk's single self-paired operand reaches ``Syrk`` once; no ``Syr2k`` invents a second operand."""
    kernel = corpus_kernel("syrk")[0]
    sdfg = polybench.fresh_sdfg(kernel)
    LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syr2k) == []
    ((state, node),) = lifted(sdfg, Syrk)
    assert inputs_of(state, node) == {"_a": "A", "_c": "C", "_alpha": "alpha", "_beta": "beta"}


@dace.program
def syrk_then_syr2k(
    C: datatype[N, N], D: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1], beta: datatype[1]
):
    for i in range(N):
        C[i, : i + 1] *= beta[0]
        for k in range(M):
            C[i, : i + 1] += alpha[0] * A[i, k] * A[: i + 1, k]
    for i in range(N):
        D[i, : i + 1] *= beta[0]
        for k in range(M):
            D[i, : i + 1] += A[: i + 1, k] * alpha[0] * B[i, k] + B[: i + 1, k] * alpha[0] * A[i, k]


def test_rank_k_and_rank_2k_nests_lifted_in_one_run():
    """One run lifts a rank-k nest to ``Syrk`` on C and a rank-2k nest to ``Syr2k`` on D, matching numpy."""
    sdfg = syrk_then_syr2k.to_sdfg(simplify=True)
    assert LoopToRankKUpdate().apply_pass(sdfg, {}) == 2

    ((syrk_state, syrk_node),) = lifted(sdfg, Syrk)
    ((syr2k_state, syr2k_node),) = lifted(sdfg, Syr2k)
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
        C[i, : i + 1] *= beta[0]
        for k in range(M):
            C[i, : i + 1] += alpha[0] * A[i, k] * B[: i + 1, k]


def test_asymmetric_pairing_not_matched():
    """A one-sided two-operand product is neither a rank-k nor a rank-2k update: the SDFG stays as it was."""
    sdfg = asymmetric_nest.to_sdfg(simplify=True)
    hash_before, loops_before = sdfg.hash_sdfg(), loop_count(sdfg)
    assert not LoopToRankKUpdate().apply_pass(sdfg, {})
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []
    assert loop_count(sdfg) == loops_before and sdfg.hash_sdfg() == hash_before


def test_evaluator_refuses_a_wcr_write():
    """A WCR edge accumulates onto its destination; reading it as a plain store would make the matcher believe the
    iteration computes the increment alone."""
    sdfg = dace.SDFG("wcr_probe")
    sdfg.add_array("C", [4, 4], dace.float64)
    sdfg.add_array("A", [4, 4], dace.float64)
    outer = LoopRegion("outer", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "k < 4", "k", "k = 0", "k = k + 1")
    outer.add_node(inner, is_start_block=True)
    state = inner.add_state("body", is_start_block=True)
    tasklet = state.add_tasklet("inc", {"__a"}, {"__o"}, "__o = __a")
    state.add_edge(state.add_read("A"), None, tasklet, "__a", dace.Memlet("A[i, k]"))
    state.add_edge(
        tasklet, "__o", state.add_write("C"), None, dace.Memlet(data="C", subset="i, 0", wcr="lambda x, y: x + y")
    )
    sdfg.validate()

    with pytest.raises(liv.Refused, match="WCR"):
        liv.summarize(sdfg, outer)


@dace.program
def syrk_loops(C: datatype[N, N], A: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


@dace.program
def syr2k_loops(C: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += A[j, k] * alpha[0] * B[i, k] + B[j, k] * alpha[0] * A[i, k]


def normalized(sdfg):
    """``sdfg`` as the phased recipe hands it to ``lift_semantics``: every phase before it, no lift run."""
    stages = build_stages(target="cpu")
    first = next(i for i, (_, unit) in enumerate(stages) if isinstance(unit, LoopToRankKUpdate))
    for _, unit in stages[:first]:
        if not isinstance(unit, LoopToRankKUpdate):
            unit.apply_pass(sdfg, {})
    return sdfg


def row_value(sdfg):
    """What one iteration of the only outer loop nest stores into its row triangle, over the names ``i``, ``j``."""
    ((parent, loop),) = [(p, lp) for p, lp in liv.outer_loop_candidates(sdfg) if p is sdfg]
    summary = liv.summarize(sdfg, loop)
    ((domain,),) = [summary.domains]
    i, j = sympy.symbols("i j")
    return liv.canonical(domain.value).xreplace({summary.var: i, domain.index[1]: j})


@pytest.mark.parametrize("name,loops", [("syrk", syrk_loops), ("syr2k", syr2k_loops)])
def test_slice_and_loop_spellings_normalize_to_one_computation(name, loops):
    """The corpus slice spelling and the scalar-loop spelling are different normalized nests (vector temporaries
    vs. one statement), but one iteration of each stores the same value into the row triangle."""
    slices = normalized(polybench.fresh_sdfg(corpus_kernel(name)[0], simplify=False))
    scalar = normalized(loops.to_sdfg(simplify=False))
    i, j = sympy.symbols("i j")
    k = sympy.Dummy("k")
    C, A, B, alpha, beta = (sympy.Function(n) for n in ("C", "A", "B", "alpha", "beta"))
    pairs = A(i, k) * A(j, k) if name == "syrk" else A(i, k) * B(j, k) + B(i, k) * A(j, k)
    want = liv.canonical(beta(0) * C(i, j) + alpha(0) * sympy.Sum(pairs, (k, 0, sympy.Symbol("M") - 1)))
    assert sympy.expand(row_value(slices) - want) == 0
    assert sympy.expand(row_value(scalar) - want) == 0


@pytest.mark.parametrize("loops,cls", [(syrk_loops, Syrk), (syr2k_loops, Syr2k)])
def test_loop_spelling_lifted_on_the_normalized_nest(loops, cls):
    """On the normalized nest the scalar-loop spelling lifts to the BLAS node, which matches numpy."""
    sdfg = normalized(loops.to_sdfg(simplify=False))
    sdfg.name = f"{sdfg.name}_normalized_lift"
    assert LoopToRankKUpdate().apply_pass(sdfg, {}) == 1
    ((_, node),) = lifted(sdfg, cls)
    assert node.uplo == "L" and node.trans == "N" and node.alpha_input and node.beta_input

    rng = np.random.default_rng(11)
    n, m = 30, 20
    A, B, C = rng.random((n, m)), rng.random((n, m)), rng.random((n, n))
    alpha, beta = np.array([1.5]), np.array([1.2])
    product = A @ A.T if cls is Syrk else A @ B.T + B @ A.T
    want = np.where(np.tril(np.ones((n, n), dtype=bool)), beta[0] * C + alpha[0] * product, C)
    operands = dict(A=A) if cls is Syrk else dict(A=A, B=B)
    sdfg(C=C, alpha=alpha, beta=beta, N=n, M=m, **operands)
    np.testing.assert_allclose(C, want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("program", [full_row_nest, asymmetric_nest])
def test_non_rank_k_nests_not_lifted_by_the_phased_recipe(program):
    """The phased recipe does not lift a full-row contraction or a one-sided product either."""
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg)
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []


# Orientation x scaling variants. Every program has the signature ``(C, A, B, alpha, beta)``; ``A``/``B`` are
# ``N x M`` for ``trans='N'`` and ``M x N`` for ``'T'``; syrk programs leave ``B`` unread.
NM, MN = datatype[N, M], datatype[M, N]


@dace.program
def syrk_upper(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i, N):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i, N):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


@dace.program
def syrk_lower_trans(C: datatype[N, N], A: MN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha[0] * A[k, i] * A[k, j]


@dace.program
def syrk_upper_trans(C: datatype[N, N], A: MN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i, N):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i, N):
                C[i, j] += alpha[0] * A[k, i] * A[k, j]


@dace.program
def syrk_lower_descending(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


@dace.program
def syrk_upper_trans_descending(C: datatype[N, N], A: MN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N - 1, -1, -1):
        for j in range(i, N):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i, N):
                C[i, j] += alpha[0] * A[k, i] * A[k, j]


@dace.program
def syr2k_upper_trans(C: datatype[N, N], A: MN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i, N):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i, N):
                C[i, j] += alpha[0] * A[k, i] * B[k, j] + alpha[0] * B[k, i] * A[k, j]


@dace.program
def syr2k_lower_descending(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += A[j, k] * alpha[0] * B[i, k] + B[j, k] * alpha[0] * A[i, k]


@dace.program
def syrk_unscaled(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


@dace.program
def syrk_overwrite(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] = 0.0
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


@dace.program
def syrk_no_alpha(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += A[i, k] * A[j, k]


@dace.program
def syrk_constant_alpha_unscaled(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += 2.0 * A[i, k] * A[j, k]


@dace.program
def syr2k_overwrite_no_alpha(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] = 0.0
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += A[j, k] * B[i, k] + B[j, k] * A[i, k]


@dace.program
def syrk_subtracted(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] -= alpha[0] * A[i, k] * A[j, k]


class Variant(NamedTuple):
    """A rank-k program and what the lift must make of it."""

    program: Any
    cls: type
    uplo: str
    trans: str
    alpha: (
        str  #: ``"runtime"``/``"negated"`` (the ``alpha`` input, ``-alpha``), ``"one"`` (absent), ``"two"`` (``2.0``)
    )
    beta: str  #: ``"runtime"`` (the ``beta`` input), ``"one"`` (no scale) or ``"zero"`` (``C`` overwritten)


VARIANTS = [
    Variant(syrk_upper, Syrk, "U", "N", "runtime", "runtime"),
    Variant(syrk_lower_trans, Syrk, "L", "T", "runtime", "runtime"),
    Variant(syrk_upper_trans, Syrk, "U", "T", "runtime", "runtime"),
    Variant(syrk_lower_descending, Syrk, "L", "N", "runtime", "runtime"),
    Variant(syrk_upper_trans_descending, Syrk, "U", "T", "runtime", "runtime"),
    Variant(syr2k_upper_trans, Syr2k, "U", "T", "runtime", "runtime"),
    Variant(syr2k_lower_descending, Syr2k, "L", "N", "runtime", "runtime"),
    Variant(syrk_unscaled, Syrk, "L", "N", "runtime", "one"),
    Variant(syrk_overwrite, Syrk, "L", "N", "runtime", "zero"),
    Variant(syrk_no_alpha, Syrk, "L", "N", "one", "runtime"),
    Variant(syrk_constant_alpha_unscaled, Syrk, "L", "N", "two", "one"),
    Variant(syr2k_overwrite_no_alpha, Syr2k, "L", "N", "one", "zero"),
    Variant(syrk_subtracted, Syrk, "L", "N", "negated", "runtime"),
]

COEFFICIENT_VALUES = {"runtime": 1, "negated": -1, "one": 1, "two": 2, "zero": 0}
RUNTIME = ("runtime", "negated")


def rank_k_inputs(trans: str, n: int = 12, m: int = 7):
    rng = np.random.default_rng(5)
    shape = (n, m) if trans == "N" else (m, n)
    return rng.random(shape), rng.random(shape), rng.random((n, n)), np.array([1.5]), np.array([1.2])


def rank_k_reference(variant: Variant, A, B, C, alpha, beta):
    """``beta*C + alpha*(A A^T | A^T A | A B^T + B A^T | A^T B + B^T A)`` on the ``uplo`` triangle, ``C`` elsewhere."""
    a = A if variant.trans == "N" else A.T
    b = B if variant.trans == "N" else B.T
    product = a @ a.T if variant.cls is Syrk else a @ b.T + b @ a.T
    alpha_value = COEFFICIENT_VALUES[variant.alpha] * (alpha[0] if variant.alpha in RUNTIME else 1)
    beta_value = beta[0] if variant.beta == "runtime" else COEFFICIENT_VALUES[variant.beta]
    n = C.shape[0]
    mask = np.tril(np.ones((n, n), dtype=bool)) if variant.uplo == "L" else np.triu(np.ones((n, n), dtype=bool))
    return np.where(mask, beta_value * C + alpha_value * product, C)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.program.name)
def test_orientation_and_scaling_variants_lift(variant):
    """Each orientation (triangle, transpose, loop direction) and scaling (``beta*C``, ``C``, overwrite; runtime,
    absent or constant ``alpha``) lifts to the right node attributes and computes what numpy does."""
    sdfg = variant.program.to_sdfg(simplify=False)
    sdfg.name = f"{sdfg.name}_variant"
    canonicalize(sdfg)
    ((state, node),) = lifted(sdfg, variant.cls)
    assert lifted(sdfg, Syr2k if variant.cls is Syrk else Syrk) == []
    assert (node.uplo, node.trans) == (variant.uplo, variant.trans)
    assert node.alpha_input == (variant.alpha in RUNTIME) and node.beta_input == (variant.beta == "runtime")
    assert (
        float(node.alpha) == COEFFICIENT_VALUES[variant.alpha] and float(node.beta) == COEFFICIENT_VALUES[variant.beta]
    )
    wired = inputs_of(state, node)
    assert wired["_a"] == "A" and wired.get("_b") == (None if variant.cls is Syrk else "B")
    assert ("_c" in wired) == (variant.beta != "zero")

    A, B, C, alpha, beta = rank_k_inputs(variant.trans)
    want = rank_k_reference(variant, A, B, C, alpha, beta)
    n, m = C.shape[0], A.shape[1] if variant.trans == "N" else A.shape[0]
    sdfg(C=C, A=A, B=B, alpha=alpha, beta=beta, N=n, M=m)
    np.testing.assert_allclose(C, want, rtol=1e-9, atol=1e-11)


@dace.program
def syr2k_one_sided_alpha(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += A[j, k] * alpha[0] * B[i, k] + B[j, k] * A[i, k]


@dace.program
def syrk_scale_lower_update_upper(C: datatype[N, N], A: NM, B: NM, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta[0]
        for k in range(M):
            for j in range(i, N):
                C[i, j] += alpha[0] * A[i, k] * A[j, k]


def deviating_reference(program, A, B, C, alpha, beta):
    """The literal loop nest each deviating program runs."""
    out = C.copy()
    n, m = A.shape
    a0, b0 = alpha[0], beta[0]
    for i in range(n):
        scaled = range(i + 1)
        updated = range(i, n) if program is syrk_scale_lower_update_upper else range(i + 1)
        for j in scaled:
            out[i, j] *= b0
        for k in range(m):
            for j in updated:
                if program is syr2k_one_sided_alpha:
                    out[i, j] += A[j, k] * a0 * B[i, k] + B[j, k] * A[i, k]
                else:
                    out[i, j] += a0 * A[i, k] * A[j, k]
    return out


DEVIATING = [syr2k_one_sided_alpha, syrk_scale_lower_update_upper]


@pytest.mark.parametrize("program", DEVIATING, ids=lambda p: p.name)
def test_deviating_rank_k_nests_not_lifted(program):
    """An ``alpha`` on one of the two syr2k terms, and a scale and an update on different triangles, are not
    rank-k updates: nothing lifts and the nest keeps computing its own result."""
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f"{sdfg.name}_deviating"
    canonicalize(sdfg)
    assert lifted(sdfg, Syrk) == [] and lifted(sdfg, Syr2k) == []

    A, B, C, alpha, beta = rank_k_inputs("N")
    want = deviating_reference(program, A, B, C, alpha, beta)
    sdfg(C=C, A=A, B=B, alpha=alpha, beta=beta, N=C.shape[0], M=A.shape[1])
    np.testing.assert_allclose(C, want, rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    test_syrk_nest_lifted_to_syrk_node()
    test_syr2k_nest_lifted_to_syr2k_node()
    test_canonicalize_pipeline_lifts("syrk", Syrk, Syr2k)
    test_canonicalize_pipeline_lifts("syr2k", Syr2k, Syrk)
    test_semantic_lifting_disabled_leaves_nest("syrk")
    test_semantic_lifting_disabled_leaves_nest("syr2k")
    test_syr2k_nest_is_not_lifted_to_syrk()
    test_syrk_nest_is_not_lifted_to_syr2k()
    test_rank_k_and_rank_2k_nests_lifted_in_one_run()
    test_full_row_nest_not_matched()
    test_asymmetric_pairing_not_matched()
    test_evaluator_refuses_a_wcr_write()
    test_slice_and_loop_spellings_normalize_to_one_computation("syrk", syrk_loops)
    test_slice_and_loop_spellings_normalize_to_one_computation("syr2k", syr2k_loops)
    test_loop_spelling_lifted_on_the_normalized_nest(syrk_loops, Syrk)
    test_loop_spelling_lifted_on_the_normalized_nest(syr2k_loops, Syr2k)
    test_non_rank_k_nests_not_lifted_by_the_phased_recipe(full_row_nest)
    test_non_rank_k_nests_not_lifted_by_the_phased_recipe(asymmetric_nest)
    for variant in VARIANTS:
        test_orientation_and_scaling_variants_lift(variant)
    for program in DEVIATING:
        test_deviating_rank_k_nests_not_lifted(program)
