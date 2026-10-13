# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToSymm`` lifts the hand-written polybench ``symm`` nest to a ``Symm`` BLAS node and is a strict no-op on
every other kernel.

The matcher reads the NORMALIZED nest the phased recipe hands ``lift_semantics``: it evaluates what one outer
iteration stores into ``C`` (the rows above ``i`` and row ``i``) and compares that against ``symm``. The npbench
slice spelling -- column slices, a ``B[:i, j] @ A[i, :i]`` inner product, a separate ``C *= beta`` prescale -- and
the scalar-loop spelling with a ``temp2`` accumulator are different nests there and must both lift; nests with the
same access shapes but other arithmetic must not.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import copy
from typing import Any, NamedTuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas.nodes.symm import Symm
from dace.libraries.tileops.dispatch import detect_host_isa
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.pipeline import build_stages
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

M = dace.symbol("M")
N = dace.symbol("N")
datatype = dace.float64


@dace.program
def symm_loops(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_sign_flip(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    """``symm_loops`` whose finalize SUBTRACTS the ``temp2`` term."""
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] - alpha[0] * temp2


@dace.program
def symm_missing_alpha(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    """``symm_loops`` whose scatter onto the rows above drops the ``alpha`` factor."""
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def gemm_loops(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]


def _symm_nodes(sdfg):
    return [n for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes() if isinstance(n, Symm)]


def _reference(A_tri, B, C, alpha, beta):
    Asym = np.tril(A_tri) + np.tril(A_tri, -1).T
    return alpha * (Asym @ B) + beta * C


@dace.program
def _symm_slice_kernel(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    temp2 = np.zeros((N,), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


@dace.program
def _symm_slice_missing_term(
    C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]
):
    temp2 = np.zeros((N,), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i]


@dace.program
def _symm_slice_transposed_dot(
    C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]
):
    temp2 = np.zeros((N,), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[:i, i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


@dace.program
def _symm_slice_temp_escapes(
    C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1], out: datatype[N]
):
    temp2 = np.zeros((N,), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2
    out[:] = temp2


def _operand(state, node, conn):
    """The array name wired into ``node``'s ``conn`` input connector."""
    return next(e.data.data for e in state.in_edges(node) if e.dst_conn == conn)


def canonicalized(program):
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    return sdfg


def _normalized(program):
    """``program`` as the phased recipe hands it to ``lift_semantics``: every phase before it, no lift run."""
    sdfg = program.to_sdfg(simplify=False)
    stages = build_stages(target="cpu")
    first = next(i for i, (_, unit) in enumerate(stages) if isinstance(unit, LoopToSymm))
    for _, unit in stages[:first]:
        unit.apply_pass(sdfg, {})
    return sdfg


def _slice_inputs(m, n):
    rng = np.random.default_rng(7)
    A = np.tril(rng.random((m, m)))
    A[np.triu_indices(m, 1)] = -999.0  # garbage in the unreferenced triangle
    return A, rng.random((m, n)), rng.random((m, n)), np.array([1.5]), np.array([1.2])


def test_symm_loop_nest_lifted_to_symm_node():
    """The scalar-loop symm nest, normalized, becomes exactly one ``Symm`` node (side L, uplo L, runtime alpha/beta
    connectors) that matches a dense reference."""
    sdfg = _normalized(symm_loops)
    sdfg.name = f"{sdfg.name}_normalized_lift"
    assert LoopToSymm().apply_pass(sdfg, {}) == 1

    symms = _symm_nodes(sdfg)
    assert len(symms) == 1
    node = symms[0]
    assert node.side == "L" and node.uplo == "L"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_b", "_c", "_alpha", "_beta"}

    m, n = 20, 30
    A, B, C, alpha, beta = _slice_inputs(m, n)
    ref = _reference(A, B, C, alpha[0], beta[0])
    got = C.copy()
    sdfg(C=got, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    assert np.allclose(got, ref), f"maxdiff {np.max(np.abs(got - ref))}"


def test_gemm_nest_not_matched():
    """A plain contraction (no triangular self-scatter) must not be lifted to Symm."""
    sdfg = _normalized(gemm_loops)
    assert not LoopToSymm().apply_pass(sdfg, {})
    assert _symm_nodes(sdfg) == []


def test_slice_nest_lifted_with_the_prescale_left_alone():
    """The npbench slice spelling lifts to one ``Symm`` with the RIGHT operands, the
    loop nest disappears, and the separate ``C *= beta`` prescale is left in place --
    so the node carries a compile-time ``beta = 1`` and no ``_beta`` connector."""
    sdfg = canonicalized(_symm_slice_kernel)
    symms = _symm_nodes(sdfg)
    assert len(symms) == 1
    node = symms[0]
    state = next(st for sd in sdfg.all_sdfgs_recursive() for st in sd.states() if node in st.nodes())
    assert node.side == "L" and node.uplo == "L"
    assert node.alpha_input and not node.beta_input and node.beta == 1
    assert "_beta" not in node.in_connectors
    assert (
        _operand(state, node, "_a"),
        _operand(state, node, "_b"),
        _operand(state, node, "_c"),
        _operand(state, node, "_alpha"),
    ) == ("A", "B", "C", "alpha")
    # The whole nest is gone: no LoopRegion is left anywhere to recompute C.
    assert [
        r
        for sd in sdfg.all_sdfgs_recursive()
        for r in sd.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion)
    ] == []

    m, n = 20, 30
    A, B, C, alpha, beta = _slice_inputs(m, n)
    ref = _reference(A, B, C, alpha[0], beta[0])
    got = C.copy()
    sdfg(C=got, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    assert np.allclose(got, ref), f"maxdiff {np.max(np.abs(got - ref))}"


def test_lifted_slice_nest_survives_vectorization():
    """The lift must hold through the multi-dim vectorizer: the hand-rolled version of
    this nest accumulates onto ``C`` from a parallel map, and the vectorizer dropped the
    WCR that made that safe -- which is only visible on more than one thread."""
    sdfg = copy.deepcopy(canonicalized(_symm_slice_kernel))
    sdfg.name = f"{sdfg.name}_vectorized"
    VectorizeCPUMultiDim(
        VectorizeConfig(
            widths=(8,),
            target_isa=detect_host_isa(),
            remainder_strategy=RemainderStrategy.MASKED_TAIL,
            branch_mode=BranchMode.MERGE,
        )
    ).apply_pass(sdfg, {})
    assert len(_symm_nodes(sdfg)) == 1, "vectorization must not dismantle the lifted node"
    # No parallel map may write C: the prescale is elementwise and the product is the
    # library node's business, so a Multicore scope storing to C means the hand-rolled
    # accumulation came back.
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            scopes = state.scope_dict()
            for dn in state.data_nodes():
                if dn.data != "C" or state.in_degree(dn) == 0:
                    continue
                scope = scopes[dn]
                while scope is not None:
                    assert not (isinstance(scope, nodes.MapEntry) and "Multicore" in str(scope.map.schedule)), (
                        f"C written inside {scope.map.params}"
                    )
                    scope = scopes[scope]

    # The race this guards is invisible on one thread, so pin the thread count HERE instead of
    # reading OMP_NUM_THREADS: a ``num_threads`` clause overrides the environment, so the check
    # holds whatever the suite exports. Every multicore scope is pinned, so a regression that
    # brings the accumulation back is racing on the run below rather than quietly serialised.
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for entry in state.nodes():
                if isinstance(entry, nodes.MapEntry) and entry.map.schedule in dtypes.CPU_SCHEDULES:
                    entry.map.omp_num_threads = 4

    m, n = 20, 30
    A, B, C, alpha, beta = _slice_inputs(m, n)
    ref = _reference(A, B, C, alpha[0], beta[0])
    got = C.copy()
    sdfg(C=got, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    assert np.allclose(got, ref), f"maxdiff {np.max(np.abs(got - ref))}"


DEVIATING_SLICE_NESTS = [
    (_symm_slice_missing_term, "finalize drops the alpha*temp2 term"),
    (_symm_slice_transposed_dot, "the inner product reads A's unreferenced column"),
    (_symm_slice_temp_escapes, "the scratch vector is read after the nest"),
]


@pytest.mark.parametrize("program,why", DEVIATING_SLICE_NESTS)
def test_deviating_slice_nest_is_not_lifted(program, why):
    """Each deviation is a DIFFERENT program from ``symm``, so the lift must decline it:
    dropping a term changes the result, reading ``A[:i, i]`` reads the triangle the
    symmetric operand does not define, and a scratch vector something else reads cannot
    be spliced away with the nest."""
    assert _symm_nodes(canonicalized(program)) == [], why


MM, NN, MN = datatype[M, M], datatype[N, N], datatype[M, N]


@dace.program
def symm_upper_from_columns(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[k, i]
                temp2 += B[k, j] * A[k, i]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_upper_descending(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M - 1, -1, -1):
        for j in range(N):
            temp2 = 0.0
            for k in range(i + 1, M):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_lower_from_columns_descending(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M - 1, -1, -1):
        for j in range(N):
            temp2 = 0.0
            for k in range(i + 1, M):
                C[k, j] += alpha[0] * B[i, j] * A[k, i]
                temp2 += B[k, j] * A[k, i]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_right_lower(C: MN, A: NN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N):
        for j in range(M):
            temp2 = 0.0
            for k in range(i):
                C[j, k] += alpha[0] * B[j, i] * A[i, k]
                temp2 += B[j, k] * A[i, k]
            C[j, i] = beta[0] * C[j, i] + alpha[0] * B[j, i] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_right_upper_descending(C: MN, A: NN, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(N - 1, -1, -1):
        for j in range(M):
            temp2 = 0.0
            for k in range(i + 1, N):
                C[j, k] += alpha[0] * B[j, i] * A[i, k]
                temp2 += B[j, k] * A[i, k]
            C[j, i] = beta[0] * C[j, i] + alpha[0] * B[j, i] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_accumulating_descending(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    """Scatters reach row ``i`` before it is finished; the finish only adds, so this is still symm."""
    for i in range(M - 1, -1, -1):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] += alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_unscaled(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] += alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_overwrite(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


@dace.program
def symm_no_alpha(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    for i in range(M):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + B[i, j] * A[i, i] + temp2


class Variant(NamedTuple):
    """A symm program and what the lift must make of it."""

    program: Any
    side: str
    uplo: str
    alpha: str  #: ``"runtime"`` (the ``alpha`` input) or ``"one"`` (absent)
    beta: str  #: ``"runtime"`` (the ``beta`` input), ``"one"`` (the finish adds) or ``"zero"`` (it overwrites)


VARIANTS = [
    Variant(symm_loops, "L", "L", "runtime", "runtime"),
    Variant(symm_upper_from_columns, "L", "U", "runtime", "runtime"),
    Variant(symm_upper_descending, "L", "U", "runtime", "runtime"),
    Variant(symm_lower_from_columns_descending, "L", "L", "runtime", "runtime"),
    Variant(symm_right_lower, "R", "L", "runtime", "runtime"),
    Variant(symm_right_upper_descending, "R", "U", "runtime", "runtime"),
    Variant(symm_accumulating_descending, "L", "L", "runtime", "one"),
    Variant(symm_unscaled, "L", "L", "runtime", "one"),
    Variant(symm_overwrite, "L", "L", "runtime", "zero"),
    Variant(symm_no_alpha, "L", "L", "one", "runtime"),
]

COEFFICIENT_VALUES = {"runtime": 1, "one": 1, "zero": 0}


def variant_inputs(side: str, m: int = 9, n: int = 7):
    """``A`` symmetric-sized for ``side`` with garbage in BOTH triangles' mirror: only the stored one is read."""
    rng = np.random.default_rng(13)
    order = m if side == "L" else n
    return rng.random((order, order)), rng.random((m, n)), rng.random((m, n)), np.array([1.5]), np.array([1.2])


def variant_reference(variant: Variant, A, B, C, alpha, beta):
    """``alpha*Asym@B + beta*C`` (side L) or ``alpha*B@Asym + beta*C`` (side R), ``Asym`` built from ``uplo``."""
    stored = np.tril(A) if variant.uplo == "L" else np.triu(A)
    asym = stored + np.tril(stored, -1).T if variant.uplo == "L" else stored + np.triu(stored, 1).T
    product = asym @ B if variant.side == "L" else B @ asym
    alpha_value = alpha[0] if variant.alpha == "runtime" else 1
    beta_value = beta[0] if variant.beta == "runtime" else COEFFICIENT_VALUES[variant.beta]
    return alpha_value * product + beta_value * C


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.program.name)
def test_orientation_and_scaling_variants_lift(variant):
    """Each side, stored triangle and loop direction, and each scaling (``beta*C``, ``C``, overwrite; runtime or
    absent ``alpha``), lifts to the right ``Symm`` attributes and computes what numpy does."""
    sdfg = canonicalized(variant.program)
    (node,) = _symm_nodes(sdfg)
    state = next(st for sd in sdfg.all_sdfgs_recursive() for st in sd.states() if node in st.nodes())
    assert (node.side, node.uplo) == (variant.side, variant.uplo)
    assert node.alpha_input == (variant.alpha == "runtime") and node.beta_input == (variant.beta == "runtime")
    assert (
        float(node.alpha) == COEFFICIENT_VALUES[variant.alpha] and float(node.beta) == COEFFICIENT_VALUES[variant.beta]
    )
    assert (_operand(state, node, "_a"), _operand(state, node, "_b")) == ("A", "B")
    assert ("_c" in node.in_connectors) == (variant.beta != "zero")

    A, B, C, alpha, beta = variant_inputs(variant.side)
    want = variant_reference(variant, A, B, C, alpha, beta)
    got = C.copy()
    sdfg(C=got, A=A, B=B, alpha=alpha, beta=beta, M=C.shape[0], N=C.shape[1])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


@dace.program
def symm_scaled_descending(C: MN, A: MM, B: MN, alpha: datatype[1], beta: datatype[1]):
    """Scatters reach row ``i`` before ``beta`` scales it: ``beta`` multiplies part of the product, not symm."""
    for i in range(M - 1, -1, -1):
        for j in range(N):
            temp2 = 0.0
            for k in range(i):
                C[k, j] += alpha[0] * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta[0] * C[i, j] + alpha[0] * B[i, j] * A[i, i] + alpha[0] * temp2


def loop_reference(program, A, B, C, alpha, beta):
    """The literal triangular loop nest each deviating program runs (not a closed-form formula)."""
    m, n = C.shape
    out = C.copy()
    a0, b0 = float(alpha[0]), float(beta[0])
    rows = range(m - 1, -1, -1) if program is symm_scaled_descending else range(m)
    for i in rows:
        for j in range(n):
            temp2 = 0.0
            for k in range(i):
                out[k, j] += (1.0 if program is symm_missing_alpha else a0) * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            sign = -1.0 if program is symm_sign_flip else 1.0
            out[i, j] = b0 * out[i, j] + a0 * B[i, j] * A[i, i] + sign * a0 * temp2
    return out


DEVIATING_LOOP_NESTS = [
    (symm_sign_flip, "the temp2 term is subtracted instead of added"),
    (symm_missing_alpha, "alpha is dropped from the triangular scatter"),
    (symm_scaled_descending, "beta scales scatters that already landed"),
]


@pytest.mark.parametrize("program,why", DEVIATING_LOOP_NESTS)
def test_a_nest_with_symm_access_shapes_but_other_arithmetic_is_not_lifted(program, why):
    """Same loops and accesses as a symm nest -- triangular scatter onto C, symmetric operand A read on one
    triangle + diagonal, matrix B -- but a different result. ``Symm`` always computes the CORRECT symm formula, so
    lifting these would silently change the program's result: the nest must stay and keep computing its own."""
    sdfg = canonicalized(program)
    assert _symm_nodes(sdfg) == [], why

    m, n = 8, 6
    A, B, C, alpha, beta = _slice_inputs(m, n)
    want = loop_reference(program, A, B, C, alpha, beta)
    got = C.copy()
    sdfg(C=got, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    np.testing.assert_allclose(got, want)


if __name__ == "__main__":
    test_symm_loop_nest_lifted_to_symm_node()
    test_gemm_nest_not_matched()
    test_slice_nest_lifted_with_the_prescale_left_alone()
    test_lifted_slice_nest_survives_vectorization()
    for case in DEVIATING_SLICE_NESTS:
        test_deviating_slice_nest_is_not_lifted(*case)
    for variant in VARIANTS:
        test_orientation_and_scaling_variants_lift(variant)
    for case in DEVIATING_LOOP_NESTS:
        test_a_nest_with_symm_access_shapes_but_other_arithmetic_is_not_lifted(*case)
