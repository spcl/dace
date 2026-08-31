# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToSymm`` lifts the hand-written polybench ``symm`` nest to a ``Symm`` BLAS
node and is a strict no-op on every other kernel shape.

The frontend emits polybench ``symm`` as a 2-D map whose NestedSDFG boundary carries
a triangular self-scatter ``C[0:i, j]`` plus a point-write ``C[i, j]`` fed by a
symmetric operand ``A`` (referenced only on its lower triangle + diagonal) and a
matrix ``B``. ``LoopToSymm`` recognises exactly that shape, extracts the operands and
the runtime ``alpha``/``beta`` scalars, and replaces the nest with a single ``Symm``
node -- verified here both structurally and numerically (bit-exact vs a dense
reference). A gemm nest (a plain contraction, no triangular self-scatter) must not
match.

The pass also recognises the npbench SLICE spelling of the same kernel -- a two-level
``LoopRegion`` nest whose statements use column slices and a ``B[:i, j] @ A[i, :i]``
inner product -- which is the form the corpus carries. That one is matched by resolving
each body state's dataflow, so it needs the full canonicalization pipeline to have fused
the body states first; the tests below drive it through ``canonicalize`` for that reason,
and check that the lift survives vectorization (an un-lifted slice nest hand-rolls its
accumulation and the vectorizer used to drop the WCR that made it safe).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import copy

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas.nodes.symm import Symm
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

M = dace.symbol("M")
N = dace.symbol("N")
datatype = dace.float64


@dace.program
def _symm_kernel(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):

    @dace.mapscope
    def comp_all(j: _[0:N], i: _[0:M]):
        temp2 = dace.define_local_scalar(datatype)

        @dace.tasklet
        def reset_tmp():
            tmp >> temp2
            tmp = 0

        @dace.map
        def comp_t2(k: _[0:i]):
            ialpha << alpha
            ia << A[i, k]
            ibi << B[i, j]
            ibk << B[k, j]
            oc >> C(1, lambda a, b: a + b)[k, j]
            ot2 >> temp2(1, lambda a, b: a + b)

            oc = ialpha * ibi * ia
            ot2 = ibk * ia

        @dace.tasklet
        def comp_rest():
            ibeta << beta
            ib << B[i, j]
            iadiag << A[i, i]
            ialpha << alpha
            it2 << temp2
            ic << C[i, j]
            oc >> C[i, j]
            oc = ibeta * ic + ialpha * ib * iadiag + ialpha * it2


@dace.program
def _gemm_kernel(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N]):

    @dace.map
    def mm(i: _[0:M], j: _[0:N], k: _[0:M]):
        ia << A[i, k]
        ib << B[k, j]
        oc >> C(1, lambda a, b: a + b)[i, j]
        oc = ia * ib


def _symm_nodes(sdfg):
    return [n for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes() if isinstance(n, Symm)]


def _reference(A_tri, B, C, alpha, beta):
    Asym = np.tril(A_tri) + np.tril(A_tri, -1).T
    return alpha * (Asym @ B) + beta * C


@dace.program
def _symm_slice_kernel(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    temp2 = np.zeros((N, ), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


@dace.program
def _symm_slice_missing_term(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1],
                             beta: datatype[1]):
    temp2 = np.zeros((N, ), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i]


@dace.program
def _symm_slice_transposed_dot(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1],
                               beta: datatype[1]):
    temp2 = np.zeros((N, ), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[:i, i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


@dace.program
def _symm_slice_temp_escapes(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1],
                             beta: datatype[1], out: datatype[N]):
    temp2 = np.zeros((N, ), dtype=C.dtype)
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


def _canonicalized(program):
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    return sdfg


def _slice_inputs(m, n):
    rng = np.random.default_rng(7)
    A = np.tril(rng.random((m, m)))
    A[np.triu_indices(m, 1)] = -999.0  # garbage in the unreferenced triangle
    return A, rng.random((m, n)), rng.random((m, n)), np.array([1.5]), np.array([1.2])


def test_symm_nest_lifted_to_symm_node():
    """The polybench symm nest becomes exactly one ``Symm`` node (side L, uplo L,
    runtime alpha/beta connectors) and stays bit-exact vs a dense reference."""
    sdfg = _symm_kernel.to_sdfg(simplify=False)
    count = LoopToSymm().apply_pass(sdfg, {})
    assert count == 1

    symms = _symm_nodes(sdfg)
    assert len(symms) == 1
    node = symms[0]
    assert node.side == "L" and node.uplo == "L"
    assert node.alpha_input and node.beta_input
    assert node.in_connectors.keys() >= {"_a", "_b", "_c", "_alpha", "_beta"}

    m, n = 20, 30
    rng = np.random.default_rng(0)
    A = np.tril(rng.random((m, m)))
    A[np.triu_indices(m, 1)] = -999.0  # garbage in the unreferenced triangle
    B = rng.random((m, n))
    C = rng.random((m, n))
    alpha = np.array([1.5])
    beta = np.array([1.2])
    ref = _reference(A, B, C, alpha[0], beta[0])

    sdfg.expand_library_nodes()
    sdfg.validate()
    Cw = C.copy()
    sdfg(C=Cw, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    assert np.allclose(Cw, ref), f"maxdiff {np.max(np.abs(Cw - ref))}"


def test_gemm_nest_not_matched():
    """A plain contraction (no triangular self-scatter) must not be lifted to Symm."""
    sdfg = _gemm_kernel.to_sdfg(simplify=False)
    count = LoopToSymm().apply_pass(sdfg, {})
    assert not count
    assert _symm_nodes(sdfg) == []


def test_slice_nest_lifted_with_the_prescale_left_alone():
    """The npbench slice spelling lifts to one ``Symm`` with the RIGHT operands, the
    loop nest disappears, and the separate ``C *= beta`` prescale is left in place --
    so the node carries a compile-time ``beta = 1`` and no ``_beta`` connector."""
    sdfg = _canonicalized(_symm_slice_kernel)
    symms = _symm_nodes(sdfg)
    assert len(symms) == 1
    node = symms[0]
    state = next(st for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() if node in st.nodes())
    assert node.side == "L" and node.uplo == "L"
    assert node.alpha_input and not node.beta_input and node.beta == 1
    assert "_beta" not in node.in_connectors
    assert (_operand(state, node, "_a"), _operand(state, node, "_b"), _operand(state, node, "_c"),
            _operand(state, node, "_alpha")) == ("A", "B", "C", "alpha")
    # The whole nest is gone: no LoopRegion is left anywhere to recompute C.
    assert [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions(recursive=True)
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
    sdfg = copy.deepcopy(_canonicalized(_symm_slice_kernel))
    sdfg.name = f"{sdfg.name}_vectorized"
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ),
                        target_isa=detect_host_isa(),
                        remainder_strategy="masked_tail",
                        branch_mode="merge")).apply_pass(sdfg, {})
    assert len(_symm_nodes(sdfg)) == 1, "vectorization must not dismantle the lifted node"
    # No parallel map may write C: the prescale is elementwise and the product is the
    # library node's business, so a Multicore scope storing to C means the hand-rolled
    # accumulation came back.
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            scopes = state.scope_dict()
            for dn in state.data_nodes():
                if dn.data != "C" or state.in_degree(dn) == 0:
                    continue
                scope = scopes[dn]
                while scope is not None:
                    assert not (isinstance(scope, nodes.MapEntry)
                                and "Multicore" in str(scope.map.schedule)), f"C written inside {scope.map.params}"
                    scope = scopes[scope]

    # The race this guards is invisible on one thread, so pin the thread count HERE instead of
    # reading OMP_NUM_THREADS: a ``num_threads`` clause overrides the environment, so the check
    # holds whatever the suite exports. Every multicore scope is pinned, so a regression that
    # brings the accumulation back is racing on the run below rather than quietly serialised.
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for entry in state.nodes():
                if isinstance(entry, nodes.MapEntry) and entry.map.schedule in dtypes.CPU_SCHEDULES:
                    entry.map.omp_num_threads = 4

    m, n = 20, 30
    A, B, C, alpha, beta = _slice_inputs(m, n)
    ref = _reference(A, B, C, alpha[0], beta[0])
    got = C.copy()
    sdfg(C=got, A=A.copy(), B=B.copy(), alpha=alpha.copy(), beta=beta.copy(), M=m, N=n)
    assert np.allclose(got, ref), f"maxdiff {np.max(np.abs(got - ref))}"


@pytest.mark.parametrize("program,why", [
    (_symm_slice_missing_term, "finalize drops the alpha*temp2 term"),
    (_symm_slice_transposed_dot, "the inner product reads A's unreferenced column"),
    (_symm_slice_temp_escapes, "the scratch vector is read after the nest"),
])
def test_deviating_slice_nest_is_not_lifted(program, why):
    """Each deviation is a DIFFERENT program from ``symm``, so the lift must decline it:
    dropping a term changes the result, reading ``A[:i, i]`` reads the triangle the
    symmetric operand does not define, and a scratch vector something else reads cannot
    be spliced away with the nest."""
    assert _symm_nodes(_canonicalized(program)) == [], why


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
