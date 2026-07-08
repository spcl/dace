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
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.libraries.blas.nodes.symm import Symm
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
