# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Absorbing a 2-D operand transpose into a BLAS library node's structural flag instead of a
physical transpose, and the tensor-contraction fallback when it cannot be absorbed.

A layout Permute of a matrix that feeds a native BLAS call should keep the single native call and
just mark the operand transposed -- the vendor kernel reads the relaid-out box and transposes it for
free (``trans`` / ``uplo`` parameter). This is exercised through:

  * ``FoldTransposeIntoMatMul`` -- an explicit ``Array -> Transpose -> MatMul`` collapses to one
    ``MatMul`` with the flag flipped (the ``atax`` ``y = A^T (A x)`` shape dace lifts as two MatMuls
    plus a Transpose node).
  * ``PermuteDimensions`` special rule -- permuting a Gemm/MatMul/Syrk/Symm operand flips
    ``transA``/``transB`` / ``trans`` / ``uplo`` (``flip_operand_transpose``), bit-exact on CPU and
    (cuBLAS) GPU. ``Syr2k`` and a general (``_b``) ``Symm`` operand cannot be absorbed and are
    refused loudly.
  * ``SyrkToTensorDot`` -- the fallback for a layout a flag cannot express: ``C = A A^T`` becomes the
    contraction ``ik,jk->ij``. It writes the WHOLE symmetric ``C`` (not just the ``uplo`` triangle),
    so it is only valid for a fresh output.
"""
import numpy
import pytest

import dace
from dace.libraries.blas import Syrk, Syr2k, Symm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.transformation.layout import PermuteDimensions
from dace.transformation.layout.rewrite_libnodes import (FoldTransposeIntoMatMul, SyrkToTensorDot,
                                                         flip_operand_transpose)

M, N = dace.symbol("M"), dace.symbol("N")


@dace.program
def atax_blas(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[N]):
    tmp = A @ x
    y[:] = A.T @ tmp


def _matmuls(sdfg):
    return [n for s in sdfg.states() for n in s.nodes() if isinstance(n, MatMul)]


def _atax_inputs(m, n, seed=3):
    rng = numpy.random.default_rng(seed)
    A = rng.random((m, n))
    x = rng.random(n)
    return A, x, A.T @ (A @ x)


# --------------------------------------------------------------------------- #
#  Fold Transpose -> MatMul into the transpose flag
# --------------------------------------------------------------------------- #
def test_fold_transpose_into_matmul_removes_node():
    sdfg = atax_blas.to_sdfg(simplify=True)
    assert any(isinstance(n, Transpose) for s in sdfg.states() for n in s.nodes())

    folded = FoldTransposeIntoMatMul().apply_pass(sdfg, {})
    assert folded == 1
    assert not any(isinstance(n, Transpose) for s in sdfg.states() for n in s.nodes())

    flags = sorted(n.transA for n in _matmuls(sdfg))
    assert flags == [False, True], "one MatMul reads A directly, the other absorbs A^T as transA=True"

    m = n = 32
    A, x, ref = _atax_inputs(m, n)
    y = numpy.zeros(n)
    sdfg(A=A.copy(), x=x.copy(), y=y, M=m, N=n)
    assert numpy.allclose(y, ref)


# --------------------------------------------------------------------------- #
#  Permuting the shared operand flips both flags -- CPU
# --------------------------------------------------------------------------- #
def _build_atax(perm):
    sdfg = atax_blas.to_sdfg(simplify=True)
    FoldTransposeIntoMatMul().apply_pass(sdfg, {})
    if perm:
        PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=False).apply_pass(sdfg, {})
    return sdfg


def test_atax_permute_flips_flags_cpu():
    m = n = 32
    A, x, ref = _atax_inputs(m, n)

    row = _build_atax(perm=False)
    assert sorted(nd.transA for nd in _matmuls(row)) == [False, True]
    y = numpy.zeros(n)
    row(A=A.copy(), x=x.copy(), y=y, M=m, N=n)
    assert numpy.allclose(y, ref)

    col = _build_atax(perm=True)
    # both flags toggled by the single Permute
    assert sorted(nd.transA for nd in _matmuls(col)) == [False, True]
    assert list(col.arrays["A"].shape) == [N, M], "A physically relaid out to [N, M]"
    col.validate()
    y = numpy.zeros(n)
    col(A=numpy.asarray(A.T, order="C").copy(), x=x.copy(), y=y, M=m, N=n)  # same logical A, [N, M]
    assert numpy.allclose(y, ref)


@pytest.mark.gpu
def test_atax_permute_flips_flags_gpu():
    m = n = 64
    A, x, ref = _atax_inputs(m, n)

    for perm in (False, True):
        sdfg = _build_atax(perm)
        for state in sdfg.states():
            for node in list(state.nodes()):
                if isinstance(node, MatMul):
                    node.expand(sdfg, state)  # specialize to Gemv before pinning cuBLAS
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.sdfg.nodes.LibraryNode):
                    node.implementation = "cuBLAS"
        sdfg.apply_gpu_transformations()

        Ain = numpy.asarray(A.T, order="C").copy() if perm else A.copy()
        y = numpy.zeros(n)
        sdfg(A=Ain, x=x.copy(), y=y, M=m, N=n)
        assert numpy.allclose(y, ref), f"perm={perm}"


# --------------------------------------------------------------------------- #
#  Syrk / Symm flag flips (triangle-preserving, in place)
# --------------------------------------------------------------------------- #
def _syrk_sdfg(nn, k):
    sdfg = dace.SDFG("syrk_flip")
    sdfg.add_array("A", [nn, k], dace.float64)
    sdfg.add_array("C", [nn, nn], dace.float64)
    st = sdfg.add_state()
    node = Syrk("syrk", uplo="L", trans="N", alpha=1, beta=0)
    node.implementation = "pure"
    st.add_node(node)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{nn}, 0:{k}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{nn}, 0:{nn}]"))
    return sdfg


def test_syrk_permute_flips_trans():
    nn, k = 12, 9
    rng = numpy.random.default_rng(0)
    A = rng.random((nn, k))
    ref = numpy.tril(A @ A.T)

    sdfg = _syrk_sdfg(nn, k)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=False).apply_pass(sdfg, {})
    syrk = next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, Syrk))
    assert syrk.trans == "T"
    assert list(sdfg.arrays["A"].shape) == [k, nn]
    sdfg.validate()

    C = numpy.zeros((nn, nn))
    sdfg(A=numpy.asarray(A.T, order="C").copy(), C=C)
    assert numpy.allclose(numpy.tril(C), ref)


def _symm_sdfg(sa, mm):
    sdfg = dace.SDFG("symm_flip")
    sdfg.add_array("A", [sa, sa], dace.float64)
    sdfg.add_array("B", [sa, mm], dace.float64)
    sdfg.add_array("C", [sa, mm], dace.float64)
    st = sdfg.add_state()
    node = Symm("symm", side="L", uplo="L", alpha=1, beta=0)
    node.implementation = "pure"
    st.add_node(node)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{sa}, 0:{sa}]"))
    st.add_edge(st.add_read("B"), None, node, "_b", dace.Memlet(f"B[0:{sa}, 0:{mm}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{sa}, 0:{mm}]"))
    return sdfg


def test_symm_permute_flips_uplo():
    sa, mm = 10, 7
    rng = numpy.random.default_rng(0)
    Atri = rng.random((sa, sa))
    B = rng.random((sa, mm))
    Asym = numpy.tril(Atri) + numpy.tril(Atri, -1).T
    ref = Asym @ B

    sdfg = _symm_sdfg(sa, mm)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=False).apply_pass(sdfg, {})
    symm = next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, Symm))
    assert symm.uplo == "U", "transposing the symmetric operand swaps its stored triangle"
    sdfg.validate()

    C = numpy.zeros((sa, mm))
    sdfg(A=numpy.asarray(Atri.T, order="C").copy(), B=B.copy(), C=C)
    assert numpy.allclose(C, ref)


# --------------------------------------------------------------------------- #
#  Refusals -- never silently miscompile an unabsorbable layout
# --------------------------------------------------------------------------- #
def test_flip_refuses_syr2k():
    node = Syr2k("s", uplo="L", trans="N", alpha=1, beta=0)
    with pytest.raises(NotImplementedError):
        flip_operand_transpose(node, "_a")


def test_flip_refuses_general_symm_operand():
    node = Symm("s", side="L", uplo="L", alpha=1, beta=0)
    with pytest.raises(NotImplementedError):
        flip_operand_transpose(node, "_b")  # B is a general matrix, no transpose flag


def test_permute_refuses_non_transpose_gemm_operand():
    # a 3-D (batched) operand permute is not a plain [1,0] transpose -> refused
    sdfg = dace.SDFG("bad")
    sdfg.add_array("A", [4, 5, 6], dace.float64)
    sdfg.add_array("B", [4, 6, 3], dace.float64)
    sdfg.add_array("C", [4, 5, 3], dace.float64)
    st = sdfg.add_state()
    node = MatMul("mm")
    st.add_node(node)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet("A[0:4, 0:5, 0:6]"))
    st.add_edge(st.add_read("B"), None, node, "_b", dace.Memlet("B[0:4, 0:6, 0:3]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet("C[0:4, 0:5, 0:3]"))
    with pytest.raises(NotImplementedError):
        PermuteDimensions(permute_map={"A": [0, 2, 1]}, add_permute_maps=False).apply_pass(sdfg, {})


# --------------------------------------------------------------------------- #
#  Tensor-contraction fallback
# --------------------------------------------------------------------------- #
def test_syrk_to_tensordot_full_output():
    nn, k = 12, 9
    rng = numpy.random.default_rng(0)
    A = rng.random((nn, k))

    sdfg = _syrk_sdfg(nn, k)
    assert SyrkToTensorDot().apply_pass(sdfg, {}) == 1
    tds = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, TensorDot)]
    assert len(tds) == 1 and tds[0].left_axes == [1] and tds[0].right_axes == [1]
    for n in tds:
        n.implementation = "pure"
    sdfg.validate()

    C = numpy.zeros((nn, nn))
    sdfg(A=A.copy(), C=C)
    assert numpy.allclose(C, A @ A.T), "TensorDot writes the full symmetric product"
    assert numpy.allclose(C, C.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
