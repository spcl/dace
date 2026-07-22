# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the P3 library-node rewrites: transform_einsum / remap_contracted_axes (making a
layout permutation reach an einsum/TensorDot operand's semantic indices) and GemmToTensorDot
(exposing a Gemm's operand layout by lowering it to a TensorDot). Correctness oracle: the rewritten
node still computes A @ B bit-exactly, and an eligible Gemm is replaced while a scaled/accumulating
one is left in place."""
import numpy
import pytest
import dace

from dace.transformation.layout.rewrite_libnodes import (transform_einsum, remap_contracted_axes, GemmToTensorDot,
                                                         permute_reduce, block_scan_stride)
from dace.transformation.layout.select_lowering import select_layout_lowering
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import Scan, ScanOp


# --------------------------------------------------------------------------- #
#  transform_einsum / remap_contracted_axes
# --------------------------------------------------------------------------- #
def test_transform_einsum_permutes_operand_subscripts():
    assert transform_einsum("ij,jk->ik", 0, (1, 0)) == "ji,jk->ik"
    assert transform_einsum("ij,jk->ik", 1, (1, 0)) == "ij,kj->ik"
    assert transform_einsum("bij,bjk->bik", 0, (0, 2, 1)) == "bji,bjk->bik"


def test_transform_einsum_no_output_arrow():
    # An implicit-output einsum keeps its (missing) arrow.
    assert transform_einsum("ij,jk", 0, (1, 0)) == "ji,jk"


def test_transform_einsum_rejects_bad_perm():
    with pytest.raises(ValueError):
        transform_einsum("ij,jk->ik", 0, (0, 1, 2))  # rank mismatch
    with pytest.raises(ValueError):
        transform_einsum("ij,jk->ik", 5, (0, 1))  # operand out of range


def test_remap_contracted_axes():
    # Operand [M, K] contracts axis 1; transpose to [K, M] -> contracts axis 0.
    assert remap_contracted_axes([1], (1, 0)) == [0]
    assert remap_contracted_axes([0], (1, 0)) == [1]
    # A 3D operand contracting axes {1,2}, permuted (0,2,1): axis1->pos2, axis2->pos1.
    assert remap_contracted_axes([1, 2], (0, 2, 1)) == [2, 1]


# --------------------------------------------------------------------------- #
#  GemmToTensorDot
# --------------------------------------------------------------------------- #
def _gemm_sdfg(name, transA=False, transB=False, alpha=1.0, beta=0.0, cin=False, shapes=None):
    M, K, Nn = shapes or (4, 5, 6)
    sa = [K, M] if transA else [M, K]
    sb = [Nn, K] if transB else [K, Nn]
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", sa, dace.float64)
    sdfg.add_array("B", sb, dace.float64)
    sdfg.add_array("C", [M, Nn], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    g = Gemm("gemm", transA=transA, transB=transB, alpha=alpha, beta=beta, cin=cin)
    st.add_node(g)
    st.add_edge(st.add_read("A"), None, g, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, g, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    if cin:
        st.add_edge(st.add_read("C"), None, g, "_cin", dace.Memlet.from_array("C", sdfg.arrays["C"]))
    st.add_edge(g, "_c", st.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg, st


def _has(state, typename):
    return any(type(n).__name__ == typename for n in state.nodes())


def test_gemm_to_tensordot_matmul_bitexact():
    sdfg, st = _gemm_sdfg("mm_nn")
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    assert not _has(st, "Gemm") and _has(st, "TensorDot")
    sdfg.validate()
    assert select_layout_lowering(sdfg, "cpu") == 1  # transform left lowering unset; pick it here

    M, K, Nn = 4, 5, 6
    A = numpy.random.rand(M, K)
    B = numpy.random.rand(K, Nn)
    C = numpy.zeros((M, Nn))
    sdfg(A=A, B=B, C=C)
    assert numpy.allclose(C, A @ B)


def test_gemm_to_tensordot_transA():
    sdfg, st = _gemm_sdfg("mm_ta", transA=True)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    select_layout_lowering(sdfg, "cpu")

    M, K, Nn = 4, 5, 6
    A = numpy.random.rand(K, M)  # stored transposed
    B = numpy.random.rand(K, Nn)
    C = numpy.zeros((M, Nn))
    sdfg(A=A, B=B, C=C)
    assert numpy.allclose(C, A.T @ B)


def test_gemm_to_tensordot_transB():
    sdfg, st = _gemm_sdfg("mm_tb", transB=True)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    select_layout_lowering(sdfg, "cpu")

    M, K, Nn = 4, 5, 6
    A = numpy.random.rand(M, K)
    B = numpy.random.rand(Nn, K)  # stored transposed
    C = numpy.zeros((M, Nn))
    sdfg(A=A, B=B, C=C)
    assert numpy.allclose(C, A @ B.T)


def test_gemm_scaled_left_in_place():
    """A Gemm with beta != 0 (accumulate) is ineligible and kept -- TensorDot has no accumulator."""
    sdfg, st = _gemm_sdfg("mm_beta", alpha=1.0, beta=1.0, cin=True)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 0
    assert _has(st, "Gemm") and not _has(st, "TensorDot")


def test_gemm_alpha_scaled_left_in_place():
    """A Gemm with alpha != 1 is ineligible (TensorDot pins alpha=1)."""
    sdfg, st = _gemm_sdfg("mm_alpha", alpha=2.0, beta=0.0, cin=False)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 0
    assert _has(st, "Gemm")


# --------------------------------------------------------------------------- #
#  Reduce: axis remap under a permuted operand
# --------------------------------------------------------------------------- #
def _reduce_sdfg(name, in_shape, out_shape, axes):
    sdfg = dace.SDFG(name)
    sdfg.add_array("X", in_shape, dace.float64)
    sdfg.add_array("Y", out_shape, dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    red = Reduce("reduce", wcr="lambda a, b: a + b", axes=axes, identity=0)
    red.add_in_connector("_in")
    red.add_out_connector("_out")
    st.add_node(red)
    st.add_edge(st.add_read("X"), None, red, "_in", dace.Memlet.from_array("X", sdfg.arrays["X"]))
    st.add_edge(red, "_out", st.add_write("Y"), None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))
    return sdfg, red


def test_permute_reduce_axes_remapped_bitexact():
    """Reducing axis 1 of [M,N] and axis 0 of the transposed [N,M] (axes remapped by permute_reduce)
    give the same row sums -- the reduction follows the operand permutation."""
    M, Nn = 4, 6
    X = numpy.random.rand(M, Nn)
    ref = X.sum(axis=1)

    base, _ = _reduce_sdfg("red_base", [M, Nn], [M], [1])
    base.validate()
    Y0 = numpy.zeros(M)
    base(X=X.copy(), Y=Y0)
    assert numpy.allclose(Y0, ref)

    # Transposed operand [N, M]: permute (1, 0) moves the reduced logical axis to position 0.
    perm, red = _reduce_sdfg("red_perm", [Nn, M], [M], [1])
    permute_reduce(red, (1, 0))
    assert red.axes == [0]
    perm.validate()
    Y1 = numpy.zeros(M)
    perm(X=X.T.copy(), Y=Y1)
    assert numpy.allclose(Y1, ref)


def test_permute_reduce_all_unchanged():
    red = Reduce("r", wcr="lambda a, b: a + b", axes=None, identity=0)
    permute_reduce(red, (2, 0, 1))
    assert red.axes is None  # reduce-all is order-independent


# --------------------------------------------------------------------------- #
#  Scan: stride under a block-interleaved operand
# --------------------------------------------------------------------------- #
def test_block_scan_stride_matches_per_lane_scan():
    """Interleaving a scan array by V makes it V independent scans (stride V). The stride-V scan on
    the flattened array equals a per-lane inclusive scan of the [N/V, V] blocked view."""
    N, V = 8, 2
    x = numpy.arange(1, N + 1, dtype=numpy.float64)

    sdfg = dace.SDFG("scan_blk")
    sdfg.add_array("x", [N], dace.float64)
    sdfg.add_array("y", [N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    sc = Scan("scan", op=ScanOp.SUM)
    assert sc.stride == 1
    block_scan_stride(sc, V)
    assert sc.stride == V
    st.add_node(sc)
    st.add_edge(st.add_read("x"), None, sc, sc.INPUT_CONNECTOR_NAME, dace.Memlet.from_array("x", sdfg.arrays["x"]))
    st.add_edge(sc, sc.OUTPUT_CONNECTOR_NAME, st.add_write("y"), None, dace.Memlet.from_array("y", sdfg.arrays["y"]))
    y = numpy.zeros(N)
    sdfg(x=x.copy(), y=y)

    # Per-lane inclusive scan of the [N/V, V] view (each column scanned down the chunk axis).
    ref = x.reshape(N // V, V).cumsum(axis=0).reshape(N)
    assert numpy.allclose(y, ref)


if __name__ == "__main__":
    test_transform_einsum_permutes_operand_subscripts()
    test_transform_einsum_no_output_arrow()
    test_transform_einsum_rejects_bad_perm()
    test_remap_contracted_axes()
    test_gemm_to_tensordot_matmul_bitexact()
    test_gemm_to_tensordot_transA()
    test_gemm_to_tensordot_transB()
    test_gemm_scaled_left_in_place()
    test_gemm_alpha_scaled_left_in_place()
    test_permute_reduce_axes_remapped_bitexact()
    test_permute_reduce_all_unchanged()
    test_block_scan_stride_matches_per_lane_scan()
    print("rewrite_libnodes tests PASS")
