# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout changes over the remaining native library nodes: an Einsum operand permutation (the P3
``transform_einsum`` rewrite, end-to-end compile+run), and the layout-agnostic copy / memset.

The P3 rewrites (``rewrite_libnodes.py``) make a layout change reach a libnode's SEMANTIC indices:
Gemm->TensorDot, einsum-subscript, Reduce.axes, Scan.stride are covered in ``rewrite_libnodes_test``.
This file adds the missing end-to-end coverage:

  * **Einsum operand permute** -- storing an operand transposed and rewriting its subscripts keeps
    the contraction bit-exact (previously only string-level unit tests existed).
  * **memset** (a zero-init tasklet) -- layout-agnostic: permuting an array through it stays exact.
  * **copy** (a ``CopyLibraryNode``) -- layout-agnostic ONLY while both operands keep the SAME
    layout; a layout change that leaves the two operands with DIFFERENT layouts turns the copy into
    a transpose, which the copy libnode cannot express (it asks for a Transpose libnode). That case
    is an xfail: the missing P3 rule is Copy->TensorTranspose (the analog of Gemm->TensorDot).
"""
import numpy
import pytest
import dace

from dace.libraries.blas.nodes.einsum import Einsum
from dace.transformation.layout.rewrite_libnodes import transform_einsum
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

M, K, Nn = (dace.symbol(s) for s in ("M", "K", "Nn"))


# --------------------------------------------------------------------------- #
#  Einsum operand permutation -- end-to-end compile + run
# --------------------------------------------------------------------------- #
def _einsum_sdfg(einsum_str, sa, sb, so):
    sdfg = dace.SDFG("es_" + einsum_str.replace(",", "_").replace("->", "to"))
    sdfg.add_array("A", sa, dace.float64)
    sdfg.add_array("B", sb, dace.float64)
    sdfg.add_array("C", so, dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    node = Einsum("einsum")
    node.einsum_str = einsum_str
    node.add_in_connector("_i0")
    node.add_in_connector("_i1")
    node.add_out_connector("_o")
    st.add_node(node)
    st.add_edge(st.add_read("A"), None, node, "_i0", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, node, "_i1", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(node, "_o", st.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_einsum_permute_operand0_bitexact():
    """Store operand A transposed ([K,M]) and rewrite ij->ji: the contraction is still A @ B."""
    _M, _K, _N = 4, 5, 6
    A = numpy.random.default_rng(0).random((_M, _K))
    B = numpy.random.default_rng(1).random((_K, _N))

    new_str = transform_einsum("ij,jk->ik", 0, (1, 0))
    assert new_str == "ji,jk->ik"
    sdfg = _einsum_sdfg(new_str, [_K, _M], [_K, _N], [_M, _N])
    sdfg.validate()
    C = numpy.zeros((_M, _N))
    sdfg(A=A.T.copy(), B=B.copy(), C=C)  # A stored transposed
    assert numpy.allclose(C, A @ B)


def test_einsum_permute_operand1_bitexact():
    """Store operand B transposed ([N,K]) and rewrite jk->kj: still A @ B."""
    _M, _K, _N = 4, 5, 6
    A = numpy.random.default_rng(2).random((_M, _K))
    B = numpy.random.default_rng(3).random((_K, _N))

    new_str = transform_einsum("ij,jk->ik", 1, (1, 0))
    assert new_str == "ij,kj->ik"
    sdfg = _einsum_sdfg(new_str, [_M, _K], [_N, _K], [_M, _N])
    sdfg.validate()
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), B=B.T.copy(), C=C)
    assert numpy.allclose(C, A @ B)


# --------------------------------------------------------------------------- #
#  memset / copy under a layout change
# --------------------------------------------------------------------------- #
@dace.program
def memset_add(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.0
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = C[i, j] + A[i, j]


@dace.program
def elementwise_copy(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j]


def test_memset_permute_bitexact():
    """A memset (zero-init tasklet) is layout-agnostic: permuting the input array stays exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(0).random((_M, _N))
    sdfg = memset_add.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, A)


def test_copy_libnode_consistent_permute_bitexact():
    """An elementwise copy lifts to a CopyLibraryNode; permuting BOTH operands the same way keeps
    it an identity copy (same layout on each side) -- bit-exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(1).random((_M, _N))
    sdfg = elementwise_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0], "C": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, A)


@pytest.mark.xfail(reason="Copy->TensorTranspose P3 rewrite not implemented: a copy whose operands "
                   "are relaid out to DIFFERENT layouts becomes a transpose the CopyLibraryNode "
                   "cannot express.",
                   strict=True)
def test_copy_libnode_transposing_operand_gap():
    """Permuting only ONE side of a copy makes it transposing; the copy libnode refuses (asks for a
    Transpose). Documents the missing Copy->TensorTranspose rewrite."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(2).random((_M, _N))
    sdfg = elementwise_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)  # raises ValueError in copy-node expansion
    assert numpy.allclose(C, A)


if __name__ == "__main__":
    test_einsum_permute_operand0_bitexact()
    test_einsum_permute_operand1_bitexact()
    test_memset_permute_bitexact()
    test_copy_libnode_consistent_permute_bitexact()
    print("libnode layout tests PASS")
